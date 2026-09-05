//! Opt-in runtime foundation for the experimental ACP protocol v2.
//!
//! Enable the `protocol-v2` crate feature to use this module. The feature maps
//! directly to the pinned SDK fork's
//! `agent-client-protocol/unstable_protocol_v2` feature. Root-level APIs remain the stable ACP v1 integration.

use std::collections::HashMap;
#[cfg(feature = "unstable-inject")]
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use agent_client_protocol::{Client, Handled, V2ConnectionTo};
use agentkit_core::{
    CancellationController, CancellationHandle, DataRef, Delta, FilePart, FinishReason, Item,
    ItemKind, MediaPart, MetadataMap, Modality, Part, PartId, PartKind,
    SessionId as AgentkitSessionId, StructuredPart, TextPart, ToolCallPart, ToolOutput,
    ToolResultPart, TurnCancellation, TurnId,
};
use agentkit_loop::{
    AgentEvent, LoopError, LoopInterrupt, LoopObserver, LoopStep, ModelAdapter, ModelSession,
    ObservedEvent,
};
use async_trait::async_trait;
use serde_json::json;
#[cfg(feature = "unstable-inject")]
use tokio::sync::Notify;
use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot};

use crate::AcpRuntimeError;

#[cfg(feature = "unstable-inject")]
fn sdk_v2_error(error: wire::Error) -> agent_client_protocol::Error {
    agent_client_protocol::Error::new(i32::from(error.code), error.message).data(error.data)
}

#[cfg(feature = "unstable-inject")]
fn session_not_found_error(session_id: &wire::SessionId) -> agent_client_protocol::Error {
    sdk_v2_error(wire::Error::resource_not_found(None)).data(serde_json::json!({
        "sessionId": session_id,
    }))
}

/// ACP v2 wire types from the pinned SDK fork.
///
/// These are gated by the SDK's `unstable_protocol_v2` feature and can change
/// while ACP v2 is under development. No stable v1 wire type is re-exported
/// from this namespace.
pub use agent_client_protocol::schema::ProtocolVersion;
pub use agent_client_protocol::schema::v2::*;

/// Explicit namespace for ACP v2 wire types from the pinned SDK fork.
pub mod wire {
    pub use agent_client_protocol::schema::ProtocolVersion;
    pub use agent_client_protocol::schema::v2::*;
}

/// Host-provided destination for ACP v2 session updates.
///
/// This unstable v2 API deliberately hides the runtime's internal channels. A
/// host can forward updates through its own ACP connection, event loop, or test
/// sink. Calls for one binding arrive in protocol order; implementations must
/// preserve that order when they enqueue them. Acknowledged updates must
/// complete only after the notification has been accepted by the destination
/// and must not overtake earlier updates.
#[async_trait]
pub trait AcpSessionUpdateSink: Send + Sync + 'static {
    /// Forwards a session update without waiting for delivery acknowledgement.
    fn update(&self, notification: wire::UpdateSessionNotification) -> Result<(), AcpRuntimeError>;

    /// Forwards a session update and waits until the destination accepts it.
    async fn update_acknowledged(
        &self,
        notification: wire::UpdateSessionNotification,
    ) -> Result<(), AcpRuntimeError>;

    /// Waits until all earlier updates have been accepted by the destination.
    async fn flush(&self) -> Result<(), AcpRuntimeError>;
}

enum ClientMessage {
    Update(Box<wire::UpdateSessionNotification>),
    AcknowledgedUpdate {
        notification: Box<wire::UpdateSessionNotification>,
        acknowledged: oneshot::Sender<Result<(), AcpRuntimeError>>,
    },
    Flush(oneshot::Sender<()>),
}

#[derive(Clone)]
struct ClientHandle {
    tx: mpsc::UnboundedSender<ClientMessage>,
    #[cfg(test)]
    before_update: Option<Arc<dyn Fn(&wire::SessionUpdate) + Send + Sync>>,
}

impl ClientHandle {
    fn channel() -> (Self, mpsc::UnboundedReceiver<ClientMessage>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                tx,
                #[cfg(test)]
                before_update: None,
            },
            rx,
        )
    }

    #[cfg(test)]
    fn channel_with_update_hook(
        hook: impl Fn(&wire::SessionUpdate) + Send + Sync + 'static,
    ) -> (Self, mpsc::UnboundedReceiver<ClientMessage>) {
        let (mut client, rx) = Self::channel();
        client.before_update = Some(Arc::new(hook));
        (client, rx)
    }

    fn update_for(
        &self,
        session_id: wire::SessionId,
        update: wire::SessionUpdate,
    ) -> Result<(), AcpRuntimeError> {
        #[cfg(test)]
        if let Some(hook) = &self.before_update {
            hook(&update);
        }
        self.update(wire::UpdateSessionNotification::new(session_id, update))
    }
}

#[async_trait]
impl AcpSessionUpdateSink for ClientHandle {
    fn update(&self, notification: wire::UpdateSessionNotification) -> Result<(), AcpRuntimeError> {
        self.tx
            .send(ClientMessage::Update(Box::new(notification)))
            .map_err(|_| AcpRuntimeError::ClientClosed)
    }

    async fn update_acknowledged(
        &self,
        notification: wire::UpdateSessionNotification,
    ) -> Result<(), AcpRuntimeError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(ClientMessage::AcknowledgedUpdate {
                notification: Box::new(notification),
                acknowledged: tx,
            })
            .map_err(|_| AcpRuntimeError::ClientClosed)?;
        rx.await.map_err(|_| AcpRuntimeError::ClientClosed)?
    }

    async fn flush(&self) -> Result<(), AcpRuntimeError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(ClientMessage::Flush(tx))
            .map_err(|_| AcpRuntimeError::ClientClosed)?;
        rx.await.map_err(|_| AcpRuntimeError::ClientClosed)
    }
}

async fn drain_client_messages(
    mut rx: mpsc::UnboundedReceiver<ClientMessage>,
    cx: V2ConnectionTo<Client>,
) {
    while let Some(message) = rx.recv().await {
        match message {
            ClientMessage::Update(notification) => {
                if let Err(error) = cx.send_notification(*notification) {
                    tracing::debug!(%error, "failed to send ACP v2 session update");
                    break;
                }
            }
            ClientMessage::AcknowledgedUpdate {
                notification,
                acknowledged,
            } => {
                let result = cx
                    .send_notification(*notification)
                    .map_err(|_| AcpRuntimeError::ClientClosed);
                let failed = result.is_err();
                let _ = acknowledged.send(result);
                if failed {
                    break;
                }
            }
            ClientMessage::Flush(response) => {
                let _ = response.send(());
            }
        }
    }
}

#[derive(Clone)]
struct CurrentMessageIds {
    agent: wire::MessageId,
    thought: wire::MessageId,
}

struct IntegrationSession {
    acp_session_id: wire::SessionId,
    agentkit_session_id: AgentkitSessionId,
    sink: Arc<dyn AcpSessionUpdateSink>,
    cancellation: CancellationController,
    closed: AtomicBool,
    lifecycle: Arc<Mutex<()>>,
    #[cfg(feature = "unstable-inject")]
    injection: Arc<InjectionController>,
    next_message: AtomicU64,
    current_messages: Mutex<Option<CurrentMessageIds>>,
    part_kinds: Mutex<HashMap<PartId, PartKind>>,
    unsupported_approval: Mutex<Option<(CancellationHandle, u64)>>,
    prompt_state: Mutex<Option<PromptState>>,
}

impl IntegrationSession {
    fn update_for(
        &self,
        session_id: wire::SessionId,
        update: wire::SessionUpdate,
    ) -> Result<(), AcpRuntimeError> {
        self.sink
            .update(wire::UpdateSessionNotification::new(session_id, update))
    }
}

struct PromptState {
    active_prompt: Arc<AtomicU64>,
    lifecycle: Arc<Mutex<()>>,
    pending_owner: Option<PromptOwner>,
    turn_owners: HashMap<TurnId, PromptOwner>,
}

#[derive(Clone)]
struct PromptOwner {
    id: u64,
    cancellation: TurnCancellation,
}

/// Unstable host-owned binding for one ACP v2 session.
///
/// This API follows the experimental ACP v2 schema and may change with the
/// pinned SDK. The update sink lets hosts keep transport ownership.
pub struct AcpSessionBinding {
    acp_session_id: wire::SessionId,
    agentkit_session_id: AgentkitSessionId,
    sink: Arc<dyn AcpSessionUpdateSink>,
    cancellation: Option<CancellationController>,
}

impl AcpSessionBinding {
    /// Creates an unstable ACP v2 session binding.
    #[must_use]
    pub fn new(
        acp_session_id: wire::SessionId,
        agentkit_session_id: AgentkitSessionId,
        sink: impl AcpSessionUpdateSink,
    ) -> Self {
        Self {
            acp_session_id,
            agentkit_session_id,
            sink: Arc::new(sink),
            cancellation: None,
        }
    }

    /// Uses a host-owned cancellation controller for this unstable v2 session.
    #[must_use]
    pub fn cancellation(mut self, cancellation: CancellationController) -> Self {
        self.cancellation = Some(cancellation);
        self
    }
}

/// Unstable host handle for one bound ACP v2 session.
#[derive(Clone)]
pub struct AcpSessionHandle {
    session: Arc<IntegrationSession>,
}

impl AcpSessionHandle {
    /// Returns the client-visible ACP v2 session ID.
    #[must_use]
    pub fn acp_session_id(&self) -> &wire::SessionId {
        &self.session.acp_session_id
    }

    /// Returns the agentkit loop session ID.
    #[must_use]
    pub fn agentkit_session_id(&self) -> &AgentkitSessionId {
        &self.session.agentkit_session_id
    }

    /// Returns the cancellation handle for this unstable v2 session.
    #[must_use]
    pub fn cancellation_handle(&self) -> CancellationHandle {
        self.session.cancellation.handle()
    }

    /// Interrupts the active turn, preserving response-committed steers.
    pub fn interrupt(&self) {
        let _lifecycle = self
            .session
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !self.session.closed.load(Ordering::Acquire) {
            #[cfg(feature = "unstable-inject")]
            self.session.injection.cancel_turn();
            self.session.cancellation.interrupt();
        }
    }

    /// Closes this unstable v2 session and discards deliverable steers.
    pub fn close(&self) {
        let _lifecycle = self
            .session
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.session.closed.store(true, Ordering::Release);
        #[cfg(feature = "unstable-inject")]
        self.session.injection.close_session();
        self.session.cancellation.interrupt();
    }

    fn is_closed(&self) -> bool {
        self.session.closed.load(Ordering::Acquire)
    }

    /// Prepares a prompt admission for unstable ACP v2 injection.
    #[cfg(feature = "unstable-inject")]
    pub fn prepare_injection_turn(&self) {
        self.session.injection.reset_cancellation();
    }

    /// Makes the admitted turn available to unstable ACP v2 injection.
    #[cfg(feature = "unstable-inject")]
    pub fn start_injection_turn(&self) {
        self.session.injection.start_turn();
    }

    /// Stops accepting unstable ACP v2 injection for the current turn.
    #[cfg(feature = "unstable-inject")]
    pub fn stop_injection_turn(&self) {
        self.session.injection.stop_turn();
    }
}

#[derive(Default)]
struct IntegrationInner {
    by_acp: HashMap<wire::SessionId, Arc<IntegrationSession>>,
    by_agentkit: HashMap<AgentkitSessionId, wire::SessionId>,
}

#[cfg(feature = "unstable-inject")]
#[derive(Default)]
struct InjectRequestOrder {
    next: AtomicU64,
    current: AtomicU64,
    changed: Notify,
}

/// Routes agentkit loop output and coordinates unstable ACP v2 host sessions.
///
/// Agent factories should install this value as their loop observer. Hosts can
/// bind sessions directly; [`AcpHeadlessRuntime`] delegates to the same public
/// coordinator. This entire API follows the experimental ACP v2 schema.
#[derive(Clone)]
pub struct AcpIntegration {
    inner: Arc<RwLock<IntegrationInner>>,
    #[cfg(feature = "unstable-inject")]
    inject_requests: Arc<InjectRequestOrder>,
}

impl Default for AcpIntegration {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(IntegrationInner::default())),
            #[cfg(feature = "unstable-inject")]
            inject_requests: Arc::new(InjectRequestOrder::default()),
        }
    }
}

impl AcpIntegration {
    /// Binds an unstable ACP v2 session to the public coordinator.
    pub fn bind_session(
        &self,
        binding: AcpSessionBinding,
    ) -> Result<AcpSessionHandle, AcpRuntimeError> {
        let cancellation = binding.cancellation.unwrap_or_default();
        let acp_session_id = binding.acp_session_id;
        let agentkit_session_id = binding.agentkit_session_id;
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(|error| error.into_inner());
        if inner.by_acp.contains_key(&acp_session_id) {
            return Err(AcpRuntimeError::SessionAlreadyBound(
                acp_session_id.to_string(),
            ));
        }
        if inner.by_agentkit.contains_key(&agentkit_session_id) {
            return Err(AcpRuntimeError::SessionAlreadyBound(
                agentkit_session_id.to_string(),
            ));
        }
        inner
            .by_agentkit
            .insert(agentkit_session_id.clone(), acp_session_id.clone());
        let session = Arc::new(IntegrationSession {
            acp_session_id: acp_session_id.clone(),
            agentkit_session_id,
            sink: binding.sink,
            cancellation,
            closed: AtomicBool::new(false),
            lifecycle: Arc::new(Mutex::new(())),
            #[cfg(feature = "unstable-inject")]
            injection: Arc::new(InjectionController::default()),
            next_message: AtomicU64::new(1),
            current_messages: Mutex::new(None),
            part_kinds: Mutex::new(HashMap::new()),
            unsupported_approval: Mutex::new(None),
            prompt_state: Mutex::new(None),
        });
        inner.by_acp.insert(acp_session_id, Arc::clone(&session));
        Ok(AcpSessionHandle { session })
    }

    #[cfg(test)]
    fn bind(
        &self,
        acp_session_id: wire::SessionId,
        agentkit_session_id: AgentkitSessionId,
        client: ClientHandle,
    ) -> Result<(), AcpRuntimeError> {
        self.bind_session(AcpSessionBinding::new(
            acp_session_id,
            agentkit_session_id,
            client,
        ))
        .map(|_| ())
    }

    /// Unbinds and closes an unstable ACP v2 session.
    pub fn unbind_session(&self, session_id: &wire::SessionId) -> Result<(), AcpRuntimeError> {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(|error| error.into_inner());
        let session = inner
            .by_acp
            .remove(session_id)
            .ok_or_else(|| AcpRuntimeError::SessionNotFound(session_id.to_string()))?;
        inner
            .by_agentkit
            .retain(|_, mapped| mapped != &session.acp_session_id);
        drop(inner);
        AcpSessionHandle { session }.close();
        Ok(())
    }

    fn session(
        &self,
        session_id: &wire::SessionId,
    ) -> Result<Arc<IntegrationSession>, AcpRuntimeError> {
        self.inner
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .by_acp
            .get(session_id)
            .cloned()
            .ok_or_else(|| AcpRuntimeError::SessionNotFound(session_id.to_string()))
    }

    fn next_user_message_id(
        &self,
        session_id: &wire::SessionId,
    ) -> Result<wire::MessageId, AcpRuntimeError> {
        let session = self.session(session_id)?;
        let sequence = session.next_message.fetch_add(1, Ordering::Relaxed);
        Ok(wire::MessageId::new(format!(
            "{session_id}-user-{sequence}"
        )))
    }

    /// Converts an unstable ACP v2 prompt into agentkit input items.
    pub fn prompt_to_items(
        &self,
        request: &wire::PromptRequest,
    ) -> Result<Vec<Item>, AcpRuntimeError> {
        self.session(&request.session_id)?;
        prompt_to_items(request)
    }

    /// Starts unstable ACP v2 prompt message routing.
    pub fn begin_prompt(
        &self,
        session_id: &wire::SessionId,
    ) -> Result<wire::MessageId, AcpRuntimeError> {
        let message_id = self.next_user_message_id(session_id)?;
        let session = self.session(session_id)?;
        finish_model_message(&session);
        Ok(message_id)
    }

    /// Finishes unstable ACP v2 prompt message routing.
    pub fn finish_prompt(&self, session_id: &wire::SessionId) {
        if let Ok(session) = self.session(session_id) {
            finish_model_message(&session);
        }
    }

    /// Flushes updates already submitted for an unstable ACP v2 session.
    pub async fn flush_session_updates(
        &self,
        session_id: &wire::SessionId,
    ) -> Result<(), AcpRuntimeError> {
        self.session(session_id)?.sink.flush().await
    }

    fn install_prompt_state(
        &self,
        session_id: &wire::SessionId,
        active_prompt: Arc<AtomicU64>,
        lifecycle: Arc<Mutex<()>>,
    ) -> Result<(), AcpRuntimeError> {
        let session = self.session(session_id)?;
        *session
            .prompt_state
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(PromptState {
            active_prompt,
            lifecycle,
            pending_owner: None,
            turn_owners: HashMap::new(),
        });
        Ok(())
    }

    fn begin_prompt_owner(
        &self,
        session_id: &wire::SessionId,
        owner: u64,
        cancellation: TurnCancellation,
    ) -> Result<wire::MessageId, AcpRuntimeError> {
        let session = self.session(session_id)?;
        {
            let mut prompt_state = session
                .prompt_state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let prompt_state = prompt_state.as_mut().ok_or_else(|| {
                AcpRuntimeError::Sdk("ACP v2 prompt state is not initialized".into())
            })?;
            prompt_state.pending_owner = Some(PromptOwner {
                id: owner,
                cancellation,
            });
        }
        let sequence = session.next_message.fetch_add(1, Ordering::Relaxed);
        Ok(wire::MessageId::new(format!(
            "{session_id}-user-{sequence}"
        )))
    }

    fn finish_prompt_owner(&self, session_id: &wire::SessionId, owner: u64) {
        if let Ok(session) = self.session(session_id) {
            if let Some(prompt_state) = session
                .prompt_state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_mut()
            {
                if prompt_state
                    .pending_owner
                    .as_ref()
                    .is_some_and(|pending| pending.id == owner)
                {
                    prompt_state.pending_owner = None;
                }
                prompt_state
                    .turn_owners
                    .retain(|_, turn_owner| turn_owner.id != owner);
            }
            finish_model_message(&session);
        }
    }

    fn mark_unsupported_approval(
        &self,
        session_id: &wire::SessionId,
        cancellation: CancellationHandle,
        generation: u64,
    ) {
        if let Ok(session) = self.session(session_id) {
            *session
                .unsupported_approval
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some((cancellation, generation));
        }
    }

    fn clear_unsupported_approval(&self, session_id: &wire::SessionId) {
        if let Ok(session) = self.session(session_id) {
            session
                .unsupported_approval
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take();
        }
    }

    fn route_turn_finished(
        session: &IntegrationSession,
        result: &agentkit_loop::TurnResult,
        unsupported_approval: Option<(CancellationHandle, u64)>,
    ) {
        let prompt_owner = session
            .prompt_state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_mut()
            .and_then(|prompt_state| {
                prompt_state
                    .turn_owners
                    .remove(&result.turn_id)
                    .map(|owner| (Arc::clone(&prompt_state.active_prompt), owner))
            });
        let Some((active_prompt, owner)) = prompt_owner else {
            return;
        };
        let prompt_cancelled = owner.cancellation.is_cancelled();
        let stop_reason = if prompt_cancelled {
            wire::StopReason::Cancelled
        } else {
            match unsupported_approval {
                Some((cancellation, generation))
                    if !cancellation.is_cancelled_since(generation) =>
                {
                    error_stop_reason()
                }
                _ => finish_reason_to_stop_reason(&result.finish_reason),
            }
        };
        release_prompt(&active_prompt, owner.id);
        if let Err(error) = session.update_for(
            session.acp_session_id.clone(),
            wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(
                wire::IdleStateUpdate::new().stop_reason(stop_reason),
            )),
        ) {
            tracing::debug!(%error, "failed to queue ACP v2 idle update");
        }
    }

    fn route_event(&self, session_id: &AgentkitSessionId, event: AgentEvent) {
        let session = {
            let inner = self.inner.read().unwrap_or_else(|error| error.into_inner());
            let Some(acp_session_id) = inner.by_agentkit.get(session_id) else {
                return;
            };
            let Some(session) = inner.by_acp.get(acp_session_id) else {
                return;
            };
            Arc::clone(session)
        };

        match &event {
            AgentEvent::TurnStarted { turn_id, .. } => {
                let prompt_owned = session
                    .prompt_state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .as_mut()
                    .and_then(|prompt_state| {
                        let owner = prompt_state.pending_owner.take()?;
                        prompt_state.turn_owners.insert(turn_id.clone(), owner);
                        Some(())
                    })
                    .is_some();
                if prompt_owned {
                    start_model_message(&session);
                    if let Err(error) = session.update_for(
                        session.acp_session_id.clone(),
                        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(
                            wire::RunningStateUpdate::new(),
                        )),
                    ) {
                        tracing::debug!(%error, "failed to queue ACP v2 running update");
                    }
                }
                return;
            }
            AgentEvent::TurnFinished(result) => {
                finish_model_message(&session);
                let unsupported_approval = session
                    .unsupported_approval
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take();
                let lifecycle = session
                    .prompt_state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .as_ref()
                    .and_then(|prompt_state| {
                        prompt_state
                            .turn_owners
                            .contains_key(&result.turn_id)
                            .then(|| Arc::clone(&prompt_state.lifecycle))
                    });
                if let Some(lifecycle) = lifecycle {
                    let _lifecycle = lifecycle.lock().unwrap_or_else(|error| error.into_inner());
                    Self::route_turn_finished(&session, result, unsupported_approval);
                }
                return;
            }
            AgentEvent::ToolExecutionStarted(_) | AgentEvent::ToolResultReceived(_) => {
                finish_model_message(&session);
            }
            AgentEvent::ContentDelta(_) | AgentEvent::ToolCallRequested(_) => {
                let has_message = session
                    .current_messages
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .is_some();
                if !has_message {
                    start_model_message(&session);
                }
            }
            _ => {}
        }

        let message_ids = session
            .current_messages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let mut part_kinds = session
            .part_kinds
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(update) = event_to_update(&event, message_ids.as_ref(), &mut part_kinds) else {
            return;
        };
        if let Err(error) = session.sink.update(wire::UpdateSessionNotification::new(
            session.acp_session_id.clone(),
            update,
        )) {
            tracing::debug!(%error, "failed to queue ACP v2 session update");
        }
    }
}

fn start_model_message(session: &IntegrationSession) {
    let sequence = session.next_message.fetch_add(1, Ordering::Relaxed);
    *session
        .current_messages
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(CurrentMessageIds {
        agent: wire::MessageId::new(format!("{}-agent-{sequence}", session.acp_session_id)),
        thought: wire::MessageId::new(format!("{}-thought-{sequence}", session.acp_session_id)),
    });
    session
        .part_kinds
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clear();
}

fn finish_model_message(session: &IntegrationSession) {
    *session
        .current_messages
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = None;
    session
        .part_kinds
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clear();
}

impl LoopObserver for AcpIntegration {
    fn handle_event(&self, event: ObservedEvent) {
        self.route_event(&event.session_id, event.event);
    }
}

/// Context passed to an ACP v2 agent factory for each new session.
#[derive(Clone)]
pub struct AcpAgentFactoryContext {
    /// ACP v2 session id visible to the client.
    pub acp_session_id: wire::SessionId,
    /// Agentkit loop session id.
    pub agentkit_session_id: AgentkitSessionId,
    /// Current working directory.
    pub cwd: PathBuf,
    /// Additional workspace roots.
    pub additional_directories: Vec<PathBuf>,
    /// ACP v2 output observer to install on the agent loop.
    pub integration: Arc<AcpIntegration>,
    /// Cancellation handle to install on the agent loop.
    pub cancellation: CancellationHandle,
    /// Session metadata.
    pub metadata: MetadataMap,
}

/// Creates one agentkit loop driver for each ACP v2 session.
#[async_trait]
pub trait AcpAgentFactory<M>: Send + Sync + 'static
where
    M: ModelAdapter,
{
    /// Builds and starts a loop driver for a new ACP v2 session.
    async fn start(
        &self,
        ctx: AcpAgentFactoryContext,
    ) -> Result<agentkit_loop::LoopDriver<M::Session>, AcpRuntimeError>;
}

/// Headless ACP v2 runtime.
pub struct AcpHeadlessRuntime<M>
where
    M: ModelAdapter,
{
    _marker: std::marker::PhantomData<M>,
}

impl<M> AcpHeadlessRuntime<M>
where
    M: ModelAdapter + Send + Sync + 'static,
    M::Session: Send + 'static,
{
    /// Starts building an ACP v2 runtime.
    #[must_use]
    pub fn builder() -> AcpHeadlessRuntimeBuilder<M> {
        AcpHeadlessRuntimeBuilder::default()
    }
}

/// Builder for [`AcpHeadlessRuntime`].
pub struct AcpHeadlessRuntimeBuilder<M>
where
    M: ModelAdapter,
{
    factory: Option<Arc<dyn AcpAgentFactory<M>>>,
    integration: AcpIntegration,
    name: String,
    version: String,
}

impl<M> Default for AcpHeadlessRuntimeBuilder<M>
where
    M: ModelAdapter,
{
    fn default() -> Self {
        Self {
            factory: None,
            integration: AcpIntegration::default(),
            name: "agentkit".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

struct ServeGuard {
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<Result<(), AcpRuntimeError>>,
}

impl Drop for ServeGuard {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

impl<M> AcpHeadlessRuntimeBuilder<M>
where
    M: ModelAdapter + Send + Sync + 'static,
    M::Session: Send + 'static,
{
    /// Sets the per-session agent factory.
    #[must_use]
    pub fn agent_factory(mut self, factory: impl AcpAgentFactory<M>) -> Self {
        self.factory = Some(Arc::new(factory));
        self
    }

    /// Uses a host-visible unstable ACP v2 session coordinator.
    ///
    /// The headless runtime delegates binding, updates, cancellation, and
    /// feature-gated injection handling to this same public value.
    #[must_use]
    pub fn integration(mut self, integration: AcpIntegration) -> Self {
        self.integration = integration;
        self
    }

    /// Sets the implementation name reported by `initialize`.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the implementation version reported by `initialize`.
    #[must_use]
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Serves ACP v2 over stdio.
    #[cfg(feature = "stdio")]
    pub async fn serve_stdio(self) -> Result<(), AcpRuntimeError> {
        self.serve(agent_client_protocol::Stdio::new()).await
    }

    /// Serves ACP v2 over a custom SDK transport.
    pub async fn serve(
        self,
        transport: impl agent_client_protocol::ConnectTo<agent_client_protocol::Agent> + 'static,
    ) -> Result<(), AcpRuntimeError> {
        let factory = self
            .factory
            .ok_or(AcpRuntimeError::MissingField("agent_factory"))?;
        let state = Arc::new(RuntimeState::new(
            factory,
            Arc::new(self.integration),
            self.name,
            self.version,
        ));
        let (shutdown, mut shutdown_rx) = oneshot::channel();
        let agent = agent_client_protocol::Agent
            .v2()
            .name(state.name.as_str())
            .on_receive_request(
                {
                    let state = Arc::clone(&state);
                    async move |request: wire::InitializeRequest, responder, _cx| {
                        responder.respond_with_result(
                            state.initialize(request).map_err(crate::sdk_error),
                        )
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let state = Arc::clone(&state);
                    async move |request: wire::NewSessionRequest, responder, cx| {
                        let state = Arc::clone(&state);
                        let connection = cx.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(
                                state
                                    .new_session(request, connection)
                                    .await
                                    .map_err(crate::sdk_error),
                            )
                        })?;
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let state = Arc::clone(&state);
                    async move |request: wire::ListSessionsRequest, responder, _cx| {
                        responder.respond_with_result(
                            state.list_sessions(request).await.map_err(crate::sdk_error),
                        )
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let state = Arc::clone(&state);
                    async move |request: wire::ResumeSessionRequest, responder, _cx| {
                        responder.respond_with_result(
                            state
                                .resume_session(request)
                                .await
                                .map_err(crate::sdk_error),
                        )
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let state = Arc::clone(&state);
                    async move |request: wire::PromptRequest, responder, cx| {
                        let state = Arc::clone(&state);
                        cx.spawn(async move {
                            match state.prompt(request).await {
                                Ok(start) => {
                                    responder.respond(wire::PromptResponse::new())?;
                                    let _ = start.send(());
                                    Ok(())
                                }
                                Err(error) => {
                                    responder.respond_with_result(Err(crate::sdk_error(error)))
                                }
                            }
                        })?;
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            );
        #[cfg(feature = "unstable-inject")]
        let agent = agent
            .on_receive_request(
                {
                    let integration = Arc::clone(&state.integration);
                    async move |request: wire::InjectSessionRequest, responder, cx| {
                        let integration = Arc::clone(&integration);
                        cx.spawn(async move {
                            integration.handle_inject_request(request, responder).await
                        })?;
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let integration = Arc::clone(&state.integration);
                    async move |request: wire::RevokeInjectSessionRequest, responder, _cx| {
                        responder.respond_with_result(integration.revoke_inject(request).await)
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let integration = Arc::clone(&state.integration);
                    async move |request: wire::ReplaceInjectSessionRequest, responder, cx| {
                        let integration = Arc::clone(&integration);
                        cx.spawn(async move {
                            responder.respond_with_result(integration.replace_inject(request).await)
                        })?;
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            );
        let connection = agent
            .on_receive_notification(
                {
                    let state = Arc::clone(&state);
                    async move |notification: wire::CancelSessionNotification, _cx| {
                        state.cancel(notification).await.map_err(crate::sdk_error)?;
                        Ok(Handled::Yes)
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                {
                    let state = Arc::clone(&state);
                    async move |request: wire::CloseSessionRequest, responder, _cx| {
                        responder.respond_with_result(
                            state.close(request).await.map_err(crate::sdk_error),
                        )
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_to(transport);
        let task = tokio::spawn(async move {
            tokio::pin!(connection);
            let result = tokio::select! {
                result = &mut connection => {
                    result.map_err(|error| AcpRuntimeError::Sdk(error.to_string()))
                }
                _ = &mut shutdown_rx => Ok(()),
            };
            state.shutdown().await;
            result
        });
        let mut guard = ServeGuard {
            shutdown: Some(shutdown),
            task,
        };
        let result = (&mut guard.task)
            .await
            .map_err(|error| AcpRuntimeError::Sdk(error.to_string()));
        guard.shutdown.take();
        result?
    }
}

/// Maximum number of pending unstable ACP v2 injections per session.
#[cfg(feature = "unstable-inject")]
pub const MAX_PENDING_INJECTIONS: usize = 64;
/// Maximum serialized content bytes retained by pending unstable injections.
#[cfg(feature = "unstable-inject")]
pub const MAX_PENDING_INJECTION_BYTES: usize = 256 * 1024;
/// Maximum accepted unstable injections tracked during one session lifetime.
///
/// This caps every accepted injection ID retained for `already_delivered`
/// classification. At 4,096 compact IDs, lifetime tracking remains bounded
/// without evicting classifications before the session closes.
#[cfg(feature = "unstable-inject")]
pub const MAX_ACCEPTED_INJECTIONS: usize = 4_096;

#[cfg(feature = "unstable-inject")]
fn validate_inject_content(
    content: &[wire::ContentBlock],
) -> Result<(Vec<Item>, usize), agent_client_protocol::Error> {
    let items = content_blocks_to_items(content).map_err(crate::sdk_error)?;
    let bytes = serde_json::to_vec(content)
        .map_err(|error| agent_client_protocol::Error::new(-32603, error.to_string()))?
        .len();
    Ok((items, bytes))
}

#[cfg(feature = "unstable-inject")]
struct PendingInject {
    message_id: wire::MessageId,
    content: Vec<wire::ContentBlock>,
    items: Vec<Item>,
    bytes: usize,
    commitment: InjectCommitment,
}

#[cfg(feature = "unstable-inject")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InjectCommitment {
    Reserved,
    Committed,
    Ready,
}

#[cfg(feature = "unstable-inject")]
#[derive(Default)]
struct InjectionState {
    running: bool,
    cancelled: bool,
    at_boundary: bool,
    pending: VecDeque<PendingInject>,
    delivering: Option<wire::MessageId>,
    pending_bytes: usize,
    accepted_count: usize,
    delivered: VecDeque<wire::MessageId>,
}

#[cfg(feature = "unstable-inject")]
#[derive(Default)]
struct InjectionController {
    state: Mutex<InjectionState>,
    changed: Notify,
}

#[cfg(feature = "unstable-inject")]
enum BoundaryAction {
    Wait,
    Deliver(PendingInject),
    Complete(AcpInjectionBoundary),
}

#[cfg(feature = "unstable-inject")]
enum PendingTransition {
    Applied,
    WaitForDelivery,
    AlreadyDelivered,
    Unknown,
}

#[cfg(feature = "unstable-inject")]
impl InjectionController {
    fn start_turn(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.running = !state.cancelled;
        state.at_boundary = false;
    }

    fn reset_cancellation(&self) {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .cancelled = false;
    }

    fn reserve(&self, pending: PendingInject) -> Result<(), agent_client_protocol::Error> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.running {
            return Err(sdk_v2_error(wire::Error::inject_no_running_turn()));
        }
        let reserved_count = state
            .pending
            .iter()
            .filter(|pending| pending.commitment == InjectCommitment::Reserved)
            .count();
        if state.accepted_count.saturating_add(reserved_count) >= MAX_ACCEPTED_INJECTIONS {
            return Err(agent_client_protocol::Error::new(
                -32000,
                "session injection lifetime limit exceeded",
            )
            .data(serde_json::json!({
                "reason": "lifetime_limit_exceeded",
                "limit": MAX_ACCEPTED_INJECTIONS,
            })));
        }
        if state.pending.len() + usize::from(state.delivering.is_some()) >= MAX_PENDING_INJECTIONS
            || state.pending_bytes.saturating_add(pending.bytes) > MAX_PENDING_INJECTION_BYTES
        {
            return Err(agent_client_protocol::Error::new(
                -32602,
                "pending session injection budget exceeded",
            ));
        }
        state.pending_bytes += pending.bytes;
        state.pending.push_back(pending);
        Ok(())
    }

    fn commit(&self, message_id: &wire::MessageId) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let pending = state
            .pending
            .iter_mut()
            .find(|pending| &pending.message_id == message_id)
            .expect("reserved injection remains until response commitment");
        debug_assert_eq!(pending.commitment, InjectCommitment::Reserved);
        pending.commitment = InjectCommitment::Committed;
        state.accepted_count += 1;
    }

    fn activate(&self, message_id: &wire::MessageId) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(pending) = state
            .pending
            .iter_mut()
            .find(|pending| &pending.message_id == message_id)
        {
            debug_assert_eq!(pending.commitment, InjectCommitment::Committed);
            pending.commitment = InjectCommitment::Ready;
        }
        drop(state);
        self.changed.notify_waiters();
    }

    fn discard(&self, message_id: &wire::MessageId) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        Self::remove_pending(&mut state, message_id);
        drop(state);
        self.changed.notify_waiters();
    }

    fn remove_pending(state: &mut InjectionState, message_id: &wire::MessageId) -> bool {
        let Some(index) = state
            .pending
            .iter()
            .position(|pending| &pending.message_id == message_id)
        else {
            return false;
        };
        if let Some(pending) = state.pending.remove(index) {
            state.pending_bytes = state.pending_bytes.saturating_sub(pending.bytes);
        }
        true
    }

    fn boundary_action(&self, terminal: bool, delivered_any: bool) -> BoundaryAction {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.at_boundary = true;
        if !state.running {
            state.at_boundary = false;
            return BoundaryAction::Complete(AcpInjectionBoundary::Stopped);
        }
        if state
            .pending
            .front()
            .is_some_and(|pending| pending.commitment != InjectCommitment::Ready)
        {
            return BoundaryAction::Wait;
        }
        if let Some(pending) = state.pending.pop_front() {
            state.delivering = Some(pending.message_id.clone());
            return BoundaryAction::Deliver(pending);
        }
        state.at_boundary = false;
        if delivered_any {
            BoundaryAction::Complete(AcpInjectionBoundary::Delivered)
        } else if terminal {
            state.running = false;
            BoundaryAction::Complete(AcpInjectionBoundary::Finished)
        } else {
            BoundaryAction::Complete(AcpInjectionBoundary::Continue)
        }
    }

    fn finish_delivery(&self, pending: &PendingInject, delivered: bool) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        debug_assert_eq!(state.delivering.as_ref(), Some(&pending.message_id));
        state.delivering = None;
        state.pending_bytes = state.pending_bytes.saturating_sub(pending.bytes);
        if delivered {
            state.delivered.push_back(pending.message_id.clone());
        }
        drop(state);
        self.changed.notify_waiters();
    }

    fn replace_transition(
        &self,
        message_id: &wire::MessageId,
        content: &[wire::ContentBlock],
        items: &[Item],
        bytes: usize,
    ) -> Result<PendingTransition, agent_client_protocol::Error> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(index) = state
            .pending
            .iter()
            .position(|pending| &pending.message_id == message_id)
        {
            if state.pending[index].commitment != InjectCommitment::Ready {
                return Ok(PendingTransition::WaitForDelivery);
            }
            // Include other pending entries and any in-flight delivery in the
            // budget. Check before mutation so failure preserves the old content.
            let pending_bytes = state.pending_bytes - state.pending[index].bytes;
            if pending_bytes.saturating_add(bytes) > MAX_PENDING_INJECTION_BYTES {
                return Err(agent_client_protocol::Error::new(
                    -32602,
                    "pending session injection budget exceeded",
                ));
            }
            let pending = &mut state.pending[index];
            pending.content = content.to_vec();
            pending.items = items.to_vec();
            pending.bytes = bytes;
            state.pending_bytes = pending_bytes + bytes;
            // Do not remove/re-enqueue: identity, commitment, and FIFO position
            // must survive replacement. The state lock also serializes delivery.
            Ok(PendingTransition::Applied)
        } else if state.delivering.as_ref() == Some(message_id) {
            Ok(PendingTransition::WaitForDelivery)
        } else if state.delivered.contains(message_id) {
            Ok(PendingTransition::AlreadyDelivered)
        } else {
            Ok(PendingTransition::Unknown)
        }
    }

    fn revoke_transition(&self, message_id: &wire::MessageId) -> PendingTransition {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let ready = state.pending.iter().any(|pending| {
            &pending.message_id == message_id && pending.commitment == InjectCommitment::Ready
        });
        if ready && Self::remove_pending(&mut state, message_id) {
            drop(state);
            self.changed.notify_waiters();
            PendingTransition::Applied
        } else if state.delivering.as_ref() == Some(message_id)
            || state
                .pending
                .iter()
                .any(|pending| &pending.message_id == message_id)
        {
            PendingTransition::WaitForDelivery
        } else if state.delivered.contains(message_id) {
            PendingTransition::AlreadyDelivered
        } else {
            PendingTransition::Unknown
        }
    }

    fn cancel_turn(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.running = false;
        state.cancelled = true;
        state.at_boundary = false;
        // Accepted steers survive cancellation and become deliverable at the
        // next valid boundary, potentially in the next prompt.
        drop(state);
        self.changed.notify_waiters();
    }

    fn stop_turn(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.running = false;
        state.at_boundary = false;
        drop(state);
        self.changed.notify_waiters();
    }

    fn close_session(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.running = false;
        state.cancelled = true;
        state.at_boundary = false;
        let discarded_bytes = state
            .pending
            .iter()
            .filter(|pending| pending.commitment == InjectCommitment::Ready)
            .map(|pending| pending.bytes)
            .sum::<usize>();
        state
            .pending
            .retain(|pending| pending.commitment != InjectCommitment::Ready);
        state.pending_bytes = state.pending_bytes.saturating_sub(discarded_bytes);
        drop(state);
        self.changed.notify_waiters();
    }
}

#[cfg(feature = "unstable-inject")]
struct ReservedInject {
    session: Arc<IntegrationSession>,
    message_id: Option<wire::MessageId>,
}

#[cfg(feature = "unstable-inject")]
impl ReservedInject {
    fn response(&self) -> wire::InjectSessionResponse {
        wire::InjectSessionResponse::new(
            self.message_id
                .as_ref()
                .expect("reserved injection has a message id")
                .clone(),
        )
    }

    fn message_id(&self) -> &wire::MessageId {
        self.message_id
            .as_ref()
            .expect("reserved injection has a message id")
    }

    fn commit(&mut self) {
        let message_id = self.message_id();
        self.session.injection.commit(message_id);
    }

    fn activate(mut self) {
        let message_id = self
            .message_id
            .take()
            .expect("reserved injection has a message id");
        let lifecycle = self
            .session
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.session.closed.load(Ordering::Acquire) {
            self.session.injection.discard(&message_id);
        } else {
            self.session.injection.activate(&message_id);
        }
        drop(lifecycle);
    }

    fn discard(mut self) {
        let message_id = self
            .message_id
            .take()
            .expect("reserved injection has a message id");
        self.session.injection.discard(&message_id);
    }
}

#[cfg(feature = "unstable-inject")]
impl Drop for ReservedInject {
    fn drop(&mut self) {
        if let Some(message_id) = self.message_id.take() {
            self.session.injection.discard(&message_id);
        }
    }
}

#[cfg(feature = "unstable-inject")]
struct InjectRequestPermit {
    integration: AcpIntegration,
    sequence: Option<u64>,
}

#[cfg(feature = "unstable-inject")]
impl Drop for InjectRequestPermit {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.integration.finish_inject_request(sequence);
        }
    }
}

/// Opaque reservation for one globally ordered unstable ACP v2 inject request.
///
/// The handle owns the SDK responder so the response cannot be committed
/// outside the session lifecycle lock or against a different cancellation
/// token. Dropping it discards the reservation and advances request ordering.
#[cfg(feature = "unstable-inject")]
#[must_use = "the reserved inject request must be responded to or dropped"]
pub struct AcpInjectRequest {
    reserved: Option<ReservedInject>,
    responder: Option<agent_client_protocol::Responder<wire::InjectSessionResponse>>,
    cancellation: agent_client_protocol::RequestCancellation,
    _permit: InjectRequestPermit,
}

#[cfg(feature = "unstable-inject")]
impl AcpInjectRequest {
    /// Returns the response that will be committed by [`respond_tracked`](Self::respond_tracked).
    #[must_use]
    pub fn response(&self) -> wire::InjectSessionResponse {
        self.reserved
            .as_ref()
            .expect("inject request has a reservation")
            .response()
    }

    /// Commits the response under the session lifecycle lock.
    ///
    /// This checks close and request cancellation immediately before calling the
    /// SDK's tracked responder. `Ok(None)` means the coordinator sent a close or
    /// cancellation error instead. `Ok(Some(_))` returns an acceptance handle
    /// bound to the resulting [`agent_client_protocol::ResponseReceipt`].
    pub fn respond_tracked(
        mut self,
    ) -> Result<Option<AcpInjectAcceptance>, agent_client_protocol::Error> {
        let session = Arc::clone(
            &self
                .reserved
                .as_ref()
                .expect("inject request has a reservation")
                .session,
        );
        let lifecycle = session
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let rejection = if session.closed.load(Ordering::Acquire) {
            Some(session_not_found_error(&session.acp_session_id))
        } else if self.cancellation.is_cancelled() {
            Some(agent_client_protocol::Error::request_cancelled())
        } else {
            None
        };
        if let Some(error) = rejection {
            drop(lifecycle);
            self.reserved.take().expect("inject reservation").discard();
            self.responder
                .take()
                .expect("inject request has a responder")
                .respond_with_result(Err(error))?;
            return Ok(None);
        }

        let response = self.response();
        let receipt = match self
            .responder
            .take()
            .expect("inject request has a responder")
            .respond_tracked(response)
        {
            Ok(receipt) => receipt,
            Err(error) => {
                drop(lifecycle);
                self.reserved.take().expect("inject reservation").discard();
                return Err(error);
            }
        };
        let mut reserved = self.reserved.take().expect("inject reservation");
        reserved.commit();
        drop(lifecycle);
        Ok(Some(AcpInjectAcceptance {
            reserved: Some(reserved),
            receipt: Some(receipt),
        }))
    }
}

/// Opaque response-committed unstable ACP v2 injection acceptance.
///
/// This handle owns both the reservation and the SDK response receipt. Dropping
/// it before activation discards the reservation.
#[cfg(feature = "unstable-inject")]
#[must_use = "the response receipt must be activated or the injection is discarded"]
pub struct AcpInjectAcceptance {
    reserved: Option<ReservedInject>,
    receipt: Option<agent_client_protocol::ResponseReceipt>,
}

#[cfg(feature = "unstable-inject")]
impl AcpInjectAcceptance {
    /// Returns the agent-owned message ID reserved for this acceptance.
    #[must_use]
    pub fn message_id(&self) -> &wire::MessageId {
        self.reserved
            .as_ref()
            .expect("inject acceptance has a reservation")
            .message_id()
    }

    /// Waits for the SDK response receipt, then makes the steer deliverable.
    ///
    /// Batch safety requires calling this only after the receive callback has
    /// returned, normally in a spawned task. Receipt failure or future
    /// cancellation drops and discards the reservation.
    pub async fn activate_after_response(mut self) -> Result<(), agent_client_protocol::Error> {
        self.receipt
            .take()
            .expect("inject acceptance has a response receipt")
            .await?;
        self.reserved
            .take()
            .expect("inject acceptance has a reservation")
            .activate();
        Ok(())
    }
}

#[cfg(feature = "unstable-inject")]
impl AcpIntegration {
    async fn wait_for_inject_request(&self, sequence: u64) {
        loop {
            let changed = self.inject_requests.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.inject_requests.current.load(Ordering::Acquire) == sequence {
                return;
            }
            changed.await;
        }
    }

    fn finish_inject_request(&self, sequence: u64) {
        let advanced = self.inject_requests.current.compare_exchange(
            sequence,
            sequence + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        debug_assert_eq!(advanced, Ok(sequence));
        self.inject_requests.changed.notify_waiters();
    }

    fn reserve_inject(
        &self,
        request: wire::InjectSessionRequest,
    ) -> Result<ReservedInject, agent_client_protocol::Error> {
        if !matches!(request.mode, wire::SessionInjectMode::Steer) {
            return Err(agent_client_protocol::Error::new(
                -32602,
                "unsupported session injection mode",
            ));
        }
        let (items, bytes) = validate_inject_content(&request.content)?;
        let session = self
            .session(&request.session_id)
            .map_err(|_| session_not_found_error(&request.session_id))?;
        let _lifecycle = session
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if session.closed.load(Ordering::Acquire) {
            return Err(session_not_found_error(&request.session_id));
        }
        let message_id = self
            .next_user_message_id(&request.session_id)
            .map_err(crate::sdk_error)?;
        session.injection.reserve(PendingInject {
            message_id: message_id.clone(),
            content: request.content,
            items,
            bytes,
            commitment: InjectCommitment::Reserved,
        })?;
        drop(_lifecycle);
        Ok(ReservedInject {
            session,
            message_id: Some(message_id),
        })
    }

    /// Reserves one globally ordered unstable ACP v2 inject request.
    ///
    /// The SDK responder is intentionally consumed here: its cancellation token
    /// and tracked response must remain paired with this reservation. This method
    /// responds to pre-reservation cancellation or validation errors itself and
    /// returns `Ok(None)`. A returned handle owns FIFO ordering and Drop cleanup.
    pub async fn reserve_inject_request(
        &self,
        request: wire::InjectSessionRequest,
        responder: agent_client_protocol::Responder<wire::InjectSessionResponse>,
    ) -> Result<Option<AcpInjectRequest>, agent_client_protocol::Error> {
        let sequence = self.inject_requests.next.fetch_add(1, Ordering::Relaxed);
        self.wait_for_inject_request(sequence).await;
        let permit = InjectRequestPermit {
            integration: self.clone(),
            sequence: Some(sequence),
        };
        let cancellation = responder.cancellation();
        // Return the receive callback first so a queued cancellation can commit
        // before reservation or response commitment.
        tokio::task::yield_now().await;
        if cancellation.is_cancelled() {
            responder
                .respond_with_result(Err(agent_client_protocol::Error::request_cancelled()))?;
            drop(permit);
            return Ok(None);
        }
        let reserved = match self.reserve_inject(request) {
            Ok(reserved) => reserved,
            Err(error) => {
                responder.respond_with_result(Err(error))?;
                drop(permit);
                return Ok(None);
            }
        };
        Ok(Some(AcpInjectRequest {
            reserved: Some(reserved),
            responder: Some(responder),
            cancellation,
            _permit: permit,
        }))
    }

    /// Handles one unstable ACP v2 `session/inject` request end to end.
    ///
    /// This convenience path uses [`reserve_inject_request`](Self::reserve_inject_request),
    /// [`AcpInjectRequest::respond_tracked`], and
    /// [`AcpInjectAcceptance::activate_after_response`]. Hosts that need staged
    /// control can call those same methods without reimplementing the race.
    pub async fn handle_inject_request(
        &self,
        request: wire::InjectSessionRequest,
        responder: agent_client_protocol::Responder<wire::InjectSessionResponse>,
    ) -> Result<(), agent_client_protocol::Error> {
        let Some(request) = self.reserve_inject_request(request, responder).await? else {
            return Ok(());
        };
        let Some(acceptance) = request.respond_tracked()? else {
            return Ok(());
        };
        let _receipt_task = tokio::spawn(async move {
            if let Err(error) = acceptance.activate_after_response().await {
                tracing::debug!(%error, "ACP v2 inject response was not accepted");
            }
        });
        Ok(())
    }

    /// Atomically replaces the complete content of a pending injection without
    /// changing its message ID, steer mode, or queue position.
    ///
    /// Like revoke, this waits for acceptance or in-flight delivery to settle.
    /// A delivered ID returns `already_delivered`; revoked or unknown IDs return
    /// `unknown_message_id`. Cancellation does not prevent replacing a retained
    /// pending steer, but closing the session does.
    pub async fn replace_inject(
        &self,
        request: wire::ReplaceInjectSessionRequest,
    ) -> Result<wire::ReplaceInjectSessionResponse, agent_client_protocol::Error> {
        let (items, bytes) = validate_inject_content(&request.content)?;
        let session = self
            .session(&request.session_id)
            .map_err(|_| session_not_found_error(&request.session_id))?;
        loop {
            let changed = session.injection.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let lifecycle = session
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if session.closed.load(Ordering::Acquire) {
                return Err(session_not_found_error(&request.session_id));
            }
            match session.injection.replace_transition(
                &request.message_id,
                &request.content,
                &items,
                bytes,
            )? {
                PendingTransition::Applied => {
                    return Ok(wire::ReplaceInjectSessionResponse::new(request.message_id));
                }
                PendingTransition::WaitForDelivery => drop(lifecycle),
                PendingTransition::AlreadyDelivered => {
                    return Err(sdk_v2_error(wire::Error::inject_already_delivered(
                        request.message_id,
                    )));
                }
                PendingTransition::Unknown => {
                    return Err(sdk_v2_error(wire::Error::inject_unknown_message_id(
                        request.message_id,
                    )));
                }
            }
            changed.await;
        }
    }

    /// Handles the state transition for unstable ACP v2 `session/revoke_inject`.
    pub async fn revoke_inject(
        &self,
        request: wire::RevokeInjectSessionRequest,
    ) -> Result<wire::RevokeInjectSessionResponse, agent_client_protocol::Error> {
        let session = self
            .session(&request.session_id)
            .map_err(|_| session_not_found_error(&request.session_id))?;
        loop {
            let changed = session.injection.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let _lifecycle = session
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if session.closed.load(Ordering::Acquire) {
                return Err(session_not_found_error(&request.session_id));
            }
            match session.injection.revoke_transition(&request.message_id) {
                PendingTransition::Applied => {
                    return Ok(wire::RevokeInjectSessionResponse::new());
                }
                PendingTransition::WaitForDelivery => drop(_lifecycle),
                PendingTransition::AlreadyDelivered => {
                    return Err(sdk_v2_error(wire::Error::inject_already_delivered(
                        request.message_id,
                    )));
                }
                PendingTransition::Unknown => {
                    return Err(sdk_v2_error(wire::Error::inject_unknown_message_id(
                        request.message_id,
                    )));
                }
            }
            changed.await;
        }
    }
}

struct SessionEntry {
    commands: mpsc::UnboundedSender<SessionCommand>,
    session: AcpSessionHandle,
    cancellation: CancellationController,
    info: wire::SessionInfo,
    active_prompt: Arc<AtomicU64>,
    driving_prompt: Arc<AtomicU64>,
    cancelled_prompt: Arc<AtomicU64>,
    next_prompt_owner: AtomicU64,
    closed: AtomicBool,
    lifecycle: Arc<Mutex<()>>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    drain_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

enum SessionCommand {
    Prompt {
        request: wire::PromptRequest,
        items: Vec<Item>,
        prompt_cancellation: TurnCancellation,
        cancellation_generation: u64,
        owner: u64,
        start: oneshot::Receiver<()>,
    },
    Shutdown,
}

struct RuntimeState<M>
where
    M: ModelAdapter,
{
    factory: Arc<dyn AcpAgentFactory<M>>,
    integration: Arc<AcpIntegration>,
    sessions: AsyncMutex<HashMap<wire::SessionId, Arc<SessionEntry>>>,
    next_session: AtomicU64,
    name: String,
    version: String,
}

impl<M> RuntimeState<M>
where
    M: ModelAdapter + Send + Sync + 'static,
    M::Session: Send + 'static,
{
    fn new(
        factory: Arc<dyn AcpAgentFactory<M>>,
        integration: Arc<AcpIntegration>,
        name: String,
        version: String,
    ) -> Self {
        Self {
            factory,
            integration,
            sessions: AsyncMutex::new(HashMap::new()),
            next_session: AtomicU64::new(1),
            name,
            version,
        }
    }

    fn initialize(
        &self,
        request: wire::InitializeRequest,
    ) -> Result<wire::InitializeResponse, AcpRuntimeError> {
        if request.protocol_version != wire::ProtocolVersion::V2 {
            return Err(AcpRuntimeError::Unsupported(
                "ACP v2 runtime requires protocol version 2".into(),
            ));
        }
        Ok(wire::InitializeResponse::new(
            wire::ProtocolVersion::V2,
            wire::Implementation::new(self.name.clone(), self.version.clone()),
        )
        .capabilities(agent_capabilities()))
    }

    async fn new_session(
        self: &Arc<Self>,
        request: wire::NewSessionRequest,
        cx: V2ConnectionTo<Client>,
    ) -> Result<wire::NewSessionResponse, AcpRuntimeError> {
        let sequence = self.next_session.fetch_add(1, Ordering::Relaxed);
        let acp_session_id = wire::SessionId::new(format!("session-{sequence}"));
        let agentkit_session_id = AgentkitSessionId::new(acp_session_id.to_string());
        let cancellation = CancellationController::new();
        let (client, client_messages) = ClientHandle::channel();
        let info = wire::SessionInfo::new(acp_session_id.clone(), request.cwd.clone())
            .additional_directories(request.additional_directories.clone());

        let mut metadata = MetadataMap::new();
        metadata.insert("acp.protocol_version".into(), json!(2));
        metadata.insert("acp.cwd".into(), json!(request.cwd));
        metadata.insert(
            "acp.additional_directories".into(),
            json!(request.additional_directories),
        );

        let active_prompt = Arc::new(AtomicU64::new(0));
        let driving_prompt = Arc::new(AtomicU64::new(0));
        let cancelled_prompt = Arc::new(AtomicU64::new(0));
        let session = self.integration.bind_session(
            AcpSessionBinding::new(
                acp_session_id.clone(),
                agentkit_session_id.clone(),
                client.clone(),
            )
            .cancellation(cancellation.clone()),
        )?;
        let lifecycle = Arc::clone(&session.session.lifecycle);
        if let Err(error) = self.integration.install_prompt_state(
            &acp_session_id,
            Arc::clone(&active_prompt),
            Arc::clone(&lifecycle),
        ) {
            let _ = self.integration.unbind_session(&acp_session_id);
            return Err(error);
        }
        let drain_task = tokio::spawn(drain_client_messages(client_messages, cx));
        let ctx = AcpAgentFactoryContext {
            acp_session_id: acp_session_id.clone(),
            agentkit_session_id: agentkit_session_id.clone(),
            cwd: request.cwd.into_inner(),
            additional_directories: request
                .additional_directories
                .into_iter()
                .map(wire::AbsolutePath::into_inner)
                .collect(),
            integration: Arc::clone(&self.integration),
            cancellation: cancellation.handle(),
            metadata,
        };
        let driver = match self.factory.start(ctx).await {
            Ok(driver) => driver,
            Err(error) => {
                let _ = self.integration.unbind_session(&acp_session_id);
                drain_task.abort();
                let _ = drain_task.await;
                return Err(error);
            }
        };

        let (commands, rx) = mpsc::unbounded_channel();
        let worker_active_prompt = Arc::clone(&active_prompt);
        let worker_driving_prompt = Arc::clone(&driving_prompt);
        let worker_cancelled_prompt = Arc::clone(&cancelled_prompt);
        let worker_lifecycle = Arc::clone(&lifecycle);
        let integration = Arc::clone(&self.integration);
        let worker_session = session.clone();
        let worker_cancellation = cancellation.handle();
        let task = tokio::spawn(async move {
            session_worker(
                worker_session,
                driver,
                client,
                integration,
                worker_cancellation,
                worker_active_prompt,
                worker_driving_prompt,
                worker_cancelled_prompt,
                worker_lifecycle,
                rx,
            )
            .await;
        });
        let entry = Arc::new(SessionEntry {
            commands,
            session,
            cancellation,
            info,
            active_prompt,
            driving_prompt,
            cancelled_prompt,
            next_prompt_owner: AtomicU64::new(1),
            closed: AtomicBool::new(false),
            lifecycle,
            task: Mutex::new(Some(task)),
            drain_task: Mutex::new(Some(drain_task)),
        });
        self.sessions
            .lock()
            .await
            .insert(acp_session_id.clone(), entry);
        Ok(wire::NewSessionResponse::new(acp_session_id))
    }

    async fn list_sessions(
        &self,
        request: wire::ListSessionsRequest,
    ) -> Result<wire::ListSessionsResponse, AcpRuntimeError> {
        if request.cursor.is_some() {
            return Err(AcpRuntimeError::Unsupported(
                "ACP v2 session list cursors are not supported".into(),
            ));
        }
        let sessions = self.sessions.lock().await;
        let mut infos = sessions
            .values()
            .filter(|entry| {
                !entry.session.is_closed()
                    && request
                        .cwd
                        .as_ref()
                        .is_none_or(|cwd| cwd == &entry.info.cwd)
            })
            .map(|entry| entry.info.clone())
            .collect::<Vec<_>>();
        infos.sort_by(|left, right| {
            left.session_id
                .to_string()
                .cmp(&right.session_id.to_string())
        });
        Ok(wire::ListSessionsResponse::new(infos))
    }

    async fn resume_session(
        &self,
        request: wire::ResumeSessionRequest,
    ) -> Result<wire::ResumeSessionResponse, AcpRuntimeError> {
        if request.replay_from.is_some() {
            return Err(AcpRuntimeError::Unsupported(
                "ACP v2 session replay is not supported".into(),
            ));
        }
        let entry = self
            .sessions
            .lock()
            .await
            .get(&request.session_id)
            .cloned()
            .ok_or_else(|| AcpRuntimeError::SessionNotFound(request.session_id.to_string()))?;
        if entry.session.is_closed() || entry.info.cwd != request.cwd {
            return Err(AcpRuntimeError::SessionNotFound(
                request.session_id.to_string(),
            ));
        }
        if entry.info.additional_directories != request.additional_directories {
            return Err(AcpRuntimeError::Unsupported(
                "changing ACP v2 session directories on resume is not supported".into(),
            ));
        }
        Ok(wire::ResumeSessionResponse::new())
    }

    async fn prompt(
        &self,
        request: wire::PromptRequest,
    ) -> Result<oneshot::Sender<()>, AcpRuntimeError> {
        let items = self.integration.prompt_to_items(&request)?;
        let entry = self
            .sessions
            .lock()
            .await
            .get(&request.session_id)
            .cloned()
            .ok_or_else(|| AcpRuntimeError::SessionNotFound(request.session_id.to_string()))?;
        let (start_tx, start_rx) = oneshot::channel();
        {
            let _lifecycle = entry
                .session
                .session
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if entry.session.is_closed() {
                return Err(AcpRuntimeError::SessionNotFound(
                    request.session_id.to_string(),
                ));
            }
            let mut owner = entry.next_prompt_owner.fetch_add(1, Ordering::Relaxed);
            if owner == 0 {
                owner = entry.next_prompt_owner.fetch_add(1, Ordering::Relaxed);
            }
            if entry
                .active_prompt
                .compare_exchange(0, owner, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(AcpRuntimeError::Unsupported(
                    "session is already running a prompt".into(),
                ));
            }
            let prompt_cancellation = entry.cancellation.handle().checkpoint();
            #[cfg(feature = "unstable-inject")]
            entry.session.prepare_injection_turn();
            let cancellation_generation = prompt_cancellation.generation();
            if entry
                .commands
                .send(SessionCommand::Prompt {
                    request,
                    items,
                    prompt_cancellation,
                    cancellation_generation,
                    owner,
                    start: start_rx,
                })
                .is_err()
            {
                release_prompt(&entry.active_prompt, owner);
                return Err(AcpRuntimeError::ClientClosed);
            }
        }
        Ok(start_tx)
    }

    async fn cancel(
        &self,
        notification: wire::CancelSessionNotification,
    ) -> Result<(), AcpRuntimeError> {
        let entry = self
            .sessions
            .lock()
            .await
            .get(&notification.session_id)
            .cloned();
        if let Some(entry) = entry {
            let _lifecycle = entry
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !entry.closed.load(Ordering::Acquire) {
                let driving = entry.driving_prompt.load(Ordering::Acquire);
                if driving != 0 {
                    #[cfg(feature = "unstable-inject")]
                    entry.session.session.injection.cancel_turn();
                    entry.cancellation.interrupt();
                } else {
                    let queued = entry.active_prompt.load(Ordering::Acquire);
                    if queued != 0 {
                        entry.cancelled_prompt.store(queued, Ordering::Release);
                    } else {
                        #[cfg(feature = "unstable-inject")]
                        entry.session.session.injection.cancel_turn();
                        entry.cancellation.interrupt();
                    }
                }
            }
        }
        Ok(())
    }

    async fn close(
        &self,
        request: wire::CloseSessionRequest,
    ) -> Result<wire::CloseSessionResponse, AcpRuntimeError> {
        let entry = self
            .sessions
            .lock()
            .await
            .remove(&request.session_id)
            .ok_or_else(|| AcpRuntimeError::SessionNotFound(request.session_id.to_string()))?;
        stop_session(Arc::clone(&entry)).await;
        self.integration.unbind_session(&request.session_id)?;
        stop_client(entry).await;
        Ok(wire::CloseSessionResponse::new())
    }

    async fn shutdown(&self) {
        let sessions = {
            let mut sessions = self.sessions.lock().await;
            sessions.drain().collect::<Vec<_>>()
        };
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut session_tasks = Vec::with_capacity(sessions.len());
        for (session_id, entry) in &sessions {
            signal_session_stop(entry);
            if let Some(task) = take_task(&entry.task) {
                session_tasks.push(task);
            }
            let _ = self.integration.unbind_session(session_id);
        }
        join_tasks_until(deadline, session_tasks).await;

        let drain_tasks = sessions
            .iter()
            .filter_map(|(_, entry)| take_task(&entry.drain_task))
            .collect();
        drop(sessions);
        join_tasks_until(deadline, drain_tasks).await;
    }
}

fn signal_session_stop(entry: &Arc<SessionEntry>) {
    entry.closed.store(true, Ordering::Release);
    entry.session.close();
    let _ = entry.commands.send(SessionCommand::Shutdown);
}

fn take_task(
    task: &Mutex<Option<tokio::task::JoinHandle<()>>>,
) -> Option<tokio::task::JoinHandle<()>> {
    task.lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
}

async fn join_tasks_until(
    deadline: tokio::time::Instant,
    mut tasks: Vec<tokio::task::JoinHandle<()>>,
) {
    if tokio::time::timeout_at(deadline, async {
        for task in &mut tasks {
            let _ = task.await;
        }
    })
    .await
    .is_err()
    {
        for task in tasks {
            task.abort();
        }
    }
}

async fn stop_session(entry: Arc<SessionEntry>) {
    signal_session_stop(&entry);
    if let Some(task) = take_task(&entry.task) {
        join_tasks_until(
            tokio::time::Instant::now() + std::time::Duration::from_secs(2),
            vec![task],
        )
        .await;
    }
}

async fn stop_client(entry: Arc<SessionEntry>) {
    let task = take_task(&entry.drain_task);
    drop(entry);
    if let Some(task) = task {
        let _ = task.await;
    }
}

fn release_prompt(active_prompt: &AtomicU64, owner: u64) {
    let _ = active_prompt.compare_exchange(owner, 0, Ordering::AcqRel, Ordering::Acquire);
}

fn clear_prompt_tracking(
    active_prompt: &AtomicU64,
    driving_prompt: &AtomicU64,
    cancelled_prompt: &AtomicU64,
    owner: u64,
) {
    let _ = driving_prompt.compare_exchange(owner, 0, Ordering::AcqRel, Ordering::Acquire);
    let _ = cancelled_prompt.compare_exchange(owner, 0, Ordering::AcqRel, Ordering::Acquire);
    release_prompt(active_prompt, owner);
}

#[allow(clippy::too_many_arguments)]
fn fail_accepted_prompt(
    client: &ClientHandle,
    integration: &AcpIntegration,
    session_id: &wire::SessionId,
    active_prompt: &AtomicU64,
    driving_prompt: &AtomicU64,
    cancelled_prompt: &AtomicU64,
    lifecycle: &Mutex<()>,
    owner: u64,
    prompt_began: bool,
) {
    let _lifecycle = lifecycle.lock().unwrap_or_else(|error| error.into_inner());
    if prompt_began {
        integration.finish_prompt_owner(session_id, owner);
    }
    clear_prompt_tracking(active_prompt, driving_prompt, cancelled_prompt, owner);
    if let Err(error) = client.update_for(
        session_id.clone(),
        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(
            wire::IdleStateUpdate::new().stop_reason(error_stop_reason()),
        )),
    ) {
        tracing::debug!(%error, owner, "failed to queue ACP v2 error idle update");
    }
}

async fn session_worker<S>(
    session: AcpSessionHandle,
    mut driver: agentkit_loop::LoopDriver<S>,
    client: ClientHandle,
    integration: Arc<AcpIntegration>,
    cancellation: CancellationHandle,
    active_prompt: Arc<AtomicU64>,
    driving_prompt: Arc<AtomicU64>,
    cancelled_prompt: Arc<AtomicU64>,
    lifecycle: Arc<Mutex<()>>,
    mut commands: mpsc::UnboundedReceiver<SessionCommand>,
) where
    S: ModelSession + Send + 'static,
{
    let session_id = session.acp_session_id().clone();
    enum SessionAction {
        Command(Option<SessionCommand>),
        LoopUpdate(Result<(), LoopError>),
    }

    // Let one fresh prompt win a race with unrelated background work, then
    // prefer the deferred update so repeated prompts cannot starve its delivery.
    let mut prefer_loop_update = false;
    loop {
        let action = if prefer_loop_update {
            tokio::select! {
                biased;
                wake = driver.wait_for_loop_update() => SessionAction::LoopUpdate(wake),
                command = commands.recv() => SessionAction::Command(command),
            }
        } else {
            tokio::select! {
                biased;
                command = commands.recv() => SessionAction::Command(command),
                wake = driver.wait_for_loop_update() => SessionAction::LoopUpdate(wake),
            }
        };
        let command = match action {
            SessionAction::Command(Some(command)) => command,
            SessionAction::Command(None) => break,
            SessionAction::LoopUpdate(wake) => {
                if let Err(error) = wake {
                    tracing::debug!(%error, "failed waiting for an ACP v2 loop update");
                    break;
                }
                prefer_loop_update = false;
                drive_prompt(
                    &mut driver,
                    &integration,
                    &session_id,
                    &cancellation,
                    cancellation.generation(),
                    #[cfg(feature = "unstable-inject")]
                    &session,
                )
                .await;
                if let Err(error) = client.flush().await {
                    tracing::debug!(%error, "failed to flush idle ACP v2 loop update");
                }
                continue;
            }
        };
        let SessionCommand::Prompt {
            request,
            items,
            prompt_cancellation,
            cancellation_generation,
            owner,
            start,
        } = command
        else {
            break;
        };
        if start.await.is_err() {
            let _lifecycle = lifecycle.lock().unwrap_or_else(|error| error.into_inner());
            clear_prompt_tracking(&active_prompt, &driving_prompt, &cancelled_prompt, owner);
            continue;
        }
        let should_start = {
            let _lifecycle = lifecycle.lock().unwrap_or_else(|error| error.into_inner());
            let still_active = active_prompt.load(Ordering::Acquire) == owner;
            let was_cancelled = cancelled_prompt.load(Ordering::Acquire) == owner;
            if !still_active || was_cancelled {
                clear_prompt_tracking(&active_prompt, &driving_prompt, &cancelled_prompt, owner);
                false
            } else {
                driving_prompt.store(owner, Ordering::Release);
                true
            }
        };
        if !should_start {
            #[cfg(feature = "unstable-inject")]
            session.stop_injection_turn();
            continue;
        }
        #[cfg(feature = "unstable-inject")]
        session.start_injection_turn();

        if let Err(error) = driver.submit_input(items) {
            tracing::debug!(%error, owner, "failed to submit accepted ACP v2 prompt");
            #[cfg(feature = "unstable-inject")]
            session.stop_injection_turn();
            fail_accepted_prompt(
                &client,
                &integration,
                &session_id,
                &active_prompt,
                &driving_prompt,
                &cancelled_prompt,
                &lifecycle,
                owner,
                false,
            );
            continue;
        }
        let user_message_id =
            match integration.begin_prompt_owner(&session_id, owner, prompt_cancellation) {
                Ok(message_id) => message_id,
                Err(error) => {
                    tracing::debug!(%error, owner, "failed to begin accepted ACP v2 prompt");
                    #[cfg(feature = "unstable-inject")]
                    session.stop_injection_turn();
                    fail_accepted_prompt(
                        &client,
                        &integration,
                        &session_id,
                        &active_prompt,
                        &driving_prompt,
                        &cancelled_prompt,
                        &lifecycle,
                        owner,
                        false,
                    );
                    continue;
                }
            };

        if let Err(error) = client.update_for(
            session_id.clone(),
            wire::SessionUpdate::UserMessage(
                wire::UserMessage::new(user_message_id).content(request.prompt),
            ),
        ) {
            tracing::debug!(%error, owner, "failed to publish accepted ACP v2 prompt");
            #[cfg(feature = "unstable-inject")]
            session.stop_injection_turn();
            fail_accepted_prompt(
                &client,
                &integration,
                &session_id,
                &active_prompt,
                &driving_prompt,
                &cancelled_prompt,
                &lifecycle,
                owner,
                true,
            );
            continue;
        }

        drive_prompt(
            &mut driver,
            &integration,
            &session_id,
            &cancellation,
            cancellation_generation,
            #[cfg(feature = "unstable-inject")]
            &session,
        )
        .await;
        {
            let _lifecycle = lifecycle.lock().unwrap_or_else(|error| error.into_inner());
            integration.finish_prompt_owner(&session_id, owner);
            clear_prompt_tracking(&active_prompt, &driving_prompt, &cancelled_prompt, owner);
        }
        #[cfg(feature = "unstable-inject")]
        session.stop_injection_turn();
        if let Err(error) = client.flush().await {
            tracing::debug!(%error, "failed to flush ACP v2 output");
        }
        prefer_loop_update = true;
    }
}

/// Outcome of an unstable ACP v2 injection boundary.
#[cfg(feature = "unstable-inject")]
#[derive(Debug, Eq, PartialEq)]
pub enum AcpInjectionBoundary {
    /// No steer was delivered; continue the current non-terminal loop.
    Continue,
    /// At least one steer was delivered and the loop must continue.
    Delivered,
    /// The terminal boundary had no steer and the turn can finish.
    Finished,
    /// Cancellation or close stopped delivery.
    Stopped,
}

#[cfg(feature = "unstable-inject")]
impl AcpSessionHandle {
    /// Delivers ready unstable ACP v2 steers at one safe loop boundary.
    ///
    /// `terminal` must be true for terminal model/input boundaries and false
    /// after tool results. The returned outcome tells the host whether to
    /// continue, finish, or report cancellation.
    pub async fn handle_injection_boundary<S>(
        &self,
        driver: &mut agentkit_loop::LoopDriver<S>,
        terminal: bool,
    ) -> Result<AcpInjectionBoundary, AcpRuntimeError>
    where
        S: ModelSession + Send + 'static,
    {
        let injection = &self.session.injection;
        let mut delivered_any = false;
        loop {
            let changed = injection.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            match injection.boundary_action(terminal, delivered_any) {
                BoundaryAction::Wait => changed.await,
                BoundaryAction::Complete(outcome) => return Ok(outcome),
                BoundaryAction::Deliver(pending) => {
                    if let Err(error) = driver
                        .submit_input(pending.items.clone())
                        .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
                    {
                        injection.finish_delivery(&pending, false);
                        return Err(error);
                    }
                    let result = self
                        .session
                        .sink
                        .update_acknowledged(wire::UpdateSessionNotification::new(
                            self.session.acp_session_id.clone(),
                            wire::SessionUpdate::UserMessage(
                                wire::UserMessage::new(pending.message_id.clone())
                                    .content(pending.content.clone()),
                            ),
                        ))
                        .await;
                    injection.finish_delivery(&pending, result.is_ok());
                    result?;
                    delivered_any = true;
                }
            }
        }
    }
}

async fn drive_prompt<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    integration: &AcpIntegration,
    session_id: &wire::SessionId,
    cancellation: &CancellationHandle,
    generation: u64,
    #[cfg(feature = "unstable-inject")] session: &AcpSessionHandle,
) -> wire::StopReason
where
    S: ModelSession + Send + 'static,
{
    loop {
        let step = match driver.next().await {
            Ok(step) => step,
            Err(error) => {
                tracing::debug!(%error, "ACP v2 agent loop failed");
                #[cfg(feature = "unstable-inject")]
                session.stop_injection_turn();
                return if cancellation.is_cancelled_since(generation) {
                    wire::StopReason::Cancelled
                } else {
                    error_stop_reason()
                };
            }
        };
        if cancellation.is_cancelled_since(generation) {
            #[cfg(feature = "unstable-inject")]
            session.stop_injection_turn();
            return wire::StopReason::Cancelled;
        }
        match step {
            LoopStep::Finished(result) => {
                if result.finish_reason == FinishReason::ToolCall {
                    continue;
                }
                #[cfg(feature = "unstable-inject")]
                match session.handle_injection_boundary(driver, true).await {
                    Ok(AcpInjectionBoundary::Delivered | AcpInjectionBoundary::Continue) => {
                        continue;
                    }
                    Ok(AcpInjectionBoundary::Stopped) => return wire::StopReason::Cancelled,
                    Ok(AcpInjectionBoundary::Finished) => {
                        return finish_reason_to_stop_reason(&result.finish_reason);
                    }
                    Err(error) => {
                        tracing::debug!(%error, "failed to deliver ACP v2 injected message");
                        session.stop_injection_turn();
                        return error_stop_reason();
                    }
                }
                #[cfg(not(feature = "unstable-inject"))]
                return finish_reason_to_stop_reason(&result.finish_reason);
            }
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                #[cfg(feature = "unstable-inject")]
                match session.handle_injection_boundary(driver, true).await {
                    Ok(AcpInjectionBoundary::Delivered | AcpInjectionBoundary::Continue) => {
                        continue;
                    }
                    Ok(AcpInjectionBoundary::Stopped) => return wire::StopReason::Cancelled,
                    Ok(AcpInjectionBoundary::Finished) => return wire::StopReason::EndTurn,
                    Err(error) => {
                        tracing::debug!(%error, "failed to deliver ACP v2 injected message");
                        session.stop_injection_turn();
                        return error_stop_reason();
                    }
                }
                #[cfg(not(feature = "unstable-inject"))]
                return wire::StopReason::EndTurn;
            }
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => {
                #[cfg(feature = "unstable-inject")]
                match session.handle_injection_boundary(driver, false).await {
                    Ok(AcpInjectionBoundary::Stopped) => return wire::StopReason::Cancelled,
                    Err(error) => {
                        tracing::debug!(%error, "failed to deliver ACP v2 injected message");
                        session.stop_injection_turn();
                        return error_stop_reason();
                    }
                    _ => {}
                }
            }
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_)) => {
                integration.mark_unsupported_approval(session_id, cancellation.clone(), generation);
                if let Err(error) = driver.cancel_pending_approvals().await {
                    integration.clear_unsupported_approval(session_id);
                    tracing::debug!(%error, "failed to cancel unsupported ACP v2 approval");
                    #[cfg(feature = "unstable-inject")]
                    session.stop_injection_turn();
                    return error_stop_reason();
                }
                #[cfg(not(feature = "unstable-inject"))]
                return error_stop_reason();
            }
        }
    }
}

fn prompt_to_items(request: &wire::PromptRequest) -> Result<Vec<Item>, AcpRuntimeError> {
    content_blocks_to_items(&request.prompt)
}

/// Converts unstable ACP v2 content blocks into agentkit input items.
///
/// The conversion preserves text, images, audio, resource links, and embedded
/// resources using the same path as prompts and session injection.
pub fn content_blocks_to_items(
    content: &[wire::ContentBlock],
) -> Result<Vec<Item>, AcpRuntimeError> {
    let mut user_parts = Vec::new();
    let mut context_items = Vec::new();

    for block in content {
        match block {
            wire::ContentBlock::Text(text) => {
                user_parts.push(Part::Text(TextPart::new(text.text.clone())));
            }
            wire::ContentBlock::Image(image) => {
                let mime_type = image.mime_type.as_ref();
                user_parts.push(Part::media(
                    Modality::Image,
                    mime_type,
                    crate::data_url_ref(mime_type, &image.data),
                ));
            }
            wire::ContentBlock::Audio(audio) => {
                let mime_type = audio.mime_type.as_ref();
                user_parts.push(Part::media(
                    Modality::Audio,
                    mime_type,
                    crate::data_url_ref(mime_type, &audio.data),
                ));
            }
            wire::ContentBlock::ResourceLink(link) => {
                context_items.push(resource_link_item(link));
            }
            wire::ContentBlock::Resource(resource) => {
                context_items.push(resource_item(resource)?);
            }
            wire::ContentBlock::Other(other) => {
                return Err(AcpRuntimeError::UnsupportedContent(format!(
                    "unknown ACP v2 content block: {}",
                    other.type_
                )));
            }
            _ => {
                return Err(AcpRuntimeError::UnsupportedContent(
                    "unknown ACP v2 content block".into(),
                ));
            }
        }
    }

    let mut items = context_items;
    if !user_parts.is_empty() {
        items.push(Item::new(ItemKind::User, user_parts));
    } else if !items.is_empty() {
        items.push(Item::text(ItemKind::User, "Use the provided context."));
    }
    Ok(items)
}

fn resource_link_item(link: &wire::ResourceLink) -> Item {
    let mut metadata = MetadataMap::new();
    metadata.insert("acp.resource.uri".into(), json!(link.uri));
    metadata.insert("acp.resource.name".into(), json!(link.name));
    if let Some(description) = &link.description {
        metadata.insert("acp.resource.description".into(), json!(description));
    }
    if let Some(mime_type) = &link.mime_type {
        metadata.insert("acp.resource.mime_type".into(), json!(mime_type));
    }
    Item::new(
        ItemKind::Context,
        vec![Part::file(DataRef::uri(link.uri.clone()))],
    )
    .with_metadata(metadata)
}

fn resource_item(resource: &wire::EmbeddedResource) -> Result<Item, AcpRuntimeError> {
    match &resource.resource {
        wire::EmbeddedResourceResource::TextResourceContents(text) => {
            let mut metadata = MetadataMap::new();
            metadata.insert("acp.resource.uri".into(), json!(text.uri));
            if let Some(mime_type) = &text.mime_type {
                metadata.insert("acp.resource.mime_type".into(), json!(mime_type));
            }
            Ok(Item::text(ItemKind::Context, text.text.clone()).with_metadata(metadata))
        }
        wire::EmbeddedResourceResource::BlobResourceContents(blob) => {
            let mime_type = blob
                .mime_type
                .as_ref()
                .map(|mime_type| mime_type.as_ref())
                .unwrap_or("application/octet-stream");
            let mut metadata = MetadataMap::new();
            metadata.insert("acp.resource.uri".into(), json!(blob.uri));
            metadata.insert("acp.resource.mime_type".into(), json!(mime_type));
            Ok(Item::new(
                ItemKind::Context,
                vec![Part::media(
                    Modality::Binary,
                    mime_type,
                    crate::data_url_ref(mime_type, &blob.blob),
                )],
            )
            .with_metadata(metadata))
        }
        _ => Err(AcpRuntimeError::UnsupportedContent(
            "unknown ACP v2 embedded resource".into(),
        )),
    }
}

fn event_to_update(
    event: &AgentEvent,
    message_ids: Option<&CurrentMessageIds>,
    part_kinds: &mut HashMap<PartId, PartKind>,
) -> Option<wire::SessionUpdate> {
    match event {
        AgentEvent::ContentDelta(delta) => delta_to_update(delta, message_ids, part_kinds),
        AgentEvent::ToolCallRequested(call) => {
            Some(wire::SessionUpdate::ToolCallUpdate(tool_call_update(call)))
        }
        AgentEvent::ToolExecutionStarted(call) => Some(wire::SessionUpdate::ToolCallUpdate(
            tool_status_update(&call.id, wire::ToolCallStatus::InProgress),
        )),
        AgentEvent::ToolExecutionProgress(result) => Some(wire::SessionUpdate::ToolCallUpdate(
            tool_result_update(result, wire::ToolCallStatus::InProgress),
        )),
        AgentEvent::ToolResultReceived(result) => {
            Some(wire::SessionUpdate::ToolCallUpdate(tool_result_update(
                result,
                if result.is_error {
                    wire::ToolCallStatus::Failed
                } else {
                    wire::ToolCallStatus::Completed
                },
            )))
        }
        AgentEvent::Warning { message } => {
            tracing::warn!(%message, "agentkit warning while routing ACP v2 event");
            None
        }
        AgentEvent::RunFailed { message } => {
            tracing::debug!(%message, "agentkit run failed while routing ACP v2 event");
            None
        }
        _ => None,
    }
}

fn delta_to_update(
    delta: &Delta,
    message_ids: Option<&CurrentMessageIds>,
    part_kinds: &mut HashMap<PartId, PartKind>,
) -> Option<wire::SessionUpdate> {
    match delta {
        Delta::BeginPart { part_id, kind } => {
            part_kinds.insert(part_id.clone(), *kind);
            None
        }
        Delta::AppendText { part_id, chunk } => {
            let message_ids = message_ids?;
            let content = wire::ContentBlock::Text(wire::TextContent::new(chunk.clone()));
            match part_kinds.get(part_id) {
                Some(PartKind::Reasoning) => Some(wire::SessionUpdate::AgentThoughtChunk(
                    wire::ContentChunk::new(content, message_ids.thought.clone()),
                )),
                Some(PartKind::Text) | None => Some(wire::SessionUpdate::AgentMessageChunk(
                    wire::ContentChunk::new(content, message_ids.agent.clone()),
                )),
                Some(_) => None,
            }
        }
        Delta::CommitPart { .. }
        | Delta::AppendBytes { .. }
        | Delta::ReplaceStructured { .. }
        | Delta::SetMetadata { .. } => None,
    }
}

fn tool_call_update(call: &ToolCallPart) -> wire::ToolCallUpdate {
    wire::ToolCallUpdate::new(call.id.to_string())
        .title(call.name.clone())
        .status(wire::ToolCallStatus::Pending)
        .raw_input(call.input.clone())
}

fn tool_status_update(
    call_id: &agentkit_core::ToolCallId,
    status: wire::ToolCallStatus,
) -> wire::ToolCallUpdate {
    wire::ToolCallUpdate::new(call_id.to_string()).status(status)
}

fn tool_result_update(
    result: &ToolResultPart,
    status: wire::ToolCallStatus,
) -> wire::ToolCallUpdate {
    wire::ToolCallUpdate::new(result.call_id.to_string())
        .status(status)
        .raw_output(crate::tool_output_raw(&result.output))
        .content(tool_output_content(&result.output))
}

fn tool_output_content(output: &ToolOutput) -> Option<Vec<wire::ToolCallContent>> {
    let content = match output {
        ToolOutput::Text(text) => vec![text_to_tool_content(text.clone())],
        ToolOutput::Structured(value) => vec![text_to_tool_content(value.to_string())],
        ToolOutput::Parts(parts) => parts.iter().filter_map(part_to_tool_content).collect(),
        ToolOutput::Files(files) => files.iter().map(file_to_tool_content).collect(),
    };
    (!content.is_empty()).then_some(content)
}

fn part_to_tool_content(part: &Part) -> Option<wire::ToolCallContent> {
    match part {
        Part::Text(text) => Some(text_to_tool_content(text.text.clone())),
        Part::Structured(value) => Some(structured_to_tool_content(value)),
        Part::Media(media) => Some(media_to_tool_content(media)),
        Part::File(file) => Some(file_to_tool_content(file)),
        Part::Reasoning(reasoning) => reasoning
            .summary
            .as_ref()
            .map(|summary| text_to_tool_content(summary.clone())),
        Part::Custom(custom) => Some(text_to_tool_content(
            custom
                .value
                .as_ref()
                .map(ToString::to_string)
                .or_else(|| custom.data.as_ref().map(crate::data_ref_payload))
                .unwrap_or_else(|| custom.kind.clone()),
        )),
        Part::ToolCall(_) | Part::ToolResult(_) => None,
    }
}

fn text_to_tool_content(text: String) -> wire::ToolCallContent {
    wire::ToolCallContent::Content(Box::new(wire::Content::new(wire::ContentBlock::Text(
        wire::TextContent::new(text),
    ))))
}

fn structured_to_tool_content(part: &StructuredPart) -> wire::ToolCallContent {
    text_to_tool_content(part.value.to_string())
}

fn media_to_tool_content(media: &MediaPart) -> wire::ToolCallContent {
    match media.modality {
        Modality::Image
            if matches!(media.data, DataRef::InlineText(_) | DataRef::InlineBytes(_)) =>
        {
            let mut image = wire::ImageContent::new(
                crate::data_ref_base64_payload(&media.data),
                media.mime_type.clone(),
            );
            if let Some(uri) = crate::data_ref_uri(&media.data) {
                image = image.uri(uri);
            }
            wire::ToolCallContent::Content(Box::new(wire::Content::new(wire::ContentBlock::Image(
                image,
            ))))
        }
        Modality::Audio
            if matches!(media.data, DataRef::InlineText(_) | DataRef::InlineBytes(_)) =>
        {
            wire::ToolCallContent::Content(Box::new(wire::Content::new(wire::ContentBlock::Audio(
                wire::AudioContent::new(
                    crate::data_ref_base64_payload(&media.data),
                    media.mime_type.clone(),
                ),
            ))))
        }
        Modality::Image | Modality::Audio | Modality::Video | Modality::Binary => {
            data_ref_to_resource_content(None, Some(&media.mime_type), &media.data)
        }
    }
}

fn file_to_tool_content(file: &FilePart) -> wire::ToolCallContent {
    data_ref_to_resource_content(file.name.as_deref(), file.mime_type.as_deref(), &file.data)
}

fn data_ref_to_resource_content(
    name: Option<&str>,
    mime_type: Option<&str>,
    data: &DataRef,
) -> wire::ToolCallContent {
    let content = match data {
        DataRef::Uri(uri) => {
            let mut link = wire::ResourceLink::new(name.unwrap_or(uri), uri.clone());
            if let Some(mime_type) = mime_type {
                link = link.mime_type(mime_type.to_string());
            }
            wire::ContentBlock::ResourceLink(link)
        }
        DataRef::Handle(handle) => {
            let uri = format!("artifact://{handle}");
            let link_name = name.map(str::to_owned).unwrap_or_else(|| uri.clone());
            let mut link = wire::ResourceLink::new(link_name, uri);
            if let Some(mime_type) = mime_type {
                link = link.mime_type(mime_type.to_string());
            }
            wire::ContentBlock::ResourceLink(link)
        }
        DataRef::InlineText(text) if mime_type.is_none_or(|mime| mime.starts_with("text/")) => {
            let mut resource = wire::TextResourceContents::new(
                text.clone(),
                crate::inline_resource_uri(name.unwrap_or("tool-output")),
            );
            if let Some(mime_type) = mime_type {
                resource = resource.mime_type(mime_type.to_string());
            }
            wire::ContentBlock::Resource(wire::EmbeddedResource::new(
                wire::EmbeddedResourceResource::TextResourceContents(resource),
            ))
        }
        _ => {
            let mut resource = wire::BlobResourceContents::new(
                crate::data_ref_base64_payload(data),
                crate::inline_resource_uri(name.unwrap_or("tool-output")),
            );
            if let Some(mime_type) = mime_type {
                resource = resource.mime_type(mime_type.to_string());
            }
            wire::ContentBlock::Resource(wire::EmbeddedResource::new(
                wire::EmbeddedResourceResource::BlobResourceContents(resource),
            ))
        }
    };
    wire::ToolCallContent::Content(Box::new(wire::Content::new(content)))
}

fn error_stop_reason() -> wire::StopReason {
    wire::StopReason::Other("_error".into())
}

fn finish_reason_to_stop_reason(reason: &FinishReason) -> wire::StopReason {
    match reason {
        FinishReason::Completed | FinishReason::ToolCall | FinishReason::Other(_) => {
            wire::StopReason::EndTurn
        }
        FinishReason::MaxTokens => wire::StopReason::MaxTokens,
        FinishReason::Cancelled => wire::StopReason::Cancelled,
        FinishReason::Blocked | FinishReason::Error => error_stop_reason(),
    }
}

/// Constructs the exact unstable ACP v2 injection capability implemented here.
///
/// This function is available only with `unstable-inject`, matching the request
/// handlers and turn-boundary APIs it advertises.
#[cfg(feature = "unstable-inject")]
pub fn session_inject_capabilities() -> wire::SessionInjectCapabilities {
    wire::SessionInjectCapabilities::new(vec![wire::SessionInjectMode::Steer])
        .steer_in_stream(vec![wire::SessionInjectSteerInStream::Finish])
        .pending(wire::SessionInjectPendingCapabilities::new().replace(true))
}

/// Constructs the honest ACP v2 capabilities supported by this build.
///
/// Injection is advertised only when `unstable-inject` is enabled, and uses
/// [`session_inject_capabilities`] so hosts and [`AcpHeadlessRuntime`] report
/// the same behavior.
pub fn agent_capabilities() -> wire::AgentCapabilities {
    let session = wire::SessionCapabilities::new().prompt(
        wire::PromptCapabilities::new()
            .image(wire::PromptImageCapabilities::new())
            .audio(wire::PromptAudioCapabilities::new())
            .embedded_context(wire::PromptEmbeddedContextCapabilities::new()),
    );
    #[cfg(feature = "unstable-inject")]
    let session = session.inject(session_inject_capabilities());
    wire::AgentCapabilities::new().session(session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use agent_client_protocol::Channel;
    #[cfg(feature = "unstable-inject")]
    use agent_client_protocol::{RawJsonRpcMessage, TransportBatch, TransportFrame};
    use agentkit_core::{ItemKind, ToolCallId, ToolOutput, ToolResultPart, TurnCancellation};
    use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
    use agentkit_integration_tests::mock_tool::BlockingTool;
    use agentkit_loop::{
        Agent, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult, SessionConfig,
        TurnRequest, TurnResult,
    };
    use agentkit_task_manager::{AsyncTaskManager, RoutingDecision};
    #[cfg(feature = "unstable-inject")]
    use agentkit_tools_core::{
        ApprovalReason, ApprovalRequest, PermissionChecker, PermissionDecision, PermissionRequest,
    };
    use agentkit_tools_core::{
        Tool, ToolContext, ToolError, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
    };
    #[cfg(feature = "unstable-inject")]
    use futures_util::StreamExt as _;
    use tokio::sync::Notify;

    #[derive(Clone, Default)]
    struct RecordingSink {
        updates: Arc<Mutex<Vec<wire::UpdateSessionNotification>>>,
        acknowledged: Arc<AtomicUsize>,
        flushes: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AcpSessionUpdateSink for RecordingSink {
        fn update(
            &self,
            notification: wire::UpdateSessionNotification,
        ) -> Result<(), AcpRuntimeError> {
            self.updates.lock().unwrap().push(notification);
            Ok(())
        }

        async fn update_acknowledged(
            &self,
            notification: wire::UpdateSessionNotification,
        ) -> Result<(), AcpRuntimeError> {
            self.updates.lock().unwrap().push(notification);
            self.acknowledged.fetch_add(1, Ordering::Release);
            Ok(())
        }

        async fn flush(&self) -> Result<(), AcpRuntimeError> {
            self.flushes.fetch_add(1, Ordering::Release);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct TestFactory<A> {
        adapter: A,
    }

    #[async_trait]
    impl<A> AcpAgentFactory<A> for TestFactory<A>
    where
        A: ModelAdapter + Clone + Send + Sync + 'static,
        A::Session: Send + 'static,
    {
        async fn start(
            &self,
            ctx: AcpAgentFactoryContext,
        ) -> Result<agentkit_loop::LoopDriver<A::Session>, AcpRuntimeError> {
            Agent::builder()
                .model(self.adapter.clone())
                .observer(ctx.integration.as_ref().clone())
                .cancellation(ctx.cancellation)
                .build()
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))?
                .start(SessionConfig::new(ctx.agentkit_session_id).with_metadata(ctx.metadata))
                .await
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        }
    }

    #[derive(Clone)]
    struct CancellationAwareTool {
        spec: ToolSpec,
        entered: Arc<AtomicUsize>,
        cleaned: Arc<AtomicUsize>,
    }

    impl CancellationAwareTool {
        fn new() -> Self {
            Self {
                spec: ToolSpec::new(
                    "blocking_tool",
                    "waits for turn cancellation",
                    json!({ "type": "object" }),
                ),
                entered: Arc::new(AtomicUsize::new(0)),
                cleaned: Arc::new(AtomicUsize::new(0)),
            }
        }

        async fn wait_for_entered(&self, count: usize) {
            tokio::time::timeout(Duration::from_secs(2), async {
                while self.entered.load(Ordering::Acquire) < count {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("tool did not start");
        }
    }

    #[async_trait]
    impl Tool for CancellationAwareTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            self.entered.fetch_add(1, Ordering::AcqRel);
            ctx.cancellation
                .as_ref()
                .expect("turn cancellation installed")
                .cancelled()
                .await;
            self.cleaned.fetch_add(1, Ordering::AcqRel);
            Ok(ToolResult::new(ToolResultPart::error(
                request.call_id,
                ToolOutput::text("cancelled"),
            )))
        }
    }

    #[derive(Clone)]
    struct ToolTestFactory {
        adapter: MockAdapter,
        tool: CancellationAwareTool,
    }

    #[derive(Clone)]
    struct BackgroundToolTestFactory<A> {
        adapter: A,
        tool: BlockingTool,
    }

    #[async_trait]
    impl AcpAgentFactory<MockAdapter> for ToolTestFactory {
        async fn start(
            &self,
            ctx: AcpAgentFactoryContext,
        ) -> Result<
            agentkit_loop::LoopDriver<<MockAdapter as ModelAdapter>::Session>,
            AcpRuntimeError,
        > {
            Agent::builder()
                .model(self.adapter.clone())
                .add_tool_source(ToolRegistry::new().with(self.tool.clone()))
                .observer(ctx.integration.as_ref().clone())
                .cancellation(ctx.cancellation)
                .build()
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))?
                .start(SessionConfig::new(ctx.agentkit_session_id).with_metadata(ctx.metadata))
                .await
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        }
    }

    #[cfg(feature = "unstable-inject")]
    struct ApprovalPermissionRequest {
        metadata: MetadataMap,
    }

    #[cfg(feature = "unstable-inject")]
    impl PermissionRequest for ApprovalPermissionRequest {
        fn kind(&self) -> &'static str {
            "custom.approval-test"
        }

        fn summary(&self) -> String {
            "approve test tool".into()
        }

        fn metadata(&self) -> &MetadataMap {
            &self.metadata
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct ApprovalTool {
        spec: ToolSpec,
    }

    #[cfg(feature = "unstable-inject")]
    impl ApprovalTool {
        fn new() -> Self {
            Self {
                spec: ToolSpec::new(
                    "approval_tool",
                    "requires approval",
                    json!({ "type": "object" }),
                ),
            }
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl Tool for ApprovalTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn proposed_requests(
            &self,
            _request: &ToolRequest,
        ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
            Ok(vec![Box::new(ApprovalPermissionRequest {
                metadata: MetadataMap::new(),
            })])
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::text("approved"),
            )))
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone, Copy)]
    struct RequireApproval;

    #[cfg(feature = "unstable-inject")]
    impl PermissionChecker for RequireApproval {
        fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision {
            PermissionDecision::RequireApproval(ApprovalRequest::new(
                "approval-test",
                request.kind(),
                ApprovalReason::PolicyRequiresConfirmation,
                request.summary(),
            ))
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct ApprovalAdapter {
        permits: Arc<tokio::sync::Semaphore>,
        next_turn: Arc<AtomicUsize>,
    }

    #[cfg(feature = "unstable-inject")]
    impl ApprovalAdapter {
        fn new() -> Self {
            Self {
                permits: Arc::new(tokio::sync::Semaphore::new(0)),
                next_turn: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn release(&self) {
            self.permits.add_permits(1);
        }
    }

    #[cfg(feature = "unstable-inject")]
    struct ApprovalSession {
        permits: Arc<tokio::sync::Semaphore>,
        next_turn: Arc<AtomicUsize>,
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl ModelAdapter for ApprovalAdapter {
        type Session = ApprovalSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(ApprovalSession {
                permits: Arc::clone(&self.permits),
                next_turn: Arc::clone(&self.next_turn),
            })
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl ModelSession for ApprovalSession {
        type Turn = GatedTurn;

        async fn begin_turn(
            &mut self,
            _request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let turn = self.next_turn.fetch_add(1, Ordering::AcqRel);
            let events = if turn == 0 {
                let call =
                    ToolCallPart::new(ToolCallId::new("approval-call"), "approval_tool", json!({}));
                VecDeque::from([
                    ModelTurnEvent::ToolCall(call.clone()),
                    ModelTurnEvent::Finished(ModelTurnResult {
                        model: None,
                        response_id: None,
                        finish_reason: FinishReason::ToolCall,
                        output_items: vec![Item::new(
                            ItemKind::Assistant,
                            vec![Part::ToolCall(call)],
                        )],
                        usage: None,
                        metadata: MetadataMap::new(),
                    }),
                ])
            } else {
                VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item::text(ItemKind::Assistant, "after approval")],
                    usage: None,
                    metadata: MetadataMap::new(),
                })])
            };
            Ok(GatedTurn {
                permits: Arc::clone(&self.permits),
                events,
                started: false,
            })
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct ApprovalFactory {
        adapter: ApprovalAdapter,
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl AcpAgentFactory<ApprovalAdapter> for ApprovalFactory {
        async fn start(
            &self,
            ctx: AcpAgentFactoryContext,
        ) -> Result<agentkit_loop::LoopDriver<ApprovalSession>, AcpRuntimeError> {
            Agent::builder()
                .model(self.adapter.clone())
                .add_tool_source(ToolRegistry::new().with(ApprovalTool::new()))
                .permissions(RequireApproval)
                .observer(ctx.integration.as_ref().clone())
                .cancellation(ctx.cancellation)
                .build()
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))?
                .start(SessionConfig::new(ctx.agentkit_session_id).with_metadata(ctx.metadata))
                .await
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct CompletionGatedTool {
        spec: ToolSpec,
        entered: Arc<AtomicUsize>,
        release: Arc<tokio::sync::Semaphore>,
    }

    #[cfg(feature = "unstable-inject")]
    impl CompletionGatedTool {
        fn new() -> Self {
            Self {
                spec: ToolSpec::new(
                    "blocking_tool",
                    "waits for deterministic release",
                    json!({ "type": "object" }),
                ),
                entered: Arc::new(AtomicUsize::new(0)),
                release: Arc::new(tokio::sync::Semaphore::new(0)),
            }
        }

        async fn wait_for_entered(&self) {
            tokio::time::timeout(Duration::from_secs(2), async {
                while self.entered.load(Ordering::Acquire) == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("tool did not start");
        }

        fn release(&self) {
            self.release.add_permits(1);
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl Tool for CompletionGatedTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            self.entered.fetch_add(1, Ordering::AcqRel);
            Arc::clone(&self.release)
                .acquire_owned()
                .await
                .expect("tool gate stays open")
                .forget();
            Ok(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::text("done"),
            )))
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct InjectToolFactory {
        adapter: MockAdapter,
        tool: CompletionGatedTool,
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl AcpAgentFactory<MockAdapter> for InjectToolFactory {
        async fn start(
            &self,
            ctx: AcpAgentFactoryContext,
        ) -> Result<
            agentkit_loop::LoopDriver<<MockAdapter as ModelAdapter>::Session>,
            AcpRuntimeError,
        > {
            Agent::builder()
                .model(self.adapter.clone())
                .add_tool_source(ToolRegistry::new().with(self.tool.clone()))
                .observer(ctx.integration.as_ref().clone())
                .cancellation(ctx.cancellation)
                .build()
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))?
                .start(SessionConfig::new(ctx.agentkit_session_id).with_metadata(ctx.metadata))
                .await
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        }
    }

    #[async_trait]
    impl<A> AcpAgentFactory<A> for BackgroundToolTestFactory<A>
    where
        A: ModelAdapter + Clone + Send + Sync + 'static,
        A::Session: Send + 'static,
    {
        async fn start(
            &self,
            ctx: AcpAgentFactoryContext,
        ) -> Result<agentkit_loop::LoopDriver<A::Session>, AcpRuntimeError> {
            Agent::builder()
                .model(self.adapter.clone())
                .add_tool_source(ToolRegistry::new().with(self.tool.clone()))
                .task_manager(
                    AsyncTaskManager::new().routing(|_: &ToolRequest| RoutingDecision::Background),
                )
                .observer(ctx.integration.as_ref().clone())
                .cancellation(ctx.cancellation)
                .build()
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))?
                .start(SessionConfig::new(ctx.agentkit_session_id).with_metadata(ctx.metadata))
                .await
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        }
    }

    struct BlockingInferenceState {
        scripts: Mutex<VecDeque<TurnScript>>,
        next_turn: AtomicUsize,
        blocked_turn: usize,
        entered: AtomicBool,
        release: Notify,
        cancelled: AtomicBool,
    }

    #[derive(Clone)]
    struct BlockingInferenceAdapter {
        state: Arc<BlockingInferenceState>,
    }

    impl BlockingInferenceAdapter {
        fn new(scripts: impl IntoIterator<Item = TurnScript>, blocked_turn: usize) -> Self {
            Self {
                state: Arc::new(BlockingInferenceState {
                    scripts: Mutex::new(scripts.into_iter().collect()),
                    next_turn: AtomicUsize::new(0),
                    blocked_turn,
                    entered: AtomicBool::new(false),
                    release: Notify::new(),
                    cancelled: AtomicBool::new(false),
                }),
            }
        }

        async fn wait_until_blocked(&self) {
            tokio::time::timeout(Duration::from_secs(2), async {
                while !self.state.entered.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("model inference did not block");
        }

        fn release(&self) {
            self.state.release.notify_one();
        }

        fn was_cancelled(&self) -> bool {
            self.state.cancelled.load(Ordering::Acquire)
        }
    }

    struct BlockingInferenceSession {
        state: Arc<BlockingInferenceState>,
    }

    struct BlockingInferenceTurn {
        state: Arc<BlockingInferenceState>,
        events: VecDeque<ModelTurnEvent>,
        blocked: bool,
    }

    #[async_trait]
    impl ModelAdapter for BlockingInferenceAdapter {
        type Session = BlockingInferenceSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(BlockingInferenceSession {
                state: Arc::clone(&self.state),
            })
        }
    }

    #[async_trait]
    impl ModelSession for BlockingInferenceSession {
        type Turn = BlockingInferenceTurn;

        async fn begin_turn(
            &mut self,
            _request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let turn = self.state.next_turn.fetch_add(1, Ordering::AcqRel);
            let script = self
                .state
                .scripts
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| LoopError::InvalidState("missing blocking turn script".into()))?;
            Ok(BlockingInferenceTurn {
                state: Arc::clone(&self.state),
                events: script.events.into(),
                blocked: turn == self.state.blocked_turn,
            })
        }
    }

    #[async_trait]
    impl ModelTurn for BlockingInferenceTurn {
        async fn next_event(
            &mut self,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if self.blocked {
                self.blocked = false;
                self.state.entered.store(true, Ordering::Release);
                if let Some(cancellation) = cancellation {
                    tokio::select! {
                        _ = self.state.release.notified() => {}
                        _ = cancellation.cancelled() => {
                            self.state.cancelled.store(true, Ordering::Release);
                            return Err(LoopError::Cancelled);
                        }
                    }
                } else {
                    self.state.release.notified().await;
                }
            }
            Ok(self.events.pop_front())
        }
    }

    fn tool_turn(call_id: &str) -> TurnScript {
        let call = ToolCallPart::new(ToolCallId::new(call_id), "blocking_tool", json!({}));
        TurnScript::new([
            ModelTurnEvent::ToolCall(call.clone()),
            ModelTurnEvent::Finished(ModelTurnResult {
                model: None,
                response_id: None,
                finish_reason: FinishReason::ToolCall,
                output_items: vec![Item::new(ItemKind::Assistant, vec![Part::ToolCall(call)])],
                usage: None,
                metadata: MetadataMap::new(),
            }),
        ])
    }

    fn streamed_text_and_tool(text: &str, call_id: &str) -> TurnScript {
        let call = ToolCallPart::new(ToolCallId::new(call_id), "missing_tool", json!({}));
        TurnScript::new([
            ModelTurnEvent::Delta(Delta::BeginPart {
                part_id: PartId::new("part-1"),
                kind: PartKind::Text,
            }),
            ModelTurnEvent::Delta(Delta::AppendText {
                part_id: PartId::new("part-1"),
                chunk: text.to_string(),
            }),
            ModelTurnEvent::ToolCall(call.clone()),
            ModelTurnEvent::Finished(ModelTurnResult {
                model: None,
                response_id: None,
                finish_reason: FinishReason::ToolCall,
                output_items: vec![Item::new(
                    ItemKind::Assistant,
                    vec![Part::text(text), Part::ToolCall(call)],
                )],
                usage: None,
                metadata: MetadataMap::new(),
            }),
        ])
    }

    fn streamed_text(text: &str) -> TurnScript {
        TurnScript::new([
            ModelTurnEvent::Delta(Delta::BeginPart {
                part_id: PartId::new("part-1"),
                kind: PartKind::Text,
            }),
            ModelTurnEvent::Delta(Delta::AppendText {
                part_id: PartId::new("part-1"),
                chunk: text.to_string(),
            }),
            ModelTurnEvent::Finished(ModelTurnResult {
                model: None,
                response_id: None,
                finish_reason: FinishReason::Completed,
                output_items: vec![Item::text(ItemKind::Assistant, text)],
                usage: None,
                metadata: MetadataMap::new(),
            }),
        ])
    }

    fn streamed_thought_and_text(thought: &str, text: &str) -> TurnScript {
        TurnScript::new([
            ModelTurnEvent::Delta(Delta::BeginPart {
                part_id: PartId::new("thought-part"),
                kind: PartKind::Reasoning,
            }),
            ModelTurnEvent::Delta(Delta::AppendText {
                part_id: PartId::new("thought-part"),
                chunk: thought.to_string(),
            }),
            ModelTurnEvent::Delta(Delta::BeginPart {
                part_id: PartId::new("text-part"),
                kind: PartKind::Text,
            }),
            ModelTurnEvent::Delta(Delta::AppendText {
                part_id: PartId::new("text-part"),
                chunk: text.to_string(),
            }),
            ModelTurnEvent::Finished(ModelTurnResult {
                model: None,
                response_id: None,
                finish_reason: FinishReason::Completed,
                output_items: vec![Item::text(ItemKind::Assistant, text)],
                usage: None,
                metadata: MetadataMap::new(),
            }),
        ])
    }

    async fn wait_for_idle(
        updates: &Arc<Mutex<Vec<(wire::SessionId, wire::SessionUpdate)>>>,
        session_id: &wire::SessionId,
    ) {
        wait_for_idle_count(updates, session_id, 1).await;
    }

    async fn wait_for_idle_count(
        updates: &Arc<Mutex<Vec<(wire::SessionId, wire::SessionUpdate)>>>,
        session_id: &wire::SessionId,
        count: usize,
    ) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let idle_count = updates
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(id, update)| {
                        id == session_id
                            && matches!(
                                update,
                                wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                            )
                    })
                    .count();
                if idle_count >= count {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("idle update timed out");
    }

    #[tokio::test]
    async fn runtime_rotates_message_ids_per_model_turn_and_session() {
        let adapter = MockAdapter::new();
        adapter.enqueue(streamed_text_and_tool("before tool", "call-1"));
        adapter.enqueue(streamed_thought_and_text("first thought", "first output"));
        adapter.enqueue(streamed_thought_and_text("second thought", "second output"));
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();

        let server = tokio::spawn({
            let factory = TestFactory {
                adapter: adapter.clone(),
            };
            async move {
                AcpHeadlessRuntime::<MockAdapter>::builder()
                    .name("agentkit-v2-test")
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let updates = Arc::clone(&updates);
                async move |cx| {
                    let initialize = cx
                        .send_request(wire::InitializeRequest::new(
                            wire::ProtocolVersion::V2,
                            wire::Implementation::new("test-client", "1"),
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(initialize.protocol_version, wire::ProtocolVersion::V2);
                    assert_eq!(initialize.info.name, "agentkit-v2-test");

                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let first = cx
                        .send_request(wire::NewSessionRequest::new(cwd.clone()))
                        .block_task()
                        .await?;
                    let second = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;
                    assert_ne!(first.session_id, second.session_id);

                    cx.send_request(wire::PromptRequest::new(
                        first.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("first"))],
                    ))
                    .block_task()
                    .await?;
                    wait_for_idle(&updates, &first.session_id).await;

                    cx.send_request(wire::PromptRequest::new(
                        second.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("second"))],
                    ))
                    .block_task()
                    .await?;
                    wait_for_idle(&updates, &second.session_id).await;

                    cx.send_request(wire::CloseSessionRequest::new(first.session_id))
                        .block_task()
                        .await?;
                    cx.send_request(wire::CloseSessionRequest::new(second.session_id))
                        .block_task()
                        .await?;
                    Ok(())
                }
            });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;

        let updates = updates.lock().unwrap();
        for session_id in ["session-1", "session-2"] {
            let session_updates = updates
                .iter()
                .filter(|(id, _)| id.to_string() == session_id)
                .map(|(_, update)| update)
                .collect::<Vec<_>>();
            assert!(matches!(
                session_updates[0],
                wire::SessionUpdate::UserMessage(_)
            ));
            assert!(matches!(
                session_updates[1],
                wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(_))
            ));
            assert_eq!(
                session_updates
                    .iter()
                    .filter(|update| matches!(
                        update,
                        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(_))
                    ))
                    .count(),
                1,
                "Running must come only from TurnStarted"
            );
            assert_eq!(
                session_updates
                    .iter()
                    .filter(|update| matches!(
                        update,
                        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                    ))
                    .count(),
                1,
                "Idle must come only from TurnFinished"
            );
            let user_id = match session_updates[0] {
                wire::SessionUpdate::UserMessage(message) => message.message_id.to_string(),
                _ => unreachable!(),
            };
            let agent_ids = session_updates
                .iter()
                .filter_map(|update| match update {
                    wire::SessionUpdate::AgentMessageChunk(chunk) => {
                        Some(chunk.message_id.to_string())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let thought_ids = session_updates
                .iter()
                .filter_map(|update| match update {
                    wire::SessionUpdate::AgentThoughtChunk(chunk) => {
                        Some(chunk.message_id.to_string())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(user_id, format!("{session_id}-user-1"));
            if session_id == "session-1" {
                assert_eq!(
                    agent_ids,
                    [
                        "session-1-agent-2".to_string(),
                        "session-1-agent-3".to_string()
                    ]
                );
                assert_eq!(thought_ids, ["session-1-thought-3"]);
                assert_ne!(agent_ids[0], agent_ids[1]);
            } else {
                assert_eq!(agent_ids, ["session-2-agent-2"]);
                assert_eq!(thought_ids, ["session-2-thought-2"]);
            }
            assert_ne!(agent_ids.last(), thought_ids.last());
            assert!(matches!(
                session_updates.last(),
                Some(wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(idle)))
                    if idle.stop_reason == Some(wire::StopReason::EndTurn)
            ));
        }
    }

    #[tokio::test]
    async fn idle_session_drives_background_completion_without_another_prompt() {
        let adapter = MockAdapter::new();
        adapter.enqueue(tool_turn("background-call"));
        adapter.enqueue(streamed_text("background observed"));
        let tool = BlockingTool::text("blocking_tool", "background done");
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();

        let server = tokio::spawn({
            let factory = BackgroundToolTestFactory {
                adapter,
                tool: tool.clone(),
            };
            async move {
                AcpHeadlessRuntime::<MockAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let updates = Arc::clone(&updates);
                async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("test-client", "1"),
                    ))
                    .block_task()
                    .await?;
                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let session = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;
                    cx.send_request(wire::PromptRequest::new(
                        session.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            "start background work",
                        ))],
                    ))
                    .block_task()
                    .await?;
                    wait_for_idle(&updates, &session.session_id).await;
                    tool.wait_until_entered().await;
                    tool.release();

                    tokio::time::timeout(Duration::from_secs(2), async {
                        loop {
                            let updates = updates.lock().unwrap();
                            let completed = updates.iter().any(|(id, update)| {
                                id == &session.session_id
                                    && matches!(
                                        update,
                                        wire::SessionUpdate::ToolCallUpdate(update)
                                            if update.tool_call_id.to_string() == "background-call"
                                                && matches!(
                                                    update.status,
                                                    agent_client_protocol::schema::MaybeUndefined::Value(
                                                        wire::ToolCallStatus::Completed
                                                    )
                                                )
                                    )
                            });
                            let message_delivered = updates.iter().any(|(id, update)| {
                                id == &session.session_id
                                    && matches!(
                                        update,
                                        wire::SessionUpdate::AgentMessageChunk(chunk)
                                            if matches!(
                                                &chunk.content,
                                                wire::ContentBlock::Text(text)
                                                    if text.text == "background observed"
                                            )
                                    )
                            });
                            if completed && message_delivered {
                                break;
                            }
                            drop(updates);
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    })
                    .await
                    .expect("background completion was not delivered while idle");
                    cx.send_request(wire::CloseSessionRequest::new(session.session_id))
                        .block_task()
                        .await?;
                    Ok(())
                }
            });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;

        let updates = updates.lock().unwrap();
        assert_eq!(
            updates
                .iter()
                .filter(|(_, update)| matches!(update, wire::SessionUpdate::UserMessage(_)))
                .count(),
            1,
            "the background continuation must not require or replay a prompt"
        );
        assert!(updates.iter().any(|(_, update)| matches!(
            update,
            wire::SessionUpdate::AgentMessageChunk(chunk)
                if matches!(
                    &chunk.content,
                    wire::ContentBlock::Text(text) if text.text == "background observed"
                )
        )));
        assert_eq!(
            updates
                .iter()
                .filter(|(_, update)| matches!(
                    update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(_))
                ))
                .count(),
            1,
            "idle background completion must not re-enter Running"
        );
        assert_eq!(
            updates
                .iter()
                .filter(|(_, update)| matches!(
                    update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                ))
                .count(),
            1,
            "idle background completion must not emit another Idle"
        );
    }

    #[tokio::test]
    async fn cancel_without_prompt_owner_interrupts_blocked_autonomous_inference() {
        let adapter = BlockingInferenceAdapter::new(
            [
                tool_turn("background-call"),
                streamed_text("autonomous complete"),
            ],
            1,
        );
        let tool = BlockingTool::text("blocking_tool", "background done");
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();

        let server = tokio::spawn({
            let factory = BackgroundToolTestFactory {
                adapter: adapter.clone(),
                tool: tool.clone(),
            };
            async move {
                AcpHeadlessRuntime::<BlockingInferenceAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let adapter = adapter.clone();
                let tool = tool.clone();
                let updates = Arc::clone(&updates);
                async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("test-client", "1"),
                    ))
                    .block_task()
                    .await?;
                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let session = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;
                    cx.send_request(wire::PromptRequest::new(
                        session.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
                    ))
                    .block_task()
                    .await?;
                    wait_for_idle(&updates, &session.session_id).await;
                    tool.wait_until_entered().await;
                    tool.release();
                    adapter.wait_until_blocked().await;

                    cx.send_notification(wire::CancelSessionNotification::new(
                        session.session_id.clone(),
                    ))?;
                    cx.send_request(wire::ListSessionsRequest::new())
                        .block_task()
                        .await?;
                    tokio::time::timeout(Duration::from_secs(2), async {
                        while !adapter.was_cancelled() {
                            tokio::task::yield_now().await;
                        }
                    })
                    .await
                    .expect("session cancel did not interrupt autonomous inference");

                    cx.send_request(wire::CloseSessionRequest::new(session.session_id))
                        .block_task()
                        .await?;
                    Ok(())
                }
            });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn queued_prompt_is_acknowledged_without_cancelling_blocked_autonomous_inference() {
        let adapter = BlockingInferenceAdapter::new(
            [
                tool_turn("background-call"),
                streamed_text("autonomous complete"),
                streamed_text("queued prompt complete"),
            ],
            1,
        );
        let tool = BlockingTool::text("blocking_tool", "background done");
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();

        let server = tokio::spawn({
            let factory = BackgroundToolTestFactory {
                adapter: adapter.clone(),
                tool: tool.clone(),
            };
            async move {
                AcpHeadlessRuntime::<BlockingInferenceAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let adapter = adapter.clone();
                let tool = tool.clone();
                let updates = Arc::clone(&updates);
                async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("test-client", "1"),
                    ))
                    .block_task()
                    .await?;
                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let session = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;
                    cx.send_request(wire::PromptRequest::new(
                        session.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
                    ))
                    .block_task()
                    .await?;
                    wait_for_idle(&updates, &session.session_id).await;
                    tool.wait_until_entered().await;
                    tool.release();
                    adapter.wait_until_blocked().await;

                    tokio::time::timeout(
                        Duration::from_millis(500),
                        cx.send_request(wire::PromptRequest::new(
                            session.session_id.clone(),
                            vec![wire::ContentBlock::Text(wire::TextContent::new(
                                "queued while idle",
                            ))],
                        ))
                        .block_task(),
                    )
                    .await
                    .expect("prompt response waited for autonomous inference")?;
                    assert_eq!(
                        updates
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|(_, update)| matches!(
                                update,
                                wire::SessionUpdate::UserMessage(_)
                            ))
                            .count(),
                        1,
                        "queued prompt update crossed the response gate"
                    );

                    cx.send_notification(wire::CancelSessionNotification::new(
                        session.session_id.clone(),
                    ))?;
                    cx.send_request(wire::ListSessionsRequest::new())
                        .block_task()
                        .await?;
                    assert!(
                        tokio::time::timeout(Duration::from_millis(50), async {
                            while !adapter.was_cancelled() {
                                tokio::task::yield_now().await;
                            }
                        })
                        .await
                        .is_err(),
                        "queued prompt cancellation interrupted autonomous inference"
                    );

                    adapter.release();
                    tokio::time::timeout(Duration::from_secs(2), async {
                        loop {
                            if updates.lock().unwrap().iter().any(|(_, update)| {
                                matches!(
                                    update,
                                    wire::SessionUpdate::AgentMessageChunk(chunk)
                                        if matches!(
                                            &chunk.content,
                                            wire::ContentBlock::Text(text)
                                                if text.text == "autonomous complete"
                                        )
                                )
                            }) {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    })
                    .await
                    .expect("autonomous inference did not complete");
                    assert!(!adapter.was_cancelled());
                    assert_eq!(
                        updates
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|(_, update)| matches!(
                                update,
                                wire::SessionUpdate::UserMessage(_)
                            ))
                            .count(),
                        1,
                        "cancelled queued prompt was driven"
                    );

                    cx.send_request(wire::CloseSessionRequest::new(session.session_id))
                        .block_task()
                        .await?;
                    Ok(())
                }
            });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;
    }

    #[test]
    fn v2_tool_updates_include_visible_text_structured_parts_and_files() {
        let outputs = [
            ToolOutput::text("plain text"),
            ToolOutput::structured(json!({ "ok": true })),
            ToolOutput::parts(vec![
                Part::text("part text"),
                Part::structured(json!({ "part": true })),
            ]),
            ToolOutput::files(vec![
                FilePart::named("artifact.txt", DataRef::inline_text("artifact body"))
                    .with_mime_type("text/plain"),
                FilePart::named("remote.txt", DataRef::uri("file:///tmp/remote.txt")),
            ]),
        ];
        let contents = outputs
            .into_iter()
            .map(|output| {
                let result = ToolResultPart::success(ToolCallId::new("call"), output);
                let update = tool_result_update(&result, wire::ToolCallStatus::Completed);
                serde_json::to_value(update).expect("serialize tool update")["content"].clone()
            })
            .collect::<Vec<_>>();

        assert_eq!(contents[0][0]["content"]["text"], "plain text");
        assert_eq!(contents[1][0]["content"]["text"], r#"{"ok":true}"#);
        assert_eq!(contents[2].as_array().map(Vec::len), Some(2));
        assert_eq!(contents[2][0]["content"]["text"], "part text");
        assert_eq!(contents[3].as_array().map(Vec::len), Some(2));
        assert_eq!(
            contents[3][0]["content"]["resource"]["text"],
            "artifact body"
        );
        assert_eq!(contents[3][1]["content"]["uri"], "file:///tmp/remote.txt");
    }

    #[tokio::test]
    async fn joining_stuck_session_tasks_uses_one_shared_deadline() {
        struct DropMarker(Arc<AtomicUsize>);
        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        let tasks = (0..3)
            .map(|_| {
                let dropped = Arc::clone(&dropped);
                tokio::spawn(async move {
                    let _marker = DropMarker(dropped);
                    std::future::pending::<()>().await;
                })
            })
            .collect();
        tokio::task::yield_now().await;
        let started = std::time::Instant::now();
        join_tasks_until(
            tokio::time::Instant::now() + Duration::from_millis(50),
            tasks,
        )
        .await;

        assert!(started.elapsed() < Duration::from_millis(250));
        tokio::time::timeout(Duration::from_millis(250), async {
            while dropped.load(Ordering::Acquire) != 3 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborted session tasks were not dropped");
    }

    #[tokio::test]
    async fn aborting_serve_cleans_up_active_sessions() {
        let adapter = MockAdapter::new();
        adapter.enqueue(tool_turn("serve-cancel"));
        let tool = CancellationAwareTool::new();
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let factory = ToolTestFactory {
                adapter,
                tool: tool.clone(),
            };
            async move {
                AcpHeadlessRuntime::<MockAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });
        let client = tokio::spawn(agent_client_protocol::Client.v2().connect_with(
            client_transport,
            async move |cx| {
                cx.send_request(wire::InitializeRequest::new(
                    wire::ProtocolVersion::V2,
                    wire::Implementation::new("test-client", "1"),
                ))
                .block_task()
                .await?;
                let cwd = std::env::current_dir()
                    .map_err(agent_client_protocol::Error::into_internal_error)?;
                let session = cx
                    .send_request(wire::NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;
                cx.send_request(wire::PromptRequest::new(
                    session.session_id,
                    vec![wire::ContentBlock::Text(wire::TextContent::new("run"))],
                ))
                .block_task()
                .await?;
                std::future::pending::<Result<(), agent_client_protocol::Error>>().await
            },
        ));

        tool.wait_for_entered(1).await;
        server.abort();
        let _ = server.await;
        tokio::time::timeout(Duration::from_secs(2), async {
            while tool.cleaned.load(Ordering::Acquire) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("serve drop did not clean the active tool");
        client.abort();
        let _ = client.await;
    }

    #[cfg(not(feature = "unstable-inject"))]
    #[test]
    fn public_capabilities_omit_inject_without_feature() {
        let capabilities = serde_json::to_value(agent_capabilities()).unwrap();
        assert!(capabilities.pointer("/session/inject").is_none());
    }

    #[cfg(feature = "unstable-inject")]
    #[test]
    fn public_host_helpers_expose_conversion_capabilities_and_limits() {
        assert_eq!(MAX_PENDING_INJECTIONS, 64);
        assert_eq!(MAX_PENDING_INJECTION_BYTES, 256 * 1024);
        assert_eq!(MAX_ACCEPTED_INJECTIONS, 4_096);

        let inject = session_inject_capabilities();
        assert_eq!(inject.modes, vec![wire::SessionInjectMode::Steer]);
        assert_eq!(
            inject.steer_in_stream,
            Some(vec![wire::SessionInjectSteerInStream::Finish])
        );
        let advertised = agent_capabilities()
            .session
            .and_then(|session| session.inject)
            .expect("inject capability must match compiled handlers");
        assert_eq!(advertised, inject);

        let items = content_blocks_to_items(&[wire::ContentBlock::Text(wire::TextContent::new(
            "converted",
        ))])
        .expect("public content conversion");
        assert_eq!(items.len(), 1);
        assert!(matches!(
            &items[0].parts[0],
            Part::Text(text) if text.text == "converted"
        ));
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn public_host_staged_inject_api_commits_and_activates_receipt() {
        let integration = AcpIntegration::default();
        let session_id = wire::SessionId::new("external-host");
        let session = integration
            .bind_session(AcpSessionBinding::new(
                session_id.clone(),
                AgentkitSessionId::new("external-host-loop"),
                RecordingSink::default(),
            ))
            .expect("bind external host session");
        session.prepare_injection_turn();
        session.start_injection_turn();

        let stages = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let integration = integration.clone();
            let stages = Arc::clone(&stages);
            async move {
                agent_client_protocol::Agent
                    .v2()
                    .on_receive_request(
                        async move |request: wire::InitializeRequest, responder, _cx| {
                            responder.respond(
                                wire::InitializeResponse::new(
                                    request.protocol_version,
                                    wire::Implementation::new("external-host", "1"),
                                )
                                .capabilities(agent_capabilities()),
                            )
                        },
                        agent_client_protocol::on_receive_request!(),
                    )
                    .on_receive_request(
                        {
                            let integration = integration.clone();
                            let stages = Arc::clone(&stages);
                            async move |request: wire::InjectSessionRequest, responder, cx| {
                                let integration = integration.clone();
                                let stages = Arc::clone(&stages);
                                cx.spawn(async move {
                                    let reserved = integration
                                        .reserve_inject_request(request, responder)
                                        .await?
                                        .expect("inject reservation");
                                    let response_id = reserved.response().message_id;
                                    stages.lock().unwrap().push("reserved");
                                    let acceptance = reserved
                                        .respond_tracked()?
                                        .expect("tracked response acceptance");
                                    assert_eq!(acceptance.message_id(), &response_id);
                                    stages.lock().unwrap().push("responded");
                                    acceptance.activate_after_response().await?;
                                    stages.lock().unwrap().push("activated");
                                    Ok(())
                                })?;
                                Ok(())
                            }
                        },
                        agent_client_protocol::on_receive_request!(),
                    )
                    .connect_to(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .connect_with(client_transport, {
                let session_id = session_id.clone();
                async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("external-client", "1"),
                    ))
                    .block_task()
                    .await?;
                    let response = cx
                        .send_request(wire::InjectSessionRequest::new(
                            session_id,
                            wire::SessionInjectMode::Steer,
                            vec![wire::ContentBlock::Text(wire::TextContent::new(
                                "external steer",
                            ))],
                        ))
                        .block_task()
                        .await?;
                    assert!(response.message_id.to_string().contains("-user-"));
                    Ok(())
                }
            });
        tokio::time::timeout(Duration::from_secs(2), client)
            .await
            .expect("external host client timed out")
            .expect("external host client failed");
        tokio::time::timeout(Duration::from_secs(2), async {
            while stages.lock().unwrap().len() != 3 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("response receipt was not activated");
        assert_eq!(
            stages.lock().unwrap().as_slice(),
            ["reserved", "responded", "activated"]
        );

        session.stop_injection_turn();
        integration.unbind_session(&session_id).unwrap();
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn public_host_binding_routes_updates_through_sink() {
        let integration = AcpIntegration::default();
        let sink = RecordingSink::default();
        let acp_id = wire::SessionId::new("public-host");
        let agentkit_id = AgentkitSessionId::new("public-host-loop");
        let session = integration
            .bind_session(AcpSessionBinding::new(
                acp_id.clone(),
                agentkit_id.clone(),
                sink.clone(),
            ))
            .expect("bind public host session");

        assert_eq!(session.acp_session_id(), &acp_id);
        assert_eq!(session.agentkit_session_id(), &agentkit_id);
        let prompt = wire::PromptRequest::new(
            acp_id.clone(),
            vec![wire::ContentBlock::Text(wire::TextContent::new("hello"))],
        );
        assert_eq!(integration.prompt_to_items(&prompt).unwrap().len(), 1);
        integration.begin_prompt(&acp_id).unwrap();
        integration.route_event(
            &agentkit_id,
            AgentEvent::ContentDelta(Delta::AppendText {
                part_id: PartId::new("part"),
                chunk: "response".into(),
            }),
        );
        integration.flush_session_updates(&acp_id).await.unwrap();

        let updates = sink.updates.lock().unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].session_id, acp_id);
        assert_eq!(sink.flushes.load(Ordering::Acquire), 1);
        drop(updates);
        integration.unbind_session(&acp_id).unwrap();
    }

    #[test]
    fn public_host_binding_rotates_message_ids_at_tool_boundary() {
        let integration = AcpIntegration::default();
        let (client, mut messages) = ClientHandle::channel();
        let acp_id = wire::SessionId::new("acp-session");
        let agentkit_id = AgentkitSessionId::new("agentkit-session");
        integration
            .bind_session(AcpSessionBinding::new(
                acp_id.clone(),
                agentkit_id.clone(),
                client,
            ))
            .expect("bind session");
        integration.begin_prompt(&acp_id).expect("begin prompt");

        for event in [
            AgentEvent::ContentDelta(Delta::BeginPart {
                part_id: PartId::new("before-tool"),
                kind: PartKind::Text,
            }),
            AgentEvent::ContentDelta(Delta::AppendText {
                part_id: PartId::new("before-tool"),
                chunk: "before".into(),
            }),
            AgentEvent::ToolExecutionStarted(ToolCallPart::new(
                ToolCallId::new("call"),
                "background_tool",
                json!({}),
            )),
            AgentEvent::ContentDelta(Delta::BeginPart {
                part_id: PartId::new("after-tool"),
                kind: PartKind::Text,
            }),
            AgentEvent::ContentDelta(Delta::AppendText {
                part_id: PartId::new("after-tool"),
                chunk: "after".into(),
            }),
        ] {
            integration.route_event(&agentkit_id, event);
        }

        let message_ids = std::iter::from_fn(|| messages.try_recv().ok())
            .filter_map(|message| match message {
                ClientMessage::Update(notification) => match notification.update {
                    wire::SessionUpdate::AgentMessageChunk(chunk) => Some(chunk.message_id),
                    _ => None,
                },
                ClientMessage::AcknowledgedUpdate { .. } | ClientMessage::Flush(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            message_ids,
            [
                wire::MessageId::new("acp-session-agent-2"),
                wire::MessageId::new("acp-session-agent-3"),
            ]
        );
    }

    #[test]
    fn public_host_binding_rejects_duplicate_agentkit_session_ids() {
        let integration = AcpIntegration::default();
        let (first_client, _first_rx) = ClientHandle::channel();
        let (second_client, _second_rx) = ClientHandle::channel();
        let agentkit_id = AgentkitSessionId::new("shared-agentkit-session");
        let first_acp = wire::SessionId::new("first-acp-session");
        let second_acp = wire::SessionId::new("second-acp-session");
        integration
            .bind_session(AcpSessionBinding::new(
                first_acp.clone(),
                agentkit_id.clone(),
                first_client,
            ))
            .expect("first binding");
        let error = integration
            .bind_session(AcpSessionBinding::new(
                second_acp.clone(),
                agentkit_id,
                second_client,
            ))
            .err()
            .expect("duplicate AgentKit session id must be rejected");
        assert!(matches!(error, AcpRuntimeError::SessionAlreadyBound(_)));
        assert!(matches!(
            integration.session(&second_acp),
            Err(AcpRuntimeError::SessionNotFound(_))
        ));
        integration
            .unbind_session(&first_acp)
            .expect("unbind first");
    }

    #[test]
    fn retained_transcript_failures_use_custom_error_stop_reason() {
        for reason in [FinishReason::Blocked, FinishReason::Error] {
            assert_eq!(
                finish_reason_to_stop_reason(&reason),
                wire::StopReason::Other("_error".into())
            );
        }
    }

    #[test]
    fn unsupported_approval_keeps_error_stop_reason_on_cancelled_turn() {
        let integration = AcpIntegration::default();
        let (client, mut messages) = ClientHandle::channel();
        let acp_id = wire::SessionId::new("acp-session");
        let agentkit_id = AgentkitSessionId::new("agentkit-session");
        let cancellation = CancellationController::new();
        let generation = cancellation.handle().generation();
        integration
            .bind(acp_id.clone(), agentkit_id.clone(), client)
            .expect("bind session");
        integration
            .install_prompt_state(
                &acp_id,
                Arc::new(AtomicU64::new(1)),
                Arc::new(Mutex::new(())),
            )
            .expect("install prompt state");
        integration
            .begin_prompt_owner(&acp_id, 1, cancellation.handle().checkpoint())
            .expect("begin prompt");

        integration.route_event(
            &agentkit_id,
            AgentEvent::TurnStarted {
                session_id: agentkit_id.clone(),
                turn_id: TurnId::new("turn-1"),
            },
        );
        integration.mark_unsupported_approval(&acp_id, cancellation.handle(), generation);
        integration.route_event(
            &agentkit_id,
            AgentEvent::TurnFinished(TurnResult {
                turn_id: TurnId::new("turn-1"),
                finish_reason: FinishReason::Cancelled,
                items: Vec::new(),
                usage: None,
                metadata: MetadataMap::new(),
            }),
        );

        assert!(matches!(
            messages.try_recv(),
            Ok(ClientMessage::Update(notification))
                if matches!(
                    &notification.update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(_))
                )
        ));
        assert!(matches!(
            messages.try_recv(),
            Ok(ClientMessage::Update(notification))
                if matches!(
                    &notification.update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(idle))
                        if idle.stop_reason.as_ref()
                            == Some(&wire::StopReason::Other("_error".into()))
                )
        ));
        assert!(messages.try_recv().is_err());
    }

    #[test]
    fn prompt_cancellation_overrides_terminal_and_unsupported_stop_reasons() {
        for (index, finish_reason, unsupported_approval) in [
            (1, FinishReason::Completed, false),
            (2, FinishReason::Error, false),
            (3, FinishReason::Completed, true),
        ] {
            let integration = AcpIntegration::default();
            let (client, mut messages) = ClientHandle::channel();
            let acp_id = wire::SessionId::new(format!("acp-session-{index}"));
            let agentkit_id = AgentkitSessionId::new(format!("agentkit-session-{index}"));
            let active_prompt = Arc::new(AtomicU64::new(index));
            let cancellation = CancellationController::new();
            let generation = cancellation.handle().generation();
            integration
                .bind(acp_id.clone(), agentkit_id.clone(), client)
                .expect("bind session");
            integration
                .install_prompt_state(&acp_id, active_prompt, Arc::new(Mutex::new(())))
                .expect("install prompt state");
            integration
                .begin_prompt_owner(&acp_id, index, cancellation.handle().checkpoint())
                .expect("begin prompt");
            let turn_id = TurnId::new(format!("turn-{index}"));
            integration.route_event(
                &agentkit_id,
                AgentEvent::TurnStarted {
                    session_id: agentkit_id.clone(),
                    turn_id: turn_id.clone(),
                },
            );
            if unsupported_approval {
                integration.mark_unsupported_approval(&acp_id, cancellation.handle(), generation);
            }
            cancellation.interrupt();
            integration.route_event(
                &agentkit_id,
                AgentEvent::TurnFinished(agentkit_loop::TurnResult {
                    turn_id,
                    finish_reason,
                    items: Vec::new(),
                    usage: None,
                    metadata: MetadataMap::new(),
                }),
            );

            assert!(matches!(
                messages.try_recv(),
                Ok(ClientMessage::Update(notification))
                    if matches!(
                        notification.update,
                        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(_))
                    )
            ));
            assert!(matches!(
                messages.try_recv(),
                Ok(ClientMessage::Update(notification))
                    if matches!(
                        &notification.update,
                        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(idle))
                            if idle.stop_reason == Some(wire::StopReason::Cancelled)
                    )
            ));
        }
    }

    #[test]
    fn accepted_prompt_failure_queues_error_idle_after_cleanup() {
        let integration = AcpIntegration::default();
        let owner = 7;
        let active_prompt = Arc::new(AtomicU64::new(owner));
        let driving_prompt = Arc::new(AtomicU64::new(owner));
        let cancelled_prompt = Arc::new(AtomicU64::new(owner));
        let lifecycle = Arc::new(Mutex::new(()));
        let prompt_at_enqueue = Arc::clone(&active_prompt);
        let lifecycle_at_enqueue = Arc::clone(&lifecycle);
        let (client, mut messages) = ClientHandle::channel_with_update_hook(move |update| {
            if matches!(
                update,
                wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
            ) {
                assert_eq!(prompt_at_enqueue.load(Ordering::Acquire), 0);
                assert!(lifecycle_at_enqueue.try_lock().is_err());
            }
        });
        let acp_id = wire::SessionId::new("acp-failed-prompt");
        integration
            .bind(
                acp_id.clone(),
                AgentkitSessionId::new("agentkit-failed-prompt"),
                client.clone(),
            )
            .expect("bind session");
        integration
            .install_prompt_state(&acp_id, Arc::clone(&active_prompt), Arc::clone(&lifecycle))
            .expect("install prompt state");
        let cancellation = CancellationController::new();
        integration
            .begin_prompt_owner(&acp_id, owner, cancellation.handle().checkpoint())
            .expect("begin prompt");

        fail_accepted_prompt(
            &client,
            &integration,
            &acp_id,
            &active_prompt,
            &driving_prompt,
            &cancelled_prompt,
            &lifecycle,
            owner,
            true,
        );

        assert_eq!(active_prompt.load(Ordering::Acquire), 0);
        assert_eq!(driving_prompt.load(Ordering::Acquire), 0);
        assert_eq!(cancelled_prompt.load(Ordering::Acquire), 0);
        let session = integration.session(&acp_id).expect("bound session");
        assert!(
            session
                .prompt_state
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|state| state.pending_owner.is_none())
        );
        assert!(matches!(
            messages.try_recv(),
            Ok(ClientMessage::Update(notification))
                if matches!(
                    &notification.update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(idle))
                        if idle.stop_reason == Some(error_stop_reason())
                )
        ));
    }

    #[test]
    fn idle_is_queued_after_releasing_its_prompt_owner() {
        let integration = AcpIntegration::default();
        let active_prompt = Arc::new(AtomicU64::new(7));
        let lifecycle = Arc::new(Mutex::new(()));
        let prompt_at_enqueue = Arc::clone(&active_prompt);
        let lifecycle_at_enqueue = Arc::clone(&lifecycle);
        let (client, mut messages) = ClientHandle::channel_with_update_hook(move |update| {
            if matches!(
                update,
                wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
            ) {
                assert_eq!(
                    prompt_at_enqueue.load(Ordering::Acquire),
                    0,
                    "Idle was enqueued before prompt ownership was released"
                );
                assert!(
                    lifecycle_at_enqueue.try_lock().is_err(),
                    "Idle was enqueued outside the prompt lifecycle lock"
                );
            }
        });
        let acp_id = wire::SessionId::new("acp-session");
        let agentkit_id = AgentkitSessionId::new("agentkit-session");
        integration
            .bind(acp_id.clone(), agentkit_id.clone(), client)
            .expect("bind session");
        integration
            .install_prompt_state(&acp_id, Arc::clone(&active_prompt), Arc::clone(&lifecycle))
            .expect("install prompt state");
        let cancellation = CancellationController::new();
        integration
            .begin_prompt_owner(&acp_id, 7, cancellation.handle().checkpoint())
            .expect("begin prompt");
        let turn_id = TurnId::new("turn-1");
        integration.route_event(
            &agentkit_id,
            AgentEvent::TurnStarted {
                session_id: agentkit_id.clone(),
                turn_id: turn_id.clone(),
            },
        );
        integration.route_event(
            &agentkit_id,
            AgentEvent::TurnFinished(agentkit_loop::TurnResult {
                turn_id,
                finish_reason: FinishReason::Completed,
                items: Vec::new(),
                usage: None,
                metadata: MetadataMap::new(),
            }),
        );

        assert_eq!(active_prompt.load(Ordering::Acquire), 0);
        assert!(
            std::iter::from_fn(|| messages.try_recv().ok()).any(|message| {
                matches!(
                    message,
                    ClientMessage::Update(notification)
                        if matches!(
                            notification.update,
                            wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                        )
                )
            })
        );
    }

    #[test]
    fn cancellation_while_turn_finish_waits_for_lifecycle_lock_wins() {
        let integration = AcpIntegration::default();
        let active_prompt = Arc::new(AtomicU64::new(9));
        let lifecycle = Arc::new(Mutex::new(()));
        let prompt_at_enqueue = Arc::clone(&active_prompt);
        let lifecycle_at_enqueue = Arc::clone(&lifecycle);
        let (client, mut messages) = ClientHandle::channel_with_update_hook(move |update| {
            if matches!(
                update,
                wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
            ) {
                assert_eq!(prompt_at_enqueue.load(Ordering::Acquire), 0);
                assert!(lifecycle_at_enqueue.try_lock().is_err());
            }
        });
        let acp_id = wire::SessionId::new("acp-race-session");
        let agentkit_id = AgentkitSessionId::new("agentkit-race-session");
        integration
            .bind(acp_id.clone(), agentkit_id.clone(), client)
            .expect("bind session");
        integration
            .install_prompt_state(&acp_id, Arc::clone(&active_prompt), Arc::clone(&lifecycle))
            .expect("install prompt state");
        let cancellation = CancellationController::new();
        integration
            .begin_prompt_owner(&acp_id, 9, cancellation.handle().checkpoint())
            .expect("begin prompt");
        let turn_id = TurnId::new("turn-race");
        integration.route_event(
            &agentkit_id,
            AgentEvent::TurnStarted {
                session_id: agentkit_id.clone(),
                turn_id: turn_id.clone(),
            },
        );

        let lifecycle_guard = lifecycle.lock().unwrap();
        let ready = Arc::new(std::sync::Barrier::new(2));
        let finish_ready = Arc::clone(&ready);
        let finish_integration = integration.clone();
        let finish_session = agentkit_id.clone();
        let finish = std::thread::spawn(move || {
            finish_ready.wait();
            finish_integration.route_event(
                &finish_session,
                AgentEvent::TurnFinished(agentkit_loop::TurnResult {
                    turn_id,
                    finish_reason: FinishReason::Completed,
                    items: Vec::new(),
                    usage: None,
                    metadata: MetadataMap::new(),
                }),
            );
        });
        ready.wait();
        cancellation.interrupt();
        drop(lifecycle_guard);
        finish.join().expect("turn finish thread");

        assert!(matches!(
            messages.try_recv(),
            Ok(ClientMessage::Update(notification))
                if matches!(
                    notification.update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(_))
                )
        ));
        assert!(matches!(
            messages.try_recv(),
            Ok(ClientMessage::Update(notification))
                if matches!(
                    &notification.update,
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(idle))
                        if idle.stop_reason == Some(wire::StopReason::Cancelled)
                )
        ));
    }

    #[test]
    fn stale_prompt_owner_cannot_release_new_prompt() {
        let active_prompt = AtomicU64::new(1);
        release_prompt(&active_prompt, 1);
        assert_eq!(active_prompt.load(Ordering::Acquire), 0);

        active_prompt.store(2, Ordering::Release);
        release_prompt(&active_prompt, 1);
        assert_eq!(active_prompt.load(Ordering::Acquire), 2);
        release_prompt(&active_prompt, 2);
        assert_eq!(active_prompt.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn cancel_then_prompt_and_close_cleanup_active_tools() {
        let adapter = MockAdapter::new();
        adapter.enqueue(tool_turn("call-cancel"));
        adapter.enqueue(streamed_text("ready again"));
        adapter.enqueue(tool_turn("call-close"));
        let tool = CancellationAwareTool::new();
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();

        let server = tokio::spawn({
            let factory = ToolTestFactory {
                adapter: adapter.clone(),
                tool: tool.clone(),
            };
            async move {
                AcpHeadlessRuntime::<MockAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let updates = Arc::clone(&updates);
                let tool = tool.clone();
                async move |cx| {
                    let initialize = cx
                        .send_request(wire::InitializeRequest::new(
                            wire::ProtocolVersion::V2,
                            wire::Implementation::new("test-client", "1"),
                        ))
                        .block_task()
                        .await?;
                    assert!(initialize.capabilities.session.is_some());
                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let first = cx
                        .send_request(wire::NewSessionRequest::new(cwd.clone()))
                        .block_task()
                        .await?;

                    let listed = cx
                        .send_request(wire::ListSessionsRequest::new())
                        .block_task()
                        .await?;
                    assert_eq!(listed.sessions.len(), 1);
                    assert_eq!(listed.sessions[0].session_id, first.session_id);
                    cx.send_request(wire::ResumeSessionRequest::new(
                        first.session_id.clone(),
                        cwd.clone(),
                    ))
                    .block_task()
                    .await?;

                    cx.send_request(wire::PromptRequest::new(
                        first.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            "start tool",
                        ))],
                    ))
                    .block_task()
                    .await?;
                    tool.wait_for_entered(1).await;
                    cx.send_notification(wire::CancelSessionNotification::new(
                        first.session_id.clone(),
                    ))?;
                    wait_for_idle_count(&updates, &first.session_id, 1).await;
                    assert_eq!(tool.cleaned.load(Ordering::Acquire), 1);

                    tokio::time::timeout(
                        Duration::from_millis(250),
                        cx.send_request(wire::PromptRequest::new(
                            first.session_id.clone(),
                            vec![wire::ContentBlock::Text(wire::TextContent::new("again"))],
                        ))
                        .block_task(),
                    )
                    .await
                    .expect("session was not ready when Idle was emitted")?;
                    wait_for_idle_count(&updates, &first.session_id, 2).await;

                    let second = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;
                    cx.send_request(wire::PromptRequest::new(
                        second.session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            "close tool",
                        ))],
                    ))
                    .block_task()
                    .await?;
                    tool.wait_for_entered(2).await;
                    tokio::time::timeout(
                        Duration::from_secs(3),
                        cx.send_request(wire::CloseSessionRequest::new(second.session_id.clone()))
                            .block_task(),
                    )
                    .await
                    .expect("close did not complete after cooperative cleanup")?;
                    assert_eq!(tool.cleaned.load(Ordering::Acquire), 2);
                    cx.send_request(wire::CloseSessionRequest::new(first.session_id))
                        .block_task()
                        .await?;
                    Ok(())
                }
            });

        tokio::time::timeout(Duration::from_secs(8), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;

        let updates = updates.lock().unwrap();
        for call_id in ["call-cancel", "call-close"] {
            let statuses = updates
                .iter()
                .filter_map(|(_, update)| match update {
                    wire::SessionUpdate::ToolCallUpdate(update)
                        if update.tool_call_id.to_string() == call_id =>
                    {
                        Some(update.status.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(statuses.iter().any(|status| matches!(
                status,
                agent_client_protocol::schema::MaybeUndefined::Value(wire::ToolCallStatus::Pending)
            )));
            assert!(statuses.iter().any(|status| matches!(
                status,
                agent_client_protocol::schema::MaybeUndefined::Value(
                    wire::ToolCallStatus::InProgress
                )
            )));
            assert!(statuses.iter().any(|status| matches!(
                status,
                agent_client_protocol::schema::MaybeUndefined::Value(wire::ToolCallStatus::Failed)
            )));
        }
    }

    #[derive(Clone, Default)]
    struct BlockingAdapter;

    struct BlockingSession;

    struct BlockingTurn {
        finished: bool,
    }

    #[async_trait]
    impl ModelAdapter for BlockingAdapter {
        type Session = BlockingSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(BlockingSession)
        }
    }

    #[async_trait]
    impl ModelSession for BlockingSession {
        type Turn = BlockingTurn;

        async fn begin_turn(
            &mut self,
            _request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            Ok(BlockingTurn { finished: false })
        }
    }

    #[async_trait]
    impl ModelTurn for BlockingTurn {
        async fn next_event(
            &mut self,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if self.finished {
                return Ok(None);
            }
            cancellation
                .expect("cancellation installed")
                .cancelled()
                .await;
            self.finished = true;
            Ok(Some(ModelTurnEvent::Finished(ModelTurnResult {
                model: None,
                response_id: None,
                finish_reason: FinishReason::Cancelled,
                output_items: Vec::new(),
                usage: None,
                metadata: MetadataMap::new(),
            })))
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct GatedAdapter {
        permits: Arc<tokio::sync::Semaphore>,
        next_turn: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<TurnRequest>>>,
    }

    #[cfg(feature = "unstable-inject")]
    impl GatedAdapter {
        fn new() -> Self {
            Self {
                permits: Arc::new(tokio::sync::Semaphore::new(0)),
                next_turn: Arc::new(AtomicUsize::new(0)),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn release(&self, count: usize) {
            self.permits.add_permits(count);
        }

        fn requests(&self) -> Vec<TurnRequest> {
            self.requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[derive(Clone)]
    struct SecondStartGatedFactory {
        adapter: GatedAdapter,
        starts: Arc<AtomicUsize>,
        second_started: Arc<Notify>,
        second_release: Arc<tokio::sync::Semaphore>,
    }

    #[cfg(feature = "unstable-inject")]
    impl SecondStartGatedFactory {
        fn new(adapter: GatedAdapter) -> Self {
            Self {
                adapter,
                starts: Arc::new(AtomicUsize::new(0)),
                second_started: Arc::new(Notify::new()),
                second_release: Arc::new(tokio::sync::Semaphore::new(0)),
            }
        }

        async fn wait_for_second_start(&self) {
            if self.starts.load(Ordering::Acquire) < 2 {
                tokio::time::timeout(Duration::from_secs(2), self.second_started.notified())
                    .await
                    .expect("batched sibling did not start");
            }
        }

        fn release_second_start(&self) {
            self.second_release.add_permits(1);
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl AcpAgentFactory<GatedAdapter> for SecondStartGatedFactory {
        async fn start(
            &self,
            ctx: AcpAgentFactoryContext,
        ) -> Result<agentkit_loop::LoopDriver<GatedSession>, AcpRuntimeError> {
            if self.starts.fetch_add(1, Ordering::AcqRel) == 1 {
                self.second_started.notify_one();
                Arc::clone(&self.second_release)
                    .acquire_owned()
                    .await
                    .expect("second-start gate stays open")
                    .forget();
            }
            Agent::builder()
                .model(self.adapter.clone())
                .observer(ctx.integration.as_ref().clone())
                .cancellation(ctx.cancellation)
                .build()
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))?
                .start(SessionConfig::new(ctx.agentkit_session_id).with_metadata(ctx.metadata))
                .await
                .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        }
    }

    #[cfg(feature = "unstable-inject")]
    async fn raw_request(
        channel: &mut Channel,
        id: i64,
        method: &str,
        params: serde_json::Value,
    ) -> serde_json::Value {
        channel
            .tx
            .unbounded_send(TransportFrame::Single(
                RawJsonRpcMessage::request(method.to_string(), params, id.into())
                    .expect("valid raw request"),
            ))
            .expect("server channel stays open");
        raw_response(channel, id).await
    }

    #[cfg(feature = "unstable-inject")]
    async fn raw_response(channel: &mut Channel, id: i64) -> serde_json::Value {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let frame = channel.rx.next().await.expect("server channel closed");
                if let TransportFrame::Single(message @ RawJsonRpcMessage::Response(_)) = frame {
                    let response = serde_json::to_value(message).expect("response serializes");
                    if response.get("id") == Some(&json!(id)) {
                        return response;
                    }
                }
            }
        })
        .await
        .expect("raw request timed out")
    }

    #[cfg(feature = "unstable-inject")]
    struct GatedSession {
        permits: Arc<tokio::sync::Semaphore>,
        next_turn: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<TurnRequest>>>,
    }

    #[cfg(feature = "unstable-inject")]
    struct GatedTurn {
        permits: Arc<tokio::sync::Semaphore>,
        events: VecDeque<ModelTurnEvent>,
        started: bool,
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl ModelAdapter for GatedAdapter {
        type Session = GatedSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(GatedSession {
                permits: Arc::clone(&self.permits),
                next_turn: Arc::clone(&self.next_turn),
                requests: Arc::clone(&self.requests),
            })
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl ModelSession for GatedSession {
        type Turn = GatedTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            self.requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(request);
            let turn = self.next_turn.fetch_add(1, Ordering::AcqRel);
            let text = format!("turn-{turn}");
            Ok(GatedTurn {
                permits: Arc::clone(&self.permits),
                events: VecDeque::from([
                    ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: PartId::new(format!("part-{turn}")),
                        kind: PartKind::Text,
                    }),
                    ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: PartId::new(format!("part-{turn}")),
                        chunk: text.clone(),
                    }),
                    ModelTurnEvent::Finished(ModelTurnResult {
                        model: None,
                        response_id: None,
                        finish_reason: FinishReason::Completed,
                        output_items: vec![Item::text(ItemKind::Assistant, text)],
                        usage: None,
                        metadata: MetadataMap::new(),
                    }),
                ]),
                started: false,
            })
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[async_trait]
    impl ModelTurn for GatedTurn {
        async fn next_event(
            &mut self,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if !self.started {
                self.started = true;
                if let Some(cancellation) = cancellation {
                    tokio::select! {
                        permit = Arc::clone(&self.permits).acquire_owned() => {
                            permit.expect("gate stays open").forget();
                        }
                        _ = cancellation.cancelled() => {
                            return Ok(Some(ModelTurnEvent::Finished(ModelTurnResult {
                                model: None,
                                response_id: None,
                                finish_reason: FinishReason::Cancelled,
                                output_items: Vec::new(),
                                usage: None,
                                metadata: MetadataMap::new(),
                            })));
                        }
                    }
                }
            }
            Ok(self.events.pop_front())
        }
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn batched_session_inject_activates_after_aggregate_response_enqueue() {
        let adapter = GatedAdapter::new();
        let factory = SecondStartGatedFactory::new(adapter.clone());
        let coordinator = AcpIntegration::default();
        let (mut client, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let factory = factory.clone();
            let coordinator = coordinator.clone();
            async move {
                AcpHeadlessRuntime::<GatedAdapter>::builder()
                    .agent_factory(factory)
                    .integration(coordinator)
                    .serve(agent_transport)
                    .await
            }
        });

        let initialize = raw_request(
            &mut client,
            1,
            "initialize",
            serde_json::to_value(wire::InitializeRequest::new(
                wire::ProtocolVersion::V2,
                wire::Implementation::new("batch-test", "1"),
            ))
            .unwrap(),
        )
        .await;
        assert!(initialize.get("result").is_some());
        let cwd = std::env::current_dir().unwrap();
        let new_session = raw_request(
            &mut client,
            2,
            "session/new",
            serde_json::to_value(wire::NewSessionRequest::new(cwd.clone())).unwrap(),
        )
        .await;
        let session_id = new_session["result"]["sessionId"]
            .as_str()
            .unwrap()
            .to_string();
        let prompt = raw_request(
            &mut client,
            3,
            "session/prompt",
            serde_json::to_value(wire::PromptRequest::new(
                wire::SessionId::new(session_id.clone()),
                vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
            ))
            .unwrap(),
        )
        .await;
        assert!(prompt.get("result").is_some());

        let inject = RawJsonRpcMessage::request(
            "session/inject".to_string(),
            serde_json::to_value(wire::InjectSessionRequest::new(
                wire::SessionId::new(session_id.clone()),
                wire::SessionInjectMode::Steer,
                vec![wire::ContentBlock::Text(wire::TextContent::new(
                    "batched steer",
                ))],
            ))
            .unwrap(),
            10.into(),
        )
        .unwrap();
        let second_inject = RawJsonRpcMessage::request(
            "session/inject".to_string(),
            serde_json::to_value(wire::InjectSessionRequest::new(
                wire::SessionId::new(session_id.clone()),
                wire::SessionInjectMode::Steer,
                vec![wire::ContentBlock::Text(wire::TextContent::new(
                    "second batched steer",
                ))],
            ))
            .unwrap(),
            12.into(),
        )
        .unwrap();
        let slow_sibling = RawJsonRpcMessage::request(
            "session/new".to_string(),
            serde_json::to_value(wire::NewSessionRequest::new(cwd)).unwrap(),
            11.into(),
        )
        .unwrap();
        client
            .tx
            .unbounded_send(TransportFrame::Batch(
                TransportBatch::from_messages([inject, second_inject, slow_sibling]).unwrap(),
            ))
            .unwrap();

        factory.wait_for_second_start().await;
        adapter.release(1);
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            adapter.requests().len(),
            1,
            "batch injections became deliverable before their aggregate response receipt"
        );
        factory.release_second_start();

        let response = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(TransportFrame::Batch(batch)) = client.rx.next().await {
                    break serde_json::to_value(batch).unwrap();
                }
            }
        })
        .await
        .expect("batched response timed out");
        let entries = response.as_array().expect("aggregate batch response");
        let inject_response = entries
            .iter()
            .find(|entry| entry.get("id") == Some(&json!(10)))
            .expect("inject response slot");
        assert!(inject_response.get("result").is_some());
        assert!(inject_response.get("error").is_none());
        assert_eq!(
            coordinator.inject_requests.next.load(Ordering::Acquire),
            2,
            "headless inject handlers did not use the supplied public coordinator"
        );
        assert!(
            entries
                .iter()
                .find(|entry| entry.get("id") == Some(&json!(12)))
                .is_some_and(|entry| entry.get("result").is_some())
        );

        adapter.release(1);
        let delivered = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let frame = client.rx.next().await.expect("server channel stays open");
                let value = match frame {
                    TransportFrame::Single(message) => serde_json::to_value(message).unwrap(),
                    TransportFrame::Batch(batch) => serde_json::to_value(batch).unwrap(),
                    TransportFrame::Malformed { raw, .. } => serde_json::Value::String(raw),
                };
                if serde_json::to_string(&value)
                    .unwrap()
                    .contains("batched steer")
                {
                    break true;
                }
            }
        })
        .await
        .expect("batched injection was not delivered");
        assert!(delivered);
        let requests = adapter.requests();
        assert!(requests.len() >= 2);
        let steers = requests[1]
            .transcript
            .iter()
            .filter(|item| item.kind == ItemKind::User)
            .flat_map(|item| &item.parts)
            .filter_map(|part| match part {
                Part::Text(text) if text.text.contains("batched steer") => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(steers, ["batched steer", "second batched steer"]);

        drop(client);
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn request_cancellation_respects_response_commit() {
        let adapter = GatedAdapter::new();
        let (mut client, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let adapter = adapter.clone();
            async move {
                AcpHeadlessRuntime::<GatedAdapter>::builder()
                    .agent_factory(TestFactory { adapter })
                    .serve(agent_transport)
                    .await
            }
        });
        raw_request(
            &mut client,
            1,
            "initialize",
            serde_json::to_value(wire::InitializeRequest::new(
                wire::ProtocolVersion::V2,
                wire::Implementation::new("cancel-test", "1"),
            ))
            .unwrap(),
        )
        .await;
        let created = raw_request(
            &mut client,
            2,
            "session/new",
            serde_json::to_value(wire::NewSessionRequest::new(
                std::env::current_dir().unwrap(),
            ))
            .unwrap(),
        )
        .await;
        let session_id =
            wire::SessionId::new(created["result"]["sessionId"].as_str().unwrap().to_string());
        raw_request(
            &mut client,
            3,
            "session/prompt",
            serde_json::to_value(wire::PromptRequest::new(
                session_id.clone(),
                vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
            ))
            .unwrap(),
        )
        .await;

        let large_steer = "x".repeat(200_000);
        client
            .tx
            .unbounded_send(TransportFrame::Single(
                RawJsonRpcMessage::request(
                    "session/inject".to_string(),
                    serde_json::to_value(wire::InjectSessionRequest::new(
                        session_id.clone(),
                        wire::SessionInjectMode::Steer,
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            large_steer,
                        ))],
                    ))
                    .unwrap(),
                    10.into(),
                )
                .unwrap(),
            ))
            .unwrap();
        client
            .tx
            .unbounded_send(TransportFrame::Single(
                RawJsonRpcMessage::notification(
                    "$/cancel_request".to_string(),
                    json!({ "requestId": 10 }),
                )
                .unwrap(),
            ))
            .unwrap();
        let cancelled_response = raw_response(&mut client, 10).await;
        assert!(cancelled_response.get("result").is_none());
        assert_eq!(cancelled_response["error"]["code"], json!(-32800));

        let committed = raw_request(
            &mut client,
            11,
            "session/inject",
            serde_json::to_value(wire::InjectSessionRequest::new(
                session_id,
                wire::SessionInjectMode::Steer,
                vec![wire::ContentBlock::Text(wire::TextContent::new(
                    "committed",
                ))],
            ))
            .unwrap(),
        )
        .await;
        assert!(committed.get("result").is_some());
        client
            .tx
            .unbounded_send(TransportFrame::Single(
                RawJsonRpcMessage::notification(
                    "$/cancel_request".to_string(),
                    json!({ "requestId": 11 }),
                )
                .unwrap(),
            ))
            .unwrap();
        adapter.release(2);

        tokio::time::timeout(Duration::from_secs(2), async {
            while adapter.requests().len() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("committed injection was removed by late request cancellation");
        assert!(adapter.requests()[1].transcript.iter().any(|item| {
            item.parts
                .iter()
                .any(|part| matches!(part, Part::Text(text) if text.text == "committed"))
        }));
        drop(client);
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn closing_session_discards_pending_inject_and_returns_not_found() {
        let adapter = GatedAdapter::new();
        let factory = SecondStartGatedFactory::new(adapter);
        let (mut client, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let factory = factory.clone();
            async move {
                AcpHeadlessRuntime::<GatedAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });
        raw_request(
            &mut client,
            1,
            "initialize",
            serde_json::to_value(wire::InitializeRequest::new(
                wire::ProtocolVersion::V2,
                wire::Implementation::new("close-test", "1"),
            ))
            .unwrap(),
        )
        .await;
        let cwd = std::env::current_dir().unwrap();
        let created = raw_request(
            &mut client,
            2,
            "session/new",
            serde_json::to_value(wire::NewSessionRequest::new(cwd.clone())).unwrap(),
        )
        .await;
        let session_id =
            wire::SessionId::new(created["result"]["sessionId"].as_str().unwrap().to_string());
        raw_request(
            &mut client,
            3,
            "session/prompt",
            serde_json::to_value(wire::PromptRequest::new(
                session_id.clone(),
                vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
            ))
            .unwrap(),
        )
        .await;
        let inject = RawJsonRpcMessage::request(
            "session/inject".to_string(),
            serde_json::to_value(wire::InjectSessionRequest::new(
                session_id.clone(),
                wire::SessionInjectMode::Steer,
                vec![wire::ContentBlock::Text(wire::TextContent::new("pending"))],
            ))
            .unwrap(),
            10.into(),
        )
        .unwrap();
        let sibling = RawJsonRpcMessage::request(
            "session/new".to_string(),
            serde_json::to_value(wire::NewSessionRequest::new(cwd)).unwrap(),
            11.into(),
        )
        .unwrap();
        client
            .tx
            .unbounded_send(TransportFrame::Batch(
                TransportBatch::from_messages([inject, sibling]).unwrap(),
            ))
            .unwrap();
        factory.wait_for_second_start().await;
        let close = raw_request(
            &mut client,
            20,
            "session/close",
            serde_json::to_value(wire::CloseSessionRequest::new(session_id.clone())).unwrap(),
        )
        .await;
        assert!(close.get("result").is_some());
        factory.release_second_start();
        tokio::time::sleep(Duration::from_millis(30)).await;
        let missing = raw_request(
            &mut client,
            21,
            "session/inject",
            serde_json::to_value(wire::InjectSessionRequest::new(
                session_id,
                wire::SessionInjectMode::Steer,
                vec![wire::ContentBlock::Text(wire::TextContent::new("late"))],
            ))
            .unwrap(),
        )
        .await;
        assert_eq!(missing["error"]["code"], json!(-32002));

        drop(client);
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn same_batch_close_prevents_inject_acceptance() {
        let adapter = GatedAdapter::new();
        let (mut client, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let adapter = adapter.clone();
            async move {
                AcpHeadlessRuntime::<GatedAdapter>::builder()
                    .agent_factory(TestFactory { adapter })
                    .serve(agent_transport)
                    .await
            }
        });
        raw_request(
            &mut client,
            1,
            "initialize",
            serde_json::to_value(wire::InitializeRequest::new(
                wire::ProtocolVersion::V2,
                wire::Implementation::new("same-batch-close-test", "1"),
            ))
            .unwrap(),
        )
        .await;
        let created = raw_request(
            &mut client,
            2,
            "session/new",
            serde_json::to_value(wire::NewSessionRequest::new(
                std::env::current_dir().unwrap(),
            ))
            .unwrap(),
        )
        .await;
        let session_id =
            wire::SessionId::new(created["result"]["sessionId"].as_str().unwrap().to_string());
        raw_request(
            &mut client,
            3,
            "session/prompt",
            serde_json::to_value(wire::PromptRequest::new(
                session_id.clone(),
                vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
            ))
            .unwrap(),
        )
        .await;

        let inject = RawJsonRpcMessage::request(
            "session/inject".to_string(),
            serde_json::to_value(wire::InjectSessionRequest::new(
                session_id.clone(),
                wire::SessionInjectMode::Steer,
                vec![wire::ContentBlock::Text(wire::TextContent::new("discard"))],
            ))
            .unwrap(),
            10.into(),
        )
        .unwrap();
        let close = RawJsonRpcMessage::request(
            "session/close".to_string(),
            serde_json::to_value(wire::CloseSessionRequest::new(session_id)).unwrap(),
            11.into(),
        )
        .unwrap();
        client
            .tx
            .unbounded_send(TransportFrame::Batch(
                TransportBatch::from_messages([inject, close]).unwrap(),
            ))
            .unwrap();

        let response = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(TransportFrame::Batch(batch)) = client.rx.next().await {
                    break serde_json::to_value(batch).unwrap();
                }
            }
        })
        .await
        .expect("same-batch close response timed out");
        let entries = response.as_array().expect("aggregate batch response");
        let inject_response = entries
            .iter()
            .find(|entry| entry.get("id") == Some(&json!(10)))
            .expect("inject response slot");
        assert!(inject_response.get("result").is_none());
        assert_eq!(inject_response["error"]["code"], json!(-32002));
        let close_response = entries
            .iter()
            .find(|entry| entry.get("id") == Some(&json!(11)))
            .expect("close response slot");
        assert!(close_response.get("result").is_some());
        assert!(close_response.get("error").is_none());
        assert_eq!(adapter.requests().len(), 1);

        drop(client);
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[test]
    fn accepted_injection_lifetime_cap_is_enforced() {
        let injection = InjectionController::default();
        injection.start_turn();
        for index in 0..MAX_ACCEPTED_INJECTIONS {
            let message_id = wire::MessageId::new(format!("accepted-{index}"));
            injection
                .reserve(PendingInject {
                    message_id: message_id.clone(),
                    content: Vec::new(),
                    items: Vec::new(),
                    bytes: 0,
                    commitment: InjectCommitment::Reserved,
                })
                .expect("injection below lifetime cap");
            injection.commit(&message_id);
            injection.activate(&message_id);
            assert!(matches!(
                injection.revoke_transition(&message_id),
                PendingTransition::Applied
            ));
        }

        let error = injection
            .reserve(PendingInject {
                message_id: wire::MessageId::new("over-limit"),
                content: Vec::new(),
                items: Vec::new(),
                bytes: 0,
                commitment: InjectCommitment::Reserved,
            })
            .expect_err("accepted injection lifetime cap must reject new reservations");
        assert_eq!(i32::from(error.code), -32000);
        assert_eq!(
            error.data,
            Some(json!({
                "reason": "lifetime_limit_exceeded",
                "limit": MAX_ACCEPTED_INJECTIONS,
            }))
        );
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn accepted_steer_survives_approval_resolution() {
        let adapter = ApprovalAdapter::new();
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let adapter = adapter.clone();
            async move {
                AcpHeadlessRuntime::<ApprovalAdapter>::builder()
                    .agent_factory(ApprovalFactory { adapter })
                    .serve(agent_transport)
                    .await
            }
        });
        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let updates = Arc::clone(&updates);
                async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("approval-test", "1"),
                    ))
                    .block_task()
                    .await?;
                    let session_id = cx
                        .send_request(wire::NewSessionRequest::new(
                            std::env::current_dir()
                                .map_err(agent_client_protocol::Error::into_internal_error)?,
                        ))
                        .block_task()
                        .await?
                        .session_id;
                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
                    ))
                    .block_task()
                    .await?;
                    cx.send_request(wire::InjectSessionRequest::new(
                        session_id.clone(),
                        wire::SessionInjectMode::Steer,
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            "approval steer",
                        ))],
                    ))
                    .block_task()
                    .await?;
                    adapter.release();
                    adapter.release();
                    wait_for_idle(&updates, &session_id).await;
                    assert!(updates.lock().unwrap().iter().any(|(id, update)| {
                        id == &session_id
                            && matches!(
                                update,
                                wire::SessionUpdate::UserMessage(message)
                                    if matches!(
                                        &message.content,
                                        agent_client_protocol::schema::MaybeUndefined::Value(content)
                                            if content.iter().any(|block| {
                                                matches!(block, wire::ContentBlock::Text(text)
                                                    if text.text == "approval steer")
                                            })
                                    )
                            )
                    }));
                    Ok(())
                }
            });
        tokio::time::timeout(Duration::from_secs(4), client)
            .await
            .expect("approval client timed out")
            .expect("approval client failed");
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn session_inject_accepted_during_tool_runs_after_tool_boundary() {
        let adapter = MockAdapter::new();
        adapter.enqueue_many([
            tool_turn("post-tool"),
            TurnScript::new([ModelTurnEvent::Finished(ModelTurnResult {
                model: None,
                response_id: None,
                finish_reason: FinishReason::Completed,
                output_items: Vec::new(),
                usage: None,
                metadata: MetadataMap::new(),
            })]),
        ]);
        let tool = CompletionGatedTool::new();
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let factory = InjectToolFactory {
                adapter: adapter.clone(),
                tool: tool.clone(),
            };
            async move {
                AcpHeadlessRuntime::<MockAdapter>::builder()
                    .agent_factory(factory)
                    .serve(agent_transport)
                    .await
            }
        });

        let client =
            agent_client_protocol::Client
                .v2()
                .connect_with(client_transport, async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("test-client", "1"),
                    ))
                    .block_task()
                    .await?;
                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let session_id = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?
                        .session_id;
                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
                    ))
                    .block_task()
                    .await?;
                    tool.wait_for_entered().await;
                    cx.send_request(wire::InjectSessionRequest::new(
                        session_id.clone(),
                        wire::SessionInjectMode::Steer,
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            "after tool",
                        ))],
                    ))
                    .block_task()
                    .await?;
                    tool.release();
                    tokio::time::timeout(Duration::from_secs(2), async {
                        while adapter.observed().len() < 2 {
                            tokio::task::yield_now().await;
                        }
                    })
                    .await
                    .expect("continuation did not start");
                    let observed = adapter.observed();
                    assert_eq!(observed.len(), 2);
                    let contains_injection = |item: &Item| {
                        item.kind == ItemKind::User
                        && item.parts.iter().any(|part| {
                            matches!(part, Part::Text(text) if text.text == "after tool")
                        })
                    };
                    assert!(!observed[0].transcript.iter().any(contains_injection));
                    assert!(observed[1].transcript.iter().any(contains_injection));
                    cx.send_request(wire::CloseSessionRequest::new(session_id))
                        .block_task()
                        .await?;
                    Ok(())
                });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn delivery_and_revoke_race_has_one_linearized_outcome() {
        let adapter = GatedAdapter::new();
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let adapter = adapter.clone();
            async move {
                AcpHeadlessRuntime::<GatedAdapter>::builder()
                    .agent_factory(TestFactory { adapter })
                    .serve(agent_transport)
                    .await
            }
        });
        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let updates = Arc::clone(&updates);
                async move |cx| {
                    cx.send_request(wire::InitializeRequest::new(
                        wire::ProtocolVersion::V2,
                        wire::Implementation::new("race-test", "1"),
                    ))
                    .block_task()
                    .await?;
                    let session_id = cx
                        .send_request(wire::NewSessionRequest::new(
                            std::env::current_dir()
                                .map_err(agent_client_protocol::Error::into_internal_error)?,
                        ))
                        .block_task()
                        .await?
                        .session_id;
                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("race"))],
                    ))
                    .block_task()
                    .await?;
                    let accepted = cx
                        .send_request(wire::InjectSessionRequest::new(
                            session_id.clone(),
                            wire::SessionInjectMode::Steer,
                            vec![wire::ContentBlock::Text(wire::TextContent::new(
                                "racing steer",
                            ))],
                        ))
                        .block_task()
                        .await?;
                    let revoke = cx
                        .send_request(wire::RevokeInjectSessionRequest::new(
                            session_id.clone(),
                            accepted.message_id.clone(),
                        ))
                        .block_task();
                    adapter.release(2);
                    let revoked = match revoke.await {
                        Ok(_) => true,
                        Err(error) => {
                            assert_eq!(i32::from(error.code), -32010);
                            false
                        }
                    };
                    wait_for_idle(&updates, &session_id).await;
                    let was_delivered = updates.lock().unwrap().iter().any(|(id, update)| {
                        id == &session_id
                            && matches!(update, wire::SessionUpdate::UserMessage(message)
                                if message.message_id == accepted.message_id)
                    });
                    assert_ne!(revoked, was_delivered);
                    Ok(())
                }
            });
        tokio::time::timeout(Duration::from_secs(4), client)
            .await
            .expect("race client timed out")
            .expect("race client failed");
        server.abort();
        let _ = server.await;
    }

    #[cfg(feature = "unstable-inject")]
    #[tokio::test]
    async fn session_inject_steers_preserves_content_and_serializes_revoke_cancel() {
        let adapter = GatedAdapter::new();
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn({
            let adapter = adapter.clone();
            async move {
                AcpHeadlessRuntime::<GatedAdapter>::builder()
                    .agent_factory(TestFactory { adapter })
                    .serve(agent_transport)
                    .await
            }
        });

        let client = agent_client_protocol::Client
            .v2()
            .on_receive_notification(
                {
                    let updates = Arc::clone(&updates);
                    async move |notification: wire::UpdateSessionNotification, _cx| {
                        updates
                            .lock()
                            .unwrap()
                            .push((notification.session_id, notification.update));
                        Ok(())
                    }
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(client_transport, {
                let adapter = adapter.clone();
                let updates = Arc::clone(&updates);
                async move |cx| {
                    let initialize = cx
                        .send_request(wire::InitializeRequest::new(
                            wire::ProtocolVersion::V2,
                            wire::Implementation::new("test-client", "1"),
                        ))
                        .block_task()
                        .await?;
                    let inject = initialize
                        .capabilities
                        .session
                        .and_then(|session| session.inject)
                        .expect("inject capability advertised");
                    assert_eq!(inject.modes, vec![wire::SessionInjectMode::Steer]);
                    assert_eq!(
                        inject.steer_in_stream,
                        Some(vec![wire::SessionInjectSteerInStream::Finish])
                    );
                    assert_eq!(
                        inject.pending.and_then(|pending| pending.replace),
                        Some(true)
                    );

                    let cwd = std::env::current_dir()
                        .map_err(agent_client_protocol::Error::into_internal_error)?;
                    let session_id = cx
                        .send_request(wire::NewSessionRequest::new(cwd))
                        .block_task()
                        .await?
                        .session_id;
                    let idle_error = cx
                        .send_request(wire::InjectSessionRequest::new(
                            session_id.clone(),
                            wire::SessionInjectMode::Steer,
                            vec![wire::ContentBlock::Text(wire::TextContent::new("idle"))],
                        ))
                        .block_task()
                        .await
                        .expect_err("idle injection must fail");
                    assert_eq!(i32::from(idle_error.code), -32010);
                    assert_eq!(
                        idle_error.data,
                        Some(json!({ "reason": "no_running_turn" }))
                    );
                    let replace_error = cx
                        .send_request(wire::ReplaceInjectSessionRequest::new(
                            session_id.clone(),
                            "not-pending",
                            vec![wire::ContentBlock::Text(wire::TextContent::new("replace"))],
                        ))
                        .block_task()
                        .await
                        .expect_err("unknown pending message must fail");
                    assert_eq!(i32::from(replace_error.code), -32002);
                    assert_eq!(
                        replace_error.data,
                        Some(json!({
                            "reason": "unknown_message_id",
                            "messageId": "not-pending",
                        }))
                    );

                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("start"))],
                    ))
                    .block_task()
                    .await?;
                    let content = vec![
                        wire::ContentBlock::Text(wire::TextContent::new("steer")),
                        wire::ContentBlock::ResourceLink(wire::ResourceLink::new(
                            "notes",
                            "file:///tmp/notes.txt",
                        )),
                    ];
                    let accepted = tokio::time::timeout(
                        Duration::from_millis(250),
                        cx.send_request(wire::InjectSessionRequest::new(
                            session_id.clone(),
                            wire::SessionInjectMode::Steer,
                            content.clone(),
                        ))
                        .block_task(),
                    )
                    .await
                    .expect("inject acceptance waited for the model")?;
                    assert!(
                        !updates.lock().unwrap().iter().any(|(id, update)| {
                            id == &session_id
                                && matches!(update, wire::SessionUpdate::UserMessage(message)
                                if message.message_id == accepted.message_id)
                        }),
                        "UserMessage was emitted before the inject response"
                    );
                    let content = vec![
                        wire::ContentBlock::Text(wire::TextContent::new("replacement")),
                        content[1].clone(),
                    ];
                    let replaced = cx
                        .send_request(wire::ReplaceInjectSessionRequest::new(
                            session_id.clone(),
                            accepted.message_id.clone(),
                            content.clone(),
                        ))
                        .block_task()
                        .await?;
                    assert_eq!(replaced.message_id, accepted.message_id);
                    adapter.release(2);
                    wait_for_idle_count(&updates, &session_id, 1).await;
                    let delivered =
                        updates
                            .lock()
                            .unwrap()
                            .iter()
                            .find_map(|(id, update)| match update {
                                wire::SessionUpdate::UserMessage(message)
                                    if id == &session_id
                                        && message.message_id == accepted.message_id =>
                                {
                                    Some(message.content.clone())
                                }
                                _ => None,
                            });
                    assert_eq!(
                        delivered,
                        Some(agent_client_protocol::schema::MaybeUndefined::Value(
                            content
                        ))
                    );
                    {
                        let ordered = updates.lock().unwrap();
                        let first_agent = ordered
                            .iter()
                            .position(|(id, update)| {
                                id == &session_id
                                    && matches!(update, wire::SessionUpdate::AgentMessageChunk(_))
                            })
                            .expect("first model stream");
                        let injected = ordered
                            .iter()
                            .position(|(id, update)| {
                                id == &session_id
                                    && matches!(update, wire::SessionUpdate::UserMessage(message)
                                if message.message_id == accepted.message_id)
                            })
                            .expect("injected user message");
                        let second_agent = ordered
                            .iter()
                            .rposition(|(id, update)| {
                                id == &session_id
                                    && matches!(update, wire::SessionUpdate::AgentMessageChunk(_))
                            })
                            .expect("continuation model stream");
                        assert!(first_agent < injected && injected < second_agent);
                        assert_eq!(
                            ordered
                                .iter()
                                .filter(|(id, update)| id == &session_id
                                    && matches!(
                                        update,
                                        wire::SessionUpdate::StateUpdate(
                                            wire::StateUpdate::Running(_)
                                        )
                                    ))
                                .count(),
                            1
                        );
                    }
                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("revoke"))],
                    ))
                    .block_task()
                    .await?;
                    let queue_error = cx
                        .send_request(wire::InjectSessionRequest::new(
                            session_id.clone(),
                            wire::SessionInjectMode::Queue,
                            vec![wire::ContentBlock::Text(wire::TextContent::new("queue"))],
                        ))
                        .block_task()
                        .await
                        .expect_err("queue mode must not be accepted");
                    assert_eq!(i32::from(queue_error.code), -32602);
                    let revoked = cx
                        .send_request(wire::InjectSessionRequest::new(
                            session_id.clone(),
                            wire::SessionInjectMode::Steer,
                            vec![wire::ContentBlock::Text(wire::TextContent::new("remove"))],
                        ))
                        .block_task()
                        .await?;
                    cx.send_request(wire::RevokeInjectSessionRequest::new(
                        session_id.clone(),
                        revoked.message_id.clone(),
                    ))
                    .block_task()
                    .await?;
                    adapter.release(1);
                    wait_for_idle_count(&updates, &session_id, 2).await;
                    assert!(!updates.lock().unwrap().iter().any(|(id, update)| {
                        id == &session_id
                            && matches!(update, wire::SessionUpdate::UserMessage(message)
                                if message.message_id == revoked.message_id)
                    }));
                    let delivered_error = cx
                        .send_request(wire::RevokeInjectSessionRequest::new(
                            session_id.clone(),
                            accepted.message_id.clone(),
                        ))
                        .block_task()
                        .await
                        .expect_err("old delivered injection cannot be revoked in a later turn");
                    assert_eq!(i32::from(delivered_error.code), -32010);
                    assert_eq!(
                        delivered_error.data,
                        Some(json!({
                            "reason": "already_delivered",
                            "messageId": accepted.message_id,
                        }))
                    );

                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new("cancel"))],
                    ))
                    .block_task()
                    .await?;
                    let cancelled = cx
                        .send_request(wire::InjectSessionRequest::new(
                            session_id.clone(),
                            wire::SessionInjectMode::Steer,
                            vec![wire::ContentBlock::Text(wire::TextContent::new("drop"))],
                        ))
                        .block_task()
                        .await?;
                    cx.send_notification(wire::CancelSessionNotification::new(session_id.clone()))?;
                    wait_for_idle_count(&updates, &session_id, 3).await;
                    assert!(!updates.lock().unwrap().iter().any(|(id, update)| {
                        id == &session_id
                            && matches!(update, wire::SessionUpdate::UserMessage(message)
                                if message.message_id == cancelled.message_id)
                    }));
                    let stop_reason = updates.lock().unwrap().iter().rev().find_map(
                        |(id, update)| match update {
                            wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(idle))
                                if id == &session_id =>
                            {
                                idle.stop_reason.clone()
                            }
                            _ => None,
                        },
                    );
                    assert_eq!(stop_reason, Some(wire::StopReason::Cancelled));

                    cx.send_request(wire::PromptRequest::new(
                        session_id.clone(),
                        vec![wire::ContentBlock::Text(wire::TextContent::new(
                            "after cancel",
                        ))],
                    ))
                    .block_task()
                    .await?;
                    adapter.release(2);
                    wait_for_idle_count(&updates, &session_id, 4).await;
                    assert!(
                        updates.lock().unwrap().iter().any(|(id, update)| {
                            id == &session_id
                                && matches!(update, wire::SessionUpdate::UserMessage(message)
                                if message.message_id == cancelled.message_id)
                        }),
                        "committed injection did not survive session cancellation"
                    );
                    cx.send_request(wire::CloseSessionRequest::new(session_id))
                        .block_task()
                        .await?;
                    Ok(())
                }
            });

        tokio::time::timeout(Duration::from_secs(8), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn same_session_prompt_race_admits_one_and_other_sessions_stay_independent() {
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn(async move {
            AcpHeadlessRuntime::<BlockingAdapter>::builder()
                .agent_factory(TestFactory {
                    adapter: BlockingAdapter,
                })
                .serve(agent_transport)
                .await
        });

        let client =
            agent_client_protocol::Client
                .v2()
                .on_receive_notification(
                    {
                        let updates = Arc::clone(&updates);
                        async move |notification: wire::UpdateSessionNotification, _cx| {
                            updates
                                .lock()
                                .unwrap()
                                .push((notification.session_id, notification.update));
                            Ok(())
                        }
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, {
                    let updates = Arc::clone(&updates);
                    async move |cx| {
                        cx.send_request(wire::InitializeRequest::new(
                            wire::ProtocolVersion::V2,
                            wire::Implementation::new("test-client", "1"),
                        ))
                        .block_task()
                        .await?;
                        let cwd = std::env::current_dir()
                            .map_err(agent_client_protocol::Error::into_internal_error)?;
                        let first = cx
                            .send_request(wire::NewSessionRequest::new(cwd.clone()))
                            .block_task()
                            .await?;
                        let second = cx
                            .send_request(wire::NewSessionRequest::new(cwd))
                            .block_task()
                            .await?;

                        let first_prompt = cx
                            .send_request(wire::PromptRequest::new(
                                first.session_id.clone(),
                                vec![wire::ContentBlock::Text(wire::TextContent::new("first"))],
                            ))
                            .block_task();
                        let competing_prompt = cx
                            .send_request(wire::PromptRequest::new(
                                first.session_id.clone(),
                                vec![wire::ContentBlock::Text(wire::TextContent::new(
                                    "competing",
                                ))],
                            ))
                            .block_task();
                        let (first_result, competing_result) =
                            tokio::time::timeout(Duration::from_millis(250), async {
                                tokio::join!(first_prompt, competing_prompt)
                            })
                            .await
                            .expect("same-session prompt admission timed out");
                        assert_ne!(
                            first_result.is_ok(),
                            competing_result.is_ok(),
                            "exactly one same-session prompt must be admitted"
                        );

                        tokio::time::timeout(
                            Duration::from_millis(250),
                            cx.send_request(wire::PromptRequest::new(
                                second.session_id.clone(),
                                vec![wire::ContentBlock::Text(wire::TextContent::new("wait"))],
                            ))
                            .block_task(),
                        )
                        .await
                        .expect("prompt acceptance was blocked by another session")?;

                        cx.send_notification(wire::CancelSessionNotification::new(
                            first.session_id.clone(),
                        ))?;
                        wait_for_idle(&updates, &first.session_id).await;
                        assert!(!updates.lock().unwrap().iter().any(|(id, update)| {
                            id == &second.session_id
                                && matches!(
                                    update,
                                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                                )
                        }));

                        cx.send_notification(wire::CancelSessionNotification::new(
                            second.session_id.clone(),
                        ))?;
                        wait_for_idle(&updates, &second.session_id).await;
                        for session_id in [&first.session_id, &second.session_id] {
                            let stop_reason = updates.lock().unwrap().iter().find_map(
                                |(id, update)| match update {
                                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(
                                        idle,
                                    )) if id == session_id => idle.stop_reason.clone(),
                                    _ => None,
                                },
                            );
                            assert_eq!(stop_reason, Some(wire::StopReason::Cancelled));
                        }
                        cx.send_request(wire::CloseSessionRequest::new(first.session_id))
                            .block_task()
                            .await?;
                        cx.send_request(wire::CloseSessionRequest::new(second.session_id))
                            .block_task()
                            .await?;
                        Ok(())
                    }
                });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;
    }
    #[tokio::test]
    async fn independent_prompts_are_accepted_immediately_and_cancel_separately() {
        let updates = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, agent_transport) = Channel::duplex();
        let server = tokio::spawn(async move {
            AcpHeadlessRuntime::<BlockingAdapter>::builder()
                .agent_factory(TestFactory {
                    adapter: BlockingAdapter,
                })
                .serve(agent_transport)
                .await
        });

        let client =
            agent_client_protocol::Client
                .v2()
                .on_receive_notification(
                    {
                        let updates = Arc::clone(&updates);
                        async move |notification: wire::UpdateSessionNotification, _cx| {
                            updates
                                .lock()
                                .unwrap()
                                .push((notification.session_id, notification.update));
                            Ok(())
                        }
                    },
                    agent_client_protocol::on_receive_notification!(),
                )
                .connect_with(client_transport, {
                    let updates = Arc::clone(&updates);
                    async move |cx| {
                        cx.send_request(wire::InitializeRequest::new(
                            wire::ProtocolVersion::V2,
                            wire::Implementation::new("test-client", "1"),
                        ))
                        .block_task()
                        .await?;
                        let cwd = std::env::current_dir()
                            .map_err(agent_client_protocol::Error::into_internal_error)?;
                        let first = cx
                            .send_request(wire::NewSessionRequest::new(cwd.clone()))
                            .block_task()
                            .await?;
                        let second = cx
                            .send_request(wire::NewSessionRequest::new(cwd))
                            .block_task()
                            .await?;

                        for session_id in [&first.session_id, &second.session_id] {
                            tokio::time::timeout(
                                Duration::from_millis(250),
                                cx.send_request(wire::PromptRequest::new(
                                    session_id.clone(),
                                    vec![wire::ContentBlock::Text(wire::TextContent::new("wait"))],
                                ))
                                .block_task(),
                            )
                            .await
                            .expect("prompt acceptance was blocked by another session")?;
                        }

                        cx.send_notification(wire::CancelSessionNotification::new(
                            first.session_id.clone(),
                        ))?;
                        wait_for_idle(&updates, &first.session_id).await;
                        assert!(!updates.lock().unwrap().iter().any(|(id, update)| {
                            id == &second.session_id
                                && matches!(
                                    update,
                                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                                )
                        }));

                        cx.send_notification(wire::CancelSessionNotification::new(
                            second.session_id.clone(),
                        ))?;
                        wait_for_idle(&updates, &second.session_id).await;
                        for session_id in [&first.session_id, &second.session_id] {
                            let stop_reason = updates.lock().unwrap().iter().find_map(
                                |(id, update)| match update {
                                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(
                                        idle,
                                    )) if id == session_id => idle.stop_reason.clone(),
                                    _ => None,
                                },
                            );
                            assert_eq!(stop_reason, Some(wire::StopReason::Cancelled));
                        }
                        cx.send_request(wire::CloseSessionRequest::new(first.session_id))
                            .block_task()
                            .await?;
                        cx.send_request(wire::CloseSessionRequest::new(second.session_id))
                            .block_task()
                            .await?;
                        Ok(())
                    }
                });

        tokio::time::timeout(Duration::from_secs(5), client)
            .await
            .expect("client timed out")
            .expect("client run");
        server.abort();
        let _ = server.await;
    }

    #[test]
    fn tool_execution_boundary_rotates_message_ids_without_a_terminal_result() {
        let integration = AcpIntegration::default();
        let (client, mut messages) = ClientHandle::channel();
        let acp_id = wire::SessionId::new("acp-session");
        let agentkit_id = AgentkitSessionId::new("agentkit-session");
        integration
            .bind(acp_id.clone(), agentkit_id.clone(), client)
            .expect("bind session");
        integration
            .install_prompt_state(
                &acp_id,
                Arc::new(AtomicU64::new(1)),
                Arc::new(Mutex::new(())),
            )
            .expect("install prompt state");
        let cancellation = CancellationController::new();
        integration
            .begin_prompt_owner(&acp_id, 1, cancellation.handle().checkpoint())
            .expect("begin prompt");

        for event in [
            AgentEvent::ContentDelta(Delta::BeginPart {
                part_id: PartId::new("before-tool"),
                kind: PartKind::Text,
            }),
            AgentEvent::ContentDelta(Delta::AppendText {
                part_id: PartId::new("before-tool"),
                chunk: "before".into(),
            }),
            AgentEvent::ToolExecutionStarted(ToolCallPart::new(
                ToolCallId::new("call"),
                "background_tool",
                json!({}),
            )),
            AgentEvent::ContentDelta(Delta::BeginPart {
                part_id: PartId::new("after-tool"),
                kind: PartKind::Text,
            }),
            AgentEvent::ContentDelta(Delta::AppendText {
                part_id: PartId::new("after-tool"),
                chunk: "after".into(),
            }),
        ] {
            integration.route_event(&agentkit_id, event);
        }

        let message_ids = std::iter::from_fn(|| messages.try_recv().ok())
            .filter_map(|message| match message {
                ClientMessage::Update(notification) => match notification.update {
                    wire::SessionUpdate::AgentMessageChunk(chunk) => Some(chunk.message_id),
                    _ => None,
                },
                ClientMessage::AcknowledgedUpdate { .. } | ClientMessage::Flush(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            message_ids,
            [
                wire::MessageId::new("acp-session-agent-2"),
                wire::MessageId::new("acp-session-agent-3"),
            ]
        );
    }

    #[test]
    fn integration_rejects_duplicate_agentkit_session_ids() {
        let integration = AcpIntegration::default();
        let (first_client, _first_rx) = ClientHandle::channel();
        let (second_client, _second_rx) = ClientHandle::channel();
        let agentkit_id = AgentkitSessionId::new("shared-agentkit-session");
        let first_acp = wire::SessionId::new("first-acp-session");
        let second_acp = wire::SessionId::new("second-acp-session");
        integration
            .bind(first_acp.clone(), agentkit_id.clone(), first_client)
            .expect("first binding");
        let error = integration
            .bind(second_acp.clone(), agentkit_id, second_client)
            .expect_err("duplicate AgentKit session id must be rejected");
        assert!(matches!(error, AcpRuntimeError::SessionAlreadyBound(_)));
        assert!(matches!(
            integration.session(&second_acp),
            Err(AcpRuntimeError::SessionNotFound(_))
        ));
        integration
            .unbind_session(&first_acp)
            .expect("unbind first");
    }
}

#[cfg(all(test, feature = "unstable-inject"))]
#[path = "v2/replace_tests.rs"]
mod replace_tests;
