//! Opt-in runtime foundation for the experimental ACP protocol v2.
//!
//! Enable the `protocol-v2` crate feature to use this module. The feature maps
//! directly to the official `agent-client-protocol/unstable_protocol_v2`
//! feature. Root-level APIs remain the stable ACP v1 integration.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use agent_client_protocol::{Client, ConnectionTo, Handled};
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
use tokio::sync::{Mutex as AsyncMutex, mpsc, oneshot};

use crate::AcpRuntimeError;

/// Official upstream ACP v2 wire types.
///
/// These are gated by upstream's `unstable_protocol_v2` feature and can change
/// while ACP v2 is under development. No stable v1 wire type is re-exported
/// from this namespace.
pub use agent_client_protocol::schema::ProtocolVersion;
pub use agent_client_protocol::schema::v2::*;

/// Explicit namespace for the official upstream ACP v2 wire types.
pub mod wire {
    pub use agent_client_protocol::schema::ProtocolVersion;
    pub use agent_client_protocol::schema::v2::*;
}

enum ClientMessage {
    Update(Box<wire::UpdateSessionNotification>),
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

    fn update(
        &self,
        session_id: wire::SessionId,
        update: wire::SessionUpdate,
    ) -> Result<(), AcpRuntimeError> {
        #[cfg(test)]
        if let Some(hook) = &self.before_update {
            hook(&update);
        }
        self.tx
            .send(ClientMessage::Update(Box::new(
                wire::UpdateSessionNotification::new(session_id, update),
            )))
            .map_err(|_| AcpRuntimeError::ClientClosed)
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
    cx: ConnectionTo<Client>,
) {
    while let Some(message) = rx.recv().await {
        match message {
            ClientMessage::Update(notification) => {
                if let Err(error) = cx.send_notification(*notification) {
                    tracing::debug!(%error, "failed to send ACP v2 session update");
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
    client: ClientHandle,
    next_message: AtomicU64,
    current_messages: Mutex<Option<CurrentMessageIds>>,
    part_kinds: Mutex<HashMap<PartId, PartKind>>,
    unsupported_approval: Mutex<Option<(CancellationHandle, u64)>>,
    prompt_state: Mutex<Option<PromptState>>,
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

#[derive(Default)]
struct IntegrationInner {
    by_acp: HashMap<wire::SessionId, Arc<IntegrationSession>>,
    by_agentkit: HashMap<AgentkitSessionId, wire::SessionId>,
}

/// Routes agentkit loop output to ACP v2 `session/update` notifications.
///
/// Agent factories should install this value as their loop observer. The
/// headless runtime binds and unbinds sessions automatically.
#[derive(Clone, Default)]
pub struct AcpIntegration {
    inner: Arc<RwLock<IntegrationInner>>,
}

impl AcpIntegration {
    fn bind(
        &self,
        acp_session_id: wire::SessionId,
        agentkit_session_id: AgentkitSessionId,
        client: ClientHandle,
    ) -> Result<(), AcpRuntimeError> {
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
            .insert(agentkit_session_id, acp_session_id.clone());
        inner.by_acp.insert(
            acp_session_id.clone(),
            Arc::new(IntegrationSession {
                acp_session_id,
                client,
                next_message: AtomicU64::new(1),
                current_messages: Mutex::new(None),
                part_kinds: Mutex::new(HashMap::new()),
                unsupported_approval: Mutex::new(None),
                prompt_state: Mutex::new(None),
            }),
        );
        Ok(())
    }

    fn unbind(&self, session_id: &wire::SessionId) -> Result<(), AcpRuntimeError> {
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

    fn begin_prompt(
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
        finish_model_message(&session);
        Ok(wire::MessageId::new(format!(
            "{session_id}-user-{sequence}"
        )))
    }

    fn finish_prompt(&self, session_id: &wire::SessionId, owner: u64) {
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
        if let Err(error) = session.client.update(
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
                    if let Err(error) = session.client.update(
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
        if let Err(error) = session
            .client
            .update(session.acp_session_id.clone(), update)
        {
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

    /// Serves ACP v2 over a custom upstream SDK transport.
    pub async fn serve(
        self,
        transport: impl agent_client_protocol::ConnectTo<agent_client_protocol::Agent> + 'static,
    ) -> Result<(), AcpRuntimeError> {
        let factory = self
            .factory
            .ok_or(AcpRuntimeError::MissingField("agent_factory"))?;
        let state = Arc::new(RuntimeState::new(factory, self.name, self.version));
        let (shutdown, mut shutdown_rx) = oneshot::channel();
        let connection = agent_client_protocol::Agent
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
            )
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

struct SessionEntry {
    commands: mpsc::UnboundedSender<SessionCommand>,
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
    fn new(factory: Arc<dyn AcpAgentFactory<M>>, name: String, version: String) -> Self {
        Self {
            factory,
            integration: Arc::new(AcpIntegration::default()),
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
        .capabilities(headless_capabilities()))
    }

    async fn new_session(
        self: &Arc<Self>,
        request: wire::NewSessionRequest,
        cx: ConnectionTo<Client>,
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
        let lifecycle = Arc::new(Mutex::new(()));
        self.integration.bind(
            acp_session_id.clone(),
            agentkit_session_id.clone(),
            client.clone(),
        )?;
        if let Err(error) = self.integration.install_prompt_state(
            &acp_session_id,
            Arc::clone(&active_prompt),
            Arc::clone(&lifecycle),
        ) {
            let _ = self.integration.unbind(&acp_session_id);
            return Err(error);
        }
        let drain_task = tokio::spawn(drain_client_messages(client_messages, cx));
        let ctx = AcpAgentFactoryContext {
            acp_session_id: acp_session_id.clone(),
            agentkit_session_id,
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
                let _ = self.integration.unbind(&acp_session_id);
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
        let worker_session_id = acp_session_id.clone();
        let worker_cancellation = cancellation.handle();
        let task = tokio::spawn(async move {
            session_worker(
                worker_session_id,
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
                !entry.closed.load(Ordering::Acquire)
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
        if entry.closed.load(Ordering::Acquire) || entry.info.cwd != request.cwd {
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
        let items = prompt_to_items(&request)?;
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
                .lifecycle
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if entry.closed.load(Ordering::Acquire) {
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
                    entry.cancellation.interrupt();
                } else {
                    let queued = entry.active_prompt.load(Ordering::Acquire);
                    if queued != 0 {
                        entry.cancelled_prompt.store(queued, Ordering::Release);
                    } else {
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
        self.integration.unbind(&request.session_id)?;
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
            let _ = self.integration.unbind(session_id);
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
    let _lifecycle = entry
        .lifecycle
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    entry.closed.store(true, Ordering::Release);
    entry.cancellation.interrupt();
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
    if let Some(task) = take_task(&entry.drain_task) {
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
        integration.finish_prompt(session_id, owner);
    }
    clear_prompt_tracking(active_prompt, driving_prompt, cancelled_prompt, owner);
    if let Err(error) = client.update(
        session_id.clone(),
        wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(
            wire::IdleStateUpdate::new().stop_reason(error_stop_reason()),
        )),
    ) {
        tracing::debug!(%error, owner, "failed to queue ACP v2 error idle update");
    }
}

async fn session_worker<S>(
    session_id: wire::SessionId,
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
            continue;
        }

        if let Err(error) = driver.submit_input(items) {
            tracing::debug!(%error, owner, "failed to submit accepted ACP v2 prompt");
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
            match integration.begin_prompt(&session_id, owner, prompt_cancellation) {
                Ok(message_id) => message_id,
                Err(error) => {
                    tracing::debug!(%error, owner, "failed to begin accepted ACP v2 prompt");
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

        if let Err(error) = client.update(
            session_id.clone(),
            wire::SessionUpdate::UserMessage(
                wire::UserMessage::new(user_message_id).content(request.prompt),
            ),
        ) {
            tracing::debug!(%error, owner, "failed to publish accepted ACP v2 prompt");
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
        )
        .await;
        {
            let _lifecycle = lifecycle.lock().unwrap_or_else(|error| error.into_inner());
            integration.finish_prompt(&session_id, owner);
            clear_prompt_tracking(&active_prompt, &driving_prompt, &cancelled_prompt, owner);
        }
        if let Err(error) = client.flush().await {
            tracing::debug!(%error, "failed to flush ACP v2 output");
        }
        prefer_loop_update = true;
    }
}

async fn drive_prompt<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    integration: &AcpIntegration,
    session_id: &wire::SessionId,
    cancellation: &CancellationHandle,
    generation: u64,
) where
    S: ModelSession + Send + 'static,
{
    loop {
        // Cancellation is installed on the driver and its model/tool work. Keep
        // polling the driver so it can close interrupted tool calls and leave a
        // resumable transcript before the session becomes idle.
        match driver.next().await {
            Ok(LoopStep::Finished(result)) => {
                if result.finish_reason == FinishReason::ToolCall {
                    continue;
                }
                return;
            }
            Ok(LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))) => return,
            Ok(LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_))) => continue,
            Ok(LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_))) => {
                integration.mark_unsupported_approval(session_id, cancellation.clone(), generation);
                if let Err(error) = driver.cancel_pending_approvals().await {
                    integration.clear_unsupported_approval(session_id);
                    tracing::debug!(%error, "failed to cancel unsupported ACP v2 approval");
                }
                return;
            }
            Err(error) => {
                tracing::debug!(%error, "ACP v2 agent loop failed");
                return;
            }
        }
    }
}

fn prompt_to_items(request: &wire::PromptRequest) -> Result<Vec<Item>, AcpRuntimeError> {
    let mut user_parts = Vec::new();
    let mut context_items = Vec::new();

    for block in &request.prompt {
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

fn headless_capabilities() -> wire::AgentCapabilities {
    wire::AgentCapabilities::new().session(
        wire::SessionCapabilities::new().prompt(
            wire::PromptCapabilities::new()
                .image(wire::PromptImageCapabilities::new())
                .audio(wire::PromptAudioCapabilities::new())
                .embedded_context(wire::PromptEmbeddedContextCapabilities::new()),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    use agent_client_protocol::Channel;
    use agentkit_core::{
        ItemKind, ToolCallId, ToolOutput, ToolResultPart, TurnCancellation, TurnId,
    };
    use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
    use agentkit_integration_tests::mock_tool::BlockingTool;
    use agentkit_loop::{
        Agent, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult, SessionConfig,
        TurnRequest, TurnResult,
    };
    use agentkit_task_manager::{AsyncTaskManager, RoutingDecision};
    use agentkit_tools_core::{
        Tool, ToolContext, ToolError, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
    };
    use tokio::sync::Notify;

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
            .begin_prompt(&acp_id, 1, cancellation.handle().checkpoint())
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
                ClientMessage::Flush(_) => None,
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
        integration.unbind(&first_acp).expect("unbind first");
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
            .begin_prompt(&acp_id, 1, cancellation.handle().checkpoint())
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
                .begin_prompt(&acp_id, index, cancellation.handle().checkpoint())
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
            .begin_prompt(&acp_id, owner, cancellation.handle().checkpoint())
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
            .begin_prompt(&acp_id, 7, cancellation.handle().checkpoint())
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
            .begin_prompt(&acp_id, 9, cancellation.handle().checkpoint())
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
}
