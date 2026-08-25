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
    CancellationController, CancellationHandle, DataRef, Delta, FinishReason, Item, ItemKind,
    MetadataMap, Modality, Part, PartId, PartKind, SessionId as AgentkitSessionId, TextPart,
};
use agentkit_loop::{
    AgentEvent, LoopInterrupt, LoopObserver, LoopStep, ModelAdapter, ModelSession, ObservedEvent,
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
}

impl ClientHandle {
    fn channel() -> (Self, mpsc::UnboundedReceiver<ClientMessage>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    fn update(
        &self,
        session_id: wire::SessionId,
        update: wire::SessionUpdate,
    ) -> Result<(), AcpRuntimeError> {
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

struct IntegrationSession {
    acp_session_id: wire::SessionId,
    client: ClientHandle,
    next_message: AtomicU64,
    current_agent_message: Mutex<Option<wire::MessageId>>,
    part_kinds: Mutex<HashMap<PartId, PartKind>>,
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
        inner
            .by_agentkit
            .insert(agentkit_session_id, acp_session_id.clone());
        inner.by_acp.insert(
            acp_session_id.clone(),
            Arc::new(IntegrationSession {
                acp_session_id,
                client,
                next_message: AtomicU64::new(1),
                current_agent_message: Mutex::new(None),
                part_kinds: Mutex::new(HashMap::new()),
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

    fn begin_prompt(
        &self,
        session_id: &wire::SessionId,
    ) -> Result<(wire::MessageId, wire::MessageId), AcpRuntimeError> {
        let session = self.session(session_id)?;
        let sequence = session.next_message.fetch_add(1, Ordering::Relaxed);
        let user_id = wire::MessageId::new(format!("{session_id}-user-{sequence}"));
        let agent_id = wire::MessageId::new(format!("{session_id}-agent-{sequence}"));
        *session
            .current_agent_message
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(agent_id.clone());
        session
            .part_kinds
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        Ok((user_id, agent_id))
    }

    fn finish_prompt(&self, session_id: &wire::SessionId) {
        if let Ok(session) = self.session(session_id) {
            *session
                .current_agent_message
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = None;
            session
                .part_kinds
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clear();
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

        let message_id = session
            .current_agent_message
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let mut part_kinds = session
            .part_kinds
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(update) = event_to_update(&event, message_id.as_ref(), &mut part_kinds) else {
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

        let result = agent_client_protocol::Agent
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
            .connect_to(transport)
            .await;
        state.shutdown().await;
        result.map_err(|error| AcpRuntimeError::Sdk(error.to_string()))
    }
}

struct SessionEntry {
    commands: mpsc::UnboundedSender<SessionCommand>,
    cancellation: CancellationController,
    busy: Arc<AtomicBool>,
    closed: AtomicBool,
    lifecycle: Mutex<()>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    drain_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

enum SessionCommand {
    Prompt {
        request: wire::PromptRequest,
        items: Vec<Item>,
        cancellation_generation: u64,
        response: oneshot::Sender<Result<oneshot::Sender<()>, AcpRuntimeError>>,
    },
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

        let mut metadata = MetadataMap::new();
        metadata.insert("acp.protocol_version".into(), json!(2));
        metadata.insert("acp.cwd".into(), json!(request.cwd));
        metadata.insert(
            "acp.additional_directories".into(),
            json!(request.additional_directories),
        );

        self.integration.bind(
            acp_session_id.clone(),
            agentkit_session_id.clone(),
            client.clone(),
        )?;
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
        let busy = Arc::new(AtomicBool::new(false));
        let worker_busy = Arc::clone(&busy);
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
                worker_busy,
                rx,
            )
            .await;
        });
        let entry = Arc::new(SessionEntry {
            commands,
            cancellation,
            busy,
            closed: AtomicBool::new(false),
            lifecycle: Mutex::new(()),
            task: Mutex::new(Some(task)),
            drain_task: Mutex::new(Some(drain_task)),
        });
        self.sessions
            .lock()
            .await
            .insert(acp_session_id.clone(), entry);
        Ok(wire::NewSessionResponse::new(acp_session_id))
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
        let (tx, rx) = oneshot::channel();
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
            if entry
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(AcpRuntimeError::Unsupported(
                    "session is already running a prompt".into(),
                ));
            }
            let cancellation_generation = entry.cancellation.handle().generation();
            if entry
                .commands
                .send(SessionCommand::Prompt {
                    request,
                    items,
                    cancellation_generation,
                    response: tx,
                })
                .is_err()
            {
                entry.busy.store(false, Ordering::Release);
                return Err(AcpRuntimeError::ClientClosed);
            }
        }
        rx.await.map_err(|_| AcpRuntimeError::ClientClosed)?
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
            if !entry.closed.load(Ordering::Acquire) && entry.busy.load(Ordering::Acquire) {
                entry.cancellation.interrupt();
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
        for (session_id, entry) in sessions {
            stop_session(Arc::clone(&entry)).await;
            let _ = self.integration.unbind(&session_id);
            stop_client(entry).await;
        }
    }
}

async fn stop_session(entry: Arc<SessionEntry>) {
    {
        let _lifecycle = entry
            .lifecycle
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        entry.closed.store(true, Ordering::Release);
        entry.cancellation.interrupt();
    }
    let task = entry
        .task
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    if let Some(task) = task {
        task.abort();
        let _ = task.await;
    }
}

async fn stop_client(entry: Arc<SessionEntry>) {
    let task = entry
        .drain_task
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    if let Some(task) = task {
        let _ = task.await;
    }
}

async fn session_worker<S>(
    session_id: wire::SessionId,
    mut driver: agentkit_loop::LoopDriver<S>,
    client: ClientHandle,
    integration: Arc<AcpIntegration>,
    cancellation: CancellationHandle,
    busy: Arc<AtomicBool>,
    mut commands: mpsc::UnboundedReceiver<SessionCommand>,
) where
    S: ModelSession + Send + 'static,
{
    while let Some(SessionCommand::Prompt {
        request,
        items,
        cancellation_generation,
        response,
    }) = commands.recv().await
    {
        if let Err(error) = driver
            .submit_input(items)
            .map_err(|error| AcpRuntimeError::Loop(error.to_string()))
        {
            busy.store(false, Ordering::Release);
            let _ = response.send(Err(error));
            continue;
        }
        let (user_message_id, _) = match integration.begin_prompt(&session_id) {
            Ok(message_ids) => message_ids,
            Err(error) => {
                busy.store(false, Ordering::Release);
                let _ = response.send(Err(error));
                continue;
            }
        };
        let (start_tx, start_rx) = oneshot::channel();
        if response.send(Ok(start_tx)).is_err() || start_rx.await.is_err() {
            integration.finish_prompt(&session_id);
            busy.store(false, Ordering::Release);
            continue;
        }

        if client
            .update(
                session_id.clone(),
                wire::SessionUpdate::UserMessage(
                    wire::UserMessage::new(user_message_id).content(request.prompt),
                ),
            )
            .and_then(|()| {
                client.update(
                    session_id.clone(),
                    wire::SessionUpdate::StateUpdate(wire::StateUpdate::Running(
                        wire::RunningStateUpdate::new(),
                    )),
                )
            })
            .is_err()
        {
            integration.finish_prompt(&session_id);
            busy.store(false, Ordering::Release);
            continue;
        }

        let stop_reason = drive_prompt(&mut driver, &cancellation, cancellation_generation).await;
        if let Err(error) = client.flush().await {
            tracing::debug!(%error, "failed to flush ACP v2 output");
        }
        let _ = client.update(
            session_id.clone(),
            wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(
                wire::IdleStateUpdate::new().stop_reason(stop_reason),
            )),
        );
        integration.finish_prompt(&session_id);
        busy.store(false, Ordering::Release);
    }
}

async fn drive_prompt<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    cancellation: &CancellationHandle,
    generation: u64,
) -> wire::StopReason
where
    S: ModelSession + Send + 'static,
{
    loop {
        let step = tokio::select! {
            step = driver.next() => step,
            () = cancellation.cancelled_since(generation) => {
                return wire::StopReason::Cancelled;
            }
        };
        match step {
            Ok(LoopStep::Finished(result)) => {
                if result.finish_reason == FinishReason::ToolCall {
                    continue;
                }
                return if cancellation.is_cancelled_since(generation) {
                    wire::StopReason::Cancelled
                } else {
                    finish_reason_to_stop_reason(&result.finish_reason)
                };
            }
            Ok(LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))) => {
                return if cancellation.is_cancelled_since(generation) {
                    wire::StopReason::Cancelled
                } else {
                    wire::StopReason::EndTurn
                };
            }
            Ok(LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_))) => continue,
            Ok(LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_))) => {
                if let Err(error) = driver.cancel_pending_approvals().await {
                    tracing::debug!(%error, "failed to cancel unsupported ACP v2 approval");
                }
                return if cancellation.is_cancelled_since(generation) {
                    wire::StopReason::Cancelled
                } else {
                    wire::StopReason::Refusal
                };
            }
            Err(error) => {
                tracing::debug!(%error, "ACP v2 agent loop failed");
                return if cancellation.is_cancelled_since(generation) {
                    wire::StopReason::Cancelled
                } else {
                    wire::StopReason::Refusal
                };
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
    message_id: Option<&wire::MessageId>,
    part_kinds: &mut HashMap<PartId, PartKind>,
) -> Option<wire::SessionUpdate> {
    match event {
        AgentEvent::ContentDelta(delta) => delta_to_update(delta, message_id, part_kinds),
        AgentEvent::TurnFinished(_) => {
            part_kinds.clear();
            None
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
    message_id: Option<&wire::MessageId>,
    part_kinds: &mut HashMap<PartId, PartKind>,
) -> Option<wire::SessionUpdate> {
    match delta {
        Delta::BeginPart { part_id, kind } => {
            part_kinds.insert(part_id.clone(), *kind);
            None
        }
        Delta::AppendText { part_id, chunk } => {
            let message_id = message_id?.clone();
            let content = wire::ContentBlock::Text(wire::TextContent::new(chunk.clone()));
            match part_kinds.get(part_id) {
                Some(PartKind::Reasoning) => Some(wire::SessionUpdate::AgentThoughtChunk(
                    wire::ContentChunk::new(content, message_id),
                )),
                Some(PartKind::Text) | None => Some(wire::SessionUpdate::AgentMessageChunk(
                    wire::ContentChunk::new(content, message_id),
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

fn finish_reason_to_stop_reason(reason: &FinishReason) -> wire::StopReason {
    match reason {
        FinishReason::Completed | FinishReason::ToolCall | FinishReason::Other(_) => {
            wire::StopReason::EndTurn
        }
        FinishReason::MaxTokens => wire::StopReason::MaxTokens,
        FinishReason::Cancelled => wire::StopReason::Cancelled,
        FinishReason::Blocked | FinishReason::Error => wire::StopReason::Refusal,
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
    use std::time::Duration;

    use agent_client_protocol::Channel;
    use agentkit_core::{ItemKind, TurnCancellation};
    use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
    use agentkit_loop::{
        Agent, LoopError, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult, SessionConfig,
        TurnRequest,
    };

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

    async fn wait_for_idle(
        updates: &Arc<Mutex<Vec<(wire::SessionId, wire::SessionUpdate)>>>,
        session_id: &wire::SessionId,
    ) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if updates.lock().unwrap().iter().any(|(id, update)| {
                    id == session_id
                        && matches!(
                            update,
                            wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_))
                        )
                }) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("idle update timed out");
    }

    #[tokio::test]
    async fn runtime_sequences_v2_updates_with_stable_message_ids_per_session() {
        let adapter = MockAdapter::new();
        adapter.enqueue(streamed_text("first output"));
        adapter.enqueue(streamed_text("second output"));
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
            assert_eq!(user_id, format!("{session_id}-user-1"));
            assert!(!agent_ids.is_empty());
            assert!(
                agent_ids
                    .iter()
                    .all(|id| id == &format!("{session_id}-agent-1"))
            );
            assert!(matches!(
                session_updates.last(),
                Some(wire::SessionUpdate::StateUpdate(wire::StateUpdate::Idle(_)))
            ));
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
}
