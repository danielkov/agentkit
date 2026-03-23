//! Runtime-agnostic agent loop orchestration for sessions, turns, tools, and interrupts.
//!
//! `agentkit-loop` is the central coordination layer in the agentkit workspace.  It
//! drives a model through a multi-turn agentic loop, executing tool calls,
//! respecting permission checks, surfacing approval and authentication interrupts
//! to the host application, and optionally compacting the transcript when it grows
//! too large.
//!
//! # Architecture
//!
//! The main entry point is [`Agent`], constructed via [`AgentBuilder`].  After
//! calling [`Agent::start`] you receive a [`LoopDriver`] that yields
//! [`LoopStep`]s -- either a finished turn or an interrupt that requires host
//! resolution before the loop can continue.
//!
//! ```text
//! Agent::builder()
//!     .model(adapter)        // ModelAdapter implementation
//!     .tools(registry)       // ToolRegistry with registered tools
//!     .permissions(checker)  // PermissionChecker for gating tool use
//!     .observer(obs)         // LoopObserver for streaming events
//!     .build()?
//!     .start(config).await?  -> LoopDriver
//!         .submit_input(...)
//!         .next().await?     -> LoopStep::Finished | LoopStep::Interrupt
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use agentkit_core::SessionId;
//! use agentkit_loop::{Agent, SessionConfig};
//! # // Trait import needed for .start()
//! # use agentkit_core::MetadataMap;
//!
//! # async fn example<M: agentkit_loop::ModelAdapter>(adapter: M) -> Result<(), agentkit_loop::LoopError> {
//! let agent = Agent::builder()
//!     .model(adapter)
//!     .build()?;
//!
//! let mut driver = agent
//!     .start(SessionConfig {
//!         session_id: SessionId::new("demo"),
//!         metadata: MetadataMap::new(),
//!     })
//!     .await?;
//! # Ok(())
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use agentkit_compaction::{
    CompactionConfig, CompactionContext, CompactionReason, CompactionResult,
};
use agentkit_core::{
    CancellationHandle, Delta, FinishReason, Item, ItemKind, MetadataMap, Part, SessionId, TaskId,
    TextPart, ToolCallPart, ToolOutput, ToolResultPart, TurnCancellation, Usage,
};
use agentkit_task_manager::{
    PendingLoopUpdates, SimpleTaskManager, TaskLaunchRequest, TaskManager, TaskResolution,
    TaskStartContext, TaskStartOutcome, TurnTaskUpdate,
};
#[cfg(test)]
use agentkit_tools_core::ToolContext;
use agentkit_tools_core::{
    ApprovalDecision, ApprovalRequest, AuthOperation, AuthRequest, AuthResolution,
    BasicToolExecutor, OwnedToolContext, PermissionChecker, ToolError, ToolExecutor, ToolRegistry,
    ToolRequest, ToolResources, ToolSpec,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const INTERRUPTED_METADATA_KEY: &str = "agentkit.interrupted";
const INTERRUPT_REASON_METADATA_KEY: &str = "agentkit.interrupt_reason";
const INTERRUPT_STAGE_METADATA_KEY: &str = "agentkit.interrupt_stage";
const USER_CANCELLED_REASON: &str = "user_cancelled";

/// Configuration required to start a new model session.
///
/// Pass this to [`Agent::start`] to initialise the underlying [`ModelSession`]
/// and obtain a [`LoopDriver`].
///
/// # Example
///
/// ```rust
/// use agentkit_core::{MetadataMap, SessionId};
/// use agentkit_loop::SessionConfig;
///
/// let config = SessionConfig {
///     session_id: SessionId::new("my-session"),
///     metadata: MetadataMap::new(),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Unique identifier for the session.
    pub session_id: SessionId,
    /// Arbitrary key-value metadata forwarded to the model adapter.
    pub metadata: MetadataMap,
}

/// Payload sent to the model at the start of each turn.
///
/// The [`LoopDriver`] constructs this automatically from its internal state
/// and passes it to [`ModelSession::begin_turn`].  Model adapter authors
/// use the fields to build the provider-specific request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRequest {
    /// Session this turn belongs to.
    pub session_id: SessionId,
    /// Unique identifier for the current turn.
    pub turn_id: agentkit_core::TurnId,
    /// Full conversation transcript accumulated so far.
    pub transcript: Vec<Item>,
    /// Tool specifications the model may invoke during this turn.
    pub available_tools: Vec<ToolSpec>,
    /// Per-turn metadata (e.g. provider hints).
    pub metadata: MetadataMap,
}

/// Final result produced by a single model turn.
///
/// Returned inside [`ModelTurnEvent::Finished`] to signal that the model has
/// completed its generation for this turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelTurnResult {
    /// Why the model stopped generating (e.g. completed, tool call, length).
    pub finish_reason: FinishReason,
    /// Items the model produced during this turn (text, tool calls, etc.).
    pub output_items: Vec<Item>,
    /// Token usage statistics, if available.
    pub usage: Option<Usage>,
    /// Provider-specific metadata about the turn.
    pub metadata: MetadataMap,
}

/// Streaming event emitted by a [`ModelTurn`] during generation.
///
/// The [`LoopDriver`] consumes these events one-by-one via
/// [`ModelTurn::next_event`] and translates them into [`AgentEvent`]s for
/// observers and into transcript mutations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelTurnEvent {
    /// Incremental text or content delta from the model.
    Delta(Delta),
    /// The model is requesting a tool call.
    ToolCall(ToolCallPart),
    /// Updated token usage statistics.
    Usage(Usage),
    /// The model has finished generating for this turn.
    Finished(ModelTurnResult),
}

/// Factory for creating model sessions.
///
/// Implement this trait to integrate a model provider (e.g. OpenRouter,
/// Anthropic, a local LLM server) with the agent loop.  [`Agent`] holds a
/// single adapter and calls [`start_session`](ModelAdapter::start_session)
/// once when [`Agent::start`] is invoked.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_loop::{ModelAdapter, ModelSession, SessionConfig, LoopError};
/// use async_trait::async_trait;
///
/// struct MyAdapter;
///
/// #[async_trait]
/// impl ModelAdapter for MyAdapter {
///     type Session = MySession;
///
///     async fn start_session(&self, config: SessionConfig) -> Result<MySession, LoopError> {
///         // Initialize provider-specific session state here.
///         Ok(MySession { /* ... */ })
///     }
/// }
/// # struct MySession;
/// # #[async_trait]
/// # impl ModelSession for MySession {
/// #     type Turn = MyTurn;
/// #     async fn begin_turn(&mut self, _r: agentkit_loop::TurnRequest, _c: Option<agentkit_core::TurnCancellation>) -> Result<MyTurn, LoopError> { todo!() }
/// # }
/// # struct MyTurn;
/// # #[async_trait]
/// # impl agentkit_loop::ModelTurn for MyTurn {
/// #     async fn next_event(&mut self, _c: Option<agentkit_core::TurnCancellation>) -> Result<Option<agentkit_loop::ModelTurnEvent>, LoopError> { todo!() }
/// # }
/// ```
#[async_trait]
pub trait ModelAdapter: Send + Sync {
    /// The session type produced by this adapter.
    type Session: ModelSession;

    /// Create a new model session from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if the provider connection or initialisation fails.
    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError>;
}

/// An active model session that can produce sequential turns.
///
/// A session is created once per [`Agent::start`] call and lives for the
/// lifetime of the [`LoopDriver`].  Each call to [`begin_turn`](ModelSession::begin_turn)
/// hands the full transcript to the model and returns a streaming
/// [`ModelTurn`].
#[async_trait]
pub trait ModelSession: Send {
    /// The turn type produced by this session.
    type Turn: ModelTurn;

    /// Start a new turn, sending the transcript and available tools to the model.
    ///
    /// # Arguments
    ///
    /// * `request` -- the turn payload including transcript and tool specs.
    /// * `cancellation` -- optional handle the implementation should poll to
    ///   detect user-initiated cancellation.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] when the turn is cancelled, or a
    /// provider-specific error wrapped in [`LoopError`].
    async fn begin_turn(
        &mut self,
        request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Self::Turn, LoopError>;
}

/// A streaming model turn that yields events one at a time.
///
/// The loop driver calls [`next_event`](ModelTurn::next_event) repeatedly
/// until it returns `Ok(None)` (stream exhausted) or
/// `Ok(Some(ModelTurnEvent::Finished(_)))`.
#[async_trait]
pub trait ModelTurn: Send {
    /// Retrieve the next event from the model's response stream.
    ///
    /// Returns `Ok(None)` when the stream is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::Cancelled`] if `cancellation` fires, or a
    /// provider-specific error wrapped in [`LoopError`].
    async fn next_event(
        &mut self,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError>;
}

/// Observer hook for streaming agent events to the host application.
///
/// Register observers via [`AgentBuilder::observer`] to receive real-time
/// notifications about deltas, tool calls, usage, warnings, and lifecycle
/// events.
///
/// # Example
///
/// ```rust
/// use agentkit_loop::{AgentEvent, LoopObserver};
///
/// struct StdoutObserver;
///
/// impl LoopObserver for StdoutObserver {
///     fn handle_event(&mut self, event: AgentEvent) {
///         println!("{event:?}");
///     }
/// }
/// ```
pub trait LoopObserver: Send {
    /// Called synchronously for every [`AgentEvent`] emitted by the loop driver.
    fn handle_event(&mut self, event: AgentEvent);
}

/// Lifecycle and streaming events emitted by the [`LoopDriver`].
///
/// Observers (see [`LoopObserver`]) receive these events in the order they
/// occur.  They are useful for building UIs, logging, or telemetry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AgentEvent {
    /// The agent run has been initialised.
    RunStarted { session_id: SessionId },
    /// A new model turn is starting.
    TurnStarted {
        session_id: SessionId,
        turn_id: agentkit_core::TurnId,
    },
    /// User input has been accepted into the pending queue.
    InputAccepted {
        session_id: SessionId,
        items: Vec<Item>,
    },
    /// Incremental content delta from the model.
    ContentDelta(Delta),
    /// The model has requested a tool call.
    ToolCallRequested(ToolCallPart),
    /// A tool call requires explicit user approval before execution.
    ApprovalRequired(ApprovalRequest),
    /// A tool call requires authentication before execution.
    AuthRequired(AuthRequest),
    /// An approval interrupt has been resolved.
    ApprovalResolved { approved: bool },
    /// An authentication interrupt has been resolved.
    AuthResolved { provided: bool },
    /// Transcript compaction is about to begin.
    CompactionStarted {
        session_id: SessionId,
        turn_id: Option<agentkit_core::TurnId>,
        reason: CompactionReason,
    },
    /// Transcript compaction has finished.
    CompactionFinished {
        session_id: SessionId,
        turn_id: Option<agentkit_core::TurnId>,
        replaced_items: usize,
        transcript_len: usize,
        metadata: MetadataMap,
    },
    /// Updated token usage statistics.
    UsageUpdated(Usage),
    /// Non-fatal warning (e.g. a tool failure that was recovered from).
    Warning { message: String },
    /// The agent run has failed with an unrecoverable error.
    RunFailed { message: String },
    /// A turn has finished (successfully, via cancellation, etc.).
    TurnFinished(TurnResult),
}

/// Descriptor for a [`LoopInterrupt::AwaitingInput`] interrupt.
///
/// Returned when the driver has no pending input and needs the host to call
/// [`LoopDriver::submit_input`] before advancing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRequest {
    /// The session that is waiting for input.
    pub session_id: SessionId,
    /// Human-readable explanation of why input is needed.
    pub reason: String,
}

/// Outcome of a completed (or cancelled) turn.
///
/// Wrapped by [`LoopStep::Finished`] and also emitted as
/// [`AgentEvent::TurnFinished`] to observers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnResult {
    /// Identifier for the turn that produced this result.
    pub turn_id: agentkit_core::TurnId,
    /// Why the turn ended (completed, tool call, cancelled, etc.).
    pub finish_reason: FinishReason,
    /// Items produced during this turn (assistant text, tool results, etc.).
    pub items: Vec<Item>,
    /// Aggregated token usage, if reported by the model.
    pub usage: Option<Usage>,
    /// Additional metadata about the turn.
    pub metadata: MetadataMap,
}

/// An interrupt that pauses the agent loop until the host resolves it.
///
/// The loop returns an interrupt inside [`LoopStep::Interrupt`] whenever it
/// cannot proceed autonomously.  The host must resolve the interrupt
/// (via [`LoopDriver::resolve_approval`], [`LoopDriver::resolve_auth`], or
/// [`LoopDriver::submit_input`]) and then call [`LoopDriver::next`] again.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_loop::{LoopInterrupt, LoopStep};
/// # use agentkit_loop::LoopDriver;
/// # use agentkit_tools_core::ApprovalDecision;
///
/// # async fn handle<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>) -> Result<(), agentkit_loop::LoopError> {
/// match driver.next().await? {
///     LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(req)) => {
///         println!("Tool {} needs approval: {}", req.request_kind, req.summary);
///         driver.resolve_approval(ApprovalDecision::Approve)?;
///     }
///     LoopStep::Interrupt(LoopInterrupt::AuthRequest(req)) => {
///         println!("Auth required from provider: {}", req.provider);
///         // ... obtain credentials and call driver.resolve_auth(...)
///     }
///     LoopStep::Interrupt(LoopInterrupt::AwaitingInput(req)) => {
///         println!("Waiting for input: {}", req.reason);
///         // ... call driver.submit_input(...)
///     }
///     LoopStep::Finished(result) => {
///         println!("Turn finished: {:?}", result.finish_reason);
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopInterrupt {
    /// A tool call requires explicit approval before it can execute.
    ApprovalRequest(ApprovalRequest),
    /// A tool call requires authentication credentials.
    AuthRequest(AuthRequest),
    /// The driver has no pending input and needs the host to supply some.
    AwaitingInput(InputRequest),
}

/// The result of advancing the agent loop by one step.
///
/// Returned by [`LoopDriver::next`].  The host should pattern-match on this
/// to decide whether to continue the loop or resolve an interrupt first.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_loop::LoopStep;
/// # use agentkit_loop::LoopDriver;
///
/// # async fn run<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>) -> Result<(), agentkit_loop::LoopError> {
/// loop {
///     match driver.next().await? {
///         LoopStep::Finished(result) => {
///             println!("Turn complete: {:?}", result.finish_reason);
///             break;
///         }
///         LoopStep::Interrupt(interrupt) => {
///             // Resolve the interrupt, then continue the loop.
///             # break;
///         }
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LoopStep {
    /// The loop is paused and requires host action.
    Interrupt(LoopInterrupt),
    /// A turn has completed (or been cancelled).
    Finished(TurnResult),
}

/// A read-only snapshot of the loop driver's current state.
///
/// Obtained via [`LoopDriver::snapshot`].  Useful for persisting or
/// inspecting the conversation transcript without holding a mutable
/// reference to the driver.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoopSnapshot {
    /// Session identifier.
    pub session_id: SessionId,
    /// The full transcript accumulated so far.
    pub transcript: Vec<Item>,
    /// Input items queued but not yet consumed by a turn.
    pub pending_input: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DriverState {
    Idle,
    AwaitingApproval,
    AwaitingAuth,
}

#[derive(Clone, Debug)]
struct PendingApprovalToolCall {
    request: ApprovalRequest,
    decision: Option<ApprovalDecision>,
    turn_id: agentkit_core::TurnId,
    task_id: TaskId,
    call: ToolCallPart,
    tool_request: ToolRequest,
}

#[derive(Clone, Debug)]
struct PendingAuthToolCall {
    request: AuthRequest,
    resolution: Option<AuthResolution>,
    turn_id: agentkit_core::TurnId,
    task_id: TaskId,
    call: ToolCallPart,
    tool_request: ToolRequest,
}

#[derive(Clone, Debug, Default)]
struct ActiveToolRound {
    turn_id: agentkit_core::TurnId,
    pending_calls: VecDeque<(ToolCallPart, ToolRequest)>,
    background_pending: bool,
    foreground_progressed: bool,
}

/// A configured agent ready to start a session.
///
/// Build one with [`Agent::builder`], supplying at minimum a [`ModelAdapter`].
/// Then call [`Agent::start`] with a [`SessionConfig`] to obtain a
/// [`LoopDriver`] that drives the agentic loop.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_core::{MetadataMap, SessionId};
/// use agentkit_loop::{Agent, SessionConfig, LoopStep};
/// use agentkit_tools_core::ToolRegistry;
///
/// # async fn example<M: agentkit_loop::ModelAdapter>(adapter: M) -> Result<(), agentkit_loop::LoopError> {
/// let agent = Agent::builder()
///     .model(adapter)
///     .tools(ToolRegistry::new())
///     .build()?;
///
/// let mut driver = agent
///     .start(SessionConfig {
///         session_id: SessionId::new("s1"),
///         metadata: MetadataMap::new(),
///     })
///     .await?;
///
/// // Submit input and advance
/// # Ok(())
/// # }
/// ```
pub struct Agent<M>
where
    M: ModelAdapter,
{
    model: M,
    tools: ToolRegistry,
    task_manager: Arc<dyn TaskManager>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    compaction: Option<CompactionConfig>,
    observers: Vec<Box<dyn LoopObserver>>,
}

impl<M> Agent<M>
where
    M: ModelAdapter,
{
    /// Create a new [`AgentBuilder`] for configuring this agent.
    pub fn builder() -> AgentBuilder<M> {
        AgentBuilder::default()
    }

    /// Consume the agent and start a new session, returning a [`LoopDriver`].
    ///
    /// This calls [`ModelAdapter::start_session`] and emits an
    /// [`AgentEvent::RunStarted`] event to all registered observers.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if the model adapter fails to create a session.
    pub async fn start(self, config: SessionConfig) -> Result<LoopDriver<M::Session>, LoopError> {
        let session_id = config.session_id.clone();
        let session = self.model.start_session(config).await?;
        let tool_executor = Arc::new(BasicToolExecutor::new(self.tools.clone()));
        let mut driver = LoopDriver {
            session_id: session_id.clone(),
            session: Some(session),
            tool_executor,
            task_manager: self.task_manager,
            permissions: self.permissions,
            resources: self.resources,
            cancellation: self.cancellation,
            compaction: self.compaction,
            observers: self.observers,
            transcript: Vec::new(),
            pending_input: Vec::new(),
            pending_approval: None,
            pending_auth: None,
            active_tool_round: None,
            next_turn_index: 1,
            state: DriverState::Idle,
        };
        driver.emit(AgentEvent::RunStarted { session_id });
        Ok(driver)
    }
}

/// Builder for constructing an [`Agent`].
///
/// Obtained via [`Agent::builder`].  The only required field is
/// [`model`](AgentBuilder::model); all others have sensible defaults
/// (no tools, allow-all permissions, no compaction, no observers).
pub struct AgentBuilder<M>
where
    M: ModelAdapter,
{
    model: Option<M>,
    tools: ToolRegistry,
    task_manager: Option<Arc<dyn TaskManager>>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    compaction: Option<CompactionConfig>,
    observers: Vec<Box<dyn LoopObserver>>,
}

impl<M> Default for AgentBuilder<M>
where
    M: ModelAdapter,
{
    fn default() -> Self {
        Self {
            model: None,
            tools: ToolRegistry::new(),
            task_manager: None,
            permissions: Arc::new(AllowAllPermissions),
            resources: Arc::new(()),
            cancellation: None,
            compaction: None,
            observers: Vec::new(),
        }
    }
}

impl<M> AgentBuilder<M>
where
    M: ModelAdapter,
{
    /// Set the model adapter (required).
    pub fn model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the tool registry.  Defaults to an empty [`ToolRegistry`].
    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Set the task manager that schedules tool-call execution.
    ///
    /// Defaults to [`SimpleTaskManager`], which preserves the existing
    /// sequential request/response behavior.
    pub fn task_manager(mut self, manager: impl TaskManager + 'static) -> Self {
        self.task_manager = Some(Arc::new(manager));
        self
    }

    /// Set the permission checker that gates tool execution.
    ///
    /// Defaults to allowing all tool calls without prompting.
    pub fn permissions(mut self, permissions: impl PermissionChecker + 'static) -> Self {
        self.permissions = Arc::new(permissions);
        self
    }

    /// Set shared resources available to tool implementations.
    pub fn resources(mut self, resources: impl ToolResources + 'static) -> Self {
        self.resources = Arc::new(resources);
        self
    }

    /// Attach a [`CancellationHandle`] for cooperative cancellation of turns.
    pub fn cancellation(mut self, handle: CancellationHandle) -> Self {
        self.cancellation = Some(handle);
        self
    }

    /// Enable transcript compaction with the given configuration.
    ///
    /// When configured, the driver checks the compaction trigger before each
    /// turn and applies the compaction strategy if the transcript is too long.
    pub fn compaction(mut self, config: CompactionConfig) -> Self {
        self.compaction = Some(config);
        self
    }

    /// Register a [`LoopObserver`] that receives [`AgentEvent`]s.
    ///
    /// Multiple observers may be registered; they are called in order.
    pub fn observer(mut self, observer: impl LoopObserver + 'static) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    /// Consume the builder and produce an [`Agent`].
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if no model adapter was provided.
    pub fn build(self) -> Result<Agent<M>, LoopError> {
        let model = self
            .model
            .ok_or_else(|| LoopError::InvalidState("model adapter is required".into()))?;
        Ok(Agent {
            model,
            tools: self.tools,
            task_manager: self
                .task_manager
                .unwrap_or_else(|| Arc::new(SimpleTaskManager::new())),
            permissions: self.permissions,
            resources: self.resources,
            cancellation: self.cancellation,
            compaction: self.compaction,
            observers: self.observers,
        })
    }
}

/// The runtime driver that advances the agent loop step by step.
///
/// Obtained from [`Agent::start`].  The typical usage pattern is:
///
/// 1. Call [`submit_input`](LoopDriver::submit_input) to enqueue user messages.
/// 2. Call [`next`](LoopDriver::next) to run the next turn.
/// 3. Handle the returned [`LoopStep`]:
///    - [`LoopStep::Finished`] -- the turn completed, inspect the result.
///    - [`LoopStep::Interrupt`] -- resolve the interrupt and call `next` again.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_core::{Item, ItemKind, MetadataMap, Part, TextPart};
/// use agentkit_loop::{LoopDriver, LoopStep};
///
/// # async fn drive<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>) -> Result<(), agentkit_loop::LoopError> {
/// driver.submit_input(vec![Item {
///     id: None,
///     kind: ItemKind::User,
///     parts: vec![Part::Text(TextPart {
///         text: "Hello!".into(),
///         metadata: MetadataMap::new(),
///     })],
///     metadata: MetadataMap::new(),
/// }])?;
///
/// let step = driver.next().await?;
/// match step {
///     LoopStep::Finished(result) => println!("Done: {:?}", result.finish_reason),
///     LoopStep::Interrupt(interrupt) => {
///         // Resolve the interrupt (approval, auth, or input), then call next() again.
///         println!("Interrupted: {interrupt:?}");
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct LoopDriver<S>
where
    S: ModelSession,
{
    session_id: SessionId,
    session: Option<S>,
    tool_executor: Arc<dyn ToolExecutor>,
    task_manager: Arc<dyn TaskManager>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    compaction: Option<CompactionConfig>,
    observers: Vec<Box<dyn LoopObserver>>,
    transcript: Vec<Item>,
    pending_input: Vec<Item>,
    pending_approval: Option<PendingApprovalToolCall>,
    pending_auth: Option<PendingAuthToolCall>,
    active_tool_round: Option<ActiveToolRound>,
    next_turn_index: u64,
    state: DriverState,
}

impl<S> LoopDriver<S>
where
    S: ModelSession,
{
    fn start_task_via_manager(
        &self,
        task_id: Option<TaskId>,
        tool_request: ToolRequest,
        approved_request: Option<ApprovalRequest>,
        cancellation: Option<TurnCancellation>,
    ) -> impl std::future::Future<Output = Result<TaskStartOutcome, LoopError>> + Send + 'static
    {
        let task_manager = self.task_manager.clone();
        let tool_executor = self.tool_executor.clone();
        let permissions = self.permissions.clone();
        let resources = self.resources.clone();
        let session_id = self.session_id.clone();
        let turn_id = tool_request.turn_id.clone();
        let metadata = tool_request.metadata.clone();

        async move {
            task_manager
                .start_task(
                    TaskLaunchRequest {
                        task_id,
                        request: tool_request.clone(),
                        approved_request,
                    },
                    TaskStartContext {
                        executor: tool_executor,
                        tool_context: OwnedToolContext {
                            session_id,
                            turn_id,
                            metadata,
                            permissions,
                            resources,
                            cancellation,
                        },
                    },
                )
                .await
                .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))
        }
    }

    fn queue_resolution_interrupt(
        &mut self,
        turn_id: &agentkit_core::TurnId,
        resolution: TaskResolution,
    ) -> Option<LoopStep> {
        match resolution {
            TaskResolution::Item(item) => {
                self.transcript.push(item);
                None
            }
            TaskResolution::Approval(task) => {
                let call = ToolCallPart {
                    id: task.tool_request.call_id.clone(),
                    name: task.tool_request.tool_name.to_string(),
                    input: task.tool_request.input.clone(),
                    metadata: task.tool_request.metadata.clone(),
                };
                self.pending_approval = Some(PendingApprovalToolCall {
                    request: task.approval.clone(),
                    decision: None,
                    turn_id: turn_id.clone(),
                    task_id: task.task_id,
                    call,
                    tool_request: task.tool_request,
                });
                self.state = DriverState::AwaitingApproval;
                self.emit(AgentEvent::ApprovalRequired(task.approval.clone()));
                Some(LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(
                    task.approval,
                )))
            }
            TaskResolution::Auth(task) => {
                let call = ToolCallPart {
                    id: task.tool_request.call_id.clone(),
                    name: task.tool_request.tool_name.to_string(),
                    input: task.tool_request.input.clone(),
                    metadata: task.tool_request.metadata.clone(),
                };
                let request = upgrade_auth_request(task.auth, &task.tool_request, &call);
                self.pending_auth = Some(PendingAuthToolCall {
                    request: request.clone(),
                    resolution: None,
                    turn_id: turn_id.clone(),
                    task_id: task.task_id,
                    call,
                    tool_request: task.tool_request,
                });
                self.state = DriverState::AwaitingAuth;
                self.emit(AgentEvent::AuthRequired(request.clone()));
                Some(LoopStep::Interrupt(LoopInterrupt::AuthRequest(request)))
            }
        }
    }

    async fn drain_pending_loop_updates(&mut self) -> Result<(bool, Option<LoopStep>), LoopError> {
        let PendingLoopUpdates { mut resolutions } = self
            .task_manager
            .take_pending_loop_updates()
            .await
            .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
        let mut saw_items = false;
        while let Some(resolution) = resolutions.pop_front() {
            match resolution {
                TaskResolution::Item(item) => {
                    self.transcript.push(item);
                    saw_items = true;
                }
                TaskResolution::Approval(task) => {
                    let call = ToolCallPart {
                        id: task.tool_request.call_id.clone(),
                        name: task.tool_request.tool_name.to_string(),
                        input: task.tool_request.input.clone(),
                        metadata: task.tool_request.metadata.clone(),
                    };
                    self.state = DriverState::AwaitingApproval;
                    self.pending_approval = Some(PendingApprovalToolCall {
                        request: task.approval.clone(),
                        decision: None,
                        turn_id: task.tool_request.turn_id.clone(),
                        task_id: task.task_id,
                        call,
                        tool_request: task.tool_request,
                    });
                    self.emit(AgentEvent::ApprovalRequired(task.approval.clone()));
                    return Ok((
                        saw_items,
                        Some(LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(
                            task.approval,
                        ))),
                    ));
                }
                TaskResolution::Auth(task) => {
                    let call = ToolCallPart {
                        id: task.tool_request.call_id.clone(),
                        name: task.tool_request.tool_name.to_string(),
                        input: task.tool_request.input.clone(),
                        metadata: task.tool_request.metadata.clone(),
                    };
                    let request = upgrade_auth_request(task.auth, &task.tool_request, &call);
                    self.state = DriverState::AwaitingAuth;
                    self.pending_auth = Some(PendingAuthToolCall {
                        request: request.clone(),
                        resolution: None,
                        turn_id: task.tool_request.turn_id.clone(),
                        task_id: task.task_id,
                        call,
                        tool_request: task.tool_request,
                    });
                    self.emit(AgentEvent::AuthRequired(request.clone()));
                    return Ok((
                        saw_items,
                        Some(LoopStep::Interrupt(LoopInterrupt::AuthRequest(request))),
                    ));
                }
            }
        }
        Ok((saw_items, None))
    }

    async fn maybe_compact(
        &mut self,
        turn_id: Option<&agentkit_core::TurnId>,
        cancellation: Option<TurnCancellation>,
    ) -> Result<(), LoopError> {
        let Some(compaction) = self.compaction.as_ref().cloned() else {
            return Ok(());
        };
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return Err(LoopError::Cancelled);
        }
        let Some(reason) =
            compaction
                .trigger
                .should_compact(&self.session_id, turn_id, &self.transcript)
        else {
            return Ok(());
        };

        self.emit(AgentEvent::CompactionStarted {
            session_id: self.session_id.clone(),
            turn_id: turn_id.cloned(),
            reason: reason.clone(),
        });

        let CompactionResult {
            transcript,
            replaced_items,
            metadata,
        } = compaction
            .strategy
            .apply(
                agentkit_compaction::CompactionRequest {
                    session_id: self.session_id.clone(),
                    turn_id: turn_id.cloned(),
                    transcript: self.transcript.clone(),
                    reason,
                    metadata: compaction.metadata.clone(),
                },
                &mut CompactionContext {
                    backend: compaction.backend.as_deref(),
                    metadata: &compaction.metadata,
                    cancellation,
                },
            )
            .await
            .map_err(|error| match error {
                agentkit_compaction::CompactionError::Cancelled => LoopError::Cancelled,
                other => LoopError::Compaction(other.to_string()),
            })?;

        self.transcript = transcript;
        self.emit(AgentEvent::CompactionFinished {
            session_id: self.session_id.clone(),
            turn_id: turn_id.cloned(),
            replaced_items,
            transcript_len: self.transcript.len(),
            metadata,
        });
        Ok(())
    }

    async fn continue_active_tool_round(&mut self) -> Result<Option<LoopStep>, LoopError> {
        let Some(_) = self.active_tool_round.as_ref() else {
            return Ok(None);
        };
        loop {
            let cancellation = self
                .cancellation
                .as_ref()
                .map(CancellationHandle::checkpoint);
            let turn_id = self
                .active_tool_round
                .as_ref()
                .map(|active| active.turn_id.clone())
                .ok_or_else(|| LoopError::InvalidState("missing active tool round".into()))?;

            if cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                self.task_manager
                    .on_turn_interrupted(&turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                self.active_tool_round = None;
                return self.finish_cancelled(turn_id, Vec::new()).map(Some);
            }

            let next_call = self
                .active_tool_round
                .as_mut()
                .and_then(|active| active.pending_calls.pop_front());
            if let Some((_call, tool_request)) = next_call {
                match self
                    .start_task_via_manager(None, tool_request.clone(), None, cancellation.clone())
                    .await?
                {
                    TaskStartOutcome::Ready(resolution) => {
                        let resolution = *resolution;
                        if matches!(resolution, TaskResolution::Item(_))
                            && let Some(active) = self.active_tool_round.as_mut()
                        {
                            active.foreground_progressed = true;
                        }
                        if let Some(step) = self.queue_resolution_interrupt(&turn_id, resolution) {
                            return Ok(Some(step));
                        }
                        continue;
                    }
                    TaskStartOutcome::Pending { kind, .. } => {
                        if kind == agentkit_task_manager::TaskKind::Background
                            && let Some(active) = self.active_tool_round.as_mut()
                        {
                            active.background_pending = true;
                        }
                        continue;
                    }
                }
            }

            match self
                .task_manager
                .wait_for_turn(&turn_id, cancellation.clone())
                .await
                .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?
            {
                Some(TurnTaskUpdate::Resolution(resolution)) => {
                    let resolution = *resolution;
                    if matches!(resolution, TaskResolution::Item(_))
                        && let Some(active) = self.active_tool_round.as_mut()
                    {
                        active.foreground_progressed = true;
                    }
                    if let Some(step) = self.queue_resolution_interrupt(&turn_id, resolution) {
                        return Ok(Some(step));
                    }
                }
                Some(TurnTaskUpdate::Detached(_)) => {
                    if let Some(active) = self.active_tool_round.as_mut() {
                        active.background_pending = true;
                    }
                }
                None => {
                    if cancellation
                        .as_ref()
                        .is_some_and(TurnCancellation::is_cancelled)
                    {
                        self.task_manager
                            .on_turn_interrupted(&turn_id)
                            .await
                            .map_err(|error| {
                                LoopError::Tool(ToolError::Internal(error.to_string()))
                            })?;
                        self.active_tool_round = None;
                        return self.finish_cancelled(turn_id, Vec::new()).map(Some);
                    }
                    let active = self.active_tool_round.take().ok_or_else(|| {
                        LoopError::InvalidState("missing active tool round".into())
                    })?;
                    if active.background_pending && !active.foreground_progressed {
                        return Ok(None);
                    }
                    return Ok(Some(Box::pin(self.drive_turn(turn_id, false)).await?));
                }
            }
        }
    }

    async fn drive_turn(
        &mut self,
        turn_id: agentkit_core::TurnId,
        emit_started: bool,
    ) -> Result<LoopStep, LoopError> {
        let cancellation = self
            .cancellation
            .as_ref()
            .map(CancellationHandle::checkpoint);
        match self
            .maybe_compact(Some(&turn_id), cancellation.clone())
            .await
        {
            Ok(()) => {}
            Err(LoopError::Cancelled) => {
                return self.finish_cancelled(turn_id, interrupted_assistant_items());
            }
            Err(error) => return Err(error),
        }
        if emit_started {
            self.emit(AgentEvent::TurnStarted {
                session_id: self.session_id.clone(),
                turn_id: turn_id.clone(),
            });
        }
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return self.finish_cancelled(turn_id, interrupted_assistant_items());
        }

        let request = TurnRequest {
            session_id: self.session_id.clone(),
            turn_id: turn_id.clone(),
            transcript: self.transcript.clone(),
            available_tools: self.tool_executor.specs(),
            metadata: MetadataMap::new(),
        };

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| LoopError::InvalidState("model session is not available".into()))?;
        let mut turn = match session.begin_turn(request, cancellation.clone()).await {
            Ok(turn) => turn,
            Err(LoopError::Cancelled) => {
                self.task_manager
                    .on_turn_interrupted(&turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                return self.finish_cancelled(turn_id, interrupted_assistant_items());
            }
            Err(error) => return Err(error),
        };
        let mut saw_tool_call = false;
        let mut finished_result = None;

        while let Some(event) = match turn.next_event(cancellation.clone()).await {
            Ok(event) => event,
            Err(LoopError::Cancelled) => {
                self.task_manager
                    .on_turn_interrupted(&turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                return self.finish_cancelled(turn_id, interrupted_assistant_items());
            }
            Err(error) => return Err(error),
        } {
            if cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                self.task_manager
                    .on_turn_interrupted(&turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                return self.finish_cancelled(turn_id, interrupted_assistant_items());
            }
            match event {
                ModelTurnEvent::Delta(delta) => self.emit(AgentEvent::ContentDelta(delta)),
                ModelTurnEvent::Usage(usage) => self.emit(AgentEvent::UsageUpdated(usage)),
                ModelTurnEvent::ToolCall(call) => {
                    saw_tool_call = true;
                    self.emit(AgentEvent::ToolCallRequested(call.clone()));
                }
                ModelTurnEvent::Finished(result) => {
                    finished_result = Some(result);
                    break;
                }
            }
        }

        let result = finished_result.ok_or_else(|| {
            LoopError::Provider("model turn ended without a Finished event".into())
        })?;
        self.transcript.extend(result.output_items.clone());

        if saw_tool_call {
            let pending_calls = extract_tool_calls(&result.output_items)
                .into_iter()
                .map(|call| {
                    let tool_request = ToolRequest {
                        call_id: call.id.clone(),
                        tool_name: agentkit_tools_core::ToolName::new(call.name.clone()),
                        input: call.input.clone(),
                        session_id: self.session_id.clone(),
                        turn_id: turn_id.clone(),
                        metadata: call.metadata.clone(),
                    };
                    (call, tool_request)
                })
                .collect();
            self.active_tool_round = Some(ActiveToolRound {
                turn_id: turn_id.clone(),
                pending_calls,
                background_pending: false,
                foreground_progressed: false,
            });
            if let Some(step) = self.continue_active_tool_round().await? {
                return Ok(step);
            }
            return Ok(LoopStep::Interrupt(LoopInterrupt::AwaitingInput(
                InputRequest {
                    session_id: self.session_id.clone(),
                    reason: "driver is waiting for input".into(),
                },
            )));
        }

        let turn_result = TurnResult {
            turn_id,
            finish_reason: result.finish_reason,
            items: result.output_items,
            usage: result.usage,
            metadata: result.metadata,
        };
        self.emit(AgentEvent::TurnFinished(turn_result.clone()));
        Ok(LoopStep::Finished(turn_result))
    }

    async fn resume_after_auth(
        &mut self,
        pending: PendingAuthToolCall,
    ) -> Result<LoopStep, LoopError> {
        let resolution = pending
            .resolution
            .clone()
            .ok_or_else(|| LoopError::InvalidState("pending auth has no resolution".into()))?;
        match resolution {
            AuthResolution::Provided { .. } => match self
                .start_task_via_manager(
                    Some(pending.task_id.clone()),
                    pending.tool_request.clone(),
                    None,
                    self.cancellation
                        .as_ref()
                        .map(CancellationHandle::checkpoint),
                )
                .await?
            {
                TaskStartOutcome::Ready(resolution) => {
                    let resolution = *resolution;
                    if let Some(step) =
                        self.queue_resolution_interrupt(&pending.turn_id, resolution)
                    {
                        return Ok(step);
                    }
                }
                TaskStartOutcome::Pending { .. } => {}
            },
            AuthResolution::Cancelled { .. } => {
                self.transcript.push(Item {
                    id: None,
                    kind: ItemKind::Tool,
                    parts: vec![Part::ToolResult(ToolResultPart {
                        call_id: pending.call.id.clone(),
                        output: ToolOutput::Text("auth cancelled".into()),
                        is_error: true,
                        metadata: pending.call.metadata.clone(),
                    })],
                    metadata: MetadataMap::new(),
                });
            }
        }

        if let Some(step) = self.continue_active_tool_round().await? {
            Ok(step)
        } else {
            self.drive_turn(pending.turn_id, false).await
        }
    }

    async fn resume_after_approval(
        &mut self,
        pending: PendingApprovalToolCall,
    ) -> Result<LoopStep, LoopError> {
        let decision = pending
            .decision
            .clone()
            .ok_or_else(|| LoopError::InvalidState("pending approval has no decision".into()))?;

        match decision {
            ApprovalDecision::Approve => match self
                .start_task_via_manager(
                    Some(pending.task_id.clone()),
                    pending.tool_request.clone(),
                    Some(pending.request.clone()),
                    self.cancellation
                        .as_ref()
                        .map(CancellationHandle::checkpoint),
                )
                .await?
            {
                TaskStartOutcome::Ready(resolution) => {
                    let resolution = *resolution;
                    if let Some(step) =
                        self.queue_resolution_interrupt(&pending.turn_id, resolution)
                    {
                        return Ok(step);
                    }
                }
                TaskStartOutcome::Pending { .. } => {}
            },
            ApprovalDecision::Deny { reason } => {
                self.transcript.push(Item {
                    id: None,
                    kind: ItemKind::Tool,
                    parts: vec![Part::ToolResult(ToolResultPart {
                        call_id: pending.call.id.clone(),
                        output: ToolOutput::Text(
                            reason.unwrap_or_else(|| "approval denied".into()),
                        ),
                        is_error: true,
                        metadata: pending.call.metadata.clone(),
                    })],
                    metadata: MetadataMap::new(),
                });
            }
        }

        if let Some(step) = self.continue_active_tool_round().await? {
            Ok(step)
        } else {
            self.drive_turn(pending.turn_id, false).await
        }
    }

    fn finish_cancelled(
        &mut self,
        turn_id: agentkit_core::TurnId,
        items: Vec<Item>,
    ) -> Result<LoopStep, LoopError> {
        self.transcript.extend(items.clone());
        let turn_result = TurnResult {
            turn_id,
            finish_reason: FinishReason::Cancelled,
            items,
            usage: None,
            metadata: interrupted_metadata("turn"),
        };
        self.emit(AgentEvent::TurnFinished(turn_result.clone()));
        Ok(LoopStep::Finished(turn_result))
    }

    /// Enqueue user input items for the next turn.
    ///
    /// Items are buffered and consumed the next time [`next`](LoopDriver::next)
    /// is called.  Must not be called while an interrupt is pending.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if an interrupt is still unresolved.
    pub fn submit_input(&mut self, input: Vec<Item>) -> Result<(), LoopError> {
        if self.state != DriverState::Idle {
            return Err(LoopError::InvalidState(
                "cannot submit input while an interrupt is pending".into(),
            ));
        }
        self.emit(AgentEvent::InputAccepted {
            session_id: self.session_id.clone(),
            items: input.clone(),
        });
        self.pending_input.extend(input);
        Ok(())
    }

    /// Resolve a pending [`LoopInterrupt::ApprovalRequest`].
    ///
    /// After calling this, invoke [`next`](LoopDriver::next) to continue the
    /// loop.  If the decision is [`ApprovalDecision::Approve`] the tool call
    /// executes; if denied, an error result is fed back to the model.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if no approval is pending.
    pub fn resolve_approval(&mut self, decision: ApprovalDecision) -> Result<(), LoopError> {
        let Some(pending) = self.pending_approval.as_mut() else {
            return Err(LoopError::InvalidState(
                "no approval request is pending".into(),
            ));
        };
        pending.decision = Some(decision.clone());
        self.state = DriverState::Idle;
        self.emit(AgentEvent::ApprovalResolved {
            approved: matches!(decision, ApprovalDecision::Approve),
        });
        Ok(())
    }

    /// Resolve a pending [`LoopInterrupt::AuthRequest`].
    ///
    /// The resolution must reference the same request id as the pending
    /// [`AuthRequest`].  After calling this, invoke [`next`](LoopDriver::next)
    /// to continue the loop.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if no auth request is pending or
    /// if the resolution's request id does not match.
    pub fn resolve_auth(&mut self, resolution: AuthResolution) -> Result<(), LoopError> {
        let Some(pending) = self.pending_auth.as_mut() else {
            return Err(LoopError::InvalidState("no auth request is pending".into()));
        };
        if pending.request.id != resolution.request().id {
            return Err(LoopError::InvalidState(
                "auth resolution does not match the pending request".into(),
            ));
        }
        pending.resolution = Some(resolution.clone());
        self.state = DriverState::Idle;
        self.emit(AgentEvent::AuthResolved {
            provided: matches!(resolution, AuthResolution::Provided { .. }),
        });
        Ok(())
    }

    /// Take a read-only snapshot of the driver's current transcript and input queue.
    pub fn snapshot(&self) -> LoopSnapshot {
        LoopSnapshot {
            session_id: self.session_id.clone(),
            transcript: self.transcript.clone(),
            pending_input: self.pending_input.clone(),
        }
    }

    /// Advance the loop by one step.
    ///
    /// This is the main method for driving the agent.  It processes pending
    /// interrupt resolutions, consumes queued input, starts a model turn,
    /// executes tool calls, and returns once the turn finishes or an
    /// interrupt occurs.
    ///
    /// If no input is queued and no interrupt is pending, returns
    /// [`LoopStep::Interrupt(LoopInterrupt::AwaitingInput(..))`](LoopInterrupt::AwaitingInput).
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if called while an unresolved
    /// interrupt is pending, or propagates provider / tool / compaction errors.
    pub async fn next(&mut self) -> Result<LoopStep, LoopError> {
        if self.state != DriverState::Idle {
            return Err(LoopError::InvalidState(
                "cannot advance while an interrupt is pending".into(),
            ));
        }

        if self
            .pending_approval
            .as_ref()
            .is_some_and(|pending| pending.decision.is_some())
        {
            let pending = self
                .pending_approval
                .take()
                .ok_or_else(|| LoopError::InvalidState("missing pending approval state".into()))?;
            return self.resume_after_approval(pending).await;
        }

        if self
            .pending_auth
            .as_ref()
            .is_some_and(|pending| pending.resolution.is_some())
        {
            let pending = self
                .pending_auth
                .take()
                .ok_or_else(|| LoopError::InvalidState("missing pending auth state".into()))?;
            return self.resume_after_auth(pending).await;
        }

        if let Some(step) = self.continue_active_tool_round().await? {
            return Ok(step);
        }

        let (had_loop_updates, loop_step) = self.drain_pending_loop_updates().await?;
        if let Some(step) = loop_step {
            return Ok(step);
        }

        if self.pending_input.is_empty() && !had_loop_updates {
            return Ok(LoopStep::Interrupt(LoopInterrupt::AwaitingInput(
                InputRequest {
                    session_id: self.session_id.clone(),
                    reason: "driver is waiting for input".into(),
                },
            )));
        }

        let turn_id = agentkit_core::TurnId::new(format!("turn-{}", self.next_turn_index));
        self.next_turn_index += 1;
        self.transcript.append(&mut self.pending_input);
        self.drive_turn(turn_id, true).await
    }

    fn emit(&mut self, event: AgentEvent) {
        for observer in &mut self.observers {
            observer.handle_event(event.clone());
        }
    }
}

fn interrupted_metadata(stage: &str) -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert(INTERRUPTED_METADATA_KEY.into(), true.into());
    metadata.insert(
        INTERRUPT_REASON_METADATA_KEY.into(),
        USER_CANCELLED_REASON.into(),
    );
    metadata.insert(INTERRUPT_STAGE_METADATA_KEY.into(), stage.into());
    metadata
}

fn interrupted_assistant_items() -> Vec<Item> {
    vec![Item {
        id: None,
        kind: ItemKind::Assistant,
        parts: vec![Part::Text(TextPart {
            text: "Previous assistant response was interrupted by the user before completion."
                .into(),
            metadata: interrupted_metadata("assistant"),
        })],
        metadata: interrupted_metadata("assistant"),
    }]
}

fn extract_tool_calls(items: &[Item]) -> Vec<ToolCallPart> {
    let mut calls = Vec::new();
    for item in items {
        for part in &item.parts {
            if let Part::ToolCall(call) = part {
                calls.push(call.clone());
            }
        }
    }
    calls
}

fn upgrade_auth_request(
    mut request: AuthRequest,
    tool_request: &ToolRequest,
    _call: &ToolCallPart,
) -> AuthRequest {
    if matches!(request.operation, AuthOperation::ToolCall { .. }) {
        return request;
    }

    let prior_server_id = request.challenge.get("server_id").cloned();
    let mut metadata = tool_request.metadata.clone();
    if let Some(server_id) = prior_server_id {
        metadata.entry("server_id".into()).or_insert(server_id);
    }
    request.operation = AuthOperation::ToolCall {
        tool_name: tool_request.tool_name.0.clone(),
        input: tool_request.input.clone(),
        call_id: Some(tool_request.call_id.clone()),
        session_id: Some(tool_request.session_id.clone()),
        turn_id: Some(tool_request.turn_id.clone()),
        metadata,
    };
    request
}

struct AllowAllPermissions;

impl PermissionChecker for AllowAllPermissions {
    fn evaluate(
        &self,
        _request: &dyn agentkit_tools_core::PermissionRequest,
    ) -> agentkit_tools_core::PermissionDecision {
        agentkit_tools_core::PermissionDecision::Allow
    }
}

/// Errors that can occur while driving the agent loop.
#[derive(Debug, Error)]
pub enum LoopError {
    /// The driver was in an unexpected state for the requested operation.
    #[error("invalid driver state: {0}")]
    InvalidState(String),
    /// The current turn was cancelled via the [`CancellationHandle`].
    #[error("turn cancelled")]
    Cancelled,
    /// An error originating from the model provider.
    #[error("provider error: {0}")]
    Provider(String),
    /// An error originating from tool execution.
    #[error("tool error: {0}")]
    Tool(#[from] ToolError),
    /// An error that occurred during transcript compaction.
    #[error("compaction error: {0}")]
    Compaction(String),
    /// The requested operation is not supported.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc as StdArc, Mutex as StdMutex};

    use agentkit_compaction::{CompactionPipeline, CompactionTrigger, KeepRecentStrategy};
    use agentkit_core::{
        CancellationController, ItemKind, Part, TextPart, ToolCallId, ToolOutput, ToolResultPart,
    };
    use agentkit_task_manager::{
        AsyncTaskManager, RoutingDecision, TaskEvent, TaskManager, TaskManagerHandle,
        TaskRoutingPolicy,
    };
    use agentkit_tools_core::{
        FileSystemPermissionRequest, PermissionCode, PermissionDecision, PermissionDenial, Tool,
        ToolAnnotations, ToolName, ToolResult, ToolSpec,
    };
    use serde_json::{Value, json};
    use tokio::sync::Notify;
    use tokio::time::{Duration, timeout};

    use super::*;

    struct FakeAdapter;
    struct SlowAdapter;
    struct RecordingAdapter {
        seen_descriptions: StdArc<StdMutex<Vec<Vec<String>>>>,
    }
    struct MultiToolAdapter;

    struct FakeSession;
    struct SlowSession;
    struct RecordingSession {
        seen_descriptions: StdArc<StdMutex<Vec<Vec<String>>>>,
    }
    struct MultiToolSession;

    struct FakeTurn {
        events: VecDeque<ModelTurnEvent>,
    }

    struct SlowTurn {
        emitted: bool,
    }

    struct RecordingTurn {
        emitted: bool,
    }
    struct MultiToolTurn {
        events: VecDeque<ModelTurnEvent>,
    }

    #[async_trait]
    impl ModelAdapter for FakeAdapter {
        type Session = FakeSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(FakeSession)
        }
    }

    #[async_trait]
    impl ModelAdapter for SlowAdapter {
        type Session = SlowSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(SlowSession)
        }
    }

    #[async_trait]
    impl ModelAdapter for RecordingAdapter {
        type Session = RecordingSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(RecordingSession {
                seen_descriptions: self.seen_descriptions.clone(),
            })
        }
    }

    #[async_trait]
    impl ModelAdapter for MultiToolAdapter {
        type Session = MultiToolSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(MultiToolSession)
        }
    }

    #[async_trait]
    impl ModelSession for FakeSession {
        type Turn = FakeTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let has_tool_result = request.transcript.iter().any(|item| {
                item.kind == ItemKind::Tool
                    && item
                        .parts
                        .iter()
                        .any(|part| matches!(part, Part::ToolResult(_)))
            });
            let tool_name = request
                .available_tools
                .first()
                .map(|tool| tool.name.0.clone())
                .unwrap_or_else(|| "echo".into());

            let events = if has_tool_result {
                let result_text = request
                    .transcript
                    .iter()
                    .rev()
                    .find_map(|item| {
                        item.parts.iter().find_map(|part| match part {
                            Part::ToolResult(ToolResultPart {
                                output: ToolOutput::Text(text),
                                ..
                            }) => Some(text.clone()),
                            _ => None,
                        })
                    })
                    .unwrap_or_else(|| "missing".into());

                VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: format!("tool said: {result_text}"),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                    }],
                    usage: None,
                    metadata: MetadataMap::new(),
                })])
            } else {
                VecDeque::from([
                    ModelTurnEvent::ToolCall(agentkit_core::ToolCallPart {
                        id: ToolCallId::new("call-1"),
                        name: tool_name.clone(),
                        input: json!({ "value": "pong" }),
                        metadata: MetadataMap::new(),
                    }),
                    ModelTurnEvent::Finished(ModelTurnResult {
                        finish_reason: FinishReason::ToolCall,
                        output_items: vec![Item {
                            id: None,
                            kind: ItemKind::Assistant,
                            parts: vec![Part::ToolCall(agentkit_core::ToolCallPart {
                                id: ToolCallId::new("call-1"),
                                name: tool_name,
                                input: json!({ "value": "pong" }),
                                metadata: MetadataMap::new(),
                            })],
                            metadata: MetadataMap::new(),
                        }],
                        usage: None,
                        metadata: MetadataMap::new(),
                    }),
                ])
            };

            Ok(FakeTurn { events })
        }
    }

    #[async_trait]
    impl ModelSession for SlowSession {
        type Turn = SlowTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let should_block = request
                .transcript
                .iter()
                .rev()
                .find(|item| item.kind == ItemKind::User)
                .is_some_and(|item| {
                    item.parts.iter().any(|part| match part {
                        Part::Text(text) => text.text == "do the long task",
                        _ => false,
                    })
                });

            if should_block && let Some(cancellation) = cancellation {
                cancellation.cancelled().await;
                return Err(LoopError::Cancelled);
            }

            Ok(SlowTurn { emitted: false })
        }
    }

    #[async_trait]
    impl ModelSession for RecordingSession {
        type Turn = RecordingTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let descriptions = request
                .available_tools
                .iter()
                .map(|tool| tool.description.clone())
                .collect::<Vec<_>>();
            self.seen_descriptions.lock().unwrap().push(descriptions);

            Ok(RecordingTurn { emitted: false })
        }
    }

    #[async_trait]
    impl ModelSession for MultiToolSession {
        type Turn = MultiToolTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let has_tool_result = request.transcript.iter().any(|item| {
                item.kind == ItemKind::Tool
                    && item
                        .parts
                        .iter()
                        .any(|part| matches!(part, Part::ToolResult(_)))
            });

            let events = if has_tool_result {
                VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "mixed tools finished".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                    }],
                    usage: None,
                    metadata: MetadataMap::new(),
                })])
            } else {
                let foreground = agentkit_core::ToolCallPart {
                    id: ToolCallId::new("call-foreground"),
                    name: "foreground-wait".into(),
                    input: json!({}),
                    metadata: MetadataMap::new(),
                };
                let background = agentkit_core::ToolCallPart {
                    id: ToolCallId::new("call-background"),
                    name: "background-wait".into(),
                    input: json!({}),
                    metadata: MetadataMap::new(),
                };
                VecDeque::from([
                    ModelTurnEvent::ToolCall(foreground.clone()),
                    ModelTurnEvent::ToolCall(background.clone()),
                    ModelTurnEvent::Finished(ModelTurnResult {
                        finish_reason: FinishReason::ToolCall,
                        output_items: vec![Item {
                            id: None,
                            kind: ItemKind::Assistant,
                            parts: vec![Part::ToolCall(foreground), Part::ToolCall(background)],
                            metadata: MetadataMap::new(),
                        }],
                        usage: None,
                        metadata: MetadataMap::new(),
                    }),
                ])
            };

            Ok(MultiToolTurn { events })
        }
    }

    #[async_trait]
    impl ModelTurn for FakeTurn {
        async fn next_event(
            &mut self,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            Ok(self.events.pop_front())
        }
    }

    #[async_trait]
    impl ModelTurn for SlowTurn {
        async fn next_event(
            &mut self,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if let Some(cancellation) = cancellation
                && cancellation.is_cancelled()
            {
                return Err(LoopError::Cancelled);
            }

            if self.emitted {
                Ok(None)
            } else {
                self.emitted = true;
                Ok(Some(ModelTurnEvent::Finished(ModelTurnResult {
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "done".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                    }],
                    usage: None,
                    metadata: MetadataMap::new(),
                })))
            }
        }
    }

    #[async_trait]
    impl ModelTurn for RecordingTurn {
        async fn next_event(
            &mut self,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if self.emitted {
                Ok(None)
            } else {
                self.emitted = true;
                Ok(Some(ModelTurnEvent::Finished(ModelTurnResult {
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "done".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                    }],
                    usage: None,
                    metadata: MetadataMap::new(),
                })))
            }
        }
    }

    #[async_trait]
    impl ModelTurn for MultiToolTurn {
        async fn next_event(
            &mut self,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            Ok(self.events.pop_front())
        }
    }

    #[derive(Clone)]
    struct EchoTool {
        spec: ToolSpec,
    }

    impl Default for EchoTool {
        fn default() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("echo"),
                    description: "Echo back a value".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "value": { "type": "string" }
                        },
                        "required": ["value"],
                        "additionalProperties": false
                    }),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[derive(Clone)]
    struct DynamicSpecTool {
        spec: ToolSpec,
        version: StdArc<AtomicUsize>,
    }

    impl DynamicSpecTool {
        fn new(version: StdArc<AtomicUsize>) -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("dynamic"),
                    description: "dynamic version 0".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
                version,
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn proposed_requests(
            &self,
            request: &agentkit_tools_core::ToolRequest,
        ) -> Result<
            Vec<Box<dyn agentkit_tools_core::PermissionRequest>>,
            agentkit_tools_core::ToolError,
        > {
            Ok(vec![Box::new(FileSystemPermissionRequest::Read {
                path: "/tmp/echo".into(),
                metadata: request.metadata.clone(),
            })])
        }

        async fn invoke(
            &self,
            request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            let value = request
                .input
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    agentkit_tools_core::ToolError::InvalidInput("missing value".into())
                })?;

            Ok(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text(value.into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    #[async_trait]
    impl Tool for DynamicSpecTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn current_spec(&self) -> Option<ToolSpec> {
            let mut spec = self.spec.clone();
            spec.description = format!("dynamic version {}", self.version.load(Ordering::SeqCst));
            Some(spec)
        }

        async fn invoke(
            &self,
            request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            Ok(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text("ok".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    struct DenyFsReads;

    impl PermissionChecker for DenyFsReads {
        fn evaluate(
            &self,
            request: &dyn agentkit_tools_core::PermissionRequest,
        ) -> PermissionDecision {
            if request.kind() == "filesystem.read" {
                return PermissionDecision::Deny(PermissionDenial {
                    code: PermissionCode::PathNotAllowed,
                    message: "reads denied in test".into(),
                    metadata: MetadataMap::new(),
                });
            }

            PermissionDecision::Allow
        }
    }

    struct ApproveFsReads;

    impl PermissionChecker for ApproveFsReads {
        fn evaluate(
            &self,
            request: &dyn agentkit_tools_core::PermissionRequest,
        ) -> PermissionDecision {
            if request.kind() == "filesystem.read" {
                return PermissionDecision::RequireApproval(ApprovalRequest {
                    task_id: None,
                    id: "approval:fs-read".into(),
                    request_kind: request.kind().into(),
                    reason: agentkit_tools_core::ApprovalReason::SensitivePath,
                    summary: request.summary(),
                    metadata: request.metadata().clone(),
                });
            }

            PermissionDecision::Allow
        }
    }

    struct CountTrigger;

    impl CompactionTrigger for CountTrigger {
        fn should_compact(
            &self,
            _session_id: &SessionId,
            _turn_id: Option<&agentkit_core::TurnId>,
            transcript: &[Item],
        ) -> Option<agentkit_compaction::CompactionReason> {
            (transcript.len() >= 2)
                .then_some(agentkit_compaction::CompactionReason::TranscriptTooLong)
        }
    }

    struct RecordingObserver {
        events: StdArc<StdMutex<Vec<AgentEvent>>>,
    }

    impl LoopObserver for RecordingObserver {
        fn handle_event(&mut self, event: AgentEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    #[derive(Clone)]
    struct AuthTool {
        spec: ToolSpec,
    }

    impl Default for AuthTool {
        fn default() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("auth-tool"),
                    description: "Always requires auth".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[async_trait]
    impl Tool for AuthTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            let mut challenge = MetadataMap::new();
            challenge.insert("server_id".into(), json!("mock"));
            challenge.insert("scope".into(), json!("secret.read"));

            Err(agentkit_tools_core::ToolError::AuthRequired(Box::new(
                AuthRequest {
                    task_id: None,
                    id: "auth-1".into(),
                    provider: "mcp.mock".into(),
                    operation: AuthOperation::ToolCall {
                        tool_name: request.tool_name.0,
                        input: request.input,
                        call_id: Some(request.call_id),
                        session_id: Some(request.session_id),
                        turn_id: Some(request.turn_id),
                        metadata: request.metadata,
                    },
                    challenge,
                },
            )))
        }
    }

    #[derive(Clone)]
    struct BlockingTool {
        spec: ToolSpec,
        entered: StdArc<AtomicBool>,
        release: StdArc<Notify>,
        output: &'static str,
    }

    impl BlockingTool {
        fn new(
            name: &str,
            entered: StdArc<AtomicBool>,
            release: StdArc<Notify>,
            output: &'static str,
        ) -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new(name),
                    description: format!("blocking tool {name}"),
                    input_schema: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
                entered,
                release,
                output,
            }
        }
    }

    #[async_trait]
    impl Tool for BlockingTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            self.entered.store(true, Ordering::SeqCst);
            self.release.notified().await;
            Ok(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text(self.output.into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    struct NameRoutingPolicy {
        routes: Vec<(String, RoutingDecision)>,
    }

    impl NameRoutingPolicy {
        fn new(routes: impl IntoIterator<Item = (impl Into<String>, RoutingDecision)>) -> Self {
            Self {
                routes: routes
                    .into_iter()
                    .map(|(name, decision)| (name.into(), decision))
                    .collect(),
            }
        }
    }

    impl TaskRoutingPolicy for NameRoutingPolicy {
        fn route(&self, request: &ToolRequest) -> RoutingDecision {
            self.routes
                .iter()
                .find(|(name, _)| name == &request.tool_name.0)
                .map(|(_, decision)| *decision)
                .unwrap_or(RoutingDecision::Foreground)
        }
    }

    async fn wait_for_task_event(handle: &TaskManagerHandle) -> TaskEvent {
        timeout(Duration::from_secs(1), handle.next_event())
            .await
            .expect("timed out waiting for task event")
            .expect("task event stream ended unexpectedly")
    }

    async fn wait_until_entered(flag: &AtomicBool) {
        timeout(Duration::from_secs(1), async {
            while !flag.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("task never entered execution");
    }

    #[tokio::test]
    async fn loop_continues_after_completed_tool_call() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-1"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();

        match result {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
                assert_eq!(turn.items.len(), 1);
                match &turn.items[0].parts[0] {
                    Part::Text(text) => assert_eq!(text.text, "tool said: pong"),
                    other => panic!("unexpected part: {other:?}"),
                }
            }
            other => panic!("unexpected loop step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_uses_injected_permission_checker() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(DenyFsReads)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-2"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();

        match result {
            LoopStep::Finished(turn) => match &turn.items[0].parts[0] {
                Part::Text(text) => assert!(text.text.contains("tool permission denied")),
                other => panic!("unexpected part: {other:?}"),
            },
            other => panic!("unexpected loop step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_surfaces_auth_interruptions_from_tools() {
        let tools = ToolRegistry::new().with(AuthTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-3"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();

        match result {
            LoopStep::Interrupt(LoopInterrupt::AuthRequest(request)) => {
                assert!(request.task_id.is_some());
                assert_eq!(request.provider, "mcp.mock");
                assert_eq!(request.challenge.get("scope"), Some(&json!("secret.read")));
                match request.operation {
                    AuthOperation::ToolCall { tool_name, .. } => {
                        assert_eq!(tool_name, "auth-tool");
                    }
                    other => panic!("unexpected auth operation: {other:?}"),
                }
            }
            other => panic!("unexpected loop step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn async_task_manager_background_round_requires_explicit_continue() {
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background-wait",
            RoutingDecision::Background,
        )]));
        let handle = task_manager.handle();
        let tools = ToolRegistry::new().with(BlockingTool::new(
            "background-wait",
            entered.clone(),
            release.clone(),
            "background-done",
        ));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(AllowAllPermissions)
            .task_manager(task_manager)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-background"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let first = driver.next().await.unwrap();
        match first {
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {}
            other => panic!("unexpected first loop step: {other:?}"),
        }

        match wait_for_task_event(&handle).await {
            TaskEvent::Started(snapshot) => assert_eq!(snapshot.tool_name, "background-wait"),
            other => panic!("unexpected task event: {other:?}"),
        }
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();

        match wait_for_task_event(&handle).await {
            TaskEvent::Completed(_, result) => {
                assert_eq!(result.output, ToolOutput::Text("background-done".into()))
            }
            other => panic!("unexpected completion event: {other:?}"),
        }

        let resumed = driver.next().await.unwrap();
        match resumed {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
                match &turn.items[0].parts[0] {
                    Part::Text(text) => assert_eq!(text.text, "tool said: background-done"),
                    other => panic!("unexpected part after resume: {other:?}"),
                }
            }
            other => panic!("unexpected resumed step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_can_cancel_a_turn_and_continue_after_new_input() {
        let controller = CancellationController::new();
        let agent = Agent::builder()
            .model(SlowAdapter)
            .cancellation(controller.handle())
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-cancel"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "do the long task".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let cancelled = tokio::join!(async { driver.next().await }, async {
            tokio::task::yield_now().await;
            controller.interrupt();
        })
        .0
        .unwrap();

        match cancelled {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Cancelled);
                assert_eq!(turn.items.len(), 1);
                assert_eq!(turn.items[0].kind, ItemKind::Assistant);
                assert_eq!(
                    turn.items[0].metadata.get(INTERRUPTED_METADATA_KEY),
                    Some(&Value::Bool(true))
                );
            }
            other => panic!("unexpected loop step: {other:?}"),
        }

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "try again".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();
        match result {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
            }
            other => panic!("unexpected loop step after retry: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_interrupt_cancels_foreground_tasks_but_keeps_background_tasks_running() {
        let controller = CancellationController::new();
        let fg_entered = StdArc::new(AtomicBool::new(false));
        let fg_release = StdArc::new(Notify::new());
        let bg_entered = StdArc::new(AtomicBool::new(false));
        let bg_release = StdArc::new(Notify::new());
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([
            ("foreground-wait", RoutingDecision::Foreground),
            ("background-wait", RoutingDecision::Background),
        ]));
        let handle = task_manager.handle();
        let tools = ToolRegistry::new()
            .with(BlockingTool::new(
                "foreground-wait",
                fg_entered.clone(),
                fg_release,
                "foreground-done",
            ))
            .with(BlockingTool::new(
                "background-wait",
                bg_entered.clone(),
                bg_release.clone(),
                "background-done",
            ));
        let agent = Agent::builder()
            .model(MultiToolAdapter)
            .tools(tools)
            .permissions(AllowAllPermissions)
            .cancellation(controller.handle())
            .task_manager(task_manager)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-mixed-cancel"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "run both".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let cancelled = tokio::join!(async { driver.next().await }, async {
            let _ = wait_for_task_event(&handle).await;
            let _ = wait_for_task_event(&handle).await;
            wait_until_entered(fg_entered.as_ref()).await;
            wait_until_entered(bg_entered.as_ref()).await;
            controller.interrupt();
        })
        .0
        .unwrap();

        match cancelled {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Cancelled),
            other => panic!("unexpected loop step after interrupt: {other:?}"),
        }

        match wait_for_task_event(&handle).await {
            TaskEvent::Cancelled(snapshot) => assert_eq!(snapshot.tool_name, "foreground-wait"),
            other => panic!("unexpected post-interrupt event: {other:?}"),
        }

        let running = handle.list_running().await;
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].tool_name, "background-wait");

        bg_release.notify_waiters();
        match wait_for_task_event(&handle).await {
            TaskEvent::Completed(snapshot, result) => {
                assert_eq!(snapshot.tool_name, "background-wait");
                assert_eq!(result.output, ToolOutput::Text("background-done".into()));
            }
            other => panic!("unexpected background completion event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_resumes_after_approved_tool_request() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(ApproveFsReads)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-approval"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let first = driver.next().await.unwrap();
        match first {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(request)) => {
                assert!(request.task_id.is_some());
                assert_eq!(request.id.0, "approval:fs-read");
            }
            other => panic!("unexpected loop step: {other:?}"),
        }

        driver.resolve_approval(ApprovalDecision::Approve).unwrap();
        let second = driver.next().await.unwrap();
        match second {
            LoopStep::Finished(turn) => match &turn.items[0].parts[0] {
                Part::Text(text) => assert_eq!(text.text, "tool said: pong"),
                other => panic!("unexpected part: {other:?}"),
            },
            other => panic!("unexpected loop step after approval: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_compacts_transcript_before_new_turns() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .compaction(CompactionConfig::new(
                CountTrigger,
                CompactionPipeline::new().with_strategy(KeepRecentStrategy::new(1)),
            ))
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-4"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        for text in ["first", "second"] {
            driver
                .submit_input(vec![Item {
                    id: None,
                    kind: ItemKind::User,
                    parts: vec![Part::Text(TextPart {
                        text: text.into(),
                        metadata: MetadataMap::new(),
                    })],
                    metadata: MetadataMap::new(),
                }])
                .unwrap();
            let _ = driver.next().await.unwrap();
        }

        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::CompactionFinished {
                replaced_items,
                ..
            } if *replaced_items > 0
        )));
    }

    #[tokio::test]
    async fn loop_refreshes_tool_specs_each_turn() {
        let seen_descriptions = StdArc::new(StdMutex::new(Vec::new()));
        let version = StdArc::new(AtomicUsize::new(1));
        let tools = ToolRegistry::new().with(DynamicSpecTool::new(version.clone()));
        let agent = Agent::builder()
            .model(RecordingAdapter {
                seen_descriptions: seen_descriptions.clone(),
            })
            .tools(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-dynamic-tools"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        for text in ["first", "second"] {
            driver
                .submit_input(vec![Item {
                    id: None,
                    kind: ItemKind::User,
                    parts: vec![Part::Text(TextPart {
                        text: text.into(),
                        metadata: MetadataMap::new(),
                    })],
                    metadata: MetadataMap::new(),
                }])
                .unwrap();

            let _ = driver.next().await.unwrap();
            if text == "first" {
                version.store(2, Ordering::SeqCst);
            }
        }

        let seen_descriptions = seen_descriptions.lock().unwrap();
        assert_eq!(seen_descriptions.len(), 2);
        assert_eq!(seen_descriptions[0], vec!["dynamic version 1".to_string()]);
        assert_eq!(seen_descriptions[1], vec!["dynamic version 2".to_string()]);
    }
}
