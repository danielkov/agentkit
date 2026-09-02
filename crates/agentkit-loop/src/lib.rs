//! Runtime-agnostic agent loop orchestration for sessions, turns, tools, and interrupts.
//!
//! `agentkit-loop` is the central coordination layer in the agentkit workspace.  It
//! drives a model through a multi-turn agentic loop, executing tool calls,
//! respecting permission checks, surfacing approval interrupts to the host
//! application, and optionally compacting the transcript when it grows too large.
//!
//! # Architecture
//!
//! The main entry point is [`Agent`], constructed via [`AgentBuilder`]. The
//! builder optionally accepts the prior conversation transcript via
//! [`AgentBuilder::transcript`] and the next user turn via
//! [`AgentBuilder::input`] — both default to empty. Calling
//! [`Agent::start`] with a [`SessionConfig`] returns a [`LoopDriver`] that
//! yields [`LoopStep`]s — either a finished turn or an interrupt that
//! requires host resolution before the loop can continue.
//!
//! If no input was preloaded, the first call to [`LoopDriver::next`] yields
//! [`LoopInterrupt::AwaitingInput`] and the host supplies the first user
//! turn via [`InputRequest::submit`]. If input was preloaded, the first
//! `next()` dispatches the model directly — convenient for one-shot calls.
//!
//! ```text
//! Agent::builder()
//!     .model(adapter)              // ModelAdapter implementation
//!     .add_tool_source(registry)   // ToolRegistry (or any ToolSource); call again to federate
//!     .permissions(checker)        // PermissionChecker for gating tool use
//!     .observer(obs)               // LoopObserver for streaming events
//!     .transcript(prior)           // optional: passive prior transcript (system prompt, resumed session)
//!     .input(first_user_turn)      // optional: preload next user turn so first next() drives a turn
//!     .build()?
//!     .start(config).await?  -> LoopDriver
//!         .next().await?     -> LoopStep::Finished | LoopStep::Interrupt(...)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use agentkit_core::{Item, ItemKind};
//! use agentkit_loop::{
//!     Agent, PromptCacheRequest, PromptCacheRetention, SessionConfig,
//! };
//!
//! # async fn example<M: agentkit_loop::ModelAdapter>(adapter: M) -> Result<(), agentkit_loop::LoopError> {
//! // One-shot: preload system prompt and first user message; first next()
//! // drives the model directly.
//! let agent = Agent::builder()
//!     .model(adapter)
//!     .transcript(vec![Item::text(ItemKind::System, "You are a helpful assistant.")])
//!     .input(vec![Item::text(ItemKind::User, "Hello!")])
//!     .build()?;
//!
//! let mut driver = agent
//!     .start(SessionConfig::new("demo").with_cache(
//!         PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
//!     ))
//!     .await?;
//!
//! let _ = driver.next().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use agentkit_core::{
    CancellationHandle, DataRef, Delta, FinishReason, Item, ItemKind, MetadataMap, Modality, Part,
    PartId, PartKind, ReasoningPart, SessionId, StructuredPart, TaskId, TextPart, Timestamp,
    ToolCallId, ToolCallPart, ToolOutput, ToolResultPart, TurnCancellation, Usage,
};
use agentkit_task_manager::{
    PendingLoopUpdates, SimpleTaskManager, TOOL_RESULT_NOT_STARTED_METADATA_KEY, TaskApproval,
    TaskLaunchKind, TaskLaunchRequest, TaskManager, TaskResolution, TaskStartContext,
    TaskStartOutcome, TurnTaskUpdate,
};
#[cfg(test)]
use agentkit_task_manager::{
    TOOL_RESULT_FAILURE_KIND_METADATA_KEY, TOOL_RESULT_FAILURE_KIND_PERMISSION_DENIED,
};
#[cfg(test)]
use agentkit_tools_core::ToolContext;
use agentkit_tools_core::{
    AllowAllPermissions, ApprovalDecision, ApprovalRequest, BasicToolExecutor, OwnedToolContext,
    PermissionChecker, ToolCatalogEvent, ToolError, ToolExecutionScope, ToolExecutor, ToolRequest,
    ToolResources, ToolSource, ToolSpec,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

const INTERRUPTED_METADATA_KEY: &str = "agentkit.interrupted";
const INTERRUPT_REASON_METADATA_KEY: &str = "agentkit.interrupt_reason";
const INTERRUPT_STAGE_METADATA_KEY: &str = "agentkit.interrupt_stage";
const USER_CANCELLED_REASON: &str = "user_cancelled";
const DETACHED_NOTIFICATION_TEXT_MAX_CHARS: usize = 512;
const DETACHED_TEXT_PREVIEW_MAX_CHARS: usize = 160;
const DETACHED_CALL_ID_MAX_CHARS: usize = 80;
const MAX_STREAMED_ASSISTANT_CONTENT_BYTES: usize = 1024 * 1024;
const MAX_STREAMED_ASSISTANT_CONTENT_PARTS: usize = 256;

/// Metadata key used by adapters to retain provider-native finish reasons.
pub const PROVIDER_FINISH_REASONS_METADATA_KEY: &str = "agentkit.provider_finish_reasons";

/// Adds provider-native finish reasons to model-turn metadata.
pub fn set_provider_finish_reasons<I, S>(metadata: &mut MetadataMap, reasons: I)
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut seen = HashSet::new();
    let reasons = reasons
        .into_iter()
        .map(Into::into)
        .filter(|reason: &String| !reason.is_empty() && seen.insert(reason.clone()))
        .map(Value::String)
        .collect::<Vec<_>>();
    metadata.remove(PROVIDER_FINISH_REASONS_METADATA_KEY);
    if !reasons.is_empty() {
        metadata.insert(
            PROVIDER_FINISH_REASONS_METADATA_KEY.into(),
            Value::Array(reasons),
        );
    }
}

fn provider_finish_reasons(metadata: &MetadataMap, fallback: &FinishReason) -> Vec<String> {
    metadata
        .get(PROVIDER_FINISH_REASONS_METADATA_KEY)
        .and_then(Value::as_array)
        .map(|values| {
            let mut seen = HashSet::new();
            values
                .iter()
                .filter_map(Value::as_str)
                .filter(|reason| !reason.is_empty() && seen.insert((*reason).to_owned()))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|reasons| !reasons.is_empty())
        .unwrap_or_else(|| vec![normalized_finish_reason(fallback).into()])
}

fn normalized_finish_reason(reason: &FinishReason) -> &str {
    match reason {
        FinishReason::Completed => "completed",
        FinishReason::ToolCall => "tool_call",
        FinishReason::MaxTokens => "max_tokens",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Blocked => "blocked",
        FinishReason::Error => "error",
        FinishReason::Other(reason) => reason,
    }
}

/// Invalid bounded-message capture configuration.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum MessageCaptureError {
    /// At least one message slot is required.
    #[error("message capture max_messages must be nonzero")]
    ZeroMessages,
    /// At least one source-content byte is required.
    #[error("message capture max_bytes must be nonzero")]
    ZeroBytes,
}

/// Bounded configuration for capturing structured messages on inference spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageCapture {
    max_messages: usize,
    max_bytes: usize,
}

impl MessageCapture {
    /// Creates validated limits without silently changing either value.
    pub fn new(max_messages: usize, max_bytes: usize) -> Result<Self, MessageCaptureError> {
        if max_messages == 0 {
            return Err(MessageCaptureError::ZeroMessages);
        }
        if max_bytes == 0 {
            return Err(MessageCaptureError::ZeroBytes);
        }
        Ok(Self {
            max_messages,
            max_bytes,
        })
    }

    /// Returns the maximum number of exported JSON message elements.
    pub fn max_messages(self) -> usize {
        self.max_messages
    }

    /// Returns the maximum source-content byte budget.
    pub fn max_bytes(self) -> usize {
        self.max_bytes
    }
}

/// Explicit, in-code configuration for inference telemetry.
///
/// Message capture is off by default. Input and output capture are independent,
/// bounded controls. AgentKit never reads
/// `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TelemetryConfig {
    input_messages: Option<MessageCapture>,
    output_messages: Option<MessageCapture>,
}

impl TelemetryConfig {
    /// Enables bounded input-message capture.
    pub fn with_input_messages(mut self, capture: MessageCapture) -> Self {
        self.input_messages = Some(capture);
        self
    }

    /// Enables bounded output-message capture.
    pub fn with_output_messages(mut self, capture: MessageCapture) -> Self {
        self.output_messages = Some(capture);
        self
    }

    /// Disables input-message capture.
    pub fn without_input_messages(mut self) -> Self {
        self.input_messages = None;
        self
    }

    /// Disables output-message capture.
    pub fn without_output_messages(mut self) -> Self {
        self.output_messages = None;
        self
    }

    /// Returns the input capture configuration, if enabled.
    pub fn input_messages(self) -> Option<MessageCapture> {
        self.input_messages
    }

    /// Returns the output capture configuration, if enabled.
    pub fn output_messages(self) -> Option<MessageCapture> {
        self.output_messages
    }
}

/// Capabilities supported by the consumer of model-turn events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConsumerCapabilities {
    /// The consumer can discard all events from a superseded response attempt.
    #[serde(default)]
    pub response_attempt_supersession: bool,
}

impl SessionConsumerCapabilities {
    /// Enables response-attempt supersession support.
    pub fn with_response_attempt_supersession(mut self) -> Self {
        self.response_attempt_supersession = true;
        self
    }
}

/// Configuration required to start a new model session.
///
/// Pass this to [`Agent::start`] to initialise the underlying [`ModelSession`]
/// and obtain a [`LoopDriver`].
///
/// # Example
///
/// ```rust
/// use agentkit_loop::{PromptCacheRequest, PromptCacheRetention, SessionConfig};
///
/// let config = SessionConfig::new("my-session").with_cache(
///     PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Unique identifier for the session.
    pub session_id: SessionId,
    /// Arbitrary key-value metadata forwarded to the model adapter.
    pub metadata: MetadataMap,
    /// Default provider-side prompt caching policy for turns in this session.
    pub cache: Option<PromptCacheRequest>,
    /// Features that the consumer of model-turn events can safely handle.
    #[serde(default)]
    pub consumer_capabilities: SessionConsumerCapabilities,
}

impl SessionConfig {
    /// Builds a session config with empty metadata and no cache policy.
    pub fn new(session_id: impl Into<SessionId>) -> Self {
        Self {
            session_id: session_id.into(),
            metadata: MetadataMap::new(),
            cache: None,
            consumer_capabilities: SessionConsumerCapabilities::default(),
        }
    }

    /// Replaces the session metadata map.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the default prompt cache request for the session.
    pub fn with_cache(mut self, cache: PromptCacheRequest) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Clears any default prompt cache request for the session.
    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    /// Declares that the event consumer can discard superseded response attempts.
    pub fn with_response_attempt_supersession(mut self) -> Self {
        self.consumer_capabilities = self
            .consumer_capabilities
            .with_response_attempt_supersession();
        self
    }
}

/// Strength of a prompt-cache request.
///
/// `BestEffort` lets adapters ignore unsupported controls while still using
/// any provider-native automatic caching they support. `Required` upgrades
/// unsupported cache requests into provider errors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptCacheMode {
    /// Disable prompt caching for this request.
    Disabled,
    /// Use caching when the provider can honor the request.
    #[default]
    BestEffort,
    /// Fail the turn if the provider cannot honor the request.
    Required,
}

/// High-level provider-neutral cache retention hint.
///
/// Providers map this to their native controls. For example, OpenAI maps
/// `Short` to in-memory retention while OpenRouter Anthropic models map it to
/// the default 5-minute ephemeral cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptCacheRetention {
    /// Use the provider's default cache retention.
    Default,
    /// Prefer the provider's short-lived cache retention mode.
    Short,
    /// Prefer the provider's longest generally available cache retention mode.
    Extended,
}

/// Provider-neutral prompt caching strategy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PromptCacheStrategy {
    /// Let the provider decide the cacheable prefix automatically.
    #[default]
    Automatic,
    /// Apply explicit cache breakpoints to selected prefix boundaries.
    Explicit {
        /// Cache breakpoints in transcript/tool order.
        breakpoints: Vec<PromptCacheBreakpoint>,
    },
}

impl PromptCacheStrategy {
    /// Uses the provider's native automatic cache behavior when available, or
    /// any adapter-provided automatic planning fallback.
    pub fn automatic() -> Self {
        Self::Automatic
    }

    /// Uses explicit cache breakpoints.
    pub fn explicit(breakpoints: impl IntoIterator<Item = PromptCacheBreakpoint>) -> Self {
        Self::Explicit {
            breakpoints: breakpoints.into_iter().collect(),
        }
    }
}

/// Prefix boundary that a provider should cache when using explicit caching.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptCacheBreakpoint {
    /// Cache the tool schema prefix through the last available tool.
    ToolsEnd,
    /// Cache through the end of the transcript item at `index`.
    TranscriptItemEnd { index: usize },
    /// Cache through the specific transcript part.
    ///
    /// Not every adapter can target every part precisely; unsupported
    /// fine-grained breakpoints become best-effort no-ops unless the request is
    /// marked [`PromptCacheMode::Required`].
    TranscriptPartEnd {
        item_index: usize,
        part_index: usize,
    },
}

impl PromptCacheBreakpoint {
    /// Cache the tool schema prefix through the last available tool.
    pub fn tools_end() -> Self {
        Self::ToolsEnd
    }

    /// Cache through the end of a transcript item.
    pub fn transcript_item_end(index: usize) -> Self {
        Self::TranscriptItemEnd { index }
    }

    /// Cache through a specific part within a transcript item.
    pub fn transcript_part_end(item_index: usize, part_index: usize) -> Self {
        Self::TranscriptPartEnd {
            item_index,
            part_index,
        }
    }
}

/// Prompt caching request sent alongside a turn.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheRequest {
    /// Strength of the caching request.
    pub mode: PromptCacheMode,
    /// Automatic or explicit caching strategy.
    pub strategy: PromptCacheStrategy,
    /// Optional provider-neutral retention hint.
    pub retention: Option<PromptCacheRetention>,
    /// Optional provider cache key or routing key.
    pub key: Option<String>,
}

impl PromptCacheRequest {
    /// Builds a best-effort automatic cache request.
    pub fn automatic() -> Self {
        Self::best_effort(PromptCacheStrategy::automatic())
    }

    /// Builds a required automatic cache request.
    pub fn automatic_required() -> Self {
        Self::required(PromptCacheStrategy::automatic())
    }

    /// Builds a best-effort explicit cache request.
    pub fn explicit(breakpoints: impl IntoIterator<Item = PromptCacheBreakpoint>) -> Self {
        Self::best_effort(PromptCacheStrategy::explicit(breakpoints))
    }

    /// Builds a required explicit cache request.
    pub fn explicit_required(breakpoints: impl IntoIterator<Item = PromptCacheBreakpoint>) -> Self {
        Self::required(PromptCacheStrategy::explicit(breakpoints))
    }

    /// Builds a disabled cache request.
    pub fn disabled() -> Self {
        Self {
            mode: PromptCacheMode::Disabled,
            strategy: PromptCacheStrategy::Automatic,
            retention: None,
            key: None,
        }
    }

    /// Builds a best-effort cache request with the given strategy.
    pub fn best_effort(strategy: PromptCacheStrategy) -> Self {
        Self {
            mode: PromptCacheMode::BestEffort,
            strategy,
            retention: None,
            key: None,
        }
    }

    /// Builds a required cache request with the given strategy.
    pub fn required(strategy: PromptCacheStrategy) -> Self {
        Self {
            mode: PromptCacheMode::Required,
            strategy,
            retention: None,
            key: None,
        }
    }

    /// Overrides the request mode.
    pub fn with_mode(mut self, mode: PromptCacheMode) -> Self {
        self.mode = mode;
        self
    }

    /// Overrides the request strategy.
    pub fn with_strategy(mut self, strategy: PromptCacheStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Applies a provider-neutral retention hint.
    pub fn with_retention(mut self, retention: PromptCacheRetention) -> Self {
        self.retention = Some(retention);
        self
    }

    /// Applies a provider cache key or routing key.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Clears any provider-neutral retention hint.
    pub fn without_retention(mut self) -> Self {
        self.retention = None;
        self
    }

    /// Clears any provider cache key or routing key.
    pub fn without_key(mut self) -> Self {
        self.key = None;
        self
    }

    /// Returns true when caching should be active for this request.
    pub fn is_enabled(&self) -> bool {
        !matches!(self.mode, PromptCacheMode::Disabled)
    }
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
    /// Provider-side prompt caching request for this turn.
    pub cache: Option<PromptCacheRequest>,
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
    /// Model identifier reported by the provider for this turn, if known.
    ///
    /// Stamped onto inference telemetry spans as `gen_ai.response.model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Provider-assigned response identifier for this turn, if known.
    ///
    /// Stamped onto inference telemetry spans as `gen_ai.response.id`.
    #[serde(default)]
    pub response_id: Option<String>,
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
    /// Supersedes every previously emitted event from the current response attempt.
    ///
    /// This marker is ordered after the failed attempt's deltas, tool calls, and usage and
    /// before replacement-attempt output. It is emitted only when the session consumer opted in.
    ResponseAttemptSuperseded,
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

    /// Name of the underlying model provider, when known.
    ///
    /// Stamped onto agent telemetry spans as the `gen_ai.provider.name`
    /// attribute from the OpenTelemetry GenAI semantic conventions. Use a
    /// lowercase identifier (e.g. `openrouter`, `ollama`). The default
    /// returns `None` for adapters without a meaningful provider identity.
    fn provider_name(&self) -> Option<&str> {
        None
    }
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

    /// Model identifier this session sends requests to, when known.
    ///
    /// Stamped onto inference telemetry spans as the `gen_ai.request.model`
    /// attribute from the OpenTelemetry GenAI semantic conventions. The
    /// default returns `None` for sessions without a fixed model.
    fn model_name(&self) -> Option<&str> {
        None
    }

    /// Concrete provider identity for this active session, when known.
    ///
    /// This value takes precedence over [`ModelAdapter::provider_name`]. The
    /// default preserves compatibility for existing session implementations.
    fn provider_name(&self) -> Option<&str> {
        None
    }
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
/// use agentkit_loop::{LoopObserver, ObservedEvent};
///
/// struct StdoutObserver;
///
/// impl LoopObserver for StdoutObserver {
///     fn handle_event(&self, event: ObservedEvent) {
///         println!("{:?}", event.event);
///     }
/// }
/// ```
pub trait LoopObserver: Send + Sync {
    /// Called synchronously for every [`AgentEvent`] emitted by the loop driver.
    /// Observers store mutable state behind interior mutability (`Mutex`,
    /// atomics, channels) so the driver can share an `Arc<dyn LoopObserver>`
    /// across reusable [`Agent`] starts.
    fn handle_event(&self, event: ObservedEvent);
}

/// Session-addressed [`AgentEvent`] envelope delivered to [`LoopObserver`]s.
///
/// Some event variants carry their own session fields, but many high-volume
/// events intentionally stay compact. The envelope gives shared observers a
/// consistent routing key without reshaping every [`AgentEvent`] variant.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservedEvent {
    /// Session this event belongs to.
    pub session_id: Arc<SessionId>,
    /// The operational event emitted by the driver.
    pub event: AgentEvent,
}

/// Receives full [`Item`]s as they are appended to the driver's transcript.
///
/// While [`LoopObserver`] surfaces operational events (deltas, tool calls,
/// lifecycle, telemetry), it can't be reconstructed back into a faithful
/// transcript on its own — content deltas span partial parts and don't
/// carry their parent-Item identity, and historically tool results were
/// pushed into the transcript with no observer event at all. A
/// `TranscriptObserver` is the loss-free counterpart: it fires once per
/// [`Item`] appended, with the full Item shape ready for persistence,
/// replication, or audit.
///
/// Observers are called *synchronously* from inside the driver, in the
/// same order items land in the transcript. Compaction-driven transcript
/// rewrites do **not** fire `on_transcript_event` — those are signaled by
/// [`AgentEvent::CompactionFinished`] instead.
///
/// Register via [`AgentBuilder::transcript_observer`]; multiple observers
/// may be registered and are called in registration order.
///
/// # Example
///
/// ```rust
/// use agentkit_core::Item;
/// use agentkit_loop::{TranscriptEvent, TranscriptObserver};
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// struct CountingObserver { items: AtomicUsize }
///
/// impl TranscriptObserver for CountingObserver {
///     fn on_transcript_event(&self, _event: TranscriptEvent<'_>) {
///         self.items.fetch_add(1, Ordering::Relaxed);
///     }
/// }
/// ```
pub trait TranscriptObserver: Send + Sync {
    /// Called synchronously every time an [`Item`] is appended to the
    /// driver's transcript, in transcript order. Observers store mutable
    /// state behind interior mutability so the driver can share an
    /// `Arc<dyn TranscriptObserver>`.
    fn on_transcript_event(&self, event: TranscriptEvent<'_>);
}

/// Session-addressed transcript append event delivered to
/// [`TranscriptObserver`]s.
#[derive(Clone, Debug)]
pub struct TranscriptEvent<'a> {
    /// Session this transcript append belongs to.
    pub session_id: &'a SessionId,
    /// Full item that was appended.
    pub item: &'a Item,
}

/// Where in the loop a [`LoopMutator`] is given a chance to modify the
/// transcript. Mutators run synchronously at these points; mid-stream
/// mutation (e.g. between content deltas) is intentionally not supported
/// because the assistant item is not yet fully constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MutationPoint {
    /// A tool result has just been appended; the next loop step will be
    /// another inference call.
    AfterToolResult,
    /// A turn has fully ended (assistant final, interrupt, or cancellation)
    /// and any new user input has not yet been dispatched.
    AfterTurnEnded,
}

/// Sink for emitting [`AgentEvent`]s from inside a [`LoopMutator`].
/// The driver supplies a concrete implementation via [`LoopCtx::emitter`].
pub trait EventEmitter: Send + Sync {
    /// Forward `event` to all registered observers.
    fn emit(&self, event: AgentEvent);
}

/// Read-only context handed to a [`LoopMutator`] alongside the cursor.
#[non_exhaustive]
pub struct LoopCtx<'a> {
    /// Session this mutation point belongs to.
    pub session_id: &'a SessionId,
    /// Turn the mutation is associated with, if any.
    pub turn_id: Option<&'a agentkit_core::TurnId>,
    /// Where in the loop the mutator is running.
    pub point: MutationPoint,
    /// Cancellation handle for the active turn, if any.
    pub cancellation: Option<TurnCancellation>,
    /// Sink for emitting events from the mutator (telemetry, progress).
    pub emitter: &'a dyn EventEmitter,
}

/// Mutable handle over the live transcript with dirty tracking.
///
/// Implements [`Deref`](std::ops::Deref)/[`DerefMut`](std::ops::DerefMut) to
/// `Vec<Item>` so mutators read and write through `Vec`'s native API
/// (`push`, `retain`, `iter`, `*cursor = ...`). Any `&mut` access marks the
/// cursor dirty; the loop validates transcript invariants when at least one
/// mutator dirtied the transcript and hard-fails on protocol violations.
pub struct TranscriptCursor<'a> {
    items: &'a mut Vec<Item>,
    pub(crate) dirty: bool,
}

impl<'a> std::ops::Deref for TranscriptCursor<'a> {
    type Target = Vec<Item>;
    fn deref(&self) -> &Vec<Item> {
        self.items
    }
}

impl<'a> std::ops::DerefMut for TranscriptCursor<'a> {
    fn deref_mut(&mut self) -> &mut Vec<Item> {
        self.dirty = true;
        self.items
    }
}

/// Async transcript mutator. Registered via [`AgentBuilder::mutator`] and
/// invoked at each [`MutationPoint`]. Mutators own their derived state
/// (e.g. running token totals via interior mutability) and decide for
/// themselves whether and how to modify the transcript.
///
/// The default implementation is a no-op so trait users override only
/// `mutate`.
#[async_trait]
pub trait LoopMutator: Send + Sync {
    /// Run this mutator. Returning without writing to `cursor` is a no-op.
    /// Errors abort the loop; protocol-violating mutations (orphaned tool
    /// uses or results) are detected by validation and turned into
    /// [`LoopError::Mutator`].
    async fn mutate(
        &self,
        cursor: &mut TranscriptCursor<'_>,
        ctx: LoopCtx<'_>,
    ) -> Result<(), LoopError> {
        let _ = (cursor, ctx);
        Ok(())
    }
}

/// Lifecycle and streaming events emitted by the [`LoopDriver`].
///
/// Observers (see [`LoopObserver`]) receive these events in the order they
/// occur.  They are useful for building UIs, logging, or telemetry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AgentEvent {
    /// The agent run has been initialised.
    RunStarted { session_id: SessionId },
    /// A new logical turn is starting.
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
    /// A tool call is about to execute after policy and approval checks.
    ToolExecutionStarted(ToolCallPart),
    /// A tool call has non-terminal progress to report.
    ///
    /// Used for updates such as background detachment. Unlike
    /// [`AgentEvent::ToolResultReceived`], this does not mean the call has
    /// reached a terminal result.
    ToolExecutionProgress(ToolResultPart),
    /// A tool call's result has landed in the transcript.
    ///
    /// Fires once per terminal [`Part::ToolResult`] that's appended.
    /// Cancellation/denial paths (auth cancelled, approval denied) also emit
    /// this with `is_error = true`.
    ///
    /// Correlate with the matching [`AgentEvent::ToolCallRequested`] via
    /// `call_id`.
    ToolResultReceived(ToolResultPart),
    /// A tool call requires explicit user approval before execution.
    ApprovalRequired(ApprovalRequest),
    /// An approval interrupt has been resolved.
    ApprovalResolved { approved: bool },
    /// The available tool catalog changed and will be reflected on the next model request.
    ToolCatalogChanged(ToolCatalogEvent),
    /// A [`LoopMutator`] is about to run at one of the mutation points.
    /// `mutator` is a stable label the implementation chooses for itself.
    MutationStarted {
        session_id: SessionId,
        turn_id: Option<agentkit_core::TurnId>,
        mutator: String,
        point: MutationPoint,
    },
    /// A [`LoopMutator`] has finished running. `dirty` indicates whether the
    /// transcript was modified; `metadata` carries mutator-specific extras
    /// (e.g. compaction reason, replaced item count).
    MutationFinished {
        session_id: SessionId,
        turn_id: Option<agentkit_core::TurnId>,
        mutator: String,
        dirty: bool,
        metadata: MetadataMap,
    },
    /// Updated token usage statistics.
    UsageUpdated(Usage),
    /// All events from the preceding model response attempt are superseded.
    ///
    /// Consumers that opted in must discard that attempt's deltas, tool calls, usage updates,
    /// and reconstruction state before handling replacement output.
    ResponseAttemptSuperseded,
    /// Non-fatal warning (e.g. a tool failure that was recovered from).
    Warning { message: String },
    /// The agent run has failed with an unrecoverable error.
    RunFailed { message: String },
    /// A logical turn has finished (successfully, via cancellation, etc.).
    TurnFinished(TurnResult),
}

/// Handle for a pending approval interrupt.
///
/// Wraps an [`ApprovalRequest`] and provides ergonomic resolution methods
/// so callers can resolve the interrupt directly instead of searching for
/// the matching method on [`LoopDriver`].
///
/// # Example
///
/// ```rust,no_run
/// # use agentkit_loop::{LoopInterrupt, LoopStep, LoopDriver};
/// # async fn handle<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>) -> Result<(), agentkit_loop::LoopError> {
/// match driver.next().await? {
///     LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
///         println!("Needs approval: {}", pending.request.summary);
///         pending.approve(driver)?;
///     }
///     _ => {}
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingApproval {
    /// The underlying approval request details.
    pub request: ApprovalRequest,
}

impl std::ops::Deref for PendingApproval {
    type Target = ApprovalRequest;
    fn deref(&self) -> &ApprovalRequest {
        &self.request
    }
}

impl PendingApproval {
    /// Approve the pending tool call.
    pub fn approve<S: ModelSession>(self, driver: &mut LoopDriver<S>) -> Result<(), LoopError> {
        let call_id = self
            .request
            .call_id
            .ok_or_else(|| LoopError::InvalidState("pending approval is missing call id".into()))?;
        driver.resolve_approval_for(call_id, ApprovalDecision::Approve)
    }

    /// Deny the pending tool call.
    pub fn deny<S: ModelSession>(self, driver: &mut LoopDriver<S>) -> Result<(), LoopError> {
        let call_id = self
            .request
            .call_id
            .ok_or_else(|| LoopError::InvalidState("pending approval is missing call id".into()))?;
        driver.resolve_approval_for(call_id, ApprovalDecision::Deny { reason: None })
    }

    /// Deny the pending tool call with a reason.
    pub fn deny_with_reason<S: ModelSession>(
        self,
        driver: &mut LoopDriver<S>,
        reason: impl Into<String>,
    ) -> Result<(), LoopError> {
        let call_id = self
            .request
            .call_id
            .ok_or_else(|| LoopError::InvalidState("pending approval is missing call id".into()))?;
        driver.resolve_approval_for(
            call_id,
            ApprovalDecision::Deny {
                reason: Some(reason.into()),
            },
        )
    }

    /// Approve the pending tool call with a patched input.
    ///
    /// The model's original tool input is replaced with `input` before the
    /// tool executes. The transcript still records the call as the model
    /// emitted it; only the executor sees the patched payload. This mirrors
    /// the `PermissionResultAllow(updated_input=...)` pattern from the
    /// Anthropic Agent SDK and is intended for hosts that want to sanitise,
    /// restrict, or augment arguments before tool execution without forcing
    /// the model to re-issue the call.
    pub fn approve_with_patched_input<S: ModelSession>(
        self,
        driver: &mut LoopDriver<S>,
        input: serde_json::Value,
    ) -> Result<(), LoopError> {
        let call_id = self
            .request
            .call_id
            .ok_or_else(|| LoopError::InvalidState("pending approval is missing call id".into()))?;
        driver.resolve_approval_for_with_patched_input(call_id, input)
    }
}

/// Descriptor for a [`LoopInterrupt::AwaitingInput`] interrupt.
///
/// Returned when the driver has no pending input and needs the host to
/// supply items before advancing. This is the entry point for every user
/// turn that wasn't preloaded via [`AgentBuilder::input`]. Transcript items
/// loaded via [`AgentBuilder::transcript`] are passive, so when no input is
/// preloaded the first [`LoopDriver::next`] call surfaces `AwaitingInput`
/// and the host injects the first user message via [`InputRequest::submit`].
///
/// # Example
///
/// ```rust,no_run
/// # use agentkit_loop::{LoopInterrupt, LoopStep, LoopDriver};
/// # use agentkit_core::Item;
/// # async fn handle<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>, items: Vec<Item>) -> Result<(), agentkit_loop::LoopError> {
/// match driver.next().await? {
///     LoopStep::Interrupt(LoopInterrupt::AwaitingInput(pending)) => {
///         pending.submit(driver, items)?;
///     }
///     _ => {}
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRequest {
    /// The session that is waiting for input.
    pub session_id: SessionId,
    /// Human-readable explanation of why input is needed.
    pub reason: String,
}

impl InputRequest {
    /// Submit input items to the driver.
    pub fn submit<S: ModelSession>(
        self,
        driver: &mut LoopDriver<S>,
        items: Vec<Item>,
    ) -> Result<(), LoopError> {
        driver.submit_input(items)
    }
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
/// cannot proceed autonomously.  Each variant carries a handle with
/// resolution methods so callers can resolve the interrupt directly.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_loop::{LoopInterrupt, LoopStep};
/// # use agentkit_loop::LoopDriver;
///
/// # async fn handle<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>) -> Result<(), agentkit_loop::LoopError> {
/// match driver.next().await? {
///     LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
///         println!("Tool {} needs approval: {}", pending.request.request_kind, pending.request.summary);
///         pending.approve(driver)?;
///     }
///     LoopStep::Interrupt(LoopInterrupt::AwaitingInput(pending)) => {
///         println!("Waiting for input: {}", pending.reason);
///         // ... call pending.submit(driver, items)
///     }
///     LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => {
///         // Cooperative yield between tool rounds.  Optionally call
///         // driver.submit_input(...) to interject a user message, then
///         // call driver.next() to resume the turn.
///         let _ = info;
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
    ApprovalRequest(PendingApproval),
    /// The driver has no pending input and needs the host to supply some.
    AwaitingInput(InputRequest),
    /// A tool round finished: all tool calls from the previous assistant
    /// message now have results in the transcript, and the driver is about to
    /// invoke the model again. The host may interject user messages via the
    /// [`ToolRoundInfo::submit`] handle before calling [`LoopDriver::next`]
    /// to resume.
    ///
    /// This is a non-blocking interrupt: callers that do not care about
    /// mid-turn interjection can treat it as a no-op (`_ => continue`) and
    /// the next `next()` call resumes the turn.
    AfterToolResult(ToolRoundInfo),
}

impl LoopInterrupt {
    /// Returns `true` if the interrupt must be explicitly resolved before
    /// the loop can make progress. Approvals are blocking;
    /// [`AwaitingInput`](LoopInterrupt::AwaitingInput) and
    /// [`AfterToolResult`](LoopInterrupt::AfterToolResult) are cooperative
    /// and can be ignored by calling [`LoopDriver::next`] again.
    pub fn is_blocking(&self) -> bool {
        matches!(self, LoopInterrupt::ApprovalRequest(_))
    }
}

/// Metadata describing a completed tool round, surfaced via
/// [`LoopInterrupt::AfterToolResult`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRoundInfo {
    /// The session that produced this tool round.
    pub session_id: SessionId,
    /// The turn that is about to continue into the next model call.
    pub turn_id: agentkit_core::TurnId,
    /// Transcript length at the yield point (for snapshots / UIs).
    pub transcript_len: usize,
}

impl ToolRoundInfo {
    /// Interject user input between tool rounds. Consumes the
    /// [`ToolRoundInfo`] handle so the same yield cannot accept input twice.
    pub fn submit<S: ModelSession>(
        self,
        driver: &mut LoopDriver<S>,
        items: Vec<Item>,
    ) -> Result<(), LoopError> {
        driver.submit_input(items)
    }
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

#[derive(Clone)]
struct PendingApprovalToolCall {
    request: ApprovalRequest,
    decision: Option<ApprovalDecision>,
    surfaced: bool,
    presentation_turn_id: agentkit_core::TurnId,
    task_id: TaskId,
    call: ToolCallPart,
    tool_request: ToolRequest,
    cancellation: Option<TurnCancellation>,
}

#[derive(Clone, Default)]
struct ActiveToolRound {
    presentation_turn_id: agentkit_core::TurnId,
    task_turn_id: agentkit_core::TurnId,
    pending_calls: VecDeque<(ToolCallPart, ToolRequest)>,
    cancellation: Option<TurnCancellation>,
    background_pending: bool,
    foreground_progressed: bool,
}

#[derive(Default)]
struct DriverLifecycle {
    active_turn: Option<agentkit_core::TurnId>,
}

/// A configured agent ready to start a session.
///
/// Build one with [`Agent::builder`], supplying at minimum a [`ModelAdapter`].
/// Optionally preload prior conversation state via
/// [`AgentBuilder::transcript`] and the next user turn via
/// [`AgentBuilder::input`]. Then call [`Agent::start`] with a
/// [`SessionConfig`] to obtain a [`LoopDriver`] that drives the agentic loop.
///
/// If no input is preloaded, the first call to [`LoopDriver::next`] yields
/// [`LoopInterrupt::AwaitingInput`] so the host can supply the first user
/// message via [`InputRequest::submit`]. If input was preloaded, the first
/// `next()` dispatches the model directly.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_core::{Item, ItemKind};
/// use agentkit_loop::{
///     Agent, PromptCacheRequest, PromptCacheRetention, SessionConfig,
/// };
/// use agentkit_tools_core::ToolRegistry;
///
/// # async fn example<M: agentkit_loop::ModelAdapter>(adapter: M) -> Result<(), agentkit_loop::LoopError> {
/// let agent = Agent::builder()
///     .model(adapter)
///     .add_tool_source(ToolRegistry::new())
///     .transcript(vec![Item::text(ItemKind::System, "You are a helpful assistant.")])
///     .input(vec![Item::text(ItemKind::User, "Hello!")])
///     .build()?;
///
/// let mut driver = agent
///     .start(SessionConfig::new("s1").with_cache(
///         PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
///     ))
///     .await?;
///
/// // First next() drives the model since input was preloaded.
/// let _ = driver.next().await?;
/// # Ok(())
/// # }
/// ```
pub struct Agent<M>
where
    M: ModelAdapter,
{
    model: M,
    tool_sources: Vec<Arc<dyn ToolSource>>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    task_manager: Arc<dyn TaskManager>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    mutators: Vec<Arc<dyn LoopMutator>>,
    observers: Vec<Arc<dyn LoopObserver>>,
    transcript_observers: Vec<Arc<dyn TranscriptObserver>>,
    transcript: Vec<Item>,
    input: Vec<Item>,
    telemetry: TelemetryConfig,
}

impl<M> Agent<M>
where
    M: ModelAdapter,
{
    /// Create a new [`AgentBuilder`] for configuring this agent.
    pub fn builder() -> AgentBuilder<M> {
        AgentBuilder::default()
    }

    /// Start a session, returning a [`LoopDriver`] preloaded with whatever
    /// transcript and input were configured on the builder. See
    /// [`AgentBuilder::transcript`] and [`AgentBuilder::input`] for what each
    /// one does and when to use them.
    ///
    /// This calls [`ModelAdapter::start_session`] and emits an
    /// [`AgentEvent::RunStarted`] event to all registered observers.
    ///
    /// `&self` so a single configured agent can mint multiple sessions over
    /// its lifetime — e.g. an outer agent that uses an inner sub-agent for
    /// transcript compaction.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError`] if the model adapter fails to create a session.
    pub async fn start(&self, config: SessionConfig) -> Result<LoopDriver<M::Session>, LoopError> {
        let session_id = config.session_id.clone();
        let default_cache = config.cache.clone();
        let session = self.model.start_session(config).await?;
        let provider_name = self.model.provider_name().map(str::to_owned);
        let tool_executor = self
            .tool_executor
            .clone()
            .unwrap_or_else(|| Arc::new(BasicToolExecutor::new(self.tool_sources.clone())));
        let driver = LoopDriver {
            session_id: session_id.clone(),
            observed_session_id: Arc::new(session_id.clone()),
            provider_name,
            telemetry: self.telemetry,
            default_cache,
            next_turn_cache: None,
            session: Some(session),
            tool_executor,
            task_manager: self.task_manager.clone(),
            permissions: self.permissions.clone(),
            resources: self.resources.clone(),
            cancellation: self.cancellation.clone(),
            mutators: self.mutators.clone(),
            observers: self.observers.clone(),
            transcript_observers: self.transcript_observers.clone(),
            transcript: self.transcript.clone(),
            pending_input: self.input.clone(),
            pending_approvals: BTreeMap::new(),
            pending_approval_order: VecDeque::new(),
            active_tool_round: None,
            pending_round_resume: None,
            pending_loop_updates: VecDeque::new(),
            next_turn_index: 1,
            lifecycle: DriverLifecycle::default(),
            background_call_ids: HashSet::new(),
            detached_call_ids: HashSet::new(),
            interrupted_background_call_ids: HashSet::new(),
            tool_cancellations: HashMap::new(),
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
    tool_sources: Vec<Arc<dyn ToolSource>>,
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    task_manager: Option<Arc<dyn TaskManager>>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    mutators: Vec<Arc<dyn LoopMutator>>,
    observers: Vec<Arc<dyn LoopObserver>>,
    transcript_observers: Vec<Arc<dyn TranscriptObserver>>,
    transcript: Vec<Item>,
    input: Vec<Item>,
    telemetry: TelemetryConfig,
}

impl<M> Default for AgentBuilder<M>
where
    M: ModelAdapter,
{
    fn default() -> Self {
        Self {
            model: None,
            tool_sources: Vec::new(),
            tool_executor: None,
            task_manager: None,
            permissions: Arc::new(AllowAllPermissions),
            resources: Arc::new(()),
            cancellation: None,
            mutators: Vec::new(),
            observers: Vec::new(),
            transcript_observers: Vec::new(),
            transcript: Vec::new(),
            input: Vec::new(),
            telemetry: TelemetryConfig::default(),
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

    /// Adds a tool source to the agent. Call multiple times to compose
    /// federated sources — for example a frozen native [`ToolRegistry`]
    /// alongside an MCP manager's [`agentkit_tools_core::CatalogReader`]
    /// and a skill-watcher reader. Sources are walked in registration
    /// order; the default [`agentkit_tools_core::CollisionPolicy`] is
    /// `FirstWins`.
    ///
    /// Accepts any sized [`ToolSource`]; the agent owns it for the
    /// session. To share a dynamic source between the agent and the
    /// subsystem mutating it, mint a [`agentkit_tools_core::CatalogReader`]
    /// from a [`agentkit_tools_core::dynamic_catalog`] pair — the reader
    /// is sized and owned, hosts never see the underlying `Arc`.
    pub fn add_tool_source<S: ToolSource + 'static>(mut self, source: S) -> Self {
        self.tool_sources.push(Arc::new(source));
        self
    }

    /// Set a custom [`ToolExecutor`]. When provided, the agent uses it
    /// instead of building a [`BasicToolExecutor`] from the configured
    /// sources. Most hosts should use [`add_tool_source`](Self::add_tool_source)
    /// instead; this is for advanced cases (custom routing, instrumentation,
    /// test fakes).
    pub fn tool_executor(mut self, executor: impl ToolExecutor + 'static) -> Self {
        self.tool_executor = Some(Arc::new(executor));
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

    /// Register a [`LoopMutator`] that runs at every [`MutationPoint`].
    ///
    /// Multiple mutators may be registered; they run in registration order
    /// and the dirty flag propagates across the pipeline. After every pass
    /// in which any mutator dirtied the transcript, the loop validates
    /// protocol invariants (tool_use/tool_result pairing); a violation is a
    /// hard [`LoopError::Mutator`] failure.
    pub fn mutator<L: LoopMutator + 'static>(mut self, mutator: L) -> Self {
        self.mutators.push(Arc::new(mutator));
        self
    }

    /// Register a [`LoopObserver`] that receives [`AgentEvent`]s.
    ///
    /// Multiple observers may be registered; they are called in order.
    pub fn observer<O: LoopObserver + 'static>(mut self, observer: O) -> Self {
        self.observers.push(Arc::new(observer));
        self
    }

    /// Register a [`TranscriptObserver`] that receives an [`Item`] every
    /// time one is appended to the transcript.
    ///
    /// Multiple observers may be registered; they are called in order.
    /// Use this when you need a loss-free view of the transcript (e.g.
    /// for persistence or replication) — [`LoopObserver`] alone is
    /// insufficient because it doesn't expose item boundaries for model
    /// output and historically did not surface tool results at all.
    pub fn transcript_observer<O: TranscriptObserver + 'static>(mut self, observer: O) -> Self {
        self.transcript_observers.push(Arc::new(observer));
        self
    }

    /// Preload the driver's transcript with prior conversation state
    /// (defaults to empty).
    ///
    /// Items pass straight into the driver's transcript without firing
    /// [`TranscriptObserver::on_transcript_event`] — the host is expected to
    /// already know about (and have persisted) anything it preloads. Use
    /// this for resumed sessions or to seed a system prompt.
    pub fn transcript(mut self, transcript: Vec<Item>) -> Self {
        self.transcript = transcript;
        self
    }

    /// Preload the driver's pending-input queue with the next user turn
    /// (defaults to empty).
    ///
    /// When non-empty, the first [`LoopDriver::next`] dispatches the model
    /// directly instead of yielding [`LoopInterrupt::AwaitingInput`]. Use
    /// this for one-shot calls and scripts where the first user turn is
    /// known up front. Items move to the transcript on turn dispatch the
    /// same way submitted input does, firing transcript observers.
    pub fn input(mut self, input: Vec<Item>) -> Self {
        self.input = input;
        self
    }

    /// Configures inference telemetry. Message capture remains off unless
    /// enabled explicitly here.
    pub fn telemetry(mut self, telemetry: TelemetryConfig) -> Self {
        self.telemetry = telemetry;
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
            tool_sources: self.tool_sources,
            tool_executor: self.tool_executor,
            task_manager: self
                .task_manager
                .unwrap_or_else(|| Arc::new(SimpleTaskManager::new())),
            permissions: self.permissions,
            resources: self.resources,
            cancellation: self.cancellation,
            mutators: self.mutators,
            observers: self.observers,
            transcript_observers: self.transcript_observers,
            transcript: self.transcript,
            input: self.input,
            telemetry: self.telemetry,
        })
    }
}

/// The runtime driver that advances the agent loop step by step.
///
/// Obtained from [`Agent::start`] with the builder's preloaded transcript
/// and pending-input queue baked in.
/// The typical usage pattern is:
///
/// 1. Call [`next`](LoopDriver::next) to advance the loop.
/// 2. Handle the returned [`LoopStep`]:
///    - [`LoopStep::Finished`] -- the turn completed, inspect the result.
///    - [`LoopStep::Interrupt`] -- resolve the interrupt via the bound
///      [`Pending*`](LoopInterrupt) handle, then call `next` again.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_core::{Item, ItemKind};
/// use agentkit_loop::{LoopDriver, LoopStep};
///
/// # async fn drive<S: agentkit_loop::ModelSession>(driver: &mut LoopDriver<S>) -> Result<(), agentkit_loop::LoopError> {
/// let step = driver.next().await?;
/// match step {
///     LoopStep::Finished(result) => println!("Done: {:?}", result.finish_reason),
///     LoopStep::Interrupt(interrupt) => {
///         // Resolve via the pending handle, then call next() again.
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
    observed_session_id: Arc<SessionId>,
    provider_name: Option<String>,
    telemetry: TelemetryConfig,
    default_cache: Option<PromptCacheRequest>,
    next_turn_cache: Option<PromptCacheRequest>,
    session: Option<S>,
    tool_executor: Arc<dyn ToolExecutor>,
    task_manager: Arc<dyn TaskManager>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    mutators: Vec<Arc<dyn LoopMutator>>,
    observers: Vec<Arc<dyn LoopObserver>>,
    transcript_observers: Vec<Arc<dyn TranscriptObserver>>,
    transcript: Vec<Item>,
    pending_input: Vec<Item>,
    pending_approvals: BTreeMap<ToolCallId, PendingApprovalToolCall>,
    pending_approval_order: VecDeque<ToolCallId>,
    active_tool_round: Option<ActiveToolRound>,
    pending_round_resume: Option<agentkit_core::TurnId>,
    pending_loop_updates: VecDeque<TaskResolution>,
    next_turn_index: u64,
    lifecycle: DriverLifecycle,
    /// Calls currently running in the background without a transcript result.
    background_call_ids: HashSet<ToolCallId>,
    /// Call ids whose original tool_use was already paired with a
    /// synthetic detach tool_result. When the real result eventually
    /// arrives via the task manager, we MUST NOT emit a second
    /// tool_result for the same id — the provider schema requires
    /// exactly one tool_result per tool_use. Instead we route the
    /// resolution into a [`ItemKind::Notification`] item that the model
    /// can react to on the next turn.
    detached_call_ids: HashSet<ToolCallId>,
    /// Background calls whose cancellation result was already terminal.
    /// Their eventual completion becomes a notification without emitting a
    /// second [`AgentEvent::ToolResultReceived`].
    interrupted_background_call_ids: HashSet<ToolCallId>,
    tool_cancellations: HashMap<ToolCallId, TurnCancellation>,
}

impl<S> LoopDriver<S>
where
    S: ModelSession,
{
    fn execute_tool_span(
        &self,
        request: &ToolRequest,
        turn_id: &agentkit_core::TurnId,
        launch_kind: &'static str,
    ) -> tracing::Span {
        tracing::info_span!(
            "agent.execute_tool",
            "otel.name" = %format!("execute_tool {}", request.tool_name),
            "gen_ai.operation.name" = "execute_tool",
            "gen_ai.tool.name" = %request.tool_name,
            "gen_ai.tool.call.id" = %request.call_id,
            "gen_ai.conversation.id" = %self.session_id,
            "error.type" = tracing::field::Empty,
            session.id = %self.session_id,
            turn.id = %turn_id,
            launch_kind = launch_kind,
        )
    }

    fn start_task_via_manager(
        &self,
        task_id: Option<TaskId>,
        tool_request: ToolRequest,
        kind: TaskLaunchKind,
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
                        kind,
                    },
                    TaskStartContext {
                        executor: tool_executor.clone(),
                        tool_context: {
                            let execution_scope = ToolExecutionScope {
                                executor: tool_executor,
                                session_id: session_id.clone(),
                                turn_id: turn_id.clone(),
                                permissions: permissions.clone(),
                                resources: resources.clone(),
                                cancellation: cancellation.clone(),
                            };
                            OwnedToolContext {
                                session_id,
                                turn_id,
                                metadata,
                                permissions,
                                resources,
                                cancellation,
                                execution_scope: Some(execution_scope),
                                approved_request: None,
                            }
                        },
                    },
                )
                .await
                .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))
        }
    }

    fn register_tool_cancellation(
        &mut self,
        call_id: &ToolCallId,
        cancellation: Option<TurnCancellation>,
    ) {
        if let Some(cancellation) = cancellation {
            self.tool_cancellations
                .insert(call_id.clone(), cancellation);
        }
    }

    fn tool_cancellation_for(
        &mut self,
        call_id: &ToolCallId,
        fallback: Option<TurnCancellation>,
    ) -> Option<TurnCancellation> {
        self.tool_cancellations.get(call_id).cloned().or(fallback)
    }

    fn clear_tool_cancellation(&mut self, call_id: &ToolCallId) {
        self.tool_cancellations.remove(call_id);
    }

    fn has_pending_interrupts(&self) -> bool {
        !self.pending_approvals.is_empty()
    }

    fn start_logical_turn(&mut self) -> agentkit_core::TurnId {
        if let Some(turn_id) = &self.lifecycle.active_turn {
            return turn_id.clone();
        }
        let turn_id = agentkit_core::TurnId::new(format!("turn-{}", self.next_turn_index));
        self.next_turn_index += 1;
        self.start_logical_turn_with(turn_id)
    }

    fn start_logical_turn_with(&mut self, turn_id: agentkit_core::TurnId) -> agentkit_core::TurnId {
        if let Some(active_turn) = &self.lifecycle.active_turn {
            return active_turn.clone();
        }
        self.lifecycle.active_turn = Some(turn_id.clone());
        self.emit(AgentEvent::TurnStarted {
            session_id: self.session_id.clone(),
            turn_id: turn_id.clone(),
        });
        turn_id
    }

    fn finish_logical_turn(&mut self, result: &TurnResult) {
        if self.pending_round_resume.as_ref() == Some(&result.turn_id) {
            self.pending_round_resume = None;
        }
        if self.lifecycle.active_turn.as_ref() == Some(&result.turn_id) {
            self.lifecycle.active_turn = None;
            self.emit(AgentEvent::TurnFinished(result.clone()));
        }
    }

    fn emit_tool_catalog_events(&mut self, events: Vec<ToolCatalogEvent>) {
        for event in events {
            self.emit(AgentEvent::ToolCatalogChanged(event));
        }
    }

    fn enqueue_pending_approval(
        &mut self,
        presentation_turn_id: &agentkit_core::TurnId,
        task: TaskApproval,
        cancellation: Option<TurnCancellation>,
    ) {
        let call_id = task.tool_request.call_id.clone();
        self.background_call_ids.remove(&call_id);
        let cancellation = self.tool_cancellation_for(&call_id, cancellation);
        let call = ToolCallPart {
            id: call_id.clone(),
            name: task.tool_request.tool_name.to_string(),
            input: task.tool_request.input.clone(),
            metadata: task.tool_request.metadata.clone(),
        };
        let mut request = task.approval;
        request.call_id = Some(call_id.clone());
        let pending = PendingApprovalToolCall {
            request: request.clone(),
            decision: None,
            surfaced: false,
            presentation_turn_id: presentation_turn_id.clone(),
            task_id: task.task_id,
            call,
            tool_request: task.tool_request,
            cancellation,
        };
        self.pending_approvals.insert(call_id.clone(), pending);
        if !self.pending_approval_order.iter().any(|id| id == &call_id) {
            self.pending_approval_order.push_back(call_id);
        }
        self.emit(AgentEvent::ApprovalRequired(request));
    }

    fn take_next_unsurfaced_approval_interrupt(&mut self) -> Option<LoopStep> {
        for call_id in self.pending_approval_order.clone() {
            let Some(pending) = self.pending_approvals.get_mut(&call_id) else {
                continue;
            };
            if pending.decision.is_none() && !pending.surfaced {
                pending.surfaced = true;
                return Some(LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(
                    PendingApproval {
                        request: pending.request.clone(),
                    },
                )));
            }
        }
        None
    }

    fn next_unresolved_approval_interrupt(&self) -> Option<LoopStep> {
        self.pending_approval_order.iter().find_map(|call_id| {
            self.pending_approvals.get(call_id).and_then(|pending| {
                pending.decision.is_none().then(|| {
                    LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(PendingApproval {
                        request: pending.request.clone(),
                    }))
                })
            })
        })
    }

    fn take_next_resolved_approval(&mut self) -> Option<PendingApprovalToolCall> {
        let call_id = self.pending_approval_order.iter().find_map(|call_id| {
            self.pending_approvals
                .get(call_id)
                .and_then(|pending| pending.decision.as_ref().map(|_| call_id.clone()))
        })?;
        self.pending_approval_order.retain(|id| id != &call_id);
        self.pending_approvals.remove(&call_id)
    }

    fn queue_resolution_interrupt(
        &mut self,
        presentation_turn_id: &agentkit_core::TurnId,
        resolution: TaskResolution,
        cancellation: Option<TurnCancellation>,
    ) -> Option<LoopStep> {
        match resolution {
            TaskResolution::Item(item) => {
                self.append_tool_result_item(item);
                None
            }
            TaskResolution::Approval(task) => {
                self.enqueue_pending_approval(presentation_turn_id, task, cancellation);
                self.take_next_unsurfaced_approval_interrupt()
            }
        }
    }

    async fn collect_pending_loop_updates(&mut self) -> Result<(), LoopError> {
        let PendingLoopUpdates { resolutions } = self
            .task_manager
            .take_pending_loop_updates()
            .await
            .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
        self.pending_loop_updates.extend(resolutions);
        Ok(())
    }

    async fn drain_pending_loop_updates(&mut self) -> Result<(bool, Option<LoopStep>), LoopError> {
        self.collect_pending_loop_updates().await?;
        let mut resolutions = std::mem::take(&mut self.pending_loop_updates);
        if !resolutions.is_empty() {
            self.start_logical_turn();
        }
        let mut saw_items = false;
        while let Some(resolution) = resolutions.pop_front() {
            match resolution {
                TaskResolution::Item(item) => {
                    self.append_tool_result_item(item);
                    saw_items = true;
                }
                TaskResolution::Approval(task) => {
                    let turn_id = self.start_logical_turn();
                    self.enqueue_pending_approval(&turn_id, task, None);
                }
            }
        }
        if let Some(step) = self.finish_cancelled_pending_approval().await? {
            return Ok((saw_items, Some(step)));
        }
        Ok((saw_items, self.take_next_unsurfaced_approval_interrupt()))
    }

    async fn finish_cancelled_pending_approval(&mut self) -> Result<Option<LoopStep>, LoopError> {
        if self.pending_approvals.is_empty() {
            return Ok(None);
        }
        if !self.pending_approvals.values().any(|pending| {
            pending
                .cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
        }) {
            return Ok(None);
        }
        self.cancel_pending_approvals().await
    }

    async fn run_mutators(
        &mut self,
        point: MutationPoint,
        turn_id: Option<&agentkit_core::TurnId>,
        cancellation: Option<TurnCancellation>,
    ) -> Result<(), LoopError> {
        if self.mutators.is_empty() {
            return Ok(());
        }
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return Err(LoopError::Cancelled);
        }
        let mutators = self.mutators.clone();
        let session_id = self.session_id.clone();
        let observed_session_id = Arc::clone(&self.observed_session_id);
        let observers = self.observers.clone();
        let emitter = DriverEmitter {
            session_id: &observed_session_id,
            observers: &observers,
        };
        let mut cursor = TranscriptCursor {
            items: &mut self.transcript,
            dirty: false,
        };
        for mutator in &mutators {
            if cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                return Err(LoopError::Cancelled);
            }
            let ctx = LoopCtx {
                session_id: &session_id,
                turn_id,
                point,
                cancellation: cancellation.clone(),
                emitter: &emitter,
            };
            mutator.mutate(&mut cursor, ctx).await?;
        }
        if cursor.dirty {
            validate_transcript_invariants(cursor.items)?;
        }
        Ok(())
    }

    async fn continue_active_tool_round(&mut self) -> Result<Option<LoopStep>, LoopError> {
        let Some((presentation_turn_id, task_turn_id, cancellation)) =
            self.active_tool_round.as_ref().map(|active| {
                (
                    active.presentation_turn_id.clone(),
                    active.task_turn_id.clone(),
                    active.cancellation.clone(),
                )
            })
        else {
            return Ok(None);
        };
        loop {
            if cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                self.task_manager
                    .on_turn_interrupted(&task_turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                self.active_tool_round = None;
                return self
                    .finish_cancelled(presentation_turn_id, Vec::new())
                    .map(Some);
            }

            let next_call = self
                .active_tool_round
                .as_mut()
                .and_then(|active| active.pending_calls.pop_front());
            if let Some((call, tool_request)) = next_call {
                use tracing::Instrument;
                self.register_tool_cancellation(&call.id, cancellation.clone());
                let dispatch_span =
                    self.execute_tool_span(&tool_request, &presentation_turn_id, "plain");
                match self
                    .start_task_via_manager(
                        None,
                        tool_request.clone(),
                        TaskLaunchKind::Plain,
                        cancellation.clone(),
                    )
                    .instrument(dispatch_span.clone())
                    .await?
                {
                    TaskStartOutcome::Ready(resolution) => {
                        let resolution = *resolution;
                        match resolution {
                            TaskResolution::Item(item) => {
                                if !tool_result_not_started(&item) {
                                    self.emit(AgentEvent::ToolExecutionStarted(call.clone()));
                                }
                                if tool_result_is_error(&item) {
                                    dispatch_span.record("error.type", "tool_error");
                                }
                                if let Some(active) = self.active_tool_round.as_mut() {
                                    active.foreground_progressed = true;
                                }
                                self.append_tool_result_item(item);
                            }
                            TaskResolution::Approval(task) => {
                                self.enqueue_pending_approval(
                                    &presentation_turn_id,
                                    task,
                                    cancellation.clone(),
                                );
                            }
                        }
                        continue;
                    }
                    TaskStartOutcome::Pending { kind, .. } => {
                        self.emit(AgentEvent::ToolExecutionStarted(call.clone()));
                        if kind == agentkit_task_manager::TaskKind::Background {
                            self.append_detach_placeholder(call.id.clone(), &call.name);
                            if let Some(active) = self.active_tool_round.as_mut() {
                                active.background_pending = true;
                            }
                        }
                        continue;
                    }
                }
            }

            match self
                .task_manager
                .wait_for_turn(&task_turn_id, cancellation.clone())
                .await
                .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?
            {
                Some(TurnTaskUpdate::Resolution(resolution)) => {
                    let resolution = *resolution;
                    match resolution {
                        TaskResolution::Item(item) => {
                            if let Some(active) = self.active_tool_round.as_mut() {
                                active.foreground_progressed = true;
                            }
                            self.append_tool_result_item(item);
                        }
                        TaskResolution::Approval(task) => {
                            self.enqueue_pending_approval(
                                &presentation_turn_id,
                                task,
                                cancellation.clone(),
                            );
                        }
                    }
                }
                Some(TurnTaskUpdate::Detached(snapshot)) => {
                    self.append_detach_placeholder(snapshot.call_id, &snapshot.tool_name);
                    if let Some(active) = self.active_tool_round.as_mut() {
                        active.background_pending = true;
                        active.foreground_progressed = true;
                    }
                }
                None => {
                    if cancellation
                        .as_ref()
                        .is_some_and(TurnCancellation::is_cancelled)
                    {
                        self.task_manager
                            .on_turn_interrupted(&task_turn_id)
                            .await
                            .map_err(|error| {
                                LoopError::Tool(ToolError::Internal(error.to_string()))
                            })?;
                        self.active_tool_round = None;
                        return self
                            .finish_cancelled(presentation_turn_id, Vec::new())
                            .map(Some);
                    }
                    let active = self.active_tool_round.take().ok_or_else(|| {
                        LoopError::InvalidState("missing active tool round".into())
                    })?;
                    if let Some(step) = self.take_next_unsurfaced_approval_interrupt() {
                        return Ok(Some(step));
                    }
                    if let Some(step) = self.next_unresolved_approval_interrupt() {
                        return Ok(Some(step));
                    }
                    if active.background_pending && !active.foreground_progressed {
                        return Ok(None);
                    }
                    // Yield control back to the host between tool rounds.
                    // All tool calls in this round have results in the
                    // transcript; the transcript is provider-valid.  The
                    // host may submit_input before calling next() to
                    // resume, which will re-enter drive_turn via
                    // pending_round_resume.
                    let info = ToolRoundInfo {
                        session_id: self.session_id.clone(),
                        turn_id: presentation_turn_id.clone(),
                        transcript_len: self.transcript.len(),
                    };
                    self.pending_round_resume = Some(presentation_turn_id);
                    return Ok(Some(LoopStep::Interrupt(LoopInterrupt::AfterToolResult(
                        info,
                    ))));
                }
            }
        }
    }

    #[tracing::instrument(
        name = "agent.turn",
        skip_all,
        fields(
            otel.name = "invoke_agent",
            gen_ai.operation.name = "invoke_agent",
            gen_ai.conversation.id = %self.session_id,
            gen_ai.provider.name = tracing::field::Empty,
            session.id = %self.session_id,
            turn.id = %turn_id,
            transcript.len = self.transcript.len(),
            saw_tool_call = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
        ),
    )]
    async fn drive_turn(
        &mut self,
        turn_id: agentkit_core::TurnId,
        mutation_point: MutationPoint,
    ) -> Result<LoopStep, LoopError> {
        let cancellation = self
            .cancellation
            .as_ref()
            .map(CancellationHandle::checkpoint);
        match self
            .run_mutators(mutation_point, Some(&turn_id), cancellation.clone())
            .await
        {
            Ok(()) => {}
            Err(LoopError::Cancelled) => {
                return self.finish_cancelled(turn_id, interrupted_assistant_items());
            }
            Err(error) => return Err(error),
        }

        // A mutator may have removed the freshly-submitted input (e.g. a
        // compaction pass that summarised the latest user turn away), leaving
        // the transcript ending in an assistant message or empty — nothing new
        // for the model to respond to. Finish the turn rather than dispatch an
        // assistant-prefill request, which most providers reject.
        if !transcript_has_pending_input(&self.transcript) {
            let turn_result = TurnResult {
                turn_id,
                finish_reason: FinishReason::Completed,
                items: Vec::new(),
                usage: None,
                metadata: MetadataMap::new(),
            };
            self.finish_logical_turn(&turn_result);
            return Ok(LoopStep::Finished(turn_result));
        }

        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return self.finish_cancelled(turn_id, interrupted_assistant_items());
        }

        let catalog_events = self.tool_executor.drain_catalog_events();
        self.emit_tool_catalog_events(catalog_events);

        let request = TurnRequest {
            session_id: self.session_id.clone(),
            turn_id: turn_id.clone(),
            transcript: self.transcript.clone(),
            available_tools: self.tool_executor.specs(),
            cache: self
                .next_turn_cache
                .take()
                .or_else(|| self.default_cache.clone()),
            metadata: MetadataMap::new(),
        };

        let session = self
            .session
            .as_mut()
            .ok_or_else(|| LoopError::InvalidState("model session is not available".into()))?;

        // Inference span per the OTel GenAI semantic conventions. It wraps the
        // model request and the full event drain rather than just `begin_turn`,
        // so attributes that streaming adapters only learn mid-stream (usage,
        // stop reason, response identity) still land before the span closes.
        // `otel.name` carries the dynamic `chat {model}` span name for
        // OpenTelemetry bridges since tracing span names are static.
        let chat_span = tracing::info_span!(
            "chat",
            "otel.name" = tracing::field::Empty,
            "otel.kind" = "client",
            "gen_ai.operation.name" = "chat",
            "gen_ai.provider.name" = tracing::field::Empty,
            "gen_ai.conversation.id" = %self.session_id,
            "gen_ai.request.model" = tracing::field::Empty,
            "gen_ai.response.model" = tracing::field::Empty,
            "gen_ai.response.id" = tracing::field::Empty,
            "gen_ai.response.finish_reasons" = tracing::field::Empty,
            "gen_ai.input.messages" = tracing::field::Empty,
            "gen_ai.output.messages" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
            "gen_ai.usage.cost" = tracing::field::Empty,
        );
        if let Some(capture) = self.telemetry.input_messages {
            record_string_array_attribute(
                &chat_span,
                "gen_ai.input.messages",
                capture_messages(&request.transcript, capture, CaptureOrder::NewestTail),
            );
        }

        // Seed known identity before begin_turn so request setup failures and
        // cancellation remain attributable. Successful per-turn routing below
        // overwrites these values with the effective selection.
        let initial_provider_name =
            effective_provider_name(session.provider_name(), self.provider_name.as_deref());
        if let Some(provider) = &initial_provider_name {
            chat_span.record("gen_ai.provider.name", provider.as_str());
            tracing::Span::current().record("gen_ai.provider.name", provider.as_str());
        }
        match session.model_name() {
            Some(model) => {
                chat_span.record("gen_ai.request.model", model);
                chat_span.record("otel.name", format!("chat {model}").as_str());
            }
            None => {
                chat_span.record("otel.name", "chat");
            }
        }

        use tracing::Instrument;
        let mut turn = match session
            .begin_turn(request, cancellation.clone())
            .instrument(chat_span.clone())
            .await
        {
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

        // begin_turn may apply per-turn routing. Sample the effective selection
        // only after that work, while the chat span still wraps begin_turn.
        let provider_name =
            effective_provider_name(session.provider_name(), self.provider_name.as_deref());
        if let Some(provider) = &provider_name {
            chat_span.record("gen_ai.provider.name", provider.as_str());
            tracing::Span::current().record("gen_ai.provider.name", provider.as_str());
        }
        match session.model_name() {
            Some(model) => {
                chat_span.record("gen_ai.request.model", model);
                chat_span.record("otel.name", format!("chat {model}").as_str());
            }
            None => {
                chat_span.record("otel.name", "chat");
            }
        }

        let mut saw_tool_call = false;
        let mut finished_result = None;
        let mut latest_usage = None;
        let mut streamed_content = cancellation
            .is_some()
            .then(StreamedAssistantContent::default);

        while let Some(event) = match turn
            .next_event(cancellation.clone())
            .instrument(chat_span.clone())
            .await
        {
            Ok(event) => event,
            Err(LoopError::Cancelled) => {
                self.task_manager
                    .on_turn_interrupted(&turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                return self.finish_cancelled(
                    turn_id,
                    interrupted_stream_items(streamed_content.as_ref()),
                );
            }
            Err(error) => return Err(error),
        } {
            let attempt_superseded = matches!(event, ModelTurnEvent::ResponseAttemptSuperseded);
            if attempt_superseded {
                saw_tool_call = false;
                latest_usage = None;
                if let Some(content) = &mut streamed_content {
                    content.reset();
                }
                self.emit(AgentEvent::ResponseAttemptSuperseded);
            }
            if cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                self.task_manager
                    .on_turn_interrupted(&turn_id)
                    .await
                    .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))?;
                return self.finish_cancelled(
                    turn_id,
                    interrupted_stream_items(streamed_content.as_ref()),
                );
            }
            if attempt_superseded {
                continue;
            }
            match event {
                ModelTurnEvent::Delta(delta) => {
                    if let Some(content) = &mut streamed_content {
                        content.apply_delta(&delta);
                    }
                    self.emit(AgentEvent::ContentDelta(delta));
                }
                ModelTurnEvent::Usage(usage) => {
                    latest_usage = Some(usage.clone());
                    self.emit(AgentEvent::UsageUpdated(usage));
                }
                ModelTurnEvent::ToolCall(call) => {
                    saw_tool_call = true;
                    if let Some(content) = &mut streamed_content {
                        content.commit_tool_call(&call);
                    }
                    self.emit(AgentEvent::ToolCallRequested(call));
                }
                ModelTurnEvent::ResponseAttemptSuperseded => {
                    unreachable!("response-attempt supersession is handled before cancellation")
                }
                ModelTurnEvent::Finished(result) => {
                    finished_result = Some(result);
                    break;
                }
            }
        }

        let mut result = finished_result.ok_or_else(|| {
            LoopError::Provider("model turn ended without a Finished event".into())
        })?;
        result.usage = merge_usage(result.usage, latest_usage);
        if let Some(model) = &result.model {
            chat_span.record("gen_ai.response.model", model.as_str());
        }
        if let Some(id) = &result.response_id {
            chat_span.record("gen_ai.response.id", id.as_str());
        }
        if let Some(tokens) = result
            .usage
            .as_ref()
            .and_then(|usage| usage.tokens.as_ref())
        {
            record_token_attribute(&chat_span, "gen_ai.usage.input_tokens", tokens.input_tokens);
            record_token_attribute(
                &chat_span,
                "gen_ai.usage.output_tokens",
                tokens.output_tokens,
            );
        }
        if let Some(cost) = result.usage.as_ref().and_then(|usage| usage.cost.as_ref()) {
            record_f64_attribute(&chat_span, "gen_ai.usage.cost", cost.amount);
        }
        record_string_array_attribute(
            &chat_span,
            "gen_ai.response.finish_reasons",
            provider_finish_reasons(&result.metadata, &result.finish_reason),
        );
        if let Some(capture) = self.telemetry.output_messages {
            record_string_array_attribute(
                &chat_span,
                "gen_ai.output.messages",
                capture_messages(&result.output_items, capture, CaptureOrder::OldestHead),
            );
        }
        drop(chat_span);
        tracing::Span::current().record("saw_tool_call", saw_tool_call);
        tracing::Span::current().record(
            "finish_reason",
            tracing::field::debug(&result.finish_reason),
        );
        let now = Timestamp::now();
        let usage = result.usage.clone();
        let finish_reason = result.finish_reason.clone();
        let output_items: Vec<Item> = result
            .output_items
            .drain(..)
            .map(|mut item| {
                if matches!(item.kind, ItemKind::Assistant) {
                    if item.usage.is_none() {
                        item.usage = usage.clone();
                    }
                    if item.finish_reason.is_none() {
                        item.finish_reason = Some(finish_reason.clone());
                    }
                }
                if item.created_at.is_none() {
                    item.created_at = Some(now);
                }
                item
            })
            .collect();
        self.extend_transcript(output_items.clone());

        if saw_tool_call {
            let pending_calls = extract_tool_calls(&output_items)
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
                presentation_turn_id: turn_id.clone(),
                task_turn_id: turn_id.clone(),
                pending_calls,
                cancellation: cancellation.clone(),
                background_pending: false,
                foreground_progressed: false,
            });
            if let Some(step) = self.continue_active_tool_round().await? {
                return Ok(step);
            }
            self.finish_logical_turn(&TurnResult {
                turn_id,
                finish_reason: result.finish_reason,
                items: output_items,
                usage: result.usage,
                metadata: result.metadata,
            });
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
            items: output_items,
            usage: result.usage,
            metadata: result.metadata,
        };
        self.finish_logical_turn(&turn_result);
        Ok(LoopStep::Finished(turn_result))
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
            ApprovalDecision::Approve => {
                use tracing::Instrument;
                self.emit(AgentEvent::ToolExecutionStarted(pending.call.clone()));
                let dispatch_span = self.execute_tool_span(
                    &pending.tool_request,
                    &pending.presentation_turn_id,
                    "approved",
                );
                let cancellation = self
                    .cancellation
                    .as_ref()
                    .map(CancellationHandle::checkpoint);
                self.register_tool_cancellation(&pending.call.id, cancellation.clone());
                let start = self
                    .start_task_via_manager(
                        Some(pending.task_id.clone()),
                        pending.tool_request.clone(),
                        TaskLaunchKind::Approved(pending.request.clone()),
                        cancellation.clone(),
                    )
                    .instrument(dispatch_span.clone())
                    .await;
                let outcome = match start {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        self.append_tool_result_item(Item {
                            id: None,
                            kind: ItemKind::Tool,
                            parts: vec![Part::ToolResult(ToolResultPart {
                                call_id: pending.call.id.clone(),
                                output: ToolOutput::Text(format!(
                                    "approved task failed to start: {error}"
                                )),
                                is_error: true,
                                metadata: pending.call.metadata.clone(),
                            })],
                            metadata: MetadataMap::new(),
                            usage: None,
                            finish_reason: None,
                            created_at: None,
                        });
                        let turn_id = pending.tool_request.turn_id.clone();
                        if let Err(cleanup_error) =
                            self.task_manager.on_turn_interrupted(&turn_id).await
                        {
                            tracing::debug!(
                                %cleanup_error,
                                %turn_id,
                                "failed to clean up turn after approved task start error"
                            );
                        }
                        return Err(error);
                    }
                };
                match outcome {
                    TaskStartOutcome::Ready(resolution) => {
                        let resolution = *resolution;
                        if let TaskResolution::Item(item) = &resolution
                            && tool_result_is_error(item)
                        {
                            dispatch_span.record("error.type", "tool_error");
                        }
                        if let Some(step) = self.queue_resolution_interrupt(
                            &pending.presentation_turn_id,
                            resolution,
                            cancellation,
                        ) {
                            return Ok(step);
                        }
                    }
                    TaskStartOutcome::Pending { kind, .. } => {
                        if kind == agentkit_task_manager::TaskKind::Background {
                            self.append_detach_placeholder(
                                pending.call.id.clone(),
                                &pending.call.name,
                            );
                        } else {
                            self.active_tool_round = Some(ActiveToolRound {
                                presentation_turn_id: pending.presentation_turn_id.clone(),
                                task_turn_id: pending.tool_request.turn_id.clone(),
                                pending_calls: VecDeque::new(),
                                cancellation: cancellation.clone(),
                                background_pending: false,
                                foreground_progressed: false,
                            });
                        }
                    }
                }
            }
            ApprovalDecision::Deny { reason } => {
                self.append_tool_result_item(Item {
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
                    usage: None,
                    finish_reason: None,
                    created_at: None,
                });
            }
        }

        if let Some(step) = self.continue_active_tool_round().await? {
            Ok(step)
        } else if let Some(step) = self.take_next_unsurfaced_approval_interrupt() {
            Ok(step)
        } else if let Some(step) = self.next_unresolved_approval_interrupt() {
            Ok(step)
        } else {
            self.drive_turn(pending.presentation_turn_id, MutationPoint::AfterToolResult)
                .await
        }
    }

    fn finish_cancelled(
        &mut self,
        turn_id: agentkit_core::TurnId,
        items: Vec<Item>,
    ) -> Result<LoopStep, LoopError> {
        let pending = self.drain_pending_approval_items();
        self.reject_drained_approvals(pending);
        self.extend_transcript(items.clone());
        self.close_interrupted_tool_calls();
        let turn_result = TurnResult {
            turn_id,
            finish_reason: FinishReason::Cancelled,
            items,
            usage: None,
            metadata: interrupted_metadata("turn"),
        };
        self.finish_logical_turn(&turn_result);
        Ok(LoopStep::Finished(turn_result))
    }

    /// Internal entry point for buffering user input. Reachable only via
    /// [`InputRequest::submit`] (resolves an `AwaitingInput` interrupt,
    /// including the very first one after [`Agent::start`]) and
    /// [`ToolRoundInfo::submit`] (interjects between tool rounds). Prior
    /// transcript items — the passive starting state of a session — are
    /// preloaded via [`AgentBuilder::transcript`]; an opening user turn for
    /// one-shot calls is preloaded via [`AgentBuilder::input`]. New input
    /// after start-up always flows through one of the typed `submit`
    /// handles.
    pub fn submit_input(&mut self, input: Vec<Item>) -> Result<(), LoopError> {
        if self.has_pending_interrupts() {
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

    /// Override the prompt cache request for the next model turn.
    ///
    /// The override is consumed the next time the driver starts a model turn.
    /// Session-level defaults still apply to later turns.
    pub fn set_next_turn_cache(&mut self, cache: PromptCacheRequest) -> Result<(), LoopError> {
        if self.has_pending_interrupts() {
            return Err(LoopError::InvalidState(
                "cannot update next-turn cache while an interrupt is pending".into(),
            ));
        }
        self.next_turn_cache = Some(cache);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn submit_input_with_cache(
        &mut self,
        input: Vec<Item>,
        cache: PromptCacheRequest,
    ) -> Result<(), LoopError> {
        self.set_next_turn_cache(cache)?;
        self.submit_input(input)
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
    pub fn resolve_approval_for(
        &mut self,
        call_id: ToolCallId,
        decision: ApprovalDecision,
    ) -> Result<(), LoopError> {
        let Some(pending) = self.pending_approvals.get_mut(&call_id) else {
            return Err(LoopError::InvalidState(format!(
                "no approval request is pending for call {}",
                call_id.0
            )));
        };
        pending.decision = Some(decision.clone());
        self.emit(AgentEvent::ApprovalResolved {
            approved: matches!(decision, ApprovalDecision::Approve),
        });
        Ok(())
    }

    /// Resolve a pending [`LoopInterrupt::ApprovalRequest`] with a patched
    /// input that replaces the model's original tool arguments.
    ///
    /// Equivalent to calling [`resolve_approval_for`] with
    /// [`ApprovalDecision::Approve`] except the tool sees `input` instead of
    /// what the model emitted. The transcript still records the model's
    /// original call.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if no approval is pending for
    /// `call_id`.
    pub fn resolve_approval_for_with_patched_input(
        &mut self,
        call_id: ToolCallId,
        input: serde_json::Value,
    ) -> Result<(), LoopError> {
        let Some(pending) = self.pending_approvals.get_mut(&call_id) else {
            return Err(LoopError::InvalidState(format!(
                "no approval request is pending for call {}",
                call_id.0
            )));
        };
        pending.tool_request.input = input;
        self.resolve_approval_for(call_id, ApprovalDecision::Approve)
    }

    /// Resolve a pending [`LoopInterrupt::ApprovalRequest`] when exactly one
    /// approval is outstanding.
    pub fn resolve_approval(&mut self, decision: ApprovalDecision) -> Result<(), LoopError> {
        let mut unresolved = self
            .pending_approval_order
            .iter()
            .filter(|call_id| {
                self.pending_approvals
                    .get(*call_id)
                    .is_some_and(|pending| pending.decision.is_none())
            })
            .cloned();
        let Some(call_id) = unresolved.next() else {
            return Err(LoopError::InvalidState(
                "no approval request is pending".into(),
            ));
        };
        if unresolved.next().is_some() {
            return Err(LoopError::InvalidState(
                "multiple approvals are pending; use resolve_approval_for".into(),
            ));
        }
        self.resolve_approval_for(call_id, decision)
    }

    /// Cancel a pending approval interrupt for a specific tool call.
    ///
    /// This clears the blocking approval and appends an error tool result so
    /// the transcript remains provider-valid if the host continues the turn.
    pub fn cancel_pending_approval_for(&mut self, call_id: ToolCallId) -> Result<(), LoopError> {
        let Some(pending) = self.drain_pending_approval_for(&call_id) else {
            return Err(LoopError::InvalidState(format!(
                "no approval request is pending for call {}",
                call_id.0
            )));
        };
        let turn_id = pending.presentation_turn_id.clone();
        self.reject_drained_approvals(vec![pending]);
        if self.pending_approvals.is_empty() && self.active_tool_round.is_none() {
            let _ = self.finish_cancelled(turn_id, Vec::new())?;
        }
        Ok(())
    }

    /// Cancel every pending approval interrupt.
    ///
    /// This is useful when the host cancels the containing turn rather than an
    /// individual approval prompt. Each pending approval is resolved as denied
    /// and receives an error tool result so the transcript remains valid.
    pub async fn cancel_pending_approvals(&mut self) -> Result<Option<LoopStep>, LoopError> {
        if self.pending_approvals.is_empty() {
            return Ok(None);
        }
        let Some(turn_id) = self
            .pending_approval_order
            .iter()
            .find_map(|call_id| self.pending_approvals.get(call_id))
            .map(|pending| pending.presentation_turn_id.clone())
        else {
            return Ok(None);
        };
        let mut seen_turns = HashSet::new();
        let mut originating_turns = Vec::new();
        for pending in self.pending_approvals.values() {
            let originating_turn = pending.tool_request.turn_id.clone();
            if seen_turns.insert(originating_turn.clone()) {
                originating_turns.push(originating_turn);
            }
        }

        let pending = self.drain_pending_approval_items();
        self.active_tool_round = None;
        let mut cleanup_error = None;
        for originating_turn in originating_turns {
            if let Err(error) = self
                .task_manager
                .on_turn_interrupted(&originating_turn)
                .await
                && cleanup_error.is_none()
            {
                cleanup_error = Some(LoopError::Tool(ToolError::Internal(error.to_string())));
            }
        }
        self.reject_drained_approvals(pending);
        if let Some(error) = cleanup_error {
            self.close_interrupted_tool_calls();
            self.finish_logical_turn(&TurnResult {
                turn_id,
                finish_reason: FinishReason::Error,
                items: Vec::new(),
                usage: None,
                metadata: MetadataMap::new(),
            });
            return Err(error);
        }
        self.finish_cancelled(turn_id, Vec::new()).map(Some)
    }

    /// Take a read-only snapshot of the driver's current transcript and input queue.
    pub fn snapshot(&self) -> LoopSnapshot {
        LoopSnapshot {
            session_id: self.session_id.clone(),
            transcript: self.transcript.clone(),
            pending_input: self.pending_input.clone(),
        }
    }

    /// Wait until an out-of-band update is available for the loop.
    ///
    /// This resolves immediately for updates already collected from the task
    /// manager but deferred behind fresh input. It does not consume the update;
    /// call [`next`](Self::next) after it resolves to append and drive the result.
    pub fn wait_for_loop_update(
        &self,
    ) -> impl std::future::Future<Output = Result<(), LoopError>> + Send + 'static {
        let has_collected_update = !self.pending_loop_updates.is_empty();
        let task_manager = self.task_manager.clone();
        async move {
            if has_collected_update {
                return Ok(());
            }
            task_manager
                .wait_for_loop_update()
                .await
                .map_err(|error| LoopError::Tool(ToolError::Internal(error.to_string())))
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
    /// This is the steady state after [`Agent::start`] when no input was
    /// preloaded via [`AgentBuilder::input`]: the prior transcript loaded
    /// via [`AgentBuilder::transcript`] is passive, so the first call
    /// surfaces `AwaitingInput` and waits for the host to supply input via
    /// [`InputRequest::submit`] before any model turn is dispatched. If
    /// input was preloaded, the first call dispatches the model directly.
    ///
    /// # Errors
    ///
    /// Returns [`LoopError::InvalidState`] if called while an unresolved
    /// interrupt is pending, or propagates provider / tool / compaction errors.
    pub async fn next(&mut self) -> Result<LoopStep, LoopError> {
        if self.lifecycle.active_turn.is_none() {
            let continuation_turn = self
                .pending_approval_order
                .iter()
                .find_map(|call_id| self.pending_approvals.get(call_id))
                .map(|pending| pending.presentation_turn_id.clone())
                .or_else(|| {
                    self.active_tool_round
                        .as_ref()
                        .map(|active| active.presentation_turn_id.clone())
                })
                .or_else(|| self.pending_round_resume.clone());
            if let Some(turn_id) = continuation_turn {
                self.start_logical_turn_with(turn_id);
            } else if !self.pending_input.is_empty() {
                self.start_logical_turn();
            }
        }

        let result = self.next_inner().await;
        match &result {
            Ok(LoopStep::Finished(turn)) => self.finish_logical_turn(turn),
            Err(_) => {
                if let Some(turn_id) = self.lifecycle.active_turn.clone() {
                    self.recover_from_next_error().await;
                    self.finish_logical_turn(&TurnResult {
                        turn_id,
                        finish_reason: FinishReason::Error,
                        items: Vec::new(),
                        usage: None,
                        metadata: MetadataMap::new(),
                    });
                }
            }
            _ => {}
        }
        result
    }

    async fn recover_from_next_error(&mut self) {
        let mut seen_turns = HashSet::new();
        let mut interrupted_turns = Vec::new();
        if let Some(active) = self.active_tool_round.take()
            && seen_turns.insert(active.task_turn_id.clone())
        {
            interrupted_turns.push(active.task_turn_id);
        }
        if let Some(turn_id) = self.pending_round_resume.take()
            && seen_turns.insert(turn_id.clone())
        {
            interrupted_turns.push(turn_id);
        }
        for pending in self.pending_approvals.values() {
            let turn_id = pending.tool_request.turn_id.clone();
            if seen_turns.insert(turn_id.clone()) {
                interrupted_turns.push(turn_id);
            }
        }

        let pending = self.drain_pending_approval_items();
        for turn_id in interrupted_turns {
            if let Err(error) = self.task_manager.on_turn_interrupted(&turn_id).await {
                tracing::debug!(%error, %turn_id, "failed to clean up turn after loop error");
            }
        }
        self.reject_drained_approvals(pending);
        self.close_interrupted_tool_calls();
    }

    async fn next_inner(&mut self) -> Result<LoopStep, LoopError> {
        if let Some(pending) = self.take_next_resolved_approval() {
            return self.resume_after_approval(pending).await;
        }

        if let Some(step) = self.finish_cancelled_pending_approval().await? {
            return Ok(step);
        }

        if let Some(step) = self.take_next_unsurfaced_approval_interrupt() {
            return Ok(step);
        }

        if let Some(step) = self.next_unresolved_approval_interrupt() {
            return Ok(step);
        }

        if let Some(step) = self.continue_active_tool_round().await? {
            return Ok(step);
        }

        // A newly submitted user turn owns the next logical turn. Drive it
        // before unrelated background completions so a delayed approval cannot
        // bind itself to that turn's TurnStarted event. AfterToolResult resumes
        // remain ordered ahead of fresh input below.
        if self.pending_round_resume.is_none() && !self.pending_input.is_empty() {
            // Take updates now to preserve the driver's once-per-step manager
            // handoff, but defer presenting them until this input turn ends.
            self.collect_pending_loop_updates().await?;
            let turn_id = self.start_logical_turn();
            let drained: Vec<Item> = std::mem::take(&mut self.pending_input);
            self.extend_transcript(drained);
            return self
                .drive_turn(turn_id, MutationPoint::AfterTurnEnded)
                .await;
        }

        let (had_loop_updates, loop_step) = self.drain_pending_loop_updates().await?;
        if let Some(step) = loop_step {
            return Ok(step);
        }

        // Resume after an AfterToolResult yield.  Any input submitted by the
        // host during the yield is folded into the transcript as part of the
        // continuation turn; background task results drained just above are
        // already in the transcript.
        if let Some(turn_id) = self.pending_round_resume.take() {
            let drained: Vec<Item> = std::mem::take(&mut self.pending_input);
            self.extend_transcript(drained);
            return self
                .drive_turn(turn_id, MutationPoint::AfterToolResult)
                .await;
        }

        if self.pending_input.is_empty() && !had_loop_updates {
            return Ok(LoopStep::Interrupt(LoopInterrupt::AwaitingInput(
                InputRequest {
                    session_id: self.session_id.clone(),
                    reason: "driver is waiting for input".into(),
                },
            )));
        }

        let turn_id = self.start_logical_turn();
        let drained: Vec<Item> = std::mem::take(&mut self.pending_input);
        self.extend_transcript(drained);
        self.drive_turn(turn_id, MutationPoint::AfterTurnEnded)
            .await
    }

    fn emit(&self, event: AgentEvent) {
        fan_out_observed_event(&self.observers, &self.observed_session_id, event);
    }

    /// Append a single [`Item`] to the transcript and notify all
    /// registered [`TranscriptObserver`]s. The single mutation point —
    /// every push to `self.transcript` should funnel through here so
    /// observers see exactly what landed in the transcript.
    fn append_item(&mut self, mut item: Item) {
        if item.created_at.is_none() {
            item.created_at = Some(Timestamp::now());
        }
        for observer in &self.transcript_observers {
            observer.on_transcript_event(TranscriptEvent {
                session_id: &self.session_id,
                item: &item,
            });
        }
        self.transcript.push(item);
    }

    fn append_detach_placeholder(&mut self, call_id: ToolCallId, tool_name: &str) {
        self.background_call_ids.insert(call_id.clone());
        if !self.detached_call_ids.insert(call_id.clone()) {
            return;
        }
        let detached_result = ToolResultPart {
            call_id: call_id.clone(),
            output: ToolOutput::Text(format!(
                "Tool {tool_name} is now running in the background. The result will be delivered when it completes."
            )),
            is_error: false,
            metadata: MetadataMap::new(),
        };
        self.emit(AgentEvent::ToolExecutionProgress(detached_result.clone()));
        self.append_item(Item {
            id: None,
            kind: ItemKind::Tool,
            parts: vec![Part::ToolResult(detached_result)],
            metadata: MetadataMap::new(),
            usage: None,
            finish_reason: None,
            created_at: None,
        });
    }

    /// Append a tool-result Item: emit one [`AgentEvent::ToolResultReceived`]
    /// per [`Part::ToolResult`] inside the Item, then funnel through
    /// [`Self::append_item`].
    ///
    /// If every `ToolResult` in the item references a `call_id` that was
    /// already paired with a synthetic detach tool_result, the item is
    /// converted to a [`ItemKind::Notification`] before appending.
    /// Without this, we would emit a second `tool_result` for the same
    /// `tool_use_id` — a provider-schema violation that
    /// Anthropic/OpenRouter reject as an "orphaned tool_result".
    /// Observers see [`AgentEvent::ToolExecutionProgress`] for the synthetic
    /// detach placeholder and see [`AgentEvent::ToolResultReceived`] only for
    /// the later terminal result.
    fn append_tool_result_item(&mut self, item: Item) {
        for part in &item.parts {
            if let Part::ToolResult(result) = part {
                if !self
                    .interrupted_background_call_ids
                    .contains(&result.call_id)
                {
                    self.emit(AgentEvent::ToolResultReceived(result.clone()));
                }
                self.background_call_ids.remove(&result.call_id);
                self.clear_tool_cancellation(&result.call_id);
            }
        }
        let item = self.maybe_convert_detached(item);
        self.append_item(item);
    }

    fn drain_pending_approval_for(
        &mut self,
        call_id: &ToolCallId,
    ) -> Option<PendingApprovalToolCall> {
        let pending = self.pending_approvals.remove(call_id)?;
        self.pending_approval_order.retain(|id| id != call_id);
        self.clear_tool_cancellation(call_id);
        Some(pending)
    }

    fn drain_pending_approval_items(&mut self) -> Vec<PendingApprovalToolCall> {
        let order = std::mem::take(&mut self.pending_approval_order);
        let pending = order
            .iter()
            .filter_map(|call_id| {
                let pending = self.pending_approvals.remove(call_id);
                self.clear_tool_cancellation(call_id);
                pending
            })
            .collect();
        self.pending_approvals.clear();
        pending
    }

    fn reject_drained_approvals(&mut self, pending: Vec<PendingApprovalToolCall>) {
        for pending in pending {
            self.emit(AgentEvent::ApprovalResolved { approved: false });
            self.append_tool_result_item(cancelled_approval_item(pending));
        }
    }

    /// Answer every tool call the cancelled turn will never come back to.
    ///
    /// A cancelled turn abandons the calls it had in flight, and a transcript
    /// carrying a `tool_use` without its `tool_result` is one that
    /// [`validate_transcript_invariants`] rejects and that providers refuse
    /// outright ("No tool output found for function call ..."). Since the
    /// results are appended through [`Self::append_tool_result_item`], hosts
    /// persisting the transcript through a [`TranscriptObserver`] record a
    /// resumable session rather than one that has to be repaired on read.
    ///
    /// This is the same closing move [`Self::reject_drained_approvals`] makes
    /// for a denied approval; cancellation owes its calls the same answer.
    ///
    /// Background tasks outlive the turn that started them. Their real results
    /// are converted to notifications; calls that never started or were
    /// cancelled in the foreground are not retained as detached work.
    fn close_interrupted_tool_calls(&mut self) {
        for call in unanswered_tool_calls(&self.transcript) {
            let call_id = call.id.clone();
            let completes_in_background = self.background_call_ids.contains(&call_id);
            self.append_tool_result_item(interrupted_tool_result_item(call));
            if completes_in_background {
                self.detached_call_ids.insert(call_id.clone());
                self.interrupted_background_call_ids.insert(call_id);
            }
        }
    }

    fn maybe_convert_detached(&mut self, mut item: Item) -> Item {
        if !matches!(item.kind, ItemKind::Tool) {
            return item;
        }
        let results: Vec<&ToolResultPart> = item
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolResult(r) => Some(r),
                _ => None,
            })
            .collect();
        if results.is_empty()
            || !results
                .iter()
                .all(|r| self.detached_call_ids.contains(&r.call_id))
        {
            return item;
        }
        let structured_results = results
            .iter()
            .map(|result| {
                Part::structured(serde_json::to_value(result).unwrap_or_else(
                    |error| serde_json::json!({ "serialization_error": error.to_string() }),
                ))
            })
            .collect::<Vec<_>>();
        let failed = results.iter().filter(|result| result.is_error).count();
        let with_metadata = results
            .iter()
            .filter(|result| !result.metadata.is_empty())
            .count();
        let mut text = format!(
            "Background tool results: {} total, {failed} failed, {with_metadata} with metadata. ",
            results.len()
        );
        for (index, result) in results.iter().enumerate() {
            self.detached_call_ids.remove(&result.call_id);
            self.interrupted_background_call_ids.remove(&result.call_id);
            if text.chars().count() >= DETACHED_NOTIFICATION_TEXT_MAX_CHARS {
                continue;
            }
            if index > 0 {
                text.push_str("; ");
            }
            let label = if result.is_error {
                "failed"
            } else {
                "completed"
            };
            let call_id = truncate_chars(&result.call_id.0, DETACHED_CALL_ID_MAX_CHARS);
            let body = render_tool_output_brief(&result.output);
            text.push_str(&format!("{call_id} {label}: {body}"));
        }
        let text = truncate_chars(&text, DETACHED_NOTIFICATION_TEXT_MAX_CHARS);
        let mut notification_parts = Vec::with_capacity(1 + structured_results.len());
        notification_parts.push(Part::text(text));
        notification_parts.extend(structured_results);
        item.kind = ItemKind::Notification;
        item.parts = notification_parts;
        item
    }

    /// Append several Items in order through [`Self::append_item`].
    /// Pre-stamps `created_at` once per batch so all items in the batch
    /// share a timestamp and `append_item` skips its own clock read.
    fn extend_transcript(&mut self, items: impl IntoIterator<Item = Item>) {
        let now = Timestamp::now();
        for mut item in items {
            if item.created_at.is_none() {
                item.created_at = Some(now);
            }
            self.append_item(item);
        }
    }
}

fn render_tool_output_brief(output: &ToolOutput) -> String {
    match output {
        ToolOutput::Text(text) => format!(
            "text preview: {}",
            truncate_chars(text, DETACHED_TEXT_PREVIEW_MAX_CHARS)
        ),
        ToolOutput::Structured(_) => "structured payload".into(),
        ToolOutput::Parts(parts) => format!("parts payload ({} parts)", parts.len()),
        ToolOutput::Files(files) => format!("files payload ({} files)", files.len()),
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let mut truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() && max_chars > 0 {
        truncated.pop();
        truncated.push('…');
    }
    truncated
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

fn record_token_attribute(span: &tracing::Span, key: &'static str, value: u64) {
    match i64::try_from(value) {
        Ok(value) => record_i64_attribute(span, key, value),
        Err(_) => tracing::warn!(attribute = key, value, "token count exceeds OTEL i64 range"),
    }
}

#[cfg(feature = "otel")]
fn record_i64_attribute(span: &tracing::Span, key: &'static str, value: i64) {
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    span.set_attribute(key, value);
}

#[cfg(not(feature = "otel"))]
fn record_i64_attribute(span: &tracing::Span, key: &'static str, value: i64) {
    span.record(key, value);
}

#[cfg(feature = "otel")]
fn record_f64_attribute(span: &tracing::Span, key: &'static str, value: f64) {
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    span.set_attribute(key, value);
}

#[cfg(not(feature = "otel"))]
fn record_f64_attribute(span: &tracing::Span, key: &'static str, value: f64) {
    span.record(key, value);
}

#[cfg(feature = "otel")]
fn otel_string_array(values: Vec<String>) -> opentelemetry::Value {
    use opentelemetry::{Array, StringValue, Value as OtelValue};
    OtelValue::Array(Array::String(
        values.into_iter().map(StringValue::from).collect(),
    ))
}

#[cfg(feature = "otel")]
fn record_string_array_attribute(span: &tracing::Span, key: &'static str, values: Vec<String>) {
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    span.set_attribute(key, otel_string_array(values));
}

#[cfg(not(feature = "otel"))]
fn record_string_array_attribute(span: &tracing::Span, key: &'static str, values: Vec<String>) {
    span.record(key, tracing::field::debug(&values));
}

#[derive(Clone, Copy)]
enum CaptureOrder {
    NewestTail,
    OldestHead,
}

fn effective_provider_name(
    session_provider: Option<&str>,
    adapter_provider: Option<&str>,
) -> Option<String> {
    session_provider.or(adapter_provider).map(str::to_owned)
}

fn merge_usage(final_usage: Option<Usage>, streamed_usage: Option<Usage>) -> Option<Usage> {
    match (final_usage, streamed_usage) {
        (None, streamed) => streamed,
        (Some(final_usage), None) => Some(final_usage),
        (Some(mut final_usage), Some(streamed)) => {
            if final_usage.tokens.is_none() {
                final_usage.tokens = streamed.tokens;
            }
            if final_usage.cost.is_none() {
                final_usage.cost = streamed.cost;
            }
            for (key, value) in streamed.metadata {
                final_usage.metadata.entry(key).or_insert(value);
            }
            Some(final_usage)
        }
    }
}

fn capture_messages(items: &[Item], capture: MessageCapture, order: CaptureOrder) -> Vec<String> {
    let mut captured = Vec::new();
    let mut used_bytes = 0usize;
    let indices: Box<dyn Iterator<Item = usize>> = match order {
        CaptureOrder::NewestTail => Box::new((0..items.len()).rev()),
        CaptureOrder::OldestHead => Box::new(0..items.len()),
    };

    for index in indices.take(capture.max_messages) {
        let item = &items[index];
        let original_bytes = source_content_bytes(item);
        let remaining = capture.max_bytes.saturating_sub(used_bytes);
        if original_bytes > remaining {
            captured.push(
                serde_json::json!({
                    "type": "truncated",
                    "original_bytes": original_bytes,
                })
                .to_string(),
            );
            break;
        }
        used_bytes += original_bytes;
        captured.push(capture_item_json(item, remaining));
    }

    if matches!(order, CaptureOrder::NewestTail) {
        captured.reverse();
    }
    captured
}

fn source_content_bytes(item: &Item) -> usize {
    item.parts.iter().fold(0, |total, part| {
        total.saturating_add(part_source_content_bytes(part))
    })
}

fn part_source_content_bytes(part: &Part) -> usize {
    match part {
        Part::Text(text) => text.text.len(),
        Part::Media(media) => media.mime_type.len(),
        Part::File(file) => file
            .name
            .as_deref()
            .map_or(0, str::len)
            .saturating_add(file.mime_type.as_deref().map_or(0, str::len)),
        Part::Structured(_) => 0,
        Part::Reasoning(reasoning) => reasoning.summary.as_deref().map_or(0, str::len),
        Part::ToolCall(call) => call.id.0.len().saturating_add(call.name.len()),
        Part::ToolResult(result) => result
            .call_id
            .0
            .len()
            .saturating_add(tool_output_source_content_bytes(&result.output)),
        Part::Custom(custom) => custom.kind.len(),
    }
}

fn tool_output_source_content_bytes(output: &ToolOutput) -> usize {
    match output {
        ToolOutput::Text(text) => text.len(),
        ToolOutput::Structured(_) | ToolOutput::Parts(_) | ToolOutput::Files(_) => 0,
    }
}

struct CaptureBudget {
    remaining: usize,
}

impl CaptureBudget {
    fn text(&mut self, text: &str) -> (String, bool) {
        let end = floor_char_boundary(text, self.remaining.min(text.len()));
        self.remaining = self.remaining.saturating_sub(end);
        (text[..end].to_owned(), end < text.len())
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

const MAX_CAPTURED_PARTS_PER_ITEM: usize = 256;

fn capture_item_json(item: &Item, max_bytes: usize) -> String {
    let mut budget = CaptureBudget {
        remaining: max_bytes,
    };
    let mut parts = item
        .parts
        .iter()
        .take(MAX_CAPTURED_PARTS_PER_ITEM)
        .map(|part| sanitized_part(part, &mut budget))
        .collect::<Vec<_>>();
    if item.parts.len() > parts.len() {
        parts.push(serde_json::json!({
            "type": "truncated",
            "reason": "part_limit",
        }));
    }
    serde_json::json!({
        "role": item_kind_name(item.kind),
        "parts": parts,
    })
    .to_string()
}

fn item_kind_name(kind: ItemKind) -> &'static str {
    match kind {
        ItemKind::System => "system",
        ItemKind::Developer => "developer",
        ItemKind::User => "user",
        ItemKind::Assistant => "assistant",
        ItemKind::Tool => "tool",
        ItemKind::Context => "context",
        ItemKind::Notification => "notification",
    }
}

fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Audio => "audio",
        Modality::Image => "image",
        Modality::Video => "video",
        Modality::Binary => "binary",
    }
}

fn omitted_data_ref(data: &DataRef) -> Value {
    let kind = match data {
        DataRef::InlineText(_) => "inline_text",
        DataRef::InlineBytes(_) => "inline_bytes",
        DataRef::Uri(_) => "uri",
        DataRef::Handle(_) => "handle",
    };
    serde_json::json!({ "kind": kind, "omitted": true })
}

fn bounded_field(text: &str, budget: &mut CaptureBudget) -> Value {
    let (text, truncated) = budget.text(text);
    serde_json::json!({ "value": text, "truncated": truncated })
}

fn sanitized_part(part: &Part, budget: &mut CaptureBudget) -> Value {
    match part {
        Part::Text(text) => serde_json::json!({
            "type": "text",
            "text": bounded_field(&text.text, budget),
        }),
        Part::Media(media) => serde_json::json!({
            "type": "media",
            "modality": modality_name(media.modality),
            "mime_type": bounded_field(&media.mime_type, budget),
            "data": omitted_data_ref(&media.data),
        }),
        Part::File(file) => serde_json::json!({
            "type": "file",
            "name": file.name.as_deref().map(|name| bounded_field(name, budget)),
            "mime_type": file.mime_type.as_deref().map(|mime| bounded_field(mime, budget)),
            "data": omitted_data_ref(&file.data),
        }),
        Part::Structured(_) => serde_json::json!({
            "type": "structured",
            "truncated": true,
        }),
        Part::Reasoning(reasoning) => serde_json::json!({
            "type": "reasoning",
            "summary": reasoning.summary.as_deref().map(|summary| bounded_field(summary, budget)),
            "redacted": reasoning.redacted,
            "data": reasoning.data.as_ref().map(omitted_data_ref),
        }),
        Part::ToolCall(call) => serde_json::json!({
            "type": "tool_call",
            "id": bounded_field(&call.id.0, budget),
            "name": bounded_field(&call.name, budget),
            "input": { "truncated": true },
        }),
        Part::ToolResult(result) => serde_json::json!({
            "type": "tool_result",
            "call_id": bounded_field(&result.call_id.0, budget),
            "is_error": result.is_error,
            "output": sanitized_tool_output(&result.output, budget),
        }),
        Part::Custom(custom) => serde_json::json!({
            "type": "custom",
            "kind": bounded_field(&custom.kind, budget),
            "data": custom.data.as_ref().map(omitted_data_ref),
            "value": custom.value.as_ref().map(|_| serde_json::json!({ "truncated": true })),
        }),
    }
}

fn sanitized_tool_output(output: &ToolOutput, budget: &mut CaptureBudget) -> Value {
    match output {
        ToolOutput::Text(text) => serde_json::json!({
            "type": "text",
            "text": bounded_field(text, budget),
        }),
        ToolOutput::Structured(_) => serde_json::json!({
            "type": "structured",
            "truncated": true,
        }),
        ToolOutput::Parts(parts) => serde_json::json!({
            "type": "parts",
            "count": parts.len(),
            "truncated": true,
        }),
        ToolOutput::Files(files) => serde_json::json!({
            "type": "files",
            "count": files.len(),
            "truncated": true,
        }),
    }
}

#[cfg(test)]
mod telemetry_tests {
    use super::*;

    #[test]
    fn message_capture_is_off_by_default_and_independent() {
        let default = TelemetryConfig::default();
        assert_eq!(default.input_messages(), None);
        assert_eq!(default.output_messages(), None);

        let capture = MessageCapture::new(2, 1).unwrap();
        let input_only = TelemetryConfig::default().with_input_messages(capture);
        assert_eq!(input_only.input_messages().unwrap().max_messages(), 2);
        assert_eq!(input_only.input_messages().unwrap().max_bytes(), 1);
        assert_eq!(input_only.output_messages(), None);
        assert_eq!(
            MessageCapture::new(0, 1),
            Err(MessageCaptureError::ZeroMessages)
        );
        assert_eq!(
            MessageCapture::new(1, 0),
            Err(MessageCaptureError::ZeroBytes)
        );
    }

    #[test]
    fn one_source_byte_is_not_rejected_for_json_envelope_overhead() {
        let items = vec![Item::text(ItemKind::User, "x")];
        let captured = capture_messages(
            &items,
            MessageCapture::new(1, 1).unwrap(),
            CaptureOrder::OldestHead,
        );
        assert_eq!(captured.len(), 1);
        let value: Value = serde_json::from_str(&captured[0]).unwrap();
        assert_eq!(value["role"], "user");
        assert_eq!(value["parts"][0]["text"]["value"], "x");
        assert_eq!(value["parts"][0]["text"]["truncated"], false);
    }

    #[test]
    fn multibyte_source_accounting_preserves_utf8_boundaries() {
        let items = vec![Item::text(ItemKind::User, "é")];

        let exact = capture_messages(
            &items,
            MessageCapture::new(1, "é".len()).unwrap(),
            CaptureOrder::OldestHead,
        );
        let value: Value = serde_json::from_str(&exact[0]).unwrap();
        assert_eq!(value["parts"][0]["text"]["value"], "é");
        assert_eq!(value["parts"][0]["text"]["truncated"], false);

        let too_small = capture_messages(
            &items,
            MessageCapture::new(1, 1).unwrap(),
            CaptureOrder::OldestHead,
        );
        let value: Value = serde_json::from_str(&too_small[0]).unwrap();
        assert_eq!(value["type"], "truncated");
        assert_eq!(value["original_bytes"], 2);
    }

    #[test]
    fn source_bytes_are_aggregated_without_charging_json_envelopes() {
        let items = vec![
            Item::text(ItemKind::User, "a"),
            Item::text(ItemKind::Assistant, "b"),
            Item::text(ItemKind::User, "cd"),
        ];
        let captured = capture_messages(
            &items,
            MessageCapture::new(3, 3).unwrap(),
            CaptureOrder::OldestHead,
        );
        assert_eq!(captured.len(), 3);
        let values = captured
            .iter()
            .map(|encoded| serde_json::from_str::<Value>(encoded).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values[0]["parts"][0]["text"]["value"], "a");
        assert_eq!(values[1]["parts"][0]["text"]["value"], "b");
        assert_eq!(values[2]["type"], "truncated");
        assert_eq!(values[2]["original_bytes"], 2);
    }

    #[test]
    fn tiny_limits_emit_valid_structured_source_byte_truncation() {
        let items = vec![Item::text(ItemKind::User, "x".repeat(1_000))];
        let captured = capture_messages(
            &items,
            MessageCapture::new(1, 1).unwrap(),
            CaptureOrder::OldestHead,
        );
        assert_eq!(captured.len(), 1);
        let value: Value = serde_json::from_str(&captured[0]).unwrap();
        assert_eq!(value["type"], "truncated");
        assert_eq!(value["original_bytes"], 1_000);
    }

    #[test]
    fn provider_finish_reason_metadata_round_trips() {
        let mut metadata = MetadataMap::new();
        set_provider_finish_reasons(&mut metadata, ["end_turn", "", "tool_use", "end_turn"]);
        assert_eq!(
            provider_finish_reasons(&metadata, &FinishReason::Completed),
            vec!["end_turn", "tool_use"]
        );
        set_provider_finish_reasons(&mut metadata, std::iter::empty::<String>());
        assert!(!metadata.contains_key(PROVIDER_FINISH_REASONS_METADATA_KEY));
        let fallbacks = [
            (FinishReason::Completed, "completed"),
            (FinishReason::ToolCall, "tool_call"),
            (FinishReason::MaxTokens, "max_tokens"),
            (FinishReason::Cancelled, "cancelled"),
            (FinishReason::Blocked, "blocked"),
            (FinishReason::Error, "error"),
            (FinishReason::Other("native".into()), "native"),
        ];
        for (reason, expected) in fallbacks {
            assert_eq!(
                provider_finish_reasons(&MetadataMap::new(), &reason),
                [expected]
            );
        }
    }

    #[test]
    fn input_is_newest_tail_output_is_head_and_data_refs_are_omitted() {
        let items = vec![
            Item::text(ItemKind::User, "old"),
            Item::text(ItemKind::User, "middle"),
            Item::text(ItemKind::User, "new"),
        ];
        let capture = MessageCapture::new(2, 10_000).unwrap();
        let input = capture_messages(&items, capture, CaptureOrder::NewestTail);
        assert!(input[0].contains("middle"));
        assert!(input[1].contains("new"));
        let output = capture_messages(&items, capture, CaptureOrder::OldestHead);
        assert!(output[0].contains("old"));
        assert!(output[1].contains("middle"));

        let media = Item::new(
            ItemKind::User,
            vec![Part::media(
                agentkit_core::Modality::Image,
                "image/png",
                agentkit_core::DataRef::uri("https://secret.invalid/image.png"),
            )],
        );
        let encoded = capture_item_json(&media, 10_000);
        assert!(!encoded.contains("secret.invalid"));
        assert!(encoded.contains("omitted"));
    }

    #[test]
    fn final_usage_wins_and_streamed_usage_fills_only_missing_fields() {
        let mut final_metadata = MetadataMap::new();
        final_metadata.insert("shared".into(), serde_json::json!("final"));
        let final_usage = Usage {
            tokens: Some(agentkit_core::TokenUsage::new(1, 2)),
            cost: None,
            metadata: final_metadata,
        };
        let mut streamed_metadata = MetadataMap::new();
        streamed_metadata.insert("shared".into(), serde_json::json!("streamed"));
        streamed_metadata.insert("stream_only".into(), serde_json::json!(true));
        let streamed_usage = Usage {
            tokens: Some(agentkit_core::TokenUsage::new(10, 20)),
            cost: Some(agentkit_core::CostUsage::new(0.5, "USD")),
            metadata: streamed_metadata,
        };
        let merged = merge_usage(Some(final_usage), Some(streamed_usage)).unwrap();
        assert_eq!(merged.tokens.unwrap().input_tokens, 1);
        assert_eq!(merged.cost.unwrap().amount, 0.5);
        assert_eq!(merged.metadata["shared"], "final");
        assert_eq!(merged.metadata["stream_only"], true);
    }

    #[test]
    fn session_provider_precedes_adapter_fallback() {
        assert_eq!(
            effective_provider_name(Some("session"), Some("adapter")).as_deref(),
            Some("session")
        );
        assert_eq!(
            effective_provider_name(None, Some("adapter")).as_deref(),
            Some("adapter")
        );
    }
}

#[cfg(all(test, feature = "otel"))]
mod true_otel_integration_tests {
    use std::fs;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn actual_layer_exports_driven_loop_spans() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp = std::env::temp_dir().join(format!(
            "agentkit-loop-true-otel-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&temp).unwrap();
        fs::create_dir(temp.join("src")).unwrap();
        let core_path = toml_path(&root.join("crates/agentkit-core"));
        let loop_path = toml_path(&root.join("crates/agentkit-loop"));
        // Pin the nested offline build to versions Cargo already fetched for
        // this workspace, without duplicating versions from Cargo.lock here.
        let async_trait_version = locked_version(root, "async-trait");
        let opentelemetry_version = locked_version(root, "opentelemetry");
        let serde_json_version = locked_version(root, "serde_json");
        let tokio_version = locked_version(root, "tokio");
        let tracing_version = locked_version(root, "tracing");
        let tracing_otel_version = locked_version(root, "tracing-opentelemetry");
        let tracing_subscriber_version = locked_version(root, "tracing-subscriber");
        let manifest = format!(
            r#"[package]
name = "agentkit-loop-true-otel-test"
version = "0.0.0"
edition = "2024"

[dependencies]
agentkit-core = {{ path = {core_path} }}
agentkit-loop = {{ path = {loop_path}, features = ["otel"] }}
async-trait = "={async_trait_version}"
opentelemetry = {{ version = "={opentelemetry_version}", default-features = false }}
serde_json = "={serde_json_version}"
tokio = {{ version = "={tokio_version}", features = ["rt"] }}
tracing = "={tracing_version}"
tracing-opentelemetry = {{ version = "={tracing_otel_version}", default-features = false }}
tracing-subscriber = "={tracing_subscriber_version}"
"#
        );
        fs::write(temp.join("Cargo.toml"), manifest).unwrap();
        fs::write(temp.join("src/main.rs"), TRUE_OTEL_HARNESS).unwrap();

        let output = Command::new(env!("CARGO"))
            .args(["run", "--quiet", "--offline"])
            .current_dir(&temp)
            .env("CARGO_TARGET_DIR", temp.join("target"))
            .output()
            .unwrap();
        let _ = fs::remove_dir_all(&temp);
        assert!(
            output.status.success(),
            "true OTEL harness failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn locked_version(root: &std::path::Path, name: &str) -> String {
        let lock = fs::read_to_string(root.join("Cargo.lock")).unwrap();
        let expected_name = format!("name = {}", serde_json::to_string(name).unwrap());
        let versions = lock
            .split("[[package]]")
            .filter(|package| {
                package
                    .lines()
                    .any(|line| line.trim() == expected_name.as_str())
            })
            .filter_map(|package| {
                package.lines().find_map(|line| {
                    line.trim()
                        .strip_prefix("version = \"")
                        .and_then(|version| version.strip_suffix('\"'))
                        .map(str::to_owned)
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(versions.len(), 1, "expected one locked version for {name}");
        versions.into_iter().next().unwrap()
    }

    fn toml_path(path: &std::path::Path) -> String {
        serde_json::to_string(&path.to_string_lossy()).unwrap()
    }

    #[test]
    fn toml_path_escapes_windows_separators_and_quotes() {
        let encoded = toml_path(std::path::Path::new(r#"C:\Users\name\quoted\"dir"#));
        assert_eq!(encoded, r#""C:\\Users\\name\\quoted\\\"dir""#);
    }

    const TRUE_OTEL_HARNESS: &str = r#"
use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use agentkit_core::{CostUsage, DataRef, FinishReason, Item, ItemKind, MetadataMap, Modality, Part, TokenUsage, TurnCancellation, Usage};
use agentkit_loop::{Agent, LoopError, LoopStep, MessageCapture, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult, SessionConfig, TelemetryConfig, TurnRequest, set_provider_finish_reasons};
use async_trait::async_trait;
use opentelemetry::trace::{Span, SpanBuilder, SpanContext, Status, Tracer};
use opentelemetry::{Array, Context, KeyValue, Value};
use tracing_subscriber::layer::SubscriberExt;

#[derive(Clone, Debug)]
struct Exported { name: String, attributes: Vec<KeyValue> }
#[derive(Clone, Default)]
struct MemoryTracer { exported: Arc<Mutex<Vec<Exported>>> }
struct MemorySpan { name: String, attributes: Vec<KeyValue>, exported: Arc<Mutex<Vec<Exported>>>, ended: bool }
impl Tracer for MemoryTracer {
    type Span = MemorySpan;
    fn build_with_context(&self, builder: SpanBuilder, _: &Context) -> MemorySpan {
        MemorySpan { name: builder.name.into_owned(), attributes: builder.attributes.unwrap_or_default(), exported: self.exported.clone(), ended: false }
    }
}
impl Span for MemorySpan {
    fn add_event_with_timestamp<T>(&mut self, _: T, _: SystemTime, _: Vec<KeyValue>) where T: Into<Cow<'static, str>> {}
    fn span_context(&self) -> &SpanContext { &SpanContext::NONE }
    fn is_recording(&self) -> bool { !self.ended }
    fn set_attribute(&mut self, attribute: KeyValue) { self.attributes.push(attribute); }
    fn set_status(&mut self, _: Status) {}
    fn update_name<T>(&mut self, name: T) where T: Into<Cow<'static, str>> { self.name = name.into().into_owned(); }
    fn add_link(&mut self, _: SpanContext, _: Vec<KeyValue>) {}
    fn end_with_timestamp(&mut self, _: SystemTime) {
        if !self.ended {
            self.ended = true;
            self.exported.lock().unwrap().push(Exported { name: self.name.clone(), attributes: self.attributes.clone() });
        }
    }
}

#[derive(Clone, Copy)]
enum BeginMode { Success, Error, Cancelled }
#[derive(Clone)]
struct ScriptedAdapter { adapter_provider: &'static str, final_usage: bool, overflow: bool, begin_mode: BeginMode }
struct ScriptedSession { selected_provider: Option<&'static str>, model: &'static str, final_usage: bool, overflow: bool, begin_mode: BeginMode }
struct ScriptedTurn { events: VecDeque<ModelTurnEvent> }
#[async_trait]
impl ModelAdapter for ScriptedAdapter {
    type Session = ScriptedSession;
    async fn start_session(&self, _: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(ScriptedSession { selected_provider: Some("before-begin"), model: "before-model", final_usage: self.final_usage, overflow: self.overflow, begin_mode: self.begin_mode })
    }
    fn provider_name(&self) -> Option<&str> { Some(self.adapter_provider) }
}
#[async_trait]
impl ModelSession for ScriptedSession {
    type Turn = ScriptedTurn;
    async fn begin_turn(&mut self, _: TurnRequest, _: Option<TurnCancellation>) -> Result<Self::Turn, LoopError> {
        match self.begin_mode {
            BeginMode::Error => return Err(LoopError::InvalidState("begin failed".into())),
            BeginMode::Cancelled => return Err(LoopError::Cancelled),
            BeginMode::Success => {}
        }
        self.selected_provider = if self.final_usage { Some("session-after-begin") } else { None };
        self.model = if self.final_usage { "model-after-begin" } else { "fallback-model-after-begin" };
        let mut stream_meta = MetadataMap::new();
        stream_meta.insert("stream_only".into(), serde_json::json!(true));
        stream_meta.insert("shared".into(), serde_json::json!("stream"));
        let streamed = Usage {
            tokens: Some(TokenUsage::new(if self.overflow { i64::MAX as u64 + 1 } else { 30 }, 40)),
            cost: Some(CostUsage::new(0.75, "USD")),
            metadata: stream_meta,
        };
        let final_usage = if self.final_usage {
            let mut metadata = MetadataMap::new();
            metadata.insert("shared".into(), serde_json::json!("final"));
            Some(Usage { tokens: Some(TokenUsage::new(if self.overflow { i64::MAX as u64 + 1 } else { 1 }, 2)), cost: None, metadata })
        } else { None };
        let mut result_metadata = MetadataMap::new();
        set_provider_finish_reasons(&mut result_metadata, ["native", "native", "done"]);
        let outputs = vec![
            Item::text(ItemKind::Assistant, "first"),
            Item::text(ItemKind::Assistant, "second"),
            Item::text(ItemKind::Assistant, "third"),
        ];
        Ok(ScriptedTurn { events: VecDeque::from([
            ModelTurnEvent::Usage(streamed),
            ModelTurnEvent::Finished(ModelTurnResult { finish_reason: FinishReason::Completed, output_items: outputs, usage: final_usage, metadata: result_metadata, model: Some(self.model.into()), response_id: Some("response-id".into()) }),
        ]) })
    }
    fn model_name(&self) -> Option<&str> { Some(self.model) }
    fn provider_name(&self) -> Option<&str> { self.selected_provider }
}
#[async_trait]
impl ModelTurn for ScriptedTurn {
    async fn next_event(&mut self, _: Option<TurnCancellation>) -> Result<Option<ModelTurnEvent>, LoopError> { Ok(self.events.pop_front()) }
}

fn attr<'a>(span: &'a Exported, key: &str) -> Option<&'a Value> {
    span.attributes.iter().rev().find(|a| a.key.as_str() == key).map(|a| &a.value)
}
fn operation(span: &Exported) -> Option<&str> {
    match attr(span, "gen_ai.operation.name") { Some(Value::String(value)) => Some(value.as_str()), _ => None }
}
fn json_array(span: &Exported, key: &str) -> Vec<serde_json::Value> {
    match attr(span, key) {
        Some(Value::Array(Array::String(values))) => values.iter().map(|v| serde_json::from_str(v.as_str()).unwrap()).collect(),
        other => panic!("{key} was not Array<String>: {other:?}"),
    }
}

fn run_attempt(adapter: ScriptedAdapter) -> (Vec<Exported>, Result<LoopStep, LoopError>) {
    let tracer = MemoryTracer::default();
    let subscriber = tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer.clone()));
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let result = tracing::subscriber::with_default(subscriber, || runtime.block_on(async {
        let media = Item::new(ItemKind::User, vec![
            Part::media(Modality::Image, "image/png", DataRef::inline_bytes([1, 2, 3])),
            Part::media(Modality::Audio, "audio/wav", DataRef::uri("https://secret.invalid/audio")),
        ]);
        let agent = Agent::builder().model(adapter).transcript(vec![
            Item::text(ItemKind::System, "old"), Item::text(ItemKind::User, "middle"), media,
        ]).input(vec![Item::text(ItemKind::User, "newest")]).telemetry(
            TelemetryConfig::default()
                .with_input_messages(MessageCapture::new(3, 100_000).unwrap())
                .with_output_messages(MessageCapture::new(2, 100_000).unwrap())
        ).build().unwrap();
        let mut driver = agent.start(SessionConfig::new("otel-test")).await.unwrap();
        driver.next().await
    }));
    (tracer.exported.lock().unwrap().clone(), result)
}
fn run(adapter: ScriptedAdapter) -> (Vec<Exported>, agentkit_loop::TurnResult) {
    let (spans, result) = run_attempt(adapter);
    let result = match result.unwrap() { LoopStep::Finished(result) => result, other => panic!("unexpected step: {other:?}") };
    (spans, result)
}

fn main() {
    let (spans, result) = run(ScriptedAdapter { adapter_provider: "adapter", final_usage: true, overflow: true, begin_mode: BeginMode::Success });
    let chat = spans.iter().find(|s| operation(s) == Some("chat")).unwrap();
    let agent = spans.iter().find(|s| operation(s) == Some("invoke_agent")).unwrap();
    assert_eq!(attr(chat, "gen_ai.provider.name"), Some(&Value::String("session-after-begin".into())));
    assert_eq!(attr(chat, "gen_ai.request.model"), Some(&Value::String("model-after-begin".into())));
    assert!(attr(chat, "gen_ai.usage.input_tokens").is_none());
    assert_eq!(attr(chat, "gen_ai.usage.output_tokens"), Some(&Value::I64(2)));
    assert_eq!(attr(chat, "gen_ai.usage.cost"), Some(&Value::F64(0.75)));
    assert_eq!(attr(agent, "gen_ai.provider.name"), Some(&Value::String("session-after-begin".into())));
    assert!(attr(agent, "gen_ai.usage.input_tokens").is_none());
    assert!(attr(agent, "gen_ai.usage.cost").is_none());
    match attr(chat, "gen_ai.response.finish_reasons") {
        Some(Value::Array(Array::String(values))) => assert_eq!(values.iter().map(|v| v.as_str()).collect::<Vec<_>>(), ["native", "done"]),
        other => panic!("finish reasons were not Array<String>: {other:?}"),
    }
    assert_eq!(result.usage.as_ref().unwrap().metadata["shared"], "final");
    assert_eq!(result.usage.as_ref().unwrap().metadata["stream_only"], true);
    let input = json_array(chat, "gen_ai.input.messages");
    assert_eq!(input.len(), 3);
    assert!(input[0].to_string().contains("middle"));
    assert!(input[1].to_string().contains("omitted"));
    assert!(input[2].to_string().contains("newest"));
    assert!(!input.iter().any(|message| message.to_string().contains("old")));
    let output = json_array(chat, "gen_ai.output.messages");
    assert_eq!(output.len(), 2);
    assert!(output[0].to_string().contains("first"));
    assert!(output[1].to_string().contains("second"));
    let encoded = format!("{input:?}{output:?}");
    assert!(!encoded.contains("secret.invalid"));
    assert!(!encoded.contains("[1,2,3]"));

    let (spans, result) = run(ScriptedAdapter { adapter_provider: "adapter-fallback", final_usage: false, overflow: false, begin_mode: BeginMode::Success });
    let chat = spans.iter().find(|s| operation(s) == Some("chat")).unwrap();
    assert_eq!(attr(chat, "gen_ai.provider.name"), Some(&Value::String("adapter-fallback".into())));
    assert_eq!(attr(chat, "gen_ai.request.model"), Some(&Value::String("fallback-model-after-begin".into())));
    assert_eq!(attr(chat, "gen_ai.usage.input_tokens"), Some(&Value::I64(30)));
    assert_eq!(attr(chat, "gen_ai.usage.cost"), Some(&Value::F64(0.75)));
    assert_eq!(result.usage.unwrap().metadata["stream_only"], true);

    for begin_mode in [BeginMode::Error, BeginMode::Cancelled] {
        let (spans, result) = run_attempt(ScriptedAdapter { adapter_provider: "adapter-before-error", final_usage: false, overflow: false, begin_mode });
        match begin_mode {
            BeginMode::Error => assert!(matches!(result, Err(LoopError::InvalidState(_)))),
            BeginMode::Cancelled => assert!(matches!(result, Ok(LoopStep::Finished(_)))),
            BeginMode::Success => unreachable!(),
        }
        let chat = spans.iter().find(|s| operation(s) == Some("chat")).unwrap();
        let agent = spans.iter().find(|s| operation(s) == Some("invoke_agent")).unwrap();
        assert_eq!(attr(chat, "gen_ai.provider.name"), Some(&Value::String("before-begin".into())));
        assert_eq!(attr(chat, "gen_ai.request.model"), Some(&Value::String("before-model".into())));
        assert_eq!(attr(agent, "gen_ai.provider.name"), Some(&Value::String("before-begin".into())));
    }
}
"#;
}

#[derive(Default)]
struct StreamedAssistantContent {
    parts: Vec<StreamedPart>,
    retained_bytes: usize,
    overflowed: bool,
}

enum StreamedPart {
    Open {
        id: PartId,
        kind: PartKind,
        text: String,
        bytes: Vec<u8>,
        structured: Option<Value>,
        structured_bytes: usize,
        metadata: MetadataMap,
        metadata_bytes: usize,
        retained_bytes: usize,
    },
    Committed {
        part: Part,
        retained_bytes: usize,
    },
}

impl StreamedAssistantContent {
    fn apply_delta(&mut self, delta: &Delta) {
        if self.overflowed {
            return;
        }

        match delta {
            Delta::BeginPart { part_id, kind } => {
                let index = self.open_part_index(part_id);
                let retained_bytes = part_id.0.len();
                if !self.reserve_slot(index, retained_bytes) {
                    return;
                }
                let open = StreamedPart::Open {
                    id: part_id.clone(),
                    kind: *kind,
                    text: String::new(),
                    bytes: Vec::new(),
                    structured: None,
                    structured_bytes: 0,
                    metadata: MetadataMap::new(),
                    metadata_bytes: 0,
                    retained_bytes,
                };
                if let Some(index) = index {
                    self.parts[index] = open;
                } else {
                    self.parts.push(open);
                }
            }
            Delta::AppendText { part_id, chunk } => {
                let Some(index) = self.open_part_index(part_id) else {
                    return;
                };
                if self.grow_slot(index, chunk.len())
                    && let StreamedPart::Open { text, .. } = &mut self.parts[index]
                {
                    text.push_str(chunk);
                }
            }
            Delta::AppendBytes { part_id, chunk } => {
                let Some(index) = self.open_part_index(part_id) else {
                    return;
                };
                if matches!(
                    self.parts[index],
                    StreamedPart::Open {
                        kind: PartKind::Media,
                        ..
                    }
                ) {
                    return;
                }
                if self.grow_slot(index, chunk.len())
                    && let StreamedPart::Open { bytes, .. } = &mut self.parts[index]
                {
                    bytes.extend_from_slice(chunk);
                }
            }
            Delta::ReplaceStructured { part_id, value } => {
                let Some(index) = self.open_part_index(part_id) else {
                    return;
                };
                let old_bytes = match &self.parts[index] {
                    StreamedPart::Open {
                        structured_bytes, ..
                    } => *structured_bytes,
                    StreamedPart::Committed { .. } => unreachable!(),
                };
                let limit = self.available_after_replacing(old_bytes);
                let Some(new_bytes) = serialized_size_with_limit(value, limit) else {
                    self.overflow();
                    return;
                };
                self.replace_slot_bytes(index, old_bytes, new_bytes);
                if let StreamedPart::Open {
                    structured,
                    structured_bytes,
                    ..
                } = &mut self.parts[index]
                {
                    *structured = Some(value.clone());
                    *structured_bytes = new_bytes;
                }
            }
            Delta::SetMetadata { part_id, metadata } => {
                let Some(index) = self.open_part_index(part_id) else {
                    return;
                };
                let old_bytes = match &self.parts[index] {
                    StreamedPart::Open { metadata_bytes, .. } => *metadata_bytes,
                    StreamedPart::Committed { .. } => unreachable!(),
                };
                let limit = self.available_after_replacing(old_bytes);
                let Some(new_bytes) = serialized_size_with_limit(metadata, limit) else {
                    self.overflow();
                    return;
                };
                self.replace_slot_bytes(index, old_bytes, new_bytes);
                if let StreamedPart::Open {
                    metadata: target,
                    metadata_bytes,
                    ..
                } = &mut self.parts[index]
                {
                    *target = metadata.clone();
                    *metadata_bytes = new_bytes;
                }
            }
            Delta::CommitPart { part } => self.commit_part(part),
        }
    }

    fn open_part_index(&self, id: &PartId) -> Option<usize> {
        self.parts.iter().position(
            |part| matches!(part, StreamedPart::Open { id: open_id, .. } if open_id == id),
        )
    }

    fn commit_part(&mut self, part: &Part) {
        if self.overflowed {
            return;
        }

        let duplicate_tool_call = if let Part::ToolCall(call) = part {
            self.parts.iter().position(|slot| {
                matches!(
                    slot,
                    StreamedPart::Committed {
                        part: Part::ToolCall(existing),
                        ..
                    } if existing.id == call.id
                )
            })
        } else {
            None
        };

        let kind = part_kind(part);
        let matching_open = self
            .parts
            .iter()
            .position(|slot| slot.open_matches_part(part));
        let mut open_with_kind = self.parts.iter().enumerate().filter_map(|(index, slot)| {
            matches!(slot, StreamedPart::Open { kind: open_kind, .. } if *open_kind == kind)
                .then_some(index)
        });
        let only_open_with_kind = match (open_with_kind.next(), open_with_kind.next()) {
            (Some(index), None) => Some(index),
            _ => None,
        };
        let index = duplicate_tool_call
            .or(matching_open)
            .or(only_open_with_kind);
        let limit = self.available_for_slot(index);
        let Some(retained_bytes) = serialized_size_with_limit(part, limit) else {
            self.overflow();
            return;
        };
        if !self.reserve_slot(index, retained_bytes) {
            return;
        }
        let committed = StreamedPart::Committed {
            part: part.clone(),
            retained_bytes,
        };
        if let Some(index) = index {
            self.parts[index] = committed;
        } else {
            self.parts.push(committed);
        }
    }

    fn commit_tool_call(&mut self, call: &ToolCallPart) {
        if self.overflowed {
            return;
        }
        let duplicate_tool_call = self.parts.iter().position(|slot| {
            matches!(
                slot,
                StreamedPart::Committed {
                    part: Part::ToolCall(existing),
                    ..
                } if existing.id == call.id
            )
        });
        let mut open_tool_calls = self.parts.iter().enumerate().filter_map(|(index, slot)| {
            matches!(
                slot,
                StreamedPart::Open {
                    kind: PartKind::ToolCall,
                    ..
                }
            )
            .then_some(index)
        });
        let only_open_tool_call = match (open_tool_calls.next(), open_tool_calls.next()) {
            (Some(index), None) => Some(index),
            _ => None,
        };
        let index = duplicate_tool_call.or(only_open_tool_call);
        let limit = self.available_for_slot(index);
        let borrowed_part = BorrowedPart::ToolCall(call);
        let Some(retained_bytes) = serialized_size_with_limit(&borrowed_part, limit) else {
            self.overflow();
            return;
        };
        if !self.reserve_slot(index, retained_bytes) {
            return;
        }
        let committed = StreamedPart::Committed {
            part: Part::ToolCall(call.clone()),
            retained_bytes,
        };
        if let Some(index) = index {
            self.parts[index] = committed;
        } else {
            self.parts.push(committed);
        }
    }

    fn reserve_slot(&mut self, index: Option<usize>, retained_bytes: usize) -> bool {
        if index.is_none() && self.parts.len() >= MAX_STREAMED_ASSISTANT_CONTENT_PARTS {
            self.overflow();
            return false;
        }
        let replaced_bytes = index.map_or(0, |index| self.parts[index].retained_bytes());
        let Some(total) = self
            .retained_bytes
            .checked_sub(replaced_bytes)
            .and_then(|bytes| bytes.checked_add(retained_bytes))
            .filter(|bytes| *bytes <= MAX_STREAMED_ASSISTANT_CONTENT_BYTES)
        else {
            self.overflow();
            return false;
        };
        self.retained_bytes = total;
        true
    }

    fn grow_slot(&mut self, index: usize, additional_bytes: usize) -> bool {
        let Some(total) = self
            .retained_bytes
            .checked_add(additional_bytes)
            .filter(|bytes| *bytes <= MAX_STREAMED_ASSISTANT_CONTENT_BYTES)
        else {
            self.overflow();
            return false;
        };
        self.retained_bytes = total;
        *self.parts[index].retained_bytes_mut() += additional_bytes;
        true
    }

    fn available_after_replacing(&self, replaced_bytes: usize) -> usize {
        MAX_STREAMED_ASSISTANT_CONTENT_BYTES - (self.retained_bytes - replaced_bytes)
    }

    fn available_for_slot(&self, index: Option<usize>) -> usize {
        self.available_after_replacing(index.map_or(0, |index| self.parts[index].retained_bytes()))
    }

    fn replace_slot_bytes(&mut self, index: usize, old_bytes: usize, new_bytes: usize) {
        self.retained_bytes = self.retained_bytes - old_bytes + new_bytes;
        let retained_bytes = self.parts[index].retained_bytes_mut();
        *retained_bytes = *retained_bytes - old_bytes + new_bytes;
    }

    fn overflow(&mut self) {
        self.parts.clear();
        self.retained_bytes = 0;
        self.overflowed = true;
    }

    fn reset(&mut self) {
        self.parts.clear();
        self.retained_bytes = 0;
        self.overflowed = false;
    }

    fn interrupted_items(&self) -> Vec<Item> {
        if self.overflowed {
            return interrupted_assistant_items();
        }
        let parts = self
            .parts
            .iter()
            .filter_map(StreamedPart::preserved_part)
            .collect::<Vec<_>>();
        if parts.is_empty() {
            return interrupted_assistant_items();
        }
        vec![Item::new(ItemKind::Assistant, parts).with_metadata(interrupted_metadata("assistant"))]
    }
}

impl StreamedPart {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Open { retained_bytes, .. } | Self::Committed { retained_bytes, .. } => {
                *retained_bytes
            }
        }
    }

    fn retained_bytes_mut(&mut self) -> &mut usize {
        match self {
            Self::Open { retained_bytes, .. } | Self::Committed { retained_bytes, .. } => {
                retained_bytes
            }
        }
    }

    fn open_matches_part(&self, committed: &Part) -> bool {
        match (self, committed) {
            (
                Self::Open {
                    kind: PartKind::Text,
                    text,
                    ..
                },
                Part::Text(part),
            ) => text == &part.text,
            (
                Self::Open {
                    kind: PartKind::Reasoning,
                    text,
                    bytes,
                    ..
                },
                Part::Reasoning(part),
            ) => {
                part.summary.as_deref() == (!text.is_empty()).then_some(text.as_str())
                    && match &part.data {
                        Some(DataRef::InlineBytes(data)) => data == bytes,
                        None => bytes.is_empty(),
                        _ => false,
                    }
            }
            (
                Self::Open {
                    kind: PartKind::Structured,
                    structured: Some(value),
                    ..
                },
                Part::Structured(part),
            ) => value == &part.value,
            _ => false,
        }
    }

    fn preserved_part(&self) -> Option<Part> {
        match self {
            Self::Committed { part, .. } if useful_streamed_part(part) => Some(part.clone()),
            Self::Committed { .. } => None,
            Self::Open {
                kind,
                text,
                bytes,
                structured,
                metadata,
                ..
            } => match kind {
                PartKind::Text if !text.is_empty() => Some(Part::Text(
                    TextPart::new(text).with_metadata(metadata.clone()),
                )),
                PartKind::Reasoning if !text.is_empty() || !bytes.is_empty() => {
                    Some(Part::Reasoning(ReasoningPart {
                        summary: (!text.is_empty()).then(|| text.clone()),
                        data: (!bytes.is_empty()).then(|| DataRef::inline_bytes(bytes.clone())),
                        redacted: false,
                        metadata: metadata.clone(),
                    }))
                }
                PartKind::Structured => structured.clone().map(|value| {
                    Part::Structured(StructuredPart::new(value).with_metadata(metadata.clone()))
                }),
                // BeginPart does not carry media modality or MIME type, so an
                // uncommitted media part cannot be reconstructed faithfully.
                PartKind::Media => None,
                _ => None,
            },
        }
    }
}

#[derive(Serialize)]
enum BorrowedPart<'a> {
    ToolCall(&'a ToolCallPart),
}

struct PayloadSizeWriter {
    remaining: usize,
    written: usize,
}

impl std::io::Write for PayloadSizeWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(std::io::Error::other(
                "streamed assistant payload limit exceeded",
            ));
        }
        self.remaining -= bytes.len();
        self.written += bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialized_size_with_limit<T: Serialize + ?Sized>(value: &T, limit: usize) -> Option<usize> {
    let mut writer = PayloadSizeWriter {
        remaining: limit,
        written: 0,
    };
    serde_json::to_writer(&mut writer, value).ok()?;
    Some(writer.written)
}

fn part_kind(part: &Part) -> PartKind {
    match part {
        Part::Text(_) => PartKind::Text,
        Part::Media(_) => PartKind::Media,
        Part::File(_) => PartKind::File,
        Part::Structured(_) => PartKind::Structured,
        Part::Reasoning(_) => PartKind::Reasoning,
        Part::ToolCall(_) => PartKind::ToolCall,
        Part::ToolResult(_) => PartKind::ToolResult,
        Part::Custom(_) => PartKind::Custom,
    }
}

fn useful_streamed_part(part: &Part) -> bool {
    match part {
        Part::Text(text) => !text.text.is_empty(),
        Part::Reasoning(reasoning) => {
            reasoning
                .summary
                .as_ref()
                .is_some_and(|text| !text.is_empty())
                || reasoning.data.is_some()
                || reasoning.redacted
        }
        Part::ToolResult(_) => false,
        _ => true,
    }
}

fn interrupted_stream_items(content: Option<&StreamedAssistantContent>) -> Vec<Item> {
    content.map_or_else(
        interrupted_assistant_items,
        StreamedAssistantContent::interrupted_items,
    )
}

fn interrupted_assistant_items() -> Vec<Item> {
    vec![
        Item::new(
            ItemKind::Assistant,
            vec![Part::Text(TextPart {
                text: "Previous assistant response was interrupted by the user before completion."
                    .into(),
                metadata: interrupted_metadata("assistant"),
            })],
        )
        .with_metadata(interrupted_metadata("assistant")),
    ]
}

/// Tool calls in `transcript` that no `tool_result` answers, in call order.
fn unanswered_tool_calls(transcript: &[Item]) -> Vec<ToolCallPart> {
    let mut open: Vec<ToolCallPart> = Vec::new();
    for part in transcript.iter().flat_map(|item| &item.parts) {
        match part {
            Part::ToolCall(call) => open.push(call.clone()),
            Part::ToolResult(result) => open.retain(|call| call.id != result.call_id),
            _ => {}
        }
    }
    open
}

/// The result recorded for a call the cancelled turn abandoned.
///
/// It is an error result: the call produced no output, and the work it started
/// may or may not have run to completion.
fn interrupted_tool_result_item(call: ToolCallPart) -> Item {
    Item {
        id: None,
        kind: ItemKind::Tool,
        parts: vec![Part::ToolResult(ToolResultPart {
            call_id: call.id,
            output: ToolOutput::Text("tool call interrupted before it reported a result".into()),
            is_error: true,
            metadata: interrupted_metadata("tool"),
        })],
        metadata: interrupted_metadata("tool"),
        usage: None,
        finish_reason: None,
        created_at: None,
    }
}

fn cancelled_approval_item(pending: PendingApprovalToolCall) -> Item {
    Item {
        id: None,
        kind: ItemKind::Tool,
        parts: vec![Part::ToolResult(ToolResultPart {
            call_id: pending.call.id,
            output: ToolOutput::Text("approval cancelled".into()),
            is_error: true,
            metadata: pending.call.metadata,
        })],
        metadata: MetadataMap::new(),
        usage: None,
        finish_reason: None,
        created_at: None,
    }
}

/// Whether the transcript ends in something the model should respond to.
///
/// Only input-bearing trailing roles should drive inference. Passive transcript
/// state (`System`, `Developer`, `Context`), an assistant tail, or an empty
/// transcript has nothing new for the model to respond to.
fn transcript_has_pending_input(transcript: &[Item]) -> bool {
    matches!(
        transcript.last().map(|item| item.kind),
        Some(ItemKind::User | ItemKind::Tool | ItemKind::Notification)
    )
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

fn tool_result_is_error(item: &Item) -> bool {
    item.parts
        .iter()
        .any(|part| matches!(part, Part::ToolResult(result) if result.is_error))
}

fn tool_result_not_started(item: &Item) -> bool {
    item.parts.iter().any(|part| {
        matches!(
            part,
            Part::ToolResult(result)
                if result
                    .metadata
                    .get(TOOL_RESULT_NOT_STARTED_METADATA_KEY)
                    .and_then(Value::as_bool)
                    == Some(true)
        )
    })
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
    /// An error reported by a [`LoopMutator`] (compaction, redaction, repair).
    #[error("mutator error: {0}")]
    Mutator(String),
    /// The requested operation is not supported.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

/// Internal [`EventEmitter`] backed by the driver's observer slice. Lives
/// only for the duration of a [`LoopDriver::run_mutators`] call so the
/// borrow against `self.observers` stays disjoint from the cursor's borrow
/// of `self.transcript`.
struct DriverEmitter<'a> {
    session_id: &'a Arc<SessionId>,
    observers: &'a [Arc<dyn LoopObserver>],
}

impl<'a> EventEmitter for DriverEmitter<'a> {
    fn emit(&self, event: AgentEvent) {
        fan_out_observed_event(self.observers, self.session_id, event);
    }
}

fn fan_out_observed_event(
    observers: &[Arc<dyn LoopObserver>],
    session_id: &Arc<SessionId>,
    event: AgentEvent,
) {
    if observers.is_empty() {
        return;
    }
    let observed = ObservedEvent {
        session_id: Arc::clone(session_id),
        event,
    };
    let last = observers.len() - 1;
    for observer in &observers[..last] {
        observer.handle_event(observed.clone());
    }
    observers[last].handle_event(observed);
}

/// Hard-fails when a mutator's edit leaves the transcript protocol-invalid.
/// The only invariant currently checked is tool_use ↔ tool_result pairing
/// — every [`Part::ToolCall`] must be followed (in transcript order) by a
/// matching [`Part::ToolResult`] with the same `call_id`.
fn validate_transcript_invariants(transcript: &[Item]) -> Result<(), LoopError> {
    let mut pending: HashSet<ToolCallId> = HashSet::new();
    let mut seen_calls: HashSet<ToolCallId> = HashSet::new();
    let mut seen_results: HashSet<ToolCallId> = HashSet::new();
    for item in transcript {
        for part in &item.parts {
            match part {
                Part::ToolCall(call) => {
                    if !seen_calls.insert(call.id.clone()) {
                        return Err(LoopError::Mutator(format!(
                            "transcript invariant violation: duplicate tool_use: {}",
                            call.id.0
                        )));
                    }
                    pending.insert(call.id.clone());
                }
                Part::ToolResult(result) => {
                    if !pending.remove(&result.call_id) {
                        let kind = if seen_results.contains(&result.call_id) {
                            "duplicate"
                        } else {
                            "orphaned"
                        };
                        return Err(LoopError::Mutator(format!(
                            "transcript invariant violation: {kind} tool_result: {}",
                            result.call_id.0
                        )));
                    }
                    seen_results.insert(result.call_id.clone());
                }
                _ => {}
            }
        }
    }
    if !pending.is_empty() {
        let missing: Vec<String> = pending.into_iter().map(|id| id.0).collect();
        return Err(LoopError::Mutator(format!(
            "transcript invariant violation: tool_use(s) without matching tool_result: {}",
            missing.join(", ")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc as StdArc, Mutex as StdMutex};

    use agentkit_core::{
        CancellationController, ItemKind, Part, TextPart, ToolCallId, ToolCallPart, ToolOutput,
        ToolResultPart,
    };
    use agentkit_task_manager::{
        AsyncTaskManager, RoutingDecision, TaskEvent, TaskManager, TaskManagerError,
        TaskManagerHandle, TaskRoutingPolicy,
    };
    use agentkit_tools_core::{
        FileSystemPermissionRequest, PermissionCode, PermissionDecision, PermissionDenial, Tool,
        ToolAnnotations, ToolCatalogEvent, ToolExecutionOutcome, ToolName, ToolRegistry,
        ToolResult, ToolSpec,
    };
    use serde_json::{Value, json};
    use tokio::sync::Notify;
    use tokio::time::{Duration, timeout};

    use super::*;

    struct FakeAdapter;
    struct SupersedingAdapter;
    struct InterruptedStreamAdapter {
        controller: CancellationController,
        scenario: InterruptedStreamScenario,
    }
    struct SlowAdapter;
    struct RecordingAdapter {
        seen_descriptions: StdArc<StdMutex<Vec<Vec<String>>>>,
        seen_caches: StdArc<StdMutex<Vec<Option<PromptCacheRequest>>>>,
    }
    struct MultiToolAdapter;
    struct DualApprovalAdapter;

    struct FakeSession;
    struct SupersedingSession {
        supersession_enabled: bool,
    }
    struct InterruptedStreamSession {
        controller: CancellationController,
        scenario: InterruptedStreamScenario,
    }
    struct SlowSession;
    struct RecordingSession {
        seen_descriptions: StdArc<StdMutex<Vec<Vec<String>>>>,
        seen_caches: StdArc<StdMutex<Vec<Option<PromptCacheRequest>>>>,
    }
    struct MultiToolSession;
    struct DualApprovalSession;

    struct FakeTurn {
        events: VecDeque<ModelTurnEvent>,
    }

    struct SupersedingTurn {
        events: VecDeque<ModelTurnEvent>,
    }

    #[derive(Clone, Copy)]
    enum InterruptedStreamScenario {
        PreserveContent,
        CancelOnSupersession,
        OverflowThenSupersession,
    }

    struct InterruptedStreamTurn {
        controller: CancellationController,
        events: VecDeque<ModelTurnEvent>,
        cancel_on_supersession: bool,
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
    struct DualApprovalTurn {
        events: VecDeque<ModelTurnEvent>,
    }

    struct TestTaskManager<T> {
        inner: T,
        start_error: Option<&'static str>,
        approved_start_error: Option<&'static str>,
        pending_update_error: Option<(usize, &'static str)>,
        pending_update_calls: AtomicUsize,
        interrupted: Option<StdArc<StdMutex<Vec<agentkit_core::TurnId>>>>,
        interrupt_error: Option<&'static str>,
    }

    impl<T> TestTaskManager<T> {
        fn new(inner: T) -> Self {
            Self {
                inner,
                start_error: None,
                approved_start_error: None,
                pending_update_error: None,
                pending_update_calls: AtomicUsize::new(0),
                interrupted: None,
                interrupt_error: None,
            }
        }

        fn fail_start(mut self, message: &'static str) -> Self {
            self.start_error = Some(message);
            self
        }

        fn fail_approved_start(mut self, message: &'static str) -> Self {
            self.approved_start_error = Some(message);
            self
        }

        fn fail_pending_update_on(mut self, call: usize, message: &'static str) -> Self {
            self.pending_update_error = Some((call, message));
            self
        }

        fn record_interrupts(
            mut self,
            interrupted: StdArc<StdMutex<Vec<agentkit_core::TurnId>>>,
        ) -> Self {
            self.interrupted = Some(interrupted);
            self
        }

        fn fail_interrupt(mut self, message: &'static str) -> Self {
            self.interrupt_error = Some(message);
            self
        }
    }

    #[async_trait]
    impl<T: TaskManager> TaskManager for TestTaskManager<T> {
        async fn start_task(
            &self,
            request: TaskLaunchRequest,
            ctx: TaskStartContext,
        ) -> Result<TaskStartOutcome, TaskManagerError> {
            if let Some(message) = self.start_error.or_else(|| {
                matches!(&request.kind, TaskLaunchKind::Approved(_))
                    .then_some(self.approved_start_error)
                    .flatten()
            }) {
                return Err(TaskManagerError::Internal(message.into()));
            }
            self.inner.start_task(request, ctx).await
        }

        async fn wait_for_turn(
            &self,
            turn_id: &agentkit_core::TurnId,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Option<TurnTaskUpdate>, TaskManagerError> {
            self.inner.wait_for_turn(turn_id, cancellation).await
        }

        async fn take_pending_loop_updates(&self) -> Result<PendingLoopUpdates, TaskManagerError> {
            if let Some((call, message)) = self.pending_update_error
                && self.pending_update_calls.fetch_add(1, Ordering::SeqCst) == call
            {
                return Err(TaskManagerError::Internal(message.into()));
            }
            self.inner.take_pending_loop_updates().await
        }

        async fn on_turn_interrupted(
            &self,
            turn_id: &agentkit_core::TurnId,
        ) -> Result<(), TaskManagerError> {
            if let Some(interrupted) = &self.interrupted {
                interrupted.lock().unwrap().push(turn_id.clone());
            }
            if let Some(message) = self.interrupt_error {
                return Err(TaskManagerError::Internal(message.into()));
            }
            self.inner.on_turn_interrupted(turn_id).await
        }

        fn handle(&self) -> TaskManagerHandle {
            self.inner.handle()
        }
    }

    struct DelayedApprovalExecutor {
        entered: StdArc<AtomicBool>,
        release: StdArc<Notify>,
        approved_entered: Option<StdArc<AtomicBool>>,
        approved_release: Option<StdArc<Notify>>,
        cancellation: Option<CancellationController>,
        spec: ToolSpec,
    }

    impl DelayedApprovalExecutor {
        fn new(entered: StdArc<AtomicBool>, release: StdArc<Notify>) -> Self {
            Self {
                entered,
                release,
                approved_entered: None,
                approved_release: None,
                cancellation: None,
                spec: ToolSpec {
                    name: ToolName::new("echo"),
                    description: "delayed approval".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "value": { "type": "string" }
                        },
                        "required": ["value"],
                        "additionalProperties": false
                    }),
                    output_schema: None,
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }

        fn cancelling_on_approval(mut self, controller: CancellationController) -> Self {
            self.cancellation = Some(controller);
            self
        }

        fn blocking_after_approval(
            mut self,
            entered: StdArc<AtomicBool>,
            release: StdArc<Notify>,
        ) -> Self {
            self.approved_entered = Some(entered);
            self.approved_release = Some(release);
            self
        }
    }

    #[async_trait]
    impl ToolExecutor for DelayedApprovalExecutor {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![self.spec.clone()]
        }

        async fn execute(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> ToolExecutionOutcome {
            self.entered.store(true, Ordering::SeqCst);
            self.release.notified().await;
            if let Some(controller) = &self.cancellation {
                controller.interrupt();
            }
            ToolExecutionOutcome::Interrupted(
                agentkit_tools_core::ToolInterruption::ApprovalRequired(ApprovalRequest {
                    task_id: None,
                    call_id: None,
                    id: "approval:delayed".into(),
                    request_kind: "delayed.approval".into(),
                    reason: agentkit_tools_core::ApprovalReason::PolicyRequiresConfirmation,
                    summary: "delayed approval".into(),
                    metadata: request.metadata,
                }),
            )
        }

        async fn execute_approved(
            &self,
            request: ToolRequest,
            approved_request: &ApprovalRequest,
            ctx: &mut ToolContext<'_>,
        ) -> ToolExecutionOutcome {
            let (Some(entered), Some(release)) = (&self.approved_entered, &self.approved_release)
            else {
                return self.execute(request, ctx).await;
            };
            let _ = approved_request;
            entered.store(true, Ordering::SeqCst);
            release.notified().await;
            ToolExecutionOutcome::Completed(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text("approved-ok".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    #[async_trait]
    impl ModelAdapter for FakeAdapter {
        type Session = FakeSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(FakeSession)
        }
    }

    #[async_trait]
    impl ModelAdapter for SupersedingAdapter {
        type Session = SupersedingSession;

        async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(SupersedingSession {
                supersession_enabled: config.consumer_capabilities.response_attempt_supersession,
            })
        }
    }

    #[async_trait]
    impl ModelAdapter for InterruptedStreamAdapter {
        type Session = InterruptedStreamSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(InterruptedStreamSession {
                controller: self.controller.clone(),
                scenario: self.scenario,
            })
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
                seen_caches: self.seen_caches.clone(),
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
    impl ModelAdapter for DualApprovalAdapter {
        type Session = DualApprovalSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(DualApprovalSession)
        }
    }

    #[async_trait]
    impl ModelSession for SupersedingSession {
        type Turn = SupersedingTurn;

        async fn begin_turn(
            &mut self,
            _request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            assert!(self.supersession_enabled);
            Ok(SupersedingTurn {
                events: VecDeque::from([
                    ModelTurnEvent::Usage(Usage::default()),
                    ModelTurnEvent::ToolCall(ToolCallPart::new(
                        "discarded-call",
                        "discarded-tool",
                        json!({}),
                    )),
                    ModelTurnEvent::ResponseAttemptSuperseded,
                    ModelTurnEvent::Finished(ModelTurnResult {
                        finish_reason: FinishReason::Completed,
                        output_items: vec![Item::text(ItemKind::Assistant, "replacement")],
                        usage: None,
                        metadata: MetadataMap::new(),
                        model: None,
                        response_id: None,
                    }),
                ]),
            })
        }
    }

    #[async_trait]
    impl ModelSession for InterruptedStreamSession {
        type Turn = InterruptedStreamTurn;

        async fn begin_turn(
            &mut self,
            _request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let events = match self.scenario {
                InterruptedStreamScenario::PreserveContent => {
                    let mut reasoning_metadata = MetadataMap::new();
                    reasoning_metadata.insert("provider.detail".into(), true.into());
                    VecDeque::from([
                        ModelTurnEvent::Delta(Delta::BeginPart {
                            part_id: PartId::new("text"),
                            kind: PartKind::Text,
                        }),
                        ModelTurnEvent::Delta(Delta::AppendText {
                            part_id: PartId::new("text"),
                            chunk: "partial answer".into(),
                        }),
                        ModelTurnEvent::Delta(Delta::CommitPart {
                            part: Part::text("partial answer"),
                        }),
                        ModelTurnEvent::Delta(Delta::BeginPart {
                            part_id: PartId::new("reasoning"),
                            kind: PartKind::Reasoning,
                        }),
                        ModelTurnEvent::Delta(Delta::AppendText {
                            part_id: PartId::new("reasoning"),
                            chunk: "partial thought".into(),
                        }),
                        ModelTurnEvent::Delta(Delta::SetMetadata {
                            part_id: PartId::new("reasoning"),
                            metadata: reasoning_metadata,
                        }),
                        ModelTurnEvent::Delta(Delta::BeginPart {
                            part_id: PartId::new("structured"),
                            kind: PartKind::Structured,
                        }),
                        ModelTurnEvent::Delta(Delta::ReplaceStructured {
                            part_id: PartId::new("structured"),
                            value: json!({ "complete": false }),
                        }),
                        ModelTurnEvent::Delta(Delta::BeginPart {
                            part_id: PartId::new("media"),
                            kind: PartKind::Media,
                        }),
                        ModelTurnEvent::Delta(Delta::AppendBytes {
                            part_id: PartId::new("media"),
                            chunk: vec![1, 2, 3],
                        }),
                        ModelTurnEvent::ToolCall(ToolCallPart::new(
                            "partial-call",
                            "unfinished-tool",
                            json!({}),
                        )),
                    ])
                }
                InterruptedStreamScenario::CancelOnSupersession => VecDeque::from([
                    ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: PartId::new("stale"),
                        kind: PartKind::Text,
                    }),
                    ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: PartId::new("stale"),
                        chunk: "discard me".into(),
                    }),
                    ModelTurnEvent::ResponseAttemptSuperseded,
                ]),
                InterruptedStreamScenario::OverflowThenSupersession => VecDeque::from([
                    ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: PartId::new("oversized"),
                        kind: PartKind::Text,
                    }),
                    ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: PartId::new("oversized"),
                        chunk: "x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES),
                    }),
                    ModelTurnEvent::ResponseAttemptSuperseded,
                    ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: PartId::new("fresh"),
                        kind: PartKind::Text,
                    }),
                    ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: PartId::new("fresh"),
                        chunk: "preserve me".into(),
                    }),
                ]),
            };
            Ok(InterruptedStreamTurn {
                controller: self.controller.clone(),
                events,
                cancel_on_supersession: matches!(
                    self.scenario,
                    InterruptedStreamScenario::CancelOnSupersession
                ),
            })
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
                        item.parts.iter().find_map(|part| match (item.kind, part) {
                            (ItemKind::Notification, Part::Text(text)) => Some(text.text.clone()),
                            (
                                _,
                                Part::ToolResult(ToolResultPart {
                                    output: ToolOutput::Text(text),
                                    ..
                                }),
                            ) => Some(text.clone()),
                            _ => None,
                        })
                    })
                    .unwrap_or_else(|| "missing".into());

                VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: format!("tool said: {result_text}"),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                        usage: None,
                        finish_reason: None,
                        created_at: None,
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
                        model: None,
                        response_id: None,
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
                            usage: None,
                            finish_reason: None,
                            created_at: None,
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
            self.seen_caches.lock().unwrap().push(request.cache.clone());

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
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "mixed tools finished".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                        usage: None,
                        finish_reason: None,
                        created_at: None,
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
                        model: None,
                        response_id: None,
                        finish_reason: FinishReason::ToolCall,
                        output_items: vec![Item {
                            id: None,
                            kind: ItemKind::Assistant,
                            parts: vec![Part::ToolCall(foreground), Part::ToolCall(background)],
                            metadata: MetadataMap::new(),
                            usage: None,
                            finish_reason: None,
                            created_at: None,
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
    impl ModelSession for DualApprovalSession {
        type Turn = DualApprovalTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let tool_results = request
                .transcript
                .iter()
                .flat_map(|item| item.parts.iter())
                .filter(|part| matches!(part, Part::ToolResult(_)))
                .count();

            let events = if tool_results >= 2 {
                VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "both approvals finished".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                        usage: None,
                        finish_reason: None,
                        created_at: None,
                    }],
                    usage: None,
                    metadata: MetadataMap::new(),
                })])
            } else {
                let first = agentkit_core::ToolCallPart {
                    id: ToolCallId::new("call-1"),
                    name: "echo".into(),
                    input: json!({ "value": "first" }),
                    metadata: MetadataMap::new(),
                };
                let second = agentkit_core::ToolCallPart {
                    id: ToolCallId::new("call-2"),
                    name: "echo".into(),
                    input: json!({ "value": "second" }),
                    metadata: MetadataMap::new(),
                };
                VecDeque::from([
                    ModelTurnEvent::ToolCall(first.clone()),
                    ModelTurnEvent::ToolCall(second.clone()),
                    ModelTurnEvent::Finished(ModelTurnResult {
                        model: None,
                        response_id: None,
                        finish_reason: FinishReason::ToolCall,
                        output_items: vec![Item {
                            id: None,
                            kind: ItemKind::Assistant,
                            parts: vec![Part::ToolCall(first), Part::ToolCall(second)],
                            metadata: MetadataMap::new(),
                            usage: None,
                            finish_reason: None,
                            created_at: None,
                        }],
                        usage: None,
                        metadata: MetadataMap::new(),
                    }),
                ])
            };

            Ok(DualApprovalTurn { events })
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
    impl ModelTurn for SupersedingTurn {
        async fn next_event(
            &mut self,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            Ok(self.events.pop_front())
        }
    }

    #[async_trait]
    impl ModelTurn for InterruptedStreamTurn {
        async fn next_event(
            &mut self,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if let Some(event) = self.events.pop_front() {
                if self.cancel_on_supersession
                    && matches!(event, ModelTurnEvent::ResponseAttemptSuperseded)
                {
                    self.controller.interrupt();
                }
                return Ok(Some(event));
            }
            self.controller.interrupt();
            Err(LoopError::Cancelled)
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
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "done".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                        usage: None,
                        finish_reason: None,
                        created_at: None,
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
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: "done".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                        usage: None,
                        finish_reason: None,
                        created_at: None,
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

    #[async_trait]
    impl ModelTurn for DualApprovalTurn {
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

    #[derive(Clone)]
    struct FailingTool {
        spec: ToolSpec,
    }

    #[derive(Clone)]
    struct RunThenDenyTool {
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
                    output_schema: None,
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    impl Default for FailingTool {
        fn default() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("failing"),
                    description: "Always fails after execution starts".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "value": { "type": "string" }
                        },
                        "additionalProperties": true
                    }),
                    output_schema: None,
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    impl Default for RunThenDenyTool {
        fn default() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("run_then_deny"),
                    description: "Runs, then returns a permission-denied error".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "value": { "type": "string" }
                        },
                        "additionalProperties": true
                    }),
                    output_schema: None,
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
                    output_schema: None,
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
    impl Tool for FailingTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            _request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            Err(agentkit_tools_core::ToolError::ExecutionFailed(
                "runtime failed".into(),
            ))
        }
    }

    #[async_trait]
    impl Tool for RunThenDenyTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            _request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            Err(agentkit_tools_core::ToolError::PermissionDenied(
                PermissionDenial {
                    code: PermissionCode::CustomPolicyDenied,
                    message: "remote 403".into(),
                    metadata: MetadataMap::new(),
                },
            ))
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
                    call_id: None,
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

    struct KeepRecentMutator {
        keep: usize,
    }

    #[async_trait]
    impl LoopMutator for KeepRecentMutator {
        async fn mutate(
            &self,
            cursor: &mut TranscriptCursor<'_>,
            ctx: LoopCtx<'_>,
        ) -> Result<(), LoopError> {
            if cursor.len() < 2 {
                return Ok(());
            }
            let drop = cursor.len().saturating_sub(self.keep);
            ctx.emitter.emit(AgentEvent::MutationStarted {
                session_id: ctx.session_id.clone(),
                turn_id: ctx.turn_id.cloned(),
                mutator: "keep-recent".into(),
                point: ctx.point,
            });
            cursor.drain(..drop);
            ctx.emitter.emit(AgentEvent::MutationFinished {
                session_id: ctx.session_id.clone(),
                turn_id: ctx.turn_id.cloned(),
                mutator: "keep-recent".into(),
                dirty: true,
                metadata: MetadataMap::new(),
            });
            Ok(())
        }
    }

    /// No-op mutator that records the [`MutationPoint`] it is invoked with at
    /// each mutation site, so a test can assert which point the loop reports.
    struct PointRecordingMutator {
        points: StdArc<StdMutex<Vec<MutationPoint>>>,
    }

    #[async_trait]
    impl LoopMutator for PointRecordingMutator {
        async fn mutate(
            &self,
            _cursor: &mut TranscriptCursor<'_>,
            ctx: LoopCtx<'_>,
        ) -> Result<(), LoopError> {
            self.points.lock().unwrap().push(ctx.point);
            Ok(())
        }
    }

    struct RecordingObserver {
        events: StdArc<StdMutex<Vec<AgentEvent>>>,
    }

    impl LoopObserver for RecordingObserver {
        fn handle_event(&self, event: ObservedEvent) {
            let event = event.event;
            self.events.lock().unwrap().push(event);
        }
    }

    #[test]
    fn session_consumer_capabilities_are_typed_and_serde_defaulted() {
        let config = SessionConfig::new("session").with_response_attempt_supersession();
        assert!(config.consumer_capabilities.response_attempt_supersession);

        let decoded: SessionConfig = serde_json::from_value(json!({
            "session_id": "session",
            "metadata": {},
            "cache": null
        }))
        .unwrap();
        assert_eq!(
            decoded.consumer_capabilities,
            SessionConsumerCapabilities::default()
        );
    }

    #[tokio::test]
    async fn response_attempt_supersession_is_forwarded_and_resets_attempt_state() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(SupersedingAdapter)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("supersession-session").with_response_attempt_supersession())
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "hello")])
            .unwrap();

        let LoopStep::Finished(result) = run_until_finished(&mut driver).await else {
            panic!("turn did not finish");
        };
        assert!(result.usage.is_none());

        let events = events.lock().unwrap();
        let tool_call = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ToolCallRequested(_)))
            .unwrap();
        let superseded = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ResponseAttemptSuperseded))
            .unwrap();
        assert!(tool_call < superseded);
    }

    #[tokio::test]
    async fn cancellation_preserves_streamed_content_in_result_and_transcript() {
        let controller = CancellationController::new();
        let agent = Agent::builder()
            .model(InterruptedStreamAdapter {
                controller: controller.clone(),
                scenario: InterruptedStreamScenario::PreserveContent,
            })
            .cancellation(controller.handle())
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("preserve-interrupted-stream"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "start")])
            .unwrap();

        let LoopStep::Finished(result) = run_until_finished(&mut driver).await else {
            panic!("turn did not finish");
        };
        assert_eq!(result.finish_reason, FinishReason::Cancelled);
        assert_eq!(result.items.len(), 1);
        assert_eq!(
            result.items[0].metadata.get(INTERRUPTED_METADATA_KEY),
            Some(&Value::Bool(true))
        );
        assert_eq!(result.items[0].parts.len(), 4);
        assert!(matches!(
            &result.items[0].parts[0],
            Part::Text(text) if text.text == "partial answer"
        ));
        assert!(matches!(
            &result.items[0].parts[1],
            Part::Reasoning(reasoning)
                if reasoning.summary.as_deref() == Some("partial thought")
                    && reasoning.metadata.get("provider.detail") == Some(&Value::Bool(true))
        ));
        assert!(matches!(
            &result.items[0].parts[2],
            Part::Structured(structured) if structured.value == json!({ "complete": false })
        ));
        assert!(matches!(
            &result.items[0].parts[3],
            Part::ToolCall(call) if call.id == ToolCallId::new("partial-call")
        ));
        assert!(
            !result.items[0]
                .parts
                .iter()
                .any(|part| matches!(part, Part::Media(_)))
        );
        assert!(!result.items[0].parts.iter().any(|part| {
            matches!(part, Part::Text(text) if text.text.contains("response was interrupted"))
        }));

        let transcript = driver.snapshot().transcript;
        let preserved = transcript
            .iter()
            .find(|item| {
                item.kind == ItemKind::Assistant
                    && item.metadata.get(INTERRUPTED_METADATA_KEY) == Some(&Value::Bool(true))
            })
            .expect("preserved assistant item in transcript");
        assert_eq!(preserved.parts, result.items[0].parts);
        assert!(transcript.iter().any(|item| {
            item.parts.iter().any(|part| {
                matches!(
                    part,
                    Part::ToolResult(tool_result)
                        if tool_result.call_id == ToolCallId::new("partial-call")
                            && tool_result.is_error
                )
            })
        }));
        validate_transcript_invariants(&transcript).unwrap();
    }

    #[test]
    fn committed_part_matches_the_correct_interleaved_open_part() {
        let mut content = StreamedAssistantContent::default();
        for delta in [
            Delta::BeginPart {
                part_id: PartId::new("first"),
                kind: PartKind::Text,
            },
            Delta::AppendText {
                part_id: PartId::new("first"),
                chunk: "one".into(),
            },
            Delta::BeginPart {
                part_id: PartId::new("second"),
                kind: PartKind::Text,
            },
            Delta::AppendText {
                part_id: PartId::new("second"),
                chunk: "two".into(),
            },
            Delta::CommitPart {
                part: Part::text("two"),
            },
        ] {
            content.apply_delta(&delta);
        }

        let items = content.interrupted_items();
        assert!(matches!(
            items[0].parts.as_slice(),
            [Part::Text(first), Part::Text(second)]
                if first.text == "one" && second.text == "two"
        ));
    }

    #[test]
    fn ambiguous_committed_part_does_not_duplicate_identical_open_parts() {
        let mut content = StreamedAssistantContent::default();
        for part_id in [PartId::new("first"), PartId::new("second")] {
            content.apply_delta(&Delta::BeginPart {
                part_id: part_id.clone(),
                kind: PartKind::Text,
            });
            content.apply_delta(&Delta::AppendText {
                part_id,
                chunk: "same".into(),
            });
        }
        content.apply_delta(&Delta::CommitPart {
            part: Part::text("same"),
        });

        let items = content.interrupted_items();
        assert_eq!(items[0].parts.len(), 2);
        assert!(
            items[0]
                .parts
                .iter()
                .all(|part| matches!(part, Part::Text(text) if text.text == "same"))
        );
    }

    #[test]
    fn deltas_without_begin_part_are_not_guessed() {
        let mut content = StreamedAssistantContent::default();
        content.apply_delta(&Delta::AppendText {
            part_id: PartId::new("missing"),
            chunk: "orphan".into(),
        });

        let items = content.interrupted_items();
        assert!(matches!(
            items[0].parts.as_slice(),
            [Part::Text(text)] if text.text.contains("response was interrupted")
        ));
    }

    #[test]
    fn streamed_content_byte_budget_covers_every_retained_payload() {
        fn open_content(kind: PartKind) -> StreamedAssistantContent {
            let mut content = StreamedAssistantContent::default();
            content.apply_delta(&Delta::BeginPart {
                part_id: PartId::new("part"),
                kind,
            });
            content
        }

        let mut content = open_content(PartKind::Text);
        content.apply_delta(&Delta::AppendText {
            part_id: PartId::new("part"),
            chunk: "x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES),
        });
        assert!(content.overflowed);

        let mut content = open_content(PartKind::Reasoning);
        content.apply_delta(&Delta::AppendBytes {
            part_id: PartId::new("part"),
            chunk: vec![0; MAX_STREAMED_ASSISTANT_CONTENT_BYTES],
        });
        assert!(content.overflowed);

        let mut content = open_content(PartKind::Structured);
        content.apply_delta(&Delta::ReplaceStructured {
            part_id: PartId::new("part"),
            value: Value::String("x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES)),
        });
        assert!(content.overflowed);

        let mut metadata = MetadataMap::new();
        metadata.insert(
            "large".into(),
            Value::String("x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES)),
        );
        let mut content = open_content(PartKind::Text);
        content.apply_delta(&Delta::SetMetadata {
            part_id: PartId::new("part"),
            metadata,
        });
        assert!(content.overflowed);

        let mut content = StreamedAssistantContent::default();
        content.apply_delta(&Delta::CommitPart {
            part: Part::text("x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES)),
        });
        assert!(content.overflowed);

        let mut content = StreamedAssistantContent::default();
        content.commit_tool_call(&ToolCallPart::new(
            "call",
            "tool",
            Value::String("x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES)),
        ));
        assert!(content.overflowed);
    }

    #[test]
    fn streamed_content_part_budget_releases_all_parts() {
        let mut content = StreamedAssistantContent::default();
        for index in 0..MAX_STREAMED_ASSISTANT_CONTENT_PARTS {
            content.apply_delta(&Delta::BeginPart {
                part_id: PartId::new(format!("part-{index}")),
                kind: PartKind::Text,
            });
        }
        assert_eq!(content.parts.len(), MAX_STREAMED_ASSISTANT_CONTENT_PARTS);

        content.apply_delta(&Delta::BeginPart {
            part_id: PartId::new("overflow"),
            kind: PartKind::Text,
        });

        assert!(content.overflowed);
        assert!(content.parts.is_empty());
        assert_eq!(content.retained_bytes, 0);
    }

    #[test]
    fn streamed_content_overflow_uses_generic_interruption_fallback() {
        let mut content = StreamedAssistantContent::default();
        content.apply_delta(&Delta::BeginPart {
            part_id: PartId::new("text"),
            kind: PartKind::Text,
        });
        content.apply_delta(&Delta::AppendText {
            part_id: PartId::new("text"),
            chunk: "release me".into(),
        });
        content.apply_delta(&Delta::AppendText {
            part_id: PartId::new("text"),
            chunk: "x".repeat(MAX_STREAMED_ASSISTANT_CONTENT_BYTES),
        });
        content.apply_delta(&Delta::BeginPart {
            part_id: PartId::new("ignored after overflow"),
            kind: PartKind::Text,
        });

        assert!(content.parts.is_empty());
        assert_eq!(content.retained_bytes, 0);
        let items = content.interrupted_items();
        assert!(matches!(
            items[0].parts.as_slice(),
            [Part::Text(text)]
                if text.text.contains("response was interrupted")
                    && !text.text.contains("release me")
        ));
    }

    #[tokio::test]
    async fn supersession_is_processed_before_concurrent_cancellation() {
        let controller = CancellationController::new();
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(InterruptedStreamAdapter {
                controller: controller.clone(),
                scenario: InterruptedStreamScenario::CancelOnSupersession,
            })
            .cancellation(controller.handle())
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("supersede-then-cancel").with_response_attempt_supersession())
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "start")])
            .unwrap();

        let LoopStep::Finished(result) = run_until_finished(&mut driver).await else {
            panic!("turn did not finish");
        };
        assert_eq!(result.finish_reason, FinishReason::Cancelled);
        assert_eq!(result.items.len(), 1);
        assert!(matches!(
            result.items[0].parts.as_slice(),
            [Part::Text(text)]
                if text.text.contains("response was interrupted")
                    && !text.text.contains("discard me")
        ));
        assert!(!driver.snapshot().transcript.iter().any(|item| {
            item.parts
                .iter()
                .any(|part| matches!(part, Part::Text(text) if text.text.contains("discard me")))
        }));

        let events = events.lock().unwrap();
        let superseded = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ResponseAttemptSuperseded))
            .expect("supersession event");
        let finished = events
            .iter()
            .position(|event| matches!(event, AgentEvent::TurnFinished(_)))
            .expect("turn-finished event");
        assert!(superseded < finished);
    }

    #[tokio::test]
    async fn supersession_resets_streamed_content_budget() {
        let controller = CancellationController::new();
        let agent = Agent::builder()
            .model(InterruptedStreamAdapter {
                controller: controller.clone(),
                scenario: InterruptedStreamScenario::OverflowThenSupersession,
            })
            .cancellation(controller.handle())
            .build()
            .unwrap();
        let mut driver = agent
            .start(
                SessionConfig::new("overflow-then-supersede").with_response_attempt_supersession(),
            )
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "start")])
            .unwrap();

        let LoopStep::Finished(result) = run_until_finished(&mut driver).await else {
            panic!("turn did not finish");
        };
        assert_eq!(result.finish_reason, FinishReason::Cancelled);
        assert!(matches!(
            result.items[0].parts.as_slice(),
            [Part::Text(text)] if text.text == "preserve me"
        ));
    }

    fn turn_lifecycle_events(
        events: &[AgentEvent],
    ) -> Vec<(agentkit_core::TurnId, Option<FinishReason>)> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TurnStarted { turn_id, .. } => Some((turn_id.clone(), None)),
                AgentEvent::TurnFinished(turn) => {
                    Some((turn.turn_id.clone(), Some(turn.finish_reason.clone())))
                }
                _ => None,
            })
            .collect()
    }

    struct CatalogExecutor {
        version: AtomicUsize,
        events: StdMutex<Vec<ToolCatalogEvent>>,
    }

    impl CatalogExecutor {
        fn new() -> Self {
            Self {
                version: AtomicUsize::new(0),
                events: StdMutex::new(Vec::new()),
            }
        }

        fn publish_change(&self, version: usize, event: ToolCatalogEvent) {
            self.version.store(version, Ordering::SeqCst);
            self.events.lock().unwrap().push(event);
        }
    }

    #[async_trait]
    impl ToolExecutor for CatalogExecutor {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![ToolSpec {
                name: ToolName::new("dynamic"),
                description: format!("dynamic version {}", self.version.load(Ordering::SeqCst)),
                input_schema: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                output_schema: None,
                annotations: ToolAnnotations::default(),
                metadata: MetadataMap::new(),
            }]
        }

        fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }

        async fn execute(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> ToolExecutionOutcome {
            ToolExecutionOutcome::Completed(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text("dynamic-ok".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
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
                    output_schema: None,
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

    async fn wait_until_completed(handle: &TaskManagerHandle) {
        timeout(Duration::from_secs(1), async {
            while handle.list_completed().await.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("task never completed");
    }

    #[tokio::test]
    async fn loop_continues_after_completed_tool_call() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-1"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let result = run_until_finished(&mut driver).await;

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

        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[0].1, None);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Completed));
    }

    /// Test helper: drives the loop, transparently resuming non-blocking
    /// cooperative interrupts (AfterToolResult), until a terminal step or a
    /// blocking interrupt is reached.
    async fn run_until_finished<S: ModelSession + Send>(driver: &mut LoopDriver<S>) -> LoopStep {
        loop {
            match driver.next().await.unwrap() {
                LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
                step => return step,
            }
        }
    }

    /// A mutator runs at the top of every `drive_turn`, and the loop labels
    /// the site via [`MutationPoint`]. The first drive of a turn is
    /// `AfterTurnEnded`; the continuation drive that follows a completed tool
    /// round must be `AfterToolResult` (a tool result was just appended and an
    /// inference call is imminent). This pins that the continuation reports the
    /// correct point.
    #[tokio::test]
    async fn post_tool_continuation_reports_after_tool_result_mutation_point() {
        let points = StdArc::new(StdMutex::new(Vec::<MutationPoint>::new()));
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .mutator(PointRecordingMutator {
                points: points.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-mutation-point"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        // FakeSession: turn 1 emits a tool call, the continuation turn finishes.
        let _ = run_until_finished(&mut driver).await;

        let recorded = points.lock().unwrap().clone();
        assert_eq!(
            recorded.first(),
            Some(&MutationPoint::AfterTurnEnded),
            "first drive of a fresh turn must report AfterTurnEnded, got {recorded:?}"
        );
        assert!(
            recorded.contains(&MutationPoint::AfterToolResult),
            "post-tool continuation must report AfterToolResult, got {recorded:?}"
        );
    }

    #[tokio::test]
    async fn no_work_awaiting_input_emits_no_turn_lifecycle() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(SlowAdapter)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-no-work"))
            .await
            .unwrap();

        for _ in 0..2 {
            assert!(matches!(
                driver.next().await.unwrap(),
                LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
            ));
        }

        assert!(turn_lifecycle_events(&events.lock().unwrap()).is_empty());
    }

    #[tokio::test]
    async fn normal_turn_emits_one_matched_lifecycle_pair() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(SlowAdapter)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-normal-lifecycle"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Finished(TurnResult {
                finish_reason: FinishReason::Completed,
                ..
            })
        ));

        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[0].1, None);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Completed));
    }

    #[tokio::test]
    async fn post_start_error_emits_terminal_error_without_run_failed() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(SlowAdapter)
            .mutator(ErrorMutator)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-error-lifecycle"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await,
            Err(LoopError::Mutator(message)) if message == "boom"
        ));

        let events = events.lock().unwrap();
        let lifecycle = turn_lifecycle_events(&events);
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[0].1, None);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Error));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::RunFailed { .. }))
        );
    }

    #[tokio::test]
    async fn active_tool_error_repairs_state_and_retry_uses_fresh_lifecycle() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let interrupted = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(ToolRegistry::new().with(EchoTool::default()))
            .task_manager(
                TestTaskManager::new(SimpleTaskManager::new())
                    .fail_start("original start failure")
                    .record_interrupts(interrupted.clone())
                    .fail_interrupt("cleanup failure"),
            )
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-active-tool-error"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "first")])
            .unwrap();

        let error = driver.next().await.unwrap_err();
        assert!(error.to_string().contains("original start failure"));
        assert!(!error.to_string().contains("cleanup failure"));
        assert!(driver.active_tool_round.is_none());
        assert!(driver.pending_round_resume.is_none());
        assert!(unanswered_tool_calls(&driver.snapshot().transcript).is_empty());
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
        assert_eq!(interrupted.lock().unwrap().len(), 1);

        driver
            .submit_input(vec![Item::text(ItemKind::User, "retry")])
            .unwrap();
        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Finished(TurnResult {
                finish_reason: FinishReason::Completed,
                ..
            })
        ));

        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 4, "{lifecycle:?}");
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Error));
        assert_eq!(lifecycle[2].0, lifecycle[3].0);
        assert_eq!(lifecycle[3].1, Some(FinishReason::Completed));
        assert_ne!(lifecycle[0].0, lifecycle[2].0);
    }

    #[tokio::test]
    async fn continuation_error_clears_resume_and_retry_uses_fresh_lifecycle() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let interrupted = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(ToolRegistry::new().with(EchoTool::default()))
            .task_manager(
                TestTaskManager::new(SimpleTaskManager::new())
                    .fail_pending_update_on(1, "original continuation failure")
                    .record_interrupts(interrupted.clone())
                    .fail_interrupt("cleanup failure"),
            )
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-continuation-error"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "first")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_))
        ));
        let error = driver.next().await.unwrap_err();
        assert!(error.to_string().contains("original continuation failure"));
        assert!(!error.to_string().contains("cleanup failure"));
        assert!(driver.pending_round_resume.is_none());
        assert!(driver.active_tool_round.is_none());
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
        assert_eq!(interrupted.lock().unwrap().len(), 1);

        driver
            .submit_input(vec![Item::text(ItemKind::User, "retry")])
            .unwrap();
        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Finished(TurnResult {
                finish_reason: FinishReason::Completed,
                ..
            })
        ));

        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 4, "{lifecycle:?}");
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Error));
        assert_eq!(lifecycle[2].0, lifecycle[3].0);
        assert_eq!(lifecycle[3].1, Some(FinishReason::Completed));
        assert_ne!(lifecycle[0].0, lifecycle[2].0);
    }

    #[test]
    fn pending_input_requires_input_bearing_tail_role() {
        assert!(!transcript_has_pending_input(&[]));
        assert!(!transcript_has_pending_input(&[Item::text(
            ItemKind::System,
            "system"
        )]));
        assert!(!transcript_has_pending_input(&[Item::text(
            ItemKind::Developer,
            "developer"
        )]));
        assert!(!transcript_has_pending_input(&[Item::text(
            ItemKind::Context,
            "context"
        )]));
        assert!(!transcript_has_pending_input(&[Item::text(
            ItemKind::Assistant,
            "assistant"
        )]));

        assert!(transcript_has_pending_input(&[Item::text(
            ItemKind::User,
            "user"
        )]));
        assert!(transcript_has_pending_input(&[Item::notification(
            "background update"
        )]));
        assert!(transcript_has_pending_input(&[Item {
            id: None,
            kind: ItemKind::Tool,
            parts: vec![Part::ToolResult(ToolResultPart {
                call_id: ToolCallId::new("call-test"),
                output: ToolOutput::Text("ok".into()),
                is_error: false,
                metadata: MetadataMap::new(),
            })],
            metadata: MetadataMap::new(),
            usage: None,
            finish_reason: None,
            created_at: None,
        }]));
    }

    /// Drops a trailing `User` item. Stands in for any mutator that removes the
    /// freshly-submitted input during `drive_turn` — e.g. a compaction pass
    /// that summarises the latest user turn away, or a normalisation step that
    /// strips an empty user prompt — leaving the transcript ending in an
    /// assistant message.
    struct DropTrailingUserMutator;
    struct ErrorMutator;

    #[async_trait]
    impl LoopMutator for ErrorMutator {
        async fn mutate(
            &self,
            _cursor: &mut TranscriptCursor<'_>,
            _ctx: LoopCtx<'_>,
        ) -> Result<(), LoopError> {
            Err(LoopError::Mutator("boom".into()))
        }
    }

    #[async_trait]
    impl LoopMutator for DropTrailingUserMutator {
        async fn mutate(
            &self,
            cursor: &mut TranscriptCursor<'_>,
            _ctx: LoopCtx<'_>,
        ) -> Result<(), LoopError> {
            if cursor.last().map(|item| item.kind) == Some(ItemKind::User) {
                cursor.pop();
            }
            Ok(())
        }
    }

    /// Mirrors the provider gram hit (Vertex/Bedrock via OpenRouter): a model
    /// that rejects any request whose final message is an assistant message
    /// ("assistant prefill — the conversation must end with a user message").
    /// Records whether it was ever asked to begin such a turn.
    struct RejectAssistantPrefillAdapter {
        saw_assistant_tail: StdArc<AtomicBool>,
    }

    struct RejectAssistantPrefillSession {
        saw_assistant_tail: StdArc<AtomicBool>,
    }

    #[async_trait]
    impl ModelAdapter for RejectAssistantPrefillAdapter {
        type Session = RejectAssistantPrefillSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(RejectAssistantPrefillSession {
                saw_assistant_tail: self.saw_assistant_tail.clone(),
            })
        }
    }

    #[async_trait]
    impl ModelSession for RejectAssistantPrefillSession {
        type Turn = FakeTurn;

        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            _cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            if request.transcript.last().map(|item| item.kind) == Some(ItemKind::Assistant) {
                self.saw_assistant_tail.store(true, Ordering::SeqCst);
                return Err(LoopError::Provider(
                    "conversation must end with a user message".into(),
                ));
            }
            Ok(FakeTurn {
                events: VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    model: None,
                    response_id: None,
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item::text(ItemKind::Assistant, "ok")],
                    usage: None,
                    metadata: MetadataMap::new(),
                })]),
            })
        }
    }

    /// Reproduces the exact failure mode observed in gram: a mutator removes
    /// the just-submitted user input during `drive_turn`, so the transcript
    /// ends in an assistant message with nothing for the model to respond to.
    /// The loop must NOT dispatch a model request in that state — there is no
    /// valid trailing input to drive with — it should finish the turn instead.
    /// The adapter stands in for a provider that rejects assistant prefill, so
    /// any dispatch in this state would surface as a provider error.
    #[tokio::test]
    async fn drive_does_not_dispatch_without_valid_trailing_input() {
        let saw_assistant_tail = StdArc::new(AtomicBool::new(false));
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(RejectAssistantPrefillAdapter {
                saw_assistant_tail: saw_assistant_tail.clone(),
            })
            .mutator(DropTrailingUserMutator)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            // Prior conversation ending in an assistant message — e.g. a cold
            // bootstrap that loaded a completed turn's history.
            .transcript(vec![
                Item::text(ItemKind::User, "kickoff"),
                Item::text(ItemKind::Assistant, "prior reply"),
            ])
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-no-valid-input"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "follow up")])
            .unwrap();

        // The mutator strips the "follow up" user item, leaving [user, assistant].
        let outcome = driver.next().await;

        assert!(
            !saw_assistant_tail.load(Ordering::SeqCst),
            "loop dispatched a model turn whose transcript ends in an assistant \
             message (outcome: {outcome:?}); with no valid trailing input the turn \
             must finish instead of driving"
        );
        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[0].1, None);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Completed));
    }

    #[tokio::test]
    async fn loop_uses_injected_permission_checker() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(DenyFsReads)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-2"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let result = run_until_finished(&mut driver).await;

        match result {
            LoopStep::Finished(turn) => match &turn.items[0].parts[0] {
                Part::Text(text) => assert!(text.text.contains("tool permission denied")),
                other => panic!("unexpected part: {other:?}"),
            },
            other => panic!("unexpected loop step: {other:?}"),
        }

        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .all(|event| !matches!(event, AgentEvent::ToolExecutionStarted(_))),
            "denied tools must not be reported as started"
        );
    }

    #[tokio::test]
    async fn failed_tool_execution_still_reports_started() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(FailingTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-failing-start-event"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        match run_until_finished(&mut driver).await {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Completed),
            other => panic!("unexpected loop step: {other:?}"),
        }

        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolExecutionStarted(call) if call.name == "failing"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolResultReceived(result) if result.is_error
        )));
    }

    #[tokio::test]
    async fn run_then_deny_tool_execution_still_reports_started() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(RunThenDenyTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-run-then-deny-start-event"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        match run_until_finished(&mut driver).await {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Completed),
            other => panic!("unexpected loop step: {other:?}"),
        }

        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolExecutionStarted(call) if call.name == "run_then_deny"
        )));
        // A mid-execution denial is still a permission denial (failure_kind),
        // but the tool DID start, so it must not carry the not-started marker.
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolResultReceived(result)
                if result.is_error
                    && result
                        .metadata
                        .get(TOOL_RESULT_FAILURE_KIND_METADATA_KEY)
                        .and_then(Value::as_str)
                        == Some(TOOL_RESULT_FAILURE_KIND_PERMISSION_DENIED)
                    && !result
                        .metadata
                        .contains_key(TOOL_RESULT_NOT_STARTED_METADATA_KEY)
        )));
    }

    #[tokio::test]
    async fn async_task_manager_background_round_requires_explicit_continue() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
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
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .task_manager(task_manager)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-background"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let first = driver.next().await.unwrap();
        match first {
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {}
            other => panic!("unexpected first loop step: {other:?}"),
        }

        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[1].1, Some(FinishReason::ToolCall));

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
                    Part::Text(text) => assert_eq!(
                        text.text,
                        "tool said: Background tool results: 1 total, 0 failed, 0 with metadata. \
                         call-1 completed: text preview: background-done"
                    ),
                    other => panic!("unexpected part after resume: {other:?}"),
                }
            }
            other => panic!("unexpected resumed step: {other:?}"),
        }

        let events = events.lock().unwrap();
        let lifecycle = turn_lifecycle_events(&events);
        assert_eq!(lifecycle.len(), 4);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[2].0, lifecycle[3].0);
        assert_ne!(lifecycle[0].0, lifecycle[2].0);
        assert_eq!(lifecycle[3].1, Some(FinishReason::Completed));

        let terminal_results: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolResultReceived(result)
                    if result.call_id == ToolCallId::new("call-1") =>
                {
                    Some(result)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            terminal_results.len(),
            1,
            "background completion must emit one terminal result event per call: {events:?}"
        );
    }

    #[tokio::test]
    async fn detached_parts_notification_preserves_full_output_and_metadata() {
        let agent = Agent::builder().model(FakeAdapter).build().unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-detached-parts"))
            .await
            .unwrap();
        let call_id = ToolCallId::new("parts-call");
        driver.detached_call_ids.insert(call_id.clone());
        let parts = vec![
            Part::text("part text"),
            Part::structured(json!({
                "nested": [1, 2, 3]
            })),
        ];
        let mut metadata = MetadataMap::new();
        metadata.insert("source".into(), json!("background"));
        let result = ToolResultPart {
            call_id,
            output: ToolOutput::Parts(parts.clone()),
            is_error: true,
            metadata: metadata.clone(),
        };
        let mut item_metadata = MetadataMap::new();
        item_metadata.insert("delivery".into(), json!("deferred"));
        let item = Item::new(ItemKind::Tool, vec![Part::ToolResult(result.clone())])
            .with_metadata(item_metadata.clone());

        let converted = driver.maybe_convert_detached(item);
        let (text, structured) = match converted.parts.as_slice() {
            [Part::Text(text), Part::Structured(structured)] => (text, structured),
            other => panic!("unexpected converted parts: {other:?}"),
        };
        assert_eq!(converted.kind, ItemKind::Notification);
        assert_eq!(converted.metadata, item_metadata);
        assert_eq!(structured.value, serde_json::to_value(&result).unwrap());
        assert_eq!(
            text.text,
            "Background tool results: 1 total, 1 failed, 1 with metadata. \
             parts-call failed: parts payload (2 parts)"
        );
        assert!(!text.text.contains("part text"));
        assert!(!text.text.contains("background"));
    }

    #[tokio::test]
    async fn detached_files_notification_preserves_full_output() {
        let agent = Agent::builder().model(FakeAdapter).build().unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-detached-files"))
            .await
            .unwrap();
        let call_id = ToolCallId::new("files-call");
        driver.detached_call_ids.insert(call_id.clone());
        let files = vec![
            agentkit_core::FilePart::named("report.txt", DataRef::inline_text("full file body"))
                .with_mime_type("text/plain"),
            agentkit_core::FilePart::named(
                "remote.json",
                DataRef::uri("https://example.test/remote.json"),
            ),
        ];
        let mut result_metadata = MetadataMap::new();
        result_metadata.insert("archive".into(), json!(true));
        let result = ToolResultPart::success(call_id, ToolOutput::Files(files.clone()))
            .with_metadata(result_metadata);
        let mut item_metadata = MetadataMap::new();
        item_metadata.insert("delivery".into(), json!("deferred"));
        let item = Item::new(ItemKind::Tool, vec![Part::ToolResult(result.clone())])
            .with_metadata(item_metadata.clone());

        let converted = driver.maybe_convert_detached(item);
        let (text, structured) = match converted.parts.as_slice() {
            [Part::Text(text), Part::Structured(structured)] => (text, structured),
            other => panic!("unexpected converted files: {other:?}"),
        };
        assert_eq!(converted.kind, ItemKind::Notification);
        assert_eq!(converted.metadata, item_metadata);
        assert_eq!(structured.value, serde_json::to_value(&result).unwrap());
        assert_eq!(
            text.text,
            "Background tool results: 1 total, 0 failed, 1 with metadata. \
             files-call completed: files payload (2 files)"
        );
        assert!(!text.text.contains("full file body"));
        assert!(!text.text.contains("remote.json"));
    }

    #[test]
    fn detached_result_summaries_are_bounded_and_do_not_serialize_structured_payloads() {
        let long_text = "é".repeat(DETACHED_TEXT_PREVIEW_MAX_CHARS + 20);
        let text_summary = render_tool_output_brief(&ToolOutput::Text(long_text.clone()));
        assert_eq!(
            text_summary.chars().count(),
            "text preview: ".chars().count() + DETACHED_TEXT_PREVIEW_MAX_CHARS
        );
        assert!(text_summary.ends_with('…'));
        assert!(!text_summary.contains(&long_text));

        let secret = "structured payload must remain out of notification text";
        let structured = ToolOutput::Structured(json!({ "secret": secret }));
        assert_eq!(render_tool_output_brief(&structured), "structured payload");

        let oversized = "x".repeat(DETACHED_NOTIFICATION_TEXT_MAX_CHARS + 20);
        let bounded = truncate_chars(&oversized, DETACHED_NOTIFICATION_TEXT_MAX_CHARS);
        assert_eq!(
            bounded.chars().count(),
            DETACHED_NOTIFICATION_TEXT_MAX_CHARS
        );
        assert!(bounded.ends_with('…'));
    }

    #[tokio::test]
    async fn detached_tool_placeholder_is_progress_not_terminal_result() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "detaching-wait",
            RoutingDecision::ForegroundThenDetachAfter(Duration::from_millis(10)),
        )]));
        let handle = task_manager.handle();
        let tools = ToolRegistry::new().with(BlockingTool::new(
            "detaching-wait",
            entered.clone(),
            release.clone(),
            "detached-done",
        ));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .task_manager(task_manager)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-detached-progress"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => {}
            other => panic!("unexpected detach step: {other:?}"),
        }

        match wait_for_task_event(&handle).await {
            TaskEvent::Started(snapshot) => assert_eq!(snapshot.tool_name, "detaching-wait"),
            other => panic!("unexpected task event: {other:?}"),
        }
        match wait_for_task_event(&handle).await {
            TaskEvent::Detached(snapshot) => assert_eq!(snapshot.tool_name, "detaching-wait"),
            other => panic!("unexpected detach event: {other:?}"),
        }
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();

        match wait_for_task_event(&handle).await {
            TaskEvent::Completed(_, result) => {
                assert_eq!(result.output, ToolOutput::Text("detached-done".into()))
            }
            other => panic!("unexpected completion event: {other:?}"),
        }

        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Completed),
            other => panic!("unexpected resumed step: {other:?}"),
        }

        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolExecutionProgress(result)
                if result.call_id == ToolCallId::new("call-1") && !result.is_error
        )));
        let terminal_results: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolResultReceived(result)
                    if result.call_id == ToolCallId::new("call-1") =>
                {
                    Some(result)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            terminal_results.len(),
            1,
            "detached call must emit one terminal result event: {events:?}"
        );
    }

    #[tokio::test]
    async fn cancelled_background_approval_auto_resolves_when_drained() {
        let controller = CancellationController::new();
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "echo",
            RoutingDecision::Background,
        )]));
        let handle = task_manager.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(DelayedApprovalExecutor::new(
                entered.clone(),
                release.clone(),
            ))
            .task_manager(task_manager)
            .cancellation(controller.handle())
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-cancel-delayed-background-approval"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {}
            other => panic!("unexpected first step: {other:?}"),
        }

        match wait_for_task_event(&handle).await {
            TaskEvent::Started(snapshot) => assert_eq!(snapshot.tool_name, "echo"),
            other => panic!("unexpected task event: {other:?}"),
        }

        wait_until_entered(entered.as_ref()).await;
        controller.interrupt();
        release.notify_waiters();
        wait_until_completed(&handle).await;

        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Cancelled),
            other => panic!("cancelled background approval should finish cancelled, got {other:?}"),
        }

        let events = events.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ApprovalResolved { approved: false }))
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolResultReceived(result)
                if result.call_id == ToolCallId::new("call-1") && result.is_error
        )));
    }

    #[tokio::test]
    async fn approved_foreground_task_waits_for_result_before_model_continuation() {
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let approved_entered = StdArc::new(AtomicBool::new(false));
        let approved_release = StdArc::new(Notify::new());
        let route_count = StdArc::new(AtomicUsize::new(0));
        let routing_count = route_count.clone();
        let task_manager = AsyncTaskManager::new().routing(move |_request: &ToolRequest| {
            if routing_count.fetch_add(1, Ordering::SeqCst) == 0 {
                RoutingDecision::Background
            } else {
                RoutingDecision::Foreground
            }
        });
        let handle = task_manager.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(
                DelayedApprovalExecutor::new(entered.clone(), release.clone())
                    .blocking_after_approval(approved_entered.clone(), approved_release.clone()),
            )
            .task_manager(task_manager)
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-approved-foreground"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        let task_turn = match wait_for_task_event(&handle).await {
            TaskEvent::Started(snapshot) => snapshot.turn_id,
            other => panic!("unexpected task event: {other:?}"),
        };
        wait_until_entered(entered.as_ref()).await;
        release.notify_one();
        wait_until_completed(&handle).await;

        let pending = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => pending,
            other => panic!("unexpected delayed approval step: {other:?}"),
        };
        let presentation_turn = driver.lifecycle.active_turn.clone().unwrap();
        assert_ne!(presentation_turn, task_turn);
        pending.approve(&mut driver).unwrap();

        let info = {
            let next = driver.next();
            tokio::pin!(next);
            tokio::select! {
                () = wait_until_entered(approved_entered.as_ref()) => {}
                result = &mut next => {
                    panic!("model continued before approved foreground result: {result:?}")
                }
            }
            assert!(
                timeout(Duration::from_millis(10), &mut next).await.is_err(),
                "model continued while approved foreground work was blocked"
            );
            approved_release.notify_one();
            let step = timeout(Duration::from_secs(1), &mut next)
                .await
                .expect("approved foreground result was not delivered")
                .unwrap();
            match step {
                LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => info,
                other => panic!("unexpected approved foreground step: {other:?}"),
            }
        };
        assert_eq!(info.turn_id, presentation_turn);

        let turn = match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => turn,
            other => panic!("model did not continue after approved result: {other:?}"),
        };
        assert_eq!(turn.finish_reason, FinishReason::Completed);
        assert_eq!(turn.turn_id, presentation_turn);
        assert!(driver.snapshot().transcript.iter().any(|item| {
            item.kind == ItemKind::Notification
                && item.parts.iter().any(
                    |part| matches!(part, Part::Text(text) if text.text.contains("approved-ok")),
                )
        }));
    }

    #[tokio::test]
    async fn approved_foreground_then_detach_waits_and_keeps_one_placeholder() {
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let approved_entered = StdArc::new(AtomicBool::new(false));
        let approved_release = StdArc::new(Notify::new());
        let route_count = StdArc::new(AtomicUsize::new(0));
        let routing_count = route_count.clone();
        let task_manager = AsyncTaskManager::new().routing(move |_request: &ToolRequest| {
            if routing_count.fetch_add(1, Ordering::SeqCst) == 0 {
                RoutingDecision::Background
            } else {
                RoutingDecision::ForegroundThenDetachAfter(Duration::from_millis(10))
            }
        });
        let handle = task_manager.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(
                DelayedApprovalExecutor::new(entered.clone(), release.clone())
                    .blocking_after_approval(approved_entered.clone(), approved_release.clone()),
            )
            .task_manager(task_manager)
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-approved-foreground-detach"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        let _ = wait_for_task_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_one();
        wait_until_completed(&handle).await;

        let pending = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => pending,
            other => panic!("unexpected delayed approval step: {other:?}"),
        };
        let presentation_turn = driver.lifecycle.active_turn.clone().unwrap();
        pending.approve(&mut driver).unwrap();

        let step = timeout(Duration::from_secs(1), driver.next())
            .await
            .expect("approved task did not detach")
            .unwrap();
        assert!(approved_entered.load(Ordering::SeqCst));
        match step {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => {
                assert_eq!(info.turn_id, presentation_turn);
            }
            other => panic!("unexpected approved detach step: {other:?}"),
        }
        let placeholders = driver
            .snapshot()
            .transcript
            .iter()
            .filter(|item| item.kind == ItemKind::Tool)
            .flat_map(|item| &item.parts)
            .filter(|part| {
                matches!(
                    part,
                    Part::ToolResult(result) if result.call_id == ToolCallId::new("call-1")
                )
            })
            .count();
        assert_eq!(placeholders, 1, "detach appended a second tool result");

        approved_release.notify_one();
        wait_until_completed(&handle).await;
        let turn = match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => turn,
            other => panic!("model did not continue after detached result: {other:?}"),
        };
        assert_eq!(turn.finish_reason, FinishReason::Completed);
        assert_eq!(turn.turn_id, presentation_turn);
        let transcript = driver.snapshot().transcript;
        assert_eq!(
            transcript
                .iter()
                .filter(|item| item.kind == ItemKind::Tool)
                .flat_map(|item| &item.parts)
                .filter(|part| {
                    matches!(
                        part,
                        Part::ToolResult(result)
                            if result.call_id == ToolCallId::new("call-1")
                    )
                })
                .count(),
            1
        );
        assert!(transcript.iter().any(|item| {
            item.kind == ItemKind::Notification
                && item.parts.iter().any(
                    |part| matches!(part, Part::Text(text) if text.text.contains("approved-ok")),
                )
        }));
    }

    #[tokio::test]
    async fn approving_detached_background_call_keeps_one_placeholder() {
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "echo",
            RoutingDecision::Background,
        )]));
        let handle = task_manager.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(DelayedApprovalExecutor::new(
                entered.clone(),
                release.clone(),
            ))
            .task_manager(task_manager)
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-approve-detached-background"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        let _ = wait_for_task_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();
        wait_until_completed(&handle).await;

        let pending = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => pending,
            other => panic!("unexpected delayed approval step: {other:?}"),
        };
        pending.approve(&mut driver).unwrap();
        release.notify_one();
        let _ = driver.next().await.unwrap();

        let placeholders = driver
            .snapshot()
            .transcript
            .iter()
            .flat_map(|item| &item.parts)
            .filter(|part| {
                matches!(
                    part,
                    Part::ToolResult(result) if result.call_id == ToolCallId::new("call-1")
                )
            })
            .count();
        assert_eq!(placeholders, 1, "approval appended a second detach result");
    }

    #[tokio::test]
    async fn failed_background_approval_cleanup_clears_queued_resume() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let inner = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "echo",
            RoutingDecision::ForegroundThenDetachAfter(Duration::from_millis(10)),
        )]));
        let handle = inner.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(DelayedApprovalExecutor::new(
                entered.clone(),
                release.clone(),
            ))
            .task_manager(TestTaskManager::new(inner).fail_interrupt("cleanup failure"))
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new(
                "session-failed-detached-background-approval-cleanup",
            ))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        let old_turn_id = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => info.turn_id,
            other => panic!("unexpected detach step: {other:?}"),
        };
        assert_eq!(driver.pending_round_resume.as_ref(), Some(&old_turn_id));
        driver
            .submit_input(vec![Item::text(ItemKind::User, "fresh input")])
            .unwrap();
        let _ = wait_for_task_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();
        wait_until_completed(&handle).await;

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_))
        ));
        let error = driver.cancel_pending_approvals().await.unwrap_err();
        assert!(error.to_string().contains("cleanup failure"));

        assert!(driver.lifecycle.active_turn.is_none());
        assert!(driver.pending_round_resume.is_none());
        assert_eq!(driver.snapshot().pending_input.len(), 1);
        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        let [started, finished] = &lifecycle[lifecycle.len() - 2..] else {
            panic!("missing terminal lifecycle events: {lifecycle:?}");
        };
        assert_eq!(started.0, finished.0);
        assert_eq!(finished.1, Some(FinishReason::Error));
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();

        let fresh_turn = match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => turn,
            other => panic!("fresh input did not start a new turn: {other:?}"),
        };
        assert_ne!(fresh_turn.turn_id, old_turn_id);
    }

    #[tokio::test]
    async fn fresh_input_runs_before_delayed_background_approval() {
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "echo",
            RoutingDecision::Background,
        )]));
        let handle = task_manager.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(DelayedApprovalExecutor::new(
                entered.clone(),
                release.clone(),
            ))
            .task_manager(task_manager)
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new(
                "session-input-before-background-approval",
            ))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        let _ = wait_for_task_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();
        wait_until_completed(&handle).await;

        driver
            .submit_input(vec![Item::text(ItemKind::User, "fresh input")])
            .unwrap();
        let fresh_turn = match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => turn.turn_id,
            other => panic!("fresh input was not driven first: {other:?}"),
        };
        assert!(driver.pending_approvals.is_empty());
        assert!(driver.snapshot().pending_input.is_empty());
        timeout(Duration::from_millis(100), driver.wait_for_loop_update())
            .await
            .expect("collected background update did not wake the loop")
            .unwrap();

        let approval = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(approval)) => approval,
            other => panic!("delayed approval was not presented separately: {other:?}"),
        };
        let approval_turn = driver.lifecycle.active_turn.clone().unwrap();
        assert_ne!(fresh_turn, approval_turn);
        assert_eq!(
            driver
                .snapshot()
                .transcript
                .iter()
                .filter(|item| {
                    item.kind == ItemKind::User
                        && item.parts.iter().any(
                            |part| matches!(part, Part::Text(text) if text.text == "fresh input"),
                        )
                })
                .count(),
            1,
            "fresh input must not be replayed while presenting the approval"
        );
        approval.deny(&mut driver).unwrap();
    }

    #[tokio::test]
    async fn delayed_background_approval_interrupts_originating_task_turn() {
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let interrupted = StdArc::new(StdMutex::new(Vec::new()));
        let inner = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "echo",
            RoutingDecision::Background,
        )]));
        let handle = inner.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(DelayedApprovalExecutor::new(
                entered.clone(),
                release.clone(),
            ))
            .task_manager(TestTaskManager::new(inner).record_interrupts(interrupted.clone()))
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new(
                "session-background-approval-origin-turn",
            ))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        let _ = wait_for_task_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();
        wait_until_completed(&handle).await;
        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_))
        ));

        let presentation_turn = driver.lifecycle.active_turn.clone().unwrap();
        let task_turn = driver
            .pending_approvals
            .values()
            .next()
            .unwrap()
            .tool_request
            .turn_id
            .clone();
        assert_ne!(presentation_turn, task_turn);
        assert!(matches!(
            driver.cancel_pending_approvals().await.unwrap(),
            Some(LoopStep::Finished(TurnResult {
                finish_reason: FinishReason::Cancelled,
                ..
            }))
        ));
        assert_eq!(interrupted.lock().unwrap().as_slice(), &[task_turn]);
        assert!(driver.lifecycle.active_turn.is_none());
    }

    #[tokio::test]
    async fn approved_background_start_error_interrupts_originating_turn() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let interrupted = StdArc::new(StdMutex::new(Vec::new()));
        let inner = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "echo",
            RoutingDecision::Background,
        )]));
        let handle = inner.handle();
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(DelayedApprovalExecutor::new(
                entered.clone(),
                release.clone(),
            ))
            .task_manager(
                TestTaskManager::new(inner)
                    .fail_approved_start("original approved start failure")
                    .record_interrupts(interrupted.clone())
                    .fail_interrupt("cleanup failure"),
            )
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-approved-start-error"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        let _ = wait_for_task_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();
        wait_until_completed(&handle).await;
        let pending = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => pending,
            other => panic!("unexpected delayed approval step: {other:?}"),
        };
        let task_turn = driver
            .pending_approvals
            .values()
            .next()
            .unwrap()
            .tool_request
            .turn_id
            .clone();
        let call_id = pending.request.call_id.clone().expect("approval call id");
        assert!(driver.detached_call_ids.contains(&call_id));
        pending.approve(&mut driver).unwrap();

        let error = driver.next().await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("original approved start failure")
        );
        assert!(!error.to_string().contains("cleanup failure"));
        assert_eq!(interrupted.lock().unwrap().as_slice(), &[task_turn]);
        assert!(driver.lifecycle.active_turn.is_none());
        assert!(!driver.detached_call_ids.contains(&call_id));
        assert!(!driver.background_call_ids.contains(&call_id));
        assert!(!driver.tool_cancellations.contains_key(&call_id));
        assert!(events.lock().unwrap().iter().any(|event| matches!(
            event,
            AgentEvent::ToolResultReceived(result)
                if result.call_id == call_id && result.is_error
        )));
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
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
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
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
                usage: None,
                finish_reason: None,
                created_at: None,
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
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .cancellation(controller.handle())
            .task_manager(task_manager)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-mixed-cancel"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
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
    async fn a_cancelled_turn_answers_the_tool_call_it_abandoned() {
        // A transcript whose tool_use has no tool_result cannot be resumed:
        // validation rejects it, and providers refuse it outright. Cancelling
        // mid-call must therefore leave the pair complete — on the driver's
        // transcript, and in whatever the host persisted from it.
        let controller = CancellationController::new();
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let items = StdArc::new(StdMutex::new(Vec::<Item>::new()));
        let task_manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "wait",
            RoutingDecision::Foreground,
        )]));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(ToolRegistry::new().with(BlockingTool::new(
                "wait",
                entered.clone(),
                release,
                "done",
            )))
            .permissions(AllowAllPermissions)
            .cancellation(controller.handle())
            .task_manager(task_manager)
            .transcript_observer(RecordingTranscriptObserver {
                items: items.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-cancel-mid-call"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "run the tool".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let cancelled = tokio::join!(async { driver.next().await }, async {
            wait_until_entered(entered.as_ref()).await;
            controller.interrupt();
        })
        .0
        .unwrap();

        match cancelled {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Cancelled),
            other => panic!("unexpected loop step after interrupt: {other:?}"),
        }

        let transcript = driver.snapshot().transcript;
        assert!(
            unanswered_tool_calls(&transcript).is_empty(),
            "the cancelled turn left a tool call unanswered: {transcript:?}"
        );
        validate_transcript_invariants(&transcript)
            .expect("a cancelled turn must leave a resumable transcript");

        let persisted = items.lock().unwrap().clone();
        let results: Vec<&ToolResultPart> = persisted
            .iter()
            .flat_map(|item| &item.parts)
            .filter_map(|part| match part {
                Part::ToolResult(result) => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 1, "{persisted:?}");
        assert_eq!(results[0].call_id, ToolCallId::new("call-1"));
        assert!(results[0].is_error);
        assert_eq!(
            results[0].metadata.get(INTERRUPTED_METADATA_KEY),
            Some(&Value::Bool(true))
        );
    }

    #[tokio::test]
    async fn regression_cancelled_background_completion_emits_one_terminal_result() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-cancelled-background-event"))
            .await
            .unwrap();

        driver.append_item(Item::new(
            ItemKind::Assistant,
            vec![Part::ToolCall(ToolCallPart {
                id: ToolCallId::new("call-1"),
                name: "wait".into(),
                input: json!({}),
                metadata: MetadataMap::new(),
            })],
        ));
        driver.background_call_ids.insert(ToolCallId::new("call-1"));
        driver.close_interrupted_tool_calls();
        driver.append_tool_result_item(Item::new(
            ItemKind::Tool,
            vec![Part::ToolResult(ToolResultPart {
                call_id: ToolCallId::new("call-1"),
                output: ToolOutput::Text("background-done".into()),
                is_error: false,
                metadata: MetadataMap::new(),
            })],
        ));

        let events = events.lock().unwrap();
        let terminal_results = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    AgentEvent::ToolResultReceived(result)
                        if result.call_id == ToolCallId::new("call-1")
                )
            })
            .count();
        assert_eq!(
            terminal_results, 1,
            "a cancelled background call emitted multiple terminal results: {events:?}"
        );
    }

    #[tokio::test]
    async fn regression_cancelled_queued_approval_is_answered_once() {
        let controller = CancellationController::new();
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        release.notify_one();
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(
                DelayedApprovalExecutor::new(entered, release)
                    .cancelling_on_approval(controller.clone()),
            )
            .cancellation(controller.handle())
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-cancelled-queued-approval"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => assert_eq!(turn.finish_reason, FinishReason::Cancelled),
            other => panic!("unexpected first cancellation step: {other:?}"),
        }
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {}
            other => panic!("unexpected post-cancellation step: {other:?}"),
        }

        let events = events.lock().unwrap();
        let terminal_results = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    AgentEvent::ToolResultReceived(result)
                        if result.call_id == ToolCallId::new("call-1")
                )
            })
            .count();
        assert_eq!(
            terminal_results, 1,
            "a cancelled queued approval was answered more than once: {events:?}"
        );
        drop(events);

        let transcript = driver.snapshot().transcript;
        assert!(
            !transcript.iter().any(|item| {
                item.kind == ItemKind::Notification
                    && item.parts.iter().any(|part| {
                        matches!(part, Part::Text(text) if text.text.contains("Background tool call"))
                    })
            }),
            "a queued approval was misreported as a background call: {transcript:?}"
        );
    }

    #[tokio::test]
    async fn regression_cancelled_unstarted_call_is_not_tracked_as_detached() {
        let agent = Agent::builder().model(FakeAdapter).build().unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-cancelled-unstarted-call"))
            .await
            .unwrap();

        driver.append_item(Item::new(
            ItemKind::Assistant,
            vec![Part::ToolCall(ToolCallPart {
                id: ToolCallId::new("call-never-started"),
                name: "wait".into(),
                input: json!({}),
                metadata: MetadataMap::new(),
            })],
        ));
        driver
            .finish_cancelled(agentkit_core::TurnId::new("turn-cancelled"), Vec::new())
            .unwrap();

        assert!(
            !driver
                .detached_call_ids
                .contains(&ToolCallId::new("call-never-started")),
            "an unstarted call can never deliver a detached result"
        );
    }

    #[tokio::test]
    async fn loop_resumes_after_approved_tool_request() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-approval"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let first = driver.next().await.unwrap();
        match first {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                assert!(pending.request.task_id.is_some());
                assert_eq!(pending.request.id.0, "approval:fs-read");
                pending.approve(&mut driver).unwrap();
            }
            other => panic!("unexpected loop step: {other:?}"),
        }
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
    async fn approval_gated_tool_does_not_start_before_approval() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-approval-start-event"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        let pending = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => pending,
            other => panic!("unexpected loop step: {other:?}"),
        };

        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .all(|event| !matches!(event, AgentEvent::ToolExecutionStarted(_))),
            "tool start must not be reported before approval"
        );

        pending.approve(&mut driver).unwrap();
        match driver.next().await.unwrap() {
            LoopStep::Finished(_) => {}
            other => panic!("unexpected loop step after approval: {other:?}"),
        }

        let started = events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolExecutionStarted(_)))
            .count();
        assert_eq!(started, 1);
        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Completed));
    }

    #[tokio::test]
    async fn cancelling_pending_approval_resolves_it_and_pairs_tool_result() {
        let controller = CancellationController::new();
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .cancellation(controller.handle())
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-cancel-pending-approval"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_)) => {}
            other => panic!("unexpected loop step: {other:?}"),
        }

        controller.interrupt();

        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Cancelled);
            }
            other => panic!("unexpected loop step after cancel: {other:?}"),
        }

        let events = events.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ApprovalResolved { approved: false })),
            "pending approval cancellation should close approval UI state"
        );
        assert!(
            events.iter().any(|event| matches!(
                event,
                AgentEvent::ToolResultReceived(result)
                    if result.call_id == ToolCallId::new("call-1") && result.is_error
            )),
            "pending approval cancellation should pair the assistant tool_use"
        );
        drop(events);

        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
    }

    #[tokio::test]
    async fn cancelling_sole_foreground_approval_for_call_finishes_turn() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(ToolRegistry::new().with(EchoTool::default()))
            .permissions(ApproveFsReads)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-cancel-foreground-approval-for"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        let call_id = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                pending.request.call_id.expect("approval call id")
            }
            other => panic!("unexpected loop step: {other:?}"),
        };
        driver.cancel_pending_approval_for(call_id).unwrap();

        assert!(driver.lifecycle.active_turn.is_none());
        assert!(driver.pending_approvals.is_empty());
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
        let lifecycle = turn_lifecycle_events(&events.lock().unwrap());
        assert_eq!(lifecycle.len(), 2, "{lifecycle:?}");
        assert_eq!(lifecycle[0].0, lifecycle[1].0);
        assert_eq!(lifecycle[1].1, Some(FinishReason::Cancelled));
    }

    #[tokio::test]
    async fn resolved_approval_runs_even_if_cancellation_also_fired() {
        let controller = CancellationController::new();
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .cancellation(controller.handle())
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-resolved-approval-cancel-race"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        let pending = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => pending,
            other => panic!("unexpected loop step: {other:?}"),
        };

        controller.interrupt();
        pending.approve(&mut driver).unwrap();

        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
                match &turn.items[0].parts[0] {
                    Part::Text(text) => assert_eq!(text.text, "tool said: pong"),
                    other => panic!("unexpected part after approval: {other:?}"),
                }
            }
            other => panic!("unexpected loop step after approved cancel race: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_resumes_with_patched_input_on_approval() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-approval-patched"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                pending
                    .approve_with_patched_input(&mut driver, json!({ "value": "patched" }))
                    .unwrap();
            }
            other => panic!("unexpected loop step: {other:?}"),
        }
        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => match &turn.items[0].parts[0] {
                Part::Text(text) => assert_eq!(text.text, "tool said: patched"),
                other => panic!("unexpected part: {other:?}"),
            },
            other => panic!("unexpected loop step after approval: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_tracks_multiple_pending_approvals_by_call_id() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(DualApprovalAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-dual-approval"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "run both approvals".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let pending_first = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                assert_eq!(
                    pending.request.call_id.as_ref().map(|id| id.0.as_str()),
                    Some("call-1")
                );
                pending
            }
            other => panic!("unexpected first loop step: {other:?}"),
        };

        let pending_second = match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                assert_eq!(
                    pending.request.call_id.as_ref().map(|id| id.0.as_str()),
                    Some("call-2")
                );
                pending
            }
            other => panic!("unexpected second loop step: {other:?}"),
        };

        pending_second.approve(&mut driver).unwrap();
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                assert_eq!(
                    pending.request.call_id.as_ref().map(|id| id.0.as_str()),
                    Some("call-1")
                );
            }
            other => panic!("unexpected step after approving second request: {other:?}"),
        }

        pending_first.approve(&mut driver).unwrap();
        match driver.next().await.unwrap() {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
                match &turn.items[0].parts[0] {
                    Part::Text(text) => assert_eq!(text.text, "both approvals finished"),
                    other => panic!("unexpected final part: {other:?}"),
                }
            }
            other => panic!("unexpected final loop step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn failed_pending_approval_cleanup_repairs_and_finishes_error() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(ToolRegistry::new().with(EchoTool::default()))
            .permissions(ApproveFsReads)
            .task_manager(
                TestTaskManager::new(SimpleTaskManager::new())
                    .fail_interrupt("interrupt cleanup failed"),
            )
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("session-failed-approval-cleanup"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_))
        ));
        let error = driver.cancel_pending_approvals().await.unwrap_err();
        assert!(error.to_string().contains("interrupt cleanup failed"));
        assert!(driver.lifecycle.active_turn.is_none());
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();

        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolResultReceived(result)
                if result.call_id == ToolCallId::new("call-1") && result.is_error
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::TurnFinished(turn) if turn.finish_reason == FinishReason::Error
        )));
    }

    #[tokio::test]
    async fn cancelling_all_pending_approvals_interrupts_every_originating_turn() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let interrupted = StdArc::new(StdMutex::new(Vec::new()));
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(DualApprovalAdapter)
            .add_tool_source(tools)
            .permissions(ApproveFsReads)
            .task_manager(
                TestTaskManager::new(SimpleTaskManager::new())
                    .record_interrupts(interrupted.clone()),
            )
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-dual-approval-cancel"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "run both approvals".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        for expected_call in ["call-1", "call-2"] {
            match driver.next().await.unwrap() {
                LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                    assert_eq!(
                        pending.request.call_id.as_ref().map(|id| id.0.as_str()),
                        Some(expected_call)
                    );
                }
                other => panic!("unexpected approval step: {other:?}"),
            }
        }

        let first_origin = driver
            .pending_approvals
            .get(&ToolCallId::new("call-1"))
            .unwrap()
            .tool_request
            .turn_id
            .clone();
        let second_origin = agentkit_core::TurnId::new("second-originating-turn");
        driver
            .pending_approvals
            .get_mut(&ToolCallId::new("call-2"))
            .unwrap()
            .tool_request
            .turn_id = second_origin.clone();

        match driver.cancel_pending_approvals().await.unwrap() {
            Some(LoopStep::Finished(turn)) => {
                assert_eq!(turn.finish_reason, FinishReason::Cancelled);
            }
            other => panic!("unexpected cancellation result: {other:?}"),
        }
        validate_transcript_invariants(&driver.snapshot().transcript).unwrap();
        let interrupted = interrupted
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(interrupted, HashSet::from([first_origin, second_origin]));

        let events = events.lock().unwrap();
        let cancelled = events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ApprovalResolved { approved: false }))
            .count();
        assert_eq!(cancelled, 2);
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::TurnFinished(turn) if turn.finish_reason == FinishReason::Cancelled
        )));
        for expected_call in ["call-1", "call-2"] {
            assert!(events.iter().any(|event| matches!(
                event,
                AgentEvent::ToolResultReceived(result)
                    if result.call_id == ToolCallId::new(expected_call) && result.is_error
            )));
        }
    }

    #[tokio::test]
    async fn loop_compacts_transcript_before_new_turns() {
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .mutator(KeepRecentMutator { keep: 1 })
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-4"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                    usage: None,
                    finish_reason: None,
                    created_at: None,
                }])
                .unwrap();
            let _ = driver.next().await.unwrap();
        }

        let events = events.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::MutationFinished { dirty: true, .. }))
        );
    }

    #[test]
    fn transcript_validation_rejects_orphaned_tool_result() {
        let transcript = vec![Item {
            id: None,
            kind: ItemKind::Tool,
            parts: vec![Part::ToolResult(ToolResultPart {
                call_id: "call-1".into(),
                output: ToolOutput::Text("result".into()),
                is_error: false,
                metadata: MetadataMap::new(),
            })],
            metadata: MetadataMap::new(),
            usage: None,
            finish_reason: None,
            created_at: None,
        }];

        let error = validate_transcript_invariants(&transcript).unwrap_err();
        assert!(error.to_string().contains("orphaned tool_result"));
    }

    #[test]
    fn transcript_validation_rejects_duplicate_tool_result() {
        let transcript = vec![
            Item {
                id: None,
                kind: ItemKind::Assistant,
                parts: vec![Part::ToolCall(ToolCallPart {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    input: serde_json::json!({}),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            },
            Item {
                id: None,
                kind: ItemKind::Tool,
                parts: vec![Part::ToolResult(ToolResultPart {
                    call_id: "call-1".into(),
                    output: ToolOutput::Text("result".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            },
            Item {
                id: None,
                kind: ItemKind::Tool,
                parts: vec![Part::ToolResult(ToolResultPart {
                    call_id: "call-1".into(),
                    output: ToolOutput::Text("again".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            },
        ];

        let error = validate_transcript_invariants(&transcript).unwrap_err();
        assert!(error.to_string().contains("duplicate tool_result"));
    }

    #[tokio::test]
    async fn loop_refreshes_tool_specs_each_turn() {
        let seen_descriptions = StdArc::new(StdMutex::new(Vec::new()));
        let version = StdArc::new(AtomicUsize::new(1));
        let tools = ToolRegistry::new().with(DynamicSpecTool::new(version.clone()));
        let agent = Agent::builder()
            .model(RecordingAdapter {
                seen_descriptions: seen_descriptions.clone(),
                seen_caches: StdArc::new(StdMutex::new(Vec::new())),
            })
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-dynamic-tools"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                    usage: None,
                    finish_reason: None,
                    created_at: None,
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

    #[tokio::test]
    async fn loop_emits_catalog_change_and_uses_updated_specs_next_turn() {
        let seen_descriptions = StdArc::new(StdMutex::new(Vec::new()));
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let executor = StdArc::new(CatalogExecutor::new());
        let executor_for_agent: Arc<dyn ToolExecutor> = executor.clone();
        let agent = Agent::builder()
            .model(RecordingAdapter {
                seen_descriptions: seen_descriptions.clone(),
                seen_caches: StdArc::new(StdMutex::new(Vec::new())),
            })
            .tool_executor(executor_for_agent)
            .permissions(AllowAllPermissions)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-catalog-events"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "first")])
            .unwrap();
        let _ = driver.next().await.unwrap();

        executor.publish_change(
            1,
            ToolCatalogEvent {
                source: "mcp:mock".into(),
                added: vec!["dynamic".into()],
                removed: Vec::new(),
                changed: Vec::new(),
            },
        );

        driver
            .submit_input(vec![Item::text(ItemKind::User, "second")])
            .unwrap();
        let _ = driver.next().await.unwrap();

        let seen_descriptions = seen_descriptions.lock().unwrap();
        assert_eq!(seen_descriptions.len(), 2);
        assert_eq!(seen_descriptions[0], vec!["dynamic version 0".to_string()]);
        assert_eq!(seen_descriptions[1], vec!["dynamic version 1".to_string()]);

        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolCatalogChanged(ToolCatalogEvent {
                source,
                added,
                removed,
                changed,
            }) if source == "mcp:mock"
                && added == &vec!["dynamic".to_string()]
                && removed.is_empty()
                && changed.is_empty()
        )));
    }

    #[tokio::test]
    async fn loop_passes_session_default_and_next_turn_cache_requests() {
        let seen_caches = StdArc::new(StdMutex::new(Vec::new()));
        let agent = Agent::builder()
            .model(RecordingAdapter {
                seen_descriptions: StdArc::new(StdMutex::new(Vec::new())),
                seen_caches: seen_caches.clone(),
            })
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let default_cache = PromptCacheRequest::best_effort(PromptCacheStrategy::Automatic)
            .with_retention(PromptCacheRetention::Short);
        let override_cache = PromptCacheRequest::required(PromptCacheStrategy::Explicit {
            breakpoints: vec![PromptCacheBreakpoint::TranscriptItemEnd { index: 0 }],
        });

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-cache"),
                metadata: MetadataMap::new(),
                cache: Some(default_cache.clone()),
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "first".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();
        let _ = driver.next().await.unwrap();

        driver
            .submit_input_with_cache(
                vec![Item {
                    id: None,
                    kind: ItemKind::User,
                    parts: vec![Part::Text(TextPart {
                        text: "second".into(),
                        metadata: MetadataMap::new(),
                    })],
                    metadata: MetadataMap::new(),
                    usage: None,
                    finish_reason: None,
                    created_at: None,
                }],
                override_cache.clone(),
            )
            .unwrap();
        let _ = driver.next().await.unwrap();

        let seen = seen_caches.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0], Some(default_cache));
        assert_eq!(seen[1], Some(override_cache));
    }

    #[tokio::test]
    async fn loop_yields_after_tool_result_between_rounds() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("yield-session"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();

        // First next() runs the model turn, resolves the tool call, and
        // yields AfterToolResult before calling the model again.
        let step = driver.next().await.unwrap();
        let info = match step {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => info,
            other => panic!("expected AfterToolResult, got {other:?}"),
        };
        assert_eq!(info.session_id, SessionId::new("yield-session"));
        // Transcript at yield: [User, Assistant(tool_call), Tool(result)]
        assert_eq!(info.transcript_len, 3);

        // The yield is cooperative, not blocking.
        let interrupt = LoopInterrupt::AfterToolResult(info.clone());
        assert!(!interrupt.is_blocking());

        // Host interjects a message mid-turn.
        driver
            .submit_input(vec![Item::text(ItemKind::User, "also: report back")])
            .unwrap();

        // Second next() resumes the turn into the next model call, which
        // sees the tool result (and the injected user message) and finishes.
        let step = driver.next().await.unwrap();
        match step {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
            }
            other => panic!("expected Finished, got {other:?}"),
        }

        // Transcript must now include the injected user message.
        let snapshot = driver.snapshot();
        let has_injected_message = snapshot.transcript.iter().any(|item| {
            item.kind == ItemKind::User
                && item.parts.iter().any(|part| match part {
                    Part::Text(text) => text.text == "also: report back",
                    _ => false,
                })
        });
        assert!(
            has_injected_message,
            "injected user message should be in transcript, got: {:?}",
            snapshot.transcript
        );
    }

    struct RecordingTranscriptObserver {
        items: StdArc<StdMutex<Vec<Item>>>,
    }

    impl TranscriptObserver for RecordingTranscriptObserver {
        fn on_transcript_event(&self, event: TranscriptEvent<'_>) {
            self.items.lock().unwrap().push(event.item.clone());
        }
    }

    #[tokio::test]
    async fn observers_see_full_tool_round() {
        // A turn with one tool call exercises every interesting path:
        //   user input drained -> model output_items (assistant w/ tool call)
        //   -> tool result Item -> next model output_items (assistant text)
        // The LoopObserver should see exactly one ToolResultReceived; the
        // TranscriptObserver should see all four items in transcript order.
        let events = StdArc::new(StdMutex::new(Vec::<AgentEvent>::new()));
        let items = StdArc::new(StdMutex::new(Vec::<Item>::new()));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .add_tool_source(ToolRegistry::new().with(EchoTool::default()))
            .permissions(AllowAllPermissions)
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .transcript_observer(RecordingTranscriptObserver {
                items: items.clone(),
            })
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("observer-session"),
                metadata: MetadataMap::new(),
                cache: None,
                consumer_capabilities: SessionConsumerCapabilities::default(),
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
                usage: None,
                finish_reason: None,
                created_at: None,
            }])
            .unwrap();

        let result = run_until_finished(&mut driver).await;
        assert!(matches!(result, LoopStep::Finished(_)), "got {result:?}");

        // LoopObserver: exactly one ToolResultReceived, with the echo
        // tool's output, correlating back to the model's tool call.
        let events = events.lock().unwrap().clone();
        let tool_call_id = events.iter().find_map(|e| match e {
            AgentEvent::ToolCallRequested(c) => Some(c.id.clone()),
            _ => None,
        });
        let tool_results: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolResultReceived(r) => Some(r.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_results.len(), 1, "events: {events:?}");
        assert_eq!(Some(tool_results[0].call_id.clone()), tool_call_id);
        assert!(!tool_results[0].is_error);

        // TranscriptObserver: every transcript mutation surfaces.
        // Expected order: User("ping"), Assistant(tool call), Tool(result),
        // Assistant("tool said: pong").
        let items = items.lock().unwrap().clone();
        assert_eq!(items.len(), 4, "items: {items:?}");
        assert_eq!(items[0].kind, ItemKind::User);
        assert_eq!(items[1].kind, ItemKind::Assistant);
        assert!(
            items[1]
                .parts
                .iter()
                .any(|p| matches!(p, Part::ToolCall(_)))
        );
        assert_eq!(items[2].kind, ItemKind::Tool);
        assert!(
            items[2]
                .parts
                .iter()
                .any(|p| matches!(p, Part::ToolResult(_)))
        );
        assert_eq!(items[3].kind, ItemKind::Assistant);
    }

    #[test]
    fn convenience_cache_builders_construct_expected_defaults() {
        let cache = PromptCacheRequest::automatic()
            .with_retention(PromptCacheRetention::Short)
            .with_key("workspace:demo");
        let session = SessionConfig::new("demo").with_cache(cache.clone());

        assert_eq!(session.session_id, SessionId::new("demo"));
        assert_eq!(session.cache, Some(cache));

        let explicit = PromptCacheRequest::explicit([
            PromptCacheBreakpoint::tools_end(),
            PromptCacheBreakpoint::transcript_item_end(2),
            PromptCacheBreakpoint::transcript_part_end(3, 1),
        ]);

        assert_eq!(explicit.mode, PromptCacheMode::BestEffort);
        assert_eq!(
            explicit.strategy,
            PromptCacheStrategy::Explicit {
                breakpoints: vec![
                    PromptCacheBreakpoint::ToolsEnd,
                    PromptCacheBreakpoint::TranscriptItemEnd { index: 2 },
                    PromptCacheBreakpoint::TranscriptPartEnd {
                        item_index: 3,
                        part_index: 1,
                    },
                ],
            }
        );
    }
}
