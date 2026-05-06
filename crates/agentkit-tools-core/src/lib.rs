//! Core abstractions for defining, registering, executing, and governing
//! tools in agentkit.
//!
//! This crate provides the [`Tool`] trait, [`ToolRegistry`],
//! [`BasicToolExecutor`], and a layered permission system built on
//! [`PermissionChecker`], [`PermissionPolicy`], and
//! [`CompositePermissionChecker`]. Together these types let you:
//!
//! - **Define tools** by implementing [`Tool`] with a [`ToolSpec`] and
//!   async `invoke` method.
//! - **Register tools** in a [`ToolRegistry`] and hand it to an executor
//!   or capability provider.
//! - **Check permissions** before execution using composable policies
//!   ([`PathPolicy`], [`CommandPolicy`], [`McpServerPolicy`],
//!   [`CustomKindPolicy`]).
//! - **Handle interruptions** (approval prompts) via the
//!   [`ToolInterruption`] / [`ApprovalRequest`] types.
//! - **Bridge to the capability layer** with [`ToolCapabilityProvider`],
//!   which wraps every registered tool as an [`Invocable`].

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use agentkit_capabilities::{
    CapabilityContext, CapabilityError, CapabilityName, CapabilityProvider, Invocable,
    InvocableOutput, InvocableRequest, InvocableResult, InvocableSpec, PromptProvider,
    ResourceProvider,
};
use agentkit_core::{
    ApprovalId, Item, ItemKind, MetadataMap, Part, SessionId, TaskId, ToolCallId, ToolOutput,
    ToolResultPart, TurnCancellation, TurnId,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Re-exports used by the `#[tool]` proc macro so generated code does not
/// require downstream crates to add `async-trait` as a direct dependency.
/// Not part of the public API; the path may change at any time.
#[doc(hidden)]
pub mod __private_async_trait {
    pub use async_trait::async_trait;
}

/// Unique name identifying a [`Tool`] within a [`ToolRegistry`].
///
/// Tool names are used as registry keys and appear in [`ToolRequest`]s to
/// route calls to the correct implementation. Names are compared in a
/// case-sensitive, lexicographic order.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::ToolName;
///
/// let name = ToolName::new("file_read");
/// assert_eq!(name.to_string(), "file_read");
///
/// // Also converts from &str:
/// let name: ToolName = "shell_exec".into();
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ToolName(pub String);

impl ToolName {
    /// Creates a new `ToolName` from any value that converts into a [`String`].
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<&str> for ToolName {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Hints that describe behavioural properties of a tool.
///
/// These flags are advisory — they influence UI presentation and permission
/// policies but do not enforce behaviour at runtime. For example, a
/// permission policy may automatically require approval for tools that
/// set `destructive_hint` to `true`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    /// The tool only reads data and has no side-effects.
    pub read_only_hint: bool,
    /// The tool may perform destructive operations (e.g. file deletion).
    pub destructive_hint: bool,
    /// Repeated calls with the same input produce the same effect.
    pub idempotent_hint: bool,
    /// The tool should prompt for user approval before execution.
    pub needs_approval_hint: bool,
    /// The tool can stream partial results during execution.
    pub supports_streaming_hint: bool,
}

impl ToolAnnotations {
    /// Builds the default advisory flags.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks the tool as read-only.
    pub fn read_only() -> Self {
        Self::default().with_read_only(true)
    }

    /// Marks the tool as destructive.
    pub fn destructive() -> Self {
        Self::default().with_destructive(true)
    }

    /// Marks the tool as requiring approval.
    pub fn needs_approval() -> Self {
        Self::default().with_needs_approval(true)
    }

    /// Marks the tool as supporting streaming.
    pub fn streaming() -> Self {
        Self::default().with_supports_streaming(true)
    }

    pub fn with_read_only(mut self, read_only_hint: bool) -> Self {
        self.read_only_hint = read_only_hint;
        self
    }

    pub fn with_destructive(mut self, destructive_hint: bool) -> Self {
        self.destructive_hint = destructive_hint;
        self
    }

    pub fn with_idempotent(mut self, idempotent_hint: bool) -> Self {
        self.idempotent_hint = idempotent_hint;
        self
    }

    pub fn with_needs_approval(mut self, needs_approval_hint: bool) -> Self {
        self.needs_approval_hint = needs_approval_hint;
        self
    }

    pub fn with_supports_streaming(mut self, supports_streaming_hint: bool) -> Self {
        self.supports_streaming_hint = supports_streaming_hint;
        self
    }
}

/// Declarative specification of a tool's identity, schema, and behavioural hints.
///
/// Every [`Tool`] implementation exposes a `ToolSpec` that the framework uses to
/// advertise the tool to an LLM, validate inputs, and drive permission checks.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::{ToolAnnotations, ToolName, ToolSpec};
/// use serde_json::json;
///
/// let spec = ToolSpec::new(
///     ToolName::new("grep_search"),
///     "Search files by regex pattern",
///     json!({
///         "type": "object",
///         "properties": {
///             "pattern": { "type": "string" },
///             "path": { "type": "string" }
///         },
///         "required": ["pattern"]
///     }),
/// )
/// .with_annotations(ToolAnnotations::read_only());
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Machine-readable name used to route tool calls.
    pub name: ToolName,
    /// Human-readable description sent to the LLM so it knows when to use this tool.
    pub description: String,
    /// JSON Schema describing the expected input object.
    pub input_schema: Value,
    /// Advisory behavioural hints (read-only, destructive, etc.).
    pub annotations: ToolAnnotations,
    /// Arbitrary key-value pairs for framework extensions.
    pub metadata: MetadataMap,
}

/// A change notification for a dynamic tool catalog.
///
/// Dynamic executors, such as MCP-backed executors, use this to tell the
/// agent loop that the model should see a refreshed tool list on the next
/// provider request.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCatalogEvent {
    /// Stable source identifier for the catalog that changed.
    pub source: String,
    /// Tool names that became available.
    pub added: Vec<String>,
    /// Tool names that are no longer available.
    pub removed: Vec<String>,
    /// Tool names whose schema, description, or metadata changed.
    pub changed: Vec<String>,
}

impl ToolCatalogEvent {
    /// Builds a catalog event with empty change sets.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            added: Vec::new(),
            removed: Vec::new(),
            changed: Vec::new(),
        }
    }

    /// Applies `f` to every tool name in `added`, `removed`, and `changed`.
    pub fn for_each_name_mut(&mut self, mut f: impl FnMut(&mut String)) {
        for vec in [&mut self.added, &mut self.removed, &mut self.changed] {
            for name in vec.iter_mut() {
                f(name);
            }
        }
    }

    /// Retains only tool names that pass `predicate` in `added`, `removed`,
    /// and `changed`.
    pub fn retain_names(&mut self, mut predicate: impl FnMut(&str) -> bool) {
        self.added.retain(|n| predicate(n));
        self.removed.retain(|n| predicate(n));
        self.changed.retain(|n| predicate(n));
    }
}

impl ToolSpec {
    /// Builds a tool spec with default annotations and empty metadata.
    pub fn new(
        name: impl Into<ToolName>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            annotations: ToolAnnotations::default(),
            metadata: MetadataMap::new(),
        }
    }

    /// Replaces the tool annotations.
    pub fn with_annotations(mut self, annotations: ToolAnnotations) -> Self {
        self.annotations = annotations;
        self
    }

    /// Replaces the tool metadata.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

/// An incoming request to execute a tool.
///
/// Created by the agent loop when the model emits a tool-call. The
/// [`BasicToolExecutor`] uses `tool_name` to look up the [`Tool`] in the
/// registry and forwards this request to [`Tool::invoke`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolRequest {
    /// Provider-assigned identifier for this specific call.
    pub call_id: ToolCallId,
    /// Name of the tool to invoke (must match a registered [`ToolName`]).
    pub tool_name: ToolName,
    /// JSON input parsed from the model's tool-call arguments.
    pub input: Value,
    /// Session that owns this call.
    pub session_id: SessionId,
    /// Turn within the session that triggered this call.
    pub turn_id: TurnId,
    /// Arbitrary key-value pairs for framework extensions.
    pub metadata: MetadataMap,
}

impl ToolRequest {
    /// Builds a tool request with empty metadata.
    pub fn new(
        call_id: impl Into<ToolCallId>,
        tool_name: impl Into<ToolName>,
        input: Value,
        session_id: impl Into<SessionId>,
        turn_id: impl Into<TurnId>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            input,
            session_id: session_id.into(),
            turn_id: turn_id.into(),
            metadata: MetadataMap::new(),
        }
    }

    /// Replaces the request metadata.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

/// The output produced by a successful tool invocation.
///
/// Returned from [`Tool::invoke`] and wrapped by [`ToolExecutionOutcome::Completed`]
/// after the executor finishes permission checks and execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The content payload sent back to the model.
    pub result: ToolResultPart,
    /// Wall-clock time the tool took to run, if measured.
    pub duration: Option<Duration>,
    /// Arbitrary key-value pairs for framework extensions.
    pub metadata: MetadataMap,
}

impl ToolResult {
    /// Builds a tool result with no duration and empty metadata.
    pub fn new(result: ToolResultPart) -> Self {
        Self {
            result,
            duration: None,
            metadata: MetadataMap::new(),
        }
    }

    /// Sets the measured duration.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Replaces the result metadata.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Trait for dependency injection into tool implementations.
///
/// Tools that need access to shared state (database handles, HTTP clients,
/// configuration, etc.) can downcast the `&dyn ToolResources` provided in
/// [`ToolContext`] to a concrete type.
///
/// The unit type `()` implements `ToolResources` and serves as the default
/// when no shared resources are needed.
///
/// # Example
///
/// ```rust
/// use std::any::Any;
/// use agentkit_tools_core::ToolResources;
///
/// struct AppResources {
///     project_root: std::path::PathBuf,
/// }
///
/// impl ToolResources for AppResources {
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
/// }
/// ```
pub trait ToolResources: Send + Sync {
    /// Returns a reference to `self` as [`Any`] so callers can downcast to
    /// the concrete resource type.
    fn as_any(&self) -> &dyn Any;
}

impl ToolResources for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Runtime context passed to every [`Tool::invoke`] call.
///
/// Provides the tool with access to session/turn metadata, the active
/// permission checker, shared resources, and a cancellation signal so the
/// tool can abort long-running work when a turn is cancelled.
pub struct ToolContext<'a> {
    /// Capability-layer context carrying session and turn identifiers.
    pub capability: CapabilityContext<'a>,
    /// The active permission checker for sub-operations the tool may perform.
    pub permissions: &'a dyn PermissionChecker,
    /// Shared resources (e.g. database handles, config) injected by the host.
    pub resources: &'a dyn ToolResources,
    /// Signal that the current turn has been cancelled by the user.
    pub cancellation: Option<TurnCancellation>,
}

/// Owned execution context that can outlive a single stack frame.
///
/// This is useful for schedulers or task managers that need to move a tool
/// execution onto another task while still constructing the borrowed
/// [`ToolContext`] expected by existing tool implementations.
#[derive(Clone)]
pub struct OwnedToolContext {
    /// Session identifier for the invocation.
    pub session_id: SessionId,
    /// Turn identifier for the invocation.
    pub turn_id: TurnId,
    /// Arbitrary invocation metadata.
    pub metadata: MetadataMap,
    /// Shared permission checker.
    pub permissions: Arc<dyn PermissionChecker>,
    /// Shared resources injected by the host.
    pub resources: Arc<dyn ToolResources>,
    /// Cooperative cancellation signal for the invocation.
    pub cancellation: Option<TurnCancellation>,
}

impl OwnedToolContext {
    /// Creates a borrowed [`ToolContext`] view over this owned context.
    pub fn borrowed(&self) -> ToolContext<'_> {
        ToolContext {
            capability: CapabilityContext {
                session_id: Some(&self.session_id),
                turn_id: Some(&self.turn_id),
                metadata: &self.metadata,
            },
            permissions: self.permissions.as_ref(),
            resources: self.resources.as_ref(),
            cancellation: self.cancellation.clone(),
        }
    }
}

/// A description of an operation that requires permission before it can proceed.
///
/// Tool implementations return `PermissionRequest` objects from
/// [`Tool::proposed_requests`] so the executor can evaluate them against the
/// active [`PermissionChecker`] before invoking the tool.
///
/// Built-in implementations include [`ShellPermissionRequest`],
/// [`FileSystemPermissionRequest`], and [`McpPermissionRequest`].
///
/// # Implementing a custom request
///
/// ```rust
/// use std::any::Any;
/// use agentkit_core::MetadataMap;
/// use agentkit_tools_core::PermissionRequest;
///
/// struct NetworkPermissionRequest {
///     url: String,
///     metadata: MetadataMap,
/// }
///
/// impl PermissionRequest for NetworkPermissionRequest {
///     fn kind(&self) -> &'static str { "network.http" }
///     fn summary(&self) -> String { format!("HTTP request to {}", self.url) }
///     fn metadata(&self) -> &MetadataMap { &self.metadata }
///     fn as_any(&self) -> &dyn Any { self }
/// }
/// ```
pub trait PermissionRequest: Send + Sync {
    /// A dot-separated category string (e.g. `"filesystem.write"`, `"shell.command"`).
    fn kind(&self) -> &'static str;
    /// Human-readable one-line description of what is being requested.
    fn summary(&self) -> String;
    /// Arbitrary metadata attached to this request.
    fn metadata(&self) -> &MetadataMap;
    /// Returns `self` as [`Any`] so policies can downcast to the concrete type.
    fn as_any(&self) -> &dyn Any;
}

/// Machine-readable code indicating why a permission was denied.
///
/// Returned inside a [`PermissionDenial`] so callers can programmatically
/// react to specific denial categories.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionCode {
    /// A filesystem path is outside the allowed set.
    PathNotAllowed,
    /// A shell command or executable is not permitted.
    CommandNotAllowed,
    /// A network operation is not permitted.
    NetworkNotAllowed,
    /// An MCP server is not in the trusted set.
    ServerNotTrusted,
    /// An MCP auth scope is not in the allowed set.
    AuthScopeNotAllowed,
    /// A custom permission policy explicitly denied the request.
    CustomPolicyDenied,
    /// No policy recognised the request kind.
    UnknownRequest,
}

/// Structured denial produced when a [`PermissionChecker`] rejects an operation.
///
/// Contains a machine-readable [`PermissionCode`] and a human-readable
/// message suitable for logging or displaying to the user.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionDenial {
    /// Machine-readable denial category.
    pub code: PermissionCode,
    /// Human-readable explanation of why the operation was denied.
    pub message: String,
    /// Arbitrary metadata carried from the original request.
    pub metadata: MetadataMap,
}

/// Why a permission policy is requesting human approval before proceeding.
///
/// Used inside [`ApprovalRequest`] so the UI layer can display context-appropriate
/// prompts to the user.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalReason {
    /// The active policy always requires confirmation for this kind of operation.
    PolicyRequiresConfirmation,
    /// The operation was flagged as higher risk than usual.
    EscalatedRisk,
    /// The target (server, path, etc.) was not recognised by any policy.
    UnknownTarget,
    /// The operation targets a filesystem path that is not in the allowed set.
    SensitivePath,
    /// The shell command is not in the pre-approved allow-list.
    SensitiveCommand,
    /// The MCP server is not in the trusted set.
    SensitiveServer,
    /// The MCP auth scope is not in the pre-approved set.
    SensitiveAuthScope,
}

/// A request sent to the host when a tool execution needs human approval.
///
/// The agent loop surfaces this to the user. Once the user responds, the
/// loop can re-submit the tool call via [`ToolExecutor::execute_approved`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Runtime task identifier associated with this approval request, if any.
    pub task_id: Option<TaskId>,
    /// The originating tool call id when this approval was raised from a
    /// tool invocation. Hosts can use this to resolve specific approvals.
    pub call_id: Option<ToolCallId>,
    /// Stable identifier so the executor can match the approval to its request.
    pub id: ApprovalId,
    /// The [`PermissionRequest::kind`] string that triggered the approval flow.
    pub request_kind: String,
    /// Why approval is needed.
    pub reason: ApprovalReason,
    /// Human-readable summary shown to the user.
    pub summary: String,
    /// Arbitrary metadata carried from the original permission request.
    pub metadata: MetadataMap,
}

impl ApprovalRequest {
    /// Builds an approval request with no task or call id.
    pub fn new(
        id: impl Into<ApprovalId>,
        request_kind: impl Into<String>,
        reason: ApprovalReason,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            task_id: None,
            call_id: None,
            id: id.into(),
            request_kind: request_kind.into(),
            reason,
            summary: summary.into(),
            metadata: MetadataMap::new(),
        }
    }

    /// Sets the associated task id.
    pub fn with_task_id(mut self, task_id: impl Into<TaskId>) -> Self {
        self.task_id = Some(task_id.into());
        self
    }

    /// Sets the associated tool call id.
    pub fn with_call_id(mut self, call_id: impl Into<ToolCallId>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// Replaces the approval metadata.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

/// The user's response to an [`ApprovalRequest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    /// The user approved the operation.
    Approve,
    /// The user denied the operation, optionally with a reason.
    Deny {
        /// Optional human-readable explanation for the denial.
        reason: Option<String>,
    },
}

/// A tool execution was paused because it needs external input.
///
/// The agent loop should handle the interruption (show a prompt, etc.) and
/// then re-submit the tool call. Source-specific interruptions (e.g. MCP
/// auth challenges) do not surface here — they are resolved by responders
/// registered with the source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolInterruption {
    /// The operation requires human approval before it can proceed.
    ApprovalRequired(ApprovalRequest),
}

/// The verdict from a [`PermissionChecker`] for a single [`PermissionRequest`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionDecision {
    /// The operation is allowed to proceed.
    Allow,
    /// The operation is denied.
    Deny(PermissionDenial),
    /// The operation may proceed only after the user approves.
    RequireApproval(ApprovalRequest),
}

/// Evaluates a [`PermissionRequest`] and returns a final [`PermissionDecision`].
///
/// The [`BasicToolExecutor`] calls `evaluate` for every permission request
/// returned by [`Tool::proposed_requests`] before invoking the tool. If any
/// request is denied, execution is aborted; if any request requires approval,
/// the executor returns a [`ToolInterruption`].
///
/// For composing multiple policies, see [`CompositePermissionChecker`].
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::{PermissionChecker, PermissionDecision, PermissionRequest};
///
/// /// A checker that allows every operation unconditionally.
/// struct AllowAll;
///
/// impl PermissionChecker for AllowAll {
///     fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
///         PermissionDecision::Allow
///     }
/// }
/// ```
pub trait PermissionChecker: Send + Sync {
    /// Evaluate a single permission request and return the decision.
    fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision;
}

/// A [`PermissionChecker`] that unconditionally allows every operation.
///
/// Useful in tests, examples, and embedding scenarios where the host has
/// already gated tool access elsewhere.
#[derive(Copy, Clone, Debug, Default)]
pub struct AllowAllPermissions;

impl PermissionChecker for AllowAllPermissions {
    fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

/// The result of a single [`PermissionPolicy`] evaluation.
///
/// Unlike [`PermissionDecision`], a policy can return [`PolicyMatch::NoOpinion`]
/// to indicate it has nothing to say about this request kind, letting other
/// policies in the [`CompositePermissionChecker`] chain decide.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyMatch {
    /// This policy does not apply to the given request kind.
    NoOpinion,
    /// This policy explicitly allows the operation.
    Allow,
    /// This policy explicitly denies the operation.
    Deny(PermissionDenial),
    /// This policy requires user approval before the operation can proceed.
    RequireApproval(ApprovalRequest),
}

/// A single, focused permission rule that contributes to a composite decision.
///
/// Policies are combined inside a [`CompositePermissionChecker`]. Each policy
/// inspects the request and either returns a definitive answer or
/// [`PolicyMatch::NoOpinion`] to defer.
///
/// Built-in policies: [`PathPolicy`], [`CommandPolicy`], [`McpServerPolicy`],
/// [`CustomKindPolicy`].
pub trait PermissionPolicy: Send + Sync {
    /// Evaluate the request and return a match or [`PolicyMatch::NoOpinion`].
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch;
}

/// Chains multiple [`PermissionPolicy`] implementations into a single [`PermissionChecker`].
///
/// Policies are evaluated in registration order. The first `Deny` short-circuits
/// immediately. If any policy returns `RequireApproval`, that is used unless a
/// later policy denies. If at least one policy returns `Allow` and none deny or
/// require approval, the result is `Allow`. Otherwise the `fallback` decision
/// is returned.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::{
///     CommandPolicy, CompositePermissionChecker, PathPolicy, PermissionDecision,
/// };
///
/// let checker = CompositePermissionChecker::new(PermissionDecision::Allow)
///     .with_policy(PathPolicy::new().allow_root("/workspace"))
///     .with_policy(CommandPolicy::new().allow_executable("git"));
/// ```
pub struct CompositePermissionChecker {
    policies: Vec<Box<dyn PermissionPolicy>>,
    fallback: PermissionDecision,
}

impl CompositePermissionChecker {
    /// Creates a new composite checker with the given fallback decision.
    ///
    /// The fallback is used when no policy has an opinion about a request.
    ///
    /// # Arguments
    ///
    /// * `fallback` - Decision returned when every policy returns [`PolicyMatch::NoOpinion`].
    pub fn new(fallback: PermissionDecision) -> Self {
        Self {
            policies: Vec::new(),
            fallback,
        }
    }

    /// Appends a policy to the evaluation chain and returns `self` for chaining.
    pub fn with_policy(mut self, policy: impl PermissionPolicy + 'static) -> Self {
        self.policies.push(Box::new(policy));
        self
    }
}

impl PermissionChecker for CompositePermissionChecker {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision {
        let mut saw_allow = false;
        let mut approval = None;

        for policy in &self.policies {
            match policy.evaluate(request) {
                PolicyMatch::NoOpinion => {}
                PolicyMatch::Allow => saw_allow = true,
                PolicyMatch::Deny(denial) => return PermissionDecision::Deny(denial),
                PolicyMatch::RequireApproval(req) => approval = Some(req),
            }
        }

        if let Some(req) = approval {
            PermissionDecision::RequireApproval(req)
        } else if saw_allow {
            PermissionDecision::Allow
        } else {
            self.fallback.clone()
        }
    }
}

/// Permission request for executing a shell command.
///
/// Evaluated by [`CommandPolicy`] to decide whether the executable, arguments,
/// working directory, and environment variables are acceptable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellPermissionRequest {
    /// The executable name or path (e.g. `"git"`, `"/usr/bin/curl"`).
    pub executable: String,
    /// Command-line arguments passed to the executable.
    pub argv: Vec<String>,
    /// Working directory for the command, if specified.
    pub cwd: Option<PathBuf>,
    /// Names of environment variables the command will receive.
    pub env_keys: Vec<String>,
    /// Arbitrary metadata for policy extensions.
    pub metadata: MetadataMap,
}

impl PermissionRequest for ShellPermissionRequest {
    fn kind(&self) -> &'static str {
        "shell.command"
    }

    fn summary(&self) -> String {
        if self.argv.is_empty() {
            self.executable.clone()
        } else {
            format!("{} {}", self.executable, self.argv.join(" "))
        }
    }

    fn metadata(&self) -> &MetadataMap {
        &self.metadata
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Permission request for a filesystem operation.
///
/// Evaluated by [`PathPolicy`] to decide whether the target path(s) fall
/// within allowed or protected directory roots.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileSystemPermissionRequest {
    /// Read a file's contents.
    Read {
        path: PathBuf,
        metadata: MetadataMap,
    },
    /// Write (create or overwrite) a file.
    Write {
        path: PathBuf,
        metadata: MetadataMap,
    },
    /// Edit (modify in place) an existing file.
    Edit {
        path: PathBuf,
        metadata: MetadataMap,
    },
    /// Delete a file or directory.
    Delete {
        path: PathBuf,
        metadata: MetadataMap,
    },
    /// Move or rename a file.
    Move {
        from: PathBuf,
        to: PathBuf,
        metadata: MetadataMap,
    },
    /// List directory contents.
    List {
        path: PathBuf,
        metadata: MetadataMap,
    },
    /// Create a directory (including parents).
    CreateDir {
        path: PathBuf,
        metadata: MetadataMap,
    },
}

impl FileSystemPermissionRequest {
    fn metadata_map(&self) -> &MetadataMap {
        match self {
            Self::Read { metadata, .. }
            | Self::Write { metadata, .. }
            | Self::Edit { metadata, .. }
            | Self::Delete { metadata, .. }
            | Self::Move { metadata, .. }
            | Self::List { metadata, .. }
            | Self::CreateDir { metadata, .. } => metadata,
        }
    }
}

impl PermissionRequest for FileSystemPermissionRequest {
    fn kind(&self) -> &'static str {
        match self {
            Self::Read { .. } => "filesystem.read",
            Self::Write { .. } => "filesystem.write",
            Self::Edit { .. } => "filesystem.edit",
            Self::Delete { .. } => "filesystem.delete",
            Self::Move { .. } => "filesystem.move",
            Self::List { .. } => "filesystem.list",
            Self::CreateDir { .. } => "filesystem.mkdir",
        }
    }

    fn summary(&self) -> String {
        match self {
            Self::Read { path, .. } => format!("Read {}", path.display()),
            Self::Write { path, .. } => format!("Write {}", path.display()),
            Self::Edit { path, .. } => format!("Edit {}", path.display()),
            Self::Delete { path, .. } => format!("Delete {}", path.display()),
            Self::Move { from, to, .. } => {
                format!("Move {} to {}", from.display(), to.display())
            }
            Self::List { path, .. } => format!("List {}", path.display()),
            Self::CreateDir { path, .. } => format!("Create directory {}", path.display()),
        }
    }

    fn metadata(&self) -> &MetadataMap {
        self.metadata_map()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Permission request for an MCP (Model Context Protocol) operation.
///
/// Evaluated by [`McpServerPolicy`] to decide whether the target server is
/// trusted and the requested auth scopes are allowed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpPermissionRequest {
    /// Connect to an MCP server.
    Connect {
        server_id: String,
        metadata: MetadataMap,
    },
    /// Invoke a tool exposed by an MCP server.
    InvokeTool {
        server_id: String,
        tool_name: String,
        metadata: MetadataMap,
    },
    /// Read a resource from an MCP server.
    ReadResource {
        server_id: String,
        resource_id: String,
        metadata: MetadataMap,
    },
    /// Fetch a prompt template from an MCP server.
    FetchPrompt {
        server_id: String,
        prompt_id: String,
        metadata: MetadataMap,
    },
    /// Request an auth scope on an MCP server.
    UseAuthScope {
        server_id: String,
        scope: String,
        metadata: MetadataMap,
    },
}

impl McpPermissionRequest {
    fn metadata_map(&self) -> &MetadataMap {
        match self {
            Self::Connect { metadata, .. }
            | Self::InvokeTool { metadata, .. }
            | Self::ReadResource { metadata, .. }
            | Self::FetchPrompt { metadata, .. }
            | Self::UseAuthScope { metadata, .. } => metadata,
        }
    }
}

impl PermissionRequest for McpPermissionRequest {
    fn kind(&self) -> &'static str {
        match self {
            Self::Connect { .. } => "mcp.connect",
            Self::InvokeTool { .. } => "mcp.invoke_tool",
            Self::ReadResource { .. } => "mcp.read_resource",
            Self::FetchPrompt { .. } => "mcp.fetch_prompt",
            Self::UseAuthScope { .. } => "mcp.use_auth_scope",
        }
    }

    fn summary(&self) -> String {
        match self {
            Self::Connect { server_id, .. } => format!("Connect MCP server {server_id}"),
            Self::InvokeTool {
                server_id,
                tool_name,
                ..
            } => format!("Invoke MCP tool {server_id}.{tool_name}"),
            Self::ReadResource {
                server_id,
                resource_id,
                ..
            } => format!("Read MCP resource {server_id}:{resource_id}"),
            Self::FetchPrompt {
                server_id,
                prompt_id,
                ..
            } => format!("Fetch MCP prompt {server_id}:{prompt_id}"),
            Self::UseAuthScope {
                server_id, scope, ..
            } => format!("Use MCP auth scope {server_id}:{scope}"),
        }
    }

    fn metadata(&self) -> &MetadataMap {
        self.metadata_map()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A [`PermissionPolicy`] that matches requests whose [`PermissionRequest::kind`]
/// starts with `"custom."` and allows or denies them by name.
///
/// Use this to govern application-defined permission categories without
/// writing a full policy implementation.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::CustomKindPolicy;
///
/// let policy = CustomKindPolicy::new(true)
///     .allow_kind("custom.analytics")
///     .deny_kind("custom.billing");
/// ```
pub struct CustomKindPolicy {
    allowed_kinds: BTreeSet<String>,
    denied_kinds: BTreeSet<String>,
    require_approval_by_default: bool,
}

impl CustomKindPolicy {
    /// Creates a new policy.
    ///
    /// # Arguments
    ///
    /// * `require_approval_by_default` - When `true`, unrecognised `custom.*`
    ///   kinds require approval instead of returning [`PolicyMatch::NoOpinion`].
    pub fn new(require_approval_by_default: bool) -> Self {
        Self {
            allowed_kinds: BTreeSet::new(),
            denied_kinds: BTreeSet::new(),
            require_approval_by_default,
        }
    }

    /// Adds a kind string to the allow-list.
    pub fn allow_kind(mut self, kind: impl Into<String>) -> Self {
        self.allowed_kinds.insert(kind.into());
        self
    }

    /// Adds a kind string to the deny-list.
    pub fn deny_kind(mut self, kind: impl Into<String>) -> Self {
        self.denied_kinds.insert(kind.into());
        self
    }
}

impl PermissionPolicy for CustomKindPolicy {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch {
        let kind = request.kind();
        if !kind.starts_with("custom.") {
            return PolicyMatch::NoOpinion;
        }
        if self.denied_kinds.contains(kind) {
            return PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::CustomPolicyDenied,
                message: format!("custom permission kind {kind} is denied"),
                metadata: request.metadata().clone(),
            });
        }
        if self.allowed_kinds.contains(kind) {
            return PolicyMatch::Allow;
        }
        if self.require_approval_by_default {
            PolicyMatch::RequireApproval(ApprovalRequest {
                task_id: None,
                call_id: None,
                id: ApprovalId::new(format!("approval:{kind}")),
                request_kind: kind.to_string(),
                reason: ApprovalReason::PolicyRequiresConfirmation,
                summary: request.summary(),
                metadata: request.metadata().clone(),
            })
        } else {
            PolicyMatch::NoOpinion
        }
    }
}

/// A [`PermissionPolicy`] that governs [`FileSystemPermissionRequest`]s by
/// checking whether target paths fall within allowed or protected directory trees.
///
/// Protected roots take priority: any path under a protected root is denied
/// immediately. Paths under an allowed root are permitted. Paths outside both
/// sets either require approval or are denied, depending on
/// `require_approval_outside_allowed`.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::PathPolicy;
///
/// let policy = PathPolicy::new()
///     .allow_root("/workspace/project")
///     .read_only_root("/workspace/project/vendor")
///     .protect_root("/workspace/project/.env")
///     .require_approval_outside_allowed(true);
/// ```
pub struct PathPolicy {
    allowed_roots: Vec<CanonicalRoot>,
    read_only_roots: Vec<CanonicalRoot>,
    protected_roots: Vec<CanonicalRoot>,
    require_approval_outside_allowed: bool,
}

impl PathPolicy {
    /// Creates a new path policy with no roots and approval required for
    /// paths outside allowed roots.
    pub fn new() -> Self {
        Self {
            allowed_roots: Vec::new(),
            read_only_roots: Vec::new(),
            protected_roots: Vec::new(),
            require_approval_outside_allowed: true,
        }
    }

    /// Adds a directory tree that filesystem operations are allowed to target.
    pub fn allow_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.allowed_roots.push(CanonicalRoot::new(root.into()));
        self
    }

    /// Adds a directory tree that may be read or listed but not mutated.
    pub fn read_only_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.read_only_roots.push(CanonicalRoot::new(root.into()));
        self
    }

    /// Adds a directory tree that filesystem operations are never allowed to target.
    pub fn protect_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.protected_roots.push(CanonicalRoot::new(root.into()));
        self
    }

    /// When `true` (the default), paths outside allowed roots trigger an
    /// approval request instead of an outright denial.
    pub fn require_approval_outside_allowed(mut self, value: bool) -> Self {
        self.require_approval_outside_allowed = value;
        self
    }
}

impl Default for PathPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolves `path` for symlink-safe containment checks; falls back to the
/// lexically-absolute path so policy decisions stay deterministic when no
/// component on disk yet exists.
fn resolve_canonical(path: &Path) -> PathBuf {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    canonicalize_with_partial_fallback(&abs).unwrap_or(abs)
}

fn canonicalize_with_partial_fallback(abs: &Path) -> Option<PathBuf> {
    if let Ok(canonical) = std::fs::canonicalize(abs) {
        return Some(canonical);
    }
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut current = abs.to_path_buf();
    loop {
        let name = current.file_name().map(|n| n.to_os_string())?;
        tail.push(name);
        if !current.pop() {
            return None;
        }
        if let Ok(canonical) = std::fs::canonicalize(&current) {
            let mut out = canonical;
            for seg in tail.iter().rev() {
                out.push(seg);
            }
            return Some(out);
        }
    }
}

/// A configured root with a lazily-cached canonical form.
///
/// Roots can be registered before they exist on disk; we only memoise once
/// `fs::canonicalize` succeeds, so symlink changes to not-yet-existent
/// components are still picked up on later evaluations.
struct CanonicalRoot {
    lexical: PathBuf,
    canonical: OnceLock<PathBuf>,
}

impl CanonicalRoot {
    fn new(lexical: PathBuf) -> Self {
        Self {
            lexical,
            canonical: OnceLock::new(),
        }
    }

    fn resolve(&self) -> std::borrow::Cow<'_, Path> {
        if let Some(canonical) = self.canonical.get() {
            return std::borrow::Cow::Borrowed(canonical);
        }
        let abs = std::path::absolute(&self.lexical).unwrap_or_else(|_| self.lexical.clone());
        if let Ok(canonical) = std::fs::canonicalize(&abs) {
            let _ = self.canonical.set(canonical);
            return std::borrow::Cow::Borrowed(self.canonical.get().unwrap());
        }
        std::borrow::Cow::Owned(canonicalize_with_partial_fallback(&abs).unwrap_or(abs))
    }
}

impl PermissionPolicy for PathPolicy {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch {
        let Some(fs) = request
            .as_any()
            .downcast_ref::<FileSystemPermissionRequest>()
        else {
            return PolicyMatch::NoOpinion;
        };

        let raw_paths: Vec<&Path> = match fs {
            FileSystemPermissionRequest::Move { from, to, .. } => {
                vec![from.as_path(), to.as_path()]
            }
            FileSystemPermissionRequest::Read { path, .. }
            | FileSystemPermissionRequest::Write { path, .. }
            | FileSystemPermissionRequest::Edit { path, .. }
            | FileSystemPermissionRequest::Delete { path, .. }
            | FileSystemPermissionRequest::List { path, .. }
            | FileSystemPermissionRequest::CreateDir { path, .. } => vec![path.as_path()],
        };

        let candidate_paths: Vec<PathBuf> =
            raw_paths.iter().map(|p| resolve_canonical(p)).collect();

        let mutates = matches!(
            fs,
            FileSystemPermissionRequest::Write { .. }
                | FileSystemPermissionRequest::Edit { .. }
                | FileSystemPermissionRequest::Delete { .. }
                | FileSystemPermissionRequest::Move { .. }
                | FileSystemPermissionRequest::CreateDir { .. }
        );

        if candidate_paths.iter().any(|path| {
            self.protected_roots
                .iter()
                .any(|root| path.starts_with(root.resolve().as_ref()))
        }) {
            return PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::PathNotAllowed,
                message: format!("path access denied for {}", fs.summary()),
                metadata: fs.metadata().clone(),
            });
        }

        if mutates
            && candidate_paths.iter().any(|path| {
                self.read_only_roots
                    .iter()
                    .any(|root| path.starts_with(root.resolve().as_ref()))
            })
        {
            return PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::PathNotAllowed,
                message: format!("path is read-only for {}", fs.summary()),
                metadata: fs.metadata().clone(),
            });
        }

        if self.allowed_roots.is_empty() {
            return PolicyMatch::NoOpinion;
        }

        let all_allowed = candidate_paths.iter().all(|path| {
            self.allowed_roots
                .iter()
                .any(|root| path.starts_with(root.resolve().as_ref()))
        });

        if all_allowed {
            PolicyMatch::Allow
        } else if self.require_approval_outside_allowed {
            PolicyMatch::RequireApproval(ApprovalRequest {
                task_id: None,
                call_id: None,
                id: ApprovalId::new(format!("approval:{}", fs.kind())),
                request_kind: fs.kind().to_string(),
                reason: ApprovalReason::SensitivePath,
                summary: fs.summary(),
                metadata: fs.metadata().clone(),
            })
        } else {
            PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::PathNotAllowed,
                message: format!("path outside allowed roots for {}", fs.summary()),
                metadata: fs.metadata().clone(),
            })
        }
    }
}

/// A [`PermissionPolicy`] that governs [`ShellPermissionRequest`]s by checking
/// the executable name, working directory, and environment variables.
///
/// Denied executables and env keys are rejected immediately. Allowed
/// executables pass. Unknown executables either require approval or are
/// denied, depending on `require_approval_for_unknown`.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::CommandPolicy;
///
/// let policy = CommandPolicy::new()
///     .allow_executable("git")
///     .allow_executable("cargo")
///     .deny_executable("rm")
///     .deny_env_key("AWS_SECRET_ACCESS_KEY")
///     .allow_cwd("/workspace")
///     .require_approval_for_unknown(true);
/// ```
pub struct CommandPolicy {
    allowed_executables: BTreeSet<String>,
    denied_executables: BTreeSet<String>,
    allowed_cwds: Vec<PathBuf>,
    denied_env_keys: BTreeSet<String>,
    require_approval_for_unknown: bool,
}

impl CommandPolicy {
    /// Creates a new command policy with no rules and approval required
    /// for unknown executables.
    pub fn new() -> Self {
        Self {
            allowed_executables: BTreeSet::new(),
            denied_executables: BTreeSet::new(),
            allowed_cwds: Vec::new(),
            denied_env_keys: BTreeSet::new(),
            require_approval_for_unknown: true,
        }
    }

    /// Adds an executable name to the allow-list.
    pub fn allow_executable(mut self, executable: impl Into<String>) -> Self {
        self.allowed_executables.insert(executable.into());
        self
    }

    /// Adds an executable name to the deny-list.
    pub fn deny_executable(mut self, executable: impl Into<String>) -> Self {
        self.denied_executables.insert(executable.into());
        self
    }

    /// Adds a directory root that commands are allowed to run in.
    pub fn allow_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.allowed_cwds.push(cwd.into());
        self
    }

    /// Adds an environment variable name to the deny-list.
    pub fn deny_env_key(mut self, key: impl Into<String>) -> Self {
        self.denied_env_keys.insert(key.into());
        self
    }

    /// When `true` (the default), executables not in the allow-list trigger
    /// an approval request instead of an outright denial.
    pub fn require_approval_for_unknown(mut self, value: bool) -> Self {
        self.require_approval_for_unknown = value;
        self
    }
}

impl Default for CommandPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionPolicy for CommandPolicy {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch {
        let Some(shell) = request.as_any().downcast_ref::<ShellPermissionRequest>() else {
            return PolicyMatch::NoOpinion;
        };

        if self.denied_executables.contains(&shell.executable)
            || shell
                .env_keys
                .iter()
                .any(|key| self.denied_env_keys.contains(key))
        {
            return PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::CommandNotAllowed,
                message: format!("command denied for {}", shell.summary()),
                metadata: shell.metadata().clone(),
            });
        }

        if let Some(cwd) = &shell.cwd
            && !self.allowed_cwds.is_empty()
            && !self.allowed_cwds.iter().any(|root| cwd.starts_with(root))
        {
            return PolicyMatch::RequireApproval(ApprovalRequest {
                task_id: None,
                call_id: None,
                id: ApprovalId::new("approval:shell.cwd"),
                request_kind: shell.kind().to_string(),
                reason: ApprovalReason::SensitiveCommand,
                summary: shell.summary(),
                metadata: shell.metadata().clone(),
            });
        }

        if self.allowed_executables.is_empty()
            || self.allowed_executables.contains(&shell.executable)
        {
            PolicyMatch::Allow
        } else if self.require_approval_for_unknown {
            PolicyMatch::RequireApproval(ApprovalRequest {
                task_id: None,
                call_id: None,
                id: ApprovalId::new("approval:shell.command"),
                request_kind: shell.kind().to_string(),
                reason: ApprovalReason::SensitiveCommand,
                summary: shell.summary(),
                metadata: shell.metadata().clone(),
            })
        } else {
            PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::CommandNotAllowed,
                message: format!("executable {} is not allowed", shell.executable),
                metadata: shell.metadata().clone(),
            })
        }
    }
}

/// A [`PermissionPolicy`] that governs [`McpPermissionRequest`]s by checking
/// whether the target server is trusted and the requested auth scopes are
/// in the allow-list.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::McpServerPolicy;
///
/// let policy = McpServerPolicy::new()
///     .trust_server("github-mcp")
///     .allow_auth_scope("repo:read");
/// ```
pub struct McpServerPolicy {
    trusted_servers: BTreeSet<String>,
    allowed_auth_scopes: BTreeSet<String>,
    require_approval_for_untrusted: bool,
}

impl McpServerPolicy {
    /// Creates a new MCP server policy with approval required for untrusted
    /// servers.
    pub fn new() -> Self {
        Self {
            trusted_servers: BTreeSet::new(),
            allowed_auth_scopes: BTreeSet::new(),
            require_approval_for_untrusted: true,
        }
    }

    /// Marks a server as trusted so operations targeting it are allowed.
    pub fn trust_server(mut self, server_id: impl Into<String>) -> Self {
        self.trusted_servers.insert(server_id.into());
        self
    }

    /// Adds an auth scope to the allow-list.
    pub fn allow_auth_scope(mut self, scope: impl Into<String>) -> Self {
        self.allowed_auth_scopes.insert(scope.into());
        self
    }
}

impl Default for McpServerPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionPolicy for McpServerPolicy {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch {
        let Some(mcp) = request.as_any().downcast_ref::<McpPermissionRequest>() else {
            return PolicyMatch::NoOpinion;
        };

        let server_id = match mcp {
            McpPermissionRequest::Connect { server_id, .. }
            | McpPermissionRequest::InvokeTool { server_id, .. }
            | McpPermissionRequest::ReadResource { server_id, .. }
            | McpPermissionRequest::FetchPrompt { server_id, .. }
            | McpPermissionRequest::UseAuthScope { server_id, .. } => server_id,
        };

        if !self.trusted_servers.is_empty() && !self.trusted_servers.contains(server_id) {
            return if self.require_approval_for_untrusted {
                PolicyMatch::RequireApproval(ApprovalRequest {
                    task_id: None,
                    call_id: None,
                    id: ApprovalId::new(format!("approval:mcp:{server_id}")),
                    request_kind: mcp.kind().to_string(),
                    reason: ApprovalReason::SensitiveServer,
                    summary: mcp.summary(),
                    metadata: mcp.metadata().clone(),
                })
            } else {
                PolicyMatch::Deny(PermissionDenial {
                    code: PermissionCode::ServerNotTrusted,
                    message: format!("MCP server {server_id} is not trusted"),
                    metadata: mcp.metadata().clone(),
                })
            };
        }

        if let McpPermissionRequest::UseAuthScope { scope, .. } = mcp
            && !self.allowed_auth_scopes.is_empty()
            && !self.allowed_auth_scopes.contains(scope)
        {
            return PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::AuthScopeNotAllowed,
                message: format!("MCP auth scope {scope} is not allowed"),
                metadata: mcp.metadata().clone(),
            });
        }

        PolicyMatch::Allow
    }
}

/// The central abstraction for an executable tool in an agentkit agent.
///
/// Implement this trait to define a tool that an LLM can call. Each tool
/// provides a [`ToolSpec`] describing its name, schema, and hints, optional
/// permission requests via [`proposed_requests`](Tool::proposed_requests),
/// and the actual execution logic in [`invoke`](Tool::invoke).
///
/// # Example
///
/// ```rust
/// use agentkit_core::{MetadataMap, ToolOutput, ToolResultPart};
/// use agentkit_tools_core::{
///     Tool, ToolContext, ToolError, ToolName, ToolRequest, ToolResult, ToolSpec,
/// };
/// use async_trait::async_trait;
/// use serde_json::json;
///
/// struct TimeTool {
///     spec: ToolSpec,
/// }
///
/// impl TimeTool {
///     fn new() -> Self {
///         Self {
///             spec: ToolSpec::new(
///                 ToolName::new("current_time"),
///                 "Returns the current UTC time",
///                 json!({ "type": "object" }),
///             ),
///         }
///     }
/// }
///
/// #[async_trait]
/// impl Tool for TimeTool {
///     fn spec(&self) -> &ToolSpec {
///         &self.spec
///     }
///
///     async fn invoke(
///         &self,
///         request: ToolRequest,
///         _ctx: &mut ToolContext<'_>,
///     ) -> Result<ToolResult, ToolError> {
///         Ok(ToolResult::new(ToolResultPart::success(
///             request.call_id,
///             ToolOutput::text("2026-03-22T12:00:00Z"),
///         )))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the static specification for this tool.
    fn spec(&self) -> &ToolSpec;

    /// Returns the current specification for this tool, if it should be
    /// advertised right now.
    ///
    /// Most tools are static and can rely on the default implementation,
    /// which clones [`spec`](Self::spec). Override this when the description
    /// or input schema should reflect runtime state, or when the tool should
    /// be temporarily hidden from the model.
    fn current_spec(&self) -> Option<ToolSpec> {
        Some(self.spec().clone())
    }

    /// Returns permission requests the executor should evaluate before calling
    /// [`invoke`](Tool::invoke).
    ///
    /// The default implementation returns an empty list (no permissions needed).
    /// Override this to declare filesystem, shell, or custom permission
    /// requirements based on the incoming request.
    ///
    /// # Errors
    ///
    /// Return [`ToolError::InvalidInput`] if the request input is malformed
    /// and permission requests cannot be constructed.
    fn proposed_requests(
        &self,
        _request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        Ok(Vec::new())
    }

    /// Executes the tool and returns a result or error.
    ///
    /// # Errors
    ///
    /// Return an appropriate [`ToolError`] variant on failure. Source-specific
    /// concerns such as MCP authentication are resolved internally by the
    /// source (via host-supplied responders) and are not surfaced as tool
    /// errors.
    async fn invoke(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError>;
}

/// A name-keyed collection of [`Tool`] implementations.
///
/// The registry owns `Arc`-wrapped tools and is passed to a
/// [`BasicToolExecutor`] (or consumed by [`ToolCapabilityProvider`]) so the
/// agent loop can look up tools by name at execution time.
///
/// # Example
///
/// ```rust
/// use agentkit_tools_core::ToolRegistry;
/// # use agentkit_tools_core::{Tool, ToolContext, ToolError, ToolName, ToolRequest, ToolResult, ToolSpec};
/// # use async_trait::async_trait;
/// # use serde_json::json;
/// # struct NoopTool(ToolSpec);
/// # #[async_trait]
/// # impl Tool for NoopTool {
/// #     fn spec(&self) -> &ToolSpec { &self.0 }
/// #     async fn invoke(&self, _r: ToolRequest, _c: &mut ToolContext<'_>) -> Result<ToolResult, ToolError> { todo!() }
/// # }
///
/// let registry = ToolRegistry::new()
///     .with(NoopTool(ToolSpec::new(
///         ToolName::new("noop"),
///         "Does nothing",
///         json!({"type": "object"}),
///     )));
///
/// assert!(registry.get(&ToolName::new("noop")).is_some());
/// assert_eq!(registry.specs().len(), 1);
/// ```
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<ToolName, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool by value and returns `&mut self` for imperative chaining.
    pub fn register<T>(&mut self, tool: T) -> &mut Self
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool.spec().name.clone(), Arc::new(tool));
        self
    }

    /// Registers a tool by value and returns `self` for builder-style chaining.
    pub fn with<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.register(tool);
        self
    }

    /// Registers a pre-wrapped `Arc<dyn Tool>`.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
        self.tools.insert(tool.spec().name.clone(), tool);
        self
    }

    /// Looks up a tool by name, returning `None` if not registered.
    pub fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Returns all registered tools as a `Vec`.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    /// Merges all tools from another registry into this one, consuming it.
    ///
    /// Supports builder-style chaining:
    ///
    /// ```ignore
    /// let registry = agentkit_tool_fs::registry()
    ///     .merge(agentkit_tool_shell::registry());
    /// ```
    pub fn merge(mut self, other: Self) -> Self {
        self.tools.extend(other.tools);
        self
    }

    /// Returns the [`ToolSpec`] for every registered tool.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .filter_map(|tool| tool.current_spec())
            .collect()
    }
}

/// Read-side contract for a federated tool catalog.
///
/// A [`BasicToolExecutor`] composes one or more `ToolSource`s — typically a
/// frozen [`ToolRegistry`] of native tools alongside one or more
/// [`CatalogReader`]s owned by subsystems (MCP server manager, skill watcher,
/// plugin loader). Each source manages its own lifecycle and concurrency
/// story; the executor only reads.
pub trait ToolSource: Send + Sync {
    /// Returns the current spec for every tool in this source.
    fn specs(&self) -> Vec<ToolSpec>;

    /// Looks up a tool by name, returning `None` if not present.
    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>>;

    /// Drains pending catalog change events. Static sources return an empty
    /// list; dynamic sources surface added/removed/changed batches that the
    /// loop forwards to the model on the next turn.
    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        Vec::new()
    }

    /// Wraps this source so every advertised tool name is prefixed with
    /// `<prefix>_`. Useful for mounting the same source under multiple
    /// namespaces, or for avoiding collisions between MCP catalogs.
    ///
    /// Lookups strip the prefix before delegating, and the wrapped tool's
    /// `spec()` reports the public (prefixed) name so the model and the
    /// tool see consistent names.
    ///
    /// To wrap an `Arc<dyn ToolSource>` instead, use [`Prefixed::new`].
    fn prefixed(self, prefix: impl Into<String>) -> Prefixed<Self>
    where
        Self: Sized,
    {
        Prefixed::new(self, prefix)
    }

    /// Wraps this source so only tools whose name passes `predicate` are
    /// advertised and resolvable. Tools rejected by the predicate are
    /// invisible to the model and return `None` on lookup.
    ///
    /// To wrap an `Arc<dyn ToolSource>` instead, use [`Filtered::new`].
    fn filtered<F>(self, predicate: F) -> Filtered<Self, F>
    where
        Self: Sized,
        F: Fn(&ToolName) -> bool + Send + Sync + 'static,
    {
        Filtered::new(self, predicate)
    }

    /// Wraps this source with a name remapping. Each `(original, new)` pair
    /// in `mapping` causes the tool to be advertised as `new` and resolved
    /// from `new` back to `original` on lookup. Tools not in the mapping
    /// pass through unchanged.
    ///
    /// To wrap an `Arc<dyn ToolSource>` instead, use [`Renamed::new`].
    fn renamed<I>(self, mapping: I) -> Renamed<Self>
    where
        Self: Sized,
        I: IntoIterator<Item = (ToolName, ToolName)>,
    {
        Renamed::new(self, mapping)
    }
}

impl ToolSource for ToolRegistry {
    fn specs(&self) -> Vec<ToolSpec> {
        ToolRegistry::specs(self)
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        ToolRegistry::get(self, name)
    }
}

impl<S> ToolSource for Arc<S>
where
    S: ToolSource + ?Sized,
{
    fn specs(&self) -> Vec<ToolSpec> {
        (**self).specs()
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        (**self).get(name)
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        (**self).drain_catalog_events()
    }
}

/// A [`ToolSource`] wrapper that prefixes every advertised tool name with
/// `<prefix>_`. Constructed via [`ToolSource::prefixed`] or directly.
pub struct Prefixed<S> {
    inner: S,
    prefix: String,
}

impl<S> Prefixed<S> {
    /// Creates a new prefixed wrapper.
    pub fn new(inner: S, prefix: impl Into<String>) -> Self {
        Self {
            inner,
            prefix: prefix.into(),
        }
    }

    fn rewrite(&self, name: &str) -> String {
        format!("{}_{}", self.prefix, name)
    }

    fn strip<'a>(&self, name: &'a str) -> Option<&'a str> {
        name.strip_prefix(self.prefix.as_str())
            .and_then(|rest| rest.strip_prefix('_'))
    }
}

impl<S> ToolSource for Prefixed<S>
where
    S: ToolSource,
{
    fn specs(&self) -> Vec<ToolSpec> {
        self.inner
            .specs()
            .into_iter()
            .map(|mut spec| {
                spec.name = ToolName::new(self.rewrite(spec.name.0.as_str()));
                spec
            })
            .collect()
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        let original = self.strip(name.0.as_str())?;
        let inner_name = ToolName::new(original);
        let inner_tool = self.inner.get(&inner_name)?;
        let mut public_spec = inner_tool.spec().clone();
        public_spec.name = name.clone();
        Some(Arc::new(RewrittenTool {
            inner: inner_tool,
            inner_name,
            public_spec,
        }))
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        self.inner
            .drain_catalog_events()
            .into_iter()
            .map(|mut event| {
                event.for_each_name_mut(|name| *name = self.rewrite(name.as_str()));
                event
            })
            .collect()
    }
}

/// A [`ToolSource`] wrapper that hides tools rejected by `predicate`.
/// Constructed via [`ToolSource::filtered`] or directly.
pub struct Filtered<S, F> {
    inner: S,
    predicate: F,
}

impl<S, F> Filtered<S, F> {
    /// Creates a new filtered wrapper.
    pub fn new(inner: S, predicate: F) -> Self {
        Self { inner, predicate }
    }
}

impl<S, F> ToolSource for Filtered<S, F>
where
    S: ToolSource,
    F: Fn(&ToolName) -> bool + Send + Sync + 'static,
{
    fn specs(&self) -> Vec<ToolSpec> {
        self.inner
            .specs()
            .into_iter()
            .filter(|spec| (self.predicate)(&spec.name))
            .collect()
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        if !(self.predicate)(name) {
            return None;
        }
        self.inner.get(name)
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        self.inner
            .drain_catalog_events()
            .into_iter()
            .map(|mut event| {
                event.retain_names(|n| (self.predicate)(&ToolName::new(n)));
                event
            })
            .collect()
    }
}

/// A [`ToolSource`] wrapper that renames specific tools. Tools whose
/// original name appears in the forward mapping are advertised under the
/// new name and resolved from the new name back to the original.
/// Unmapped names pass through unchanged.
///
/// Constructed via [`ToolSource::renamed`] or directly.
pub struct Renamed<S> {
    inner: S,
    forward: BTreeMap<ToolName, ToolName>,
    backward: BTreeMap<ToolName, ToolName>,
}

impl<S> Renamed<S> {
    /// Creates a new renaming wrapper from a `(original, new)` mapping.
    pub fn new<I>(inner: S, mapping: I) -> Self
    where
        I: IntoIterator<Item = (ToolName, ToolName)>,
    {
        let forward: BTreeMap<ToolName, ToolName> = mapping.into_iter().collect();
        let backward = forward
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();
        Self {
            inner,
            forward,
            backward,
        }
    }
}

impl<S> ToolSource for Renamed<S>
where
    S: ToolSource,
{
    fn specs(&self) -> Vec<ToolSpec> {
        self.inner
            .specs()
            .into_iter()
            .map(|mut spec| {
                if let Some(new_name) = self.forward.get(&spec.name) {
                    spec.name = new_name.clone();
                }
                spec
            })
            .collect()
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        if let Some(original) = self.backward.get(name) {
            let inner_tool = self.inner.get(original)?;
            let mut public_spec = inner_tool.spec().clone();
            public_spec.name = name.clone();
            Some(Arc::new(RewrittenTool {
                inner: inner_tool,
                inner_name: original.clone(),
                public_spec,
            }))
        } else if self.forward.contains_key(name) {
            // Original name of a remapped tool — hidden under its new name.
            None
        } else {
            self.inner.get(name)
        }
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        self.inner
            .drain_catalog_events()
            .into_iter()
            .map(|mut event| {
                event.for_each_name_mut(|name| {
                    if let Some(new) = self.forward.get(&ToolName::new(name.as_str())) {
                        *name = new.0.clone();
                    }
                });
                event
            })
            .collect()
    }
}

/// Builds a JSON Schema [`Value`] for the given input type. Requires the
/// `schemars` feature.
///
/// This is the bridge between Rust types and the
/// [`ToolSpec::input_schema`] field — instead of hand-writing JSON Schema,
/// derive [`schemars::JsonSchema`] on your input struct and call this.
///
/// # Example
///
/// ```rust,ignore
/// use agentkit_tools_core::schema_for;
/// use schemars::JsonSchema;
///
/// #[derive(JsonSchema)]
/// struct WeatherInput {
///     /// City name to look up.
///     location: String,
///     /// Use celsius (default false).
///     #[serde(default)]
///     celsius: bool,
/// }
///
/// let schema = schema_for::<WeatherInput>();
/// assert!(schema.is_object());
/// ```
#[cfg(feature = "schemars")]
pub fn schema_for<T: schemars::JsonSchema>() -> Value {
    let schema = schemars::schema_for!(T);
    serde_json::to_value(schema)
        .expect("schemars produces valid JSON; this conversion is infallible")
}

/// Builds a [`ToolSpec`] from `T`'s derived JSON Schema. Requires the
/// `schemars` feature. The generated schema is exactly what
/// [`schema_for::<T>`] produces; this helper just wraps it with a name and
/// description.
///
/// # Example
///
/// ```rust,ignore
/// use agentkit_tools_core::tool_spec_for;
/// use schemars::JsonSchema;
///
/// #[derive(JsonSchema)]
/// struct WeatherInput { location: String }
///
/// let spec = tool_spec_for::<WeatherInput>("get_weather", "Fetch current weather");
/// assert_eq!(spec.name.0, "get_weather");
/// ```
#[cfg(feature = "schemars")]
pub fn tool_spec_for<T: schemars::JsonSchema>(
    name: impl Into<ToolName>,
    description: impl Into<String>,
) -> ToolSpec {
    ToolSpec::new(name, description, schema_for::<T>())
}

/// A [`Tool`] wrapper used by [`Prefixed`] and [`Renamed`] to bridge between
/// the public (rewritten) tool name and the inner tool's own name. The
/// wrapper reports the public spec but rewrites `request.tool_name` back to
/// the inner name before delegating to the wrapped tool, so tools that
/// inspect their own name (e.g. for logging or routing) see the original.
struct RewrittenTool {
    inner: Arc<dyn Tool>,
    inner_name: ToolName,
    public_spec: ToolSpec,
}

#[async_trait]
impl Tool for RewrittenTool {
    fn spec(&self) -> &ToolSpec {
        &self.public_spec
    }

    fn current_spec(&self) -> Option<ToolSpec> {
        let inner_current = self.inner.current_spec()?;
        Some(ToolSpec {
            name: self.public_spec.name.clone(),
            description: inner_current.description,
            input_schema: inner_current.input_schema,
            annotations: inner_current.annotations,
            metadata: inner_current.metadata,
        })
    }

    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let mut inner_request = request.clone();
        inner_request.tool_name = self.inner_name.clone();
        self.inner.proposed_requests(&inner_request)
    }

    async fn invoke(
        &self,
        mut request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        request.tool_name = self.inner_name.clone();
        self.inner.invoke(request, ctx).await
    }
}

/// Catalog storage with poison-recovery encoded in the type. The wrapped
/// `RwLock` is private; only [`read`](Self::read) and [`write`](Self::write)
/// are exposed, both infallible. Recovery is safe because every callsite
/// honors the invariant below.
///
/// **Invariant for callers:** write critical sections must compute all
/// derived state — diffs, comparisons, anything that may run user code
/// (`Tool` impls in particular) — BEFORE mutating the map. If a panic fires
/// between two mutations in the same critical section, recovery would hand
/// the next caller a partially-updated map. The current callsites all hold
/// this: `upsert`/`remove` perform a single op with no user code;
/// `replace_all` completes its diff (which calls `Tool::current_spec`)
/// before the swap.
///
/// The `catalog_recovers_from_panicked_writer` test exercises the recovery
/// path; if you change a write critical section, re-check that it still
/// computes-then-mutates.
struct ToolMap {
    inner: std::sync::RwLock<BTreeMap<ToolName, Arc<dyn Tool>>>,
}

impl ToolMap {
    fn new() -> Self {
        Self {
            inner: std::sync::RwLock::new(BTreeMap::new()),
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<ToolName, Arc<dyn Tool>>> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<ToolName, Arc<dyn Tool>>> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// Shared inner state of a dynamic catalog. Held by both [`CatalogWriter`]
/// (mutates) and [`CatalogReader`] (reads), behind `Arc`s that hosts never see.
struct DynamicCatalogInner {
    source_id: String,
    tools: ToolMap,
    events_tx: tokio::sync::broadcast::Sender<ToolCatalogEvent>,
}

/// Constructs a fresh dynamic tool catalog as a writer/reader pair.
///
/// The writer mutates the catalog; the reader implements [`ToolSource`] and
/// is what gets handed to an `Agent`. Both sides share storage internally —
/// callers see only sized, owned values. Modeled on
/// `tokio::sync::watch::channel`.
///
/// `source_id` appears as the `source` field on every emitted
/// [`ToolCatalogEvent`].
///
/// ```
/// use agentkit_tools_core::dynamic_catalog;
///
/// let (writer, reader) = dynamic_catalog("plugins");
/// assert_eq!(writer.source_id(), "plugins");
/// assert_eq!(reader.source_id(), "plugins");
/// ```
pub fn dynamic_catalog(source_id: impl Into<String>) -> (CatalogWriter, CatalogReader) {
    let (events_tx, events_rx) = tokio::sync::broadcast::channel(128);
    let inner = Arc::new(DynamicCatalogInner {
        source_id: source_id.into(),
        tools: ToolMap::new(),
        events_tx,
    });
    (
        CatalogWriter {
            inner: Arc::clone(&inner),
        },
        CatalogReader {
            inner,
            events_rx: std::sync::Mutex::new(events_rx),
        },
    )
}

/// Mutating side of a dynamic tool catalog. Owned by subsystems that
/// discover or refresh tools at runtime (MCP server manager, skill watcher,
/// plugin loader). Each [`upsert`](Self::upsert), [`remove`](Self::remove),
/// or [`replace_all`](Self::replace_all) emits a [`ToolCatalogEvent`] that
/// every [`CatalogReader`] minted from the same [`dynamic_catalog`] call
/// (or its clones) observes via [`ToolSource::drain_catalog_events`].
pub struct CatalogWriter {
    inner: Arc<DynamicCatalogInner>,
}

impl CatalogWriter {
    /// Stable source identifier appearing on emitted catalog events.
    pub fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    /// Mints an additional [`CatalogReader`] over the same shared state.
    /// The new reader subscribes from now forward — it does not see events
    /// emitted before this call.
    pub fn reader(&self) -> CatalogReader {
        CatalogReader {
            inner: Arc::clone(&self.inner),
            events_rx: std::sync::Mutex::new(self.inner.events_tx.subscribe()),
        }
    }

    /// Inserts or replaces a tool. Emits a single-entry catalog event with
    /// the tool's name in `added` (new) or `changed` (replaced).
    pub fn upsert(&self, tool: Arc<dyn Tool>) {
        let name = tool.spec().name.clone();
        let mut guard = self.inner.tools.write();
        let existed = guard.insert(name.clone(), tool).is_some();
        drop(guard);
        let mut event = ToolCatalogEvent::new(self.inner.source_id.clone());
        if existed {
            event.changed.push(name.0);
        } else {
            event.added.push(name.0);
        }
        let _ = self.inner.events_tx.send(event);
    }

    /// Removes a tool by name. Emits a catalog event with the name in
    /// `removed` if it existed.
    pub fn remove(&self, name: &ToolName) -> bool {
        let mut guard = self.inner.tools.write();
        let removed = guard.remove(name).is_some();
        drop(guard);
        if removed {
            let mut event = ToolCatalogEvent::new(self.inner.source_id.clone());
            event.removed.push(name.0.clone());
            let _ = self.inner.events_tx.send(event);
        }
        removed
    }

    /// Atomically replaces the entire tool set. Emits one catalog event
    /// describing the diff against the previous contents.
    pub fn replace_all(&self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) {
        let new_map: BTreeMap<ToolName, Arc<dyn Tool>> = tools
            .into_iter()
            .map(|tool| (tool.spec().name.clone(), tool))
            .collect();

        let mut guard = self.inner.tools.write();
        let mut event = ToolCatalogEvent::new(self.inner.source_id.clone());

        for (name, new_tool) in new_map.iter() {
            match guard.get(name) {
                None => event.added.push(name.0.clone()),
                Some(existing)
                    if !Arc::ptr_eq(existing, new_tool)
                        && existing.current_spec() != new_tool.current_spec() =>
                {
                    event.changed.push(name.0.clone());
                }
                Some(_) => {}
            }
        }
        for name in guard.keys() {
            if !new_map.contains_key(name) {
                event.removed.push(name.0.clone());
            }
        }

        *guard = new_map;
        drop(guard);

        if !event.added.is_empty() || !event.removed.is_empty() || !event.changed.is_empty() {
            let _ = self.inner.events_tx.send(event);
        }
    }

    /// Subscribes a fresh broadcast receiver. Lower-level than
    /// [`CatalogReader`] — for hosts that consume catalog events directly
    /// rather than through the loop.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ToolCatalogEvent> {
        self.inner.events_tx.subscribe()
    }
}

/// Read side of a dynamic tool catalog. Implements [`ToolSource`] and is the
/// value handed to [`agentkit_loop::AgentBuilder::tools`]. Cloning subscribes
/// a fresh broadcast receiver, so independent observers don't compete for
/// catalog events.
pub struct CatalogReader {
    inner: Arc<DynamicCatalogInner>,
    events_rx: std::sync::Mutex<tokio::sync::broadcast::Receiver<ToolCatalogEvent>>,
}

impl CatalogReader {
    /// Stable source identifier appearing on emitted catalog events.
    pub fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    /// Subscribes a fresh broadcast receiver — equivalent to
    /// [`CatalogWriter::subscribe`].
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ToolCatalogEvent> {
        self.inner.events_tx.subscribe()
    }
}

impl Clone for CatalogReader {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            events_rx: std::sync::Mutex::new(self.inner.events_tx.subscribe()),
        }
    }
}

impl ToolSource for CatalogReader {
    fn specs(&self) -> Vec<ToolSpec> {
        self.inner
            .tools
            .read()
            .values()
            .filter_map(|tool| tool.current_spec())
            .collect()
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        self.inner.tools.read().get(name).cloned()
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        // try_recv on a broadcast::Receiver has no panic source, so the only
        // way this Mutex poisons is if a panic somehow originates outside the
        // try_recv loop while held — recover defensively, the receiver state
        // is independent of this lock.
        let mut rx = self.events_rx.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(event) => out.push(event),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
                Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
            }
        }
        out
    }
}

impl ToolSpec {
    /// Converts this spec into an [`InvocableSpec`] for use with the
    /// capability layer.
    pub fn as_invocable_spec(&self) -> InvocableSpec {
        InvocableSpec::new(
            CapabilityName::new(self.name.0.clone()),
            self.description.clone(),
            self.input_schema.clone(),
        )
        .with_metadata(self.metadata.clone())
    }
}

/// Wraps a [`Tool`] as an [`Invocable`] so it can be surfaced through the
/// agentkit capability layer.
///
/// Created automatically by [`ToolCapabilityProvider::from_registry`]; you
/// rarely need to construct one yourself.
pub struct ToolInvocableAdapter {
    spec: InvocableSpec,
    tool: Arc<dyn Tool>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    next_call_id: AtomicU64,
}

impl ToolInvocableAdapter {
    /// Creates a new adapter that wraps `tool` with the given permission
    /// checker and shared resources.
    pub fn new(
        tool: Arc<dyn Tool>,
        permissions: Arc<dyn PermissionChecker>,
        resources: Arc<dyn ToolResources>,
    ) -> Option<Self> {
        let spec = tool.current_spec()?.as_invocable_spec();
        Some(Self {
            spec,
            tool,
            permissions,
            resources,
            next_call_id: AtomicU64::new(1),
        })
    }
}

#[async_trait]
impl Invocable for ToolInvocableAdapter {
    fn spec(&self) -> &InvocableSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: InvocableRequest,
        ctx: &mut CapabilityContext<'_>,
    ) -> Result<InvocableResult, CapabilityError> {
        let tool_request = ToolRequest {
            call_id: ToolCallId::new(format!(
                "tool-call-{}",
                self.next_call_id.fetch_add(1, Ordering::Relaxed)
            )),
            tool_name: self.tool.spec().name.clone(),
            input: request.input,
            session_id: ctx
                .session_id
                .cloned()
                .unwrap_or_else(|| SessionId::new("capability-session")),
            turn_id: ctx
                .turn_id
                .cloned()
                .unwrap_or_else(|| TurnId::new("capability-turn")),
            metadata: request.metadata,
        };

        for permission_request in self
            .tool
            .proposed_requests(&tool_request)
            .map_err(|error| CapabilityError::InvalidInput(error.to_string()))?
        {
            match self.permissions.evaluate(permission_request.as_ref()) {
                PermissionDecision::Allow => {}
                PermissionDecision::Deny(denial) => {
                    return Err(CapabilityError::ExecutionFailed(format!(
                        "tool permission denied: {denial:?}"
                    )));
                }
                PermissionDecision::RequireApproval(req) => {
                    return Err(CapabilityError::Unavailable(format!(
                        "tool invocation requires approval: {}",
                        req.summary
                    )));
                }
            }
        }

        let mut tool_ctx = ToolContext {
            capability: CapabilityContext {
                session_id: ctx.session_id,
                turn_id: ctx.turn_id,
                metadata: ctx.metadata,
            },
            permissions: self.permissions.as_ref(),
            resources: self.resources.as_ref(),
            cancellation: None,
        };

        let result = self
            .tool
            .invoke(tool_request, &mut tool_ctx)
            .await
            .map_err(|error| CapabilityError::ExecutionFailed(error.to_string()))?;

        Ok(InvocableResult {
            output: match result.result.output {
                ToolOutput::Text(text) => InvocableOutput::Text(text),
                ToolOutput::Structured(value) => InvocableOutput::Structured(value),
                ToolOutput::Parts(parts) => InvocableOutput::Items(vec![Item {
                    id: None,
                    kind: ItemKind::Tool,
                    parts,
                    metadata: MetadataMap::new(),
                }]),
                ToolOutput::Files(files) => {
                    let parts = files.into_iter().map(Part::File).collect();
                    InvocableOutput::Items(vec![Item {
                        id: None,
                        kind: ItemKind::Tool,
                        parts,
                        metadata: MetadataMap::new(),
                    }])
                }
            },
            metadata: result.metadata,
        })
    }
}

/// A [`CapabilityProvider`] that exposes every tool in a [`ToolRegistry`]
/// as an [`Invocable`] in the agentkit capability layer.
///
/// This is the bridge between the tool subsystem and the generic capability
/// API that the agent loop consumes.
pub struct ToolCapabilityProvider {
    invocables: Vec<Arc<dyn Invocable>>,
}

impl ToolCapabilityProvider {
    /// Builds a provider from all tools in `registry`, sharing the given
    /// permission checker and resources across every adapter.
    pub fn from_registry(
        registry: &ToolRegistry,
        permissions: Arc<dyn PermissionChecker>,
        resources: Arc<dyn ToolResources>,
    ) -> Self {
        let invocables = registry
            .tools()
            .into_iter()
            .filter_map(|tool| {
                ToolInvocableAdapter::new(tool, permissions.clone(), resources.clone())
                    .map(|adapter| Arc::new(adapter) as Arc<dyn Invocable>)
            })
            .collect();

        Self { invocables }
    }
}

impl CapabilityProvider for ToolCapabilityProvider {
    fn invocables(&self) -> Vec<Arc<dyn Invocable>> {
        self.invocables.clone()
    }

    fn resources(&self) -> Vec<Arc<dyn ResourceProvider>> {
        Vec::new()
    }

    fn prompts(&self) -> Vec<Arc<dyn PromptProvider>> {
        Vec::new()
    }
}

/// The three-way result of a [`ToolExecutor::execute`] call.
///
/// Unlike a simple `Result`, this type distinguishes between a successful
/// completion, an interruption requiring user input (approval or auth), and
/// an outright failure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolExecutionOutcome {
    /// The tool ran to completion and produced a result.
    Completed(ToolResult),
    /// The tool was interrupted and needs user input before it can continue.
    Interrupted(ToolInterruption),
    /// The tool failed with an error.
    Failed(ToolError),
}

/// Trait for executing tool calls with permission checking and interruption
/// handling.
///
/// The agent loop calls [`execute`](ToolExecutor::execute) for every tool
/// call the model emits. If execution returns
/// [`ToolExecutionOutcome::Interrupted`], the loop collects user input and
/// retries with [`execute_approved`](ToolExecutor::execute_approved).
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Returns the current specification for every available tool.
    fn specs(&self) -> Vec<ToolSpec>;

    /// Drains any pending dynamic catalog events.
    ///
    /// Static executors return an empty list. Dynamic executors should use
    /// interior mutability to return each catalog event once.
    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        Vec::new()
    }

    /// Looks up the tool, evaluates permissions, and invokes it.
    async fn execute(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome;

    /// Looks up the tool, evaluates permissions, and invokes it using an
    /// owned execution context.
    async fn execute_owned(
        &self,
        request: ToolRequest,
        ctx: OwnedToolContext,
    ) -> ToolExecutionOutcome {
        let mut borrowed = ctx.borrowed();
        self.execute(request, &mut borrowed).await
    }

    /// Re-executes a tool call that was previously interrupted for approval.
    ///
    /// The default implementation ignores `approved_request` and delegates
    /// to [`execute`](ToolExecutor::execute). [`BasicToolExecutor`]
    /// overrides this to skip the approval gate for the matching request.
    async fn execute_approved(
        &self,
        request: ToolRequest,
        approved_request: &ApprovalRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        let _ = approved_request;
        self.execute(request, ctx).await
    }

    /// Re-executes a tool call that was previously interrupted for approval
    /// using an owned execution context.
    async fn execute_approved_owned(
        &self,
        request: ToolRequest,
        approved_request: &ApprovalRequest,
        ctx: OwnedToolContext,
    ) -> ToolExecutionOutcome {
        let mut borrowed = ctx.borrowed();
        self.execute_approved(request, approved_request, &mut borrowed)
            .await
    }
}

#[async_trait]
impl<T> ToolExecutor for Arc<T>
where
    T: ToolExecutor + ?Sized,
{
    fn specs(&self) -> Vec<ToolSpec> {
        (**self).specs()
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        (**self).drain_catalog_events()
    }

    async fn execute(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        (**self).execute(request, ctx).await
    }

    async fn execute_approved(
        &self,
        request: ToolRequest,
        approved_request: &ApprovalRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        (**self)
            .execute_approved(request, approved_request, ctx)
            .await
    }
}

/// Policy applied when the same tool name appears in more than one
/// [`ToolSource`] of a [`BasicToolExecutor`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CollisionPolicy {
    /// First source wins (in iteration order). Subsequent definitions of
    /// the same name are ignored.
    #[default]
    FirstWins,
    /// Later sources overwrite earlier ones at lookup time.
    LastWins,
}

/// The default [`ToolExecutor`] that walks one or more [`ToolSource`]s,
/// checks permissions via [`Tool::proposed_requests`], and invokes the tool.
///
/// Compose static native tools (a frozen [`ToolRegistry`]) alongside
/// dynamic sources (a [`CatalogReader`] minted by [`dynamic_catalog`] and
/// owned by an MCP manager, skill watcher, plugin loader, etc.) without
/// merging into a single mutable registry.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use agentkit_tools_core::{BasicToolExecutor, ToolRegistry, ToolSource};
///
/// let static_registry: Arc<dyn ToolSource> = Arc::new(ToolRegistry::new());
/// let executor = BasicToolExecutor::new([static_registry]);
/// // Pass `executor` to the agent loop.
/// ```
pub struct BasicToolExecutor {
    sources: Vec<Arc<dyn ToolSource>>,
    collision: CollisionPolicy,
}

impl BasicToolExecutor {
    /// Creates an executor that walks `sources` in order on every lookup.
    pub fn new(sources: impl IntoIterator<Item = Arc<dyn ToolSource>>) -> Self {
        Self {
            sources: sources.into_iter().collect(),
            collision: CollisionPolicy::default(),
        }
    }

    /// Back-compat shorthand: wrap a single [`ToolRegistry`] as the only source.
    pub fn from_registry(registry: ToolRegistry) -> Self {
        Self::new([Arc::new(registry) as Arc<dyn ToolSource>])
    }

    /// Sets the collision policy applied when the same tool name appears in
    /// multiple sources.
    pub fn with_collision_policy(mut self, policy: CollisionPolicy) -> Self {
        self.collision = policy;
        self
    }

    /// Returns the [`ToolSpec`] for every tool across all sources, deduped
    /// by [`CollisionPolicy`].
    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        let iter: Box<dyn Iterator<Item = &Arc<dyn ToolSource>>> = match self.collision {
            CollisionPolicy::FirstWins => Box::new(self.sources.iter()),
            CollisionPolicy::LastWins => Box::new(self.sources.iter().rev()),
        };
        for source in iter {
            for spec in source.specs() {
                if seen.insert(spec.name.clone()) {
                    out.push(spec);
                }
            }
        }
        out
    }

    fn lookup(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        match self.collision {
            CollisionPolicy::FirstWins => self.sources.iter().find_map(|s| s.get(name)),
            CollisionPolicy::LastWins => self.sources.iter().rev().find_map(|s| s.get(name)),
        }
    }

    async fn execute_inner(
        &self,
        request: ToolRequest,
        approved_request_id: Option<&ApprovalId>,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        let Some(tool) = self.lookup(&request.tool_name) else {
            return ToolExecutionOutcome::Failed(ToolError::NotFound(request.tool_name));
        };

        match tool.proposed_requests(&request) {
            Ok(requests) => {
                for permission_request in requests {
                    match ctx.permissions.evaluate(permission_request.as_ref()) {
                        PermissionDecision::Allow => {}
                        PermissionDecision::Deny(denial) => {
                            return ToolExecutionOutcome::Failed(ToolError::PermissionDenied(
                                denial,
                            ));
                        }
                        PermissionDecision::RequireApproval(mut req) => {
                            req.call_id = Some(request.call_id.clone());
                            if approved_request_id != Some(&req.id) {
                                return ToolExecutionOutcome::Interrupted(
                                    ToolInterruption::ApprovalRequired(req),
                                );
                            }
                        }
                    }
                }
            }
            Err(error) => return ToolExecutionOutcome::Failed(error),
        }

        match tool.invoke(request, ctx).await {
            Ok(result) => ToolExecutionOutcome::Completed(result),
            Err(error) => ToolExecutionOutcome::Failed(error),
        }
    }
}

#[async_trait]
impl ToolExecutor for BasicToolExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        BasicToolExecutor::specs(self)
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        self.sources
            .iter()
            .flat_map(|s| s.drain_catalog_events())
            .collect()
    }

    async fn execute(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        self.execute_inner(request, None, ctx).await
    }

    async fn execute_approved(
        &self,
        request: ToolRequest,
        approved_request: &ApprovalRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        self.execute_inner(request, Some(&approved_request.id), ctx)
            .await
    }
}

/// Errors that can occur during tool lookup, permission checking, or execution.
///
/// Returned from [`Tool::invoke`] and also used internally by
/// [`BasicToolExecutor`] to represent lookup and permission failures.
#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolError {
    /// No tool with the given name exists in the registry.
    #[error("tool not found: {0}")]
    NotFound(ToolName),
    /// The input JSON did not match the tool's expected schema.
    #[error("invalid tool input: {0}")]
    InvalidInput(String),
    /// A permission policy denied the operation.
    #[error("tool permission denied: {0:?}")]
    PermissionDenied(PermissionDenial),
    /// The tool ran but encountered a runtime error.
    #[error("tool execution failed: {0}")]
    ExecutionFailed(String),
    /// The tool is temporarily unavailable.
    #[error("tool unavailable: {0}")]
    Unavailable(String),
    /// The turn was cancelled while the tool was running.
    #[error("tool execution cancelled")]
    Cancelled,
    /// An unexpected internal error.
    #[error("internal tool error: {0}")]
    Internal(String),
}

impl ToolError {
    /// Convenience constructor for the [`PermissionDenied`](ToolError::PermissionDenied) variant.
    pub fn permission_denied(denial: PermissionDenial) -> Self {
        Self::PermissionDenied(denial)
    }
}

impl From<PermissionDenial> for ToolError {
    fn from(value: PermissionDenial) -> Self {
        Self::permission_denied(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;

    #[test]
    fn command_policy_can_deny_unknown_executables_without_approval() {
        let policy = CommandPolicy::new()
            .allow_executable("pwd")
            .require_approval_for_unknown(false);
        let request = ShellPermissionRequest {
            executable: "rm".into(),
            argv: vec!["-rf".into(), "/tmp/demo".into()],
            cwd: None,
            env_keys: Vec::new(),
            metadata: MetadataMap::new(),
        };

        match policy.evaluate(&request) {
            PolicyMatch::Deny(denial) => {
                assert_eq!(denial.code, PermissionCode::CommandNotAllowed);
            }
            other => panic!("unexpected policy match: {other:?}"),
        }
    }

    #[test]
    fn path_policy_allows_reads_under_read_only_roots() {
        let policy = PathPolicy::new().read_only_root("/workspace/vendor");
        let request = FileSystemPermissionRequest::Read {
            path: PathBuf::from("/workspace/vendor/lib.rs"),
            metadata: MetadataMap::new(),
        };

        match policy.evaluate(&request) {
            PolicyMatch::NoOpinion | PolicyMatch::Allow => {}
            other => panic!("unexpected policy match: {other:?}"),
        }
    }

    #[test]
    fn path_policy_denies_mutations_under_read_only_roots() {
        let policy = PathPolicy::new().read_only_root("/workspace/vendor");
        let request = FileSystemPermissionRequest::Edit {
            path: PathBuf::from("/workspace/vendor/lib.rs"),
            metadata: MetadataMap::new(),
        };

        match policy.evaluate(&request) {
            PolicyMatch::Deny(denial) => {
                assert_eq!(denial.code, PermissionCode::PathNotAllowed);
                assert!(denial.message.contains("read-only"));
            }
            other => panic!("unexpected policy match: {other:?}"),
        }
    }

    #[test]
    fn path_policy_denies_moves_into_read_only_roots() {
        let policy = PathPolicy::new().read_only_root("/workspace/vendor");
        let request = FileSystemPermissionRequest::Move {
            from: PathBuf::from("/workspace/src/lib.rs"),
            to: PathBuf::from("/workspace/vendor/lib.rs"),
            metadata: MetadataMap::new(),
        };

        match policy.evaluate(&request) {
            PolicyMatch::Deny(denial) => {
                assert_eq!(denial.code, PermissionCode::PathNotAllowed);
                assert!(denial.message.contains("read-only"));
            }
            other => panic!("unexpected policy match: {other:?}"),
        }
    }

    #[cfg(unix)]
    struct SymlinkTmpDir(PathBuf);

    #[cfg(unix)]
    impl SymlinkTmpDir {
        fn new(label: &str) -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "agentkit-pathpolicy-{}-{}-{}",
                label,
                std::process::id(),
                nanos
            ));
            std::fs::create_dir_all(&dir).unwrap();
            // Canonicalise so callers compare against the resolved tmp path
            // (macOS `/tmp` is a symlink to `/private/tmp`, etc.).
            Self(std::fs::canonicalize(&dir).unwrap())
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    #[cfg(unix)]
    impl Drop for SymlinkTmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    fn assert_path_denied(
        policy: &PathPolicy,
        request: FileSystemPermissionRequest,
    ) -> PermissionDenial {
        match policy.evaluate(&request) {
            PolicyMatch::Deny(denial) => denial,
            other => panic!("expected deny, got: {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn path_policy_blocks_symlink_escape_from_allowed_root() {
        let tmp = SymlinkTmpDir::new("allow-escape");
        let allowed = tmp.path().join("workspace");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let secret = outside.join("secret.txt");
        std::fs::write(&secret, b"top-secret").unwrap();
        let escape = allowed.join("leak");
        std::os::unix::fs::symlink(&secret, &escape).unwrap();

        let policy = PathPolicy::new()
            .allow_root(&allowed)
            .require_approval_outside_allowed(false);
        let denial = assert_path_denied(
            &policy,
            FileSystemPermissionRequest::Read {
                path: escape,
                metadata: MetadataMap::new(),
            },
        );
        assert_eq!(denial.code, PermissionCode::PathNotAllowed);
    }

    #[cfg(unix)]
    #[test]
    fn path_policy_blocks_symlink_into_protected_root() {
        let tmp = SymlinkTmpDir::new("protect-bypass");
        let workspace = tmp.path().join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let secret = workspace.join(".env");
        std::fs::write(&secret, b"API_KEY=xxx").unwrap();
        let alias = workspace.join("config");
        std::os::unix::fs::symlink(&secret, &alias).unwrap();

        let policy = PathPolicy::new()
            .allow_root(&workspace)
            .protect_root(&secret);
        let denial = assert_path_denied(
            &policy,
            FileSystemPermissionRequest::Read {
                path: alias,
                metadata: MetadataMap::new(),
            },
        );
        assert_eq!(denial.code, PermissionCode::PathNotAllowed);
        assert!(denial.message.contains("denied"));
    }

    #[cfg(unix)]
    #[test]
    fn path_policy_blocks_symlink_write_into_read_only_root() {
        let tmp = SymlinkTmpDir::new("readonly-bypass");
        let workspace = tmp.path().join("workspace");
        let vendor = workspace.join("vendor");
        std::fs::create_dir_all(&vendor).unwrap();
        let target = vendor.join("lib.rs");
        std::fs::write(&target, b"// vendored").unwrap();
        let writable_alias = workspace.join("writable");
        std::os::unix::fs::symlink(&target, &writable_alias).unwrap();

        let policy = PathPolicy::new()
            .allow_root(&workspace)
            .read_only_root(&vendor);
        let denial = assert_path_denied(
            &policy,
            FileSystemPermissionRequest::Edit {
                path: writable_alias,
                metadata: MetadataMap::new(),
            },
        );
        assert_eq!(denial.code, PermissionCode::PathNotAllowed);
        assert!(denial.message.contains("read-only"));
    }

    #[cfg(unix)]
    #[test]
    fn path_policy_resolves_symlink_parent_for_nonexistent_leaf() {
        let tmp = SymlinkTmpDir::new("create-escape");
        let allowed = tmp.path().join("workspace");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let escape_dir = allowed.join("escape");
        std::os::unix::fs::symlink(&outside, &escape_dir).unwrap();
        let new_file = escape_dir.join("new.txt");

        let policy = PathPolicy::new()
            .allow_root(&allowed)
            .require_approval_outside_allowed(false);
        let denial = assert_path_denied(
            &policy,
            FileSystemPermissionRequest::Write {
                path: new_file,
                metadata: MetadataMap::new(),
            },
        );
        assert_eq!(denial.code, PermissionCode::PathNotAllowed);
    }

    #[derive(Clone)]
    struct HiddenTool {
        spec: ToolSpec,
    }

    impl HiddenTool {
        fn new() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("hidden"),
                    description: "hidden".into(),
                    input_schema: json!({"type": "object"}),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[async_trait]
    impl Tool for HiddenTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn current_spec(&self) -> Option<ToolSpec> {
            None
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text("hidden".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    #[test]
    fn hidden_tools_are_omitted_from_specs_and_capabilities() {
        let registry = ToolRegistry::new().with(HiddenTool::new());

        assert!(registry.specs().is_empty());

        let provider = ToolCapabilityProvider::from_registry(
            &registry,
            Arc::new(AllowAllPermissionChecker),
            Arc::new(()),
        );
        assert!(provider.invocables().is_empty());
    }

    struct AllowAllPermissionChecker;

    impl PermissionChecker for AllowAllPermissionChecker {
        fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    /// Tool whose `current_spec()` panics — used to exercise the
    /// catalog's poison-recovery guarantee.
    #[derive(Clone)]
    struct PanickingSpecTool {
        spec: ToolSpec,
    }

    impl PanickingSpecTool {
        fn new(name: &str) -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new(name),
                    description: "panics on current_spec".into(),
                    input_schema: json!({"type": "object"}),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[async_trait]
    impl Tool for PanickingSpecTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn current_spec(&self) -> Option<ToolSpec> {
            panic!("PanickingSpecTool::current_spec");
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text("never".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    /// If a tool's `current_spec()` panics during `replace_all`'s diff phase,
    /// the inner `RwLock` would normally poison and brick the catalog forever.
    /// `ToolMap` recovers from poison; this test pins the behavior so a future
    /// patch can't accidentally reintroduce the brick.
    ///
    /// The recovery is only safe because `replace_all` computes the diff
    /// (running user code) BEFORE swapping the map. If you change a write
    /// critical section to mutate before/between user-code calls, this test
    /// will still pass — but the catalog WILL be in a half-mutated state
    /// after a panic. Re-read `ToolMap`'s invariant before changing.
    #[test]
    fn catalog_recovers_from_panicked_writer() {
        let (writer, reader) = dynamic_catalog("test");

        // Pre-seed with a panicker so the next `replace_all` enters the
        // diff branch that calls `existing.current_spec()`. `upsert`
        // itself never calls `current_spec`, so this insertion is safe.
        writer.upsert(Arc::new(PanickingSpecTool::new("boom")));
        let _ = reader.drain_catalog_events();

        // `replace_all` with a different Arc of the same name forces the
        // diff to call `existing.current_spec()` → panics. The swap has
        // NOT happened yet at this point (the diff runs before the
        // `*guard = new_map`), so the catalog state is still consistent.
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            writer.replace_all(vec![
                Arc::new(PanickingSpecTool::new("boom")) as Arc<dyn Tool>
            ]);
        }));
        assert!(
            panic_result.is_err(),
            "PanickingSpecTool::current_spec must propagate"
        );

        // Without recovery, every subsequent lock acquisition would panic
        // with "dynamic catalog poisoned". `get` doesn't call `current_spec`,
        // so it's a clean probe of whether the lock recovered.
        assert!(
            reader.get(&ToolName::new("boom")).is_some(),
            "catalog still readable after poisoning panic"
        );

        // Writes also recover. Remove the panicker so subsequent `specs()`
        // calls don't re-trigger its panic.
        assert!(writer.remove(&ToolName::new("boom")));

        // Add a well-behaved tool and round-trip through both sides.
        // (HiddenTool::current_spec returns None, so it's intentionally
        // filtered out of specs() — probe via get() instead.)
        writer.upsert(Arc::new(HiddenTool::new()));
        assert!(
            reader.get(&ToolName::new("hidden")).is_some(),
            "catalog usable for further writes + reads"
        );
    }

    #[derive(Clone)]
    struct EchoTool {
        spec: ToolSpec,
    }

    impl EchoTool {
        fn new(name: &str) -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new(name),
                    description: format!("echo {name}"),
                    input_schema: json!({"type": "object"}),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::text(request.tool_name.0.clone()),
            )))
        }
    }

    fn registry_with(names: &[&str]) -> ToolRegistry {
        names.iter().fold(ToolRegistry::new(), |reg, name| {
            reg.with(EchoTool::new(name))
        })
    }

    #[test]
    fn prefixed_rewrites_specs_and_resolves_lookups() {
        let source = registry_with(&["get_temp", "get_humidity"]).prefixed("weather");
        let names: Vec<_> = source.specs().into_iter().map(|s| s.name.0).collect();
        assert_eq!(names, vec!["weather_get_humidity", "weather_get_temp"]);

        assert!(source.get(&ToolName::new("weather_get_temp")).is_some());
        assert!(
            source.get(&ToolName::new("get_temp")).is_none(),
            "original name must not resolve when prefixed"
        );
        assert!(source.get(&ToolName::new("unknown")).is_none());
    }

    #[tokio::test]
    async fn prefixed_invoke_sees_inner_name_on_request() {
        let source = registry_with(&["get_temp"]).prefixed("weather");
        let tool = source.get(&ToolName::new("weather_get_temp")).unwrap();

        // The wrapper must report the public name on its spec...
        assert_eq!(tool.spec().name.0, "weather_get_temp");

        // ...but the inner tool must see its own name in the request.
        let owned = OwnedToolContext {
            session_id: SessionId::new("s"),
            turn_id: TurnId::new("t"),
            metadata: MetadataMap::new(),
            permissions: Arc::new(AllowAllPermissions),
            resources: Arc::new(()),
            cancellation: None,
        };
        let mut ctx = owned.borrowed();
        let request = ToolRequest {
            call_id: ToolCallId::new("c"),
            tool_name: ToolName::new("weather_get_temp"),
            input: json!({}),
            session_id: SessionId::new("s"),
            turn_id: TurnId::new("t"),
            metadata: MetadataMap::new(),
        };
        let result = tool.invoke(request, &mut ctx).await.unwrap();
        match result.result.output {
            ToolOutput::Text(text) => assert_eq!(text, "get_temp"),
            other => panic!("unexpected output: {other:?}"),
        }
    }

    #[test]
    fn filtered_hides_tools_rejected_by_predicate() {
        let source = registry_with(&["safe", "danger_drop", "danger_delete"])
            .filtered(|name| !name.0.starts_with("danger_"));
        let names: Vec<_> = source.specs().into_iter().map(|s| s.name.0).collect();
        assert_eq!(names, vec!["safe"]);

        assert!(source.get(&ToolName::new("safe")).is_some());
        assert!(source.get(&ToolName::new("danger_drop")).is_none());
    }

    #[test]
    fn renamed_remaps_specs_and_lookups() {
        let source = registry_with(&["legacy_name", "passthrough"])
            .renamed([(ToolName::new("legacy_name"), ToolName::new("modern_name"))]);
        let mut names: Vec<_> = source.specs().into_iter().map(|s| s.name.0).collect();
        names.sort();
        assert_eq!(names, vec!["modern_name", "passthrough"]);

        assert!(source.get(&ToolName::new("modern_name")).is_some());
        assert!(
            source.get(&ToolName::new("legacy_name")).is_none(),
            "original name is hidden after renaming"
        );
        assert!(source.get(&ToolName::new("passthrough")).is_some());
    }

    #[cfg(feature = "schemars")]
    mod schemars_helpers {
        use super::*;
        use schemars::JsonSchema;
        use serde::Deserialize;

        #[derive(JsonSchema, Deserialize)]
        #[allow(dead_code)]
        struct WeatherInput {
            /// City name to look up.
            location: String,
            /// Use celsius (default false).
            #[serde(default)]
            celsius: bool,
        }

        #[test]
        fn schema_for_emits_object_schema_with_typed_fields() {
            let schema = schema_for::<WeatherInput>();
            let obj = schema.as_object().expect("schema is a JSON object");
            assert_eq!(
                obj.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "root type should be object"
            );
            let properties = obj
                .get("properties")
                .and_then(|v| v.as_object())
                .expect("properties block");
            assert!(properties.contains_key("location"));
            assert!(properties.contains_key("celsius"));
        }

        #[test]
        fn tool_spec_for_carries_schema_name_and_description() {
            let spec = tool_spec_for::<WeatherInput>("get_weather", "Fetch current weather");
            assert_eq!(spec.name.0, "get_weather");
            assert_eq!(spec.description, "Fetch current weather");
            assert!(spec.input_schema.is_object());
        }
    }

    #[test]
    fn transforms_compose_via_chained_methods() {
        let source = registry_with(&["read_file", "write_file", "delete_file"])
            .filtered(|name| name.0 != "delete_file")
            .prefixed("fs");
        let mut names: Vec<_> = source.specs().into_iter().map(|s| s.name.0).collect();
        names.sort();
        assert_eq!(names, vec!["fs_read_file", "fs_write_file"]);
    }
}
