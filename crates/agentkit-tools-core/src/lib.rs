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
//! - **Handle interruptions** (approval prompts, OAuth flows) via the
//!   [`ToolInterruption`] / [`ApprovalRequest`] / [`AuthRequest`] types.
//! - **Bridge to the capability layer** with [`ToolCapabilityProvider`],
//!   which wraps every registered tool as an [`Invocable`].

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Declarative specification of a tool's identity, schema, and behavioural hints.
///
/// Every [`Tool`] implementation exposes a `ToolSpec` that the framework uses to
/// advertise the tool to an LLM, validate inputs, and drive permission checks.
///
/// # Example
///
/// ```rust
/// use agentkit_core::MetadataMap;
/// use agentkit_tools_core::{ToolAnnotations, ToolName, ToolSpec};
/// use serde_json::json;
///
/// let spec = ToolSpec {
///     name: ToolName::new("grep_search"),
///     description: "Search files by regex pattern".into(),
///     input_schema: json!({
///         "type": "object",
///         "properties": {
///             "pattern": { "type": "string" },
///             "path": { "type": "string" }
///         },
///         "required": ["pattern"]
///     }),
///     annotations: ToolAnnotations {
///         read_only_hint: true,
///         ..Default::default()
///     },
///     metadata: MetadataMap::new(),
/// };
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

/// A request for authentication credentials before a tool can proceed.
///
/// Emitted as [`ToolInterruption::AuthRequired`] when a tool (typically an
/// MCP integration) needs OAuth tokens, API keys, or other credentials that
/// the user must supply interactively.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthRequest {
    /// Runtime task identifier associated with this auth request, if any.
    pub task_id: Option<TaskId>,
    /// Unique identifier for this auth challenge.
    pub id: String,
    /// Name of the authentication provider (e.g. `"github"`, `"google"`).
    pub provider: String,
    /// The operation that triggered the auth requirement.
    pub operation: AuthOperation,
    /// Provider-specific challenge data (e.g. OAuth URLs, scopes).
    pub challenge: MetadataMap,
}

/// Describes the operation that triggered an [`AuthRequest`].
///
/// The agent loop can inspect this to decide how to present the auth
/// challenge and where to deliver the resulting credentials.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthOperation {
    /// A local tool call that requires auth.
    ToolCall {
        tool_name: String,
        input: Value,
        call_id: Option<ToolCallId>,
        session_id: Option<SessionId>,
        turn_id: Option<TurnId>,
        metadata: MetadataMap,
    },
    /// Connecting to an MCP server that requires auth.
    McpConnect {
        server_id: String,
        metadata: MetadataMap,
    },
    /// Invoking a tool on an MCP server that requires auth.
    McpToolCall {
        server_id: String,
        tool_name: String,
        input: Value,
        metadata: MetadataMap,
    },
    /// Reading a resource from an MCP server that requires auth.
    McpResourceRead {
        server_id: String,
        resource_id: String,
        metadata: MetadataMap,
    },
    /// Fetching a prompt from an MCP server that requires auth.
    McpPromptGet {
        server_id: String,
        prompt_id: String,
        args: Value,
        metadata: MetadataMap,
    },
    /// An application-defined operation that requires auth.
    Custom {
        kind: String,
        payload: Value,
        metadata: MetadataMap,
    },
}

impl AuthOperation {
    /// Returns the MCP server ID if this operation targets one, or looks it
    /// up in metadata for `ToolCall` and `Custom` variants.
    pub fn server_id(&self) -> Option<&str> {
        match self {
            Self::McpConnect { server_id, .. }
            | Self::McpToolCall { server_id, .. }
            | Self::McpResourceRead { server_id, .. }
            | Self::McpPromptGet { server_id, .. } => Some(server_id.as_str()),
            Self::ToolCall { metadata, .. } | Self::Custom { metadata, .. } => {
                metadata.get("server_id").and_then(Value::as_str)
            }
        }
    }
}

/// The outcome of an [`AuthRequest`] after the user interacts with the auth flow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthResolution {
    /// The user completed authentication and supplied credentials.
    Provided {
        /// The original auth request.
        request: AuthRequest,
        /// Credentials the user provided (tokens, keys, etc.).
        credentials: MetadataMap,
    },
    /// The user cancelled the authentication flow.
    Cancelled {
        /// The original auth request that was cancelled.
        request: AuthRequest,
    },
}

impl AuthResolution {
    /// Returns a reference to the underlying [`AuthRequest`] regardless of
    /// the resolution variant.
    pub fn request(&self) -> &AuthRequest {
        match self {
            Self::Provided { request, .. } | Self::Cancelled { request } => request,
        }
    }
}

impl AuthRequest {
    /// Convenience accessor that delegates to [`AuthOperation::server_id`].
    pub fn server_id(&self) -> Option<&str> {
        self.operation.server_id()
    }
}

/// A tool execution was paused because it needs external input.
///
/// The agent loop should handle the interruption (show a prompt, open an
/// OAuth flow, etc.) and then re-submit the tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolInterruption {
    /// The operation requires human approval before it can proceed.
    ApprovalRequired(ApprovalRequest),
    /// The operation requires authentication credentials.
    AuthRequired(AuthRequest),
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
///     .protect_root("/workspace/project/.env")
///     .require_approval_outside_allowed(true);
/// ```
pub struct PathPolicy {
    allowed_roots: Vec<PathBuf>,
    protected_roots: Vec<PathBuf>,
    require_approval_outside_allowed: bool,
}

impl PathPolicy {
    /// Creates a new path policy with no roots and approval required for
    /// paths outside allowed roots.
    pub fn new() -> Self {
        Self {
            allowed_roots: Vec::new(),
            protected_roots: Vec::new(),
            require_approval_outside_allowed: true,
        }
    }

    /// Adds a directory tree that filesystem operations are allowed to target.
    pub fn allow_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.allowed_roots.push(root.into());
        self
    }

    /// Adds a directory tree that filesystem operations are never allowed to target.
    pub fn protect_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.protected_roots.push(root.into());
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

impl PermissionPolicy for PathPolicy {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch {
        let Some(fs) = request
            .as_any()
            .downcast_ref::<FileSystemPermissionRequest>()
        else {
            return PolicyMatch::NoOpinion;
        };

        let candidate_paths: Vec<&Path> = match fs {
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

        if candidate_paths.iter().any(|path| {
            self.protected_roots
                .iter()
                .any(|root| path.starts_with(root))
        }) {
            return PolicyMatch::Deny(PermissionDenial {
                code: PermissionCode::PathNotAllowed,
                message: format!("path access denied for {}", fs.summary()),
                metadata: fs.metadata().clone(),
            });
        }

        if self.allowed_roots.is_empty() {
            return PolicyMatch::NoOpinion;
        }

        let all_allowed = candidate_paths
            .iter()
            .all(|path| self.allowed_roots.iter().any(|root| path.starts_with(root)));

        if all_allowed {
            PolicyMatch::Allow
        } else if self.require_approval_outside_allowed {
            PolicyMatch::RequireApproval(ApprovalRequest {
                task_id: None,
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
///             spec: ToolSpec {
///                 name: ToolName::new("current_time"),
///                 description: "Returns the current UTC time".into(),
///                 input_schema: json!({ "type": "object" }),
///                 annotations: Default::default(),
///                 metadata: MetadataMap::new(),
///             },
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
///         Ok(ToolResult {
///             result: ToolResultPart {
///                 call_id: request.call_id,
///                 output: ToolOutput::Text("2026-03-22T12:00:00Z".into()),
///                 is_error: false,
///                 metadata: MetadataMap::new(),
///             },
///             duration: None,
///             metadata: MetadataMap::new(),
///         })
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
    /// Return an appropriate [`ToolError`] variant on failure. Returning
    /// [`ToolError::AuthRequired`] causes the executor to emit a
    /// [`ToolInterruption::AuthRequired`] instead of treating it as a
    /// hard failure.
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
/// # use agentkit_core::MetadataMap;
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
///     .with(NoopTool(ToolSpec {
///         name: ToolName::new("noop"),
///         description: "Does nothing".into(),
///         input_schema: json!({"type": "object"}),
///         annotations: Default::default(),
///         metadata: MetadataMap::new(),
///     }));
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

    /// Returns the [`ToolSpec`] for every registered tool.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .filter_map(|tool| tool.current_spec())
            .collect()
    }
}

impl ToolSpec {
    /// Converts this spec into an [`InvocableSpec`] for use with the
    /// capability layer.
    pub fn as_invocable_spec(&self) -> InvocableSpec {
        InvocableSpec {
            name: CapabilityName::new(self.name.0.clone()),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            metadata: self.metadata.clone(),
        }
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

/// The default [`ToolExecutor`] that looks up tools in a [`ToolRegistry`],
/// checks permissions via [`Tool::proposed_requests`], and invokes the tool.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_tools_core::{BasicToolExecutor, ToolRegistry};
///
/// let registry = ToolRegistry::new();
/// let executor = BasicToolExecutor::new(registry);
/// // Pass `executor` to the agent loop.
/// ```
pub struct BasicToolExecutor {
    registry: ToolRegistry,
}

impl BasicToolExecutor {
    /// Creates an executor backed by the given registry.
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    /// Returns the [`ToolSpec`] for every tool in the underlying registry.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.registry.specs()
    }

    async fn execute_inner(
        &self,
        request: ToolRequest,
        approved_request_id: Option<&ApprovalId>,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        let Some(tool) = self.registry.get(&request.tool_name) else {
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
                        PermissionDecision::RequireApproval(req) => {
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
            Err(ToolError::AuthRequired(request)) => {
                ToolExecutionOutcome::Interrupted(ToolInterruption::AuthRequired(*request))
            }
            Err(error) => ToolExecutionOutcome::Failed(error),
        }
    }
}

#[async_trait]
impl ToolExecutor for BasicToolExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        self.registry.specs()
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
    /// The tool needs authentication credentials to proceed.
    ///
    /// The executor converts this into [`ToolInterruption::AuthRequired`].
    #[error("tool auth required: {0:?}")]
    AuthRequired(Box<AuthRequest>),
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
}
