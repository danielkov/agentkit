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
    ApprovalId, Item, ItemKind, MetadataMap, Part, SessionId, ToolCallId, ToolOutput,
    ToolResultPart, TurnId,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ToolName(pub String);

impl ToolName {
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only_hint: bool,
    pub destructive_hint: bool,
    pub idempotent_hint: bool,
    pub needs_approval_hint: bool,
    pub supports_streaming_hint: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: ToolName,
    pub description: String,
    pub input_schema: Value,
    pub annotations: ToolAnnotations,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolRequest {
    pub call_id: ToolCallId,
    pub tool_name: ToolName,
    pub input: Value,
    pub session_id: SessionId,
    pub turn_id: TurnId,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub result: ToolResultPart,
    pub duration: Option<Duration>,
    pub metadata: MetadataMap,
}

pub trait ToolResources: Send + Sync {}

impl ToolResources for () {}

pub struct ToolContext<'a> {
    pub capability: CapabilityContext<'a>,
    pub permissions: &'a dyn PermissionChecker,
    pub resources: &'a dyn ToolResources,
}

pub trait PermissionRequest: Send + Sync {
    fn kind(&self) -> &'static str;
    fn summary(&self) -> String;
    fn metadata(&self) -> &MetadataMap;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionCode {
    PathNotAllowed,
    CommandNotAllowed,
    NetworkNotAllowed,
    ServerNotTrusted,
    AuthScopeNotAllowed,
    CustomPolicyDenied,
    UnknownRequest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionDenial {
    pub code: PermissionCode,
    pub message: String,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalReason {
    PolicyRequiresConfirmation,
    EscalatedRisk,
    UnknownTarget,
    SensitivePath,
    SensitiveCommand,
    SensitiveServer,
    SensitiveAuthScope,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: ApprovalId,
    pub request_kind: String,
    pub reason: ApprovalReason,
    pub summary: String,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    Approve,
    Deny { reason: Option<String> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthRequest {
    pub provider: String,
    pub details: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolInterruption {
    ApprovalRequired(ApprovalRequest),
    AuthRequired(AuthRequest),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionDecision {
    Allow,
    Deny(PermissionDenial),
    RequireApproval(ApprovalRequest),
}

pub trait PermissionChecker: Send + Sync {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyMatch {
    NoOpinion,
    Allow,
    Deny(PermissionDenial),
    RequireApproval(ApprovalRequest),
}

pub trait PermissionPolicy: Send + Sync {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PolicyMatch;
}

pub struct CompositePermissionChecker {
    policies: Vec<Box<dyn PermissionPolicy>>,
    fallback: PermissionDecision,
}

impl CompositePermissionChecker {
    pub fn new(fallback: PermissionDecision) -> Self {
        Self {
            policies: Vec::new(),
            fallback,
        }
    }

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellPermissionRequest {
    pub executable: String,
    pub argv: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env_keys: Vec<String>,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileSystemPermissionRequest {
    Read {
        path: PathBuf,
        metadata: MetadataMap,
    },
    Write {
        path: PathBuf,
        metadata: MetadataMap,
    },
    Edit {
        path: PathBuf,
        metadata: MetadataMap,
    },
    Delete {
        path: PathBuf,
        metadata: MetadataMap,
    },
    Move {
        from: PathBuf,
        to: PathBuf,
        metadata: MetadataMap,
    },
    List {
        path: PathBuf,
        metadata: MetadataMap,
    },
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpPermissionRequest {
    Connect {
        server_id: String,
        metadata: MetadataMap,
    },
    InvokeTool {
        server_id: String,
        tool_name: String,
        metadata: MetadataMap,
    },
    ReadResource {
        server_id: String,
        resource_id: String,
        metadata: MetadataMap,
    },
    FetchPrompt {
        server_id: String,
        prompt_id: String,
        metadata: MetadataMap,
    },
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

pub struct CustomKindPolicy {
    allowed_kinds: BTreeSet<String>,
    denied_kinds: BTreeSet<String>,
    require_approval_by_default: bool,
}

impl CustomKindPolicy {
    pub fn new(require_approval_by_default: bool) -> Self {
        Self {
            allowed_kinds: BTreeSet::new(),
            denied_kinds: BTreeSet::new(),
            require_approval_by_default,
        }
    }

    pub fn allow_kind(mut self, kind: impl Into<String>) -> Self {
        self.allowed_kinds.insert(kind.into());
        self
    }

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

pub struct PathPolicy {
    allowed_roots: Vec<PathBuf>,
    protected_roots: Vec<PathBuf>,
    require_approval_outside_allowed: bool,
}

impl PathPolicy {
    pub fn new() -> Self {
        Self {
            allowed_roots: Vec::new(),
            protected_roots: Vec::new(),
            require_approval_outside_allowed: true,
        }
    }

    pub fn allow_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.allowed_roots.push(root.into());
        self
    }

    pub fn protect_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.protected_roots.push(root.into());
        self
    }

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

pub struct CommandPolicy {
    allowed_executables: BTreeSet<String>,
    denied_executables: BTreeSet<String>,
    allowed_cwds: Vec<PathBuf>,
    denied_env_keys: BTreeSet<String>,
    require_approval_for_unknown: bool,
}

impl CommandPolicy {
    pub fn new() -> Self {
        Self {
            allowed_executables: BTreeSet::new(),
            denied_executables: BTreeSet::new(),
            allowed_cwds: Vec::new(),
            denied_env_keys: BTreeSet::new(),
            require_approval_for_unknown: true,
        }
    }

    pub fn allow_executable(mut self, executable: impl Into<String>) -> Self {
        self.allowed_executables.insert(executable.into());
        self
    }

    pub fn deny_executable(mut self, executable: impl Into<String>) -> Self {
        self.denied_executables.insert(executable.into());
        self
    }

    pub fn allow_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.allowed_cwds.push(cwd.into());
        self
    }

    pub fn deny_env_key(mut self, key: impl Into<String>) -> Self {
        self.denied_env_keys.insert(key.into());
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

pub struct McpServerPolicy {
    trusted_servers: BTreeSet<String>,
    allowed_auth_scopes: BTreeSet<String>,
    require_approval_for_untrusted: bool,
}

impl McpServerPolicy {
    pub fn new() -> Self {
        Self {
            trusted_servers: BTreeSet::new(),
            allowed_auth_scopes: BTreeSet::new(),
            require_approval_for_untrusted: true,
        }
    }

    pub fn trust_server(mut self, server_id: impl Into<String>) -> Self {
        self.trusted_servers.insert(server_id.into());
        self
    }

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

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;

    fn proposed_requests(
        &self,
        _request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        Ok(Vec::new())
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError>;
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<ToolName, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T) -> &mut Self
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool.spec().name.clone(), Arc::new(tool));
        self
    }

    pub fn with<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.register(tool);
        self
    }

    pub fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| tool.spec().clone())
            .collect()
    }
}

impl ToolSpec {
    pub fn as_invocable_spec(&self) -> InvocableSpec {
        InvocableSpec {
            name: CapabilityName::new(self.name.0.clone()),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

pub struct ToolInvocableAdapter {
    spec: InvocableSpec,
    tool: Arc<dyn Tool>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    next_call_id: AtomicU64,
}

impl ToolInvocableAdapter {
    pub fn new(
        tool: Arc<dyn Tool>,
        permissions: Arc<dyn PermissionChecker>,
        resources: Arc<dyn ToolResources>,
    ) -> Self {
        let spec = tool.spec().as_invocable_spec();
        Self {
            spec,
            tool,
            permissions,
            resources,
            next_call_id: AtomicU64::new(1),
        }
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

pub struct ToolCapabilityProvider {
    invocables: Vec<Arc<dyn Invocable>>,
}

impl ToolCapabilityProvider {
    pub fn from_registry(
        registry: &ToolRegistry,
        permissions: Arc<dyn PermissionChecker>,
        resources: Arc<dyn ToolResources>,
    ) -> Self {
        let invocables = registry
            .tools()
            .into_iter()
            .map(|tool| {
                Arc::new(ToolInvocableAdapter::new(
                    tool,
                    permissions.clone(),
                    resources.clone(),
                )) as Arc<dyn Invocable>
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolExecutionOutcome {
    Completed(ToolResult),
    Interrupted(ToolInterruption),
    Failed(ToolError),
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome;
}

pub struct BasicToolExecutor {
    registry: ToolRegistry,
}

impl BasicToolExecutor {
    pub fn new(registry: ToolRegistry) -> Self {
        Self { registry }
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.registry.specs()
    }
}

#[async_trait]
impl ToolExecutor for BasicToolExecutor {
    async fn execute(
        &self,
        request: ToolRequest,
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
                            return ToolExecutionOutcome::Interrupted(
                                ToolInterruption::ApprovalRequired(req),
                            );
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

#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolError {
    #[error("tool not found: {0}")]
    NotFound(ToolName),
    #[error("invalid tool input: {0}")]
    InvalidInput(String),
    #[error("tool permission denied: {0:?}")]
    PermissionDenied(PermissionDenial),
    #[error("tool execution failed: {0}")]
    ExecutionFailed(String),
    #[error("tool unavailable: {0}")]
    Unavailable(String),
    #[error("internal tool error: {0}")]
    Internal(String),
}

impl ToolError {
    pub fn permission_denied(denial: PermissionDenial) -> Self {
        Self::PermissionDenied(denial)
    }
}

impl From<PermissionDenial> for ToolError {
    fn from(value: PermissionDenial) -> Self {
        Self::permission_denied(value)
    }
}
