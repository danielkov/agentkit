//! Model Context Protocol integration for agentkit, built on top of [`rmcp`].
//!
//! This crate exposes:
//!
//! - [`McpServerConfig`] / [`McpTransportBinding`] / [`StdioTransportConfig`] /
//!   [`StreamableHttpTransportConfig`] — declarative transport configuration.
//! - [`McpConnection`] — a live, single-server connection wrapping
//!   [`rmcp::service::RunningService`].
//! - [`McpServerManager`] — multi-server lifecycle, discovery, catalog diffing,
//!   and auth replay.
//! - [`McpServerHandle`], [`McpToolExecutor`], [`McpToolAdapter`],
//!   [`McpInvocable`], [`McpResourceHandle`], [`McpPromptHandle`],
//!   [`McpCapabilityProvider`] — bridges into the agentkit `Tool` / capabilities
//!   systems.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex as StdMutex, RwLock};

use agentkit_capabilities::{
    CapabilityContext, CapabilityError, CapabilityProvider, Invocable, PromptContents,
    PromptDescriptor, PromptId, PromptProvider, ResourceContents, ResourceDescriptor, ResourceId,
    ResourceProvider,
};
use agentkit_core::{
    DataRef, Item, ItemKind, MediaPart, MetadataMap, Modality, Part, TextPart, ToolOutput,
    ToolResultPart,
};
use agentkit_tools_core::{
    AuthOperation, AuthRequest, AuthResolution, PermissionChecker, PermissionDecision,
    PermissionRequest, Tool, ToolAnnotations, ToolCapabilityProvider, ToolCatalogEvent,
    ToolContext, ToolError, ToolExecutionOutcome, ToolExecutor, ToolName, ToolRegistry,
    ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use rmcp::ServiceExt;
use rmcp::handler::client::ClientHandler;
use rmcp::model as rmcp_model;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig as RmcpStreamableHttpClientTransportConfig;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::{Mutex, broadcast, mpsc};

/// Re-exports of the rmcp wire-protocol types this crate now surfaces directly
/// instead of wrapping. Pull these in to pattern-match on tool annotations,
/// content blocks, structured tool output, embedded resources, sampling /
/// elicitation requests, progress and log notifications, etc.
pub use rmcp::model::{
    Annotations as McpAnnotations, AudioContent, CallToolResult,
    CancelledNotificationParam as McpCancelledNotificationParam,
    ClientCapabilities as McpClientCapabilities, Content,
    CreateElicitationRequestParams as McpCreateElicitationRequestParams,
    CreateElicitationResult as McpCreateElicitationResult,
    CreateMessageRequestParams as McpCreateMessageRequestParams,
    CreateMessageResult as McpCreateMessageResult, ElicitationAction as McpElicitationAction,
    ElicitationCapability as McpElicitationCapability, EmbeddedResource,
    FormElicitationCapability as McpFormElicitationCapability, GetPromptResult, ImageContent,
    Implementation as McpImplementation, ListRootsResult as McpListRootsResult,
    LoggingLevel as McpLoggingLevel,
    LoggingMessageNotificationParam as McpLoggingMessageNotificationParam,
    ProgressNotificationParam as McpProgressNotificationParam, Prompt as McpPrompt, PromptArgument,
    PromptMessage, PromptMessageContent, PromptMessageRole, RawAudioContent, RawContent,
    RawEmbeddedResource, RawImageContent, RawResource as McpRawResource, RawTextContent,
    ReadResourceResult, Resource as McpResource,
    ResourceContents as McpResourceContents,
    ResourceUpdatedNotificationParam as McpResourceUpdatedNotificationParam, Root as McpRoot,
    RootsCapabilities as McpRootsCapabilities, SamplingCapability as McpSamplingCapability,
    SamplingMessage as McpSamplingMessage, SetLevelRequestParams as McpSetLevelRequestParams,
    TextContent, Tool as McpTool, ToolAnnotations as McpToolAnnotations,
    UrlElicitationCapability as McpUrlElicitationCapability,
};

/// Backwards-compatible alias — descriptors are now the rmcp wire types.
pub type McpToolDescriptor = McpTool;
/// Backwards-compatible alias — descriptors are now the rmcp wire types.
pub type McpResourceDescriptor = McpResource;
/// Backwards-compatible alias — descriptors are now the rmcp wire types.
pub type McpPromptDescriptor = McpPrompt;

/// Unique identifier for a registered MCP server.
///
/// Each MCP server in a [`McpServerManager`] is addressed by its `McpServerId`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct McpServerId(pub String);

impl McpServerId {
    /// Creates a new server identifier from any string-like value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for McpServerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Configuration for an MCP server that communicates over standard I/O.
///
/// The specified command is spawned as a child process; rmcp drives the
/// JSON-RPC framing over its stdin/stdout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StdioTransportConfig {
    /// The executable to launch (e.g. `"npx"`, `"python"`, `"node"`).
    pub command: String,
    /// Command-line arguments passed to the executable.
    pub args: Vec<String>,
    /// Additional environment variables set for the child process.
    pub env: Vec<(String, String)>,
    /// Optional working directory for the child process.
    pub cwd: Option<std::path::PathBuf>,
}

impl StdioTransportConfig {
    /// Creates a new stdio transport configuration for the given command.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
        }
    }

    /// Appends a command-line argument. Returns `self` for chaining.
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Adds an environment variable for the child process. Returns `self` for chaining.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Sets the working directory for the child process. Returns `self` for chaining.
    pub fn with_cwd(mut self, cwd: impl Into<std::path::PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

/// Configuration for an MCP server that communicates over the MCP Streamable HTTP transport.
#[derive(Clone, Debug, Default)]
pub struct StreamableHttpTransportConfig {
    /// The MCP endpoint URL to connect to.
    pub url: String,
    /// Static bearer token sent as an HTTP `Authorization: Bearer ...` header.
    pub bearer_token: Option<String>,
    /// Static custom HTTP headers sent with every Streamable HTTP request.
    pub headers: Vec<(HeaderName, HeaderValue)>,
}

impl StreamableHttpTransportConfig {
    /// Creates a new Streamable HTTP transport configuration for the given MCP endpoint.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            bearer_token: None,
            headers: Vec::new(),
        }
    }

    /// Sets a static bearer token for Streamable HTTP authorization.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Adds a static HTTP header for every Streamable HTTP request.
    ///
    /// Reserved MCP session and protocol headers are still managed by RMCP.
    pub fn with_header<N, V>(mut self, name: N, value: V) -> Result<Self, McpError>
    where
        N: TryInto<HeaderName>,
        N::Error: fmt::Display,
        V: TryInto<HeaderValue>,
        V::Error: fmt::Display,
    {
        let name = name
            .try_into()
            .map_err(|error| McpError::Transport(format!("invalid HTTP header name: {error}")))?;
        let value = value
            .try_into()
            .map_err(|error| McpError::Transport(format!("invalid HTTP header value: {error}")))?;
        self.headers.push((name, value));
        Ok(self)
    }
}

/// Selects which transport an MCP server should use.
#[derive(Clone, Debug)]
pub enum McpTransportBinding {
    /// Communicate over the child process's stdin/stdout.
    Stdio(StdioTransportConfig),
    /// Communicate over the MCP Streamable HTTP transport.
    StreamableHttp(StreamableHttpTransportConfig),
}

/// Full configuration for a single MCP server.
#[derive(Clone, Debug)]
pub struct McpServerConfig {
    /// Unique identifier for this server.
    pub id: McpServerId,
    /// Transport binding that determines how communication happens.
    pub transport: McpTransportBinding,
    /// Arbitrary metadata attached to this server configuration.
    pub metadata: MetadataMap,
}

impl McpServerConfig {
    /// Creates a new server configuration with the given identifier and transport.
    pub fn new(id: impl Into<String>, transport: McpTransportBinding) -> Self {
        Self {
            id: McpServerId::new(id),
            transport,
            metadata: MetadataMap::new(),
        }
    }

    /// Creates a stdio-backed server configuration.
    pub fn stdio(id: impl Into<String>, command: impl Into<String>) -> Self {
        Self::new(
            id,
            McpTransportBinding::Stdio(StdioTransportConfig::new(command)),
        )
    }

    /// Creates a Streamable HTTP-backed server configuration.
    pub fn streamable_http(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(
            id,
            McpTransportBinding::StreamableHttp(StreamableHttpTransportConfig::new(url)),
        )
    }

    /// Replaces the configuration metadata.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Strategy used to derive the agentkit-side tool name for an MCP tool.
///
/// The default (`Default`) preserves agentkit's historical
/// `mcp_<server>_<tool>` shape so that names satisfy provider validators
/// that only allow `[a-zA-Z0-9_-]` (e.g. Anthropic on Vertex). Use
/// [`McpToolNamespace::None`] when the calling provider already namespaces
/// remote tools, or [`McpToolNamespace::Custom`] for a bespoke scheme.
#[derive(Clone)]
pub enum McpToolNamespace {
    /// Format names as `mcp_<server>_<tool>`.
    Default,
    /// Use the raw MCP tool name with no prefix at all.
    None,
    /// Apply a caller-supplied function for full control.
    Custom(Arc<dyn Fn(&McpServerId, &str) -> String + Send + Sync>),
}

impl Default for McpToolNamespace {
    fn default() -> Self {
        Self::Default
    }
}

impl fmt::Debug for McpToolNamespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("McpToolNamespace::Default"),
            Self::None => f.write_str("McpToolNamespace::None"),
            Self::Custom(_) => f.write_str("McpToolNamespace::Custom(<fn>)"),
        }
    }
}

impl McpToolNamespace {
    /// Builds a custom namespace from a closure.
    pub fn custom(
        f: impl Fn(&McpServerId, &str) -> String + Send + Sync + 'static,
    ) -> Self {
        Self::Custom(Arc::new(f))
    }

    /// Applies the namespace strategy to produce the agentkit tool name.
    pub fn apply(&self, server_id: &McpServerId, tool_name: &str) -> String {
        match self {
            Self::Default => format!("mcp_{server_id}_{tool_name}"),
            Self::None => tool_name.to_string(),
            Self::Custom(f) => f(server_id, tool_name),
        }
    }
}

/// A snapshot of all capabilities discovered from a single MCP server.
///
/// Tools, resources, and prompts are stored as raw rmcp wire types
/// ([`McpTool`], [`McpResource`], [`McpPrompt`]) so that consumers see the
/// full typed surface — `Tool::annotations`, `Tool::output_schema`,
/// `Tool::execution`, `Tool::icons`; `Resource::title` / `mime_type` /
/// `size`; `Prompt::arguments` (with the typed `required` flag and per-arg
/// `description`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpDiscoverySnapshot {
    /// The server this snapshot was taken from.
    pub server_id: McpServerId,
    /// Tools advertised by the server.
    pub tools: Vec<McpTool>,
    /// Resources advertised by the server.
    pub resources: Vec<McpResource>,
    /// Prompts advertised by the server.
    pub prompts: Vec<McpPrompt>,
    /// Arbitrary metadata attached to this snapshot.
    pub metadata: MetadataMap,
}

/// Catalog and lifecycle events emitted by [`McpServerManager`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpCatalogEvent {
    /// A server connected and completed initial discovery.
    ServerConnected { server_id: McpServerId },
    /// A server disconnected.
    ServerDisconnected { server_id: McpServerId },
    /// The server's tool list changed.
    ToolsChanged {
        server_id: McpServerId,
        added: Vec<String>,
        removed: Vec<String>,
        changed: Vec<String>,
    },
    /// The server's resource list changed.
    ResourcesChanged {
        server_id: McpServerId,
        added: Vec<String>,
        removed: Vec<String>,
        changed: Vec<String>,
    },
    /// The server's prompt list changed.
    PromptsChanged {
        server_id: McpServerId,
        added: Vec<String>,
        removed: Vec<String>,
        changed: Vec<String>,
    },
    /// Authentication state changed for a server.
    AuthChanged { server_id: McpServerId },
    /// A catalog refresh failed.
    RefreshFailed {
        server_id: McpServerId,
        message: String,
    },
}

impl McpCatalogEvent {
    fn as_tool_catalog_event(&self) -> Option<ToolCatalogEvent> {
        match self {
            Self::ToolsChanged {
                server_id,
                added,
                removed,
                changed,
            } => Some(ToolCatalogEvent {
                source: format!("mcp:{server_id}"),
                added: added
                    .iter()
                    .map(|name| format!("mcp_{server_id}_{name}"))
                    .collect(),
                removed: removed
                    .iter()
                    .map(|name| format!("mcp_{server_id}_{name}"))
                    .collect(),
                changed: changed
                    .iter()
                    .map(|name| format!("mcp_{server_id}_{name}"))
                    .collect(),
            }),
            _ => None,
        }
    }
}

/// Capabilities advertised by an MCP server during the `initialize` handshake.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    /// Advertised `tools` capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Advertised `resources` capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Advertised `prompts` capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Advertised `logging` capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

impl McpServerCapabilities {
    /// Returns a capabilities struct with every top-level capability
    /// advertised. Useful for tests.
    pub fn all() -> Self {
        Self {
            tools: Some(ToolsCapability::default()),
            resources: Some(ResourcesCapability::default()),
            prompts: Some(PromptsCapability::default()),
            logging: Some(LoggingCapability::default()),
        }
    }
}

/// Tools sub-capability flags from the MCP `initialize` response.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// Server emits `notifications/tools/list_changed` when the catalog changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resources sub-capability flags from the MCP `initialize` response.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    /// Server supports `resources/subscribe`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// Server emits `notifications/resources/list_changed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Prompts sub-capability flags from the MCP `initialize` response.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    /// Server emits `notifications/prompts/list_changed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Logging sub-capability. Spec reserves the key with no defined sub-fields yet.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingCapability {}

/// Server-originated catalog notifications observed by [`McpClientHandler`].
///
/// Drained by [`McpConnection`] inside
/// [`McpServerManager::refresh_changed_catalogs`] to trigger re-discovery of
/// the affected capability lists. For richer push-style consumption (progress,
/// logging, resource updates, cancellation), subscribe via
/// [`McpConnection::subscribe_events`] and pattern-match on
/// [`McpServerEvent`].
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Debug)]
pub enum McpServerNotification {
    /// Server announced `notifications/tools/list_changed`.
    ToolsChanged,
    /// Server announced `notifications/resources/list_changed`.
    ResourcesChanged,
    /// Server announced `notifications/prompts/list_changed`.
    PromptsChanged,
}

/// Server-pushed events broadcast to every [`McpConnection::subscribe_events`]
/// receiver.
///
/// Covers the rmcp client-handler notification surface that does not feed the
/// catalog refresh path: progress, logging, resource updates, cancellation,
/// plus list-changed announcements (also delivered over the legacy
/// [`McpServerNotification`] channel).
#[derive(Clone, Debug)]
pub enum McpServerEvent {
    /// `notifications/progress` from the server, scoped to a
    /// `progress_token` issued in a previous request.
    Progress(McpProgressNotificationParam),
    /// `notifications/message` (server log emission). Drives the optional
    /// log-level negotiation initiated by [`McpConnection::set_logging_level`].
    Logging(McpLoggingMessageNotificationParam),
    /// `notifications/resources/updated` for a resource the client previously
    /// subscribed to via [`McpConnection::subscribe_resource`].
    ResourceUpdated(McpResourceUpdatedNotificationParam),
    /// `notifications/tools/list_changed`.
    ToolListChanged,
    /// `notifications/resources/list_changed`.
    ResourceListChanged,
    /// `notifications/prompts/list_changed`.
    PromptListChanged,
    /// `notifications/cancelled` from the server, requesting cancellation of
    /// an in-flight client request.
    Cancelled(McpCancelledNotificationParam),
}

/// Pluggable handler invoked when an MCP server issues `sampling/createMessage`.
///
/// Wire one in via
/// [`McpClientHandlerBuilder::with_sampling_responder`] (or
/// [`McpServerManager::with_sampling_responder`]) to expose the host
/// application's LLM as a sampling target for connected MCP servers.
#[async_trait]
pub trait McpSamplingResponder: Send + Sync + 'static {
    /// Produces a sampled completion in response to a server-initiated
    /// `sampling/createMessage` request.
    async fn create_message(
        &self,
        params: McpCreateMessageRequestParams,
    ) -> Result<McpCreateMessageResult, McpError>;
}

/// Pluggable handler invoked when an MCP server issues `elicitation/create`.
///
/// Wire one in via
/// [`McpClientHandlerBuilder::with_elicitation_responder`] (or
/// [`McpServerManager::with_elicitation_responder`]) to drive the
/// host application's user-input UI.
#[async_trait]
pub trait McpElicitationResponder: Send + Sync + 'static {
    /// Returns the user's response to a server-initiated elicitation request.
    async fn create_elicitation(
        &self,
        params: McpCreateElicitationRequestParams,
    ) -> Result<McpCreateElicitationResult, McpError>;
}

/// Pluggable handler invoked when an MCP server issues `roots/list`.
///
/// Wire one in via
/// [`McpClientHandlerBuilder::with_roots_provider`] (or
/// [`McpServerManager::with_roots_provider`]) to surface workspace roots
/// that scope the server's filesystem access.
#[async_trait]
pub trait McpRootsProvider: Send + Sync + 'static {
    /// Returns the roots the server should consider in scope.
    async fn list_roots(&self) -> Result<Vec<McpRoot>, McpError>;
}

/// Default broadcast capacity for [`McpServerEvent`] subscribers.
const DEFAULT_EVENTS_CAPACITY: usize = 128;

/// Channels paired with an [`McpClientHandler`] returned by
/// [`McpClientHandlerBuilder::build`].
///
/// `notifications` is the legacy mpsc receiver consumed by the catalog refresh
/// path inside [`McpServerManager::refresh_changed_catalogs`]. `events` is the
/// broadcast sender that surfaces every [`McpServerEvent`] — clone it once and
/// pass it into [`McpConnection::from_running_service_with_events`] when
/// adopting an externally constructed [`rmcp::service::RunningService`].
pub struct McpClientChannels {
    /// Legacy mpsc receiver for catalog list-changed announcements.
    pub notifications: mpsc::UnboundedReceiver<McpServerNotification>,
    /// Broadcast sender that forwards every [`McpServerEvent`] to subscribers.
    pub events: broadcast::Sender<McpServerEvent>,
}

/// Builder for [`McpClientHandler`].
///
/// Configure responders for server-initiated `sampling/createMessage`,
/// `elicitation/create`, and `roots/list` requests, then call [`Self::build`]
/// to obtain the handler and its paired [`McpClientChannels`].
#[derive(Clone, Default)]
pub struct McpClientHandlerBuilder {
    sampling: Option<Arc<dyn McpSamplingResponder>>,
    elicitation: Option<Arc<dyn McpElicitationResponder>>,
    roots: Option<Arc<dyn McpRootsProvider>>,
    events_capacity: Option<usize>,
}

impl McpClientHandlerBuilder {
    /// Creates an empty builder. By default no responders are wired.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a responder for server-initiated `sampling/createMessage`
    /// requests. The handler advertises `sampling` in `ClientCapabilities`.
    pub fn with_sampling_responder(mut self, responder: Arc<dyn McpSamplingResponder>) -> Self {
        self.sampling = Some(responder);
        self
    }

    /// Registers a responder for server-initiated `elicitation/create`
    /// requests. The handler advertises `elicitation` (form mode) in
    /// `ClientCapabilities`.
    pub fn with_elicitation_responder(
        mut self,
        responder: Arc<dyn McpElicitationResponder>,
    ) -> Self {
        self.elicitation = Some(responder);
        self
    }

    /// Registers a provider for `roots/list`. The handler advertises `roots`
    /// in `ClientCapabilities`.
    pub fn with_roots_provider(mut self, provider: Arc<dyn McpRootsProvider>) -> Self {
        self.roots = Some(provider);
        self
    }

    /// Overrides the default broadcast capacity for the [`McpServerEvent`]
    /// channel.
    pub fn with_events_capacity(mut self, capacity: usize) -> Self {
        self.events_capacity = Some(capacity);
        self
    }

    /// Consumes the builder, returning the configured handler plus the channel
    /// receiver and broadcast sender it writes into.
    pub fn build(self) -> (McpClientHandler, McpClientChannels) {
        let (notifications_tx, notifications_rx) = mpsc::unbounded_channel();
        let events_capacity = self.events_capacity.unwrap_or(DEFAULT_EVENTS_CAPACITY);
        let (events_tx, _) = broadcast::channel(events_capacity);

        let mut capabilities = rmcp_model::ClientCapabilities::default();
        if self.sampling.is_some() {
            capabilities.sampling = Some(McpSamplingCapability::default());
        }
        if self.elicitation.is_some() {
            capabilities.elicitation = Some(McpElicitationCapability {
                form: Some(McpFormElicitationCapability::default()),
                url: None,
            });
        }
        if self.roots.is_some() {
            capabilities.roots = Some(McpRootsCapabilities::default());
        }

        let handler = McpClientHandler {
            info: rmcp_model::ClientInfo::new(
                capabilities,
                rmcp_model::Implementation::new("agentkit-mcp", env!("CARGO_PKG_VERSION"))
                    .with_title("agentkit MCP client"),
            )
            .with_protocol_version(rmcp_model::ProtocolVersion::LATEST),
            notifications: notifications_tx,
            events: events_tx.clone(),
            sampling: self.sampling,
            elicitation: self.elicitation,
            roots: self.roots,
        };

        (
            handler,
            McpClientChannels {
                notifications: notifications_rx,
                events: events_tx,
            },
        )
    }
}

/// rmcp [`ClientHandler`] used by [`McpConnection`].
///
/// You only need to construct this directly if you're wiring rmcp transports
/// that [`McpTransportBinding`] does not cover (in-memory pipes, websockets,
/// custom IO). Build one via [`McpClientHandlerBuilder`] (for full event +
/// responder access) or the back-compat [`Self::with_channel`] shortcut, then
/// pair the resulting service with [`McpConnection::from_running_service`] /
/// [`McpConnection::from_running_service_with_events`].
#[derive(Clone)]
pub struct McpClientHandler {
    info: rmcp_model::ClientInfo,
    notifications: mpsc::UnboundedSender<McpServerNotification>,
    events: broadcast::Sender<McpServerEvent>,
    sampling: Option<Arc<dyn McpSamplingResponder>>,
    elicitation: Option<Arc<dyn McpElicitationResponder>>,
    roots: Option<Arc<dyn McpRootsProvider>>,
}

impl McpClientHandler {
    /// Returns a default [`McpClientHandlerBuilder`].
    pub fn builder() -> McpClientHandlerBuilder {
        McpClientHandlerBuilder::new()
    }

    /// Builds a handler together with the notification receiver that
    /// [`McpConnection::from_running_service`] expects. No responders are
    /// wired, and the broadcast events sender is dropped — use
    /// [`McpClientHandlerBuilder`] + [`McpConnection::from_running_service_with_events`]
    /// when you need server-initiated request handling or event subscription.
    pub fn with_channel() -> (Self, mpsc::UnboundedReceiver<McpServerNotification>) {
        let (handler, channels) = McpClientHandlerBuilder::new().build();
        (handler, channels.notifications)
    }
}

impl ClientHandler for McpClientHandler {
    fn create_message(
        &self,
        params: rmcp_model::CreateMessageRequestParams,
        _context: rmcp::service::RequestContext<RoleClient>,
    ) -> impl Future<Output = Result<rmcp_model::CreateMessageResult, rmcp_model::ErrorData>>
    + rmcp::service::MaybeSendFuture
    + '_ {
        let responder = self.sampling.clone();
        async move {
            match responder {
                Some(responder) => responder.create_message(params).await.map_err(Into::into),
                None => Err(rmcp_model::ErrorData::method_not_found::<
                    rmcp_model::CreateMessageRequestMethod,
                >()),
            }
        }
    }

    fn list_roots(
        &self,
        _context: rmcp::service::RequestContext<RoleClient>,
    ) -> impl Future<Output = Result<rmcp_model::ListRootsResult, rmcp_model::ErrorData>>
    + rmcp::service::MaybeSendFuture
    + '_ {
        let provider = self.roots.clone();
        async move {
            match provider {
                Some(provider) => provider
                    .list_roots()
                    .await
                    .map(McpListRootsResult::new)
                    .map_err(Into::into),
                None => Ok(McpListRootsResult::default()),
            }
        }
    }

    fn create_elicitation(
        &self,
        params: rmcp_model::CreateElicitationRequestParams,
        _context: rmcp::service::RequestContext<RoleClient>,
    ) -> impl Future<Output = Result<rmcp_model::CreateElicitationResult, rmcp_model::ErrorData>>
    + rmcp::service::MaybeSendFuture
    + '_ {
        let responder = self.elicitation.clone();
        async move {
            match responder {
                Some(responder) => responder
                    .create_elicitation(params)
                    .await
                    .map_err(Into::into),
                None => Ok(McpCreateElicitationResult::new(McpElicitationAction::Decline)),
            }
        }
    }

    fn on_progress(
        &self,
        params: rmcp_model::ProgressNotificationParam,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self.events.send(McpServerEvent::Progress(params));
        std::future::ready(())
    }

    fn on_logging_message(
        &self,
        params: rmcp_model::LoggingMessageNotificationParam,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self.events.send(McpServerEvent::Logging(params));
        std::future::ready(())
    }

    fn on_resource_updated(
        &self,
        params: rmcp_model::ResourceUpdatedNotificationParam,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self.events.send(McpServerEvent::ResourceUpdated(params));
        std::future::ready(())
    }

    fn on_cancelled(
        &self,
        params: rmcp_model::CancelledNotificationParam,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self.events.send(McpServerEvent::Cancelled(params));
        std::future::ready(())
    }

    fn on_tool_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self.notifications.send(McpServerNotification::ToolsChanged);
        let _ = self.events.send(McpServerEvent::ToolListChanged);
        std::future::ready(())
    }

    fn on_resource_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self
            .notifications
            .send(McpServerNotification::ResourcesChanged);
        let _ = self.events.send(McpServerEvent::ResourceListChanged);
        std::future::ready(())
    }

    fn on_prompt_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self
            .notifications
            .send(McpServerNotification::PromptsChanged);
        let _ = self.events.send(McpServerEvent::PromptListChanged);
        std::future::ready(())
    }

    fn get_info(&self) -> rmcp_model::ClientInfo {
        self.info.clone()
    }
}

impl From<McpError> for rmcp_model::ErrorData {
    fn from(error: McpError) -> Self {
        rmcp_model::ErrorData::internal_error(error.to_string(), None)
    }
}

type RmcpClientService = RunningService<RoleClient, McpClientHandler>;

/// Configuration applied to every [`McpClientHandler`] this crate builds on
/// behalf of a connection or [`McpServerManager`].
///
/// Holds the optional sampling / elicitation / roots responders plus the
/// broadcast capacity for [`McpServerEvent`] subscribers. Pass an instance to
/// [`McpConnection::connect_with_handler`] to drive a single connection, or
/// install one on the manager via
/// [`McpServerManager::with_handler_config`] / per-trait builders.
#[derive(Clone, Default)]
pub struct McpHandlerConfig {
    /// Responder for server-initiated `sampling/createMessage` requests.
    pub sampling: Option<Arc<dyn McpSamplingResponder>>,
    /// Responder for server-initiated `elicitation/create` requests.
    pub elicitation: Option<Arc<dyn McpElicitationResponder>>,
    /// Provider for `roots/list`.
    pub roots: Option<Arc<dyn McpRootsProvider>>,
    /// Broadcast capacity for the [`McpServerEvent`] channel. Defaults to
    /// `DEFAULT_EVENTS_CAPACITY` when `None`.
    pub events_capacity: Option<usize>,
}

impl McpHandlerConfig {
    /// Returns an empty handler config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the sampling responder.
    pub fn with_sampling_responder(mut self, responder: Arc<dyn McpSamplingResponder>) -> Self {
        self.sampling = Some(responder);
        self
    }

    /// Sets the elicitation responder.
    pub fn with_elicitation_responder(
        mut self,
        responder: Arc<dyn McpElicitationResponder>,
    ) -> Self {
        self.elicitation = Some(responder);
        self
    }

    /// Sets the roots provider.
    pub fn with_roots_provider(mut self, provider: Arc<dyn McpRootsProvider>) -> Self {
        self.roots = Some(provider);
        self
    }

    /// Sets the broadcast capacity for [`McpServerEvent`] subscribers.
    pub fn with_events_capacity(mut self, capacity: usize) -> Self {
        self.events_capacity = Some(capacity);
        self
    }

    fn build_handler(
        &self,
        events: Option<broadcast::Sender<McpServerEvent>>,
    ) -> (McpClientHandler, McpClientChannels) {
        let (notifications_tx, notifications_rx) = mpsc::unbounded_channel();
        let events_tx = events.unwrap_or_else(|| {
            let capacity = self.events_capacity.unwrap_or(DEFAULT_EVENTS_CAPACITY);
            let (tx, _) = broadcast::channel(capacity);
            tx
        });

        let mut capabilities = rmcp_model::ClientCapabilities::default();
        if self.sampling.is_some() {
            capabilities.sampling = Some(McpSamplingCapability::default());
        }
        if self.elicitation.is_some() {
            capabilities.elicitation = Some(McpElicitationCapability {
                form: Some(McpFormElicitationCapability::default()),
                url: None,
            });
        }
        if self.roots.is_some() {
            capabilities.roots = Some(McpRootsCapabilities::default());
        }

        let handler = McpClientHandler {
            info: rmcp_model::ClientInfo::new(
                capabilities,
                rmcp_model::Implementation::new("agentkit-mcp", env!("CARGO_PKG_VERSION"))
                    .with_title("agentkit MCP client"),
            )
            .with_protocol_version(rmcp_model::ProtocolVersion::LATEST),
            notifications: notifications_tx,
            events: events_tx.clone(),
            sampling: self.sampling.clone(),
            elicitation: self.elicitation.clone(),
            roots: self.roots.clone(),
        };

        (
            handler,
            McpClientChannels {
                notifications: notifications_rx,
                events: events_tx,
            },
        )
    }
}

/// A live connection to a single MCP server, wrapping an
/// [`rmcp::service::RunningService`].
pub struct McpConnection {
    server_id: McpServerId,
    config: Option<McpServerConfig>,
    inner: Mutex<RmcpClientService>,
    auth: Mutex<Option<MetadataMap>>,
    notifications: Mutex<mpsc::UnboundedReceiver<McpServerNotification>>,
    events: broadcast::Sender<McpServerEvent>,
    handler_config: McpHandlerConfig,
    capabilities: McpServerCapabilities,
}

/// The result of replaying an MCP operation after auth resolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum McpOperationResult {
    /// The server was successfully (re)connected; contains the discovery snapshot.
    Connected(McpDiscoverySnapshot),
    /// A tool call completed; contains the typed rmcp [`CallToolResult`].
    Tool(CallToolResult),
    /// A resource was read successfully.
    Resource(ReadResourceResult),
    /// A prompt was retrieved successfully.
    Prompt(GetPromptResult),
}

impl McpConnection {
    /// Connects to an MCP server, performs the rmcp `initialize` handshake,
    /// and returns a ready-to-use connection. No sampling / elicitation /
    /// roots responders are wired; use [`Self::connect_with_handler`] when
    /// the server may issue those requests.
    pub async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        Self::connect_with_auth(config, None, McpHandlerConfig::default()).await
    }

    /// Connects to an MCP server with a fully configured [`McpHandlerConfig`].
    pub async fn connect_with_handler(
        config: &McpServerConfig,
        handler_config: McpHandlerConfig,
    ) -> Result<Self, McpError> {
        Self::connect_with_auth(config, None, handler_config).await
    }

    async fn connect_with_auth(
        config: &McpServerConfig,
        auth: Option<&MetadataMap>,
        handler_config: McpHandlerConfig,
    ) -> Result<Self, McpError> {
        let (handler, channels) = handler_config.build_handler(None);
        let McpClientChannels {
            notifications: notification_rx,
            events: events_tx,
        } = channels;
        let (service, capabilities) = match &config.transport {
            McpTransportBinding::Stdio(binding) => {
                connect_rmcp_stdio(config, binding, handler).await?
            }
            McpTransportBinding::StreamableHttp(binding) => {
                connect_rmcp_streamable_http(config, binding, auth, handler).await?
            }
        };

        Ok(Self {
            server_id: config.id.clone(),
            config: Some(config.clone()),
            inner: Mutex::new(service),
            auth: Mutex::new(auth.cloned()),
            notifications: Mutex::new(notification_rx),
            events: events_tx,
            handler_config,
            capabilities,
        })
    }

    /// Adopts an externally constructed [`rmcp::service::RunningService`] as
    /// an [`McpConnection`].
    ///
    /// Use this when you need a transport rmcp supports but
    /// [`McpTransportBinding`] does not (in-memory pipes for tests, websockets,
    /// custom IO). Pair the service with the notification receiver returned by
    /// [`McpClientHandler::with_channel`] so list-change notifications stay
    /// observable.
    ///
    /// The connection has no [`McpServerConfig`] attached, so reconnect-on-auth
    /// is unavailable; [`resolve_auth`](Self::resolve_auth) only updates stored
    /// credentials in this mode. Server-pushed events from the underlying
    /// handler are *not* forwarded to subscribers — use
    /// [`Self::from_running_service_with_events`] paired with the broadcast
    /// sender from [`McpClientChannels`] when you need event delivery.
    pub fn from_running_service(
        server_id: impl Into<McpServerId>,
        service: RmcpClientService,
        notifications: mpsc::UnboundedReceiver<McpServerNotification>,
    ) -> Self {
        let (events_tx, _) = broadcast::channel(DEFAULT_EVENTS_CAPACITY);
        Self::from_running_service_with_events(server_id, service, notifications, events_tx)
    }

    /// Variant of [`Self::from_running_service`] that wires the broadcast
    /// sender returned by [`McpClientHandlerBuilder::build`] / [`McpClientHandler::builder`]
    /// so [`Self::subscribe_events`] receivers observe the same stream the
    /// handler is publishing into.
    pub fn from_running_service_with_events(
        server_id: impl Into<McpServerId>,
        service: RmcpClientService,
        notifications: mpsc::UnboundedReceiver<McpServerNotification>,
        events: broadcast::Sender<McpServerEvent>,
    ) -> Self {
        let capabilities = service
            .peer_info()
            .map(|info| rmcp_server_capabilities_to_agentkit(&info.capabilities))
            .unwrap_or_default();
        Self {
            server_id: server_id.into(),
            config: None,
            inner: Mutex::new(service),
            auth: Mutex::new(None),
            notifications: Mutex::new(notifications),
            events,
            handler_config: McpHandlerConfig::default(),
            capabilities,
        }
    }

    async fn reconnect_inner(&self, auth: Option<&MetadataMap>) -> Result<(), McpError> {
        let Some(config) = self.config.clone() else {
            return Ok(());
        };
        let (handler, channels) = self
            .handler_config
            .build_handler(Some(self.events.clone()));
        let McpClientChannels {
            notifications: notification_rx,
            ..
        } = channels;
        let (service, _capabilities) = match &config.transport {
            McpTransportBinding::Stdio(binding) => {
                connect_rmcp_stdio(&config, binding, handler).await?
            }
            McpTransportBinding::StreamableHttp(binding) => {
                connect_rmcp_streamable_http(&config, binding, auth, handler).await?
            }
        };
        *self.notifications.lock().await = notification_rx;
        *self.inner.lock().await = service;
        Ok(())
    }

    /// Returns the [`McpServerId`] for this connection.
    pub fn server_id(&self) -> &McpServerId {
        &self.server_id
    }

    /// Returns the capabilities advertised by the server during `initialize`.
    pub fn capabilities(&self) -> &McpServerCapabilities {
        &self.capabilities
    }

    /// Subscribes to the per-connection [`McpServerEvent`] broadcast.
    ///
    /// Receivers buffer up to `events_capacity` (configured via
    /// [`McpHandlerConfig::with_events_capacity`], defaults to
    /// `DEFAULT_EVENTS_CAPACITY`) before slow consumers are signalled with
    /// [`broadcast::error::RecvError::Lagged`]. Catalog `*ListChanged` events
    /// are also delivered through the legacy [`McpServerNotification`]
    /// receiver consumed by [`McpServerManager::refresh_changed_catalogs`].
    pub fn subscribe_events(&self) -> broadcast::Receiver<McpServerEvent> {
        self.events.subscribe()
    }

    /// Subscribes to `notifications/resources/updated` for the given URI.
    ///
    /// Updates surface as [`McpServerEvent::ResourceUpdated`] on every
    /// receiver returned by [`Self::subscribe_events`].
    pub async fn subscribe_resource(&self, uri: impl Into<String>) -> Result<(), McpError> {
        let uri = uri.into();
        let inner = self.inner.lock().await;
        inner
            .subscribe(rmcp_model::SubscribeRequestParams::new(uri.clone()))
            .await
            .map_err(|error| {
                rmcp_operation_error(
                    &self.server_id,
                    "resources/subscribe",
                    json!({ "uri": uri }),
                    error,
                )
            })
    }

    /// Cancels a previous [`Self::subscribe_resource`] subscription.
    pub async fn unsubscribe_resource(&self, uri: impl Into<String>) -> Result<(), McpError> {
        let uri = uri.into();
        let inner = self.inner.lock().await;
        inner
            .unsubscribe(rmcp_model::UnsubscribeRequestParams::new(uri.clone()))
            .await
            .map_err(|error| {
                rmcp_operation_error(
                    &self.server_id,
                    "resources/unsubscribe",
                    json!({ "uri": uri }),
                    error,
                )
            })
    }

    /// Negotiates the minimum severity the server should emit through
    /// `notifications/message`. Surfaced as [`McpServerEvent::Logging`].
    pub async fn set_logging_level(&self, level: McpLoggingLevel) -> Result<(), McpError> {
        let inner = self.inner.lock().await;
        inner
            .set_level(rmcp_model::SetLevelRequestParams::new(level))
            .await
            .map_err(|error| {
                rmcp_operation_error(
                    &self.server_id,
                    "logging/setLevel",
                    json!({ "level": format!("{level:?}") }),
                    error,
                )
            })
    }

    /// Sends a `notifications/cancelled` to the server, asking it to stop
    /// processing a previously issued request.
    pub async fn notify_cancelled(
        &self,
        params: McpCancelledNotificationParam,
    ) -> Result<(), McpError> {
        let inner = self.inner.lock().await;
        inner
            .notify_cancelled(params)
            .await
            .map_err(rmcp_service_error)
    }

    /// Notifies the server that the client's roots list has changed; servers
    /// may respond by re-issuing `roots/list`.
    pub async fn notify_roots_list_changed(&self) -> Result<(), McpError> {
        let inner = self.inner.lock().await;
        inner
            .notify_roots_list_changed()
            .await
            .map_err(rmcp_service_error)
    }

    /// Gracefully closes the underlying rmcp service.
    ///
    /// For Streamable HTTP this drives the rmcp transport to issue a `DELETE`
    /// against the negotiated session, releasing server-side state.
    pub async fn close(&self) -> Result<(), McpError> {
        let mut inner = self.inner.lock().await;
        inner
            .close()
            .await
            .map(|_| ())
            .map_err(|error| McpError::Transport(format!("rmcp service close failed: {error}")))
    }

    /// Stores or clears authentication credentials and, when configured to do
    /// so via [`McpServerConfig`], reconnects to apply them.
    pub async fn resolve_auth(&self, resolution: AuthResolution) -> Result<(), McpError> {
        let mut auth_slot = self.auth.lock().await;
        match resolution {
            AuthResolution::Provided { credentials, .. } => {
                *auth_slot = Some(credentials);
            }
            AuthResolution::Cancelled { .. } => {
                *auth_slot = None;
            }
        }
        let snapshot = auth_slot.clone();
        drop(auth_slot);
        // Only reconnect if we have a config to reconnect with. Without one
        // (e.g. constructed via [`from_running_service`]) the auth is stored
        // but not pushed to the live transport.
        if self.config.is_some() {
            self.reconnect_inner(snapshot.as_ref()).await?;
        }
        Ok(())
    }

    /// Discovers tools, resources, and prompts that the server advertised.
    pub async fn discover(&self) -> Result<McpDiscoverySnapshot, McpError> {
        let tools = if self.capabilities.tools.is_some() {
            self.list_tools().await?
        } else {
            Vec::new()
        };
        let resources = if self.capabilities.resources.is_some() {
            self.list_resources().await?
        } else {
            Vec::new()
        };
        let prompts = if self.capabilities.prompts.is_some() {
            self.list_prompts().await?
        } else {
            Vec::new()
        };
        Ok(McpDiscoverySnapshot {
            server_id: self.server_id.clone(),
            tools,
            resources,
            prompts,
            metadata: MetadataMap::new(),
        })
    }

    async fn drain_notifications(&self) -> Vec<McpServerNotification> {
        let mut notifications = self.notifications.lock().await;
        let mut drained = Vec::new();
        while let Ok(notification) = notifications.try_recv() {
            drained.push(notification);
        }
        drained
    }

    /// Lists all tools advertised by the connected MCP server.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        let inner = self.inner.lock().await;
        inner.list_all_tools().await.map_err(rmcp_service_error)
    }

    /// Lists all resources advertised by the connected MCP server.
    pub async fn list_resources(&self) -> Result<Vec<McpResource>, McpError> {
        let inner = self.inner.lock().await;
        inner.list_all_resources().await.map_err(rmcp_service_error)
    }

    /// Lists all prompts advertised by the connected MCP server.
    pub async fn list_prompts(&self) -> Result<Vec<McpPrompt>, McpError> {
        let inner = self.inner.lock().await;
        inner.list_all_prompts().await.map_err(rmcp_service_error)
    }

    /// Invokes a tool on the MCP server.
    ///
    /// Returns the typed [`CallToolResult`] — the [`Vec<Content>`] block list,
    /// the optional `structured_content` field, and the `is_error` flag are
    /// all preserved. Adapters convert this into agentkit
    /// [`ToolOutput`]/[`InvocableOutput`] at the boundary.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<CallToolResult, McpError> {
        let inner = self.inner.lock().await;
        let mut params = rmcp_model::CallToolRequestParams::new(name.to_string());
        if !arguments.is_null() {
            params =
                params.with_arguments(value_to_json_object(arguments, "tools/call arguments")?);
        }
        inner.call_tool(params).await.map_err(|error| {
            rmcp_operation_error(
                &self.server_id,
                "tools/call",
                json!({ "name": name }),
                error,
            )
        })
    }

    /// Reads a resource from the MCP server by URI.
    ///
    /// Returns the typed [`ReadResourceResult`] — the full
    /// [`Vec<McpResourceContents>`] is preserved (text vs blob, mime types,
    /// metadata). Use [`McpResourceHandle`] for the agentkit
    /// [`ResourceProvider`] view that collapses to a single inline `DataRef`.
    pub async fn read_resource(&self, uri: &str) -> Result<ReadResourceResult, McpError> {
        let inner = self.inner.lock().await;
        inner
            .read_resource(rmcp_model::ReadResourceRequestParams::new(uri))
            .await
            .map_err(|error| {
                rmcp_operation_error(
                    &self.server_id,
                    "resources/read",
                    json!({ "uri": uri }),
                    error,
                )
            })
    }

    /// Retrieves a prompt from the MCP server, rendering it with the given
    /// arguments.
    ///
    /// Returns the typed [`GetPromptResult`] — message role and content
    /// blocks (text/image/audio/embedded resource) are preserved. Use
    /// [`McpPromptHandle`] for the collapsed agentkit [`PromptProvider`]
    /// view.
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<GetPromptResult, McpError> {
        let inner = self.inner.lock().await;
        let mut params = rmcp_model::GetPromptRequestParams::new(name);
        if !arguments.is_null() {
            params =
                params.with_arguments(value_to_json_object(arguments, "prompts/get arguments")?);
        }
        inner.get_prompt(params).await.map_err(|error| {
            rmcp_operation_error(
                &self.server_id,
                "prompts/get",
                json!({ "name": name }),
                error,
            )
        })
    }

    /// Replays an MCP operation that previously failed with an auth challenge.
    pub async fn replay_auth_operation(
        &self,
        operation: &AuthOperation,
    ) -> Result<McpOperationResult, McpError> {
        match operation {
            AuthOperation::McpToolCall {
                server_id,
                tool_name,
                input,
                ..
            } => {
                self.ensure_server_match(server_id)?;
                self.call_tool(tool_name, input.clone())
                    .await
                    .map(McpOperationResult::Tool)
            }
            AuthOperation::McpResourceRead {
                server_id,
                resource_id,
                ..
            } => {
                self.ensure_server_match(server_id)?;
                self.read_resource(resource_id)
                    .await
                    .map(McpOperationResult::Resource)
            }
            AuthOperation::McpPromptGet {
                server_id,
                prompt_id,
                args,
                ..
            } => {
                self.ensure_server_match(server_id)?;
                self.get_prompt(prompt_id, args.clone())
                    .await
                    .map(McpOperationResult::Prompt)
            }
            AuthOperation::ToolCall {
                tool_name,
                input,
                metadata,
                ..
            } => {
                if let Some(server_id) = metadata.get("server_id").and_then(Value::as_str) {
                    self.ensure_server_match(server_id)?;
                }
                let tool_name = normalize_mcp_tool_name(self.server_id(), tool_name);
                self.call_tool(&tool_name, input.clone())
                    .await
                    .map(McpOperationResult::Tool)
            }
            AuthOperation::McpConnect { .. } => Err(McpError::AuthResolution(
                "connect operations must be replayed through the server manager".into(),
            )),
            AuthOperation::Custom { kind, .. } => Err(McpError::AuthResolution(format!(
                "unsupported auth operation for replay: {kind}"
            ))),
        }
    }

    fn ensure_server_match(&self, server_id: &str) -> Result<(), McpError> {
        if self.server_id.0 == server_id {
            Ok(())
        } else {
            Err(McpError::AuthResolution(format!(
                "auth operation targets server {server_id}, but connection is for {}",
                self.server_id
            )))
        }
    }
}

async fn connect_rmcp_stdio(
    config: &McpServerConfig,
    binding: &StdioTransportConfig,
    handler: McpClientHandler,
) -> Result<(RmcpClientService, McpServerCapabilities), McpError> {
    let transport = TokioChildProcess::new(
        tokio::process::Command::new(&binding.command).configure(|command| {
            command.args(&binding.args);
            if let Some(cwd) = &binding.cwd {
                command.current_dir(cwd);
            }
            for (key, value) in &binding.env {
                command.env(key, value);
            }
        }),
    )
    .map_err(McpError::Io)?;

    let service = handler
        .serve(transport)
        .await
        .map_err(|error| rmcp_initialize_error(config, error))?;
    let capabilities = service
        .peer_info()
        .map(|info| rmcp_server_capabilities_to_agentkit(&info.capabilities))
        .unwrap_or_default();

    Ok((service, capabilities))
}

async fn connect_rmcp_streamable_http(
    config: &McpServerConfig,
    binding: &StreamableHttpTransportConfig,
    auth: Option<&MetadataMap>,
    handler: McpClientHandler,
) -> Result<(RmcpClientService, McpServerCapabilities), McpError> {
    let auth_header = auth
        .and_then(bearer_token_from_metadata)
        .or_else(|| binding.bearer_token.clone());
    let mut rmcp_config = RmcpStreamableHttpClientTransportConfig::with_uri(binding.url.clone());
    if let Some(auth_header) = auth_header {
        rmcp_config = rmcp_config.auth_header(auth_header);
    }
    rmcp_config = rmcp_config.custom_headers(binding.headers.iter().cloned().collect());
    let transport = StreamableHttpClientTransport::from_config(rmcp_config);

    let service = handler
        .serve(transport)
        .await
        .map_err(|error| rmcp_initialize_error(config, error))?;
    let capabilities = service
        .peer_info()
        .map(|info| rmcp_server_capabilities_to_agentkit(&info.capabilities))
        .unwrap_or_default();

    Ok((service, capabilities))
}

/// Adapter exposing a single MCP resource as a [`ResourceProvider`].
pub struct McpResourceHandle {
    connection: Arc<McpConnection>,
    descriptor: ResourceDescriptor,
}

#[async_trait]
impl ResourceProvider for McpResourceHandle {
    async fn list_resources(&self) -> Result<Vec<ResourceDescriptor>, CapabilityError> {
        Ok(vec![self.descriptor.clone()])
    }

    async fn read_resource(
        &self,
        id: &ResourceId,
        _ctx: &mut CapabilityContext<'_>,
    ) -> Result<ResourceContents, CapabilityError> {
        let result = self
            .connection
            .read_resource(&id.0)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => {
                    CapabilityError::Unavailable(format!("auth required: {:?}", request))
                }
                other => CapabilityError::ExecutionFailed(other.to_string()),
            })?;
        read_resource_result_to_capabilities(result)
            .map_err(|error| CapabilityError::ExecutionFailed(error.to_string()))
    }
}

/// Adapter exposing a single MCP prompt as a [`PromptProvider`].
pub struct McpPromptHandle {
    connection: Arc<McpConnection>,
    descriptor: PromptDescriptor,
}

#[async_trait]
impl PromptProvider for McpPromptHandle {
    async fn list_prompts(&self) -> Result<Vec<PromptDescriptor>, CapabilityError> {
        Ok(vec![self.descriptor.clone()])
    }

    async fn get_prompt(
        &self,
        id: &PromptId,
        args: Value,
        _ctx: &mut CapabilityContext<'_>,
    ) -> Result<PromptContents, CapabilityError> {
        let result = self
            .connection
            .get_prompt(&id.0, args)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => {
                    CapabilityError::Unavailable(format!("auth required: {:?}", request))
                }
                other => CapabilityError::ExecutionFailed(other.to_string()),
            })?;
        Ok(get_prompt_result_to_capabilities(result))
    }
}

/// A [`CapabilityProvider`] that surfaces MCP tools, resources, and prompts.
///
/// The tool side is built by wrapping [`McpToolAdapter`]s in
/// [`agentkit_tools_core::ToolInvocableAdapter`], so the same
/// permission-check + adapter-spec plumbing the rest of agentkit uses also
/// applies to MCP tools — this crate no longer ships its own
/// `McpInvocable`.
pub struct McpCapabilityProvider {
    invocables: Vec<Arc<dyn Invocable>>,
    resources: Vec<Arc<dyn ResourceProvider>>,
    prompts: Vec<Arc<dyn PromptProvider>>,
}

impl McpCapabilityProvider {
    /// Builds a capability provider from an existing connection and snapshot,
    /// using the [`McpToolNamespace::Default`] tool naming strategy.
    pub fn from_snapshot(connection: Arc<McpConnection>, snapshot: &McpDiscoverySnapshot) -> Self {
        Self::from_snapshot_with_namespace(connection, snapshot, &McpToolNamespace::Default)
    }

    /// Builds a capability provider with a custom tool naming strategy.
    pub fn from_snapshot_with_namespace(
        connection: Arc<McpConnection>,
        snapshot: &McpDiscoverySnapshot,
        namespace: &McpToolNamespace,
    ) -> Self {
        let server_id = connection.server_id().clone();
        let registry = snapshot.tools.iter().cloned().fold(
            ToolRegistry::new(),
            |registry, tool| {
                registry.with(McpToolAdapter::with_namespace(
                    &server_id,
                    connection.clone(),
                    tool,
                    namespace,
                ))
            },
        );
        let permissions: Arc<dyn PermissionChecker> = Arc::new(McpAllowAllPermissions);
        let resources_arc: Arc<dyn agentkit_tools_core::ToolResources> = Arc::new(());
        let invocables = ToolCapabilityProvider::from_registry(
            &registry,
            permissions,
            resources_arc,
        )
        .invocables();

        let resources = snapshot
            .resources
            .iter()
            .cloned()
            .map(|resource| {
                Arc::new(McpResourceHandle {
                    connection: connection.clone(),
                    descriptor: resource_descriptor_from_rmcp(resource),
                }) as Arc<dyn ResourceProvider>
            })
            .collect();

        let prompts = snapshot
            .prompts
            .iter()
            .cloned()
            .map(|prompt| {
                Arc::new(McpPromptHandle {
                    connection: connection.clone(),
                    descriptor: prompt_descriptor_from_rmcp(prompt),
                }) as Arc<dyn PromptProvider>
            })
            .collect();

        Self {
            invocables,
            resources,
            prompts,
        }
    }

    /// Merges multiple capability providers into one.
    pub fn merge<I>(providers: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        let mut invocables = Vec::new();
        let mut resources = Vec::new();
        let mut prompts = Vec::new();

        for provider in providers {
            invocables.extend(provider.invocables);
            resources.extend(provider.resources);
            prompts.extend(provider.prompts);
        }

        Self {
            invocables,
            resources,
            prompts,
        }
    }

    /// Connects to an MCP server, performs discovery, and builds a provider.
    pub async fn connect(
        config: &McpServerConfig,
    ) -> Result<(Arc<McpConnection>, Self, McpDiscoverySnapshot), McpError> {
        let connection = Arc::new(McpConnection::connect(config).await?);
        let snapshot = connection.discover().await?;
        let provider = Self::from_snapshot(connection.clone(), &snapshot);

        Ok((connection, provider, snapshot))
    }
}

impl CapabilityProvider for McpCapabilityProvider {
    fn invocables(&self) -> Vec<Arc<dyn Invocable>> {
        self.invocables.clone()
    }

    fn resources(&self) -> Vec<Arc<dyn ResourceProvider>> {
        self.resources.clone()
    }

    fn prompts(&self) -> Vec<Arc<dyn PromptProvider>> {
        self.prompts.clone()
    }
}

/// Permission checker that approves every request. Used internally by
/// [`McpCapabilityProvider`] when bridging MCP tools through the standard
/// agentkit invocable adapter — MCP servers are gated upstream at connection
/// time, not per-call.
struct McpAllowAllPermissions;

impl PermissionChecker for McpAllowAllPermissions {
    fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

/// A connected MCP server together with its configuration and snapshot.
#[derive(Clone)]
pub struct McpServerHandle {
    config: McpServerConfig,
    connection: Arc<McpConnection>,
    snapshot: McpDiscoverySnapshot,
    namespace: McpToolNamespace,
}

impl McpServerHandle {
    /// Returns the original configuration used to connect this server.
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Returns the server's unique identifier.
    pub fn server_id(&self) -> &McpServerId {
        self.connection.server_id()
    }

    /// Returns a shared reference to the underlying [`McpConnection`].
    pub fn connection(&self) -> Arc<McpConnection> {
        self.connection.clone()
    }

    /// Returns the discovery snapshot captured when the server was connected.
    pub fn snapshot(&self) -> &McpDiscoverySnapshot {
        &self.snapshot
    }

    /// Returns the tool naming strategy in effect for this server.
    pub fn namespace(&self) -> &McpToolNamespace {
        &self.namespace
    }

    /// Builds a [`ToolRegistry`] containing an [`McpToolAdapter`] for each tool.
    pub fn tool_registry(&self) -> ToolRegistry {
        self.snapshot
            .tools
            .iter()
            .cloned()
            .fold(ToolRegistry::new(), |registry, tool| {
                registry.with(McpToolAdapter::with_namespace(
                    self.server_id(),
                    self.connection.clone(),
                    tool,
                    &self.namespace,
                ))
            })
    }

    /// Builds an [`McpCapabilityProvider`] from this server's snapshot.
    pub fn capability_provider(&self) -> McpCapabilityProvider {
        McpCapabilityProvider::from_snapshot_with_namespace(
            self.connection.clone(),
            &self.snapshot,
            &self.namespace,
        )
    }
}

/// Manages the lifecycle of one or more MCP servers.
pub struct McpServerManager {
    configs: BTreeMap<McpServerId, McpServerConfig>,
    connections: BTreeMap<McpServerId, McpServerHandle>,
    auth: BTreeMap<McpServerId, MetadataMap>,
    catalog_tx: broadcast::Sender<McpCatalogEvent>,
    namespace: McpToolNamespace,
    handler_config: McpHandlerConfig,
}

impl Default for McpServerManager {
    fn default() -> Self {
        let (catalog_tx, _) = broadcast::channel(128);
        Self {
            configs: BTreeMap::new(),
            connections: BTreeMap::new(),
            auth: BTreeMap::new(),
            catalog_tx,
            namespace: McpToolNamespace::Default,
            handler_config: McpHandlerConfig::default(),
        }
    }
}

impl McpServerManager {
    /// Creates an empty server manager with no registered servers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the tool naming strategy for every adapter built by this manager.
    pub fn with_namespace(mut self, namespace: McpToolNamespace) -> Self {
        self.namespace = namespace;
        self
    }

    /// Replaces the tool naming strategy in place.
    pub fn set_namespace(&mut self, namespace: McpToolNamespace) -> &mut Self {
        self.namespace = namespace;
        self
    }

    /// Returns the active tool naming strategy.
    pub fn namespace(&self) -> &McpToolNamespace {
        &self.namespace
    }

    /// Replaces the [`McpHandlerConfig`] applied to every connection this
    /// manager opens.
    pub fn with_handler_config(mut self, handler_config: McpHandlerConfig) -> Self {
        self.handler_config = handler_config;
        self
    }

    /// Sets the [`McpHandlerConfig`] in place.
    pub fn set_handler_config(&mut self, handler_config: McpHandlerConfig) -> &mut Self {
        self.handler_config = handler_config;
        self
    }

    /// Returns the active [`McpHandlerConfig`].
    pub fn handler_config(&self) -> &McpHandlerConfig {
        &self.handler_config
    }

    /// Convenience builder that installs a sampling responder on the manager's
    /// [`McpHandlerConfig`]. Subsequent connections will advertise the
    /// `sampling` client capability.
    pub fn with_sampling_responder(mut self, responder: Arc<dyn McpSamplingResponder>) -> Self {
        self.handler_config.sampling = Some(responder);
        self
    }

    /// Convenience builder that installs an elicitation responder.
    pub fn with_elicitation_responder(
        mut self,
        responder: Arc<dyn McpElicitationResponder>,
    ) -> Self {
        self.handler_config.elicitation = Some(responder);
        self
    }

    /// Convenience builder that installs a roots provider.
    pub fn with_roots_provider(mut self, provider: Arc<dyn McpRootsProvider>) -> Self {
        self.handler_config.roots = Some(provider);
        self
    }

    /// Registers a server configuration. Returns `self` for chaining.
    pub fn with_server(mut self, config: McpServerConfig) -> Self {
        self.register_server(config);
        self
    }

    /// Registers a server configuration by mutable reference.
    pub fn register_server(&mut self, config: McpServerConfig) -> &mut Self {
        self.configs.insert(config.id.clone(), config);
        self
    }

    /// Returns the handle for a connected server, or `None` if not connected.
    pub fn connected_server(&self, server_id: &McpServerId) -> Option<&McpServerHandle> {
        self.connections.get(server_id)
    }

    /// Returns handles for all currently connected servers.
    pub fn connected_servers(&self) -> Vec<&McpServerHandle> {
        self.connections.values().collect()
    }

    /// Subscribes to MCP catalog and lifecycle events.
    pub fn subscribe_catalog_events(&self) -> broadcast::Receiver<McpCatalogEvent> {
        self.catalog_tx.subscribe()
    }

    fn emit_catalog_event(&self, event: McpCatalogEvent) {
        let _ = self.catalog_tx.send(event);
    }

    /// Connects a single registered server by its identifier.
    pub async fn connect_server(
        &mut self,
        server_id: &McpServerId,
    ) -> Result<McpServerHandle, McpError> {
        let config = self
            .configs
            .get(server_id)
            .cloned()
            .ok_or_else(|| McpError::UnknownServer(server_id.to_string()))?;
        let connection = Arc::new(
            McpConnection::connect_with_auth(
                &config,
                self.auth.get(server_id),
                self.handler_config.clone(),
            )
            .await?,
        );
        let snapshot = connection.discover().await?;
        let handle = McpServerHandle {
            config,
            connection,
            snapshot,
            namespace: self.namespace.clone(),
        };
        self.connections.insert(server_id.clone(), handle.clone());
        self.emit_catalog_event(McpCatalogEvent::ServerConnected {
            server_id: server_id.clone(),
        });
        Ok(handle)
    }

    /// Connects all registered servers sequentially.
    pub async fn connect_all(&mut self) -> Result<Vec<McpServerHandle>, McpError> {
        let server_ids = self.configs.keys().cloned().collect::<Vec<_>>();
        let mut handles = Vec::with_capacity(server_ids.len());
        for server_id in server_ids {
            handles.push(self.connect_server(&server_id).await?);
        }
        Ok(handles)
    }

    /// Re-discovers capabilities for a connected server.
    pub async fn refresh_server(
        &mut self,
        server_id: &McpServerId,
    ) -> Result<McpDiscoverySnapshot, McpError> {
        let handle = self
            .connections
            .get_mut(server_id)
            .ok_or_else(|| McpError::UnknownServer(server_id.to_string()))?;
        let previous = handle.snapshot.clone();
        let snapshot = match handle.connection.discover().await {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.emit_catalog_event(McpCatalogEvent::RefreshFailed {
                    server_id: server_id.clone(),
                    message: error.to_string(),
                });
                return Err(error);
            }
        };
        handle.snapshot = snapshot.clone();
        for event in diff_discovery_snapshots(server_id, &previous, &snapshot) {
            self.emit_catalog_event(event);
        }
        Ok(snapshot)
    }

    /// Processes pending server list-change notifications.
    pub async fn refresh_changed_catalogs(&mut self) -> Result<Vec<McpCatalogEvent>, McpError> {
        let server_ids = self.connections.keys().cloned().collect::<Vec<_>>();
        let mut emitted = Vec::new();

        for server_id in server_ids {
            let Some(connection) = self
                .connections
                .get(&server_id)
                .map(McpServerHandle::connection)
            else {
                continue;
            };
            let notifications = connection.drain_notifications().await;
            if notifications.is_empty() {
                continue;
            }

            let handle = self
                .connections
                .get_mut(&server_id)
                .ok_or_else(|| McpError::UnknownServer(server_id.to_string()))?;
            let previous = handle.snapshot.clone();
            let snapshot = match handle.connection.discover().await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    let event = McpCatalogEvent::RefreshFailed {
                        server_id: server_id.clone(),
                        message: error.to_string(),
                    };
                    self.emit_catalog_event(event.clone());
                    emitted.push(event);
                    return Err(error);
                }
            };
            handle.snapshot = snapshot.clone();
            for event in diff_discovery_snapshots(&server_id, &previous, &snapshot) {
                self.emit_catalog_event(event.clone());
                emitted.push(event);
            }
        }

        Ok(emitted)
    }

    /// Disconnects a server and removes it from active connections.
    pub async fn disconnect_server(&mut self, server_id: &McpServerId) -> Result<(), McpError> {
        let Some(handle) = self.connections.remove(server_id) else {
            return Err(McpError::UnknownServer(server_id.to_string()));
        };
        handle.connection.close().await?;
        self.emit_catalog_event(McpCatalogEvent::ServerDisconnected {
            server_id: server_id.clone(),
        });
        Ok(())
    }

    /// Stores or clears authentication credentials for a server.
    pub async fn resolve_auth(&mut self, resolution: AuthResolution) -> Result<(), McpError> {
        let server_id = resolution
            .request()
            .server_id()
            .ok_or_else(|| McpError::AuthResolution("auth resolution missing server id".into()))?;
        let server_id = McpServerId::new(server_id);
        match &resolution {
            AuthResolution::Provided { credentials, .. } => {
                self.auth.insert(server_id.clone(), credentials.clone());
            }
            AuthResolution::Cancelled { .. } => {
                self.auth.remove(&server_id);
            }
        }

        if let Some(handle) = self.connections.get(&server_id) {
            handle.connection.resolve_auth(resolution).await?;
            self.emit_catalog_event(McpCatalogEvent::AuthChanged { server_id });
            return Ok(());
        }

        if self.configs.contains_key(&server_id) {
            self.emit_catalog_event(McpCatalogEvent::AuthChanged { server_id });
            Ok(())
        } else {
            Err(McpError::UnknownServer(server_id.to_string()))
        }
    }

    /// Resolves auth and immediately replays the operation that triggered the challenge.
    pub async fn resolve_auth_and_resume(
        &mut self,
        resolution: AuthResolution,
    ) -> Result<McpOperationResult, McpError> {
        let request = resolution.request().clone();
        self.resolve_auth(resolution).await?;
        self.replay_auth_request(&request).await
    }

    /// Replays an auth request's original MCP operation using stored credentials.
    pub async fn replay_auth_request(
        &mut self,
        request: &AuthRequest,
    ) -> Result<McpOperationResult, McpError> {
        match &request.operation {
            AuthOperation::McpConnect { server_id, .. } => {
                let server_id = McpServerId::new(server_id);
                let handle = self.connect_server(&server_id).await?;
                Ok(McpOperationResult::Connected(handle.snapshot.clone()))
            }
            AuthOperation::McpToolCall { server_id, .. }
            | AuthOperation::McpResourceRead { server_id, .. }
            | AuthOperation::McpPromptGet { server_id, .. } => {
                let connection = self.connection_for_auth_server(server_id).await?;
                connection.replay_auth_operation(&request.operation).await
            }
            AuthOperation::ToolCall { metadata, .. } => {
                let server_id = metadata
                    .get("server_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        McpError::AuthResolution(
                            "tool-call auth replay requires metadata.server_id".into(),
                        )
                    })?;
                let connection = self.connection_for_auth_server(server_id).await?;
                connection.replay_auth_operation(&request.operation).await
            }
            AuthOperation::Custom { kind, .. } => Err(McpError::AuthResolution(format!(
                "unsupported auth operation for replay: {kind}"
            ))),
        }
    }

    async fn connection_for_auth_server(
        &mut self,
        server_id: &str,
    ) -> Result<Arc<McpConnection>, McpError> {
        let server_id = McpServerId::new(server_id);
        if !self.connections.contains_key(&server_id) {
            self.connect_server(&server_id).await?;
        }
        self.connections
            .get(&server_id)
            .map(McpServerHandle::connection)
            .ok_or_else(|| McpError::UnknownServer(server_id.to_string()))
    }

    /// Builds a combined [`ToolRegistry`] for every tool across all connected servers.
    ///
    /// Tool names are prefixed `mcp_<server_id>_<tool_name>`.
    pub fn tool_registry(&self) -> ToolRegistry {
        self.connections
            .values()
            .fold(ToolRegistry::new(), |mut registry, handle| {
                for tool in handle.snapshot.tools.iter().cloned() {
                    registry.register(McpToolAdapter::new(
                        handle.server_id(),
                        handle.connection.clone(),
                        tool,
                    ));
                }
                registry
            })
    }

    /// Builds a combined [`McpCapabilityProvider`] from all connected servers.
    pub fn capability_provider(&self) -> McpCapabilityProvider {
        McpCapabilityProvider::merge(
            self.connections
                .values()
                .map(McpServerHandle::capability_provider),
        )
    }

    /// Builds an MCP-backed executor from the current snapshot.
    pub fn tool_executor(&self) -> McpToolExecutor {
        McpToolExecutor::from_manager(self)
    }

    fn connection_map(&self) -> BTreeMap<McpServerId, Arc<McpConnection>> {
        self.connections
            .iter()
            .map(|(server_id, handle)| (server_id.clone(), handle.connection()))
            .collect()
    }
}

fn diff_discovery_snapshots(
    server_id: &McpServerId,
    previous: &McpDiscoverySnapshot,
    current: &McpDiscoverySnapshot,
) -> Vec<McpCatalogEvent> {
    let mut events = Vec::new();
    let (added, removed, changed) = diff_named_items(
        previous
            .tools
            .iter()
            .map(|item| (item.name.to_string(), item)),
        current
            .tools
            .iter()
            .map(|item| (item.name.to_string(), item)),
    );
    if !added.is_empty() || !removed.is_empty() || !changed.is_empty() {
        events.push(McpCatalogEvent::ToolsChanged {
            server_id: server_id.clone(),
            added,
            removed,
            changed,
        });
    }

    let (added, removed, changed) = diff_named_items(
        previous
            .resources
            .iter()
            .map(|item| (item.uri.clone(), item)),
        current
            .resources
            .iter()
            .map(|item| (item.uri.clone(), item)),
    );
    if !added.is_empty() || !removed.is_empty() || !changed.is_empty() {
        events.push(McpCatalogEvent::ResourcesChanged {
            server_id: server_id.clone(),
            added,
            removed,
            changed,
        });
    }

    let (added, removed, changed) = diff_named_items(
        previous
            .prompts
            .iter()
            .map(|item| (item.name.clone(), item)),
        current
            .prompts
            .iter()
            .map(|item| (item.name.clone(), item)),
    );
    if !added.is_empty() || !removed.is_empty() || !changed.is_empty() {
        events.push(McpCatalogEvent::PromptsChanged {
            server_id: server_id.clone(),
            added,
            removed,
            changed,
        });
    }

    events
}

fn diff_named_items<'a, T>(
    previous: impl IntoIterator<Item = (String, &'a T)>,
    current: impl IntoIterator<Item = (String, &'a T)>,
) -> (Vec<String>, Vec<String>, Vec<String>)
where
    T: PartialEq + 'a,
{
    let previous = previous.into_iter().collect::<BTreeMap<_, _>>();
    let current = current.into_iter().collect::<BTreeMap<_, _>>();

    let added = current
        .keys()
        .filter(|name| !previous.contains_key(*name))
        .cloned()
        .collect();
    let removed = previous
        .keys()
        .filter(|name| !current.contains_key(*name))
        .cloned()
        .collect();
    let changed = current
        .iter()
        .filter(|(name, item)| {
            previous
                .get(*name)
                .is_some_and(|previous_item| previous_item != *item)
        })
        .map(|(name, _)| name.clone())
        .collect();

    (added, removed, changed)
}

/// A tool executor backed by MCP tool adapters.
#[derive(Clone)]
pub struct McpToolExecutor {
    registry: Arc<RwLock<ToolRegistry>>,
    connections: Arc<RwLock<BTreeMap<McpServerId, Arc<McpConnection>>>>,
    events: Arc<StdMutex<Vec<ToolCatalogEvent>>>,
}

impl McpToolExecutor {
    /// Creates an executor from a manager's current connected snapshot.
    pub fn from_manager(manager: &McpServerManager) -> Self {
        Self {
            registry: Arc::new(RwLock::new(manager.tool_registry())),
            connections: Arc::new(RwLock::new(manager.connection_map())),
            events: Arc::new(StdMutex::new(Vec::new())),
        }
    }

    /// Refreshes the executor registry from the manager's current snapshot.
    pub fn refresh_from_manager(&self, manager: &McpServerManager) {
        *self
            .registry
            .write()
            .expect("MCP tool registry lock poisoned") = manager.tool_registry();
        *self
            .connections
            .write()
            .expect("MCP connection map lock poisoned") = manager.connection_map();
    }

    /// Queues a catalog event for the loop-facing [`ToolExecutor`] API.
    pub fn push_catalog_event(&self, event: McpCatalogEvent) {
        if let Some(event) = event.as_tool_catalog_event() {
            self.events
                .lock()
                .expect("MCP catalog event lock poisoned")
                .push(event);
        }
    }

    fn executor(&self) -> Result<agentkit_tools_core::BasicToolExecutor, ToolError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| ToolError::Internal("MCP tool registry lock poisoned".into()))?
            .clone();
        Ok(agentkit_tools_core::BasicToolExecutor::new(registry))
    }
}

#[async_trait]
impl ToolExecutor for McpToolExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        self.registry
            .read()
            .expect("MCP tool registry lock poisoned")
            .specs()
    }

    fn drain_catalog_events(&self) -> Vec<ToolCatalogEvent> {
        std::mem::take(&mut *self.events.lock().expect("MCP catalog event lock poisoned"))
    }

    async fn execute(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        let Ok(executor) = self.executor() else {
            return ToolExecutionOutcome::Failed(ToolError::Internal(
                "MCP tool registry lock poisoned".into(),
            ));
        };
        executor.execute(request, ctx).await
    }

    async fn execute_approved(
        &self,
        request: ToolRequest,
        approved_request: &agentkit_tools_core::ApprovalRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        let Ok(executor) = self.executor() else {
            return ToolExecutionOutcome::Failed(ToolError::Internal(
                "MCP tool registry lock poisoned".into(),
            ));
        };
        executor
            .execute_approved(request, approved_request, ctx)
            .await
    }

    async fn execute_after_auth(
        &self,
        request: ToolRequest,
        auth_resolution: &AuthResolution,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        if let Some(server_id) = auth_resolution.request().server_id() {
            let connection = self
                .connections
                .read()
                .map_err(|_| ToolError::Internal("MCP connection map lock poisoned".into()))
                .and_then(|connections| {
                    connections
                        .get(&McpServerId::new(server_id))
                        .cloned()
                        .ok_or_else(|| {
                            ToolError::Unavailable(format!("MCP server not connected: {server_id}"))
                        })
                });
            match connection {
                Ok(connection) => {
                    if let Err(error) = connection.resolve_auth(auth_resolution.clone()).await {
                        return ToolExecutionOutcome::Failed(ToolError::ExecutionFailed(
                            error.to_string(),
                        ));
                    }
                }
                Err(error) => return ToolExecutionOutcome::Failed(error),
            }
        }

        let Ok(executor) = self.executor() else {
            return ToolExecutionOutcome::Failed(ToolError::Internal(
                "MCP tool registry lock poisoned".into(),
            ));
        };
        executor
            .execute_after_auth(request, auth_resolution, ctx)
            .await
    }
}

/// Adapter exposing an MCP tool as an agentkit [`Tool`].
pub struct McpToolAdapter {
    tool_name: String,
    connection: Arc<McpConnection>,
    spec: ToolSpec,
}

impl McpToolAdapter {
    /// Creates a new tool adapter for the given MCP tool, using the
    /// [`McpToolNamespace::Default`] naming strategy.
    pub fn new(server_id: &McpServerId, connection: Arc<McpConnection>, tool: McpTool) -> Self {
        Self::with_namespace(server_id, connection, tool, &McpToolNamespace::Default)
    }

    /// Creates a new tool adapter with a custom name-namespacing strategy.
    pub fn with_namespace(
        server_id: &McpServerId,
        connection: Arc<McpConnection>,
        tool: McpTool,
        namespace: &McpToolNamespace,
    ) -> Self {
        let spec = tool_spec_from_tool(server_id, &tool, namespace);
        Self {
            tool_name: tool.name.into_owned(),
            connection,
            spec,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let result = self
            .connection
            .call_tool(&self.tool_name, request.input)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => ToolError::AuthRequired(request),
                other => ToolError::ExecutionFailed(other.to_string()),
            })?;

        let is_error = result.is_error.unwrap_or(false);
        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: call_tool_result_to_tool_output(result),
                is_error,
                metadata: MetadataMap::new(),
            },
            duration: None,
            metadata: MetadataMap::new(),
        })
    }
}

fn rmcp_server_capabilities_to_agentkit(
    capabilities: &rmcp_model::ServerCapabilities,
) -> McpServerCapabilities {
    McpServerCapabilities {
        tools: capabilities.tools.as_ref().map(|tools| ToolsCapability {
            list_changed: tools.list_changed,
        }),
        resources: capabilities
            .resources
            .as_ref()
            .map(|resources| ResourcesCapability {
                subscribe: resources.subscribe,
                list_changed: resources.list_changed,
            }),
        prompts: capabilities
            .prompts
            .as_ref()
            .map(|prompts| PromptsCapability {
                list_changed: prompts.list_changed,
            }),
        logging: capabilities.logging.as_ref().map(|_| LoggingCapability {}),
    }
}

fn tool_spec_from_tool(
    server_id: &McpServerId,
    tool: &McpTool,
    namespace: &McpToolNamespace,
) -> ToolSpec {
    ToolSpec {
        name: ToolName::new(namespace.apply(server_id, &tool.name)),
        description: tool
            .description
            .as_ref()
            .map(|d| d.to_string())
            .unwrap_or_else(|| tool.name.to_string()),
        input_schema: Value::Object((*tool.input_schema).clone()),
        annotations: tool_annotations_from_rmcp(tool.annotations.as_ref()),
        metadata: MetadataMap::new(),
    }
}

fn tool_annotations_from_rmcp(annotations: Option<&McpToolAnnotations>) -> ToolAnnotations {
    let Some(annotations) = annotations else {
        return ToolAnnotations::default();
    };
    // rmcp expresses each hint as `Option<bool>` (advisory; absent means
    // unspecified). agentkit collapses absent → false. Tools that need to
    // distinguish "absent" from "false" should inspect the underlying
    // `McpTool::annotations` directly via the snapshot.
    ToolAnnotations {
        read_only_hint: annotations.read_only_hint.unwrap_or(false),
        destructive_hint: annotations.destructive_hint.unwrap_or(false),
        idempotent_hint: annotations.idempotent_hint.unwrap_or(false),
        needs_approval_hint: annotations.destructive_hint.unwrap_or(false),
        supports_streaming_hint: false,
    }
}

fn resource_descriptor_from_rmcp(resource: McpResource) -> ResourceDescriptor {
    let raw = resource.raw;
    ResourceDescriptor {
        id: ResourceId::new(raw.uri),
        name: raw.name,
        description: raw.description,
        mime_type: raw.mime_type,
        metadata: MetadataMap::new(),
    }
}

fn prompt_descriptor_from_rmcp(prompt: McpPrompt) -> PromptDescriptor {
    let arguments = prompt.arguments.unwrap_or_default();
    let mut required = Vec::new();
    let properties = arguments
        .into_iter()
        .map(|argument| {
            let mut schema = serde_json::Map::new();
            schema.insert("type".into(), Value::String("string".into()));
            if let Some(description) = argument.description {
                schema.insert("description".into(), Value::String(description));
            }
            if argument.required.unwrap_or(false) {
                required.push(Value::String(argument.name.clone()));
            }
            (argument.name, Value::Object(schema))
        })
        .collect::<serde_json::Map<String, Value>>();
    let mut input_schema = serde_json::Map::new();
    input_schema.insert("type".into(), Value::String("object".into()));
    input_schema.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        input_schema.insert("required".into(), Value::Array(required));
    }

    PromptDescriptor {
        id: PromptId::new(prompt.name.clone()),
        name: prompt.name,
        description: prompt.description,
        input_schema: Value::Object(input_schema),
        metadata: MetadataMap::new(),
    }
}

fn read_resource_result_to_capabilities(
    result: ReadResourceResult,
) -> Result<ResourceContents, McpError> {
    let content = result
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| McpError::Protocol("resources/read returned no contents".into()))?;
    Ok(resource_contents_to_capabilities(content))
}

fn resource_contents_to_capabilities(content: McpResourceContents) -> ResourceContents {
    let mut metadata = MetadataMap::new();
    let data = match content {
        McpResourceContents::TextResourceContents {
            text, mime_type, ..
        } => {
            if let Some(mime) = mime_type {
                metadata.insert("mime_type".into(), Value::String(mime));
            }
            DataRef::InlineText(text)
        }
        McpResourceContents::BlobResourceContents {
            blob,
            mime_type,
            uri,
            ..
        } => {
            if let Some(mime) = mime_type {
                metadata.insert("mime_type".into(), Value::String(mime));
            }
            metadata.insert("uri".into(), Value::String(uri));
            // rmcp delivers blobs as base64-encoded text on the wire.
            DataRef::InlineText(blob)
        }
    };
    ResourceContents { data, metadata }
}

fn get_prompt_result_to_capabilities(result: GetPromptResult) -> PromptContents {
    let items = result
        .messages
        .into_iter()
        .map(prompt_message_to_item)
        .collect();
    let mut metadata = MetadataMap::new();
    if let Some(description) = result.description {
        metadata.insert("description".into(), Value::String(description));
    }
    PromptContents { items, metadata }
}

fn prompt_message_to_item(message: PromptMessage) -> Item {
    let kind = match message.role {
        PromptMessageRole::Assistant => ItemKind::Assistant,
        PromptMessageRole::User => ItemKind::User,
    };
    Item {
        id: None,
        kind,
        parts: vec![prompt_message_content_to_part(message.content)],
        metadata: MetadataMap::new(),
    }
}

fn prompt_message_content_to_part(content: PromptMessageContent) -> Part {
    match content {
        PromptMessageContent::Text { text } => Part::Text(TextPart::new(text)),
        PromptMessageContent::Image { image } => Part::Media(MediaPart::new(
            Modality::Image,
            image.mime_type.clone(),
            DataRef::InlineText(image.data.clone()),
        )),
        PromptMessageContent::Resource { resource } => {
            let agentkit_resource = resource_contents_to_capabilities(resource.resource.clone());
            agentkit_part_from_resource(agentkit_resource)
        }
        PromptMessageContent::ResourceLink { link } => {
            Part::Text(TextPart::new(link.uri.clone()))
        }
    }
}

fn agentkit_part_from_resource(resource: ResourceContents) -> Part {
    let mime = resource
        .metadata
        .get("mime_type")
        .and_then(Value::as_str)
        .unwrap_or("text/plain")
        .to_string();
    Part::Media(MediaPart::new(Modality::Binary, mime, resource.data))
}

fn call_tool_result_to_tool_output(result: CallToolResult) -> ToolOutput {
    if let Some(structured) = result.structured_content {
        return ToolOutput::Structured(structured);
    }
    let parts = call_tool_content_to_parts(result.content);
    if parts.iter().all(|part| matches!(part, Part::Text(_))) {
        let text = parts
            .iter()
            .filter_map(|part| match part {
                Part::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        ToolOutput::Text(text)
    } else {
        ToolOutput::Parts(parts)
    }
}

fn call_tool_content_to_parts(contents: Vec<Content>) -> Vec<Part> {
    contents.into_iter().map(content_to_part).collect()
}

fn content_to_part(content: Content) -> Part {
    match content.raw {
        RawContent::Text(text) => Part::Text(TextPart::new(text.text)),
        RawContent::Image(image) => Part::Media(MediaPart::new(
            Modality::Image,
            image.mime_type,
            DataRef::InlineText(image.data),
        )),
        RawContent::Audio(audio) => Part::Media(MediaPart::new(
            Modality::Audio,
            audio.mime_type,
            DataRef::InlineText(audio.data),
        )),
        RawContent::Resource(embedded) => {
            agentkit_part_from_resource(resource_contents_to_capabilities(embedded.resource))
        }
        RawContent::ResourceLink(link) => Part::Text(TextPart::new(link.uri)),
    }
}

fn value_to_json_object(value: Value, context: &str) -> Result<rmcp_model::JsonObject, McpError> {
    match value {
        Value::Object(object) => Ok(object),
        Value::Null => Ok(serde_json::Map::new()),
        other => Err(McpError::Protocol(format!(
            "{context} must be a JSON object, got {other}"
        ))),
    }
}

fn bearer_token_from_metadata(metadata: &MetadataMap) -> Option<String> {
    ["bearer_token", "access_token", "token", "api_key"]
        .into_iter()
        .find_map(|key| metadata.get(key).and_then(Value::as_str).map(str::to_owned))
}

fn rmcp_initialize_error(
    config: &McpServerConfig,
    error: rmcp::service::ClientInitializeError,
) -> McpError {
    let message = error.to_string();
    if let Some(auth_request) =
        parse_transport_auth_request(&config.id, "initialize", &json!({}), &message)
    {
        return McpError::AuthRequired(Box::new(auth_request));
    }
    McpError::Transport(message)
}

fn rmcp_service_error(error: rmcp::ServiceError) -> McpError {
    McpError::Invocation(error.to_string())
}

fn rmcp_operation_error(
    server_id: &McpServerId,
    method: &str,
    params: Value,
    error: rmcp::ServiceError,
) -> McpError {
    let message = error.to_string();
    if let Some(auth_request) = parse_transport_auth_request(server_id, method, &params, &message) {
        return McpError::AuthRequired(Box::new(auth_request));
    }
    McpError::Invocation(message)
}

fn parse_transport_auth_request(
    server_id: &McpServerId,
    method: &str,
    params: &Value,
    message: &str,
) -> Option<AuthRequest> {
    let lower = message.to_ascii_lowercase();
    if !(lower.contains("auth required")
        || lower.contains("unauthorized")
        || lower.contains("insufficient scope")
        || lower.contains("insufficient_scope"))
    {
        return None;
    }

    let mut challenge = MetadataMap::new();
    challenge.insert("server_id".into(), Value::String(server_id.to_string()));
    challenge.insert("method".into(), Value::String(method.into()));
    challenge.insert("message".into(), Value::String(message.into()));
    challenge.insert("flow_kind".into(), Value::String("http_bearer".into()));
    if lower.contains("insufficient scope") || lower.contains("insufficient_scope") {
        challenge.insert("insufficient_scope".into(), Value::Bool(true));
    }

    Some(AuthRequest {
        task_id: None,
        id: format!("mcp:{}:{}", server_id, method),
        provider: format!("mcp.{}", server_id),
        operation: auth_operation_for_method(server_id, method, params),
        challenge,
    })
}

fn auth_operation_for_method(
    server_id: &McpServerId,
    method: &str,
    params: &Value,
) -> AuthOperation {
    match method {
        "initialize" => AuthOperation::McpConnect {
            server_id: server_id.to_string(),
            metadata: MetadataMap::new(),
        },
        "tools/call" => AuthOperation::McpToolCall {
            server_id: server_id.to_string(),
            tool_name: params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            input: params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({})),
            metadata: MetadataMap::new(),
        },
        "resources/read" => AuthOperation::McpResourceRead {
            server_id: server_id.to_string(),
            resource_id: params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            metadata: MetadataMap::new(),
        },
        "prompts/get" => AuthOperation::McpPromptGet {
            server_id: server_id.to_string(),
            prompt_id: params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            args: params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({})),
            metadata: MetadataMap::new(),
        },
        other => AuthOperation::Custom {
            kind: format!("mcp.{other}"),
            payload: params.clone(),
            metadata: {
                let mut metadata = MetadataMap::new();
                metadata.insert("server_id".into(), Value::String(server_id.to_string()));
                metadata
            },
        },
    }
}

fn normalize_mcp_tool_name(server_id: &McpServerId, tool_name: &str) -> String {
    let prefix = format!("mcp_{server_id}_");
    tool_name
        .strip_prefix(&prefix)
        .unwrap_or(tool_name)
        .to_string()
}

/// Errors produced by MCP transport, protocol, and lifecycle operations.
#[derive(Debug, Error)]
pub enum McpError {
    /// An underlying I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A JSON serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    /// A transport-level error.
    #[error("transport error: {0}")]
    Transport(String),
    /// An MCP protocol violation.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// The server requires authentication before the operation can proceed.
    #[error("MCP auth required: {0:?}")]
    AuthRequired(Box<AuthRequest>),
    /// An error occurred while resolving or replaying authentication.
    #[error("auth resolution error: {0}")]
    AuthResolution(String),
    /// The MCP server returned an error for the invoked method.
    #[error("invocation error: {0}")]
    Invocation(String),
    /// The referenced server ID is not registered in the [`McpServerManager`].
    #[error("unknown MCP server: {0}")]
    UnknownServer(String),
}

impl From<&str> for McpServerId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for McpServerId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}
