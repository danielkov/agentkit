use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, RwLock};
use std::time::Duration;

use agentkit_capabilities::{
    CapabilityContext, CapabilityError, CapabilityName, CapabilityProvider, Invocable,
    InvocableOutput, InvocableRequest, InvocableResult, InvocableSpec, PromptContents,
    PromptDescriptor, PromptId, PromptProvider, ResourceContents, ResourceDescriptor, ResourceId,
    ResourceProvider,
};
use agentkit_core::{
    DataRef, Item, ItemKind, MetadataMap, Part, TextPart, ToolOutput, ToolResultPart,
};
use agentkit_http::{
    HeaderMap, Http, HttpError, HttpRequestBuilder, HttpResponse, StatusCode, header as http_header,
};
use agentkit_tools_core::{
    AuthOperation, AuthRequest, AuthResolution, Tool, ToolAnnotations, ToolCatalogEvent,
    ToolContext, ToolError, ToolExecutionOutcome, ToolExecutor, ToolName, ToolRegistry,
    ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use rmcp::ServiceExt;
use rmcp::handler::client::ClientHandler;
use rmcp::model as rmcp_model;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig as RmcpStreamableHttpClientTransportConfig;
use rmcp::transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_util::io::StreamReader;
use url::Url;

const MCP_LATEST_PROTOCOL_VERSION: &str = "2025-11-25";
const MCP_SUPPORTED_PROTOCOL_VERSIONS: &[&str] =
    &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

/// Unique identifier for a registered MCP server.
///
/// Each MCP server in a [`McpServerManager`] is addressed by its `McpServerId`.
/// The inner string is typically a short, human-readable name such as `"filesystem"`
/// or `"github"`.
///
/// # Example
///
/// ```rust
/// use agentkit_mcp::McpServerId;
///
/// let id = McpServerId::new("filesystem");
/// assert_eq!(id.to_string(), "filesystem");
/// ```
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

/// Configuration for an MCP server that communicates over standard I/O (stdin/stdout).
///
/// This is the most common transport for local MCP servers. The specified command is
/// spawned as a child process, and JSON-RPC messages are exchanged line-by-line over
/// its stdin and stdout streams.
///
/// # Example
///
/// ```rust
/// use agentkit_mcp::StdioTransportConfig;
///
/// let config = StdioTransportConfig::new("npx")
///     .with_arg("-y")
///     .with_arg("@modelcontextprotocol/server-filesystem")
///     .with_env("HOME", "/home/user")
///     .with_cwd("/tmp");
/// ```
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

/// Configuration for an MCP server that communicates over Server-Sent Events (SSE).
///
/// Use this transport for remote MCP servers exposed over HTTP. The client opens an
/// SSE stream to the given URL, receives an `endpoint` event pointing to the POST
/// endpoint, and then exchanges JSON-RPC messages over that endpoint.
///
/// Auth headers and other per-request customisation live on the [`Http`] client
/// — either via `reqwest::ClientBuilder::default_headers` or a custom
/// [`agentkit_http::HttpClient`] implementation — passed in through
/// [`with_client`](Self::with_client).
#[derive(Clone, Debug)]
pub struct SseTransportConfig {
    /// The SSE endpoint URL to connect to.
    pub url: String,
    /// HTTP client used for all requests on this transport. When `None`, the
    /// factory builds a default reqwest-backed client with agentkit's user agent.
    pub client: Option<Http>,
}

impl SseTransportConfig {
    /// Creates a new SSE transport configuration for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: None,
        }
    }

    /// Supplies a pre-configured HTTP client. Use this to attach auth headers
    /// (via `reqwest::ClientBuilder::default_headers`), install retry/tracing
    /// middleware, or plug in a non-reqwest backend.
    pub fn with_client(mut self, client: Http) -> Self {
        self.client = Some(client);
        self
    }
}

/// Configuration for an MCP server that communicates over Streamable HTTP.
///
/// Use this transport for modern remote MCP servers that expose a single HTTP
/// endpoint supporting JSON-RPC over POST, with optional SSE responses for
/// streaming server messages.
///
/// Auth headers and other per-request customisation live on the [`Http`] client
/// passed in via [`with_client`](Self::with_client).
#[derive(Clone, Debug)]
pub struct StreamableHttpTransportConfig {
    /// The MCP endpoint URL to connect to.
    pub url: String,
    /// HTTP client used for all requests on this transport. When `None`, the
    /// factory builds a default reqwest-backed client with agentkit's user agent.
    pub client: Option<Http>,
    /// Static bearer token sent as an HTTP `Authorization: Bearer ...` header.
    pub bearer_token: Option<String>,
    /// Static custom HTTP headers sent with every Streamable HTTP request.
    pub headers: HeaderMap,
}

impl StreamableHttpTransportConfig {
    /// Creates a new Streamable HTTP transport configuration for the given MCP endpoint.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: None,
            bearer_token: None,
            headers: HeaderMap::new(),
        }
    }

    /// Supplies a pre-configured HTTP client. See
    /// [`SseTransportConfig::with_client`] for the typical use cases.
    pub fn with_client(mut self, client: Http) -> Self {
        self.client = Some(client);
        self
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
        N: TryInto<agentkit_http::HeaderName>,
        N::Error: std::fmt::Display,
        V: TryInto<agentkit_http::HeaderValue>,
        V::Error: std::fmt::Display,
    {
        let name = name
            .try_into()
            .map_err(|error| McpError::Transport(format!("invalid HTTP header name: {error}")))?;
        let value = value
            .try_into()
            .map_err(|error| McpError::Transport(format!("invalid HTTP header value: {error}")))?;
        self.headers.insert(name, value);
        Ok(self)
    }
}

/// Selects which transport an MCP server should use.
///
/// This enum is passed into [`McpServerConfig`] and determines how the client will
/// communicate with the MCP server. The built-in options are [`Stdio`](Self::Stdio),
/// [`StreamableHttp`](Self::StreamableHttp), and the legacy [`Sse`](Self::Sse);
/// use [`Custom`](Self::Custom) to provide your own [`McpTransportFactory`].
#[derive(Clone)]
pub enum McpTransportBinding {
    /// Communicate over the child process's stdin/stdout.
    Stdio(StdioTransportConfig),
    /// Communicate over the MCP Streamable HTTP transport.
    StreamableHttp(StreamableHttpTransportConfig),
    /// Communicate over HTTP Server-Sent Events.
    Sse(SseTransportConfig),
    /// A user-supplied transport factory.
    Custom(Arc<dyn McpTransportFactory>),
}

/// Full configuration for a single MCP server, combining an identifier, a transport
/// binding, and optional metadata.
///
/// Register one or more of these with [`McpServerManager`] to manage the lifecycle
/// of MCP servers in an agentkit runtime.
///
/// # Example
///
/// ```rust
/// use agentkit_mcp::{McpServerConfig, McpTransportBinding, StdioTransportConfig};
///
/// let config = McpServerConfig::new(
///     "filesystem",
///     McpTransportBinding::Stdio(
///         StdioTransportConfig::new("npx")
///             .with_arg("-y")
///             .with_arg("@modelcontextprotocol/server-filesystem"),
///     ),
/// );
/// ```
#[derive(Clone)]
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
    ///
    /// # Arguments
    ///
    /// * `id` - A unique name for this server (e.g. `"filesystem"`).
    /// * `transport` - The [`McpTransportBinding`] that determines how to connect.
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

    /// Creates an SSE-backed server configuration.
    pub fn sse(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(id, McpTransportBinding::Sse(SseTransportConfig::new(url)))
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

/// A single JSON-RPC frame exchanged with an MCP server.
///
/// This is the low-level wire unit. Most users will not interact with `McpFrame`
/// directly; instead use [`McpConnection`] or the higher-level adapters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpFrame {
    /// The raw JSON-RPC value (request, response, or notification).
    pub value: Value,
}

/// Factory trait for creating new [`McpTransport`] connections.
///
/// Implement this trait to provide a custom transport mechanism. The built-in
/// [`StdioTransportFactory`] and [`SseTransportFactory`] cover the two standard
/// MCP transports; use this trait for in-memory, WebSocket, or other custom
/// transports.
///
/// # Errors
///
/// Returns [`McpError`] if the connection cannot be established.
#[async_trait]
pub trait McpTransportFactory: Send + Sync {
    /// Establishes a new transport connection and returns it.
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError>;
}

/// Bidirectional transport for exchanging [`McpFrame`] messages with an MCP server.
///
/// Implement this trait to provide a custom transport. Each transport instance
/// represents a single, live connection.
///
/// # Errors
///
/// All methods return [`McpError`] on I/O or protocol failures.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Sends a JSON-RPC frame to the server.
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError>;
    /// Receives the next JSON-RPC frame from the server, or `None` if the stream has ended.
    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError>;
    /// Closes the transport, releasing any underlying resources.
    async fn close(&mut self) -> Result<(), McpError>;
}

/// Factory that spawns a child process and connects via stdin/stdout.
///
/// Created from a [`StdioTransportConfig`]. Each call to
/// [`connect`](McpTransportFactory::connect) spawns a new child process.
pub struct StdioTransportFactory {
    config: StdioTransportConfig,
}

impl StdioTransportFactory {
    /// Creates a new factory from the given stdio transport configuration.
    pub fn new(config: StdioTransportConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl McpTransportFactory for StdioTransportFactory {
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError> {
        let mut command = Command::new(&self.config.command);
        command.args(&self.config.args);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::inherit());

        if let Some(cwd) = &self.config.cwd {
            command.current_dir(cwd);
        }

        for (key, value) in &self.config.env {
            command.env(key, value);
        }

        let mut child = command.spawn().map_err(McpError::Io)?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("failed to capture MCP stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("failed to capture MCP stdout".into()))?;

        Ok(Box::new(StdioTransport {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        }))
    }
}

/// Factory that opens an HTTP SSE stream and connects via Server-Sent Events.
///
/// Created from an [`SseTransportConfig`]. Each call to
/// [`connect`](McpTransportFactory::connect) opens a new HTTP connection.
pub struct SseTransportFactory {
    config: SseTransportConfig,
}

impl SseTransportFactory {
    /// Creates a new factory from the given SSE transport configuration.
    pub fn new(config: SseTransportConfig) -> Self {
        Self { config }
    }
}

/// Factory that connects to a Streamable HTTP MCP endpoint.
///
/// Created from a [`StreamableHttpTransportConfig`]. Each call to
/// [`connect`](McpTransportFactory::connect) creates a new HTTP-backed MCP session.
pub struct StreamableHttpTransportFactory {
    config: StreamableHttpTransportConfig,
}

impl StreamableHttpTransportFactory {
    /// Creates a new factory from the given Streamable HTTP transport configuration.
    pub fn new(config: StreamableHttpTransportConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl McpTransportFactory for SseTransportFactory {
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError> {
        let client = resolve_http_client(self.config.client.as_ref())?;

        let response = client
            .get(self.config.url.as_str())
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable response body>".into());
            return Err(McpError::Transport(format!(
                "SSE connection failed with status {status}: {body}"
            )));
        }

        let response_url = Url::parse(response.url())
            .map_err(|error| McpError::Transport(format!("invalid SSE response URL: {error}")))?;
        let stream = response.bytes_stream().map_err(std::io::Error::other);
        let reader = BufReader::new(StreamReader::new(stream));
        let (frame_tx, frame_rx) = mpsc::unbounded_channel();
        let (endpoint_tx, endpoint_rx) = oneshot::channel();
        let read_task = tokio::spawn(read_sse_stream(reader, response_url, frame_tx, endpoint_tx));

        let endpoint_url = endpoint_rx
            .await
            .map_err(|_| McpError::Transport("SSE stream closed before endpoint event".into()))??;

        Ok(Box::new(SseTransport {
            client,
            endpoint_url,
            frame_rx,
            read_task,
        }))
    }
}

#[async_trait]
impl McpTransportFactory for StreamableHttpTransportFactory {
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError> {
        let client = resolve_http_client(self.config.client.as_ref())?;

        let endpoint_url = Url::parse(&self.config.url)
            .map_err(|error| McpError::Transport(format!("invalid MCP endpoint URL: {error}")))?;

        Ok(Box::new(StreamableHttpTransport {
            client,
            endpoint_url,
            protocol_version: None,
            session_id: None,
            pending_frames: VecDeque::new(),
        }))
    }
}

fn resolve_http_client(configured: Option<&Http>) -> Result<Http, McpError> {
    if let Some(client) = configured {
        return Ok(client.clone());
    }
    #[cfg(feature = "reqwest-client")]
    {
        reqwest::Client::builder()
            .user_agent(concat!("agentkit-mcp/", env!("CARGO_PKG_VERSION")))
            .build()
            .map(Http::new)
            .map_err(|error| McpError::Http(HttpError::request(error)))
    }
    #[cfg(not(feature = "reqwest-client"))]
    {
        Err(McpError::Transport(
            "no HTTP client configured; enable the `reqwest-client` feature or supply one via `with_client`".into(),
        ))
    }
}

struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct SseTransport {
    client: Http,
    endpoint_url: Url,
    frame_rx: mpsc::UnboundedReceiver<Result<McpFrame, McpError>>,
    read_task: JoinHandle<()>,
}

struct StreamableHttpTransport {
    client: Http,
    endpoint_url: Url,
    protocol_version: Option<String>,
    session_id: Option<String>,
    pending_frames: VecDeque<McpFrame>,
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError> {
        let mut encoded = serde_json::to_vec(&message.value).map_err(McpError::Serialize)?;
        encoded.push(b'\n');
        self.stdin.write_all(&encoded).await.map_err(McpError::Io)?;
        self.stdin.flush().await.map_err(McpError::Io)?;
        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError> {
        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .await
            .map_err(McpError::Io)?;
        if read == 0 {
            return Ok(None);
        }

        let value = serde_json::from_str(line.trim()).map_err(McpError::Serialize)?;
        Ok(Some(McpFrame { value }))
    }

    async fn close(&mut self) -> Result<(), McpError> {
        let _ = self.stdin.shutdown().await;
        let _ = self.child.kill().await;
        Ok(())
    }
}

#[async_trait]
impl McpTransport for SseTransport {
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError> {
        let response = self
            .client
            .post(self.endpoint_url.as_str())
            .header("Content-Type", "application/json")
            .json(&message.value)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable response body>".into());
            return Err(McpError::Transport(format!(
                "SSE POST failed with status {status}: {body}"
            )));
        }

        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError> {
        match self.frame_rx.recv().await {
            Some(Ok(frame)) => Ok(Some(frame)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> Result<(), McpError> {
        self.read_task.abort();
        Ok(())
    }
}

#[async_trait]
impl McpTransport for StreamableHttpTransport {
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError> {
        let is_request = is_jsonrpc_request(&message.value);
        let request_id = message.value.get("id").cloned();
        let is_initialize =
            message.value.get("method").and_then(Value::as_str) == Some("initialize");

        let mut request = self
            .client
            .post(self.endpoint_url.as_str())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        request = apply_streamable_http_headers(
            request,
            self.protocol_version.as_deref(),
            self.session_id.as_deref(),
        );

        let response = request.json(&message.value).send().await?;

        if is_initialize {
            self.capture_session_id(response.headers());
        }

        let status = response.status();
        if !status.is_success() {
            return Err(
                streamable_http_status_error("Streamable HTTP POST", status, response).await,
            );
        }

        if !is_request {
            return Ok(());
        }

        let content_type = response
            .headers()
            .get(http_header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();

        if content_type.starts_with("application/json") {
            let value: Value = response.json().await?;
            self.maybe_update_protocol_version(&message.value, &value)?;
            self.pending_frames.push_back(McpFrame { value });
            return Ok(());
        }

        if !content_type.starts_with("text/event-stream") {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable response body>".into());
            return Err(McpError::Transport(format!(
                "unexpected Streamable HTTP response content type {content_type:?}: {body}"
            )));
        }

        let request_id = request_id.ok_or_else(|| {
            McpError::Protocol("JSON-RPC request over Streamable HTTP is missing an id".into())
        })?;
        self.collect_streamable_http_response(response, &message.value, &request_id)
            .await
    }

    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError> {
        Ok(self.pending_frames.pop_front())
    }

    async fn close(&mut self) -> Result<(), McpError> {
        let Some(session_id) = self.session_id.clone() else {
            return Ok(());
        };

        let mut request = self.client.delete(self.endpoint_url.as_str());
        request = apply_streamable_http_headers(
            request,
            self.protocol_version.as_deref(),
            Some(session_id.as_str()),
        );

        let response = request.send().await?;
        if response.status().is_success()
            || response.status() == StatusCode::METHOD_NOT_ALLOWED
            || response.status() == StatusCode::NOT_FOUND
        {
            self.session_id = None;
            return Ok(());
        }

        Err(
            streamable_http_status_error("Streamable HTTP DELETE", response.status(), response)
                .await,
        )
    }
}

impl StreamableHttpTransport {
    async fn collect_streamable_http_response(
        &mut self,
        response: HttpResponse,
        request_message: &Value,
        request_id: &Value,
    ) -> Result<(), McpError> {
        let mut retry_delay = Duration::from_millis(0);
        let mut last_event_id = None;
        let mut saw_response = false;

        saw_response |= self
            .read_streamable_http_events(
                response,
                request_message,
                request_id,
                &mut last_event_id,
                &mut retry_delay,
            )
            .await?;

        while !saw_response && last_event_id.is_some() {
            if !retry_delay.is_zero() {
                sleep(retry_delay).await;
            }

            let response = self
                .resume_streamable_http_stream(last_event_id.as_deref().unwrap())
                .await?;
            saw_response |= self
                .read_streamable_http_events(
                    response,
                    request_message,
                    request_id,
                    &mut last_event_id,
                    &mut retry_delay,
                )
                .await?;
        }

        Ok(())
    }

    async fn read_streamable_http_events(
        &mut self,
        response: HttpResponse,
        request_message: &Value,
        request_id: &Value,
        last_event_id: &mut Option<String>,
        retry_delay: &mut Duration,
    ) -> Result<bool, McpError> {
        let stream = response.bytes_stream().map_err(std::io::Error::other);
        let mut reader = BufReader::new(StreamReader::new(stream));
        let mut saw_response = false;

        while let Some(event) = read_next_sse_event(&mut reader).await? {
            if let Some(id) = event.id.clone() {
                *last_event_id = Some(id);
            }
            if let Some(retry_ms) = event.retry_ms {
                *retry_delay = Duration::from_millis(retry_ms);
            }

            let Some(frame) = streamable_http_event_to_frame(event)? else {
                continue;
            };

            self.maybe_update_protocol_version(request_message, &frame.value)?;
            if frame.value.get("id") == Some(request_id) {
                saw_response = true;
            }
            self.pending_frames.push_back(frame);
        }

        Ok(saw_response)
    }

    async fn resume_streamable_http_stream(
        &self,
        last_event_id: &str,
    ) -> Result<HttpResponse, McpError> {
        let mut request = self
            .client
            .get(self.endpoint_url.as_str())
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Last-Event-ID", last_event_id);

        request = apply_streamable_http_headers(
            request,
            self.protocol_version.as_deref(),
            self.session_id.as_deref(),
        );

        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(
                streamable_http_status_error("Streamable HTTP GET", status, response).await,
            );
        }

        let content_type = response
            .headers()
            .get(http_header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !content_type.starts_with("text/event-stream") {
            let content_type = content_type.to_string();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable response body>".into());
            return Err(McpError::Transport(format!(
                "Streamable HTTP GET expected text/event-stream, got {content_type:?}: {body}"
            )));
        }

        Ok(response)
    }

    fn maybe_update_protocol_version(
        &mut self,
        request_message: &Value,
        response_value: &Value,
    ) -> Result<(), McpError> {
        if request_message.get("method").and_then(Value::as_str) != Some("initialize") {
            return Ok(());
        }

        let protocol_version = response_value
            .get("result")
            .and_then(|result| result.get("protocolVersion"))
            .and_then(Value::as_str);

        if let Some(protocol_version) = protocol_version {
            self.protocol_version = Some(protocol_version.to_string());
        }

        Ok(())
    }

    fn capture_session_id(&mut self, headers: &HeaderMap) {
        self.session_id = headers
            .get("MCP-Session-Id")
            .and_then(|value| value.to_str().ok())
            .map(|value| value.to_string());
    }
}

/// Descriptor for a tool advertised by an MCP server.
///
/// Returned as part of a [`McpDiscoverySnapshot`] after server discovery. The
/// [`input_schema`](Self::input_schema) field is the JSON Schema that describes
/// the tool's expected input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    /// The tool name as reported by the MCP server.
    pub name: String,
    /// Optional human-readable description of the tool.
    pub description: Option<String>,
    /// JSON Schema describing the tool's input parameters.
    pub input_schema: Value,
    /// Arbitrary metadata attached to this descriptor.
    pub metadata: MetadataMap,
}

/// Descriptor for a resource advertised by an MCP server.
///
/// Resources represent data that the server can provide (e.g. files, database
/// records). Each resource is identified by a URI.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceDescriptor {
    /// The resource URI (e.g. `"file:///tmp/example.txt"`).
    pub id: String,
    /// Human-readable name of the resource.
    pub name: String,
    /// Optional description of the resource.
    pub description: Option<String>,
    /// Optional MIME type (e.g. `"text/plain"`, `"application/json"`).
    pub mime_type: Option<String>,
    /// Arbitrary metadata attached to this descriptor.
    pub metadata: MetadataMap,
}

/// Descriptor for a prompt template advertised by an MCP server.
///
/// Prompts are reusable message templates that can be parameterized with arguments.
/// The [`input_schema`](Self::input_schema) describes the expected arguments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpPromptDescriptor {
    /// Unique identifier for the prompt (typically the same as `name`).
    pub id: String,
    /// Human-readable name of the prompt.
    pub name: String,
    /// Optional description of what the prompt does.
    pub description: Option<String>,
    /// JSON Schema describing the prompt's input arguments.
    pub input_schema: Value,
    /// Arbitrary metadata attached to this descriptor.
    pub metadata: MetadataMap,
}

/// A snapshot of all capabilities discovered from a single MCP server.
///
/// Obtained by calling [`McpConnection::discover`] or as part of a
/// [`McpServerHandle`]. Contains the full list of tools, resources, and prompts
/// that the server advertised at discovery time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpDiscoverySnapshot {
    /// The server this snapshot was taken from.
    pub server_id: McpServerId,
    /// Tools advertised by the server.
    pub tools: Vec<McpToolDescriptor>,
    /// Resources advertised by the server.
    pub resources: Vec<McpResourceDescriptor>,
    /// Prompts advertised by the server.
    pub prompts: Vec<McpPromptDescriptor>,
    /// Arbitrary metadata attached to this snapshot.
    pub metadata: MetadataMap,
}

/// Catalog and lifecycle events emitted by [`McpServerManager`].
///
/// Hosts can subscribe to these events and forward tool-catalog changes to
/// `agentkit-loop` as next-turn model capability updates.
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

#[derive(Clone)]
struct AgentkitRmcpClientHandler {
    info: rmcp_model::ClientInfo,
    notifications: mpsc::UnboundedSender<RmcpServerNotification>,
}

#[derive(Clone, Debug)]
enum RmcpServerNotification {
    ToolsChanged,
    ResourcesChanged,
    PromptsChanged,
}

impl AgentkitRmcpClientHandler {
    fn new(notifications: mpsc::UnboundedSender<RmcpServerNotification>) -> Self {
        Self {
            info: rmcp_model::ClientInfo::new(
                rmcp_model::ClientCapabilities::default(),
                rmcp_model::Implementation::new("agentkit-mcp", env!("CARGO_PKG_VERSION"))
                    .with_title("agentkit MCP client"),
            )
            .with_protocol_version(rmcp_model::ProtocolVersion::LATEST),
            notifications,
        }
    }
}

impl ClientHandler for AgentkitRmcpClientHandler {
    fn on_tool_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self
            .notifications
            .send(RmcpServerNotification::ToolsChanged);
        std::future::ready(())
    }

    fn on_resource_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self
            .notifications
            .send(RmcpServerNotification::ResourcesChanged);
        std::future::ready(())
    }

    fn on_prompt_list_changed(
        &self,
        _context: rmcp::service::NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + rmcp::service::MaybeSendFuture + '_ {
        let _ = self
            .notifications
            .send(RmcpServerNotification::PromptsChanged);
        std::future::ready(())
    }

    fn get_info(&self) -> rmcp_model::ClientInfo {
        self.info.clone()
    }
}

type RmcpClientService = RunningService<RoleClient, AgentkitRmcpClientHandler>;

enum McpConnectionInner {
    Rmcp(RmcpClientService),
    Legacy(Box<dyn McpTransport>),
}

/// A live connection to a single MCP server.
///
/// Wraps an RMCP client service for built-in transports and exposes
/// high-level methods for tool calls, resource reads, prompt retrieval, and
/// capability discovery. Custom transports use the compatibility JSON-RPC
/// path.
///
/// Create a connection with [`McpConnection::connect`] or indirectly through
/// [`McpServerManager::connect_server`].
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_mcp::{McpConnection, McpServerConfig, McpTransportBinding, StdioTransportConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = McpServerConfig::new(
///     "filesystem",
///     McpTransportBinding::Stdio(StdioTransportConfig::new("npx")
///         .with_arg("-y")
///         .with_arg("@modelcontextprotocol/server-filesystem")),
/// );
///
/// let connection = McpConnection::connect(&config).await?;
/// let snapshot = connection.discover().await?;
/// println!("found {} tools", snapshot.tools.len());
/// # Ok(())
/// # }
/// ```
pub struct McpConnection {
    server_id: McpServerId,
    config: McpServerConfig,
    inner: Mutex<McpConnectionInner>,
    auth: Mutex<Option<MetadataMap>>,
    notifications: Mutex<mpsc::UnboundedReceiver<RmcpServerNotification>>,
    next_id: AtomicU64,
    capabilities: McpServerCapabilities,
}

/// Capabilities advertised by an MCP server during the `initialize` handshake.
///
/// Per the MCP specification, a client must only call `<capability>/list`
/// (and related) methods for capabilities the server has advertised. Calling
/// an unadvertised method generally results in a `-32601 Method not found`
/// JSON-RPC error.
///
/// Each capability is `Some(_)` when the server advertised the top-level key
/// (regardless of sub-field contents) and `None` otherwise. Sub-structs carry
/// the optional feature flags from the spec (`listChanged`, `subscribe`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    /// Advertised `tools` capability → `tools/list` and `tools/call` supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Advertised `resources` capability → `resources/list` and `resources/read`
    /// supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Advertised `prompts` capability → `prompts/list` and `prompts/get`
    /// supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Advertised `logging` capability — currently informational; no discovery
    /// call is made for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

/// Tools sub-capability flags from the MCP `initialize` response.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// Server will emit `notifications/tools/list_changed` when the tool
    /// catalog changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resources sub-capability flags from the MCP `initialize` response.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    /// Server supports `resources/subscribe` for change notifications on
    /// individual resources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// Server will emit `notifications/resources/list_changed` when the
    /// resource catalog changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Prompts sub-capability flags from the MCP `initialize` response.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    /// Server will emit `notifications/prompts/list_changed` when the prompt
    /// catalog changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Logging sub-capability. Spec reserves the key with no defined sub-fields
/// yet; kept as a unit struct for forward-compat.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggingCapability {}

impl McpServerCapabilities {
    /// Build capabilities from an MCP `initialize` response's `capabilities`
    /// object. Absence of a key means the capability is not supported.
    /// Malformed sub-structures degrade to `Some(Default::default())` rather
    /// than erroring — the presence of the key is the load-bearing signal.
    pub fn from_initialize_value(value: Option<&Value>) -> Self {
        let Some(obj) = value.and_then(Value::as_object) else {
            return Self::default();
        };
        fn lift<T: Default + for<'de> Deserialize<'de>>(v: Option<&Value>) -> Option<T> {
            v.map(|value| serde_json::from_value(value.clone()).unwrap_or_default())
        }
        Self {
            tools: lift(obj.get("tools")),
            resources: lift(obj.get("resources")),
            prompts: lift(obj.get("prompts")),
            logging: lift(obj.get("logging")),
        }
    }

    /// Return a capabilities struct with every top-level capability advertised
    /// (sub-flags left at their defaults). Useful for tests and for callers
    /// that want to attempt discovery regardless of what the server
    /// advertised.
    pub fn all() -> Self {
        Self {
            tools: Some(ToolsCapability::default()),
            resources: Some(ResourcesCapability::default()),
            prompts: Some(PromptsCapability::default()),
            logging: Some(LoggingCapability::default()),
        }
    }
}

/// Wire shape of an MCP `initialize` response's `result` object. Used
/// internally by [`McpConnection::connect`] to validate the protocol version
/// and extract advertised capabilities in one typed step.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResult {
    protocol_version: String,
    #[serde(default)]
    capabilities: McpServerCapabilities,
}

/// Minimal envelope used to decode list-style responses (`tools/list`,
/// `resources/list`, `prompts/list`). Each MCP list method wraps its payload
/// in a single-field object.
#[derive(Debug, Deserialize)]
struct ToolsListPayload {
    #[serde(default)]
    tools: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct ResourcesListPayload {
    #[serde(default)]
    resources: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct PromptsListPayload {
    #[serde(default)]
    prompts: Vec<Value>,
}

/// Wire shape of a `resources/read` response. Servers return one or more
/// content blocks; agentkit currently consumes the first one and discriminates
/// on `text` vs. `uri`.
#[derive(Debug, Deserialize)]
struct ResourcesReadPayload {
    #[serde(default)]
    contents: Vec<ResourceContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ResourceContentBlock {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    uri: Option<String>,
}

/// Wire shape of a `prompts/get` response.
#[derive(Debug, Deserialize)]
struct PromptsGetPayload {
    #[serde(default)]
    messages: Vec<Value>,
}

/// The result of replaying an MCP operation after auth resolution.
///
/// Returned by [`McpConnection::replay_auth_operation`] and
/// [`McpServerManager::resolve_auth_and_resume`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum McpOperationResult {
    /// The server was successfully (re)connected; contains the discovery snapshot.
    Connected(McpDiscoverySnapshot),
    /// A tool call completed; contains the raw JSON result.
    Tool(Value),
    /// A resource was read successfully.
    Resource(ResourceContents),
    /// A prompt was retrieved successfully.
    Prompt(PromptContents),
}

impl McpConnection {
    /// Connects to an MCP server, performs the JSON-RPC `initialize` handshake, and
    /// returns a ready-to-use connection.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the transport fails to connect, the handshake is
    /// rejected, or the server requires authentication ([`McpError::AuthRequired`]).
    pub async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        Self::connect_with_auth(config, None).await
    }

    async fn connect_with_auth(
        config: &McpServerConfig,
        auth: Option<&MetadataMap>,
    ) -> Result<Self, McpError> {
        let (notification_tx, notification_rx) = mpsc::unbounded_channel();
        let (inner, capabilities) = match &config.transport {
            McpTransportBinding::Stdio(binding) => {
                connect_rmcp_stdio(config, binding, notification_tx).await?
            }
            McpTransportBinding::StreamableHttp(binding) => {
                connect_rmcp_streamable_http(config, binding, auth, notification_tx).await?
            }
            McpTransportBinding::Sse(binding) => {
                #[cfg(feature = "legacy-sse")]
                {
                    connect_legacy_transport(
                        config,
                        Arc::new(SseTransportFactory::new(binding.clone())),
                        auth,
                    )
                    .await?
                }
                #[cfg(not(feature = "legacy-sse"))]
                {
                    let _ = binding;
                    return Err(McpError::Transport(
                        "legacy HTTP+SSE MCP transport is disabled; enable the `legacy-sse` feature or use Streamable HTTP".into(),
                    ));
                }
            }
            McpTransportBinding::Custom(factory) => {
                connect_legacy_transport(config, factory.clone(), auth).await?
            }
        };

        Ok(Self {
            server_id: config.id.clone(),
            config: config.clone(),
            inner: Mutex::new(inner),
            auth: Mutex::new(auth.cloned()),
            notifications: Mutex::new(notification_rx),
            next_id: AtomicU64::new(1),
            capabilities,
        })
    }

    async fn reconnect_rmcp_inner(
        &self,
        auth: Option<&MetadataMap>,
    ) -> Result<McpConnectionInner, McpError> {
        let (notification_tx, notification_rx) = mpsc::unbounded_channel();
        let (inner, _capabilities) = match &self.config.transport {
            McpTransportBinding::Stdio(binding) => {
                connect_rmcp_stdio(&self.config, binding, notification_tx).await?
            }
            McpTransportBinding::StreamableHttp(binding) => {
                connect_rmcp_streamable_http(&self.config, binding, auth, notification_tx).await?
            }
            McpTransportBinding::Sse(binding) => {
                #[cfg(feature = "legacy-sse")]
                {
                    connect_legacy_transport(
                        &self.config,
                        Arc::new(SseTransportFactory::new(binding.clone())),
                        auth,
                    )
                    .await?
                }
                #[cfg(not(feature = "legacy-sse"))]
                {
                    let _ = binding;
                    return Err(McpError::Transport(
                        "legacy HTTP+SSE MCP transport is disabled; enable the `legacy-sse` feature or use Streamable HTTP".into(),
                    ));
                }
            }
            McpTransportBinding::Custom(factory) => {
                connect_legacy_transport(&self.config, factory.clone(), auth).await?
            }
        };
        *self.notifications.lock().await = notification_rx;
        Ok(inner)
    }
}

async fn connect_legacy_transport(
    config: &McpServerConfig,
    factory: Arc<dyn McpTransportFactory>,
    auth: Option<&MetadataMap>,
) -> Result<(McpConnectionInner, McpServerCapabilities), McpError> {
    let mut transport = factory.connect().await?;
    let mut params = serde_json::Map::new();
    params.insert(
        "protocolVersion".into(),
        Value::String(MCP_LATEST_PROTOCOL_VERSION.into()),
    );
    params.insert("capabilities".into(), json!({}));
    params.insert(
        "clientInfo".into(),
        json!({
            "name": "agentkit-mcp",
            "version": env!("CARGO_PKG_VERSION")
        }),
    );
    if let Some(auth) = auth {
        params.insert("auth".into(), metadata_to_value(auth));
    }
    let init_params = Value::Object(params.clone());
    transport
        .send(McpFrame {
            value: json!({
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": init_params.clone()
            }),
        })
        .await?;
    let init_response = transport
        .recv()
        .await?
        .ok_or_else(|| McpError::Transport("transport closed during MCP initialization".into()))?;
    if let Some(error) = init_response.value.get("error") {
        if let Some(auth_request) =
            parse_auth_request(&config.id, "initialize", &init_params, error)
        {
            return Err(McpError::AuthRequired(Box::new(auth_request)));
        }
        return Err(McpError::Invocation(error.to_string()));
    }
    let result_value = init_response
        .value
        .get("result")
        .cloned()
        .ok_or_else(|| McpError::Protocol("initialize response missing result".into()))?;
    let initialize: InitializeResult = serde_json::from_value(result_value)
        .map_err(|error| McpError::Protocol(format!("malformed initialize result: {error}")))?;
    if !MCP_SUPPORTED_PROTOCOL_VERSIONS.contains(&initialize.protocol_version.as_str()) {
        return Err(McpError::Protocol(format!(
            "unsupported MCP protocol version negotiated during initialize: {}",
            initialize.protocol_version
        )));
    }
    let capabilities = initialize.capabilities;
    transport
        .send(McpFrame {
            value: json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }),
        })
        .await?;

    Ok((McpConnectionInner::Legacy(transport), capabilities))
}

async fn connect_rmcp_stdio(
    config: &McpServerConfig,
    binding: &StdioTransportConfig,
    notification_tx: mpsc::UnboundedSender<RmcpServerNotification>,
) -> Result<(McpConnectionInner, McpServerCapabilities), McpError> {
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

    let service = AgentkitRmcpClientHandler::new(notification_tx)
        .serve(transport)
        .await
        .map_err(|error| rmcp_initialize_error(config, error))?;
    let capabilities = service
        .peer_info()
        .map(|info| rmcp_server_capabilities_to_agentkit(&info.capabilities))
        .unwrap_or_default();

    Ok((McpConnectionInner::Rmcp(service), capabilities))
}

async fn connect_rmcp_streamable_http(
    config: &McpServerConfig,
    binding: &StreamableHttpTransportConfig,
    auth: Option<&MetadataMap>,
    notification_tx: mpsc::UnboundedSender<RmcpServerNotification>,
) -> Result<(McpConnectionInner, McpServerCapabilities), McpError> {
    if binding.client.is_some() {
        return Err(McpError::Transport(
            "custom agentkit HTTP clients are not supported by the RMCP Streamable HTTP transport; use static headers or the default reqwest client".into(),
        ));
    }

    let auth_header = auth
        .and_then(bearer_token_from_metadata)
        .or_else(|| binding.bearer_token.clone());
    let mut rmcp_config = RmcpStreamableHttpClientTransportConfig::with_uri(binding.url.clone());
    if let Some(auth_header) = auth_header {
        rmcp_config = rmcp_config.auth_header(auth_header);
    }
    rmcp_config = rmcp_config.custom_headers(
        binding
            .headers
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    );
    let transport = StreamableHttpClientTransport::from_config(rmcp_config);

    let service = AgentkitRmcpClientHandler::new(notification_tx)
        .serve(transport)
        .await
        .map_err(|error| rmcp_initialize_error(config, error))?;
    let capabilities = service
        .peer_info()
        .map(|info| rmcp_server_capabilities_to_agentkit(&info.capabilities))
        .unwrap_or_default();

    Ok((McpConnectionInner::Rmcp(service), capabilities))
}

impl McpConnection {
    /// Returns the [`McpServerId`] for this connection.
    pub fn server_id(&self) -> &McpServerId {
        &self.server_id
    }

    /// Closes the underlying transport, shutting down the connection to the server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the transport cannot be closed cleanly.
    pub async fn close(&self) -> Result<(), McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => {
                service.close().await.map(|_| ()).map_err(|error| {
                    McpError::Transport(format!("RMCP service close failed: {error}"))
                })
            }
            McpConnectionInner::Legacy(transport) => transport.close().await,
        }
    }

    /// Stores or clears authentication credentials for future requests on this
    /// connection.
    ///
    /// After calling this method with [`AuthResolution::Provided`], every subsequent
    /// JSON-RPC request will include the credentials in an `auth` field.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the resolution cannot be applied.
    pub async fn resolve_auth(&self, resolution: AuthResolution) -> Result<(), McpError> {
        let mut auth = self.auth.lock().await;
        match resolution {
            AuthResolution::Provided { credentials, .. } => {
                *auth = Some(credentials);
            }
            AuthResolution::Cancelled { .. } => {
                *auth = None;
            }
        }
        if matches!(&*self.inner.lock().await, McpConnectionInner::Legacy(_)) {
            return Ok(());
        }
        let replacement = self.reconnect_rmcp_inner(auth.as_ref()).await?;
        *self.inner.lock().await = replacement;
        Ok(())
    }

    /// Performs capability discovery by listing tools, resources, and
    /// prompts — but only for the capabilities the server advertised during
    /// `initialize`. Unadvertised capabilities produce empty collections
    /// without making any requests, avoiding `-32601 Method not found`
    /// errors from spec-compliant servers.
    ///
    /// Returns an [`McpDiscoverySnapshot`] containing everything the server
    /// advertised.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if any advertised list request fails.
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

    /// Returns the capabilities advertised by the server during `initialize`.
    pub fn capabilities(&self) -> &McpServerCapabilities {
        &self.capabilities
    }

    async fn drain_notifications(&self) -> Vec<RmcpServerNotification> {
        let mut notifications = self.notifications.lock().await;
        let mut drained = Vec::new();
        while let Ok(notification) = notifications.try_recv() {
            drained.push(notification);
        }
        drained
    }

    /// Lists all tools advertised by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the `tools/list` request fails.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => service
                .list_all_tools()
                .await
                .map_err(rmcp_service_error)
                .and_then(|tools| {
                    tools
                        .into_iter()
                        .map(rmcp_tool_descriptor)
                        .collect::<Result<Vec<_>, _>>()
                }),
            McpConnectionInner::Legacy(_) => {
                drop(inner);
                let result = self.request("tools/list", json!({})).await?;
                let payload: ToolsListPayload =
                    serde_json::from_value(result).map_err(McpError::Serialize)?;
                payload
                    .tools
                    .into_iter()
                    .map(parse_tool_descriptor)
                    .collect()
            }
        }
    }

    /// Lists all resources advertised by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the `resources/list` request fails.
    pub async fn list_resources(&self) -> Result<Vec<McpResourceDescriptor>, McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => service
                .list_all_resources()
                .await
                .map_err(rmcp_service_error)
                .map(|resources| {
                    resources
                        .into_iter()
                        .map(rmcp_resource_descriptor)
                        .collect()
                }),
            McpConnectionInner::Legacy(_) => {
                drop(inner);
                let result = self.request("resources/list", json!({})).await?;
                let payload: ResourcesListPayload =
                    serde_json::from_value(result).map_err(McpError::Serialize)?;
                payload
                    .resources
                    .into_iter()
                    .map(parse_resource_descriptor)
                    .collect()
            }
        }
    }

    /// Lists all prompts advertised by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the `prompts/list` request fails.
    pub async fn list_prompts(&self) -> Result<Vec<McpPromptDescriptor>, McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => service
                .list_all_prompts()
                .await
                .map_err(rmcp_service_error)
                .map(|prompts| prompts.into_iter().map(rmcp_prompt_descriptor).collect()),
            McpConnectionInner::Legacy(_) => {
                drop(inner);
                let result = self.request("prompts/list", json!({})).await?;
                let payload: PromptsListPayload =
                    serde_json::from_value(result).map_err(McpError::Serialize)?;
                payload
                    .prompts
                    .into_iter()
                    .map(parse_prompt_descriptor)
                    .collect()
            }
        }
    }

    /// Invokes a tool on the MCP server and returns the raw JSON result.
    ///
    /// # Arguments
    ///
    /// * `name` - The tool name as it appears in the server's tool list.
    /// * `arguments` - A JSON value matching the tool's input schema.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::AuthRequired`] if the server demands authentication,
    /// or another [`McpError`] variant on transport or protocol failures.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value, McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => {
                let mut params = rmcp_model::CallToolRequestParams::new(name.to_string());
                if !arguments.is_null() {
                    params = params
                        .with_arguments(value_to_json_object(arguments, "tools/call arguments")?);
                }
                service
                    .call_tool(params)
                    .await
                    .map_err(|error| {
                        rmcp_operation_error(
                            &self.server_id,
                            "tools/call",
                            json!({ "name": name }),
                            error,
                        )
                    })
                    .and_then(|result| serde_json::to_value(result).map_err(McpError::Serialize))
            }
            McpConnectionInner::Legacy(_) => {
                drop(inner);
                self.request(
                    "tools/call",
                    json!({
                        "name": name,
                        "arguments": arguments,
                    }),
                )
                .await
            }
        }
    }

    /// Reads a resource from the MCP server by URI.
    ///
    /// # Arguments
    ///
    /// * `uri` - The resource URI (e.g. `"file:///tmp/example.txt"`).
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the resource cannot be read or the response is malformed.
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceContents, McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => service
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
                .and_then(rmcp_resource_contents),
            McpConnectionInner::Legacy(_) => {
                drop(inner);
                let result = self
                    .request(
                        "resources/read",
                        json!({
                            "uri": uri,
                        }),
                    )
                    .await?;
                legacy_resource_contents(result)
            }
        }
    }

    /// Retrieves a prompt from the MCP server, rendering it with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `name` - The prompt name as it appears in the server's prompt list.
    /// * `arguments` - A JSON value containing the prompt's input arguments.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the prompt cannot be retrieved or the response is malformed.
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<PromptContents, McpError> {
        let mut inner = self.inner.lock().await;
        match &mut *inner {
            McpConnectionInner::Rmcp(service) => {
                let mut params = rmcp_model::GetPromptRequestParams::new(name);
                if !arguments.is_null() {
                    params = params
                        .with_arguments(value_to_json_object(arguments, "prompts/get arguments")?);
                }
                service
                    .get_prompt(params)
                    .await
                    .map_err(|error| {
                        rmcp_operation_error(
                            &self.server_id,
                            "prompts/get",
                            json!({ "name": name }),
                            error,
                        )
                    })
                    .and_then(rmcp_prompt_contents)
            }
            McpConnectionInner::Legacy(_) => {
                drop(inner);
                let result = self
                    .request(
                        "prompts/get",
                        json!({
                            "name": name,
                            "arguments": arguments,
                        }),
                    )
                    .await?;
                legacy_prompt_contents(result)
            }
        }
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let params = self.enrich_params(params.clone()).await;
        let mut inner = self.inner.lock().await;
        let McpConnectionInner::Legacy(transport) = &mut *inner else {
            return Err(McpError::Protocol(format!(
                "raw JSON-RPC request path is not available for RMCP-backed connection method {method}"
            )));
        };
        transport
            .send(McpFrame {
                value: json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "method": method,
                    "params": params,
                }),
            })
            .await?;

        loop {
            let Some(frame) = transport.recv().await? else {
                return Err(McpError::Transport(
                    "transport closed while waiting for MCP response".into(),
                ));
            };

            if frame.value.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }

            if let Some(error) = frame.value.get("error") {
                if let Some(auth_request) =
                    parse_auth_request(&self.server_id, method, &params, error)
                {
                    return Err(McpError::AuthRequired(Box::new(auth_request)));
                }
                return Err(McpError::Invocation(error.to_string()));
            }

            return frame
                .value
                .get("result")
                .cloned()
                .ok_or_else(|| McpError::Protocol("MCP response missing result".into()));
        }
    }

    async fn enrich_params(&self, params: Value) -> Value {
        let auth = self.auth.lock().await;
        let Some(auth) = auth.as_ref() else {
            return params;
        };

        match params {
            Value::Object(mut object) => {
                object
                    .entry("auth")
                    .or_insert_with(|| metadata_to_value(auth));
                Value::Object(object)
            }
            other => other,
        }
    }

    /// Replays an MCP operation that previously failed with an auth challenge.
    ///
    /// This is called after credentials have been resolved via [`resolve_auth`](Self::resolve_auth).
    /// The operation is re-issued with the stored credentials attached.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::AuthResolution`] if the operation targets a different server,
    /// or other [`McpError`] variants if the replayed operation itself fails.
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

/// Adapter that exposes an MCP tool as an [`Invocable`] for the capabilities system.
///
/// This is the capabilities-layer adapter. For the tool-layer adapter, see
/// [`McpToolAdapter`]. Names are prefixed with `mcp_<server_id>_<tool_name>`
/// so they satisfy provider validators that only allow `[a-zA-Z0-9_-]`
/// (e.g. Anthropic on Vertex).
pub struct McpInvocable {
    connection: Arc<McpConnection>,
    descriptor: McpToolDescriptor,
    spec: InvocableSpec,
}

impl McpInvocable {
    /// Creates a new invocable adapter for the given MCP tool.
    ///
    /// # Arguments
    ///
    /// * `connection` - A shared connection to the MCP server that owns the tool.
    /// * `descriptor` - The tool descriptor obtained from discovery.
    pub fn new(connection: Arc<McpConnection>, descriptor: McpToolDescriptor) -> Self {
        let spec = InvocableSpec {
            name: CapabilityName::new(format!(
                "mcp_{}_{}",
                connection.server_id(),
                descriptor.name
            )),
            description: descriptor
                .description
                .clone()
                .unwrap_or_else(|| descriptor.name.clone()),
            input_schema: descriptor.input_schema.clone(),
            metadata: descriptor.metadata.clone(),
        };

        Self {
            connection,
            descriptor,
            spec,
        }
    }
}

#[async_trait]
impl Invocable for McpInvocable {
    fn spec(&self) -> &InvocableSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: InvocableRequest,
        _ctx: &mut CapabilityContext<'_>,
    ) -> Result<InvocableResult, CapabilityError> {
        let result = self
            .connection
            .call_tool(&self.descriptor.name, request.input)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => {
                    CapabilityError::Unavailable(format!("auth required: {:?}", request))
                }
                other => CapabilityError::ExecutionFailed(other.to_string()),
            })?;

        Ok(InvocableResult {
            output: value_to_invocable_output(result),
            metadata: MetadataMap::new(),
        })
    }
}

/// Adapter that exposes a single MCP resource as a [`ResourceProvider`].
///
/// Created automatically by [`McpCapabilityProvider::from_snapshot`] for each
/// resource discovered on the server.
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
        self.connection
            .read_resource(&id.0)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => {
                    CapabilityError::Unavailable(format!("auth required: {:?}", request))
                }
                other => CapabilityError::ExecutionFailed(other.to_string()),
            })
    }
}

/// Adapter that exposes a single MCP prompt as a [`PromptProvider`].
///
/// Created automatically by [`McpCapabilityProvider::from_snapshot`] for each
/// prompt discovered on the server.
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
        self.connection
            .get_prompt(&id.0, args)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => {
                    CapabilityError::Unavailable(format!("auth required: {:?}", request))
                }
                other => CapabilityError::ExecutionFailed(other.to_string()),
            })
    }
}

/// A [`CapabilityProvider`] that surfaces MCP tools, resources, and prompts into the
/// agentkit capabilities system.
///
/// Built from a discovery snapshot, this provider wraps each MCP tool as an
/// [`McpInvocable`], each resource as an [`McpResourceHandle`], and each prompt as
/// an [`McpPromptHandle`].
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use agentkit_mcp::{McpCapabilityProvider, McpServerConfig, McpTransportBinding, StdioTransportConfig};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = McpServerConfig::new(
///     "filesystem",
///     McpTransportBinding::Stdio(StdioTransportConfig::new("npx")
///         .with_arg("-y")
///         .with_arg("@modelcontextprotocol/server-filesystem")),
/// );
/// let (connection, provider, snapshot) = McpCapabilityProvider::connect(&config).await?;
/// // `provider` implements CapabilityProvider and can be registered with an agent.
/// # Ok(())
/// # }
/// ```
pub struct McpCapabilityProvider {
    invocables: Vec<Arc<dyn Invocable>>,
    resources: Vec<Arc<dyn ResourceProvider>>,
    prompts: Vec<Arc<dyn PromptProvider>>,
}

impl McpCapabilityProvider {
    /// Creates a capability provider from an existing connection and its discovery
    /// snapshot.
    ///
    /// Each tool, resource, and prompt in the snapshot is wrapped in the appropriate
    /// adapter type.
    pub fn from_snapshot(connection: Arc<McpConnection>, snapshot: &McpDiscoverySnapshot) -> Self {
        let invocables = snapshot
            .tools
            .iter()
            .cloned()
            .map(|descriptor| {
                Arc::new(McpInvocable::new(connection.clone(), descriptor)) as Arc<dyn Invocable>
            })
            .collect();

        let resources = snapshot
            .resources
            .iter()
            .cloned()
            .map(|descriptor| {
                Arc::new(McpResourceHandle {
                    connection: connection.clone(),
                    descriptor: ResourceDescriptor {
                        id: ResourceId::new(descriptor.id),
                        name: descriptor.name,
                        description: descriptor.description,
                        mime_type: descriptor.mime_type,
                        metadata: descriptor.metadata,
                    },
                }) as Arc<dyn ResourceProvider>
            })
            .collect();

        let prompts = snapshot
            .prompts
            .iter()
            .cloned()
            .map(|descriptor| {
                Arc::new(McpPromptHandle {
                    connection: connection.clone(),
                    descriptor: PromptDescriptor {
                        id: PromptId::new(descriptor.id),
                        name: descriptor.name,
                        description: descriptor.description,
                        input_schema: descriptor.input_schema,
                        metadata: descriptor.metadata,
                    },
                }) as Arc<dyn PromptProvider>
            })
            .collect();

        Self {
            invocables,
            resources,
            prompts,
        }
    }

    /// Merges multiple capability providers into a single provider.
    ///
    /// This is useful when managing several MCP servers through a
    /// [`McpServerManager`] and you want one combined provider for the agent.
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

    /// Connects to an MCP server, performs discovery, and builds a capability
    /// provider in one step.
    ///
    /// Returns the shared connection, the provider, and the discovery snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if connection or discovery fails.
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

/// A connected MCP server together with its configuration and discovery snapshot.
///
/// Obtained from [`McpServerManager::connect_server`] or
/// [`McpServerManager::connect_all`]. Provides convenience methods to create
/// tool registries and capability providers from the server's discovered capabilities.
#[derive(Clone)]
pub struct McpServerHandle {
    config: McpServerConfig,
    connection: Arc<McpConnection>,
    snapshot: McpDiscoverySnapshot,
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

    /// Builds a [`ToolRegistry`] containing an [`McpToolAdapter`] for each tool
    /// discovered on this server.
    pub fn tool_registry(&self) -> ToolRegistry {
        self.snapshot
            .tools
            .iter()
            .cloned()
            .fold(ToolRegistry::new(), |registry, descriptor| {
                registry.with(McpToolAdapter::new(
                    self.server_id(),
                    self.connection.clone(),
                    descriptor,
                ))
            })
    }

    /// Builds an [`McpCapabilityProvider`] from this server's discovery snapshot.
    pub fn capability_provider(&self) -> McpCapabilityProvider {
        McpCapabilityProvider::from_snapshot(self.connection.clone(), &self.snapshot)
    }
}

/// Manages the lifecycle of one or more MCP servers: registration, connection,
/// discovery, refresh, disconnection, and auth resolution.
///
/// This is the primary entry point for integrating MCP servers into an agentkit
/// application. Register server configurations, connect them, and then obtain a
/// combined [`ToolRegistry`] or [`McpCapabilityProvider`] for use in an agent loop.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_mcp::{
///     McpServerConfig, McpServerManager, McpTransportBinding, StdioTransportConfig,
/// };
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut manager = McpServerManager::new()
///     .with_server(McpServerConfig::new(
///         "filesystem",
///         McpTransportBinding::Stdio(
///             StdioTransportConfig::new("npx")
///                 .with_arg("-y")
///                 .with_arg("@modelcontextprotocol/server-filesystem"),
///         ),
///     ))
///     .with_server(McpServerConfig::new(
///         "github",
///         McpTransportBinding::Stdio(
///             StdioTransportConfig::new("npx")
///                 .with_arg("-y")
///                 .with_arg("@modelcontextprotocol/server-github"),
///         ),
///     ));
///
/// let handles = manager.connect_all().await?;
/// let registry = manager.tool_registry();
/// println!("tools: {:?}", registry.specs().iter().map(|s| &s.name).collect::<Vec<_>>());
/// # Ok(())
/// # }
/// ```
pub struct McpServerManager {
    configs: BTreeMap<McpServerId, McpServerConfig>,
    connections: BTreeMap<McpServerId, McpServerHandle>,
    auth: BTreeMap<McpServerId, MetadataMap>,
    catalog_tx: broadcast::Sender<McpCatalogEvent>,
}

impl Default for McpServerManager {
    fn default() -> Self {
        let (catalog_tx, _) = broadcast::channel(128);
        Self {
            configs: BTreeMap::new(),
            connections: BTreeMap::new(),
            auth: BTreeMap::new(),
            catalog_tx,
        }
    }
}

impl McpServerManager {
    /// Creates an empty server manager with no registered servers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a server configuration and returns `self` for chaining.
    ///
    /// The server is not connected until [`connect_server`](Self::connect_server) or
    /// [`connect_all`](Self::connect_all) is called.
    pub fn with_server(mut self, config: McpServerConfig) -> Self {
        self.register_server(config);
        self
    }

    /// Registers a server configuration by mutable reference.
    ///
    /// The server is not connected until [`connect_server`](Self::connect_server) or
    /// [`connect_all`](Self::connect_all) is called.
    pub fn register_server(&mut self, config: McpServerConfig) -> &mut Self {
        self.configs.insert(config.id.clone(), config);
        self
    }

    /// Returns the handle for a connected server, or `None` if it is not connected.
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
    ///
    /// Performs the MCP handshake and full capability discovery.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::UnknownServer`] if the server ID has not been registered,
    /// or other [`McpError`] variants if connection or discovery fails.
    pub async fn connect_server(
        &mut self,
        server_id: &McpServerId,
    ) -> Result<McpServerHandle, McpError> {
        let config = self
            .configs
            .get(server_id)
            .cloned()
            .ok_or_else(|| McpError::UnknownServer(server_id.to_string()))?;
        let connection =
            Arc::new(McpConnection::connect_with_auth(&config, self.auth.get(server_id)).await?);
        let snapshot = connection.discover().await?;
        let handle = McpServerHandle {
            config,
            connection,
            snapshot,
        };
        self.connections.insert(server_id.clone(), handle.clone());
        self.emit_catalog_event(McpCatalogEvent::ServerConnected {
            server_id: server_id.clone(),
        });
        Ok(handle)
    }

    /// Connects all registered servers sequentially.
    ///
    /// Returns a handle for each server in registration order. If any server fails
    /// to connect, the error is returned immediately and remaining servers are
    /// not attempted.
    ///
    /// # Errors
    ///
    /// Returns the first [`McpError`] encountered during connection.
    pub async fn connect_all(&mut self) -> Result<Vec<McpServerHandle>, McpError> {
        let server_ids = self.configs.keys().cloned().collect::<Vec<_>>();
        let mut handles = Vec::with_capacity(server_ids.len());

        for server_id in server_ids {
            handles.push(self.connect_server(&server_id).await?);
        }

        Ok(handles)
    }

    /// Re-discovers capabilities for a connected server, updating the stored snapshot.
    ///
    /// Call this after the server's capabilities may have changed (e.g. after
    /// installing a plugin).
    ///
    /// # Errors
    ///
    /// Returns [`McpError::UnknownServer`] if the server is not connected, or other
    /// [`McpError`] variants if discovery fails.
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

    /// Processes pending server list-change notifications and refreshes affected snapshots.
    ///
    /// RMCP receives `notifications/tools/list_changed`,
    /// `notifications/resources/list_changed`, and
    /// `notifications/prompts/list_changed` on the transport task. This method
    /// drains those notifications, re-runs discovery for each affected server,
    /// updates the manager snapshot, and emits diffed [`McpCatalogEvent`]s.
    ///
    /// Hosts can call this between model turns to make capability changes visible
    /// on the next request without injecting synthetic transcript text.
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

    /// Disconnects a server and removes it from the active connections.
    ///
    /// The server configuration remains registered and can be reconnected later
    /// with [`connect_server`](Self::connect_server).
    ///
    /// # Errors
    ///
    /// Returns [`McpError::UnknownServer`] if the server is not connected.
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

    /// Stores or clears authentication credentials for a server and, if already
    /// connected, updates the live connection as well.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::UnknownServer`] if the server ID from the resolution
    /// does not match any registered server.
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

    /// Resolves authentication and immediately replays the operation that originally
    /// triggered the auth challenge.
    ///
    /// This is a convenience method combining [`resolve_auth`](Self::resolve_auth)
    /// and [`replay_auth_request`](Self::replay_auth_request).
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if auth resolution or the replayed operation fails.
    pub async fn resolve_auth_and_resume(
        &mut self,
        resolution: AuthResolution,
    ) -> Result<McpOperationResult, McpError> {
        let request = resolution.request().clone();
        self.resolve_auth(resolution).await?;
        self.replay_auth_request(&request).await
    }

    /// Replays an auth request's original MCP operation using stored credentials.
    ///
    /// For connect operations the server is (re)connected. For tool calls, resource
    /// reads, and prompt retrievals the request is re-issued on the existing or
    /// newly established connection.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the operation cannot be replayed.
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

    /// Builds a combined [`ToolRegistry`] containing [`McpToolAdapter`]s for every
    /// tool discovered across all connected servers.
    ///
    /// Tool names are prefixed as `mcp_<server_id>_<tool_name>`.
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

    /// Builds a combined [`McpCapabilityProvider`] from all connected servers,
    /// merging their tools, resources, and prompts.
    pub fn capability_provider(&self) -> McpCapabilityProvider {
        McpCapabilityProvider::merge(
            self.connections
                .values()
                .map(McpServerHandle::capability_provider),
        )
    }

    /// Builds an MCP-backed executor from the current discovered tool snapshot.
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
        previous.tools.iter().map(|item| (item.name.clone(), item)),
        current.tools.iter().map(|item| (item.name.clone(), item)),
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
            .map(|item| (item.id.clone(), item)),
        current.resources.iter().map(|item| (item.id.clone(), item)),
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
        previous.prompts.iter().map(|item| (item.id.clone(), item)),
        current.prompts.iter().map(|item| (item.id.clone(), item)),
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
        .filter_map(|(name, item)| {
            previous
                .get(name)
                .is_some_and(|previous_item| previous_item != item)
                .then(|| name.clone())
        })
        .collect();

    (added, removed, changed)
}

/// A tool executor backed by MCP tool adapters.
///
/// The executor stores a replaceable registry snapshot so hosts can refresh it
/// after [`McpServerManager`] catalog events without rebuilding the entire
/// agent loop.
#[derive(Clone)]
pub struct McpToolExecutor {
    registry: Arc<RwLock<ToolRegistry>>,
    connections: Arc<RwLock<BTreeMap<McpServerId, Arc<McpConnection>>>>,
    events: Arc<StdMutex<Vec<ToolCatalogEvent>>>,
}

impl McpToolExecutor {
    /// Creates an executor from a manager's current connected server snapshot.
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

/// Adapter that exposes an MCP tool as an agentkit [`Tool`].
///
/// This is the tool-layer adapter for the tool registry. For the capabilities-layer
/// adapter, see [`McpInvocable`]. Tool names are prefixed as
/// `mcp_<server_id>_<tool_name>`.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use agentkit_core::MetadataMap;
/// use agentkit_mcp::{McpToolAdapter, McpToolDescriptor, McpServerId};
/// # // McpToolAdapter::new requires a connection which we cannot construct in a doc test,
/// # // so this example only shows the construction pattern.
/// ```
pub struct McpToolAdapter {
    descriptor: McpToolDescriptor,
    connection: Arc<McpConnection>,
    spec: ToolSpec,
}

impl McpToolAdapter {
    /// Creates a new tool adapter for the given MCP tool.
    ///
    /// # Arguments
    ///
    /// * `server_id` - The server's identifier, used to namespace the tool name.
    /// * `connection` - A shared connection to the owning MCP server.
    /// * `descriptor` - The tool descriptor obtained from discovery.
    pub fn new(
        server_id: &McpServerId,
        connection: Arc<McpConnection>,
        descriptor: McpToolDescriptor,
    ) -> Self {
        let spec = ToolSpec {
            name: ToolName::new(format!("mcp_{}_{}", server_id, descriptor.name)),
            description: descriptor
                .description
                .clone()
                .unwrap_or_else(|| descriptor.name.clone()),
            input_schema: descriptor.input_schema.clone(),
            annotations: ToolAnnotations::default(),
            metadata: descriptor.metadata.clone(),
        };

        Self {
            descriptor,
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
            .call_tool(&self.descriptor.name, request.input)
            .await
            .map_err(|error| match error {
                McpError::AuthRequired(request) => ToolError::AuthRequired(request),
                other => ToolError::ExecutionFailed(other.to_string()),
            })?;

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: invocable_output_to_tool_output(value_to_invocable_output(result)),
                is_error: false,
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

fn rmcp_tool_descriptor(tool: rmcp_model::Tool) -> Result<McpToolDescriptor, McpError> {
    let mut metadata = MetadataMap::new();
    if let Some(title) = tool.title.clone() {
        metadata.insert("title".into(), Value::String(title));
    }
    if let Some(output_schema) = tool.output_schema.as_ref() {
        metadata.insert(
            "output_schema".into(),
            Value::Object((**output_schema).clone()),
        );
    }
    if let Some(annotations) = tool.annotations.as_ref() {
        metadata.insert(
            "annotations".into(),
            serde_json::to_value(annotations).map_err(McpError::Serialize)?,
        );
    }
    if let Some(execution) = tool.execution.as_ref() {
        metadata.insert(
            "execution".into(),
            serde_json::to_value(execution).map_err(McpError::Serialize)?,
        );
    }
    if let Some(icons) = tool.icons.as_ref() {
        metadata.insert(
            "icons".into(),
            serde_json::to_value(icons).map_err(McpError::Serialize)?,
        );
    }
    if let Some(meta) = tool.meta.as_ref() {
        metadata.insert(
            "_meta".into(),
            serde_json::to_value(meta).map_err(McpError::Serialize)?,
        );
    }

    Ok(McpToolDescriptor {
        name: tool.name.into_owned(),
        description: tool.description.map(|description| description.into_owned()),
        input_schema: Value::Object((*tool.input_schema).clone()),
        metadata,
    })
}

fn rmcp_resource_descriptor(resource: rmcp_model::Resource) -> McpResourceDescriptor {
    let mut metadata = MetadataMap::new();
    if let Some(title) = resource.title.clone() {
        metadata.insert("title".into(), Value::String(title));
    }
    if let Some(size) = resource.size {
        metadata.insert("size".into(), Value::Number(size.into()));
    }
    if let Some(icons) = resource.icons.as_ref() {
        if let Ok(value) = serde_json::to_value(icons) {
            metadata.insert("icons".into(), value);
        }
    }
    if let Some(meta) = resource.meta.as_ref() {
        if let Ok(value) = serde_json::to_value(meta) {
            metadata.insert("_meta".into(), value);
        }
    }

    McpResourceDescriptor {
        id: resource.uri.clone(),
        name: resource.name.clone(),
        description: resource.description.clone(),
        mime_type: resource.mime_type.clone(),
        metadata,
    }
}

fn rmcp_prompt_descriptor(prompt: rmcp_model::Prompt) -> McpPromptDescriptor {
    let mut metadata = MetadataMap::new();
    if let Some(title) = prompt.title.clone() {
        metadata.insert("title".into(), Value::String(title));
    }
    if let Some(icons) = prompt.icons.as_ref() {
        if let Ok(value) = serde_json::to_value(icons) {
            metadata.insert("icons".into(), value);
        }
    }
    if let Some(meta) = prompt.meta.as_ref() {
        if let Ok(value) = serde_json::to_value(meta) {
            metadata.insert("_meta".into(), value);
        }
    }

    let properties = prompt
        .arguments
        .unwrap_or_default()
        .into_iter()
        .map(|argument| (argument.name, json!({ "type": "string" })))
        .collect::<serde_json::Map<String, Value>>();

    McpPromptDescriptor {
        id: prompt.name.clone(),
        name: prompt.name,
        description: prompt.description,
        input_schema: json!({
            "type": "object",
            "properties": properties,
        }),
        metadata,
    }
}

fn parse_tool_descriptor(value: Value) -> Result<McpToolDescriptor, McpError> {
    Ok(McpToolDescriptor {
        name: required_string(&value, "name")?,
        description: value
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        input_schema: value
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object" })),
        metadata: MetadataMap::new(),
    })
}

fn parse_resource_descriptor(value: Value) -> Result<McpResourceDescriptor, McpError> {
    Ok(McpResourceDescriptor {
        id: required_string(&value, "uri")?,
        name: value
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                value
                    .get("uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            }),
        description: value
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        mime_type: value
            .get("mimeType")
            .and_then(Value::as_str)
            .map(str::to_owned),
        metadata: MetadataMap::new(),
    })
}

fn parse_prompt_descriptor(value: Value) -> Result<McpPromptDescriptor, McpError> {
    let name = required_string(&value, "name")?;
    let properties = value
        .get("arguments")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|arg| {
            let name = arg.get("name")?.as_str()?.to_string();
            Some((name, json!({ "type": "string" })))
        })
        .collect::<serde_json::Map<String, Value>>();

    Ok(McpPromptDescriptor {
        id: name.clone(),
        name,
        description: value
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned),
        input_schema: json!({
            "type": "object",
            "properties": properties,
        }),
        metadata: MetadataMap::new(),
    })
}

fn legacy_resource_contents(result: Value) -> Result<ResourceContents, McpError> {
    let payload: ResourcesReadPayload =
        serde_json::from_value(result).map_err(McpError::Serialize)?;
    let content = payload
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| McpError::Protocol("resources/read returned no contents".into()))?;

    let data = if let Some(text) = content.text {
        DataRef::InlineText(text)
    } else if let Some(found_uri) = content.uri {
        DataRef::Uri(found_uri)
    } else {
        return Err(McpError::Protocol(
            "unsupported resource content shape".into(),
        ));
    };

    Ok(ResourceContents {
        data,
        metadata: MetadataMap::new(),
    })
}

fn rmcp_resource_contents(
    result: rmcp_model::ReadResourceResult,
) -> Result<ResourceContents, McpError> {
    let content = result
        .contents
        .into_iter()
        .next()
        .ok_or_else(|| McpError::Protocol("resources/read returned no contents".into()))?;

    let data = match content {
        rmcp_model::ResourceContents::TextResourceContents { text, .. } => {
            DataRef::InlineText(text)
        }
        rmcp_model::ResourceContents::BlobResourceContents { uri, .. } => DataRef::Uri(uri),
    };

    Ok(ResourceContents {
        data,
        metadata: MetadataMap::new(),
    })
}

fn legacy_prompt_contents(result: Value) -> Result<PromptContents, McpError> {
    let payload: PromptsGetPayload = serde_json::from_value(result).map_err(McpError::Serialize)?;
    let items = payload
        .messages
        .into_iter()
        .map(parse_prompt_message)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PromptContents {
        items,
        metadata: MetadataMap::new(),
    })
}

fn rmcp_prompt_contents(result: rmcp_model::GetPromptResult) -> Result<PromptContents, McpError> {
    let items = result
        .messages
        .into_iter()
        .map(|message| serde_json::to_value(message).map_err(McpError::Serialize))
        .map(|value| value.and_then(parse_prompt_message))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PromptContents {
        items,
        metadata: MetadataMap::new(),
    })
}

fn parse_prompt_message(value: Value) -> Result<Item, McpError> {
    let role = value.get("role").and_then(Value::as_str).unwrap_or("user");
    let kind = match role {
        "assistant" => ItemKind::Assistant,
        "system" => ItemKind::System,
        _ => ItemKind::User,
    };

    let content = value.get("content").cloned().unwrap_or(Value::Null);
    let text = if let Some(text) = content.get("text").and_then(Value::as_str) {
        text.to_string()
    } else if let Some(text) = content.as_str() {
        text.to_string()
    } else {
        content.to_string()
    };

    Ok(Item {
        id: None,
        kind,
        parts: vec![Part::Text(TextPart {
            text,
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    })
}

fn required_string(value: &Value, field: &str) -> Result<String, McpError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| McpError::Protocol(format!("missing string field {field}")))
}

fn value_to_invocable_output(value: Value) -> InvocableOutput {
    if let Some(content) = value.get("content").and_then(Value::as_array) {
        let text = content
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            return InvocableOutput::Text(text);
        }
    }

    if let Some(text) = value.as_str() {
        InvocableOutput::Text(text.to_string())
    } else {
        InvocableOutput::Structured(value)
    }
}

fn invocable_output_to_tool_output(output: InvocableOutput) -> ToolOutput {
    match output {
        InvocableOutput::Text(text) => ToolOutput::Text(text),
        InvocableOutput::Structured(value) => ToolOutput::Structured(value),
        InvocableOutput::Items(items) => {
            ToolOutput::Parts(items.into_iter().flat_map(|item| item.parts).collect())
        }
        InvocableOutput::Data(data) => ToolOutput::Structured(json!({ "data": data })),
    }
}

fn metadata_to_value(metadata: &MetadataMap) -> Value {
    Value::Object(
        metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
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

fn parse_auth_request(
    server_id: &McpServerId,
    method: &str,
    params: &Value,
    error: &Value,
) -> Option<AuthRequest> {
    let code = error.get("code").and_then(Value::as_i64);
    let message = error.get("message").and_then(Value::as_str);
    let data = error.get("data");

    let auth_marker = matches!(code, Some(401 | -32001))
        || data
            .and_then(|data| data.get("auth_required"))
            .and_then(Value::as_bool)
            == Some(true)
        || data.and_then(|data| data.get("auth")).is_some();

    if !auth_marker {
        return None;
    }

    let mut challenge = MetadataMap::new();
    challenge.insert("server_id".into(), Value::String(server_id.to_string()));
    challenge.insert("method".into(), Value::String(method.into()));

    if let Some(code) = code {
        challenge.insert("code".into(), Value::Number(code.into()));
    }
    if let Some(message) = message {
        challenge.insert("message".into(), Value::String(message.into()));
    }
    if let Some(data) = data {
        challenge.insert("data".into(), data.clone());
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

async fn read_sse_stream<R>(
    mut reader: R,
    response_url: Url,
    frame_tx: mpsc::UnboundedSender<Result<McpFrame, McpError>>,
    endpoint_tx: oneshot::Sender<Result<Url, McpError>>,
) where
    R: AsyncBufRead + Unpin,
{
    let mut endpoint_tx = Some(endpoint_tx);
    loop {
        match read_next_sse_event(&mut reader).await {
            Ok(Some(event)) => {
                if let Some(endpoint) = legacy_sse_event_to_endpoint(&response_url, &event) {
                    if let Some(tx) = endpoint_tx.take() {
                        let _ = tx.send(endpoint);
                    }
                    continue;
                }

                if let Some(frame) = legacy_sse_event_to_frame(event) {
                    let _ = frame_tx.send(frame);
                }
            }
            Ok(None) => break,
            Err(error) => {
                if let Some(tx) = endpoint_tx.take() {
                    let _ = tx.send(Err(error));
                } else {
                    let _ = frame_tx.send(Err(error));
                }
                return;
            }
        }
    }

    if let Some(tx) = endpoint_tx.take() {
        let _ = tx.send(Err(McpError::Transport(
            "SSE stream ended before endpoint event".into(),
        )));
    }
}

fn resolve_sse_endpoint(response_url: &Url, endpoint: &str) -> Result<Url, McpError> {
    response_url
        .join(endpoint.trim())
        .map_err(|error| McpError::Transport(format!("invalid SSE endpoint URL: {error}")))
}

#[derive(Debug)]
struct SseEvent {
    event_name: Option<String>,
    data: String,
    id: Option<String>,
    retry_ms: Option<u64>,
}

async fn read_next_sse_event<R>(reader: &mut R) -> Result<Option<SseEvent>, McpError>
where
    R: AsyncBufRead + Unpin,
{
    let mut event_name = None;
    let mut data_lines = Vec::new();
    let mut id = None;
    let mut retry_ms = None;

    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).await.map_err(McpError::Io)?;
        if read == 0 {
            if event_name.is_none() && data_lines.is_empty() && id.is_none() && retry_ms.is_none() {
                return Ok(None);
            }
            return Ok(Some(SseEvent {
                event_name,
                data: data_lines.join("\n"),
                id,
                retry_ms,
            }));
        }

        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            if event_name.is_none() && data_lines.is_empty() && id.is_none() && retry_ms.is_none() {
                continue;
            }
            return Ok(Some(SseEvent {
                event_name,
                data: data_lines.join("\n"),
                id,
                retry_ms,
            }));
        }

        if line.starts_with(':') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim_start().to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start().to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix("id:") {
            id = Some(rest.trim_start().to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix("retry:") {
            retry_ms = rest.trim_start().parse().ok();
        }
    }
}

fn legacy_sse_event_to_endpoint(
    response_url: &Url,
    event: &SseEvent,
) -> Option<Result<Url, McpError>> {
    if event.event_name.as_deref() != Some("endpoint") {
        return None;
    }
    if event.data.is_empty() {
        return Some(Err(McpError::Transport(
            "legacy SSE endpoint event is missing data".into(),
        )));
    }
    Some(resolve_sse_endpoint(response_url, &event.data))
}

fn legacy_sse_event_to_frame(event: SseEvent) -> Option<Result<McpFrame, McpError>> {
    let event_name = event.event_name.unwrap_or_else(|| "message".into());
    if event_name != "message" || event.data.is_empty() {
        return None;
    }

    Some(
        serde_json::from_str(&event.data)
            .map_err(McpError::Serialize)
            .map(|value| McpFrame { value }),
    )
}

fn streamable_http_event_to_frame(event: SseEvent) -> Result<Option<McpFrame>, McpError> {
    let event_name = event.event_name.unwrap_or_else(|| "message".into());
    if event_name != "message" || event.data.is_empty() {
        return Ok(None);
    }

    let value = serde_json::from_str(&event.data).map_err(McpError::Serialize)?;
    Ok(Some(McpFrame { value }))
}

fn is_jsonrpc_request(value: &Value) -> bool {
    value.get("method").is_some() && value.get("id").is_some()
}

fn apply_streamable_http_headers(
    mut request: HttpRequestBuilder,
    protocol_version: Option<&str>,
    session_id: Option<&str>,
) -> HttpRequestBuilder {
    if let Some(protocol_version) = protocol_version {
        request = request.header("MCP-Protocol-Version", protocol_version);
    }
    if let Some(session_id) = session_id {
        request = request.header("MCP-Session-Id", session_id);
    }

    request
}

async fn streamable_http_status_error(
    operation: &str,
    status: StatusCode,
    response: HttpResponse,
) -> McpError {
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<unreadable response body>".into());
    McpError::Transport(format!("{operation} failed with status {status}: {body}"))
}

/// Errors produced by MCP transport, protocol, and lifecycle operations.
#[derive(Debug, Error)]
pub enum McpError {
    /// An underlying I/O error (e.g. spawning a child process or reading from a pipe).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// An HTTP-level error surfaced by the configured [`agentkit_http::HttpClient`].
    #[error("http error: {0}")]
    Http(#[from] HttpError),
    /// A JSON serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    /// A transport-level error (e.g. unexpected disconnection or bad SSE response).
    #[error("transport error: {0}")]
    Transport(String),
    /// An MCP protocol violation (e.g. missing required fields in a response).
    #[error("protocol error: {0}")]
    Protocol(String),
    /// The server requires authentication before the operation can proceed.
    /// Contains the [`AuthRequest`] that describes the challenge.
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

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc as StdArc, Mutex as StdMutex};

    use super::*;
    use agentkit_tools_core::{PermissionChecker, PermissionDecision, PermissionRequest};
    #[cfg(feature = "reqwest-client")]
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    #[cfg(feature = "reqwest-client")]
    use tokio::net::TcpListener;

    struct AllowAll;

    impl PermissionChecker for AllowAll {
        fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    struct FakeTransport {
        recv: VecDeque<Value>,
    }

    #[async_trait]
    impl McpTransport for FakeTransport {
        async fn send(&mut self, _message: McpFrame) -> Result<(), McpError> {
            Ok(())
        }

        async fn recv(&mut self) -> Result<Option<McpFrame>, McpError> {
            Ok(self.recv.pop_front().map(|value| McpFrame { value }))
        }

        async fn close(&mut self) -> Result<(), McpError> {
            Ok(())
        }
    }

    fn fake_connection(responses: Vec<Value>) -> McpConnection {
        let (_notification_tx, notification_rx) = mpsc::unbounded_channel();
        McpConnection {
            server_id: McpServerId::new("fake"),
            config: McpServerConfig::new(
                "fake",
                McpTransportBinding::Custom(Arc::new(FakeTransportFactory::new(Vec::new()))),
            ),
            inner: Mutex::new(McpConnectionInner::Legacy(Box::new(FakeTransport {
                recv: responses.into(),
            }))),
            auth: Mutex::new(None),
            notifications: Mutex::new(notification_rx),
            next_id: AtomicU64::new(1),
            capabilities: McpServerCapabilities::all(),
        }
    }

    #[derive(Clone)]
    struct FakeTransportFactory {
        responses: StdArc<StdMutex<VecDeque<Vec<Value>>>>,
    }

    impl FakeTransportFactory {
        fn new(sequences: Vec<Vec<Value>>) -> Self {
            Self {
                responses: StdArc::new(StdMutex::new(sequences.into())),
            }
        }
    }

    #[async_trait]
    impl McpTransportFactory for FakeTransportFactory {
        async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError> {
            let responses =
                self.responses.lock().unwrap().pop_front().ok_or_else(|| {
                    McpError::Transport("no fake transport responses left".into())
                })?;
            Ok(Box::new(FakeTransport {
                recv: responses.into(),
            }))
        }
    }

    #[tokio::test]
    async fn discovery_parses_snapshot() {
        let connection = fake_connection(vec![
            json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [{ "name": "echo", "description": "Echo", "inputSchema": {"type": "object"} }] } }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [{ "uri": "file:///tmp/example.txt", "name": "example.txt", "mimeType": "text/plain" }] } }),
            json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [{ "name": "summarize", "description": "Summarize", "arguments": [{ "name": "path" }] }] } }),
        ]);

        let snapshot = connection.discover().await.unwrap();
        assert_eq!(snapshot.tools[0].name, "echo");
        assert_eq!(snapshot.resources[0].id, "file:///tmp/example.txt");
        assert_eq!(snapshot.prompts[0].id, "summarize");
    }

    #[tokio::test]
    async fn discover_skips_unadvertised_capabilities() {
        // Server advertises ONLY tools — discover must not send
        // resources/list or prompts/list (real servers like Linear return
        // -32601 for those). The fixture deliberately provides a single
        // tools/list response and nothing else; if discover erroneously
        // tried to call resources/list, FakeTransport's empty queue would
        // yield `None` and the test would fail with a transport error.
        let factory = FakeTransportFactory::new(vec![vec![
            json!({
                "jsonrpc": "2.0",
                "id": 0,
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": { "tools": { "listChanged": true } },
                    "serverInfo": { "name": "tools-only", "version": "1.0.0" }
                }
            }),
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        { "name": "ping", "description": "ping", "inputSchema": {"type":"object"} }
                    ]
                }
            }),
        ]]);
        let config =
            McpServerConfig::new("tools-only", McpTransportBinding::Custom(Arc::new(factory)));
        let connection = McpConnection::connect(&config).await.unwrap();
        assert!(connection.capabilities().tools.is_some());
        assert_eq!(
            connection.capabilities().tools,
            Some(ToolsCapability {
                list_changed: Some(true),
            })
        );
        assert!(connection.capabilities().resources.is_none());
        assert!(connection.capabilities().prompts.is_none());

        let snapshot = connection.discover().await.unwrap();
        assert_eq!(snapshot.tools.len(), 1);
        assert_eq!(snapshot.tools[0].name, "ping");
        assert!(snapshot.resources.is_empty());
        assert!(snapshot.prompts.is_empty());
    }

    #[tokio::test]
    async fn tool_adapter_returns_text_output() {
        let connection = Arc::new(fake_connection(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "content": [{ "type": "text", "text": "pong" }] }
        })]));
        let server_id = connection.server_id().clone();
        let adapter = McpToolAdapter::new(
            &server_id,
            connection,
            McpToolDescriptor {
                name: "echo".into(),
                description: Some("Echo".into()),
                input_schema: json!({ "type": "object" }),
                metadata: MetadataMap::new(),
            },
        );
        let metadata = MetadataMap::new();
        let mut ctx = ToolContext {
            capability: CapabilityContext {
                session_id: None,
                turn_id: None,
                metadata: &metadata,
            },
            permissions: &AllowAll,
            resources: &(),
            cancellation: None,
        };

        let result = adapter
            .invoke(
                ToolRequest {
                    call_id: "call-1".into(),
                    tool_name: ToolName::new("mcp_fake_echo"),
                    input: json!({}),
                    session_id: "session-1".into(),
                    turn_id: "turn-1".into(),
                    metadata: MetadataMap::new(),
                },
                &mut ctx,
            )
            .await
            .unwrap();

        assert_eq!(result.result.output, ToolOutput::Text("pong".into()));
    }

    #[tokio::test]
    async fn request_surfaces_auth_required_errors() {
        let connection = fake_connection(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32001,
                "message": "authentication required",
                "data": {
                    "auth_required": true,
                    "scope": "secrets.read"
                }
            }
        })]);

        let error = connection.call_tool("echo", json!({})).await.unwrap_err();
        match error {
            McpError::AuthRequired(request) => {
                assert_eq!(request.provider, "mcp.fake");
                assert_eq!(
                    request.challenge.get("method"),
                    Some(&Value::String("tools/call".into()))
                );
                assert!(matches!(
                    request.operation,
                    AuthOperation::McpToolCall { ref tool_name, .. } if tool_name == "echo"
                ));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_adapter_maps_auth_required_into_tool_error() {
        let connection = Arc::new(fake_connection(vec![json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32001,
                "message": "authentication required",
                "data": { "auth_required": true }
            }
        })]));
        let server_id = connection.server_id().clone();
        let adapter = McpToolAdapter::new(
            &server_id,
            connection,
            McpToolDescriptor {
                name: "echo".into(),
                description: Some("Echo".into()),
                input_schema: json!({ "type": "object" }),
                metadata: MetadataMap::new(),
            },
        );
        let metadata = MetadataMap::new();
        let mut ctx = ToolContext {
            capability: CapabilityContext {
                session_id: None,
                turn_id: None,
                metadata: &metadata,
            },
            permissions: &AllowAll,
            resources: &(),
            cancellation: None,
        };

        let error = adapter
            .invoke(
                ToolRequest {
                    call_id: "call-1".into(),
                    tool_name: ToolName::new("mcp_fake_echo"),
                    input: json!({}),
                    session_id: "session-1".into(),
                    turn_id: "turn-1".into(),
                    metadata: MetadataMap::new(),
                },
                &mut ctx,
            )
            .await
            .unwrap_err();

        match error {
            ToolError::AuthRequired(request) => {
                assert_eq!(request.provider, "mcp.fake");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    struct RecordingTransport {
        recv: VecDeque<Value>,
        sent: StdArc<StdMutex<Vec<Value>>>,
    }

    #[async_trait]
    impl McpTransport for RecordingTransport {
        async fn send(&mut self, message: McpFrame) -> Result<(), McpError> {
            self.sent.lock().unwrap().push(message.value);
            Ok(())
        }

        async fn recv(&mut self) -> Result<Option<McpFrame>, McpError> {
            Ok(self.recv.pop_front().map(|value| McpFrame { value }))
        }

        async fn close(&mut self) -> Result<(), McpError> {
            Ok(())
        }
    }

    #[derive(Clone)]
    struct RecordingTransportFactory {
        responses: StdArc<StdMutex<VecDeque<Vec<Value>>>>,
        sent: StdArc<StdMutex<Vec<Value>>>,
    }

    impl RecordingTransportFactory {
        fn new(sequences: Vec<Vec<Value>>) -> Self {
            Self {
                responses: StdArc::new(StdMutex::new(sequences.into())),
                sent: StdArc::new(StdMutex::new(Vec::new())),
            }
        }

        fn sent(&self) -> Vec<Value> {
            self.sent.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl McpTransportFactory for RecordingTransportFactory {
        async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError> {
            let responses = self.responses.lock().unwrap().pop_front().ok_or_else(|| {
                McpError::Transport("no recording transport responses left".into())
            })?;
            Ok(Box::new(RecordingTransport {
                recv: responses.into(),
                sent: self.sent.clone(),
            }))
        }
    }

    #[tokio::test]
    async fn connection_includes_resolved_auth_in_future_requests() {
        let factory = RecordingTransportFactory::new(vec![vec![
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "protocolVersion": "2025-11-25", "capabilities": { "tools": {}, "resources": {}, "prompts": {} }, "serverInfo": { "name": "recording", "version": "1.0.0" } } }),
            json!({ "jsonrpc": "2.0", "id": 1, "result": { "content": [{ "type": "text", "text": "ok" }] } }),
        ]]);
        let config = McpServerConfig::new(
            "recording",
            McpTransportBinding::Custom(Arc::new(factory.clone())),
        );
        let connection = McpConnection::connect(&config).await.unwrap();
        let mut auth = MetadataMap::new();
        auth.insert("token".into(), json!("secret-token"));
        let request = AuthRequest {
            task_id: None,
            id: "auth-recording-tool".into(),
            provider: "mcp.recording".into(),
            operation: AuthOperation::McpToolCall {
                server_id: "recording".into(),
                tool_name: "echo".into(),
                input: json!({}),
                metadata: MetadataMap::new(),
            },
            challenge: MetadataMap::new(),
        };
        connection
            .resolve_auth(agentkit_tools_core::AuthResolution::Provided {
                request,
                credentials: auth,
            })
            .await
            .unwrap();

        let _ = connection.call_tool("echo", json!({})).await.unwrap();
        let sent = factory.sent();
        assert!(
            sent.iter().any(|value| {
                value
                    .get("params")
                    .and_then(|params| params.get("auth"))
                    .and_then(|auth| auth.get("token"))
                    == Some(&json!("secret-token"))
            }),
            "expected an MCP request to include the resolved auth payload, saw {:?}",
            sent
        );
    }

    #[tokio::test]
    async fn manager_reuses_stored_auth_on_connect() {
        let factory = RecordingTransportFactory::new(vec![vec![
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "protocolVersion": "2025-11-25", "capabilities": { "tools": {}, "resources": {}, "prompts": {} }, "serverInfo": { "name": "recording", "version": "1.0.0" } } }),
            json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [] } }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [] } }),
            json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [] } }),
        ]]);
        let server_id = McpServerId::new("recording");
        let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
            server_id.to_string(),
            McpTransportBinding::Custom(Arc::new(factory.clone())),
        ));
        let mut auth = MetadataMap::new();
        auth.insert("token".into(), json!("seed-token"));
        let request = AuthRequest {
            task_id: None,
            id: "auth-recording-connect".into(),
            provider: "mcp.recording".into(),
            operation: AuthOperation::McpConnect {
                server_id: server_id.to_string(),
                metadata: MetadataMap::new(),
            },
            challenge: MetadataMap::new(),
        };
        manager
            .resolve_auth(agentkit_tools_core::AuthResolution::Provided {
                request,
                credentials: auth,
            })
            .await
            .unwrap();

        manager.connect_server(&server_id).await.unwrap();
        let sent = factory.sent();
        assert!(
            sent.iter().any(|value| {
                value.get("method").and_then(Value::as_str) == Some("initialize")
                    && value
                        .get("params")
                        .and_then(|params| params.get("auth"))
                        .and_then(|auth| auth.get("token"))
                        == Some(&json!("seed-token"))
            }),
            "expected initialize to include stored auth, saw {:?}",
            sent
        );
    }

    #[tokio::test]
    async fn manager_resolves_auth_and_replays_resource_read() {
        let factory = RecordingTransportFactory::new(vec![vec![
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "protocolVersion": "2025-11-25", "capabilities": { "tools": {}, "resources": {}, "prompts": {} }, "serverInfo": { "name": "recording", "version": "1.0.0" } } }),
            json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [] } }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [] } }),
            json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [] } }),
            json!({
                "jsonrpc": "2.0",
                "id": 4,
                "result": {
                    "contents": [
                        {
                            "uri": "file:///tmp/secret.txt",
                            "text": "secret from resource"
                        }
                    ]
                }
            }),
        ]]);
        let server_id = McpServerId::new("recording");
        let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
            server_id.to_string(),
            McpTransportBinding::Custom(Arc::new(factory.clone())),
        ));
        let mut auth = MetadataMap::new();
        auth.insert("token".into(), json!("resource-token"));
        let request = AuthRequest {
            task_id: None,
            id: "auth-recording-resource".into(),
            provider: "mcp.recording".into(),
            operation: AuthOperation::McpResourceRead {
                server_id: server_id.to_string(),
                resource_id: "file:///tmp/secret.txt".into(),
                metadata: MetadataMap::new(),
            },
            challenge: MetadataMap::new(),
        };

        let result = manager
            .resolve_auth_and_resume(agentkit_tools_core::AuthResolution::Provided {
                request,
                credentials: auth,
            })
            .await
            .unwrap();

        match result {
            McpOperationResult::Resource(contents) => {
                assert_eq!(
                    contents.data,
                    DataRef::InlineText("secret from resource".into())
                );
            }
            other => panic!("unexpected replay result: {other:?}"),
        }

        let sent = factory.sent();
        assert!(
            sent.iter().any(|value| {
                value.get("method").and_then(Value::as_str) == Some("resources/read")
                    && value
                        .get("params")
                        .and_then(|params| params.get("auth"))
                        .and_then(|auth| auth.get("token"))
                        == Some(&json!("resource-token"))
            }),
            "expected resources/read to include resolved auth, saw {:?}",
            sent
        );
    }

    #[tokio::test]
    async fn manager_resolves_auth_and_replays_connect() {
        let factory = RecordingTransportFactory::new(vec![vec![
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "protocolVersion": "2025-11-25", "capabilities": { "tools": {}, "resources": {}, "prompts": {} }, "serverInfo": { "name": "recording", "version": "1.0.0" } } }),
            json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [] } }),
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [] } }),
            json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [] } }),
        ]]);
        let server_id = McpServerId::new("recording");
        let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
            server_id.to_string(),
            McpTransportBinding::Custom(Arc::new(factory.clone())),
        ));
        let mut auth = MetadataMap::new();
        auth.insert("token".into(), json!("connect-token"));
        let request = AuthRequest {
            task_id: None,
            id: "auth-recording-connect-replay".into(),
            provider: "mcp.recording".into(),
            operation: AuthOperation::McpConnect {
                server_id: server_id.to_string(),
                metadata: MetadataMap::new(),
            },
            challenge: MetadataMap::new(),
        };

        let result = manager
            .resolve_auth_and_resume(agentkit_tools_core::AuthResolution::Provided {
                request,
                credentials: auth,
            })
            .await
            .unwrap();

        match result {
            McpOperationResult::Connected(snapshot) => {
                assert_eq!(snapshot.server_id, server_id);
            }
            other => panic!("unexpected replay result: {other:?}"),
        }
    }

    #[cfg(feature = "reqwest-client")]
    #[tokio::test]
    async fn sse_transport_posts_messages_and_receives_frames() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = StdArc::new(StdMutex::new(Vec::new()));
        let captured = requests.clone();

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0_u8; 4096];
                let read = socket.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();

                if request.starts_with("GET /sse ") {
                    let body = concat!(
                        "event: endpoint\n",
                        "data: /messages\n\n",
                        "event: message\n",
                        "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n"
                    );
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    socket.write_all(response.as_bytes()).await.unwrap();
                } else {
                    captured.lock().unwrap().push(request);
                    socket
                        .write_all(
                            b"HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                }
            }
        });

        let factory =
            SseTransportFactory::new(SseTransportConfig::new(format!("http://{address}/sse")));
        let mut transport = factory.connect().await.unwrap();
        transport
            .send(McpFrame {
                value: json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                }),
            })
            .await
            .unwrap();
        let frame = transport.recv().await.unwrap().unwrap();
        transport.close().await.unwrap();
        server.await.unwrap();

        assert_eq!(frame.value["result"]["tools"], json!([]));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /messages "));
        assert!(requests[0].contains("\"method\":\"tools/list\""));
    }

    #[cfg(feature = "reqwest-client")]
    #[tokio::test]
    async fn streamable_http_connection_tracks_session_and_protocol_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = StdArc::new(StdMutex::new(Vec::new()));
        let captured = requests.clone();

        let server = tokio::spawn(async move {
            for _ in 0..5 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0_u8; 8192];
                let read = socket.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                captured.lock().unwrap().push(request.clone());

                let response = if request.contains("\"method\":\"initialize\"") {
                    let body = "{\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"serverInfo\":{\"name\":\"remote\",\"version\":\"1.0.0\"}}}";
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nMCP-Session-Id: session-123\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else if request.contains("\"method\":\"notifications/initialized\"") {
                    "HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        .to_string()
                } else if request.starts_with("DELETE /mcp ") {
                    "HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        .to_string()
                } else {
                    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}";
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                };

                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let config = McpServerConfig::new(
            "remote",
            McpTransportBinding::StreamableHttp(StreamableHttpTransportConfig::new(format!(
                "http://{address}/mcp"
            ))),
        );
        let connection = McpConnection::connect(&config).await.unwrap();
        let _ = connection.list_tools().await.unwrap();
        connection.close().await.unwrap();
        server.await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 5);
        let normalized = requests
            .iter()
            .map(|request| request.to_ascii_lowercase())
            .collect::<Vec<_>>();
        assert!(requests[0].starts_with("POST /mcp "));
        assert!(!requests[0].contains("MCP-Session-Id:"));
        assert!(normalized[1].contains("mcp-session-id: session-123"));
        assert!(normalized[1].contains("mcp-protocol-version: 2025-11-25"));
        assert!(normalized[2].contains("mcp-session-id: session-123"));
        assert!(normalized[2].contains("mcp-protocol-version: 2025-11-25"));
        let delete = requests
            .iter()
            .zip(normalized.iter())
            .find(|(request, _)| request.starts_with("DELETE /mcp "))
            .expect("expected RMCP transport to delete the Streamable HTTP session on close");
        assert!(delete.1.contains("mcp-session-id: session-123"));
    }

    #[cfg(feature = "reqwest-client")]
    #[tokio::test]
    async fn streamable_http_transport_resumes_sse_streams_until_response_arrives() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let requests = StdArc::new(StdMutex::new(Vec::new()));
        let captured = requests.clone();

        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0_u8; 8192];
                let read = socket.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                captured.lock().unwrap().push(request.clone());

                let response = if request.starts_with("POST /mcp ") {
                    let body = concat!(
                        "id: evt-1\n",
                        "event: message\n",
                        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"phase\":\"stream-start\"}}\n\n"
                    );
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else {
                    let body = concat!(
                        "id: evt-2\n",
                        "event: message\n",
                        "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n"
                    );
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                };

                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let factory = StreamableHttpTransportFactory::new(StreamableHttpTransportConfig::new(
            format!("http://{address}/mcp"),
        ));
        let mut transport = factory.connect().await.unwrap();
        transport
            .send(McpFrame {
                value: json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                }),
            })
            .await
            .unwrap();

        let first = transport.recv().await.unwrap().unwrap();
        let second = transport.recv().await.unwrap().unwrap();
        transport.close().await.unwrap();
        server.await.unwrap();

        assert_eq!(
            first.value["method"],
            Value::String("notifications/message".into())
        );
        assert_eq!(second.value["result"]["tools"], json!([]));

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /mcp "));
        assert!(requests[1].starts_with("GET /mcp "));
        assert!(
            requests[1].contains("last-event-id: evt-1")
                || requests[1].contains("Last-Event-ID: evt-1")
        );
    }

    #[tokio::test]
    async fn server_manager_connects_refreshes_and_aggregates_tools() {
        let alpha = McpServerConfig::new(
            "alpha",
            McpTransportBinding::Custom(Arc::new(FakeTransportFactory::new(vec![vec![
                json!({ "jsonrpc": "2.0", "id": 0, "result": { "protocolVersion": "2025-11-25", "capabilities": { "tools": {}, "resources": {}, "prompts": {} }, "serverInfo": { "name": "alpha", "version": "1.0.0" } } }),
                json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [{ "name": "echo", "description": "Echo", "inputSchema": {"type": "object"} }] } }),
                json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [] } }),
                json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [] } }),
                json!({ "jsonrpc": "2.0", "id": 4, "result": { "tools": [{ "name": "echo_v2", "description": "Echo 2", "inputSchema": {"type": "object"} }] } }),
                json!({ "jsonrpc": "2.0", "id": 5, "result": { "resources": [] } }),
                json!({ "jsonrpc": "2.0", "id": 6, "result": { "prompts": [] } }),
            ]]))),
        );
        let beta = McpServerConfig::new(
            "beta",
            McpTransportBinding::Custom(Arc::new(FakeTransportFactory::new(vec![vec![
                json!({ "jsonrpc": "2.0", "id": 0, "result": { "protocolVersion": "2025-11-25", "capabilities": { "tools": {}, "resources": {}, "prompts": {} }, "serverInfo": { "name": "beta", "version": "1.0.0" } } }),
                json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [{ "name": "search", "description": "Search", "inputSchema": {"type": "object"} }] } }),
                json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [] } }),
                json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [] } }),
            ]]))),
        );

        let mut manager = McpServerManager::new().with_server(alpha).with_server(beta);
        let mut catalog_events = manager.subscribe_catalog_events();

        let handles = manager.connect_all().await.unwrap();
        assert_eq!(handles.len(), 2);
        assert!(matches!(
            catalog_events.recv().await.unwrap(),
            McpCatalogEvent::ServerConnected { ref server_id } if server_id == &McpServerId::new("alpha")
        ));
        assert!(matches!(
            catalog_events.recv().await.unwrap(),
            McpCatalogEvent::ServerConnected { ref server_id } if server_id == &McpServerId::new("beta")
        ));
        assert_eq!(
            manager
                .tool_registry()
                .specs()
                .into_iter()
                .map(|spec| spec.name.0)
                .collect::<Vec<_>>(),
            vec!["mcp_alpha_echo".to_string(), "mcp_beta_search".to_string()]
        );

        let refreshed = manager
            .refresh_server(&McpServerId::new("alpha"))
            .await
            .unwrap();
        assert_eq!(refreshed.tools[0].name, "echo_v2");
        assert!(matches!(
            catalog_events.recv().await.unwrap(),
            McpCatalogEvent::ToolsChanged {
                ref server_id,
                ref added,
                ref removed,
                ref changed,
            } if server_id == &McpServerId::new("alpha")
                && added == &vec!["echo_v2".to_string()]
                && removed == &vec!["echo".to_string()]
                && changed.is_empty()
        ));
        assert_eq!(
            manager
                .connected_server(&McpServerId::new("alpha"))
                .unwrap()
                .snapshot()
                .tools[0]
                .name,
            "echo_v2"
        );

        let capabilities = manager.capability_provider();
        assert_eq!(capabilities.invocables().len(), 2);

        manager
            .disconnect_server(&McpServerId::new("alpha"))
            .await
            .unwrap();
        assert!(matches!(
            catalog_events.recv().await.unwrap(),
            McpCatalogEvent::ServerDisconnected { ref server_id } if server_id == &McpServerId::new("alpha")
        ));
        assert!(
            manager
                .connected_server(&McpServerId::new("alpha"))
                .is_none()
        );
    }
}
