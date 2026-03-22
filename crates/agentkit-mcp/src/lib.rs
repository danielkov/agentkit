use std::collections::BTreeMap;
use std::fmt;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agentkit_capabilities::{
    CapabilityContext, CapabilityError, CapabilityName, CapabilityProvider, Invocable,
    InvocableOutput, InvocableRequest, InvocableResult, InvocableSpec, PromptContents,
    PromptDescriptor, PromptId, PromptProvider, ResourceContents, ResourceDescriptor, ResourceId,
    ResourceProvider,
};
use agentkit_core::{
    DataRef, Item, ItemKind, MetadataMap, Part, TextPart, ToolOutput, ToolResultPart,
};
use agentkit_tools_core::{
    AuthOperation, AuthRequest, AuthResolution, Tool, ToolAnnotations, ToolContext, ToolError,
    ToolName, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::io::StreamReader;

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
/// # Example
///
/// ```rust
/// use agentkit_mcp::SseTransportConfig;
///
/// let config = SseTransportConfig::new("https://mcp.example.com/sse")
///     .with_header("Authorization", "Bearer tok_abc123");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseTransportConfig {
    /// The SSE endpoint URL to connect to.
    pub url: String,
    /// Additional HTTP headers sent with every request (e.g. authentication tokens).
    pub headers: Vec<(String, String)>,
}

impl SseTransportConfig {
    /// Creates a new SSE transport configuration for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: Vec::new(),
        }
    }

    /// Adds an HTTP header to include with every request. Returns `self` for chaining.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }
}

/// Selects which transport an MCP server should use.
///
/// This enum is passed into [`McpServerConfig`] and determines how the client will
/// communicate with the MCP server. The two built-in options are [`Stdio`](Self::Stdio)
/// and [`Sse`](Self::Sse); use [`Custom`](Self::Custom) to provide your own
/// [`McpTransportFactory`].
#[derive(Clone)]
pub enum McpTransportBinding {
    /// Communicate over the child process's stdin/stdout.
    Stdio(StdioTransportConfig),
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

#[async_trait]
impl McpTransportFactory for SseTransportFactory {
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError> {
        let client = Client::builder()
            .user_agent("agentkit-mcp/0.1.0")
            .build()
            .map_err(McpError::Http)?;

        let mut request = client
            .get(&self.config.url)
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache");

        for (key, value) in &self.config.headers {
            request = request.header(key, value);
        }

        let response = request.send().await.map_err(McpError::Http)?;
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

        let response_url = response.url().clone();
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
            headers: self.config.headers.clone(),
            frame_rx,
            read_task,
        }))
    }
}

struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct SseTransport {
    client: Client,
    endpoint_url: Url,
    headers: Vec<(String, String)>,
    frame_rx: mpsc::UnboundedReceiver<Result<McpFrame, McpError>>,
    read_task: JoinHandle<()>,
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
        let mut request = self
            .client
            .post(self.endpoint_url.clone())
            .header("Content-Type", "application/json");

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        let response = request
            .json(&message.value)
            .send()
            .await
            .map_err(McpError::Http)?;
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

/// A live connection to a single MCP server.
///
/// Handles JSON-RPC request/response framing, automatic auth enrichment, and
/// high-level methods for tool calls, resource reads, prompt retrieval, and
/// capability discovery.
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
    transport: Mutex<Box<dyn McpTransport>>,
    auth: Mutex<Option<MetadataMap>>,
    next_id: AtomicU64,
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
        let factory: Arc<dyn McpTransportFactory> = match &config.transport {
            McpTransportBinding::Stdio(binding) => {
                Arc::new(StdioTransportFactory::new(binding.clone()))
            }
            McpTransportBinding::Sse(binding) => {
                Arc::new(SseTransportFactory::new(binding.clone()))
            }
            McpTransportBinding::Custom(factory) => factory.clone(),
        };

        let mut transport = factory.connect().await?;
        let mut params = serde_json::Map::new();
        params.insert("protocolVersion".into(), Value::String("2024-11-05".into()));
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
        let init_response = transport.recv().await?.ok_or_else(|| {
            McpError::Transport("transport closed during MCP initialization".into())
        })?;
        if let Some(error) = init_response.value.get("error") {
            if let Some(auth_request) =
                parse_auth_request(&config.id, "initialize", &init_params, error)
            {
                return Err(McpError::AuthRequired(Box::new(auth_request)));
            }
            return Err(McpError::Invocation(error.to_string()));
        }
        transport
            .send(McpFrame {
                value: json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }),
            })
            .await?;

        Ok(Self {
            server_id: config.id.clone(),
            transport: Mutex::new(transport),
            auth: Mutex::new(auth.cloned()),
            next_id: AtomicU64::new(1),
        })
    }

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
        let mut transport = self.transport.lock().await;
        transport.close().await
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
        Ok(())
    }

    /// Performs full capability discovery by listing tools, resources, and prompts.
    ///
    /// Returns an [`McpDiscoverySnapshot`] containing everything the server advertises.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if any of the list requests fail.
    pub async fn discover(&self) -> Result<McpDiscoverySnapshot, McpError> {
        Ok(McpDiscoverySnapshot {
            server_id: self.server_id.clone(),
            tools: self.list_tools().await?,
            resources: self.list_resources().await?,
            prompts: self.list_prompts().await?,
            metadata: MetadataMap::new(),
        })
    }

    /// Lists all tools advertised by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the `tools/list` request fails.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError> {
        let result = self.request("tools/list", json!({})).await?;
        result
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(parse_tool_descriptor)
            .collect()
    }

    /// Lists all resources advertised by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the `resources/list` request fails.
    pub async fn list_resources(&self) -> Result<Vec<McpResourceDescriptor>, McpError> {
        let result = self.request("resources/list", json!({})).await?;
        result
            .get("resources")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(parse_resource_descriptor)
            .collect()
    }

    /// Lists all prompts advertised by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError`] if the `prompts/list` request fails.
    pub async fn list_prompts(&self) -> Result<Vec<McpPromptDescriptor>, McpError> {
        let result = self.request("prompts/list", json!({})).await?;
        result
            .get("prompts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(parse_prompt_descriptor)
            .collect()
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
        self.request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments,
            }),
        )
        .await
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
        let result = self
            .request(
                "resources/read",
                json!({
                    "uri": uri,
                }),
            )
            .await?;
        let content = result
            .get("contents")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
            .cloned()
            .ok_or_else(|| McpError::Protocol("resources/read returned no contents".into()))?;

        let data = if let Some(text) = content.get("text").and_then(Value::as_str) {
            DataRef::InlineText(text.into())
        } else if let Some(found_uri) = content.get("uri").and_then(Value::as_str) {
            DataRef::Uri(found_uri.into())
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
        let result = self
            .request(
                "prompts/get",
                json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;
        let items = result
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(parse_prompt_message)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(PromptContents {
            items,
            metadata: MetadataMap::new(),
        })
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let params = self.enrich_params(params.clone()).await;
        let mut transport = self.transport.lock().await;
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
/// [`McpToolAdapter`]. Names are prefixed with `mcp.<server_id>.<tool_name>`.
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
                "mcp.{}.{}",
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
#[derive(Default)]
pub struct McpServerManager {
    configs: BTreeMap<McpServerId, McpServerConfig>,
    connections: BTreeMap<McpServerId, McpServerHandle>,
    auth: BTreeMap<McpServerId, MetadataMap>,
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
        let snapshot = handle.connection.discover().await?;
        handle.snapshot = snapshot.clone();
        Ok(snapshot)
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
        handle.connection.close().await
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
            return Ok(());
        }

        if self.configs.contains_key(&server_id) {
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
    /// Tool names are prefixed as `mcp.<server_id>.<tool_name>`.
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
}

/// Adapter that exposes an MCP tool as an agentkit [`Tool`].
///
/// This is the tool-layer adapter for the tool registry. For the capabilities-layer
/// adapter, see [`McpInvocable`]. Tool names are prefixed as
/// `mcp.<server_id>.<tool_name>`.
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
            name: ToolName::new(format!("mcp.{}.{}", server_id, descriptor.name)),
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
    let prefix = format!("mcp.{server_id}.");
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
    let mut event_name: Option<String> = None;
    let mut data_lines = Vec::new();

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() {
                    dispatch_sse_event(
                        &response_url,
                        &mut endpoint_tx,
                        &frame_tx,
                        event_name.take(),
                        std::mem::take(&mut data_lines),
                    );
                    continue;
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
                }
            }
            Err(error) => {
                let error = McpError::Io(error);
                if let Some(tx) = endpoint_tx.take() {
                    let _ = tx.send(Err(error));
                } else {
                    let _ = frame_tx.send(Err(error));
                }
                return;
            }
        }
    }

    if event_name.is_some() || !data_lines.is_empty() {
        dispatch_sse_event(
            &response_url,
            &mut endpoint_tx,
            &frame_tx,
            event_name.take(),
            std::mem::take(&mut data_lines),
        );
    }

    if let Some(tx) = endpoint_tx.take() {
        let _ = tx.send(Err(McpError::Transport(
            "SSE stream ended before endpoint event".into(),
        )));
    }
}

fn dispatch_sse_event(
    response_url: &Url,
    endpoint_tx: &mut Option<oneshot::Sender<Result<Url, McpError>>>,
    frame_tx: &mpsc::UnboundedSender<Result<McpFrame, McpError>>,
    event_name: Option<String>,
    data_lines: Vec<String>,
) {
    if data_lines.is_empty() {
        return;
    }

    let event_name = event_name.unwrap_or_else(|| "message".into());
    let data = data_lines.join("\n");

    if event_name == "endpoint" {
        if let Some(tx) = endpoint_tx.take() {
            let _ = tx.send(resolve_sse_endpoint(response_url, &data));
        }
        return;
    }

    if event_name != "message" {
        return;
    }

    let value = serde_json::from_str(&data).map_err(McpError::Serialize);
    let _ = frame_tx.send(value.map(|value| McpFrame { value }));
}

fn resolve_sse_endpoint(response_url: &Url, endpoint: &str) -> Result<Url, McpError> {
    response_url
        .join(endpoint.trim())
        .map_err(|error| McpError::Transport(format!("invalid SSE endpoint URL: {error}")))
}

/// Errors produced by MCP transport, protocol, and lifecycle operations.
#[derive(Debug, Error)]
pub enum McpError {
    /// An underlying I/O error (e.g. spawning a child process or reading from a pipe).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// An HTTP-level error from the SSE transport.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
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
        McpConnection {
            server_id: McpServerId::new("fake"),
            transport: Mutex::new(Box::new(FakeTransport {
                recv: responses.into(),
            })),
            auth: Mutex::new(None),
            next_id: AtomicU64::new(1),
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
                    tool_name: ToolName::new("mcp.fake.echo"),
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
                    tool_name: ToolName::new("mcp.fake.echo"),
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
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "capabilities": {} } }),
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
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "capabilities": {} } }),
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
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "capabilities": {} } }),
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
            json!({ "jsonrpc": "2.0", "id": 0, "result": { "capabilities": {} } }),
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

    #[tokio::test]
    async fn server_manager_connects_refreshes_and_aggregates_tools() {
        let alpha = McpServerConfig::new(
            "alpha",
            McpTransportBinding::Custom(Arc::new(FakeTransportFactory::new(vec![vec![
                json!({ "jsonrpc": "2.0", "id": 0, "result": { "capabilities": {} } }),
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
                json!({ "jsonrpc": "2.0", "id": 0, "result": { "capabilities": {} } }),
                json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [{ "name": "search", "description": "Search", "inputSchema": {"type": "object"} }] } }),
                json!({ "jsonrpc": "2.0", "id": 2, "result": { "resources": [] } }),
                json!({ "jsonrpc": "2.0", "id": 3, "result": { "prompts": [] } }),
            ]]))),
        );

        let mut manager = McpServerManager::new().with_server(alpha).with_server(beta);

        let handles = manager.connect_all().await.unwrap();
        assert_eq!(handles.len(), 2);
        assert_eq!(
            manager
                .tool_registry()
                .specs()
                .into_iter()
                .map(|spec| spec.name.0)
                .collect::<Vec<_>>(),
            vec!["mcp.alpha.echo".to_string(), "mcp.beta.search".to_string()]
        );

        let refreshed = manager
            .refresh_server(&McpServerId::new("alpha"))
            .await
            .unwrap();
        assert_eq!(refreshed.tools[0].name, "echo_v2");
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
        assert!(
            manager
                .connected_server(&McpServerId::new("alpha"))
                .is_none()
        );
    }
}
