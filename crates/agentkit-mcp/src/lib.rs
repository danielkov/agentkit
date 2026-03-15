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
    AuthRequest, Tool, ToolAnnotations, ToolContext, ToolError, ToolName, ToolRegistry,
    ToolRequest, ToolResult, ToolSpec,
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

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct McpServerId(pub String);

impl McpServerId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for McpServerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StdioTransportConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<std::path::PathBuf>,
}

impl StdioTransportConfig {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
        }
    }

    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    pub fn with_cwd(mut self, cwd: impl Into<std::path::PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseTransportConfig {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

impl SseTransportConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }
}

#[derive(Clone)]
pub enum McpTransportBinding {
    Stdio(StdioTransportConfig),
    Sse(SseTransportConfig),
    Custom(Arc<dyn McpTransportFactory>),
}

#[derive(Clone)]
pub struct McpServerConfig {
    pub id: McpServerId,
    pub transport: McpTransportBinding,
    pub metadata: MetadataMap,
}

impl McpServerConfig {
    pub fn new(id: impl Into<String>, transport: McpTransportBinding) -> Self {
        Self {
            id: McpServerId::new(id),
            transport,
            metadata: MetadataMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpFrame {
    pub value: Value,
}

#[async_trait]
pub trait McpTransportFactory: Send + Sync {
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError>;
}

#[async_trait]
pub trait McpTransport: Send + Sync {
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError>;
    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError>;
    async fn close(&mut self) -> Result<(), McpError>;
}

pub struct StdioTransportFactory {
    config: StdioTransportConfig,
}

impl StdioTransportFactory {
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

pub struct SseTransportFactory {
    config: SseTransportConfig,
}

impl SseTransportFactory {
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceDescriptor {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpPromptDescriptor {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpDiscoverySnapshot {
    pub server_id: McpServerId,
    pub tools: Vec<McpToolDescriptor>,
    pub resources: Vec<McpResourceDescriptor>,
    pub prompts: Vec<McpPromptDescriptor>,
    pub metadata: MetadataMap,
}

pub struct McpConnection {
    server_id: McpServerId,
    transport: Mutex<Box<dyn McpTransport>>,
    auth: Mutex<Option<MetadataMap>>,
    next_id: AtomicU64,
}

impl McpConnection {
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
        transport
            .send(McpFrame {
                value: json!({
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": Value::Object(params)
                }),
            })
            .await?;
        let init_response = transport.recv().await?.ok_or_else(|| {
            McpError::Transport("transport closed during MCP initialization".into())
        })?;
        if let Some(error) = init_response.value.get("error") {
            if let Some(auth_request) = parse_auth_request(&config.id, "initialize", error) {
                return Err(McpError::AuthRequired(auth_request));
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

    pub fn server_id(&self) -> &McpServerId {
        &self.server_id
    }

    pub async fn close(&self) -> Result<(), McpError> {
        let mut transport = self.transport.lock().await;
        transport.close().await
    }

    pub async fn resolve_auth(
        &self,
        resolution: agentkit_tools_core::AuthResolution,
    ) -> Result<(), McpError> {
        let mut auth = self.auth.lock().await;
        match resolution {
            agentkit_tools_core::AuthResolution::Provided(details) => {
                *auth = Some(details);
            }
            agentkit_tools_core::AuthResolution::Cancelled => {
                *auth = None;
            }
        }
        Ok(())
    }

    pub async fn discover(&self) -> Result<McpDiscoverySnapshot, McpError> {
        Ok(McpDiscoverySnapshot {
            server_id: self.server_id.clone(),
            tools: self.list_tools().await?,
            resources: self.list_resources().await?,
            prompts: self.list_prompts().await?,
            metadata: MetadataMap::new(),
        })
    }

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
        let params = self.enrich_params(params).await;
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
                if let Some(auth_request) = parse_auth_request(&self.server_id, method, error) {
                    return Err(McpError::AuthRequired(auth_request));
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
}

pub struct McpInvocable {
    connection: Arc<McpConnection>,
    descriptor: McpToolDescriptor,
    spec: InvocableSpec,
}

impl McpInvocable {
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

pub struct McpCapabilityProvider {
    invocables: Vec<Arc<dyn Invocable>>,
    resources: Vec<Arc<dyn ResourceProvider>>,
    prompts: Vec<Arc<dyn PromptProvider>>,
}

impl McpCapabilityProvider {
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

#[derive(Clone)]
pub struct McpServerHandle {
    config: McpServerConfig,
    connection: Arc<McpConnection>,
    snapshot: McpDiscoverySnapshot,
}

impl McpServerHandle {
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    pub fn server_id(&self) -> &McpServerId {
        self.connection.server_id()
    }

    pub fn connection(&self) -> Arc<McpConnection> {
        self.connection.clone()
    }

    pub fn snapshot(&self) -> &McpDiscoverySnapshot {
        &self.snapshot
    }

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

    pub fn capability_provider(&self) -> McpCapabilityProvider {
        McpCapabilityProvider::from_snapshot(self.connection.clone(), &self.snapshot)
    }
}

#[derive(Default)]
pub struct McpServerManager {
    configs: BTreeMap<McpServerId, McpServerConfig>,
    connections: BTreeMap<McpServerId, McpServerHandle>,
    auth: BTreeMap<McpServerId, MetadataMap>,
}

impl McpServerManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_server(mut self, config: McpServerConfig) -> Self {
        self.register_server(config);
        self
    }

    pub fn register_server(&mut self, config: McpServerConfig) -> &mut Self {
        self.configs.insert(config.id.clone(), config);
        self
    }

    pub fn connected_server(&self, server_id: &McpServerId) -> Option<&McpServerHandle> {
        self.connections.get(server_id)
    }

    pub fn connected_servers(&self) -> Vec<&McpServerHandle> {
        self.connections.values().collect()
    }

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

    pub async fn connect_all(&mut self) -> Result<Vec<McpServerHandle>, McpError> {
        let server_ids = self.configs.keys().cloned().collect::<Vec<_>>();
        let mut handles = Vec::with_capacity(server_ids.len());

        for server_id in server_ids {
            handles.push(self.connect_server(&server_id).await?);
        }

        Ok(handles)
    }

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

    pub async fn disconnect_server(&mut self, server_id: &McpServerId) -> Result<(), McpError> {
        let Some(handle) = self.connections.remove(server_id) else {
            return Err(McpError::UnknownServer(server_id.to_string()));
        };
        handle.connection.close().await
    }

    pub async fn resolve_auth(
        &mut self,
        server_id: &McpServerId,
        resolution: agentkit_tools_core::AuthResolution,
    ) -> Result<(), McpError> {
        match &resolution {
            agentkit_tools_core::AuthResolution::Provided(details) => {
                self.auth.insert(server_id.clone(), details.clone());
            }
            agentkit_tools_core::AuthResolution::Cancelled => {
                self.auth.remove(server_id);
            }
        }

        if let Some(handle) = self.connections.get(server_id) {
            handle.connection.resolve_auth(resolution).await?;
            return Ok(());
        }

        if self.configs.contains_key(server_id) {
            Ok(())
        } else {
            Err(McpError::UnknownServer(server_id.to_string()))
        }
    }

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

    pub fn capability_provider(&self) -> McpCapabilityProvider {
        McpCapabilityProvider::merge(
            self.connections
                .values()
                .map(McpServerHandle::capability_provider),
        )
    }
}

pub struct McpToolAdapter {
    descriptor: McpToolDescriptor,
    connection: Arc<McpConnection>,
    spec: ToolSpec,
}

impl McpToolAdapter {
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

fn parse_auth_request(server_id: &McpServerId, method: &str, error: &Value) -> Option<AuthRequest> {
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

    let mut details = MetadataMap::new();
    details.insert("server_id".into(), Value::String(server_id.to_string()));
    details.insert("method".into(), Value::String(method.into()));

    if let Some(code) = code {
        details.insert("code".into(), Value::Number(code.into()));
    }
    if let Some(message) = message {
        details.insert("message".into(), Value::String(message.into()));
    }
    if let Some(data) = data {
        details.insert("data".into(), data.clone());
    }

    Some(AuthRequest {
        provider: format!("mcp.{}", server_id),
        details,
    })
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

#[derive(Debug, Error)]
pub enum McpError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("MCP auth required: {0:?}")]
    AuthRequired(AuthRequest),
    #[error("invocation error: {0}")]
    Invocation(String),
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
                    request.details.get("method"),
                    Some(&Value::String("tools/call".into()))
                );
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
        connection
            .resolve_auth(agentkit_tools_core::AuthResolution::Provided(auth))
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
        manager
            .resolve_auth(
                &server_id,
                agentkit_tools_core::AuthResolution::Provided(auth),
            )
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
