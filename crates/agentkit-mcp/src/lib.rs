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
    Tool, ToolAnnotations, ToolContext, ToolError, ToolName, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

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

#[derive(Clone)]
pub enum McpTransportBinding {
    Stdio(StdioTransportConfig),
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

struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
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
    next_id: AtomicU64,
}

impl McpConnection {
    pub async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        let factory: Arc<dyn McpTransportFactory> = match &config.transport {
            McpTransportBinding::Stdio(binding) => {
                Arc::new(StdioTransportFactory::new(binding.clone()))
            }
            McpTransportBinding::Custom(factory) => factory.clone(),
        };

        let mut transport = factory.connect().await?;
        transport
            .send(McpFrame {
                value: json!({
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "agentkit-mcp",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }
                }),
            })
            .await?;
        let _ = transport.recv().await?;
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
            next_id: AtomicU64::new(1),
        })
    }

    pub fn server_id(&self) -> &McpServerId {
        &self.server_id
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
                return Err(McpError::Invocation(error.to_string()));
            }

            return frame
                .value
                .get("result")
                .cloned()
                .ok_or_else(|| McpError::Protocol("MCP response missing result".into()));
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
            .map_err(|error| CapabilityError::ExecutionFailed(error.to_string()))?;

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
            .map_err(|error| CapabilityError::ExecutionFailed(error.to_string()))
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
            .map_err(|error| CapabilityError::ExecutionFailed(error.to_string()))
    }
}

pub struct McpCapabilityProvider {
    invocables: Vec<Arc<dyn Invocable>>,
    resources: Vec<Arc<dyn ResourceProvider>>,
    prompts: Vec<Arc<dyn PromptProvider>>,
}

impl McpCapabilityProvider {
    pub async fn connect(
        config: &McpServerConfig,
    ) -> Result<(Arc<McpConnection>, Self, McpDiscoverySnapshot), McpError> {
        let connection = Arc::new(McpConnection::connect(config).await?);
        let snapshot = connection.discover().await?;

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

        Ok((
            connection,
            Self {
                invocables,
                resources,
                prompts,
            },
            snapshot,
        ))
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
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;

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

#[derive(Debug, Error)]
pub enum McpError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("invocation error: {0}")]
    Invocation(String),
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use agentkit_tools_core::{PermissionChecker, PermissionDecision, PermissionRequest};

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
            next_id: AtomicU64::new(1),
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
}
