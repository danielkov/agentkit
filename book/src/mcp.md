# MCP

`agentkit-mcp` integrates Model Context Protocol servers into the agent stack.

## Server configuration

```rust
pub struct McpServerConfig {
    pub id: McpServerId,
    pub transport: McpTransportBinding,
    pub auth: McpAuthConfig,
    pub metadata: MetadataMap,
}
```

Built-in transports: stdio, SSE. Hosts can provide custom transport implementations.

## Discovery

After connecting to a server, `agentkit-mcp` produces a discovery snapshot:

```rust
pub struct McpDiscoverySnapshot {
    pub server_id: McpServerId,
    pub tools: Vec<McpToolDescriptor>,
    pub resources: Vec<McpResourceDescriptor>,
    pub prompts: Vec<McpPromptDescriptor>,
    pub metadata: MetadataMap,
}
```

## Tool adaptation

MCP tools are adapted into the shared tool system via `McpToolAdapter`. They appear as ordinary `ToolSpec`s and execute through the same `ToolExecutor` path as native tools.

Default naming convention: `mcp.<server_id>.<tool_name>`

## Resources and prompts

Resources and prompts are **not** tools. They have dedicated APIs:

```rust
pub trait McpResourceStore {
    async fn list_resources(&self, server: &McpServerId) -> Result<Vec<McpResourceDescriptor>, McpError>;
    async fn read_resource(&self, server: &McpServerId, resource: &McpResourceId) -> Result<McpResourceContents, McpError>;
}

pub trait McpPromptStore {
    async fn list_prompts(&self, server: &McpServerId) -> Result<Vec<McpPromptDescriptor>, McpError>;
    async fn get_prompt(&self, server: &McpServerId, prompt: &McpPromptId, args: serde_json::Value) -> Result<McpPromptContents, McpError>;
}
```

## Auth model

Auth is an explicit interruption, not hidden retry logic:

- Tool invocation may interrupt with `AuthRequired`
- Server/session startup may interrupt with `AuthRequired`
- The host resolves the auth flow
- The operation resumes

MCP auth fits into the loop interrupt system via `ToolInterruption::AuthRequired`.
