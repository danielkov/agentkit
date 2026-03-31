# MCP integration

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) lets agents discover and use tools, resources, and prompts from external servers. This chapter covers [`agentkit-mcp`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-mcp): how MCP fits into the capability and tool layers, and how auth and lifecycle are managed.

## What MCP solves

Without MCP, every external integration is a custom tool. Connecting to GitHub means writing a GitHub tool. Connecting to a database means writing a database tool. Each one has bespoke connection logic, auth handling, and discovery.

MCP standardizes this: external servers expose capabilities through a uniform protocol, and the agent discovers them at runtime instead of compile time.

```text
Without MCP:                          With MCP:

  Agent                                Agent
  ├── GitHubTool (custom)              ├── MCP client
  ├── DatabaseTool (custom)            │   ├── github-server (discovered)
  ├── SlackTool (custom)               │   ├── database-server (discovered)
  └── JiraTool (custom)                │   └── slack-server (discovered)
                                       │
  Each tool: custom code,              Each server: standard protocol,
  custom auth, custom schema           standard auth, standard schema
```

## MCP in the capability model

MCP servers expose three capability types, which map directly to agentkit's capability layer:

| MCP concept   | agentkit abstraction            | How it's used                              |
| ------------- | ------------------------------- | ------------------------------------------ |
| MCP tools     | `Invocable` → adapted to `Tool` | Model calls them during turns              |
| MCP resources | `ResourceProvider`              | Host reads them for context loading        |
| MCP prompts   | `PromptProvider`                | Host renders them for transcript injection |

An MCP server implements `CapabilityProvider`, exposing all three through one registration point.

## Server configuration

```rust
pub struct McpServerConfig {
    pub id: McpServerId,
    pub transport: McpTransportBinding,
    pub auth: McpAuthConfig,
    pub metadata: MetadataMap,
}
```

Built-in transports: **stdio** (local child process) and **SSE** (HTTP streaming). Custom transports implement the `McpTransportFactory` trait.

## Discovery

After connecting, the server's capabilities are captured in a snapshot:

```rust
pub struct McpDiscoverySnapshot {
    pub server_id: McpServerId,
    pub tools: Vec<McpToolDescriptor>,
    pub resources: Vec<McpResourceDescriptor>,
    pub prompts: Vec<McpPromptDescriptor>,
    pub metadata: MetadataMap,
}
```

Snapshots are cacheable and refreshable. Hosts choose which capabilities to expose — discovery doesn't automatically register everything.

## Tool adaptation

`McpToolAdapter` wraps an MCP tool as a `Tool` implementation:

- Exposes a `ToolSpec` derived from the MCP tool descriptor
- Translates `ToolRequest` into MCP invocation
- Translates MCP responses into normalized `ToolResult`
- Surfaces auth interruptions as `ToolInterruption::AuthRequired`

### Namespacing

MCP tools are namespaced by default: `mcp.<server_id>.<tool_name>`. This prevents collisions with native tools. Hosts can override names if they want a cleaner surface.

## Auth as interruption

MCP auth follows the same interrupt pattern as tool approvals:

1. Tool invocation triggers an auth requirement
2. The tool adapter returns `ToolInterruption::AuthRequired`
3. The loop surfaces it as `LoopStep::Interrupt(AuthRequest)`
4. The host performs the auth flow (OAuth, API key entry, etc.)
5. The host resolves the interrupt and the operation resumes

Auth is never hidden retry logic. The host always knows when auth is happening and controls the flow.

For non-tool MCP operations (connecting, reading resources), auth follows the same pattern but through the MCP manager API rather than the loop interrupt system.

## Resources and prompts

Resources and prompts have dedicated APIs, separate from the tool path:

```rust
pub trait McpResourceStore {
    async fn list_resources(&self, server: &McpServerId) -> ...;
    async fn read_resource(&self, server: &McpServerId, resource: &McpResourceId) -> ...;
}

pub trait McpPromptStore {
    async fn list_prompts(&self, server: &McpServerId) -> ...;
    async fn get_prompt(&self, server: &McpServerId, prompt: &McpPromptId, args: Value) -> ...;
}
```

MCP resources integrate with `agentkit-context` for injecting project-specific data into the transcript. MCP prompts integrate with context loading for template-based prompt assembly.

## Transports

Transport details stay inside `agentkit-mcp`:

```rust
pub trait McpTransport: Send {
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError>;
    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError>;
    async fn close(&mut self) -> Result<(), McpError>;
}
```

Built-in transports:

| Transport | Connection                              | Use case            |
| --------- | --------------------------------------- | ------------------- |
| stdio     | Spawn child process, pipe stdin/stdout  | Local tool servers  |
| SSE       | HTTP connection with server-sent events | Remote tool servers |

The rest of agentkit doesn't know whether a server is reached via stdio, TCP, or WebSocket. The transport is configured in `McpServerConfig` and the MCP manager handles the connection lifecycle.

### stdio transport

The most common pattern for local MCP servers. The agent spawns the server as a child process and communicates over stdin/stdout:

```text
Agent process ──── stdin ────▶ MCP server process
              ◀── stdout ────
```

This is how tools like GitHub's MCP server, filesystem tools, and database connectors typically run. The server starts on demand and exits when the agent disconnects.

### SSE transport

For remote MCP servers that run as HTTP services. The agent connects via HTTP and receives responses as server-sent events:

```text
Agent ──── HTTP POST ────▶ Remote MCP server
      ◀── SSE stream ────
```

This is used for shared MCP servers that serve multiple agents or run in cloud infrastructure.

## The full picture

```text
┌──────────────────────────────────────────────────────────┐
│  Agent loop                                              │
│                                                          │
│  ┌──────────────────────┐   ┌──────────────────────┐     │
│  │  Native tools        │   │  MCP tools           │     │
│  │  (ToolRegistry)      │   │  (McpToolAdapter)    │     │
│  │  fs.read_file        │   │  mcp.github.search   │     │
│  │  shell.exec          │   │  mcp.db.query        │     │
│  └──────────┬───────────┘   └──────────┬───────────┘     │
│             │                          │                 │
│             └──── unified tool list ───┘                 │
│                        │                                 │
│               presented to model                         │
│                                                          │
│  MCP resources ──▶ ContextLoader ──▶ transcript          │
│  MCP prompts   ──▶ ContextLoader ──▶ transcript          │
└──────────────────────────────────────────────────────────┘
```

Native tools and MCP tools appear as a single list to the model. The model doesn't know (or need to know) which tools come from MCP and which are native. The `mcp.<server_id>.` prefix distinguishes them in the tool name for human readers and policy evaluation, but the model just sees a tool spec with a name and schema.

> **Example:** [`openrouter-mcp-tool`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-mcp-tool) demonstrates MCP tool discovery and invocation. [`openrouter-agent-cli`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-agent-cli) shows MCP integrated into a full agent with context, tools, and compaction.
>
> **Crate:** [`agentkit-mcp`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-mcp) — depends on [`agentkit-capabilities`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-capabilities), [`agentkit-tools-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-tools-core), and [`agentkit-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-core).
