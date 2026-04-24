# MCP integration

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) lets agents discover and use tools, resources, and prompts from external servers. This chapter covers [`agentkit-mcp`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-mcp): how MCP fits into the capability and tool layers, how auth and lifecycle are managed, and how the client surfaces server-initiated requests and events.

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

`agentkit-mcp` is built on top of the official [`rmcp`](https://crates.io/crates/rmcp) Rust SDK. The wire-protocol types (`CallToolResult`, `ReadResourceResult`, `Content`, `Tool`, `Prompt`, …) are re-exported as `McpTool`, `McpResource`, `McpPrompt`, etc. — there is no parallel agentkit-side vocabulary.

## MCP in the capability model

MCP servers expose three capability types, which map directly to agentkit's capability layer:

| MCP concept   | agentkit abstraction            | How it's used                              |
| ------------- | ------------------------------- | ------------------------------------------ |
| MCP tools     | `Invocable` → adapted to `Tool` | Model calls them during turns              |
| MCP resources | `ResourceProvider`              | Host reads them for context loading        |
| MCP prompts   | `PromptProvider`                | Host renders them for transcript injection |

An MCP server implements `CapabilityProvider`, exposing all three through one registration point — `McpCapabilityProvider`.

## Server configuration

```rust,ignore
pub struct McpServerConfig {
    pub id: McpServerId,
    pub transport: McpTransportBinding,
    pub metadata: MetadataMap,
}
```

Built-in transports: **stdio** (local child process) and **Streamable HTTP** (modern remote MCP). Both are driven by rmcp's transport implementations — agentkit-mcp does not maintain its own JSON-RPC framing.

## Discovery

After connecting, the server's capabilities are captured in a snapshot of rmcp wire types:

```rust,ignore
pub struct McpDiscoverySnapshot {
    pub server_id: McpServerId,
    pub tools: Vec<McpTool>,        // = rmcp::model::Tool
    pub resources: Vec<McpResource>, // = rmcp::model::Resource (Annotated<RawResource>)
    pub prompts: Vec<McpPrompt>,     // = rmcp::model::Prompt
    pub metadata: MetadataMap,
}
```

Snapshots are cacheable and refreshable. Hosts choose which capabilities to expose — discovery doesn't automatically register everything. Pattern-matching directly against the rmcp types gives access to `output_schema`, `annotations`, `mime_type`, prompt arguments, etc. without a wrapping layer.

## Tool adaptation

`McpToolAdapter` wraps an MCP tool as a `Tool` implementation:

- exposes a `ToolSpec` derived from the `McpTool` descriptor (annotations included)
- translates `ToolRequest` into an rmcp `tools/call`
- translates `CallToolResult` (content blocks + optional `structured_content`) into a normalized `ToolResult`
- surfaces auth challenges as `ToolError::AuthRequired`

### Namespacing

MCP tools are namespaced by default as `mcp_<server_id>_<tool_name>`. This prevents collisions with native tools. Hosts can swap the strategy via `McpToolNamespace::None` (strip the prefix) or `McpToolNamespace::Custom(...)` (e.g. `remote.<server>.<tool>`):

```rust,ignore
let manager = McpServerManager::new().with_namespace(McpToolNamespace::custom(
    |server, name| format!("remote.{server}.{name}"),
));
```

## Sampling, elicitation, and roots

MCP is bidirectional: a server can ask the client to do work too. agentkit-mcp surfaces three responder traits — install one to handle each request type. The client only advertises the corresponding `ClientCapabilities` entry when a responder is wired in, so servers see exactly the surface the host opted into.

| Server request             | Trait                       | Use                                                |
| -------------------------- | --------------------------- | -------------------------------------------------- |
| `sampling/createMessage`   | `McpSamplingResponder`      | Server asks the host LLM to generate a completion  |
| `elicitation/create`       | `McpElicitationResponder`   | Server asks the user for input                     |
| `roots/list`               | `McpRootsProvider`          | Server enumerates workspace roots in scope         |

```rust,ignore
let manager = McpServerManager::new()
    .with_sampling_responder(Arc::new(host_sampling))
    .with_elicitation_responder(Arc::new(prompt_user))
    .with_roots_provider(Arc::new(workspace_roots));
```

## Server-pushed events

Servers also push notifications: progress updates for long-running tools, log messages, resource updates the client subscribed to, list-changed announcements, and cancellation. Subscribe to `McpConnection::subscribe_events` to receive them as `McpServerEvent`:

```rust,ignore
let mut events = connection.subscribe_events();
connection.subscribe_resource("memo:welcome").await?;
while let Ok(event) = events.recv().await {
    match event {
        McpServerEvent::Progress(progress) => /* update UI */ {},
        McpServerEvent::Logging(message)   => /* write to log */ {},
        McpServerEvent::ResourceUpdated(_) => /* re-read resource */ {},
        McpServerEvent::ToolListChanged
        | McpServerEvent::ResourceListChanged
        | McpServerEvent::PromptListChanged => /* refresh discovery */ {},
        McpServerEvent::Cancelled(_)       => /* stop in-flight work */ {},
    }
}
```

`set_logging_level(LoggingLevel::Info)` negotiates the minimum severity the server emits. `subscribe_resource` / `unsubscribe_resource` toggle per-URI watch on resources that support it.

List-changed events are *also* delivered through the legacy `McpServerNotification` mpsc receiver that `McpServerManager::refresh_changed_catalogs` drains to re-run discovery — the two channels coexist: events for live UI/observability, mpsc for catalog re-sync.

## Auth as interruption

MCP auth follows the same interrupt pattern as tool approvals:

1. Tool invocation triggers an auth requirement
2. The tool adapter returns `ToolError::AuthRequired(AuthRequest)`
3. The loop surfaces it as `LoopStep::Interrupt(...)`
4. The host performs the auth flow (OAuth, API key entry, etc.)
5. The host resolves the interrupt with `manager.resolve_auth(...)` and the operation resumes

Auth is never hidden retry logic. The host always knows when auth is happening and controls the flow. For non-tool MCP operations (connecting, reading resources, fetching prompts), auth follows the same pattern but through the MCP manager API rather than the loop interrupt system.

To rotate credentials at runtime — e.g. a Streamable HTTP bearer that expires every hour — drive an `AuthResolution::Provided { credentials, .. }` through `McpServerManager::resolve_auth`. The next operation reconnects with the new credentials.

## Transports

| Transport       | Connection                              | Use case                   |
| --------------- | --------------------------------------- | -------------------------- |
| stdio           | Spawn child process, pipe stdin/stdout  | Local tool servers         |
| Streamable HTTP | HTTP POST with JSON or SSE responses    | Modern remote tool servers |

The rest of agentkit doesn't know whether a server is reached via stdio, HTTP, or in-memory pipes. Transport is configured in `McpServerConfig` and the MCP manager handles the connection lifecycle.

### stdio transport

The most common pattern for local MCP servers. The agent spawns the server as a child process and communicates over stdin/stdout:

```text
Agent process ──── stdin ────▶ MCP server process
              ◀── stdout ────
```

This is how tools like GitHub's MCP server, filesystem tools, and database connectors typically run. The server starts on demand and exits when the agent disconnects.

### Streamable HTTP transport

For modern remote MCP servers that run as HTTP services. rmcp drives JSON-RPC over HTTP POST, accepts either JSON or SSE responses, and tracks the negotiated session/protocol headers:

```text
Agent ──── HTTP POST ────▶ Remote MCP server
      ◀── JSON or SSE ───
```

If an SSE response stream is interrupted before the matching response arrives, the client resumes with `Last-Event-ID`. Bearer tokens and arbitrary custom headers are configured declaratively on `StreamableHttpTransportConfig`.

### Custom transports

When you need a transport rmcp supports but `McpTransportBinding` does not (in-memory pipes for tests, websockets, custom IO), build the rmcp `RunningService` yourself and adopt it through `McpConnection::from_running_service_with_events`. Pair the service with the channels returned by `McpClientHandler::builder().build()` so list-change notifications and `McpServerEvent` subscribers stay observable.

## The full picture

```text
┌──────────────────────────────────────────────────────────┐
│  Agent loop                                              │
│                                                          │
│  ┌──────────────────────┐   ┌──────────────────────┐     │
│  │  Native tools        │   │  MCP tools           │     │
│  │  (ToolRegistry)      │   │  (McpToolAdapter)    │     │
│  │  fs_read_file        │   │  mcp_github_search   │     │
│  │  shell_exec          │   │  mcp_db_query        │     │
│  └──────────┬───────────┘   └──────────┬───────────┘     │
│             │                          │                 │
│             └──── unified tool list ───┘                 │
│                        │                                 │
│               presented to model                         │
│                                                          │
│  MCP resources ──▶ ContextLoader ──▶ transcript          │
│  MCP prompts   ──▶ ContextLoader ──▶ transcript          │
│                                                          │
│  MCP server events ──▶ McpConnection::subscribe_events   │
│  MCP server requests ──▶ host responders                 │
│    (sampling / elicitation / roots)                      │
└──────────────────────────────────────────────────────────┘
```

Native tools and MCP tools appear as a single list to the model. The model doesn't know (or need to know) which tools come from MCP and which are native. The `mcp_<server_id>_` prefix distinguishes them in the tool name for human readers and policy evaluation, but the model just sees a tool spec with a name and schema.

> **Example:** [`openrouter-mcp-tool`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-mcp-tool) demonstrates MCP tool discovery and invocation. [`openrouter-agent-cli`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-agent-cli) shows MCP integrated into a full agent with context, tools, and compaction. [`mcp-reference-interop`](https://github.com/danielkov/agentkit/tree/main/examples/mcp-reference-interop) and [`mcp-dynamic-auth`](https://github.com/danielkov/agentkit/tree/main/examples/mcp-dynamic-auth) cover transport interop and credential rotation respectively.
>
> **Crate:** [`agentkit-mcp`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-mcp) — depends on [`agentkit-capabilities`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-capabilities), [`agentkit-tools-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-tools-core), [`agentkit-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-core), and [`rmcp`](https://crates.io/crates/rmcp).
