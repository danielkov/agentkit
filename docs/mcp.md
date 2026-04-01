# agentkit-mcp design

## Purpose

`agentkit-mcp` is the MCP integration crate.

It should make MCP servers usable from `agentkit` without forcing the rest of the stack to understand MCP transport details.

It should own:

- MCP server configuration and lifecycle
- transport/session management
- auth handshakes and auth-required interruptions
- discovery of MCP tools, resources, and prompts
- adaptation of MCP tools into the shared tool system
- access APIs for MCP resources and prompts

It should not pretend that all of MCP is a tool.

## Non-goals

`agentkit-mcp` should not own:

- the main loop driver
- the generic tool registry contract
- shell/process execution outside MCP server management
- host UI for auth or approval
- long-term caching or persistence
- provider-specific model logic

This crate integrates MCP into `agentkit`; it does not replace the rest of the architecture.

## Design principles

### 1. MCP tools should plug into the normal tool system

If an MCP server exposes tools, those should appear as ordinary `ToolSpec`s and execute through the same `ToolExecutor` path as native tools.

That gives the loop one unified tool flow:

- tool discovered
- tool spec exposed to model
- tool called
- permission checked
- auth/approval interruption surfaced if needed
- tool result returned

### 2. MCP resources and prompts stay first-class MCP concepts

Resources and prompts are not tools.

They should be exposed through dedicated MCP-facing APIs rather than awkward fake tools such as:

- `mcp.read_resource`
- `mcp.get_prompt`

That preserves MCP’s structure and avoids flattening unlike concepts into one trait.

### 3. Auth is an interruption, not hidden retry logic

MCP often involves auth or capability negotiation.

`agentkit-mcp` should surface auth requirements explicitly so the host can resolve them.

That should align with the loop/tool interruption model:

- tool invocation may interrupt with `AuthRequired`
- server/session startup may interrupt with `AuthRequired`
- the host resolves the auth flow
- the same operation can then resume

### 4. Transport details stay inside this crate

The rest of `agentkit` should not care whether an MCP server is reached via:

- stdio
- local child process
- TCP
- websocket
- HTTP streaming

Those should all normalize into one server/session abstraction.

The transport layer should be pluggable.

Built-in transports for v1 should be:

- stdio
- Streamable HTTP
- legacy SSE compatibility

Hosts should also be able to provide their own transport implementation.

### 5. Discovery should be explicit and cacheable

MCP server capabilities may change over time, but hosts should not be forced to re-discover on every loop step.

The crate should support:

- explicit discovery/refresh
- stable snapshots of tools/resources/prompts
- invalidation when server configuration changes

### 6. MCP should implement the lower-level capability layer

`agentkit-mcp` should build on `agentkit-capabilities`, not define a parallel universe.

That means:

- MCP tools map to invocables
- MCP resources map to resource providers
- MCP prompts map to prompt providers

## Main boundary

The clean separation is:

- `agentkit-capabilities` owns the lower-level invocable/resource/prompt contracts
- `agentkit-tools-core` owns generic tool execution contracts
- `agentkit-mcp` adapts MCP tools into those contracts
- `agentkit-mcp` separately exposes MCP resources/prompts/server lifecycle/auth

So:

- MCP tools participate in the shared tool registry
- MCP resources/prompts do not get forced into the tool system

## Core concepts

## 1. Server configuration

Hosts need a way to define MCP servers declaratively.

Recommended shape:

```rust
pub struct McpServerConfig {
    pub id: McpServerId,
    pub transport: McpTransportBinding,
    pub auth: McpAuthConfig,
    pub metadata: MetadataMap,
}
```

`McpTransportBinding` should support:

- built-in stdio transport configuration
- built-in Streamable HTTP transport configuration
- built-in legacy SSE transport configuration
- host-provided custom transport factories

This gives you a transport-agnostic surface without closing the door on user-defined transports.

Recommended transport boundary:

```rust
pub trait McpTransportFactory: Send + Sync {
    async fn connect(&self) -> Result<Box<dyn McpTransport>, McpError>;
}

pub trait McpTransport: Send {
    async fn send(&mut self, message: McpFrame) -> Result<(), McpError>;
    async fn recv(&mut self) -> Result<Option<McpFrame>, McpError>;
    async fn close(&mut self) -> Result<(), McpError>;
}
```

The built-in stdio, Streamable HTTP, and legacy SSE implementations should just be default `McpTransportFactory` implementations.

## 2. Server manager

There should be one subsystem that owns MCP server lifecycle.

Recommended shape:

```rust
pub trait McpServerManager {
    async fn connect(
        &self,
        config: &McpServerConfig,
    ) -> Result<McpConnection, McpError>;
}
```

`McpConnection` is the live handle to one configured MCP server.

It should own:

- transport session
- negotiated capabilities
- auth state
- discovery snapshot

This gives you one operational boundary for server management.

## 3. Connection snapshot and discovery

Hosts and the loop should not have to ask the live server for static capability details repeatedly.

Recommended discovery artifact:

```rust
pub struct McpDiscoverySnapshot {
    pub server_id: McpServerId,
    pub tools: Vec<McpToolDescriptor>,
    pub resources: Vec<McpResourceDescriptor>,
    pub prompts: Vec<McpPromptDescriptor>,
    pub metadata: MetadataMap,
}
```

This snapshot should be:

- serializable
- refreshable
- usable to populate tool registries and host UX

## 4. MCP tool adaptation

MCP tools should be adapted into `Tool` implementations, but the crate should also expose them through the lower-level invocable layer.

Recommended shape:

```rust
pub struct McpToolAdapter {
    server_id: McpServerId,
    descriptor: McpToolDescriptor,
    client: Arc<McpConnection>,
}
```

This adapter should:

- expose a `ToolSpec`
- translate `ToolRequest` into MCP tool invocation
- translate MCP responses into normalized `ToolResult`
- surface auth or capability issues as `ToolInterruption`

This is the key place where MCP joins the shared tool path.

## 5. MCP resources API

Resources should have their own access layer.

Recommended shape:

```rust
pub trait McpResourceStore {
    async fn list_resources(
        &self,
        server: &McpServerId,
    ) -> Result<Vec<McpResourceDescriptor>, McpError>;

    async fn read_resource(
        &self,
        server: &McpServerId,
        resource: &McpResourceId,
    ) -> Result<McpResourceContents, McpError>;
}
```

This is separate from the tool system on purpose.

It allows:

- host-side browsing
- context loading integrations
- prompt assembly using MCP resources

without pretending resource access is a tool call.

## 6. MCP prompts API

Prompts should also stay distinct.

Recommended shape:

```rust
pub trait McpPromptStore {
    async fn list_prompts(
        &self,
        server: &McpServerId,
    ) -> Result<Vec<McpPromptDescriptor>, McpError>;

    async fn get_prompt(
        &self,
        server: &McpServerId,
        prompt: &McpPromptId,
        args: serde_json::Value,
    ) -> Result<McpPromptContents, McpError>;
}
```

This matters because prompts are closer to context-generation helpers than executable tools.

That makes them more relevant to:

- host UX
- context sources
- prompt assembly

than to the tool executor path.

## 7. Auth model

Auth needs to be explicit and resumable.

Recommended types:

```rust
pub struct AuthRequest {
    pub server_id: McpServerId,
    pub kind: AuthKind,
    pub details: MetadataMap,
}

pub enum AuthResolution {
    Provided(MetadataMap),
    Cancelled,
}
```

Where auth may arise during:

- server connection
- capability negotiation
- tool invocation
- resource read
- prompt retrieval

Recommended rule:

- the MCP crate should not run host UX for auth
- it should surface an `AuthRequest`
- the host resolves it and the operation resumes

## 8. Integration with loop interrupts

The loop already has a blocking interrupt model.

MCP should fit into it by translation, not by inventing a parallel control system.

Recommended mapping:

- MCP tool auth interruption -> `ToolInterruption::AuthRequired`
- tool interruption -> loop interrupt
- non-tool MCP auth needs, if initiated outside the loop, remain MCP-level API results for the host to resolve directly

This is important because not all MCP interactions happen in a running agent turn.

## 9. Permission boundary

MCP needs its own structured action representation for policy decisions.

Recommended `McpAction` variants:

- connect server
- invoke tool
- read resource
- fetch prompt
- use auth scope

These actions should plug into the shared `ProposedToolAction::Mcp(...)` path where relevant.

For non-tool MCP operations, the same structured action model should still be reusable even if the execution path is not through `ToolExecutor`.

This is one reason MCP should be designed before the detailed permissions doc: it expands the policy surface beyond local shell/fs operations.

## 10. Discovery and registration

Hosts need a practical way to take discovered MCP tools and register them.

Recommended flow:

1. host configures server(s)
2. `agentkit-mcp` connects and discovers capabilities
3. it produces an `McpDiscoverySnapshot`
4. the host selects which MCP tools to expose
5. `McpToolAdapter`s are created and registered in `ToolRegistry`

Important design choice:

- discovery should not automatically expose every tool by default

Hosts should be able to:

- expose all
- expose a filtered subset
- rename or namespace tools
- attach policy overrides per server or per tool

## 11. Namespacing

MCP tools need predictable names to coexist with native tools.

Recommended default convention:

- `mcp.<server_id>.<tool_name>`

Examples:

- `mcp.github.search_code`
- `mcp.linear.get_issue`

Hosts may override the public name if they want a cleaner surface, but the default should be collision-safe.

## 12. Error model

`McpError` should be specific enough to support retries, reporting, and host intervention.

Recommended categories:

- transport failure
- protocol error
- capability missing
- auth required
- auth failed
- invocation failed
- discovery failed
- unavailable

When MCP tools are adapted into generic tool execution, these should map cleanly into:

- `ToolInterruption`
- `ToolError`
- observer events

without losing useful metadata.

## 13. Context integration

Because MCP resources and prompts are not tools, they should integrate with `agentkit-context`, not only with the loop.

Examples:

- load an MCP resource into the effective context for a session
- materialize an MCP prompt as one or more `Item`s

This bridge should be owned by `agentkit-mcp`.

Reason:

- MCP already owns the resource and prompt semantics
- MCP now implements the shared capability layer directly
- the context layer should be able to consume MCP-backed resources/prompts through adapters exposed by `agentkit-mcp`

The interface should still follow the lower-level capability abstractions rather than a bespoke MCP-only path.

The important point is that MCP prompts/resources should be usable in prompt assembly without going through fake tool calls.

## 14. Runtime considerations

Unlike `core` and ideally `loop`, `agentkit-mcp` may need stronger runtime assumptions depending on transports.

Practical recommendation:

- keep public MCP abstractions runtime-neutral where possible
- allow transport implementations to depend on `tokio` behind feature flags

Possible feature split:

- `stdio`
- `sse`
- `auth`

This keeps the integration flexible without pretending process/network handling is runtime-free.

## Suggested public API shape

Recommended first-pass types:

```rust
pub struct McpServerConfig { /* ... */ }
pub struct McpDiscoverySnapshot { /* ... */ }
pub struct McpToolAdapter { /* ... */ }

pub trait McpServerManager { /* ... */ }
pub trait McpResourceStore { /* ... */ }
pub trait McpPromptStore { /* ... */ }
```

And operational result/interrupt types:

```rust
pub struct AuthRequest { /* ... */ }
pub enum AuthResolution { /* ... */ }
pub enum McpError { /* ... */ }
```

This is enough to support:

- MCP server connection
- discovery
- tool adaptation
- resource access
- prompt access
- explicit auth handshakes

## Suggested module layout

```text
agentkit-mcp/
  src/
    lib.rs
    config.rs
    manager.rs
    connection.rs
    discovery.rs
    tool_adapter.rs
    resources.rs
    prompts.rs
    auth.rs
    action.rs
    error.rs
```

Module intent:

- `config.rs`: server and transport configuration
- `manager.rs`: connection/lifecycle interfaces
- `connection.rs`: live connection abstraction
- `discovery.rs`: descriptors and discovery snapshots
- `tool_adapter.rs`: MCP tool integration with `agentkit-tools-core`
- `resources.rs`: resource listing and read APIs
- `prompts.rs`: prompt listing and fetch APIs
- `auth.rs`: auth requests and resolution
- `action.rs`: MCP policy action types
- `error.rs`: MCP-specific errors

## What we should validate early

Before locking the MCP API, prove:

1. one MCP server can be connected and discovered from config alone
2. discovered MCP tools can be registered into `ToolRegistry` with collision-safe names
3. MCP tool invocation can interrupt for auth and resume cleanly
4. MCP resources can be read without going through the tool path
5. MCP prompts can be fetched and turned into context inputs
6. server-specific metadata survives adaptation without polluting the generic tool model

If any of those are awkward, the boundary between MCP, tools, and context is still wrong.
