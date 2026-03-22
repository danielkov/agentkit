# agentkit-mcp

Model Context Protocol integration for agentkit.

This crate covers:

- stdio and SSE MCP transports
- server configuration and lifecycle management
- discovery of MCP tools, resources, and prompts
- adapters that expose MCP servers as agentkit tools and capabilities
- auth replay support for MCP operations

Use it when you want MCP servers to participate in an agentkit runtime without writing custom transport glue.

## Configuring and connecting MCP servers

Register one or more MCP server configurations with `McpServerManager`, then connect
them. Each connected server is represented by an `McpServerHandle` that holds the
live connection and the discovery snapshot.

```rust,no_run
use agentkit_mcp::{
    McpServerConfig, McpServerManager, McpTransportBinding, StdioTransportConfig,
};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut manager = McpServerManager::new()
    .with_server(McpServerConfig::new(
        "filesystem",
        McpTransportBinding::Stdio(
            StdioTransportConfig::new("npx")
                .with_arg("-y")
                .with_arg("@modelcontextprotocol/server-filesystem"),
        ),
    ))
    .with_server(McpServerConfig::new(
        "github",
        McpTransportBinding::Stdio(
            StdioTransportConfig::new("npx")
                .with_arg("-y")
                .with_arg("@modelcontextprotocol/server-github")
                .with_env("GITHUB_TOKEN", "ghp_..."),
        ),
    ));

let handles = manager.connect_all().await?;
println!("connected {} MCP server(s)", handles.len());
# Ok(())
# }
```

## Discovering tools

After connecting, each server's capabilities are available through its discovery
snapshot. You can also build a combined `ToolRegistry` that spans all connected
servers.

```rust,no_run
use agentkit_mcp::{
    McpServerConfig, McpServerManager, McpServerId, McpTransportBinding,
    StdioTransportConfig,
};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
#     "filesystem",
#     McpTransportBinding::Stdio(StdioTransportConfig::new("npx")
#         .with_arg("-y").with_arg("@modelcontextprotocol/server-filesystem")),
# ));
# manager.connect_all().await?;
// Inspect a single server's tools:
let handle = manager.connected_server(&McpServerId::new("filesystem")).unwrap();
for tool in &handle.snapshot().tools {
    println!("  {} - {}", tool.name, tool.description.as_deref().unwrap_or(""));
}

// Build a combined tool registry across all servers:
let registry = manager.tool_registry();
for spec in registry.specs() {
    println!("{}", spec.name);  // e.g. "mcp.filesystem.read_file"
}
# Ok(())
# }
```

## Using MCP tools in an agent

The tool registry and capability provider produced by `McpServerManager` integrate
directly with the agentkit agent loop. Tools are namespaced as
`mcp.<server_id>.<tool_name>`.

```rust,no_run
use agentkit_mcp::{
    McpServerConfig, McpServerManager, McpTransportBinding, StdioTransportConfig,
};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
#     "filesystem",
#     McpTransportBinding::Stdio(StdioTransportConfig::new("npx")
#         .with_arg("-y").with_arg("@modelcontextprotocol/server-filesystem")),
# ));
# manager.connect_all().await?;
// Get a ToolRegistry for the agent loop:
let tool_registry = manager.tool_registry();

// Or get a CapabilityProvider for the capabilities system:
let capability_provider = manager.capability_provider();
# Ok(())
# }
```

## SSE transport

For remote MCP servers exposed over HTTP, use the SSE transport:

```rust,no_run
use agentkit_mcp::{
    McpServerConfig, McpServerManager, McpTransportBinding, SseTransportConfig,
};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
    "remote",
    McpTransportBinding::Sse(
        SseTransportConfig::new("https://mcp.example.com/sse")
            .with_header("Authorization", "Bearer tok_abc123"),
    ),
));

let handles = manager.connect_all().await?;
# Ok(())
# }
```

## Lifecycle management

Servers can be individually connected, refreshed, and disconnected:

```rust,no_run
use agentkit_mcp::{
    McpServerConfig, McpServerManager, McpServerId, McpTransportBinding,
    StdioTransportConfig,
};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
#     "filesystem",
#     McpTransportBinding::Stdio(StdioTransportConfig::new("npx")
#         .with_arg("-y").with_arg("@modelcontextprotocol/server-filesystem")),
# ));
let server_id = McpServerId::new("filesystem");

// Connect a single server:
let handle = manager.connect_server(&server_id).await?;

// Re-discover capabilities after changes:
let snapshot = manager.refresh_server(&server_id).await?;
println!("now has {} tools", snapshot.tools.len());

// Disconnect when done:
manager.disconnect_server(&server_id).await?;
# Ok(())
# }
```
