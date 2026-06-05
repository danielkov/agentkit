# agentkit-tool-compose

Lua tool composition for agentkit.

This crate exposes a single `compose` tool. The model supplies a Lua script and
optional JSON input; the script can call the current tool catalog with
`tool(name, input)` and inspect available tools with `tools()`.

```rust
let registry = agentkit_tool_compose::registry();
```

Compose is opt-in. Add this registry explicitly with
`AgentBuilder::add_tool_source`.

