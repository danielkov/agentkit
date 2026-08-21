# agentkit-tool-compose

Scripted tool composition for agentkit.

This crate exposes a single `compose` tool. The model supplies a script and
optional JSON input; the script can call the current tool catalog with
`tool(name, input)` and inspect available tools with `tools()`.

Sandboxed Lua is enabled by default. To use only the Runlet backend:

```toml
agentkit-tool-compose = { version = "0.10.9", default-features = false, features = ["runlet"] }
```

```rust
let registry = agentkit_tool_compose::registry();
```

Compose is opt-in. Add this registry explicitly with
`AgentBuilder::add_tool_source`.

For a richer tool description, wrap an existing tool source:

```rust
let tools = agentkit_tool_compose::ComposeTool::wrap(child_source);
```

The wrapped source still advertises and executes its child tools directly, while
`compose` renders child output schemas into its own description. Dynamic sources
remain live: catalog events and child lookups delegate to the wrapped source.

## Runlet ordering

Runlet schedules independent calls concurrently, including effectful calls such
as writes. Ordinary data references establish dependencies: a call that uses an
earlier result waits for that result. When a call must wait for earlier work it
does not read, express the ordering edge explicitly:

```runlet
prepared = prepare_workspace({ path: input.path })
result = after prepared {
    return publish_workspace({ path: input.path })
}
return result
```

Calls lexically created inside an `after` block are created only after every
prerequisite succeeds. Use `after` for required sequencing, not source order;
two adjacent calls with no data dependency or explicit `after` edge may run in
parallel.

The final compose result enters the transcript as compact JSON by default.
With the `toon` feature enabled,
`ComposeConfig::with_result_encoding(ResultEncoding::Toon)` switches it to
[TOON](https://docs.rs/serde_toon2) (Token-Oriented Object Notation), which
renders uniform object lists as a header plus one row per element — smaller
than JSON for the list-shaped values compose scripts tend to return. The tool
description gains a note explaining the format so the model can read it.
