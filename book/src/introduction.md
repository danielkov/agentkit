# agentkit

`agentkit` is a Rust toolkit for building LLM agent applications such as coding agents, assistant CLIs, and multi-agent tools.

The project is intentionally split into small crates behind feature flags so hosts can pull in only the pieces they need.

## Current status

`agentkit` includes working implementations for:

- normalized transcript, content-part, and delta types
- a runtime-agnostic loop driver with blocking interrupts for approval, auth, and input
- trait-based tools, permissions, approvals, and auth handoff
- built-in filesystem and shell tools
- context loading for `AGENTS.md` and skills directories
- MCP transports, discovery, tool/resource/prompt adapters, auth replay, and lifecycle management
- reporting observers
- compaction triggers, strategy pipelines, and backend-driven semantic compaction
- async task management with foreground/background scheduling, routing policies, and detach-after-timeout
- optional turn cancellation with resumable sessions
- an OpenRouter provider adapter

## Installation

Add `agentkit` to your project:

```sh
cargo add agentkit
```

Or add it to your `Cargo.toml` directly:

```toml
[dependencies]
agentkit = "0.1"
```

Enable only the features you need:

```toml
[dependencies]
agentkit = { version = "0.1", default-features = false, features = ["core", "tools", "loop"] }
```

See [Feature flags](./feature-flags.md) for the full list.

## Minimal composition

The smallest useful assembly is a model adapter, a tool registry, a permission checker, and a loop observer:

```rust
let agent = Agent::builder()
    .model(adapter)
    .tools(agentkit_tool_fs::registry())
    .permissions(my_permissions)
    .observer(my_reporter)
    .build()?;

let mut driver = agent
    .start(SessionConfig {
        session_id: SessionId::new("demo"),
        metadata: MetadataMap::new(),
    })
    .await?;

driver.submit_input(vec![system_item, user_item])?;

match driver.next().await? {
    LoopStep::Finished(result) => { /* render output */ }
    LoopStep::Interrupt(interrupt) => { /* approval, auth, or input */ }
}
```
