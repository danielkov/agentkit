# agentkit

`agentkit` is a Rust toolkit for building LLM agent applications such as coding agents, assistant CLIs, and multi-agent tools.

The project is intentionally split into small crates behind feature flags so hosts can pull in only the pieces they need.

## Current status

`agentkit` is past the design-only stage. The repo currently includes working implementations for:

- normalized transcript, content-part, and delta types
- a runtime-agnostic loop driver with blocking interrupts for approval, auth, and input
- trait-based tools, permissions, approvals, and auth handoff
- built-in filesystem and shell tools
- context loading for `AGENTS.md` and skills directories
- MCP transports, discovery, tool/resource/prompt adapters, auth replay, and lifecycle management
- reporting observers
- compaction triggers, strategy pipelines, and backend-driven semantic compaction
- optional turn cancellation with resumable sessions
- an OpenRouter provider adapter

The repo also ships multiple examples that exercise these pieces end to end.

## Crates

- `agentkit-core`
  - transcript, parts, deltas, IDs, usage, and cancellation primitives
- `agentkit-capabilities`
  - lower-level invocable/resource/prompt abstraction
- `agentkit-tools-core`
  - tools, registry, executor, permissions, approvals, auth requests
- `agentkit-loop`
  - model session abstraction, driver, interrupts, tool roundtrips
- `agentkit-context`
  - `AGENTS.md` and skills loading
- `agentkit-mcp`
  - MCP transports, discovery, lifecycle, auth, replay, adapters
- `agentkit-reporting`
  - loop observers and reporting adapters
- `agentkit-compaction`
  - compaction triggers, strategies, pipelines, backend hooks
- `agentkit-tool-fs`
  - filesystem tools
- `agentkit-tool-shell`
  - shell execution tool
- `agentkit-provider-openrouter`
  - OpenRouter adapter
- `agentkit`
  - umbrella crate with feature-gated re-exports

## Built-in tools today

Filesystem:

- `fs.read_file`
  - supports optional `from` / `to` line ranges
- `fs.write_file`
- `fs.replace_in_file`
- `fs.move`
- `fs.delete`
- `fs.list_directory`
- `fs.create_directory`

Shell:

- `shell.exec`

The filesystem crate also supports session-scoped read-before-write enforcement through `FileSystemToolResources` and `FileSystemToolPolicy`.

## Quick start

1. Put your OpenRouter credentials in `.env`.
2. Run one of the examples.

Example `.env`:

```env
OPENROUTER_API_KEY=replace_me
OPENROUTER_MODEL=openrouter/hunter-alpha
```

Example commands:

```bash
cargo run -p openrouter-chat -- "hello"
```

```bash
cargo run -p openrouter-coding-agent -- \
  "Use fs.read_file on ./Cargo.toml and return only the workspace member count as an integer."
```

```bash
cargo run -p openrouter-agent-cli -- --mcp-mock \
  "Return only the secret from the MCP tool."
```

## Example progression

- `openrouter-chat`
  - minimal chat loop
  - now supports `Ctrl-C` turn cancellation
- `openrouter-coding-agent`
  - one-shot coding-oriented prompt runner with filesystem tools
- `openrouter-context-agent`
  - context loading from `AGENTS.md` and skills
- `openrouter-mcp-tool`
  - MCP tool discovery and invocation
- `openrouter-subagent-tool`
  - custom tool that runs a nested agent
- `openrouter-compaction-agent`
  - structural, semantic, and hybrid compaction
  - semantic compaction uses a nested agent as the backend
- `openrouter-agent-cli`
  - combined example using context, tools, shell, MCP, compaction, and reporting

## Minimal composition

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

## Feature flags

The umbrella crate re-exports subcrates behind feature flags.

Default flags:

- `core`
- `capabilities`
- `tools`
- `loop`
- `reporting`

Optional flags:

- `compaction`
- `context`
- `mcp`
- `provider-openrouter`
- `tool-fs`
- `tool-shell`

More detail is in [docs/feature-flags.md](./docs/feature-flags.md).

## Docs

- [docs/getting-started.md](./docs/getting-started.md)
- [docs/architecture.md](./docs/architecture.md)
- [docs/core.md](./docs/core.md)
- [docs/tools.md](./docs/tools.md)
- [docs/mcp.md](./docs/mcp.md)
- [docs/compaction.md](./docs/compaction.md)
- [docs/README.md](./docs/README.md)

## Notable gaps

The main remaining work is polish and hardening rather than missing foundation:

- public-release decisions around the default helper policy set
- richer docs and onboarding polish
- deeper built-in tool ergonomics
- more end-to-end examples and integration coverage
