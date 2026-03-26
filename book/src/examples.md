# Examples

The examples are meant to build up in complexity. Each one exercises different pieces of the toolkit.

## `openrouter-chat`

Minimal chat loop with provider + loop. Supports `Ctrl-C` turn cancellation.

```sh
cargo run -p openrouter-chat -- "hello"
```

## `openrouter-coding-agent`

One-shot coding-oriented prompt runner with filesystem tools and permissions.

```sh
cargo run -p openrouter-coding-agent -- \
  "Use fs.read_file on ./Cargo.toml and return only the workspace member count as an integer."
```

## `openrouter-context-agent`

Context loading from `AGENTS.md` and skills directories.

## `openrouter-mcp-tool`

MCP tool discovery and invocation.

## `openrouter-subagent-tool`

Custom tool that runs a nested agent.

## `openrouter-compaction-agent`

Structural, semantic, and hybrid compaction with a nested-loop compaction backend.

## `openrouter-parallel-agent`

Async task manager with foreground filesystem tools and detach-after-timeout shell tools. `TaskManagerHandle` event stream printed to stderr.

## `openrouter-agent-cli`

Combined example using context, tools, shell, MCP, compaction, and reporting.

```sh
cargo run -p openrouter-agent-cli -- --mcp-mock \
  "Return only the secret from the MCP tool."
```
