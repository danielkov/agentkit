# Crates

## `agentkit-core`

Normalized transcript, part, delta, usage, and tool-call/result types.

## `agentkit-capabilities`

Lower-level invocable/resource/prompt abstraction.

## `agentkit-tools-core`

Tool traits, registry, executor, permissions, approvals, auth requests.

## `agentkit-loop`

Model session abstraction, driver, interrupts, loop events, tool roundtrips.

## `agentkit-context`

`AGENTS.md` and skills loading.

## `agentkit-mcp`

MCP transports, discovery, lifecycle, auth, replay, tool/resource/prompt adapters.

## `agentkit-reporting`

Loop observers for stdout, JSONL, usage, transcripts, fanout.

## `agentkit-compaction`

Compaction trigger, strategy pipeline, backend hooks, and basic built-ins.

## `agentkit-task-manager`

Task scheduling for tool execution: foreground, background, and detach-after-timeout routing.

## `agentkit-tool-fs`

Built-in filesystem tools.

## `agentkit-tool-shell`

Built-in shell tool.

## `agentkit-tool-skills`

Skill discovery and activation tool.

## `agentkit-provider-openrouter`

OpenRouter model adapter.

## `agentkit`

Umbrella crate with feature-gated re-exports of all subcrates.
