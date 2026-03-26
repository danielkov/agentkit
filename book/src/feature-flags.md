# Feature flags

The umbrella crate `agentkit` re-exports subcrates behind feature flags.

## Default flags

- `core` — `agentkit-core`
- `capabilities` — `agentkit-capabilities`
- `tools` — `agentkit-tools-core`
- `task-manager` — `agentkit-task-manager`
- `loop` — `agentkit-loop`
- `reporting` — `agentkit-reporting`

## Optional flags

- `compaction` — `agentkit-compaction`
- `context` — `agentkit-context`
- `mcp` — `agentkit-mcp`
- `provider-openrouter` — `agentkit-provider-openrouter`
- `tool-fs` — `agentkit-tool-fs`
- `tool-shell` — `agentkit-tool-shell`

## Typical combinations

**Minimal orchestration:**

```toml
agentkit = { version = "0.1", features = ["core", "capabilities", "tools", "loop"] }
```

**Coding agent:**

```toml
agentkit = { version = "0.1", features = [
    "core", "capabilities", "context", "tools",
    "loop", "tool-fs", "tool-shell", "reporting",
] }
```

**MCP-enabled agent:**

```toml
agentkit = { version = "0.1", features = [
    "core", "capabilities", "context", "tools",
    "loop", "tool-fs", "tool-shell", "reporting", "mcp",
] }
```

**OpenRouter-backed example host:**

```toml
agentkit = { version = "0.1", features = [
    "core", "capabilities", "tools", "loop",
    "reporting", "provider-openrouter",
] }
```
