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
- `adapter-completions` — `agentkit-adapter-completions`
- `provider-groq` — `agentkit-provider-groq`
- `provider-mistral` — `agentkit-provider-mistral`
- `provider-ollama` — `agentkit-provider-ollama`
- `provider-openai` — `agentkit-provider-openai`
- `provider-openrouter` — `agentkit-provider-openrouter`
- `provider-vllm` — `agentkit-provider-vllm`
- `tool-fs` — `agentkit-tool-fs`
- `tool-shell` — `agentkit-tool-shell`
- `tool-skills` — `agentkit-tool-skills`

## Typical combinations

**Minimal orchestration:**

```toml
agentkit = { version = "0.2", features = ["core", "capabilities", "tools", "loop"] }
```

**Coding agent:**

```toml
agentkit = { version = "0.2", features = [
    "core", "capabilities", "context", "tools",
    "loop", "tool-fs", "tool-shell", "reporting",
] }
```

**MCP-enabled agent:**

```toml
agentkit = { version = "0.2", features = [
    "core", "capabilities", "context", "tools",
    "loop", "tool-fs", "tool-shell", "reporting", "mcp",
] }
```

**OpenRouter-backed example host:**

```toml
agentkit = { version = "0.2", features = [
    "core", "capabilities", "tools", "loop",
    "reporting", "provider-openrouter",
] }
```
