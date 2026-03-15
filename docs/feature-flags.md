# feature flags

The umbrella crate `agentkit` re-exports subcrates behind feature flags.

## Flags

- `core`
  - enables `agentkit-core`
- `compaction`
  - enables `agentkit-compaction`
- `capabilities`
  - enables `agentkit-capabilities`
- `context`
  - enables `agentkit-context`
- `tools`
  - enables `agentkit-tools-core`
- `loop`
  - enables `agentkit-loop`
- `mcp`
  - enables `agentkit-mcp`
- `provider-openrouter`
  - enables `agentkit-provider-openrouter`
- `reporting`
  - enables `agentkit-reporting`
- `tool-fs`
  - enables `agentkit-tool-fs`
- `tool-shell`
  - enables `agentkit-tool-shell`

## Default flags

The current default set is:

- `core`
- `capabilities`
- `tools`
- `loop`
- `reporting`

## Typical combinations

Minimal orchestration:

- `core`
- `capabilities`
- `tools`
- `loop`

Coding agent:

- `core`
- `capabilities`
- `context`
- `tools`
- `loop`
- `tool-fs`
- `tool-shell`
- `reporting`

MCP-enabled agent:

- everything above
- `mcp`

OpenRouter-backed example host:

- everything needed for the host
- `provider-openrouter`
