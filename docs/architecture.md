# agentkit architecture

`agentkit` is split into small crates with one umbrella crate for feature-gated re-exports.

## Current crate layout

- `agentkit-core`
  - normalized transcript, part, delta, usage, and tool-call/result types
- `agentkit-compaction`
  - compaction trigger, strategy pipeline, backend hooks, and basic built-ins
- `agentkit-capabilities`
  - lower-level invocable/resource/prompt abstraction
- `agentkit-tools-core`
  - tool traits, registry, executor, permissions, approvals, auth requests
- `agentkit-loop`
  - model session abstraction, driver, interrupts, loop events, tool roundtrips
- `agentkit-context`
  - `AGENTS.md` and skills loading
- `agentkit-mcp`
  - MCP transports, discovery, lifecycle, auth, replay, tool/resource/prompt adapters
- `agentkit-reporting`
  - loop observers for stdout, JSONL, usage, transcripts, fanout
- `agentkit-tool-fs`
  - built-in filesystem tools
- `agentkit-tool-shell`
  - built-in shell tool
- `agentkit-tool-skills`
  - progressive Agent Skills discovery and activation
- `agentkit-http`
  - HTTP transport abstraction shared by provider crates (`HttpClient`,
    `Http`, `HttpRequestBuilder`); default reqwest-backed, with optional
    `reqwest-middleware` client
- `agentkit-adapter-completions`
  - generic OpenAI-compatible chat completions adapter base
- `agentkit-provider-openrouter`
  - OpenRouter model adapter
- `agentkit-provider-openai`
  - OpenAI model adapter
- `agentkit-provider-anthropic`
  - Anthropic Messages API adapter; implements `ModelAdapter` directly (the
    Messages API is not OpenAI-compatible)
- `agentkit-provider-cerebras`
  - Cerebras Inference API adapter; implements `ModelAdapter` directly to
    carry provider-specific surface (msgpack/gzip request compression,
    `X-Cerebras-Version-Patch`, typed reasoning config, strict JSON-Schema
    output, rate-limit snapshot, Files + Batch) that does not fit the
    `CompletionsProvider` hook shape
- `agentkit-provider-ollama`
  - Ollama model adapter
- `agentkit-provider-vllm`
  - vLLM model adapter
- `agentkit-provider-groq`
  - Groq model adapter
- `agentkit-provider-mistral`
  - Mistral model adapter
- `agentkit`
  - umbrella crate with feature flags

## Control flow

The main runtime path is:

1. host constructs an `Agent`
2. host starts a `LoopDriver`
3. host submits input items
4. `LoopDriver::next()` runs a model turn
5. non-blocking activity is emitted as `AgentEvent`
6. blocking control points are returned as `LoopStep::Interrupt(...)`
7. when the turn completes, `LoopStep::Finished(...)` is returned

## Blocking vs non-blocking boundaries

Blocking:

- approval requests
- auth requests
- awaiting input

Non-blocking:

- deltas
- usage updates
- tool-call observations
- compaction observations
- warnings

Non-blocking events go to loop observers and reporting adapters. They do not control loop progress.

## Tool execution path

Tool calls requested by the model go through:

1. `ToolRegistry`
2. `ToolExecutor`
3. permission preflight
4. tool invocation
5. normalization into transcript `ToolResult` items

Approval and auth both pause the loop and can resume the original operation.

## MCP path

MCP is split into two layers:

- transport/lifecycle/discovery in `agentkit-mcp`
- adaptation into tools and capabilities for the rest of the stack

MCP auth requests now carry typed operation data and can be resumed through:

- loop auth resume for tool-path interruptions
- `McpServerManager::resolve_auth_and_resume(...)` for non-tool MCP operations

## Compaction path

Compaction is optional.

If configured, the loop asks the compaction trigger whether transcript replacement should happen before a turn. If so:

1. `AgentEvent::CompactionStarted` is emitted
2. the configured compaction strategy or pipeline receives the current transcript
3. the loop replaces its in-memory transcript with the result
4. `AgentEvent::CompactionFinished` is emitted

The loop owns when compaction hooks are checked, but not how compaction is performed. Semantic compaction, if needed, is provided through an injected backend rather than a built-in model client.
