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
- `agentkit-plugins`
  - Agent Plugins package validation and portable skill/MCP asset discovery; runtime composition stays in the satellite crates
- `agentkit-acp`
  - Agent Client Protocol integration: session binding, observer routing, prompt
    conversion, approval resolvers, and a headless stdio runtime built on the
    official `agent-client-protocol` SDK
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
- `agentkit-provider-baseten`
  - Baseten Model API adapter
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

## ACP path

ACP makes agent sessions addressable by external clients (editors, IDEs). The
integration is split the same way as MCP:

- protocol schema, JSON-RPC framing, and transports live in the upstream
  `agent-client-protocol` SDK, re-exported by `agentkit-acp`
- agentkit owns the glue: `AcpIntegration` implements `LoopObserver` and routes
  session-addressed `ObservedEvent`s into per-session ACP `session/update`
  notifications, `AcpInputPort` converts ACP prompts into input items,
  per-session `CancellationController`s service `session/cancel`, and
  `AcpApprovalResolver` bridges loop approval interrupts to ACP
  `session/request_permission`

Hybrid hosts wire those pieces into their own `AgentBuilder`; standalone agents
use `AcpHeadlessRuntime`, which owns the session table and serves ACP over
stdio or a custom transport.

## Mutator path

Transcript edits — compaction, redaction, repair — plug into the loop through one generic seam, `LoopMutator`. Mutators run at well-defined `MutationPoint`s (`AfterToolResult`, `AfterTurnEnded`) and decide for themselves whether to touch the transcript via a `TranscriptCursor`.

For each registered mutator the loop:

1. emits `AgentEvent::MutationStarted { mutator, point, .. }` (mutators are expected to emit their own start/finish events with a stable label)
2. runs the mutator with read-only context and a mutable cursor over the transcript
3. emits `AgentEvent::MutationFinished { mutator, dirty, metadata, .. }`

If any mutator in the pass dirtied the cursor, the loop validates transcript invariants (tool_use ↔ tool_result pairing) and hard-fails with `LoopError::Mutator` if a mutator left the transcript protocol-invalid.

Compaction is the canonical mutator: `agentkit-compaction` provides a `Compactor` trait, `StrategyCompactor`, and trigger helpers (`item_count_trigger`, `context_window_trigger`) that wire into this seam through the `AgentBuilderCompactorExt::compactor` extension. Semantic compaction is provided through an injected `CompactionBackend` (e.g. `AgentCompactor`, which runs a nested sub-agent) rather than a built-in model client.

## Recovering from context-length rejection

Enable context recovery when your host can compact or otherwise reduce conversation history after a provider rejects an oversized request:

```rust,ignore
use agentkit_loop::ContextRecoveryPolicy;
use std::time::Duration;

let builder = Agent::builder()
    .context_recovery(ContextRecoveryPolicy {
        max_attempts: 3,
        budget: Duration::from_secs(60),
    });
```

Register a `LoopMutator` that handles `MutationPoint::ContextLengthRejected`. At that point, `LoopCtx::recovery` provides the one-based attempt and an absolute deadline. The mutator chooses what history to summarize or shorten; AgentKit does not estimate token counts or implement a degradation policy. Preserve applicable instructions, the current user request, and complete tool-call/result relationships.

Recovery is disabled unless configured. It starts only on a typed context-length rejection, never a generic HTTP 400, authentication error, or rate limit. The provider adapter must preserve that classification; `LoopError::is_context_length_exceeded()` exposes it without parsing rendered error messages. Supported classification includes OpenAI Responses HTTP/SSE/WebSocket errors, OpenAI chat-completions HTTP errors, and Anthropic HTTP/SSE prompt-length errors. Other compatible-provider routes may still return an unclassified error.

Recovery mutators edit a staged transcript. AgentKit validates and commits the candidate only after the mutation pass succeeds within its cancellation and deadline bounds. An unchanged candidate stops recovery. Failed, cancelled, invalid, or timed-out mutation passes do not replace canonical history. Mutators must not persist staged candidates themselves: implement `TranscriptObserver::on_transcript_replaced` to update a persisted or mirrored transcript after commitment.

The driver retries only model generation, preserving the logical turn and avoiding repeated tool execution. Recovery is refused after model response content has been exposed. If a rejected attempt emitted usage, recovery also requires `SessionConfig::with_response_attempt_supersession()`: the host must discard that attempt's usage when it receives `ResponseAttemptSuperseded`, before handling replacement output. Without that capability, the rejection terminates without a recovery mutation. All recovery attempts and replacement generation share the configured budget, starting at the first rejection; zero attempts or a zero budget disables recovery. The deadline is cooperative and cannot preempt blocking synchronous code.

If the retry limit is reached or protected context cannot be reduced, surface the failure rather than discarding instructions or repeatedly submitting the same request. A successful summary does not guarantee the resumed request fits: instructions, tools, attachments, or the current request can themselves exceed the provider's limit.
