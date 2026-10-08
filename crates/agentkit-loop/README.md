# agentkit-loop

<p align="center">
  <a href="https://crates.io/crates/agentkit-loop"><img src="https://img.shields.io/crates/v/agentkit-loop.svg?logo=rust" alt="Crates.io" /></a>
  <a href="https://docs.rs/agentkit-loop"><img src="https://img.shields.io/docsrs/agentkit-loop?logo=docsdotrs" alt="Documentation" /></a>
  <a href="https://github.com/danielkov/agentkit/blob/main/LICENSE"><img src="https://img.shields.io/crates/l/agentkit-loop.svg" alt="License" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/MSRV-1.92-blue?logo=rust" alt="MSRV" /></a>
</p>

Runtime-agnostic agent loop orchestration for sessions, turns, tools, and interrupts.

This crate provides:

- **Model adapter traits** -- `ModelAdapter`, `ModelSession`, and `ModelTurn` abstract away the model provider so you can swap between OpenRouter, Anthropic, or a local LLM without changing loop logic.
- **`Agent` builder and `LoopDriver`** -- configure tools, permissions, observers, and compaction, then drive the loop step-by-step.
- **Interrupt handling** -- the loop pauses and yields `LoopStep::Interrupt` on blocking events (tool approval) and cooperative yields (`AwaitingInput` at end-of-turn, `AfterToolResult` between tool rounds). The host either resolves the interrupt or just calls `next()` again depending on whether `LoopInterrupt::is_blocking()` is `true`.
- **Observer hooks** -- attach `LoopObserver` implementations to receive streaming `AgentEvent`s (deltas, tool calls, usage, warnings, lifecycle events).
- **Transcript compaction** -- optionally compact the transcript when it grows too large, via the `agentkit-compaction` integration.

Use it as the central coordinator between model providers, tool execution, and application UI or control flow.

## Interception hooks

Besides `mutate`, a `LoopMutator` registered with `AgentBuilder::mutator` can intercept values before the loop consumes or commits them:

- `on_session_start` edits session options before the model adapter starts the session.
- `on_model_request` edits a single inference request (transcript, tools, cache, metadata). These edits are not persisted.
- `on_model_response` edits complete model output before it is committed, returned, or sent to tools. Content may change — including tool-call arguments, which reach the executor. Item identity, accounting and tool-call linkage may not. `payload.disposition` says whether the loop will dispatch this response's tool calls (`ContinueWithTools`) or take the normal finish branch (`FinishTurnCandidate`), computed from the loop's own branch predicate rather than the response content.

All hooks default to no-ops and run in registration order. Read-only notifications belong in `LoopObserver`; individual tool interception belongs at `ToolExecutor`.

## Transcript mutation

`mutate` runs at every `MutationPoint`: `TurnStarted` (once per logical turn the driver creates, with queued input already appended, and the only point that also runs for turns that never dispatch an inference), `AfterToolResult`, and `AfterTurnEnded`.

The chain is transactional. Mutators edit a candidate copy; the loop validates invariants, re-checks cancellation, then assigns the result to the live transcript in one synchronous step. A mutator that errors, produces a protocol-invalid transcript, cancels, or whose future is dropped leaves the live transcript untouched. A committed change is published to every `TranscriptObserver` as one `on_transcript_rewrite` carrying the complete canonical transcript; writing the same value back is not a change and publishes nothing.

## Awaited delivery

`LoopObserver` is synchronous and infallible. A host that must *await* its own delivery of a fact at the moment it happens registers a `NativeDelivery` with `AgentBuilder::delivery`. The driver awaits each target at the fact's emission site:

- `NativeFact::Progress` — model deltas, usage, tool calls, attempt supersession, and the loop-authored background-detach placeholder, delivered as the driver consumes them rather than buffered until `next()` returns.
- `NativeFact::BeforeFinish` — the logical turn is about to finish, before its own terminal items are appended, so a consumer sees terminal output and cancellation partials before they commit.
- `NativeFact::TurnFinished` — the turn finished and its items are committed.

The terminal pair is delivered exactly once per logical turn — including turns that end through cancellation, an error, a failed cleanup, or `retire_interrupted_turn` — for uninterrupted calls and for cooperative cancellation followed by retirement. A hard abort breaks that: `BeforeFinish` is awaited before anything commits, so dropping the `next()` future inside it loses the turn's terminal output candidate and leaves the turn active, and a later `retire_interrupted_turn` delivers a second prefinish for the same turn with a cancelled result. Treat prefinish as at-least-once if you drop driver futures.

`HookCtx::cancellation` is `None` for the terminal facts, and also `None` for `Progress` when the agent was built without `AgentBuilder::cancellation` — match on the `NativeFact` variant rather than on the handle's presence.

Delivery is read-only; a `DeliveryError` is diagnostics only — it never fails the operation, never stops the stream, never rolls a commit back, never replays and never produces a second terminal event, and later facts for the same turn still arrive. Drain it with `LoopDriver::take_delivery_errors`, which resets both the retained failures and the `dropped` count of what the bounded buffer discarded.

## Quick start

```rust,no_run
use agentkit_core::{Item, ItemKind};
use agentkit_loop::{
    Agent, LoopInterrupt, LoopStep, PromptCacheRequest, PromptCacheRetention, SessionConfig,
};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// 1. Create a model adapter
let adapter = OpenRouterAdapter::new(
    OpenRouterConfig::new("sk-or-v1-...", "openrouter/auto"),
)?;

// 2. Build an agent. Preload the system prompt and first user turn so the
//    very first `next()` call dispatches the model directly.
let agent = Agent::builder()
    .model(adapter)
    .transcript(vec![Item::text(ItemKind::System, "You are a helpful assistant.")])
    .input(vec![Item::text(ItemKind::User, "Hello, agent!")])
    .build()?;

// 3. Start a session to get a LoopDriver
let mut driver = agent
    .start(
        SessionConfig::new("demo").with_cache(
            PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
        ),
    )
    .await?;

// 4. Drive the loop. Subsequent user turns are supplied via the
//    `InputRequest::submit` handle yielded by `LoopInterrupt::AwaitingInput`.
loop {
    match driver.next().await? {
        LoopStep::Finished(result) => {
            println!("Turn finished ({:?}): {:?}", result.finish_reason, result.items);
            break;
        }
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
            // No more input to feed in this example; stop here.
            break;
        }
        LoopStep::Interrupt(interrupt) => {
            // See "Handling interrupts" below for how to resolve each variant.
            println!("Loop paused: {interrupt:?}");
            break;
        }
    }
}
# Ok(())
# }
```

## Adding tools and observers

`AgentBuilder::add_tool_source` accepts any `ToolSource`. A `ToolRegistry`
implements `ToolSource` directly, so you can hand it in by value; call the
method again to federate additional sources (MCP catalogs, plugin loaders,
etc.).

```rust,no_run
use agentkit_loop::{Agent, LoopObserver, ObservedEvent};
use agentkit_tools_core::ToolRegistry;

struct PrintObserver;

impl LoopObserver for PrintObserver {
    fn handle_event(&self, event: ObservedEvent) {
        println!("[event] {:?}", event.event);
    }
}

# fn example<M: agentkit_loop::ModelAdapter>(adapter: M, registry: ToolRegistry) -> Result<(), agentkit_loop::LoopError> {
let agent = Agent::builder()
    .model(adapter)
    .add_tool_source(registry)
    .observer(PrintObserver)
    .build()?;
# Ok(())
# }
```

## Handling interrupts

When a tool call requires approval the loop yields a blocking interrupt;
`AwaitingInput` and `AfterToolResult` are cooperative (use
`LoopInterrupt::is_blocking` to tell them apart). Resolve any pending
approval and call `next()` again to resume:

```rust,no_run
use agentkit_core::{Item, ItemKind};
use agentkit_loop::{LoopInterrupt, LoopStep};

# async fn handle<S: agentkit_loop::ModelSession>(
#     driver: &mut agentkit_loop::LoopDriver<S>,
# ) -> Result<(), agentkit_loop::LoopError> {
loop {
    match driver.next().await? {
        LoopStep::Finished(result) => {
            println!("Done: {:?}", result.finish_reason);
            break;
        }
        LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
            println!("Approve {}? (auto-approving)", pending.summary);
            pending.approve(driver)?;
        }
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(request)) => {
            // Hand the next user turn to the driver, or break to stop.
            request.submit(driver, vec![Item::text(ItemKind::User, "continue")])?;
        }
        // Cooperative yield between tool rounds. Interactive hosts may use
        // `info.submit(driver, items)` to interject a user message before
        // the next model call; non-interactive callers just loop.
        LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_info)) => continue,
    }
}
# Ok(())
# }
```

## OpenTelemetry telemetry

The loop emits GenAI `tracing` spans without requiring an OpenTelemetry SDK. Enable the optional `otel` feature when exporting through `tracing-opentelemetry`; this preserves token counts as signed 64-bit attributes and provider-native finish reasons as string-array attributes. With that feature enabled, usage, finish-reason, and message attributes are written directly to the OpenTelemetry span and are not visible to a plain `tracing-subscriber` formatting layer.

Message content capture is always off by default and is configured only in code. Input and output limits are independent and bounded:

```rust,ignore
use agentkit_loop::{Agent, MessageCapture, TelemetryConfig};

let agent = Agent::builder()
    .model(adapter)
    .telemetry(
        TelemetryConfig::default()
            .with_input_messages(MessageCapture::new(32, 16 * 1024)?)
            .with_output_messages(MessageCapture::new(16, 8 * 1024)?),
    )
    .build()?;
```

AgentKit does not read `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. `MessageCapture::new` rejects zero message and byte limits; it never silently clamps them. The exported `gen_ai.input.messages` and `gen_ai.output.messages` attributes are OpenTelemetry `Array<String>` values whose elements are compact valid JSON. Input capture keeps the newest bounded tail in transcript order; output capture keeps the bounded head. Data references are omitted, so inline image/audio/binary data and URLs are neither exported nor dereferenced. An item that exceeds the remaining source-content budget becomes a structured JSON truncation record, including when the configured byte budget is tiny.
