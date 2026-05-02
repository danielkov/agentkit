# What is an agent loop?

A chat completion takes a transcript and returns a response. An agent loop extends this by inspecting the response for tool calls, executing them, appending the results to the transcript, and sending the updated transcript back to the model. This repeats until the model produces a response with no tool calls, or until the host intervenes.

This chapter defines the structure of that loop and maps it to agentkit's core types.

## The basic loop

An agent loop repeats five steps:

1. Send the current transcript to the model
2. Receive the model's response (which may contain text, tool calls, or both)
3. If the response contains tool calls, execute them
4. Append the tool results to the transcript
5. Go to 1

```text
┌───────────────────────────────────────────┐
│              Host application             │
│                                           │
│   submit user input                       │
│        │                                  │
│        ▼                                  │
│   ┌──────────┐   ┌────────────────────┐   │
│   │Transcript│──▶│  Model inference   │   │
│   │          │◀──│  (streaming turn)  │   │
│   └──────────┘   └────────┬───────────┘   │
│        │                  │               │
│        │          ┌───────▼───────┐       │
│        │          │ Tool calls?   │       │
│        │          └───┬───────┬───┘       │
│        │           no │       │ yes       │
│        │              ▼       ▼           │
│        │          [return] [execute]      │
│        │                      │           │
│        └──────────────────────┘           │
│              append results               │
└───────────────────────────────────────────┘
```

The number of iterations is determined by the model at runtime. The loop may execute zero tool calls (a plain text response) or dozens across multiple turns. This is what distinguishes an agent from a pipeline — the control flow is dynamic.

## Loop vs pipeline

In a pipeline, data flows through a fixed sequence of stages. The topology is known at compile time. An agent loop has a dynamic topology: the model decides which tools to call, in what order, and how many times.

This has architectural consequences. A pipeline framework optimises for stage composition and throughput. An agent framework must handle:

- **variable iteration count** — the loop runs until the model stops requesting tools
- **interrupt points** — the host may need to intervene mid-loop (user cancellation, approval gates, auth challenges)
- **transcript growth** — each iteration adds items, eventually requiring compaction
- **parallel execution** — independent tool calls within a single turn can run concurrently

## The control boundary

The central design question is: where does the framework yield control to the host?

agentkit uses a pull-based model. The host calls `driver.next().await` and receives one of two outcomes:

```rust
pub enum LoopStep {
    Interrupt(LoopInterrupt),
    Finished(TurnResult),
}
```

`Finished` means the model completed a turn — the host can inspect the results, submit new input, and call `next()` again. `Interrupt` means the loop cannot proceed without host action.

```text
Host                                    LoopDriver
 │                                         │
 │  Agent::builder()                       │
 │    .transcript(prior)                   │  passive, optional
 │    .input([first_user_msg])             │  optional one-shot opener
 │    .build()?                            │
 │                                         │
 │  agent.start(cfg)                       │
 │────────────────────────────────────────▶│
 │                                         │
 │  next().await                           │
 │────────────────────────────────────────▶│
 │                                         │
 │  if no input was preloaded:             │
 │  LoopStep::Interrupt(AwaitingInput)     │
 │◀────────────────────────────────────────│  no input queued yet
 │  req.submit(driver, [first_user_msg])   │
 │────────────────────────────────────────▶│
 │  next().await                           │
 │────────────────────────────────────────▶│
 │                                         ├── send transcript to model
 │                                         ├── stream response
 │                                         ├── execute tool calls
 │                                         ├── (possibly loop internally)
 │                                         │
 │  LoopStep::Finished(result)             │
 │◀────────────────────────────────────────│
 │                                         │
 │  next().await                           │
 │────────────────────────────────────────▶│
 │                                         │
 │  LoopStep::Interrupt(...)               │
 │◀────────────────────────────────────────│  needs host decision
 │                                         │
 │  pending.approve(driver)                │
 │────────────────────────────────────────▶│  host resolves, loop resumes
 │                                         │
```

There is no polling, no callback registration, and no event queue the host must drain. The `next()` call is the only synchronisation point.

## Interrupts

An interrupt pauses the loop and returns control to the host. agentkit defines three interrupt types, split into blocking (must be resolved before the loop continues) and cooperative (host may ignore the yield and call `next()` again):

```rust
pub enum LoopInterrupt {
    ApprovalRequest(PendingApproval),     // blocking
    AwaitingInput(InputRequest),          // cooperative
    AfterToolResult(ToolRoundInfo),       // cooperative
}
```

| Interrupt         | Trigger                                                   | Resolution                                                                        |
| ----------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `ApprovalRequest` | A tool call requires explicit permission                  | Host calls `approve()` or `deny()` on the `PendingApproval` handle                |
| `AwaitingInput`   | The model finished and the loop has no pending input      | Host calls `req.submit(&mut driver, items)` on the `InputRequest` handle          |
| `AfterToolResult` | A tool round completed, model is about to be called again | Optional — host calls `info.submit(&mut driver, items)` to interject, or `next()` |

> **MCP auth.** Earlier versions of agentkit surfaced `AuthRequest` as a fourth loop interrupt. Auth is now an MCP-only concept: the manager raises `McpError::AuthRequired(AuthRequest)` from non-tool operations and `ToolError::AuthRequired(_)` from tool calls, and the host resolves them out-of-band via [`McpServerManager::resolve_auth`](./ch17-mcp.md). The loop never blocks on auth.
>
> This pattern generalises. Userland and third-party tools can define their own strongly-typed pause-and-resume protocols (think payment confirmations, hardware key taps, multi-step wizards) by surfacing a tool-specific error variant and exposing a resolver on the tool handle. The host catches the typed error, drives the resolution through the tool's own API, and retries. The loop interrupt enum stays closed: there is no need to plumb every new interaction shape through `LoopInterrupt`, and no need for a catch-all `Custom(Box<dyn Any>)` variant that would erase the type at the boundary.

Interrupts are the mechanism for user cancellation and external preemption. A user who wants to abort a loop heading in the wrong direction triggers a cancellation (via `CancellationController::interrupt()`), which causes the current turn to end with `FinishReason::Cancelled`. The host sees this in the `TurnResult` and can decide how to proceed — submit corrected input, adjust the system prompt, or stop entirely.

## Non-blocking events

Not everything requires host intervention. Streaming deltas, usage updates, tool lifecycle events, and compaction notifications are delivered to `LoopObserver` implementations:

```rust
pub trait LoopObserver: Send {
    fn handle_event(&mut self, event: AgentEvent);
}
```

Observers are informational — they cannot stall the loop or alter its control flow. This keeps the driver's state machine simple: `next()` either returns a `LoopStep` or doesn't return yet. There is no interleaving of observer handling with loop logic.

| Control flow (`LoopStep`) | Observation (`AgentEvent`)                 |
| ------------------------- | ------------------------------------------ |
| `ApprovalRequest`         | `ContentDelta`                             |
| `AwaitingInput`           | `ToolCallRequested` / `ToolResultReceived` |
| `AfterToolResult`         | `ToolCatalogChanged`                       |
| `Finished(TurnResult)`    | `UsageUpdated`                             |
|                           | `CompactionStarted` / `CompactionFinished` |
|                           | `TurnStarted` / `TurnFinished` / `Warning` |

For loss-free transcript reconstruction (persistence, replication, audit), register a `TranscriptObserver` alongside the operational `LoopObserver`. The two observe different things:

- **`LoopObserver`** sees a stream of `AgentEvent`s. Content arrives as deltas — partial fragments that don't carry their parent-`Item` identity — interleaved with lifecycle and telemetry events. Useful for UIs and logging, but a consumer cannot reassemble the canonical transcript from this stream alone.
- **`TranscriptObserver`** fires exactly once per `Item` appended, with the fully-formed `Item` ready to persist. Calls happen synchronously from the driver, in transcript order, at the single mutation point that owns the transcript — so what the observer sees is what the loop will send to the model on the next turn.

Compaction rewrites the transcript without firing `on_item_appended`; those rewrites are signalled separately by `AgentEvent::CompactionFinished`, which a persistence layer can use to snapshot the post-compaction state.

## The three-layer model

agentkit splits the runtime into three layers:

```text
┌─────────────────────────────────────────────┐
│  Agent                                      │
│  (config: adapter, tools, permissions,      │
│   observers, compaction)                    │
│                                             │
│   ┌─────────────────────────────────────┐   │
│   │  LoopDriver                         │   │
│   │  (mutable state: transcript,        │   │
│   │   pending input, interrupt state)   │   │
│   │                                     │   │
│   │   ┌─────────────────────────────┐   │   │
│   │   │  ModelSession               │   │   │
│   │   │  (provider connection,      │   │   │
│   │   │   turn management)          │   │   │
│   │   └─────────────────────────────┘   │   │
│   └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

- **`Agent`** — immutable configuration assembled via a builder. Holds the model adapter, tool registry, permission checker, observers, and compaction config. Can start multiple sessions.
- **`LoopDriver<S>`** — the mutable runtime for a single session. Owns the transcript, manages pending input, tracks interrupt state, and drives the turn loop. Generic over the session type `S`.
- **`ModelSession`** — the provider-owned session handle. Created by the adapter, consumed by the driver. Each turn calls `begin_turn()` which returns a streaming `ModelTurn`.

This separation means: configure once, run many sessions. Or run multiple concurrent sessions from the same `Agent`, each with its own `LoopDriver` and independent transcript.

## A minimal example

The [`openrouter-chat`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-chat) example demonstrates the simplest possible host loop. The key parts:

**Setup** — build an `Agent` with just a model adapter and a preloaded first user message, then start the session:

```rust
let agent = Agent::builder()
    .model(adapter)
    .cancellation(cancellation.handle())
    // Optional — preload a prior transcript (system prompt or resumed
    // session) and the next user turn so the first next() drives a turn
    // directly. Both default to empty.
    .input(vec![Item::text(ItemKind::User, prompt)])
    .build()?;

let mut driver = agent
    .start(SessionConfig::new("openrouter-chat").with_cache(
        PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
    ))
    .await?;
```

`AgentBuilder::transcript` takes the prior transcript — typically `[system_item]` for a fresh session, or a transcript loaded from a database / file when resuming. It is loaded passively. `AgentBuilder::input` preloads the next user turn into the driver's pending-input queue: when non-empty, the first `next()` dispatches the model directly; when left empty (the default for turn-based loops), the first `next()` yields `AwaitingInput` and the host supplies input via `InputRequest::submit`.

The session-level `cache` configures prompt caching for the session. It is optional, but most long-running agents benefit from setting it. See [Chapter 15](./ch15-caching.md) for the full cache request shape.

**Drive the loop** — call `next().await` and match on the result:

```rust
match driver.next().await? {
    LoopStep::Finished(result) => {
        // Render assistant items from result.items
    }
    LoopStep::Interrupt(LoopInterrupt::AwaitingInput(req)) => {
        // Model finished — call req.submit(&mut driver, more_items)? to
        // continue the conversation, or drop the handle to stop.
    }
    LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => {
        // Tool round done. Non-interactive callers just loop back; an
        // interactive host may call info.submit(&mut driver, items)?
        // to interject a user message before the next model call.
    }
    LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
        // A tool needs permission — pending.approve(&mut driver)? /
        // pending.deny(&mut driver)?
    }
}
```

This is the entire host-side contract. The loop handles streaming, tool execution, transcript accumulation, and compaction internally. The host only sees `LoopStep` values — either results to render or interrupts to resolve.

No tools are registered in this example, so the model cannot make tool calls and the loop always returns `Finished` after a single inference turn. Adding tools is covered in [Chapter 9](./ch09-tool-system.md).

## What comes next

The following chapters build up each piece of this system:

- [Chapter 3](./ch03-transcript-model.md) defines the data model — the `Item` and `Part` types that make up the transcript
- [Chapter 4](./ch04-streaming-and-deltas.md) covers how streaming works and how deltas fold into durable parts
- [Chapter 5](./ch05-model-adapter.md) defines the boundary between the loop and model providers
- [Chapter 6](./ch06-driving-the-loop.md) walks through the driver implementation
- [Chapter 7](./ch07-interrupts-and-control-flow.md) covers the interrupt system in detail
