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
Host                          LoopDriver
 │                               │
 │  submit_input(items)          │
 │──────────────────────────────▶│
 │                               │
 │  next().await                 │
 │──────────────────────────────▶│
 │                               ├── send transcript to model
 │                               ├── stream response
 │                               ├── execute tool calls
 │                               ├── (possibly loop internally)
 │                               │
 │  LoopStep::Finished(result)   │
 │◀──────────────────────────────│
 │                               │
 │  next().await                 │
 │──────────────────────────────▶│
 │                               │
 │  LoopStep::Interrupt(...)     │
 │◀──────────────────────────────│  needs host decision
 │                               │
 │  resolve_approval(...)        │
 │──────────────────────────────▶│  host resolves, loop resumes
 │                               │
```

There is no polling, no callback registration, and no event queue the host must drain. The `next()` call is the only synchronisation point.

## Interrupts

An interrupt pauses the loop and returns control to the host. agentkit defines three interrupt types:

```rust
pub enum LoopInterrupt {
    ApprovalRequest(PendingApproval),
    AuthRequest(PendingAuth),
    AwaitingInput(InputRequest),
}
```

| Interrupt         | Trigger                                              | Resolution                                                         |
| ----------------- | ---------------------------------------------------- | ------------------------------------------------------------------ |
| `ApprovalRequest` | A tool call requires explicit permission             | Host calls `approve()` or `deny()` on the `PendingApproval` handle |
| `AuthRequest`     | A tool needs credentials the loop doesn't have       | Host provides credentials or cancels                               |
| `AwaitingInput`   | The model finished and the loop has no pending input | Host calls `submit_input()` with new items                         |

Interrupts are the mechanism for user cancellation and external preemption. A user who wants to abort a loop heading in the wrong direction triggers a cancellation (via `CancellationController::interrupt()`), which causes the current turn to end with `FinishReason::Cancelled`. The host sees this in the `TurnResult` and can decide how to proceed — submit corrected input, adjust the system prompt, or stop entirely.

## Non-blocking events

Not everything requires host intervention. Streaming deltas, usage updates, tool lifecycle events, and compaction notifications are delivered to `LoopObserver` implementations:

```rust
pub trait LoopObserver: Send {
    fn handle_event(&mut self, event: AgentEvent);
}
```

Observers are informational — they cannot stall the loop or alter its control flow. This keeps the driver's state machine simple: `next()` either returns a `LoopStep` or doesn't return yet. There is no interleaving of observer handling with loop logic.

| Blocking (`LoopStep`)  | Non-blocking (`AgentEvent`)      |
| ---------------------- | -------------------------------- |
| `ApprovalRequest`      | `ContentDelta`                   |
| `AuthRequest`          | `ToolCallRequested`              |
| `AwaitingInput`        | `UsageUpdated`                   |
| `Finished(TurnResult)` | `CompactionStarted` / `Finished` |
|                        | `TurnStarted` / `TurnFinished`   |
|                        | `Warning`                        |

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

**Setup** — build an `Agent` with just a model adapter, start a session:

```rust
let agent = Agent::builder()
    .model(adapter)
    .cancellation(cancellation.handle())
    .build()?;

let mut driver = agent
    .start(SessionConfig {
        session_id: SessionId::new("openrouter-chat"),
        metadata: MetadataMap::new(),
        cache: Some(PromptCacheRequest {
            mode: PromptCacheMode::BestEffort,
            strategy: PromptCacheStrategy::Automatic,
            retention: Some(PromptCacheRetention::Short),
            key: None,
        }),
    })
    .await?;
```

The `cache` field configures provider-side prompt caching for the session. It is optional, but most long-running agents should set it deliberately. The full model is covered in [Chapter 15](./ch15-caching.md).

**Submit input** — construct an `Item` with `ItemKind::User` and a `TextPart`:

```rust
driver.submit_input(vec![Item {
    id: None,
    kind: ItemKind::User,
    parts: vec![Part::Text(TextPart {
        text: prompt.into(),
        metadata: MetadataMap::new(),
    })],
    metadata: MetadataMap::new(),
}])?;
```

**Drive the loop** — call `next().await` and match on the result:

```rust
match driver.next().await? {
    LoopStep::Finished(result) => {
        // Render assistant items from result.items
    }
    LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
        // Model finished, prompt user for more input
    }
    LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
        // A tool needs permission — approve or deny
    }
    LoopStep::Interrupt(LoopInterrupt::AuthRequest(request)) => {
        // A tool needs credentials
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
