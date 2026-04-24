# Reporting and observability

An agent that you can't observe is an agent you can't debug. This chapter covers `agentkit-reporting`: how events flow from the loop to observers, and the built-in reporter implementations.

## The observer contract

```rust
pub trait LoopObserver: Send {
    fn handle_event(&mut self, event: AgentEvent);
}
```

Observers are synchronous and called in deterministic order. This is a deliberate choice:

- **Deterministic ordering** — if event B depends on event A, observers always see A first
- **No async leakage** — the loop stays runtime-agnostic
- **Simple reasoning** — observer behavior is fully predictable

The cost is that observers must be fast. Heavy processing should happen behind a channel adapter.

```text
Event flow:

  LoopDriver
       │
       ├── emit(AgentEvent)
       │        │
       │        ├──▶ Observer 1 (StdoutReporter)    → print to terminal
       │        ├──▶ Observer 2 (JsonlReporter)      → write to log file
       │        └──▶ Observer 3 (UsageReporter)      → accumulate counters
       │
       │   Observers are called in registration order.
       │   Each observer blocks until it returns.
       │   Total time = sum of all observer handle_event() calls.
       │
       └── continue loop execution
```

## Built-in reporters

### StdoutReporter

Human-readable terminal output. Handles streaming text deltas, tool lifecycle notices, approval prompts, and turn summaries. Intentionally conservative — line-oriented output, no cursor management or advanced TUI tricks.

### JsonlReporter

One structured JSON object per event, newline-delimited. Useful for audit logs, debugging, and external system ingestion. Uses a stable envelope format with event type, timestamp, session ID, turn ID, and payload.

### UsageReporter

Aggregates token usage across a session: input tokens, output tokens, reasoning tokens, cached input tokens, cache write tokens, estimated cost. Exposes query methods for per-turn and cumulative totals.

### TranscriptReporter

Reconstructs an inspectable transcript from events. Useful for debugging, persistence, and testing. Important constraint: the reporter reconstructs a _derived_ view — the loop owns the authoritative working transcript.

### CompositeReporter

Fans out events to multiple child reporters:

```rust
let reporter = CompositeReporter::new()
    .with_observer(StdoutReporter::new(std::io::stderr()))
    .with_observer(JsonlReporter::new(file))
    .with_observer(UsageReporter::new());
```

## Adapter reporters

For expensive or async reporting:

- **BufferedReporter** — enqueues events for batch flushing
- **ChannelReporter** — forwards events to another thread or task via a sender
- **TracingReporter** — converts events into tracing spans and events

These adapters wrap the synchronous observer contract without changing it.

## Failure policy

Reporter failures are non-fatal by default. A broken log writer shouldn't crash the agent. Hosts can configure stricter behavior:

- `Ignore` — swallow errors
- `Log` — log errors to stderr
- `Accumulate` — collect errors for later inspection
- `FailFast` — abort on first error

## Writing a custom observer

The trait is simple enough that custom observers are straightforward:

```rust
struct ToolCallCounter {
    count: usize,
}

impl LoopObserver for ToolCallCounter {
    fn handle_event(&mut self, event: AgentEvent) {
        if matches!(event, AgentEvent::ToolCallRequested(_)) {
            self.count += 1;
        }
    }
}
```

A more practical example — a reporter that writes tool calls to a structured log:

```rust
struct AuditLogger {
    writer: BufWriter<File>,
}

impl LoopObserver for AuditLogger {
    fn handle_event(&mut self, event: AgentEvent) {
        match &event {
            AgentEvent::ToolCallRequested(call) => {
                writeln!(self.writer, "TOOL_CALL: {} input={}", call.name,
                    serde_json::to_string(&call.input).unwrap_or_default()
                ).ok();
            }
            AgentEvent::ApprovalRequired(req) => {
                writeln!(self.writer, "APPROVAL_REQUIRED: {} reason={:?}",
                    req.summary, req.reason
                ).ok();
            }
            _ => {}
        }
    }
}
```

## AgentEvent categories

| Category   | Events                                                   |
| ---------- | -------------------------------------------------------- |
| Lifecycle  | `RunStarted`, `TurnStarted`, `TurnFinished`, `RunFailed` |
| Input      | `InputAccepted`                                          |
| Streaming  | `ContentDelta`                                           |
| Tools      | `ToolCallRequested`                                      |
| Approval   | `ApprovalRequired`, `ApprovalResolved`                   |
| Auth       | `AuthRequired`, `AuthResolved`                           |
| Compaction | `CompactionStarted`, `CompactionFinished`                |
| Usage      | `UsageUpdated`                                           |
| Diagnostic | `Warning`                                                |

### Event timeline for a typical turn

```text
RunStarted { session_id }
│
├── InputAccepted { items: [User("Fix the bug")] }
├── TurnStarted { session_id, turn_id: "turn-1" }
│   ├── ContentDelta(BeginPart { kind: Text })
│   ├── ContentDelta(AppendText { chunk: "I'll " })
│   ├── ContentDelta(AppendText { chunk: "read the file." })
│   ├── ContentDelta(CommitPart { part: Text("I'll read the file.") })
│   ├── ToolCallRequested(ToolCallPart { name: "fs_read_file", ... })
│   └── UsageUpdated(Usage { input: 1500, output: 200 })
│
├── TurnStarted { session_id, turn_id: "turn-2" }  ← automatic tool roundtrip
│   ├── ContentDelta(...)                            ← model response after reading file
│   ├── ToolCallRequested(ToolCallPart { name: "fs_replace_in_file", ... })
│   └── UsageUpdated(Usage { ... })
│
└── TurnFinished(TurnResult { finish_reason: Completed, ... })
```

> **Example:** [`openrouter-agent-cli`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-agent-cli) uses a composite reporter with stdout and usage reporting.
>
> **Crate:** [`agentkit-reporting`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-reporting) — depends on [`agentkit-loop`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-loop) for event types.
