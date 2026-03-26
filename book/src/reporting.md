# Reporting

`agentkit-reporting` provides reusable observer implementations for consuming loop events.

## Observer contract

```rust
pub trait LoopObserver {
    fn handle_event(&mut self, event: AgentEvent);
}
```

Observers are synchronous and called in deterministic order within a driver instance.

## Built-in reporters

### `StdoutReporter`

Human-readable terminal output: text deltas, tool lifecycle, approval notices, warnings, turn summaries.

### `JsonlReporter`

One structured JSON object per event. Useful for audit logs, debugging, and external system ingestion.

### `UsageReporter`

Aggregates token usage across a session: input tokens, output tokens, reasoning tokens, estimated cost, per-turn and cumulative totals.

### `TranscriptReporter`

Reconstructs an inspectable transcript view from events. Useful for debugging, persistence, and testing.

### `CompositeReporter`

Fans out one event to multiple reporters synchronously:

```rust
pub struct CompositeReporter {
    children: Vec<Box<dyn LoopObserver>>,
}
```

## Adapters

Optional adapters for more advanced use cases:

- `BufferedReporter` — enqueue events for batch flushing
- `ChannelReporter` — forward events to another thread or task
- `TracingReporter` — convert events into tracing spans/events

## Failure policy

Reporter failures are non-fatal by default. Hosts can configure stricter behavior (Ignore, Log, Accumulate, FailFast).
