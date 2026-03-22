# agentkit-reporting

Observers for turning loop events into logs, summaries, and transcript views.

This crate provides [`LoopObserver`] implementations for [`agentkit-loop`].
Instead of baking reporting into the driver, you attach one or more reporters
to the loop and they react to every `AgentEvent` that flows through it.

## Included reporters

| Reporter | Purpose |
|---|---|
| `StdoutReporter` | Human-readable bracketed log lines (`[turn] started ...`) |
| `JsonlReporter` | Machine-readable newline-delimited JSON envelopes |
| `UsageReporter` | Aggregated token counts and cost totals |
| `TranscriptReporter` | Growing snapshot of conversation items |
| `CompositeReporter` | Fan-out wrapper that forwards events to multiple reporters |

## Quick start

Compose several reporters with `CompositeReporter` and hand it to the loop:

```rust
use agentkit_reporting::{
    CompositeReporter, JsonlReporter, StdoutReporter,
    TranscriptReporter, UsageReporter,
};

// Build a composite reporter that fans out to all four reporters.
let reporter = CompositeReporter::new()
    .with_observer(StdoutReporter::new(std::io::stderr()).with_usage(false))
    .with_observer(JsonlReporter::new(Vec::new()).with_flush_each_event(false))
    .with_observer(UsageReporter::new())
    .with_observer(TranscriptReporter::new());

// Pass `reporter` as the observer when constructing the agent loop.
```

## Accessing outputs after the loop

Reporters that accumulate state (`UsageReporter`, `TranscriptReporter`,
`JsonlReporter`) expose accessors for reading back data once the loop
finishes:

```rust
use agentkit_reporting::{UsageReporter, TranscriptReporter, JsonlReporter};

// Usage totals
let reporter = UsageReporter::new();
// ...run the loop...
let summary = reporter.summary();
println!(
    "tokens: {} in / {} out, turns: {}",
    summary.totals.input_tokens,
    summary.totals.output_tokens,
    summary.turn_results_seen,
);

// Transcript items
let reporter = TranscriptReporter::new();
// ...run the loop...
for item in &reporter.transcript().items {
    println!("{:?}: {} parts", item.kind, item.parts.len());
}

// JSONL buffer
let mut reporter = JsonlReporter::new(Vec::new());
// ...run the loop...
let jsonl = String::from_utf8(reporter.writer().clone()).unwrap();
let errors = reporter.take_errors();
assert!(errors.is_empty(), "reporting errors: {:?}", errors);
```

## Writing to a file

`JsonlReporter` and `StdoutReporter` accept any `std::io::Write`
implementation, so you can point them at files, network sockets, or
in-memory buffers:

```rust,no_run
use agentkit_reporting::JsonlReporter;
use std::io::BufWriter;
use std::fs::File;

let file = File::create("events.jsonl").expect("open file");
let reporter = JsonlReporter::new(BufWriter::new(file));
```

## Error handling

`JsonlReporter` and `StdoutReporter` never panic on write failures.
Errors are collected internally and can be drained after the loop with
`take_errors()`. This keeps the `LoopObserver::handle_event` signature
infallible while still giving you full visibility into any I/O issues.
