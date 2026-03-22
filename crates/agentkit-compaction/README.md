# agentkit-compaction

Transcript compaction primitives for reducing context size while preserving useful state.

This crate includes:

- **Triggers** (`CompactionTrigger`, `ItemCountTrigger`) that decide _when_ compaction should run
- **Strategies** (`DropReasoningStrategy`, `DropFailedToolResultsStrategy`, `KeepRecentStrategy`, `SummarizeOlderStrategy`) that drop, keep, or summarize transcript items
- **Backends** (`CompactionBackend`) for provider-backed summarization
- **Pipelines** (`CompactionPipeline`) for composing multiple compaction steps into a single pass

Use it from `agentkit-loop` or your own runtime when you need to trim older transcript state without losing essential context.

## Quick start

Combine a trigger with a multi-step pipeline to build a `CompactionConfig`:

```rust
use agentkit_compaction::{
    CompactionConfig, CompactionPipeline, DropFailedToolResultsStrategy,
    DropReasoningStrategy, ItemCountTrigger, KeepRecentStrategy,
};
use agentkit_core::ItemKind;

// Trigger compaction once the transcript exceeds 32 items.
let trigger = ItemCountTrigger::new(32);

// Build a pipeline that:
//   1. Strips chain-of-thought reasoning parts
//   2. Removes failed tool results
//   3. Keeps only the 24 most recent items (preserving system/context)
let pipeline = CompactionPipeline::new()
    .with_strategy(DropReasoningStrategy::new())
    .with_strategy(DropFailedToolResultsStrategy::new())
    .with_strategy(
        KeepRecentStrategy::new(24)
            .preserve_kind(ItemKind::System)
            .preserve_kind(ItemKind::Context),
    );

let config = CompactionConfig::new(trigger, pipeline);
assert!(config.backend.is_none());
```

## Using a summarization backend

When you want older items to be condensed rather than dropped, use
`SummarizeOlderStrategy` together with a `CompactionBackend`:

```rust
use agentkit_compaction::{
    CompactionBackend, CompactionConfig, CompactionError, CompactionPipeline,
    DropReasoningStrategy, ItemCountTrigger, SummarizeOlderStrategy, SummaryRequest,
    SummaryResult,
};
use agentkit_core::{Item, ItemKind, MetadataMap, Part, TextPart, TurnCancellation};
use async_trait::async_trait;

struct MyBackend;

#[async_trait]
impl CompactionBackend for MyBackend {
    async fn summarize(
        &self,
        request: SummaryRequest,
        _cancellation: Option<TurnCancellation>,
    ) -> Result<SummaryResult, CompactionError> {
        // Call your LLM here to produce a summary.
        let summary_text = format!("Summary of {} items", request.items.len());
        Ok(SummaryResult {
            items: vec![Item {
                id: None,
                kind: ItemKind::Context,
                parts: vec![Part::Text(TextPart {
                    text: summary_text,
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }],
            metadata: MetadataMap::new(),
        })
    }
}

let config = CompactionConfig::new(
    ItemCountTrigger::new(64),
    CompactionPipeline::new()
        .with_strategy(DropReasoningStrategy::new())
        .with_strategy(
            SummarizeOlderStrategy::new(16)
                .preserve_kind(ItemKind::System),
        ),
)
.with_backend(MyBackend);

assert!(config.backend.is_some());
```

## Checking the trigger manually

You can query the trigger yourself outside of the agent loop:

```rust
use agentkit_compaction::{CompactionTrigger, ItemCountTrigger};
use agentkit_core::SessionId;

let trigger = ItemCountTrigger::new(10);
let transcript = Vec::new();
assert!(trigger.should_compact(&SessionId::new("sess-1"), None, &transcript).is_none());
```
