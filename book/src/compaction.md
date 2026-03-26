# Compaction

`agentkit-compaction` is the optional transcript replacement layer.

## Core types

- `CompactionTrigger` — decides whether compaction should run
- `CompactionStrategy` — applies one transcript transformation
- `CompactionPipeline` — composes multiple strategies in order
- `CompactionBackend` — optional host-provided semantic summarization backend
- `CompactionContext` — gives strategies access to the optional backend and host metadata

## Built-in helpers

- `ItemCountTrigger` — triggers when transcript length exceeds a configured item count
- `DropReasoningStrategy` — removes reasoning parts, optionally drops empty items
- `DropFailedToolResultsStrategy` — removes failed tool-result parts
- `KeepRecentStrategy` — keeps the last N non-preserved items
- `SummarizeOlderStrategy` — summarizes older removable items through the optional backend

## Configuration

```rust
let agent = Agent::builder()
    .model(adapter)
    .compaction(CompactionConfig::new(
        ItemCountTrigger::new(12),
        CompactionPipeline::new()
            .with_strategy(DropReasoningStrategy::new())
            .with_strategy(DropFailedToolResultsStrategy::new())
            .with_strategy(
                KeepRecentStrategy::new(8)
                    .preserve_kind(ItemKind::System)
                    .preserve_kind(ItemKind::Context),
            ),
    ))
    .build()?;
```

For semantic summarization, inject a backend:

```rust
let config = CompactionConfig::new(trigger, strategy).with_backend(my_backend);
```

## Loop integration

When compaction is configured, `agentkit-loop` emits:

- `AgentEvent::CompactionStarted`
- `AgentEvent::CompactionFinished`

The loop checks compaction before a turn begins. The configured strategy or pipeline decides how much transcript to keep, trim, replace, or summarize.
