# agentkit-compaction

`agentkit-compaction` is the optional transcript replacement layer.

It is intentionally narrow in v1.

## Core types

- `CompactionTrigger`
  - decides whether compaction should run
- `CompactionStrategy`
  - applies one transcript transformation
- `CompactionPipeline`
  - composes multiple strategies in order
- `CompactionBackend`
  - optional host-provided semantic summarization backend
- `CompactionContext`
  - gives strategies access to the optional backend and host metadata
- `CompactionRequest`
  - current transcript, session, turn, and reason
- `CompactionResult`
  - replacement transcript plus metadata
- `CompactionReason`
  - why compaction was triggered
- `SummaryRequest` / `SummaryResult`
  - typed backend contract for semantic summary strategies

## Included helpers

- `ItemCountTrigger`
  - triggers when transcript length exceeds a configured item count
- `DropReasoningStrategy`
  - removes reasoning parts and optionally drops empty items
- `DropFailedToolResultsStrategy`
  - removes failed tool-result parts and optionally drops empty items
- `KeepRecentStrategy`
  - keeps the last N non-preserved items
- `SummarizeOlderStrategy`
  - summarizes older removable items through the optional backend

## Loop integration

Compaction is configured on the agent builder:

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

If a strategy needs semantic summarization, the host can inject a backend:

```rust
let config = CompactionConfig::new(trigger, strategy).with_backend(my_backend);
```

When compaction is configured, `agentkit-loop` emits:

- `AgentEvent::CompactionStarted`
- `AgentEvent::CompactionFinished`

The loop currently checks compaction before a turn begins. The configured strategy or pipeline decides how much transcript to keep, trim, replace, or summarize.

## Scope

This crate does not implement:

- memory products
- retrieval
- provider-specific LLM calls
- persistence

It is the hook and artifact boundary for transcript replacement and optional host-provided semantic compaction.
