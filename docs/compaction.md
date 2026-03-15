# agentkit-compaction

`agentkit-compaction` is the optional transcript replacement layer.

It is intentionally narrow in v1.

## Core types

- `CompactionTrigger`
  - decides whether compaction should run
- `Compactor`
  - produces a replacement transcript
- `CompactionRequest`
  - current transcript, session, turn, and reason
- `CompactionResult`
  - replacement transcript plus metadata
- `CompactionReason`
  - why compaction was triggered

## Included helper

- `ItemCountTrigger`
  - triggers when transcript length exceeds a configured item count

## Loop integration

Compaction is configured on the agent builder:

```rust
let agent = Agent::builder()
    .model(adapter)
    .compaction(CompactionConfig::new(
        ItemCountTrigger::new(12),
        my_compactor,
    ))
    .build()?;
```

When compaction is configured, `agentkit-loop` emits:

- `AgentEvent::CompactionStarted`
- `AgentEvent::CompactionFinished`

The loop currently checks compaction before a turn begins. The compactor decides how much transcript to keep or replace.

## Scope

This crate does not implement:

- memory products
- retrieval
- summarization prompts
- persistence

It is only the hook and artifact boundary for transcript replacement.
