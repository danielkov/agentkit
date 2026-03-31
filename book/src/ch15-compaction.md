# Transcript compaction

Long conversations exceed context windows. Compaction is how you keep an agent session viable without losing important context. This chapter covers `agentkit-compaction`: the trigger, strategy, and pipeline system.

## The design

Compaction is optional and host-configured. It has three concerns:

1. **When** to compact — the trigger
2. **How** to compact — the strategy (or pipeline of strategies)
3. **What** to use for semantic summarization — the optional backend

```rust
let agent = Agent::builder()
    .model(adapter)
    .compaction(CompactionConfig::new(trigger, strategy))
    .build()?;
```

## Triggers

A `CompactionTrigger` decides whether compaction should run before a turn:

```rust
pub trait CompactionTrigger {
    fn should_compact(&self, transcript: &[Item], reason: &CompactionReason) -> bool;
}
```

Built-in: `ItemCountTrigger::new(12)` fires when the transcript exceeds 12 items.

## Strategies

A `CompactionStrategy` transforms the transcript:

```rust
pub trait CompactionStrategy {
    async fn compact(
        &self,
        request: CompactionRequest,
        ctx: &mut CompactionContext,
    ) -> Result<CompactionResult, CompactionError>;
}
```

Built-in strategies:

| Strategy                        | Description                                  |
| ------------------------------- | -------------------------------------------- |
| `DropReasoningStrategy`         | Removes reasoning parts from assistant items |
| `DropFailedToolResultsStrategy` | Removes tool results where `is_error: true`  |
| `KeepRecentStrategy`            | Keeps the last N non-preserved items         |
| `SummarizeOlderStrategy`        | Summarizes older items through the backend   |

### Preservation

`KeepRecentStrategy` supports preservation rules:

```rust
KeepRecentStrategy::new(8)
    .preserve_kind(ItemKind::System)
    .preserve_kind(ItemKind::Context)
```

System and context items are kept regardless of age. Only user/assistant/tool items are subject to trimming.

## Pipelines

Multiple strategies compose into a pipeline:

```rust
CompactionPipeline::new()
    .with_strategy(DropReasoningStrategy::new())
    .with_strategy(DropFailedToolResultsStrategy::new())
    .with_strategy(KeepRecentStrategy::new(8)
        .preserve_kind(ItemKind::System)
        .preserve_kind(ItemKind::Context))
```

Strategies execute in order. Each one receives the output of the previous.

## Semantic compaction

For summarization, the host injects a `CompactionBackend`:

```rust
let config = CompactionConfig::new(trigger, strategy).with_backend(my_backend);
```

The backend receives a `SummaryRequest` and returns a `SummaryResult`. agentkit does not include a built-in LLM client — the backend is host-provided. The [`openrouter-compaction-agent`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-compaction-agent) example uses a nested agent loop as the summarization backend.

## A compaction example

Before and after a compaction pipeline run:

```text
Before (20 items, trigger threshold: 12):

  [0]  System: "You are a coding assistant"           ← preserved
  [1]  Context: "Project uses Rust 2024..."            ← preserved
  [2]  User: "What files are in src/?"
  [3]  Asst: (reasoning) "Let me list the directory"
             (text) "I'll check..."
             (tool_call) fs.list_directory
  [4]  Tool: ["main.rs", "lib.rs", "parser.rs"]
  [5]  Asst: "There are three files..."
  [6]  User: "Read parser.rs"
  [7]  Asst: (tool_call) fs.read_file
  [8]  Tool: "fn parse() { ... }"
  [9]  Asst: "The parser contains..."
  [10] User: "Add error handling"
  [11] Asst: (tool_call) fs.replace_in_file
  [12] Tool: { is_error: true, "old_text not found" }  ← failed
  [13] Asst: "Let me try again..."
             (tool_call) fs.replace_in_file
  [14] Tool: "Replacement successful"
  [15] Asst: (tool_call) shell.exec("cargo check")
  [16] Tool: "Compiling... 0 errors"
  [17] Asst: "Done! I added error handling..."
  [18] User: "Now add tests"
  [19] Asst: (thinking about tests...)


Pipeline:
  1. DropReasoningStrategy     → removes reasoning parts from [3], [19]
  2. DropFailedToolResultsStrategy → removes failed result [12]
  3. KeepRecentStrategy(8, preserve System+Context)

After (10 items):

  [0]  System: "You are a coding assistant"            ← preserved
  [1]  Context: "Project uses Rust 2024..."             ← preserved
  [2]  Asst: "Let me try again..."                      ← recent 8 start here
             (tool_call) fs.replace_in_file
  [3]  Tool: "Replacement successful"
  [4]  Asst: (tool_call) shell.exec("cargo check")
  [5]  Tool: "Compiling... 0 errors"
  [6]  Asst: "Done! I added error handling..."
  [7]  User: "Now add tests"
  [8]  Asst: (now without reasoning part)
```

The model lost the early conversation but retains the system prompt, project context, and the most recent work. This is usually a good trade-off — the model's attention is strongest on recent items anyway.

## Loop integration

When compaction fires:

1. `AgentEvent::CompactionStarted` is emitted (with the trigger reason)
2. The strategy pipeline transforms the transcript
3. The loop replaces its working transcript with the compacted result
4. `AgentEvent::CompactionFinished` is emitted (with before/after item counts)

```text
Turn lifecycle with compaction:

  submit_input()
       │
       ▼
  ┌── compaction check ──┐
  │                      │
  │  trigger fires?      │
  │  yes → run pipeline  │
  │  no  → skip          │
  └──────────┬───────────┘
             │
             ▼
  begin model turn (with post-compaction transcript)
```

This happens _before_ the model sees the transcript for the next turn. The model never observes raw compaction artifacts — it just sees a shorter transcript.

### Compaction is not summarization

Most compaction strategies are structural — they drop parts or trim items without understanding semantics. `DropReasoningStrategy` removes reasoning blocks because they're verbose and not needed for future turns. `KeepRecentStrategy` drops old items because the model's attention is weakest on them.

Only `SummarizeOlderStrategy` (with a `CompactionBackend`) does semantic work — it summarizes old items into a shorter form. This requires an LLM call, which adds latency and cost. The [`openrouter-compaction-agent`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-compaction-agent) example uses a nested agent loop as the summarization backend.

> **Example:** [`openrouter-compaction-agent`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-compaction-agent) demonstrates all three types: structural (drop reasoning), hybrid (keep recent + summarize older), and semantic (nested-agent summarization backend).
>
> **Crate:** [`agentkit-compaction`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-compaction) — depends on [`agentkit-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-core). The loop integration is in [`agentkit-loop`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-loop).
