# agentkit-core

Shared primitives for agentkit transcripts, content parts, usage accounting, identifiers, and cancellation.

This crate defines the data model used across the rest of the workspace:

- transcript items and roles
- multimodal content parts
- tool call and tool result payloads
- streaming deltas
- token and cost usage
- cancellation checkpoints for turns and tools

Most other crates in the workspace depend on `agentkit-core` as their common language for messages and events.

## Building a transcript for the agent loop

Every agent turn starts with a `Vec<Item>` transcript. System instructions,
user messages, and context documents are all items; the loop appends assistant
and tool items as the turn progresses.

```rust
use agentkit_core::{Item, ItemKind};

let transcript = vec![
    Item::text(ItemKind::System, "You are a careful coding agent."),
    Item::text(ItemKind::Context, "Project uses Rust 1.80, workspace has 12 crates."),
    Item::text(ItemKind::User, "Summarize the release notes."),
];

assert_eq!(transcript.len(), 3);
assert_eq!(transcript[0].kind, ItemKind::System);
```

## Representing tool calls and results

When the model invokes a tool the loop emits a `ToolCallPart`. After execution
the tool executor wraps the output in a `ToolResultPart` and appends it back
to the transcript as a `Tool` item so the model can observe the result.

```rust
use agentkit_core::{
    Item, ItemKind, Part, ToolCallId, ToolCallPart, ToolOutput, ToolResultPart,
};
use serde_json::json;

// The model asks to read a file.
let tool_call = ToolCallPart::new(
    ToolCallId::new("call-1"),
    "fs_read_file",
    json!({ "path": "CHANGELOG.md" }),
);

// After execution, the tool executor produces a result item.
let tool_result_item = Item::new(
    ItemKind::Tool,
    vec![Part::ToolResult(ToolResultPart::success(
        tool_call.id.clone(),
        ToolOutput::text("## v0.3.0\n- Added compaction."),
    ))],
);

assert!(matches!(tool_result_item.parts[0], Part::ToolResult(_)));
```

## Tracking token usage across turns

`Usage` and `TokenUsage` let you accumulate costs and token counts reported by
model providers. Reporters and compaction triggers inspect these values to
decide when to summarize or stop.

```rust
use agentkit_core::{CostUsage, TokenUsage, Usage};

let turn_usage = Usage::new(
    TokenUsage::new(1200, 350)
        .with_reasoning_tokens(50)
        .with_cached_input_tokens(800),
)
.with_cost(CostUsage::new(0.0042, "USD"));

let tokens = turn_usage.tokens.as_ref().unwrap();
assert_eq!(tokens.input_tokens + tokens.output_tokens, 1550);
```

## Cancelling a running turn

Wire a `CancellationController` into the agent and spawn a Ctrl-C listener
that fires `interrupt()`. The loop checks the handle between steps and
returns `FinishReason::Cancelled` when triggered.

```rust,no_run
use agentkit_core::CancellationController;
use agentkit_loop::{Agent, LoopStep, SessionConfig};
use agentkit_core::FinishReason;
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let cancellation = CancellationController::new();
let agent = Agent::builder()
    .model(OpenRouterAdapter::new(
        OpenRouterConfig::new("sk-or-v1-...", "openrouter/auto"),
    )?)
    .cancellation(cancellation.handle())
    .build()?;

let mut driver = agent
    .start(SessionConfig::new("demo"))
    .await?;

// Ctrl-C fires the interrupt, cancelling the in-flight turn.
let interrupt = cancellation.clone();
let ctrl_c = tokio::spawn(async move {
    let _ = tokio::signal::ctrl_c().await;
    interrupt.interrupt();
});

// ... submit input and drive the loop ...

let step = driver.next().await;
ctrl_c.abort();

if let Ok(LoopStep::Finished(result)) = step {
    if result.finish_reason == FinishReason::Cancelled {
        eprintln!("turn cancelled");
    }
}
# Ok(())
# }
```
