# agentkit-loop

Runtime-agnostic agent loop orchestration for sessions, turns, tools, and interrupts.

This crate provides:

- **Model adapter traits** -- `ModelAdapter`, `ModelSession`, and `ModelTurn` abstract away the model provider so you can swap between OpenRouter, Anthropic, or a local LLM without changing loop logic.
- **`Agent` builder and `LoopDriver`** -- configure tools, permissions, observers, and compaction, then drive the loop step-by-step.
- **Interrupt handling** -- the loop pauses and yields `LoopStep::Interrupt` when a tool call requires user approval, authentication, or when no input is queued.
- **Observer hooks** -- attach `LoopObserver` implementations to receive streaming `AgentEvent`s (deltas, tool calls, usage, warnings, lifecycle events).
- **Transcript compaction** -- optionally compact the transcript when it grows too large, via the `agentkit-compaction` integration.

Use it as the central coordinator between model providers, tool execution, and application UI or control flow.

## Quick start

```rust,no_run
use agentkit_core::{Item, ItemKind};
use agentkit_loop::{
    Agent, LoopStep, PromptCacheRequest, PromptCacheRetention, SessionConfig,
};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// 1. Create a model adapter
let adapter = OpenRouterAdapter::new(
    OpenRouterConfig::new("sk-or-v1-...", "openrouter/auto"),
)?;

// 2. Build an agent
let agent = Agent::builder()
    .model(adapter)
    .build()?;

// 3. Start a session to get a LoopDriver
let mut driver = agent
    .start(
        SessionConfig::new("demo").with_cache(
            PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
        ),
    )
    .await?;

// 4. Submit user input and drive the loop
driver.submit_input(vec![Item::text(ItemKind::User, "Hello, agent!")])?;

loop {
    match driver.next().await? {
        LoopStep::Finished(result) => {
            println!("Turn finished ({:?}): {:?}", result.finish_reason, result.items);
            break;
        }
        LoopStep::Interrupt(interrupt) => {
            // See "Handling interrupts" below for how to resolve each variant.
            println!("Loop paused: {interrupt:?}");
            break;
        }
    }
}
# Ok(())
# }
```

## Adding tools and observers

```rust,no_run
use agentkit_loop::{Agent, AgentEvent, LoopObserver};
use agentkit_tools_core::ToolRegistry;

struct PrintObserver;

impl LoopObserver for PrintObserver {
    fn handle_event(&mut self, event: AgentEvent) {
        println!("[event] {event:?}");
    }
}

# fn example<M: agentkit_loop::ModelAdapter>(adapter: M, registry: ToolRegistry) -> Result<(), agentkit_loop::LoopError> {
let agent = Agent::builder()
    .model(adapter)
    .tools(registry)
    .observer(PrintObserver)
    .build()?;
# Ok(())
# }
```

## Handling interrupts

When a tool call requires approval or auth, the loop yields an interrupt.
Resolve it and call `next()` again to resume:

```rust,no_run
use agentkit_loop::{LoopInterrupt, LoopStep};
use agentkit_tools_core::ApprovalDecision;

# async fn handle<S: agentkit_loop::ModelSession>(
#     driver: &mut agentkit_loop::LoopDriver<S>,
# ) -> Result<(), agentkit_loop::LoopError> {
loop {
    match driver.next().await? {
        LoopStep::Finished(result) => {
            println!("Done: {:?}", result.finish_reason);
            break;
        }
        LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(req)) => {
            println!("Approve {}? (auto-approving)", req.summary);
            driver.resolve_approval(ApprovalDecision::Approve)?;
        }
        LoopStep::Interrupt(LoopInterrupt::AuthRequest(req)) => {
            println!("Auth required: {}", req.provider);
            // Obtain credentials, then call driver.resolve_auth(...)
            break;
        }
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
            println!("No more input, stopping.");
            break;
        }
    }
}
# Ok(())
# }
```
