# agentkit

Feature-gated umbrella crate for assembling agent applications from the workspace crates.

By default this crate re-exports the core runtime pieces:

- `core` -- shared types: `Item`, `Part`, `SessionId`, `Usage`, cancellation primitives
- `capabilities` -- capability traits: `Invocable`, `CapabilityProvider`, `PermissionChecker`
- `tools` -- tool abstractions: `Tool`, `ToolRegistry`, `ToolSpec`, permission types
- `loop` -- agent loop orchestration: `Agent`, `AgentBuilder`, `LoopDriver`, `LoopStep`
- `reporting` -- loop observers: `StdoutReporter`, `JsonlReporter`, `UsageReporter`, `CompositeReporter`

Additional integrations are available behind optional Cargo features:

| Feature               | Crate                          | Purpose                                                         |
| --------------------- | ------------------------------ | --------------------------------------------------------------- |
| `compaction`          | `agentkit-compaction`          | Transcript compaction triggers, strategies, and pipelines       |
| `context`             | `agentkit-context`             | `AGENTS.md` discovery and loading                               |
| `mcp`                 | `agentkit-mcp`                 | Model Context Protocol server connections                       |
| `provider-openrouter` | `agentkit-provider-openrouter` | OpenRouter `ModelAdapter` implementation                        |
| `tool-fs`             | `agentkit-tool-fs`             | Filesystem tools (read, write, edit, move, delete, list, mkdir) |
| `tool-shell`          | `agentkit-tool-shell`          | Shell execution tool (`shell.exec`)                             |
| `tool-skills`         | `agentkit-tool-skills`         | Progressive Agent Skills discovery and activation               |

## Quick start

Add agentkit with the features you need:

```toml
[dependencies]
agentkit = { version = "0.1", features = ["provider-openrouter", "tool-fs", "tool-shell"] }
tokio = { version = "1", features = ["full"] }
```

## Examples

### Minimal agent with OpenRouter

```rust,no_run
use agentkit::core::{Item, ItemKind, MetadataMap, Part, SessionId, TextPart};
use agentkit::loop_::{Agent, LoopStep, SessionConfig};
use agentkit::provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit::reporting::StdoutReporter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenRouterConfig::from_env()?;
    let adapter = OpenRouterAdapter::new(config)?;

    let agent = Agent::builder()
        .model(adapter)
        .observer(StdoutReporter::new(std::io::stdout()))
        .build()?;

    let mut driver = agent
        .start(SessionConfig {
            session_id: SessionId::new("demo"),
            metadata: MetadataMap::new(),
        })
        .await?;

    // Submit a user message and drive the loop to completion.
    driver.submit_input(vec![Item {
        id: None,
        kind: ItemKind::User,
        parts: vec![Part::Text(TextPart {
            text: "What is the capital of France?".into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    }])?;

    loop {
        match driver.next().await? {
            LoopStep::Finished(result) => {
                println!("Turn finished: {:?}", result.finish_reason);
                break;
            }
            LoopStep::Interrupt(interrupt) => match interrupt {
                agentkit::loop_::LoopInterrupt::ApprovalRequest(req) => {
                    // Auto-approve for this demo; a real app would prompt the user.
                    driver.resolve_approval(req.id, true)?;
                }
                agentkit::loop_::LoopInterrupt::AuthRequest(req) => {
                    eprintln!("Auth required: {}", req.provider);
                    break;
                }
                agentkit::loop_::LoopInterrupt::AwaitingInput(_) => {
                    // Feed the next user message and continue the loop.
                    break;
                }
            },
        }
    }

    Ok(())
}
```

### Agent with filesystem and shell tools

```rust,no_run
use agentkit::core::{MetadataMap, SessionId};
use agentkit::loop_::{Agent, SessionConfig};
use agentkit::reporting::{CompositeReporter, StdoutReporter, UsageReporter};
use agentkit::provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit::tools::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenRouterAdapter::new(OpenRouterConfig::from_env()?)?;

    // Build a tool registry with filesystem and shell tools.
    let mut tools = agentkit::tool_fs::registry();
    for tool in agentkit::tool_shell::registry().tools() {
        tools.register_arc(tool);
    }

    let agent = Agent::builder()
        .model(adapter)
        .tools(tools)
        .observer(StdoutReporter::new(std::io::stdout()))
        .build()?;

    let _driver = agent
        .start(SessionConfig {
            session_id: SessionId::new("coding-agent"),
            metadata: MetadataMap::new(),
        })
        .await?;

    Ok(())
}
```

### Default features only (no provider)

When writing a custom `ModelAdapter`, only the default features are needed:

```rust,ignore
use agentkit::core::{Item, SessionId};
use agentkit::loop_::{Agent, ModelAdapter, SessionConfig};
use agentkit::tools::ToolRegistry;
use agentkit::reporting::StdoutReporter;

// Implement ModelAdapter for your own backend, then:
// let agent = Agent::builder().model(my_adapter).build()?;
```
