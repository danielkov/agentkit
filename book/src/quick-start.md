# Quick start

## Prerequisites

- Rust 1.88+
- An [OpenRouter](https://openrouter.ai) API key (for the included examples)

## Running an example

1. Clone the repo:

```sh
git clone https://github.com/danielkov/agentkit.git
cd agentkit
```

2. Create a `.env` file:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/hunter-alpha
```

3. Run an example:

```sh
cargo run -p openrouter-chat -- "hello"
```

## Example progression

The examples build up in complexity:

| Example | What it demonstrates |
|---------|---------------------|
| `openrouter-chat` | Minimal chat loop with `Ctrl-C` turn cancellation |
| `openrouter-coding-agent` | One-shot prompt runner with filesystem tools |
| `openrouter-context-agent` | Context loading from `AGENTS.md` and skills |
| `openrouter-mcp-tool` | MCP tool discovery and invocation |
| `openrouter-subagent-tool` | Custom tool running a nested agent |
| `openrouter-compaction-agent` | Structural, semantic, and hybrid compaction |
| `openrouter-parallel-agent` | Async task manager with foreground/background routing |
| `openrouter-agent-cli` | Combined: context + tools + shell + MCP + compaction + reporting |

## Minimal composition

The smallest useful assembly is a model adapter, a tool registry, a permission checker, and a loop observer:

```rust
let agent = Agent::builder()
    .model(adapter)
    .tools(agentkit_tool_fs::registry())
    .permissions(my_permissions)
    .observer(my_reporter)
    .build()?;

let mut driver = agent
    .start(SessionConfig {
        session_id: SessionId::new("demo"),
        metadata: MetadataMap::new(),
    })
    .await?;

driver.submit_input(vec![system_item, user_item])?;

match driver.next().await? {
    LoopStep::Finished(result) => { /* render output */ }
    LoopStep::Interrupt(interrupt) => { /* approval, auth, or input */ }
}
```
