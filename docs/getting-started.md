# getting started

## Quick start

1. set `OPENROUTER_API_KEY` in `.env`
2. optionally set `OPENROUTER_MODEL`
3. run one of the examples

Examples:

- `cargo run -p openrouter-chat -- "hello"`
- `cargo run -p openrouter-coding-agent -- "Use fs.read_file on ./Cargo.toml and return only the workspace member count as an integer."`
- `cargo run -p openrouter-agent-cli -- --mcp-mock "Return only the secret from the MCP tool."`

## Minimal composition

The smallest useful assembly is:

- one model adapter
- one tool registry
- one permission checker
- one loop observer

```rust
let agent = Agent::builder()
    .model(adapter)
    .tools(agentkit_tool_fs::registry())
    .permissions(my_permissions)
    .observer(my_reporter)
    .build()?;
```

Then:

```rust
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

## Example progression

The examples are meant to build up in complexity:

- `openrouter-chat`
  - provider + loop
- `openrouter-coding-agent`
  - provider + loop + fs tools + permissions
- `openrouter-context-agent`
  - provider + loop + context loading
- `openrouter-mcp-tool`
  - provider + loop + MCP tool adaptation
- `openrouter-subagent-tool`
  - custom tool extension
- `openrouter-agent-cli`
  - combined example: context + tools + shell + MCP + compaction + reporting
