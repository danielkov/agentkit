# Architecture

`agentkit` is split into small crates with one umbrella crate for feature-gated re-exports.

## Control flow

The main runtime path is:

1. Host constructs an `Agent`
2. Host starts a `LoopDriver`
3. Host submits input items
4. `LoopDriver::next()` runs a model turn
5. Non-blocking activity is emitted as `AgentEvent`
6. Blocking control points are returned as `LoopStep::Interrupt(...)`
7. When the turn completes, `LoopStep::Finished(...)` is returned

## Blocking vs non-blocking boundaries

**Blocking:**

- Approval requests
- Auth requests
- Awaiting input

**Non-blocking:**

- Deltas
- Usage updates
- Tool-call observations
- Compaction observations
- Warnings

Non-blocking events go to loop observers and reporting adapters. They do not control loop progress.

## Tool execution path

Tool calls requested by the model go through:

1. `ToolRegistry`
2. `ToolExecutor`
3. Permission preflight
4. Tool invocation
5. Normalization into transcript `ToolResult` items

Approval and auth both pause the loop and can resume the original operation.

## MCP path

MCP is split into two layers:

- Transport/lifecycle/discovery in `agentkit-mcp`
- Adaptation into tools and capabilities for the rest of the stack

MCP auth requests carry typed operation data and can be resumed through:

- Loop auth resume for tool-path interruptions
- `McpServerManager::resolve_auth_and_resume(...)` for non-tool MCP operations

## Compaction path

Compaction is optional. If configured, the loop asks the compaction trigger whether transcript replacement should happen before a turn. If so:

1. `AgentEvent::CompactionStarted` is emitted
2. The configured compaction strategy or pipeline receives the current transcript
3. The loop replaces its in-memory transcript with the result
4. `AgentEvent::CompactionFinished` is emitted

The loop owns when compaction hooks are checked, but not how compaction is performed.
