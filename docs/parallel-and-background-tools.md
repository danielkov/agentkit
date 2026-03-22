# Parallel And Background Tool Execution In AgentKit

## Current State

AgentKit already asks OpenRouter for parallel tool calls, but the loop does not exploit that capability yet.

- [`crates/agentkit-provider-openrouter/src/lib.rs`](../crates/agentkit-provider-openrouter/src/lib.rs) sets `parallel_tool_calls = true`.
- [`crates/agentkit-loop/src/lib.rs`](../crates/agentkit-loop/src/lib.rs) executes every `ModelTurnEvent::ToolCall` inline as soon as it is observed.
- [`crates/agentkit-loop/src/lib.rs`](../crates/agentkit-loop/src/lib.rs) stores only one pending approval and one pending auth interruption.
- [`crates/agentkit-tools-core/src/lib.rs`](../crates/agentkit-tools-core/src/lib.rs) treats each tool execution as atomic: completed, interrupted, or failed.

That means:

- multiple tool calls in one assistant message are handled serially
- one blocked tool call prevents every other tool call in the same batch from progressing
- there is no first-class concept of a running background tool job

## What Is Already Possible

There are two different targets here, and they need different designs.

### 1. Parallel execution inside a model tool roundtrip

This means:

- the model emits multiple tool calls in one assistant message
- AgentKit runs the ready ones concurrently
- AgentKit feeds all results back to the model before continuing generation

This is not supported today, but it fits the existing transcript model.

### 2. Detached background jobs

This means:

- a tool starts work that may outlive the current loop step or turn
- the model does not necessarily wait for completion before continuing
- completion arrives later through polling, callbacks, or host scheduling

This is not supported as a first-class runtime feature today.

The only current workaround is tool-level job semantics:

- one tool starts a remote job and returns a job id
- another tool polls for job status

That works without loop changes, but it is not true background execution managed by AgentKit.

## Main Blockers

### The loop executes tool calls inline and serially

[`crates/agentkit-loop/src/lib.rs`](../crates/agentkit-loop/src/lib.rs) currently does this for every `ModelTurnEvent::ToolCall`:

1. build `ToolRequest`
2. call `tool_executor.execute(...).await`
3. append one `ToolResultPart`
4. move to the next event

That design prevents batching and parallel scheduling.

### The driver only knows about one pending interruption

The loop stores:

- `pending_approval: Option<PendingApprovalToolCall>`
- `pending_auth: Option<PendingAuthToolCall>`

and `next()` errors if any interrupt is pending.

That shape makes it impossible to represent:

- two approvals from two tool calls in the same assistant message
- one auth interruption while other tool calls continue running
- one finished tool and one still-running tool in the same batch

### Approval resolution is not keyed

`resolve_auth(...)` validates request ids, but `resolve_approval(...)` only accepts a decision and assumes exactly one pending approval.

Parallel batches need approval resolution to target a specific request or tool call.

### Tool execution has no “running” state

`ToolExecutionOutcome` is currently:

- `Completed`
- `Interrupted`
- `Failed`

There is no outcome for:

- queued
- running
- spawned
- background handle returned

Without that, AgentKit cannot represent long-lived execution in the tool runtime itself.

### MCP calls are serialized per connection

[`crates/agentkit-mcp/src/lib.rs`](../crates/agentkit-mcp/src/lib.rs) keeps a `Mutex<Box<dyn McpTransport>>` on each connection and holds the lock across send + wait-for-response.

That means parallel loop scheduling would still serialize:

- multiple MCP tool calls against the same server

Parallelism would still help for:

- filesystem tools
- shell tools
- custom native tools
- MCP calls to different servers

But same-server MCP concurrency needs an MCP-side transport refactor too.

### The loop is runtime-agnostic

[`docs/loop.md`](./loop.md) explicitly keeps `agentkit-loop` free of direct `tokio` dependency.

That is compatible with in-loop concurrency using portable futures utilities, but true detached background execution needs an injected spawning abstraction rather than hard-coding `tokio::spawn`.

## Recommended Path

## Phase 1: Parallel tool batches inside one turn

This is the highest-value change and should come first.

### Execution model

Change the loop from “execute tools as they stream in” to “collect tool calls for the assistant item, then schedule the batch”.

Suggested flow:

1. Read the model turn through `Finished`.
2. Extract the assistant item and all `ToolCallPart`s.
3. Build a batch state object keyed by `ToolCallId`.
4. Start all ready tool calls concurrently.
5. Keep collecting completions until:
   - every call has a terminal tool result, or
   - the batch reaches a blocking host interrupt state
6. Append results to the transcript in original tool-call order.
7. Continue the same host-visible turn.

This still preserves the current user-visible contract:

- one `next()` call runs until a blocking interrupt or a finished turn

### Loop changes

In `agentkit-loop`:

- replace `pending_approval` and `pending_auth` with a batch-aware structure such as:
  - `pending_tool_batch: Option<PendingToolBatch>`
- add per-call state:
  - `Queued`
  - `Running`
  - `AwaitingApproval`
  - `AwaitingAuth`
  - `Completed`
  - `Failed`
- add batch scheduling using `FuturesUnordered`
- emit lifecycle events for tool execution start and finish

The docs already anticipate `ToolExecutionStarted` and `ToolExecutionFinished`; the code just needs to catch up.

### Public API changes

Minimal viable API changes:

- add `resolve_approval_by_id(request_id, decision)`
- keep `resolve_approval(decision)` only as a convenience when exactly one approval is pending
- optionally add:
  - `pending_interrupts()` for hosts that want to inspect the whole batch

The host-facing `LoopInterrupt` can stay simple at first:

- return one pending approval or auth request at a time
- keep the rest queued in driver state

That avoids a large breaking API change while still enabling internal concurrency.

### Tool executor changes

`ToolExecutor` does not need a fully new model for phase 1.

The loop can execute multiple `execute(...)` futures concurrently if it stops borrowing transient context and instead creates owned per-call execution inputs.

Practical change:

- introduce an owned execution context built from cloned `Arc`s and metadata
- keep `Tool::invoke(...)` itself request/response for now

### Transcript and event ordering

To keep behavior deterministic:

- emit completion events in actual completion order
- append transcript `ToolResultPart`s in original assistant tool-call order

That gives good observability without making provider behavior dependent on race order.

### MCP follow-up

After phase 1 lands, refactor `agentkit-mcp` request handling so one connection can have multiple in-flight JSON-RPC requests.

Likely shape:

- one writer path
- one background reader task
- a map from request id to `oneshot::Sender`

That lets same-server MCP tool calls benefit from loop-level parallelism.

## Phase 2: First-class background tool jobs

This should be treated as a separate feature, not folded into phase 1.

### Why it is different

Parallel batch execution still returns one tool result per tool call before the model continues.

Background execution changes the semantics:

- a tool call may not have a final result yet
- the loop may need to continue while work is still running
- the result may arrive after the model has already moved on

That requires a new runtime concept, not just a scheduler tweak.

### Recommended abstraction

Add an optional background execution layer rather than overloading `ToolExecutionOutcome::Completed`.

One possible direction:

```rust
pub enum ToolExecutionOutcome {
    Completed(ToolResult),
    Interrupted(ToolInterruption),
    Running(BackgroundToolHandle),
    Failed(ToolError),
}
```

with a host/runtime abstraction like:

```rust
pub trait ToolTaskRunner: Send + Sync {
    fn spawn(
        &self,
        task: BackgroundToolTask,
    ) -> Result<BackgroundToolHandle, ToolError>;
}
```

Important constraint:

- `agentkit-loop` should not directly depend on `tokio`

So the runner must be injected by the host or implemented in a runtime-specific crate.

### Driver responsibilities

The driver would need:

- a stable background task id
- background task state tracking
- completion polling or completion injection
- cancellation propagation
- events such as:
  - `ToolBackgroundStarted`
  - `ToolBackgroundFinished`
  - `ToolBackgroundCancelled`

### Transcript strategy

Do not silently mutate the original tool call transcript entry after the fact.

Safer options:

- background tools return an immediate structured acknowledgement with a job id
- later completion is injected as a new tool item or explicit host item
- or the model uses an explicit poll/wait tool

The explicit poll/wait model is the least invasive and may be enough for many real use cases.

## Suggested Implementation Order

1. Add `ToolExecutionStarted` and `ToolExecutionFinished` events.
2. Refactor loop internals to batch tool calls after `Finished`.
3. Replace single pending approval/auth state with batch-aware per-call state.
4. Add concurrent execution with `FuturesUnordered`.
5. Add keyed approval resolution.
6. Refactor MCP request handling to support multiple in-flight requests per connection.
7. Decide whether first-class background jobs are worth the API surface, or whether job-style tools are sufficient.

## Lowest-Risk MVP

If the goal is to get meaningful value quickly without opening a large API surface:

- implement phase 1 only
- keep background jobs as explicit tool-level job ids and polling

That gets AgentKit to:

- actually use the provider’s parallel tool-calling capability
- reduce latency for independent tools
- preserve the current loop mental model
- avoid committing too early to a detached-job API

## Files To Change First

- [`crates/agentkit-loop/src/lib.rs`](../crates/agentkit-loop/src/lib.rs)
  - batching, scheduling, pending batch state, keyed interrupt resolution
- [`crates/agentkit-tools-core/src/lib.rs`](../crates/agentkit-tools-core/src/lib.rs)
  - owned execution inputs or executor helpers for concurrent polling
- [`crates/agentkit-mcp/src/lib.rs`](../crates/agentkit-mcp/src/lib.rs)
  - request multiplexing per connection
- [`crates/agentkit-reporting/src/lib.rs`](../crates/agentkit-reporting/src/lib.rs)
  - render new tool lifecycle events

