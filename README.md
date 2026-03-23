# agentkit

`agentkit` is a Rust toolkit for building LLM agent applications such as coding agents, assistant CLIs, and multi-agent tools.

The project is intentionally split into small crates behind feature flags so hosts can pull in only the pieces they need.

## Current status

`agentkit` is past the design-only stage. The repo currently includes working implementations for:

- normalized transcript, content-part, and delta types
- a runtime-agnostic loop driver with blocking interrupts for approval, auth, and input
- trait-based tools, permissions, approvals, and auth handoff
- built-in filesystem and shell tools
- context loading for `AGENTS.md` and skills directories
- MCP transports, discovery, tool/resource/prompt adapters, auth replay, and lifecycle management
- reporting observers
- compaction triggers, strategy pipelines, and backend-driven semantic compaction
- async task management with foreground/background scheduling, routing policies, and detach-after-timeout
- optional turn cancellation with resumable sessions
- an OpenRouter provider adapter

The repo also ships multiple examples that exercise these pieces end to end.

## Crates

- `agentkit-core`
  - transcript, parts, deltas, IDs, usage, and cancellation primitives
- `agentkit-capabilities`
  - lower-level invocable/resource/prompt abstraction
- `agentkit-tools-core`
  - tools, registry, executor, permissions, approvals, auth requests
- `agentkit-loop`
  - model session abstraction, driver, interrupts, tool roundtrips
- `agentkit-context`
  - `AGENTS.md` and skills loading
- `agentkit-mcp`
  - MCP transports, discovery, lifecycle, auth, replay, adapters
- `agentkit-reporting`
  - loop observers and reporting adapters
- `agentkit-compaction`
  - compaction triggers, strategies, pipelines, backend hooks
- `agentkit-task-manager`
  - task scheduling for tool execution: foreground, background, and detach-after-timeout routing
- `agentkit-tool-fs`
  - filesystem tools
- `agentkit-tool-shell`
  - shell execution tool
- `agentkit-provider-openrouter`
  - OpenRouter adapter
- `agentkit`
  - umbrella crate with feature-gated re-exports

## Built-in tools today

Filesystem:

- `fs.read_file`
  - supports optional `from` / `to` line ranges
- `fs.write_file`
- `fs.replace_in_file`
- `fs.move`
- `fs.delete`
- `fs.list_directory`
- `fs.create_directory`

Shell:

- `shell.exec`

The filesystem crate also supports session-scoped read-before-write enforcement through `FileSystemToolResources` and `FileSystemToolPolicy`.

## Quick start

1. Put your OpenRouter credentials in `.env`.
2. Run one of the examples.

Example `.env`:

```env
OPENROUTER_API_KEY=replace_me
OPENROUTER_MODEL=openrouter/hunter-alpha
```

Example commands:

```bash
cargo run -p openrouter-chat -- "hello"
```

```bash
cargo run -p openrouter-coding-agent -- \
  "Use fs.read_file on ./Cargo.toml and return only the workspace member count as an integer."
```

```bash
cargo run -p openrouter-agent-cli -- --mcp-mock \
  "Return only the secret from the MCP tool."
```

## Example progression

- `openrouter-chat`
  - minimal chat loop
  - now supports `Ctrl-C` turn cancellation
- `openrouter-coding-agent`
  - one-shot coding-oriented prompt runner with filesystem tools
- `openrouter-context-agent`
  - context loading from `AGENTS.md` and skills
- `openrouter-mcp-tool`
  - MCP tool discovery and invocation
- `openrouter-subagent-tool`
  - custom tool that runs a nested agent
- `openrouter-compaction-agent`
  - structural, semantic, and hybrid compaction
  - semantic compaction uses a nested agent as the backend
- `openrouter-agent-cli`
  - combined example using context, tools, shell, MCP, compaction, and reporting

## Minimal composition

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

## Task management

The `agentkit-task-manager` crate controls how the loop driver schedules tool-call execution. Two implementations are provided out of the box, and you can supply your own by implementing the `TaskManager` trait.

### `SimpleTaskManager` (default)

Runs every tool call inline on the current task, returning the result before the driver continues. This is the default when no task manager is configured and preserves purely sequential behavior.

### `AsyncTaskManager`

Spawns each tool call as a Tokio task. Tasks are classified as **foreground** or **background** through a pluggable `TaskRoutingPolicy`. Foreground tasks block the current turn until they resolve; background tasks run independently and deliver results back to the loop (or to a manual drain queue) when they complete.

A third routing mode, `ForegroundThenDetachAfter(Duration)`, starts a task in the foreground and automatically promotes it to background if it hasn't finished within the given timeout.

```rust
use agentkit_task_manager::{AsyncTaskManager, RoutingDecision, TaskRoutingPolicy};
use agentkit_tools_core::ToolRequest;
use std::time::Duration;

struct MyRoutingPolicy;

impl TaskRoutingPolicy for MyRoutingPolicy {
    fn route(&self, request: &ToolRequest) -> RoutingDecision {
        if request.tool_name.as_ref() == "shell.exec" {
            // Shell commands that take too long get detached automatically.
            RoutingDecision::ForegroundThenDetachAfter(Duration::from_secs(5))
        } else if request.metadata.get::<String>("background").is_some() {
            RoutingDecision::Background
        } else {
            RoutingDecision::Foreground
        }
    }
}

let agent = Agent::builder()
    .model(adapter)
    .tools(agentkit_tool_fs::registry())
    .task_manager(AsyncTaskManager::new().routing(MyRoutingPolicy))
    .build()?;
```

`TaskRoutingPolicy` has a blanket impl for `Fn(&ToolRequest) -> RoutingDecision`, so you can pass a closure directly:

```rust
use agentkit_task_manager::{AsyncTaskManager, RoutingDecision};

let agent = Agent::builder()
    .model(adapter)
    .task_manager(AsyncTaskManager::new().routing(|req| {
        if req.tool_name.as_ref() == "shell.exec" {
            RoutingDecision::Background
        } else {
            RoutingDecision::Foreground
        }
    }))
    .build()?;
```

### Implementing a custom `TaskManager`

The `TaskManager` trait lets you control how tool calls are scheduled, awaited, and delivered back to the loop.

```rust
use agentkit_task_manager::{
    PendingLoopUpdates, TaskLaunchRequest, TaskManager, TaskManagerError,
    TaskManagerHandle, TaskStartContext, TaskStartOutcome,
    TurnTaskUpdate,
};
use agentkit_core::{TurnCancellation, TurnId};
use async_trait::async_trait;

struct MyTaskManager { /* ... */ }

#[async_trait]
impl TaskManager for MyTaskManager {
    /// Launch a tool call. Return `Ready` for an immediate result
    /// or `Pending` to indicate the task is running asynchronously.
    async fn start_task(
        &self,
        request: TaskLaunchRequest,
        ctx: TaskStartContext,
    ) -> Result<TaskStartOutcome, TaskManagerError> {
        // Execute immediately via the provided executor, apply
        // your own scheduling, queue to a thread pool, etc.
        todo!()
    }

    /// Block until the next foreground task for this turn resolves,
    /// or return `None` when no foreground tasks remain.
    async fn wait_for_turn(
        &self,
        turn_id: &TurnId,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, TaskManagerError> {
        todo!()
    }

    /// Drain any background-task results that are ready to be
    /// injected into the next loop iteration.
    async fn take_pending_loop_updates(
        &self,
    ) -> Result<PendingLoopUpdates, TaskManagerError> {
        todo!()
    }

    /// Called when a turn is interrupted (e.g. by cancellation).
    /// Cancel or clean up any foreground tasks for this turn.
    async fn on_turn_interrupted(
        &self,
        turn_id: &TurnId,
    ) -> Result<(), TaskManagerError> {
        todo!()
    }

    /// Return a clonable handle for out-of-band task control:
    /// listing, cancelling, draining, and subscribing to events.
    fn handle(&self) -> TaskManagerHandle {
        todo!()
    }
}
```

The `TaskManagerHandle` returned by `handle()` lets host code outside the loop observe task lifecycle events, cancel running tasks, list running/completed snapshots, and drain manually-delivered items.

## Feature flags

The umbrella crate re-exports subcrates behind feature flags.

Default flags:

- `core`
- `capabilities`
- `tools`
- `task-manager`
- `loop`
- `reporting`

Optional flags:

- `compaction`
- `context`
- `mcp`
- `provider-openrouter`
- `tool-fs`
- `tool-shell`

More detail is in [docs/feature-flags.md](./docs/feature-flags.md).

## Docs

- [docs/getting-started.md](./docs/getting-started.md)
- [docs/architecture.md](./docs/architecture.md)
- [docs/core.md](./docs/core.md)
- [docs/tools.md](./docs/tools.md)
- [docs/mcp.md](./docs/mcp.md)
- [docs/compaction.md](./docs/compaction.md)
- [docs/README.md](./docs/README.md)

## Notable gaps

The main remaining work is polish and hardening rather than missing foundation:

- public-release decisions around the default helper policy set
- richer docs and onboarding polish
- deeper built-in tool ergonomics
- more end-to-end examples and integration coverage
