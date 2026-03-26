# Task management

The `agentkit-task-manager` crate controls how the loop driver schedules tool-call execution.

## `SimpleTaskManager` (default)

Runs every tool call inline on the current task. This preserves purely sequential behavior and is the default when no task manager is configured.

## `AsyncTaskManager`

Spawns each tool call as a Tokio task. Tasks are classified through a pluggable `TaskRoutingPolicy`:

- **Foreground** — blocks the current turn until resolved
- **Background** — runs independently, delivers results when complete
- **ForegroundThenDetachAfter(Duration)** — starts foreground, promotes to background after a timeout

```rust
use agentkit_task_manager::{AsyncTaskManager, RoutingDecision, TaskRoutingPolicy};
use agentkit_tools_core::ToolRequest;
use std::time::Duration;

struct MyRoutingPolicy;

impl TaskRoutingPolicy for MyRoutingPolicy {
    fn route(&self, request: &ToolRequest) -> RoutingDecision {
        if request.tool_name.as_ref() == "shell.exec" {
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

`TaskRoutingPolicy` has a blanket impl for `Fn(&ToolRequest) -> RoutingDecision`, so closures work too:

```rust
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

## Custom `TaskManager`

Implement the `TaskManager` trait for full control over scheduling:

```rust
#[async_trait]
impl TaskManager for MyTaskManager {
    async fn start_task(&self, request: TaskLaunchRequest, ctx: TaskStartContext)
        -> Result<TaskStartOutcome, TaskManagerError> { todo!() }

    async fn wait_for_turn(&self, turn_id: &TurnId, cancellation: Option<TurnCancellation>)
        -> Result<Option<TurnTaskUpdate>, TaskManagerError> { todo!() }

    async fn take_pending_loop_updates(&self)
        -> Result<PendingLoopUpdates, TaskManagerError> { todo!() }

    async fn on_turn_interrupted(&self, turn_id: &TurnId)
        -> Result<(), TaskManagerError> { todo!() }

    fn handle(&self) -> TaskManagerHandle { todo!() }
}
```

The `TaskManagerHandle` returned by `handle()` lets host code observe task lifecycle events, cancel running tasks, list snapshots, and drain manually-delivered items.
