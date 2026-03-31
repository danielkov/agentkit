# Driving the loop

This chapter walks through the `LoopDriver` — the runtime heart of the agent. We'll trace a complete turn from input submission through model invocation, tool execution, and final result.

## The driver API

The `LoopDriver` is generic over the model session type:

```rust
pub struct LoopDriver<S: ModelSession> {
    session_id: SessionId,
    session: Option<S>,
    tool_executor: Arc<BasicToolExecutor>,
    task_manager: Arc<dyn TaskManager>,
    permissions: Arc<dyn PermissionChecker>,
    resources: Arc<dyn ToolResources>,
    cancellation: Option<CancellationHandle>,
    compaction: Option<CompactionConfig>,
    observers: Vec<Box<dyn LoopObserver>>,
    transcript: Vec<Item>,
    pending_input: Vec<Item>,
    pending_approvals: BTreeMap<ToolCallId, PendingApprovalToolCall>,
    pending_auth: Option<PendingAuthToolCall>,
    active_tool_round: Option<ActiveToolRound>,
    next_turn_index: u64,
}
```

The public API is narrow:

```rust
impl<S: ModelSession> LoopDriver<S> {
    pub fn submit_input(&mut self, input: Vec<Item>) -> Result<(), LoopError>;
    pub fn resolve_approval_for(&mut self, call_id: ToolCallId, decision: ApprovalDecision)
        -> Result<(), LoopError>;
    pub fn resolve_auth(&mut self, resolution: AuthResolution) -> Result<(), LoopError>;
    pub async fn next(&mut self) -> Result<LoopStep, LoopError>;
    pub fn snapshot(&self) -> LoopSnapshot;
}
```

The host code is a simple loop:

```rust
driver.submit_input(vec![system_item, user_item])?;

loop {
    match driver.next().await? {
        LoopStep::Interrupt(interrupt) => handle_interrupt(interrupt),
        LoopStep::Finished(result) => break,
    }
}
```

### State machine semantics

`next()` is the only async method. It advances the driver through its internal state machine until it hits a yield point — either a finished turn or an interrupt. There is no polling, no callback registration, and no event queue to drain.

```text
Driver state machine:

                submit_input()
                      │
                      ▼
  ┌─────────────────────────────────┐
  │         Has pending input?      │
  │                                 │
  │  yes ──▶ merge into transcript  │
  │  no  ──▶ AwaitingInput         ─┼──▶ Interrupt
  └─────────────┬───────────────────┘
                │
                ▼
  ┌─────────────────────────────────┐
  │      Compaction trigger?        │
  │                                 │
  │  yes ──▶ run compaction pipeline│
  │  no  ──▶ skip                   │
  └─────────────┬───────────────────┘
                │
                ▼
  ┌─────────────────────────────────┐
  │      Model turn                 │
  │                                 │
  │  stream events from model       │
  │  collect tool calls             │
  │  emit AgentEvents to observers  │
  └─────────────┬───────────────────┘
                │
                ▼
  ┌─────────────────────────────────┐
  │      Tool calls present?        │
  │                                 │
  │  no  ──▶ Finished(TurnResult)  ─┼──▶ return
  │  yes ──▶ permission preflight   │
  └─────────────┬───────────────────┘
                │
                ▼
  ┌─────────────────────────────────┐
  │    Any require approval?        │
  │                                 │
  │  yes ──▶ ApprovalRequest       ─┼──▶ Interrupt
  │  no  ──▶ execute tools          │
  └─────────────┬───────────────────┘
                │
                ▼
       append tool results
                │
                ▼
       go to "Model turn" ◀─── automatic tool roundtrip
```

The host cannot call `next()` twice without resolving an outstanding interrupt — that's a state error. This is intentional. The driver forces the host to deal with interrupts before proceeding. You can't accidentally ignore an approval request.

## Anatomy of a turn

Here's what happens inside `next()`, step by step:

### 1. Merge input

Pending items (submitted via `submit_input()`) are appended to the working transcript. The driver emits `AgentEvent::InputAccepted` to observers.

```text
Before:
  transcript: [System, Context, User("hello"), Assistant("Hi!")]
  pending:    [User("Read main.rs")]

After merge:
  transcript: [System, Context, User("hello"), Assistant("Hi!"), User("Read main.rs")]
  pending:    []
```

### 2. Check compaction

If a `CompactionConfig` is set, the trigger evaluates the transcript. If it fires, the strategy pipeline transforms the transcript before the model sees it:

```text
Before compaction (18 items, trigger threshold: 12):
  [System, Context, User, Asst, Tool, User, Asst, Tool, Tool, User, Asst, Tool, User, Asst, Tool, User, Asst, Tool]

After compaction (keep recent 8 + preserve System/Context):
  [System, Context, User, Asst, Tool, User, Asst, Tool]
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    most recent 8 non-preserved items
```

Compaction happens before the model turn, not after. The model always sees the post-compaction transcript.

### 3. Construct TurnRequest

The loop builds a `TurnRequest` from the working transcript and tool registry:

```rust
TurnRequest {
    session_id: self.session_id.clone(),
    turn_id: TurnId::new(format!("turn-{}", self.next_turn_index)),
    transcript: self.transcript.clone(),
    available_tools: self.tool_executor.specs(),
    metadata: MetadataMap::new(),
    cache: self.next_turn_cache.take().or_else(|| self.default_cache.clone()),
}
```

### 4. Start model turn

`session.begin_turn(request, cancellation)` sends the transcript to the provider and returns a streaming turn handle.

### 5. Stream model output

The driver polls `turn.next_event()` in a loop:

```text
Loop:
  next_event() ──▶ Some(Delta(BeginPart))       ──▶ emit ContentDelta to observers
  next_event() ──▶ Some(Delta(AppendText))      ──▶ emit ContentDelta to observers
  next_event() ──▶ Some(Delta(CommitPart))      ──▶ emit ContentDelta to observers
  next_event() ──▶ Some(ToolCall(ToolCallPart)) ──▶ collect for execution
  next_event() ──▶ Some(Usage(Usage))           ──▶ emit UsageUpdated to observers
  next_event() ──▶ Some(Finished(result))       ──▶ break
```

### 6. Execute tools

If the model requested tool calls (indicated by `FinishReason::ToolCall`):

1. The driver constructs a `ToolRequest` for each `ToolCallPart`
2. Each request goes through the task manager for scheduling
3. The task manager routes each tool call (foreground, background, or foreground-then-detach)
4. The executor runs permission preflight on each tool
5. If any tool requires approval → the driver surfaces `LoopStep::Interrupt(ApprovalRequest)`
6. If any tool requires auth → the driver surfaces `LoopStep::Interrupt(AuthRequest)`
7. Otherwise → tools execute and results are appended to the transcript as `ToolResultPart`s

### 7. Tool roundtrip

If tools were executed, the driver starts another model turn automatically (back to step 3). The model sees the tool results and may request more tools or produce a final response.

### 8. Return result

When the model finishes without pending tool calls, the driver returns:

```rust
LoopStep::Finished(TurnResult {
    turn_id,
    finish_reason: FinishReason::Completed,
    items: /* assistant items from this turn */,
    usage: /* accumulated usage */,
    metadata: MetadataMap::new(),
})
```

### Multiple tool roundtrips per user turn

A single user message can trigger many tool roundtrips:

```text
User: "Add error handling to src/parser.rs"

  Turn 1: model → ToolCall(fs.read_file)
          execute → result appended
  Turn 2: model → ToolCall(fs.replace_in_file)
          execute → result appended
  Turn 3: model → ToolCall(shell.exec("cargo check"))
          execute → result appended
  Turn 4: model → Text("I've added error handling...")
          no tool calls → Finished

Host sees: one call to next(), one TurnResult with all items.
```

From the host's perspective, this is one call to `next()` that returns one `TurnResult` containing all items produced across all internal turns. This is a critical feature for coding agents — the model must be able to chain tool calls without returning control to the host after each one.

## Event delivery during a turn

While the driver processes a turn, non-blocking events are delivered to observers synchronously:

```rust
pub trait LoopObserver: Send {
    fn handle_event(&mut self, event: AgentEvent);
}
```

The full event taxonomy:

| Event                      | When it fires                                |
| -------------------------- | -------------------------------------------- |
| `RunStarted`               | `Agent::start()` completes                   |
| `TurnStarted`              | Before each model turn begins                |
| `InputAccepted`            | `submit_input()` is called                   |
| `ContentDelta(Delta)`      | Model streams a delta                        |
| `ToolCallRequested`        | Model requests a tool call                   |
| `ApprovalRequired`         | A tool requires approval                     |
| `AuthRequired`             | A tool requires auth                         |
| `ApprovalResolved`         | An approval interrupt is resolved            |
| `AuthResolved`             | An auth interrupt is resolved                |
| `CompactionStarted`        | Compaction trigger fires                     |
| `CompactionFinished`       | Compaction pipeline completes                |
| `UsageUpdated(Usage)`      | Token usage reported                         |
| `Warning(String)`          | Non-fatal issue (recovered tool error, etc.) |
| `RunFailed(String)`        | Unrecoverable error                          |
| `TurnFinished(TurnResult)` | A turn completes                             |

Observers are called inline, synchronously, in registration order. The loop task blocks briefly for each observer call. This is acceptable because observers should be fast — write to stderr, increment a counter, append to a buffer. Expensive processing should happen asynchronously behind a channel adapter.

## Building the agent

The `Agent` is built with a builder:

```rust
let agent = Agent::builder()
    .model(adapter)                          // required
    .tools(registry)                         // default: empty
    .permissions(checker)                    // default: allow all
    .resources(resources)                    // default: ()
    .task_manager(manager)                   // default: SimpleTaskManager
    .cancellation(cancellation_handle)       // default: none
    .compaction(config)                      // default: none
    .observer(reporter)                      // default: none
    .build()?;

let mut driver = agent.start(session_config).await?;
```

The builder validates that a model adapter is set. Everything else has sensible defaults:

| Field          | Default               | Effect                            |
| -------------- | --------------------- | --------------------------------- |
| `tools`        | empty `ToolRegistry`  | Model can't call any tools        |
| `permissions`  | `AllowAllPermissions` | Every tool call is auto-approved  |
| `resources`    | `()`                  | No shared resources               |
| `task_manager` | `SimpleTaskManager`   | Sequential, inline tool execution |
| `cancellation` | `None`                | No cancellation support           |
| `compaction`   | `None`                | Transcript grows without bounds   |
| `observers`    | `[]`                  | No event reporting                |

`Agent::start()` consumes the agent and returns a `LoopDriver`. The agent's immutable configuration (adapter, tools, permissions) is moved into the driver. Multiple drivers can be created from the same `Agent` type by cloning it first.

## Snapshots

The driver exposes a read-only snapshot for inspection or persistence:

```rust
let snapshot: LoopSnapshot = driver.snapshot();
// snapshot.session_id, snapshot.transcript, snapshot.pending_input
```

This is useful for debugging (inspect the transcript mid-session), persistence (serialize and resume later), and testing (assert on transcript state).

## Cancellation

If the host connects a `CancellationHandle` (e.g. wired to a Ctrl-C handler), the driver creates `TurnCancellation` checkpoints and passes them to model turns and tool executions:

```text
Host wires Ctrl-C:

  ctrlc::set_handler(move || controller.interrupt());

Driver flow:

  1. checkpoint = cancellation.checkpoint()
  2. session.begin_turn(request, Some(checkpoint.clone()))
  3. turn.next_event(Some(checkpoint.clone()))
     └── if cancelled → LoopError::Cancelled
  4. tool.invoke(request, ctx)  // ctx.cancellation = Some(checkpoint)
     └── if cancelled → ToolError::Cancelled
```

When cancellation fires, the current turn ends with `FinishReason::Cancelled`. The driver adds metadata (`agentkit.interrupted: true`, `agentkit.interrupt_reason: "user_cancelled"`) to the turn result so the host can distinguish cancellation from normal completion.

> **Example:** [`openrouter-coding-agent`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-coding-agent) demonstrates a driver executing filesystem tool calls across multiple roundtrips in a single turn.
>
> **Crate:** [`agentkit-loop`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-loop) — the `Agent`, `AgentBuilder`, `LoopDriver`, `LoopStep`, `TurnResult`, and `LoopSnapshot` types.
