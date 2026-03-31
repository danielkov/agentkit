# Interrupts and control flow

The loop runs autonomously until it hits something that requires a human decision. These blocking points are _interrupts_. This chapter covers how interrupts work, why they exist, and how hosts resolve them.

## The interrupt model

```rust
pub enum LoopStep {
    Interrupt(LoopInterrupt),
    Finished(TurnResult),
}

pub enum LoopInterrupt {
    ApprovalRequest(PendingApproval),
    AuthRequest(PendingAuth),
    AwaitingInput(InputRequest),
}
```

Each interrupt type represents a different reason the loop cannot proceed without host intervention. The variants carry handle types (`PendingApproval`, `PendingAuth`, `InputRequest`) with ergonomic resolution methods, so hosts can resolve the interrupt directly on the handle rather than reaching back into the driver.

```text
Loop autonomy boundary:

  ┌──────────────────────────────────────────────────────┐
  │                Autonomous zone                       │
  │                                                      │
  │   model turn → stream deltas → collect tool calls    │
  │   permission check → tool execution → append result  │
  │   compaction → next model turn → ...                 │
  │                                                      │
  │   The loop runs here without host involvement.       │
  └──────────────────────────┬───────────────────────────┘
                             │
                    yield point (interrupt)
                             │
  ┌──────────────────────────▼───────────────────────────┐
  │                Host decision zone                    │
  │                                                      │
  │   "Approve this shell command?"                      │
  │   "Enter your GitHub OAuth token"                    │
  │   "Type your next message"                           │
  │                                                      │
  │   The host handles this, then calls next() again.    │
  └──────────────────────────────────────────────────────┘
```

## Approval interrupts

When a tool's permission policy returns `RequireApproval`, the loop pauses and surfaces the request:

```rust
pub struct ApprovalRequest {
    pub task_id: Option<TaskId>,
    pub call_id: Option<ToolCallId>,
    pub id: ApprovalId,
    pub request_kind: String,      // e.g. "filesystem.write", "shell.command"
    pub reason: ApprovalReason,
    pub summary: String,
    pub metadata: MetadataMap,
}
```

The `reason` field tells the host _why_ approval is needed:

```rust
pub enum ApprovalReason {
    PolicyRequiresConfirmation,   // Policy always requires approval for this kind
    EscalatedRisk,                // Operation flagged as higher risk than usual
    UnknownTarget,                // Target not recognised by any policy
    SensitivePath,                // Filesystem path outside the allowed set
    SensitiveCommand,             // Shell command not in the allow-list
    SensitiveServer,              // MCP server not in the trusted set
    SensitiveAuthScope,           // MCP auth scope not pre-approved
}
```

The host resolves using the `PendingApproval` handle:

```rust
match driver.next().await? {
    LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
        println!("Tool needs approval: {}", pending.request.summary);

        // Option 1: approve
        pending.approve(&mut driver)?;

        // Option 2: deny
        pending.deny(&mut driver)?;

        // Option 3: deny with reason (fed back to model)
        pending.deny_with_reason(&mut driver, "User declined")?;
    }
    ...
}
```

After resolution, the host calls `next()` again. If approved, the tool executes and the turn continues. If denied, the denial is reported back to the model as a tool error — the model sees the denial reason and can adjust its approach.

### The approval flow in detail

```text
1. Model emits ToolCall(fs.replace_in_file, { path: "/etc/hosts", ... })
                          │
2. Executor runs permission preflight
   └── PathPolicy: /etc/hosts is outside workspace → RequireApproval(SensitivePath)
                          │
3. Driver emits AgentEvent::ApprovalRequired to observers (for UI/logging)
                          │
4. Driver returns LoopStep::Interrupt(ApprovalRequest(PendingApproval { ... }))
                          │
                   ─── host decision ───
                          │
5a. host calls pending.approve(driver)
    └── tool executes → result appended → loop resumes
                    OR
5b. host calls pending.deny(driver)
    └── denial sent to model as ToolResult { is_error: true, output: "Permission denied: ..." }
    └── model sees the error and may try a different approach
```

### Multiple pending approvals

When the model requests several tool calls in a single turn, some may require approval while others don't. The driver surfaces one approval at a time, in the order the model emitted them:

```text
Model response: [ToolCall("fs.write", ...), ToolCall("shell.exec", ...), ToolCall("fs.read", ...)]

Permission check:
  fs.write     → RequireApproval (outside workspace)
  shell.exec   → RequireApproval (unknown command)
  fs.read      → Allow

next() → Interrupt(ApprovalRequest for fs.write)
  host approves
next() → Interrupt(ApprovalRequest for shell.exec)
  host denies
next() → tools execute (fs.write runs, shell.exec denied, fs.read runs)
       → results appended, loop continues
```

The driver tracks pending approvals in a `BTreeMap<ToolCallId, PendingApprovalToolCall>` with a `VecDeque` for ordering. Each approval is surfaced individually, but they belong to the same tool round — the driver only starts tool execution once all pending approvals are resolved.

### Why interrupts, not callbacks

An alternative design would pass a callback or channel into the tool executor. agentkit uses interrupts instead because:

1. **Explicit control flow** — the host's main loop always knows what state the driver is in. There's no hidden state machine running in the background.
2. **No hidden concurrency** — approval doesn't happen on a background thread while the loop keeps running. The loop is genuinely paused.
3. **Testability** — interrupt-based flows are easy to test: submit input, call `next()`, assert you get the expected interrupt, resolve it, call `next()` again. No mocking of async channels.
4. **Serializable state** — an interrupted driver can be snapshotted and resumed later, because the interrupt carries all state needed for resolution.

```text
Callback model (rejected):

  loop calls tool → tool calls approval_callback → callback calls host code
  └── Who owns the stack? Can the host do async work? What if the host panics?
      What if multiple tools need approval concurrently?

Interrupt model (adopted):

  loop calls tool → tool needs approval → loop returns Interrupt to host
  └── Host owns the stack. Host does whatever it needs. Calls next() when ready.
```

## Auth interrupts

MCP servers and external tools may require authentication. Auth interrupts follow the same pattern:

```rust
pub struct AuthRequest {
    pub task_id: Option<TaskId>,
    pub id: String,
    pub provider: String,           // e.g. "github", "google"
    pub operation: AuthOperation,   // what triggered the auth
    pub challenge: MetadataMap,     // OAuth URLs, scopes, etc.
}
```

The `AuthOperation` enum describes what triggered the auth requirement:

```rust
pub enum AuthOperation {
    ToolCall { tool_name, input, ... },
    McpConnect { server_id, ... },
    McpToolCall { server_id, tool_name, input, ... },
    McpResourceRead { server_id, resource_id, ... },
    McpPromptGet { server_id, prompt_id, args, ... },
    Custom { kind, payload, ... },
}
```

The host resolves using the `PendingAuth` handle:

```rust
match driver.next().await? {
    LoopStep::Interrupt(LoopInterrupt::AuthRequest(pending)) => {
        println!("Auth required from: {}", pending.request.provider);

        // Option 1: provide credentials
        let mut creds = MetadataMap::new();
        creds.insert("token".into(), json!("ghp_..."));
        pending.provide(&mut driver, creds)?;

        // Option 2: cancel
        pending.cancel(&mut driver)?;
    }
    ...
}
```

## Input interrupts

When the model finishes a turn and the loop has no pending input, it returns `AwaitingInput`:

```rust
pub struct InputRequest {
    pub session_id: SessionId,
    pub reason: String,
}
```

The host reads the next user message and submits it:

```rust
LoopStep::Interrupt(LoopInterrupt::AwaitingInput(pending)) => {
    let user_message = read_line()?;
    pending.submit(&mut driver, vec![user_item(user_message)])?;
}
```

This is the most common interrupt in an interactive session. The pattern is: model finishes → host gets `AwaitingInput` → host reads user input → host calls `submit` → host calls `next()` → loop runs another turn.

## Interrupt ordering and state safety

The driver enforces strict state transitions:

```text
Valid transitions:

  submit_input() ──▶ next() ──▶ Finished
                               ──▶ Interrupt(Approval) ──▶ resolve_approval() ──▶ next()
                               ──▶ Interrupt(Auth)     ──▶ resolve_auth()     ──▶ next()
                               ──▶ Interrupt(Awaiting) ──▶ submit_input()     ──▶ next()

Invalid (state errors):

  next() while an approval is pending                    → LoopError::InvalidState
  resolve_approval() with no pending approval            → LoopError::InvalidState
  resolve_approval() for a ToolCallId that doesn't exist → LoopError::InvalidState
```

These constraints prevent subtle bugs where the host accidentally skips or duplicates an interrupt resolution. The cost is that the host must handle interrupts immediately, but this matches the reality that an unanswered approval request means the agent genuinely cannot proceed.

## The event/interrupt duality

Some actions are reported both as non-blocking observations _and_ as blocking interrupts:

| Observer receives                   | Host receives                                   |
| ----------------------------------- | ----------------------------------------------- |
| `AgentEvent::ApprovalRequired(req)` | `LoopStep::Interrupt(ApprovalRequest(pending))` |
| `AgentEvent::AuthRequired(req)`     | `LoopStep::Interrupt(AuthRequest(pending))`     |
| `AgentEvent::TurnFinished(result)`  | `LoopStep::Finished(result)`                    |

This duplication is intentional. The event is for observability — a reporter logs it, a UI updates a status indicator. The interrupt is for control flow — the host must answer it before the loop can continue. These are different concerns served by different mechanisms.

A reporter that displays "Waiting for approval..." needs the event. The host code that prompts the user needs the interrupt. Neither should have to reach into the other's channel.

## Practical patterns

### Auto-approve by policy

If your permission policy already knows which operations are safe, it returns `Allow` instead of `RequireApproval`. The loop never interrupts for those operations. Configure your policies conservatively and expand allowlists as you build confidence.

### Session-scoped approvals

A host can maintain a session-local allowlist. When the user approves a command like `cargo build`, add it to the allowlist. On subsequent approval interrupts, check the allowlist before prompting:

```rust
LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
    if session_allowlist.contains(&pending.request.request_kind) {
        pending.approve(&mut driver)?;
    } else {
        let decision = prompt_user(&pending.request)?;
        if decision == "always" {
            session_allowlist.insert(pending.request.request_kind.clone());
        }
        // resolve based on decision
    }
}
```

### Headless operation

For non-interactive agents (CI, background jobs), either configure permissive policies or auto-approve everything:

```rust
LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
    pending.approve(&mut driver)?;
}
```

The approval system still runs — it's just that the policy answers "yes" to everything. The events are still emitted, so audit logging captures every approved operation.

> **Example:** [`openrouter-coding-agent`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-coding-agent) handles approval interrupts for filesystem writes in its main loop.
>
> **Crate:** [`agentkit-loop`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-loop) — `LoopStep`, `LoopInterrupt`, `PendingApproval`, `PendingAuth`, `InputRequest`. Approval types come from [`agentkit-tools-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-tools-core).
