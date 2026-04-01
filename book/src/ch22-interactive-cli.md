# The interactive CLI

This chapter covers the host-side implementation of an interactive coding agent CLI: input handling, output rendering, approval UX, session lifecycle, and error recovery.

Everything in this chapter is host code — agentkit doesn't include a CLI. The library provides the loop, and the host provides the user interface. This separation means the same agentkit crates power a terminal CLI, a web server, an IDE plugin, or a headless CI agent.

## The host loop skeleton

Before diving into details, here's the complete structure of an interactive CLI host:

```rust
// Setup
let agent = Agent::builder()
    .model(adapter)
    .tools(tools)
    .permissions(permissions)
    .observer(reporter)
    .compaction(compaction)
    .cancellation(cancellation_handle)
    .build()?;

let mut driver = agent.start(session_config).await?;

// Submit system prompt and context
driver.submit_input(system_items)?;
driver.submit_input(context_items)?;

// Main interaction loop
loop {
    // Read user input
    let input = read_user_input()?;
    if input == "/exit" { break; }

    driver.submit_input(vec![user_item(&input)])?;

    // Drive the turn to completion
    loop {
        match driver.next().await? {
            LoopStep::Finished(_) => break,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => break,
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(p)) => {
                handle_approval(p, &mut driver)?;
            }
            LoopStep::Interrupt(LoopInterrupt::AuthRequest(p)) => {
                handle_auth(p, &mut driver)?;
            }
        }
    }
}
```

Every section below fills in a piece of this skeleton.

## Input handling

A coding agent CLI needs to handle:

- Single-line user messages
- Multi-line input (pasted code, heredocs)
- Special commands (exit, help, clear)
- Ctrl-C for turn cancellation (not process exit)

### Cancellation wiring

Wire Ctrl-C to the `CancellationController`, not to process exit:

```rust
let controller = CancellationController::new();
let handle = controller.handle();

ctrlc::set_handler(move || {
    controller.interrupt();
})?;
```

The first Ctrl-C cancels the current turn. The model sees `FinishReason::Cancelled` and the turn ends cleanly. The second Ctrl-C (if nothing is running) exits the process.

## Output rendering

### Streaming text

The `StdoutReporter` renders `ContentDelta` events as they arrive. For a CLI, this means writing each text chunk to stdout immediately:

```rust
fn handle_event(&mut self, event: AgentEvent) {
    if let AgentEvent::ContentDelta(Delta::AppendText { chunk, .. }) = event {
        print!("{}", chunk);
        std::io::stdout().flush().ok();
    }
}
```

### Tool activity

Display tool calls as they happen so the user knows what the agent is doing:

```
→ fs.read_file(path: "src/main.rs")
→ fs.replace_in_file(path: "src/main.rs", ...)
→ shell.exec(executable: "cargo", argv: ["build"])
```

### Usage reporting

At the end of each turn, display token counts and cost:

```
tokens: 1,234 in / 567 out | cost: $0.02
```

## Approval UX

When the loop returns an approval interrupt, present it clearly:

```
⚠ shell.exec wants to run: rm -rf target/
  Allow? [y/n/always]:
```

Consider supporting:

- `y` — approve once
- `n` — deny
- `always` — approve and add to allowlist for this session

The approval response maps to `ApprovalDecision::Approve` or `ApprovalDecision::Deny`.

## Session lifecycle

### Multi-turn sessions

A coding agent session typically spans many user turns. The driver persists across turns — the transcript accumulates, compaction fires as needed, and the model retains context from earlier in the conversation.

### Graceful shutdown

On exit, flush any buffered reporters, print a final usage summary, and clean up resources. If MCP servers are connected, shut them down cleanly.

## Error recovery

### Model errors

If the model returns an error (rate limit, content filter, network failure), the driver returns `Err(LoopError::...)`. Display the error and let the user decide:

```rust
match driver.next().await {
    Ok(step) => handle_step(step),
    Err(LoopError::Provider(msg)) => {
        eprintln!("Model error: {msg}");
        eprintln!("Press Enter to retry, or type a new message:");
        // Don't exit — the session is still valid
    }
    Err(LoopError::Cancelled) => {
        eprintln!("Turn cancelled.");
        // Session is still valid, user can send another message
    }
    Err(e) => {
        eprintln!("Fatal error: {e}");
        break;  // Only exit on truly unrecoverable errors
    }
}
```

The key insight: most errors are recoverable. A rate limit resolves after waiting. A content filter can be worked around by rephrasing. A network timeout may succeed on retry. Only exit the session on errors that genuinely corrupt the driver state.

### Tool errors

Tool failures are returned to the model as a `ToolResultPart` with `is_error: true`. The model sees the error message and can decide to retry, try a different approach, or report the failure. The CLI doesn't need to handle tool errors specially — they're part of the normal conversation flow.

```text
Tool error flow (handled entirely within the loop):

  Model: ToolCall(fs.read_file, { path: "main.rs" })
  Tool:  ToolResultPart { is_error: true, output: "File not found" }
  Model: "The file doesn't exist in the current directory. Let me check..."
  Model: ToolCall(shell.exec, { executable: "find", argv: [".", "-name", "main.rs"] })
  Tool:  ToolResultPart { output: "./src/main.rs" }
  Model: ToolCall(fs.read_file, { path: "./src/main.rs" })
  Tool:  ToolResultPart { output: "fn main() { ... }" }

  The host never saw the error — the model handled it autonomously.
```

## Design checklist

A production interactive CLI should handle all of these:

- [ ] Ctrl-C cancels the current turn, not the process
- [ ] Second Ctrl-C (when no turn is running) exits cleanly
- [ ] Streaming text renders as it arrives
- [ ] Tool calls are displayed with name and key arguments
- [ ] Approval prompts clearly show what's being requested
- [ ] Usage is displayed after each turn
- [ ] Model errors are displayed and the session continues
- [ ] Graceful shutdown flushes reporters and disconnects MCP
- [ ] Multi-line input is supported for pasting code

> **Example:** [`openrouter-agent-cli`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-agent-cli) implements most of these patterns. The remaining work for a production CLI is polish: better terminal rendering, richer approval UX, and configuration management.
