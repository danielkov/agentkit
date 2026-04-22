# `openrouter-coding-agent`

Claude-Code-style REPL on top of the agentkit loop. Wires together:

- `agentkit-loop` — driver, observer hooks, compaction plumbing.
- `agentkit-provider-openrouter` — model provider.
- `agentkit-tool-fs` + `agentkit-tool-shell` — editing and execution toolset.
- `agentkit-compaction` — transcript compaction pipeline.
- A custom `LoopObserver` that streams assistant deltas inline, renders tool
  calls, surfaces compaction events, and feeds token usage into a
  context-window trigger.
- A `ModelSession` wrapper that injects type-ahead user messages into the
  current turn at the next tool-round boundary (no cancel/restart).

## Run

```bash
cargo run -p openrouter-coding-agent
```

You'll get a `›` prompt. Type a message and press enter.

- Type while the agent is mid-turn (e.g. during a slow `shell.exec`): the line
  is echoed as `⎿ queued` and reaches the model at the next tool round. The
  turn itself is not cancelled.
- `Ctrl-C` cancels the in-flight turn. The next prompt appears.
- `/cancel` cancels the turn without quitting (same as `Ctrl-C` but via the
  queue).
- `/exit` or `/quit` quits — at the idle prompt, immediately; during a turn,
  after the current turn's cancellation settles.

## Permissions

- Filesystem access inside the current working directory is allowed; paths
  outside request interactive approval.
- Shell commands with cwd inside the working directory are allowed; unknown
  executables request approval.
- Approval prompts: `y`/`yes` to approve, empty/`n`/`no` to deny, or type a
  sentence to deny with that reason.

## Compaction

Fires when reported `input_tokens + output_tokens` reach 80% of the configured
context window (default 200k). Pipeline drops reasoning, drops failed tool
results, then keeps the 16 most recent items while preserving `System` and
`Context` items.

Override the context size with `AGENTKIT_MAX_CONTEXT_TOKENS`:

```bash
AGENTKIT_MAX_CONTEXT_TOKENS=128000 cargo run -p openrouter-coding-agent
```

## Environment

Loads from the workspace `.env`. Requires `OPENROUTER_API_KEY` and
`OPENROUTER_MODEL`.
