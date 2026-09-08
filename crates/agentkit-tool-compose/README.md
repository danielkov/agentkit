# agentkit-tool-compose

Scripted tool composition for agentkit.

This crate exposes a single `compose` tool. The model supplies a script and
optional JSON input; the script can call the current tool catalog with
`tool(name, input)` and inspect available tools with `tools()`.

Sandboxed Lua is enabled by default. To use only the Runlet backend:

```toml
agentkit-tool-compose = { version = "0.10.10", default-features = false, features = ["runlet"] }
```

```rust
let registry = agentkit_tool_compose::registry();
```

Compose is opt-in. Add this registry explicitly with
`AgentBuilder::add_tool_source`.

For a richer tool description, wrap an existing tool source:

```rust
let tools = agentkit_tool_compose::ComposeTool::wrap(child_source);
```

The wrapped source still advertises and executes its child tools directly, while
`compose` renders child output schemas into its own description. Dynamic sources
remain live: catalog events and child lookups delegate to the wrapped source.

## Runlet ordering

Runlet schedules independent calls concurrently, including effectful calls such
as writes. Ordinary data references establish dependencies: a call that uses an
earlier result waits for that result. When a call must wait for earlier work it
does not read, express the ordering edge explicitly:

```runlet
prepared = prepare_workspace({ path: input.path })
result = after prepared {
    return publish_workspace({ path: input.path })
}
return result
```

Calls lexically created inside an `after` block are created only after every
prerequisite succeeds. Use `after` for required sequencing, not source order;
two adjacent calls with no data dependency or explicit `after` edge may run in
parallel.

The final compose result enters the transcript as compact JSON by default.
With the `toon` feature enabled,
`ComposeConfig::with_result_encoding(ResultEncoding::Toon)` switches it to
[TOON](https://docs.rs/serde_toon2) (Token-Oriented Object Notation), which
renders uniform object lists as a header plus one row per element — smaller
than JSON for the list-shaped values compose scripts tend to return. The tool
description gains a note explaining the format so the model can read it.

## Scoped Runlet progress

A custom `ComposeBackend` can delegate to
`RunletBackend.execute_with_progress(run, sink, capacity).await` instead of
`RunletBackend.execute(run).await`. `sink` is a host-owned bounded
`tokio::sync::mpsc::Sender<RunletProgress>`; `capacity` is a `NonZeroUsize`
bounding each execution's metadata queue. The unit backend and `BackendRun`
remain unchanged.

Each received envelope starts observation of one compiled execution and carries
its exact `parent_call_id`, process-local unique `incarnation`, compiled
`source_digest`, and `healed` flag. Poll `RunletProgress::try_recv` in the host's
own scoped task until `RunletProgressEnd`. No observer threads or callbacks are
created. A full/dropped sink means the execution is unobserved; a dropped
per-execution receiver or overflowing queue never blocks execution.

Events contain only typed Runlet metadata, not inputs, outputs, or error text.
Raw runtime completion is withheld until the compose host checks its outcome.
`Interrupted` and `Incomplete` invalidate the entire observed incarnation;
`Lagged` means state beyond the received prefix is unknown. Approval replay
gets a fresh incarnation. Dropping the execution future invalidates observation
even if its existing blocking runtime is still completing.

Byte spans refer only to the digest-matching compiled source. In particular,
never map `healed` spans onto the submitted script. Source text is not included
because it can contain secrets; if matching source is unavailable, display
metadata without source snippets. Node IDs are execution-local, and neither a
tool name nor an absent event proves which call is running.
