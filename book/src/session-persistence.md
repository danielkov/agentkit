# Session persistence

The agent loop has no built-in storage backend. Persistence is intentionally a host concern, but agentkit ships the three primitives you need to compose any backend you like. This chapter documents the contract and walks through the [`openrouter-session-persistence`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-session-persistence) example that puts the pieces together.

## The three primitives

| Primitive                                        | Purpose                                                                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `AgentBuilder::transcript(items)`                  | Restore prior transcript before the loop starts.                                                                         |
| `TranscriptObserver::on_transcript_event(event)`   | Mirror every newly-appended item to durable storage as the loop runs (`TranscriptEvent` carries `session_id` + `item`).   |
| `TranscriptObserver::on_transcript_rewrite(event)` | Replace the stored transcript when a mutator rewrote history (`TranscriptRewriteEvent` carries `session_id` + `items`).   |
| `LoopDriver::snapshot() -> LoopSnapshot`           | Read-only point-in-time view of `transcript` and `pending_input` for ad-hoc dumps, audit, or full-state checkpoints.      |

That is the whole protocol. Any storage backend — in-memory map, sqlite, Postgres, S3, Redis — implements the same shape:

1. **On startup**: load the prior `Vec<Item>` for the session id (or empty for a fresh session) and pass it to `AgentBuilder::transcript`.
2. **During the run**: register a `TranscriptObserver` that appends each `Item` to durable storage and replaces it on rewrite.
3. **On shutdown** (graceful or not): nothing required — the observer has already persisted every appended item.

## Two important guarantees

**Append-only ordering.** `on_transcript_event` is called synchronously by the loop, in the exact order items land in the transcript. The observer is the single mutation point — every push to the transcript funnels through it. This means a strictly monotonic `seq` column on a sqlite `items` table reproduces the transcript byte-for-byte on reload.

**Mutator rewrites come through the second method.** Mutator-driven transcript rewrites (compaction, redaction, repair) replace history rather than appending to it, so they do **not** fire `on_transcript_event`. They fire `on_transcript_rewrite` with the complete canonical transcript — once per mutation point whose mutator chain produced a transcript that actually differs from the live one. Both methods are required, so an append-only consumer has to decide explicitly (a one-line no-op body) rather than silently diverging from the driver after a compaction pass.

**Best-effort post-commit observation, not a commit gate.** Both methods are synchronous and infallible, and the loop has already committed by the time they run. A failed write cannot roll the turn back; a host that needs acknowledged durable publication owns that linearization point itself.

## A complete sqlite implementation

The shape below is from the [example crate](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-session-persistence). Two tables, three operations:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);
CREATE TABLE items (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

The observer is a struct holding an `Arc<SqliteSessionStore>` and a session id:

```rust,ignore
struct SqliteTranscriptObserver {
    store: Arc<SqliteSessionStore>,
    session_id: String,
}

impl TranscriptObserver for SqliteTranscriptObserver {
    fn on_transcript_event(&self, event: TranscriptEvent<'_>) {
        if let Err(error) = self.store.append(&self.session_id, event.item) {
            eprintln!("[persistence] failed to append item: {error}");
        }
    }
}
```

Restore on startup is a single SELECT:

```rust,ignore
let prior = store.load(&session_id)?;       // Vec<Item> in transcript order

let agent = Agent::builder()
    .model(adapter)
    .transcript(prior)                        // <- starting state
    .transcript_observer(SqliteTranscriptObserver {
        store: Arc::clone(&store),
        session_id: session_id.clone(),
    })
    .build()?;
```

```rust,ignore
impl TranscriptObserver for SqliteTranscriptObserver {
    fn on_transcript_event(&self, event: TranscriptEvent<'_>) {
        // append one item
    }

    fn on_transcript_rewrite(&self, event: TranscriptRewriteEvent<'_>) {
        // replace the stored rows with event.items
    }
}
```

That is the entire round-trip. Run the example twice with the same `--session` flag and the second run resumes mid-conversation — the first `next()` call returns `AwaitingInput` because the transcript is loaded but no input is queued, and the host supplies the next user message in response.

## Choosing a backend

Sqlite is the easiest to drop into a single-process CLI. For multi-process or distributed agents, swap the storage backend; the observer interface is unchanged:

- **Postgres / MySQL** — same two-table schema, use a connection pool. `on_transcript_event` runs on the loop's task; if your write latency is significant, queue items into a buffered channel and persist on a dedicated task to avoid stalling the loop.
- **Redis** — `RPUSH session:<id> <item-json>` and `LRANGE session:<id> 0 -1` for restore. Atomic, fast, no schema migrations.
- **S3 / GCS** — write a JSONL blob per session, append-on-flush. Higher latency, but cheap and infinitely scalable for archival workloads. Use `LoopDriver::snapshot()` to take periodic full-state checkpoints rather than streaming each item.
- **In-memory `HashMap<SessionId, Vec<Item>>`** — for tests and ephemeral demos. The observer is a one-liner.

## Why no `SessionStore` trait

A `SessionStore` trait would force every backend to implement the same four or five methods. That is what Anthropic's claude-agent-sdk-python does — five methods plus a thirteen-test conformance harness — and it works because their SDK consumes session storage.

agentkit doesn't consume the backend. The loop just calls `on_transcript_event`. Restoration is the host's job. Picking your own shape (one table, three tables, a stream, a directory of JSON files) is the right default for a library that doesn't know how you want to query, archive, or share session state.

The integration test crate exercises the round-trip pattern internally; see `crates/agentkit-integration-tests` for the canonical worked tests.

## Mutation-aware persistence

If your agent registers any `LoopMutator`s (compaction, redaction, repair), the second observer method carries it:

1. `TranscriptObserver::on_transcript_event` continues to mirror new items as they arrive.
2. `TranscriptObserver::on_transcript_rewrite` delivers the complete canonical transcript after a mutator chain committed a change. Replace the stored rows with `event.items`; subsequent `on_transcript_event` calls resume appending from the new tail.
3. A mutation that errored, produced a protocol-invalid transcript, was cancelled, or whose future was dropped leaves the live transcript untouched and notifies nothing — there is no partial rewrite to reconcile. A mutator that writes the same value back is not a change and notifies nothing either.

`AgentEvent::MutationStarted` / `MutationFinished` remain the mutator's own telemetry (which mutator ran, why, how much it replaced). They are not the persistence signal: a mutator chooses its own `dirty` label, while `on_transcript_rewrite` fires on the driver's own value comparison.
