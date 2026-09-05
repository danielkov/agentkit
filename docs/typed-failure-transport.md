# Typed failure transport

AgentKit transports diagnostic facts as real tool failures, not successful values containing an error object. Retry observations describe what happened; they never authorize replay of a child invocation.

## Values and compatibility

The canonical retry values now live in `agentkit_core::retry`. The existing `agentkit_loop` root exports refer to those exact types; `RetryObserver` remains loop-owned. Existing enum spellings and `Duration` serde objects (`secs`, `nanos`) are unchanged.

`agentkit_core::failure::FailureMetadataV1` is a closed version-1 value with a local `FailureCode` and optional retry, host fatal receipt, and `PossibleEffects` leaves. Missing leaves mean unknown, not zero attempts or no effects. Construction validates HTTP status; the new envelope's reader also closes every nested retry object and duration without changing standalone retry serde.

Use `FailureMetadataV1::from_slice` before parsing untrusted transport bytes. It enforces the 4096-byte budget and returns a static validation error. Unsupported versions, unknown local enum variants or fields, malformed counts, invalid durations, and claims of complete effects observation are rejected. Unknown provider type/code spellings normalize to `UpstreamErrorKind::Unknown`; the original spelling is not retained. The fixed object layout bounds depth. The wire contains no provider message, URL, path, arbitrary map, recursive cause, or raw error source.

`HostReceiptId` validates 1–128 ASCII letters, digits, underscores, or hyphens. This is grammar validation, **not authentication**. Hosts must issue receipt IDs and validate external envelopes against the expected child/session route. Allocate a fatal receipt once and retain that same identity across projections:

- `Stored`: the store operation succeeded at emission time, not a promise of eternal availability.
- `MemoryOnly`: an actual retrievable in-memory record exists.
- `Unavailable`: no record is retained, including a failed storage operation.

Never infer storage success from allocation, reconstruct a receipt from display text, or replace a child's receipt with a parent's local diagnostic.

`ToolError::Diagnostic(Box<DiagnosticToolFailure>)` has static display text and an `ExecutionFailed`, `Internal`, or `Cancelled` kind. The box bounds the enum size, not a recursive cause chain. It deliberately cannot represent retryable `Unavailable`. Use `ToolError::is_cancelled`, `is_permission_denied`, `failure_kind`, `failure_metadata`, and `failure_info` instead of matching only legacy variants.

All legacy `ToolError` serde variants remain readable and retain their previous output shapes and messages. There is no artifact rewrite or migration of old strings into trusted facts. Older exhaustive enum consumers/readers must be updated before they receive the new Diagnostic variant. The new context/snapshot fields also require updating Rust struct literals; `TurnTaskUpdate::Detached` now boxes its snapshot to bound enum size. This change does not publish crates or bump versions.

## Compose catch and rethrow

Uncaught native failures preserve the exact original `ToolError` in both backends. A child `Completed` result with `is_error = true` is rejected as a generic failure and is not cached as successful nested execution. Producers must return `Failed(error)` for typed transport.

Runlet 0.5 flattens host details into the caught error:

```text
return boundary {
    return child({})
} catch err {
    return fail(err.code, err.message, {
        agentkit_failure_token: err.agentkit_failure_token
    })
}
```

`err.agentkit_failure` is the closed advisory projection (`kind`, optional `metadata`). Runlet has signed integers. If a counter or duration cannot be represented exactly, the bridge reports a static, nonretryable internal range failure, issues no capability for that error, and cannot be recovered into success by a catch. Canonical durable values keep their full range; no rounded or saturated observation is published. The projection is never parsed back into an authoritative failure.

The explicit third argument forwards an ephemeral, cryptographically random capability. The per-run bridge restores only the saved native error when its issued token and original code/message all match. Forwarding the full `err` object is also explicit capability forwarding; supplied facts cannot replace saved metadata.

- `fail(err.code, err.message)` creates a **new generic execution failure**, dropping native classification and facts.
- A changed code/message, invented token, consumed token, or token from another run cannot restore metadata.
- A caught capability may intentionally rethrow its original error later in the same run. It does not identify a different concurrent failure.
- A catch that returns a value explicitly handles the failure; no native facts are implicitly attached to that successful result.
- Native diagnostic failures and cancellation are not implicitly retried, regardless of provider retry facts. Legacy `Unavailable` retry behavior is unchanged.

Do not persist tokens. The bridge retains at most 4096 failures per run, does not overwrite on token collision, and releases the table at run end. Entropy failure, collision, capacity exhaustion, or an unrepresentable fact latches a nonretryable host internal failure rather than evicting live identities or allowing a catch to hide broken transport. Tokens are redacted from runtime error rendering. The optional, no-extra-feature `getrandom` dependency is enabled only with Runlet; Lua remains the default backend.

Lua retains its native external error identity: `pcall` followed by `error(err)` preserves diagnostics; stringifying and recreating an error does not.

## Observations that survive cancellation

A task manager installs a fresh `FailureObservationSlot` publisher into its own `OwnedToolContext` copy. A host producer obtains it with `ToolContext::failure_observer()`. Borrowed contexts and callbacks within that invocation can clone the publisher. Nested execution scopes do **not** inherit it; nested tasks receive their own fresh slot. Request metadata is never a publisher or an observation source.

The controller retains one bounded latest value, not an event queue. Publishers independently set final receipt, final retry summary, and effects snapshot:

- Receipt and retry are write-once. Identical republishing is unchanged; a different value conflicts. Retry progress belongs to `RetryObserver`, not this slot.
- Effects may advance only monotonically with the same source. Clearing a true flag or changing source is rejected without mutation. There is no cross-source OR/merge.
- `LocalSession` covers the live owner's cumulative observations, not earlier persisted history or only the failing prompt. `AcpNotifications` covers reports during the selected child prompt. Completion is existential, not proof that every tool finished, committed, or rolled back.
- `seal()` freezes all producer clones. Publications accepted before sealing survive cancellation; later publications fail with `Sealed`. No publication means unknown.

A task's native child error and its local observations can describe different scopes. They remain two separately labeled, bounded projections:

| Surface | Native terminal facts | Task-owned frozen facts |
| --- | --- | --- |
| `TaskSnapshot` | `failure` | `failure_observations` |
| `ToolResultPart.metadata` | `agentkit.tool.failure` | `agentkit.tool.failure_observations` |
| Typed reader | `tool_failure_info` | `task_failure_observations` |

Failed outcomes emit `TaskEvent::Failed`; cancellation emits `TaskEvent::Cancelled` with the typed snapshot. Foreground delivery, background loop updates, manual ready items, and detached notifications retain the same selected facts. Actual success remains Completed and gets no failure-only fields. Cancellation never fabricates a receipt, retry exhaustion, or completion observation.

Cancellation and executor completion select a single terminal result under the task lock. The slot seals before a winning abort. Repeated cancellation cannot reclassify completed work. Launch identities prevent a stale approved generation's completion or detach timer from modifying a replacement task. Before executor admission, cancellation may truthfully mark `not_started`; task scheduling itself never publishes an execution-start observation.

The loop drains `TaskManager::take_interrupted_task_updates(session_id, call_ids)` before synthesizing results for remaining unanswered calls, preserving observed foreground cancellation facts without a second result. Custom task-manager wrappers that delegate interruption must also delegate that scoped drain. Approved continuation verifies immutable session/turn/call/tool and approval identity while permitting host-approved input patches, preserves delivery policy, and seeds a fresh publisher from previously frozen facts; old publishers remain sealed. `list_suspended` distinguishes approval from terminal completion. Inline invocation owners also seal and retain a cancellation projection when their future is dropped.

The host strips `agentkit.tool.failure`, `agentkit.tool.failure_observations`, `agentkit.tool.failure_kind`, and `agentkit.tool.not_started` from request/context-derived snapshots and successful result/item metadata before writing authoritative fields. Unrelated application metadata is preserved. Typed readers validate shape, not the provenance of arbitrary persisted JSON.

## Downstream release boundary

Compatible published AgentKit crates and downstream exhaustive-reader updates are still required. Kit must validate both ACP protocol envelopes and expected routing, wire its host producers to invocation-scoped publishers, allocate/store a fatal receipt once, and use cancellation helpers throughout child/subagent/wrapper handling. This upstream implementation does not install those Kit producers, release either project, or establish replay safety. No production Cargo patches are required or included.

At untrusted boundaries, use the bounded metadata/observations entrypoints and the static-error terminal projectors. Do not deserialize raw leaf enums or arbitrary legacy `ToolError` blobs from provider/child data and log their serde errors: standalone leaf/legacy serde is not a redaction boundary. The new Diagnostic and ToolFailureInfo wrappers normalize their own direct serde errors. Host-approved input patching remains supported during a verified logical-call continuation; a routing policy may intentionally reroute that continuation while delivery and continue policy are retained.
