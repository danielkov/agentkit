# agentkit-provider-openai

<p align="center">
  <a href="https://crates.io/crates/agentkit-provider-openai"><img src="https://img.shields.io/crates/v/agentkit-provider-openai.svg?logo=rust" alt="Crates.io" /></a>
  <a href="https://docs.rs/agentkit-provider-openai"><img src="https://img.shields.io/docsrs/agentkit-provider-openai?logo=docsdotrs" alt="Documentation" /></a>
  <a href="https://github.com/danielkov/agentkit/blob/main/LICENSE"><img src="https://img.shields.io/crates/l/agentkit-provider-openai.svg" alt="License" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/MSRV-1.92-blue?logo=rust" alt="MSRV" /></a>
</p>

OpenAI model adapter for the agentkit agent loop.

This crate provides two OpenAI adapters:

- `OpenAIChatCompletionsAdapter` for `/v1/chat/completions`. The historical
  `OpenAIAdapter` name remains a compatibility alias.
- `OpenAIResponsesAdapter` for the public `/v1/responses` API and configurable
  private Responses-compatible deployments.

Both adapters translate AgentKit transcripts, tools, usage, and finish reasons.
Chat-completions streaming is enabled by default; use `.with_streaming(false)`
to force its buffered response path.

Applications that want an OpenAI-powered agent will usually use this crate
through the umbrella `agentkit` crate's `provider-openai` feature, or depend on
it directly when assembling a smaller runtime.

## Configuration

Create a config with `OpenAIConfig::new(authentication, model)` and chain `.with_*()` builders for optional parameters. Alternatively, `OpenAIConfig::from_env()` reads from environment variables:

| Variable          | Required | Default                                      |
| ----------------- | -------- | -------------------------------------------- |
| `OPENAI_API_KEY`  | yes      | --                                           |
| `OPENAI_MODEL`    | no       | `gpt-4o`                                     |
| `OPENAI_BASE_URL` | no       | `https://api.openai.com/v1/chat/completions` |

## Authentication and resilience

`OpenAIConfig` and `OpenAIResponsesConfig` store credentials as first-class
`agentkit_http::Authentication` values. A bare string passed to either `new`
constructor or `.with_authentication(...)` is shorthand for bearer
authentication. Both configs and adapters provide
`.with_authentication_provider(...)` for a custom refresh-capable
`AuthenticationProvider`; the provider receives the opaque prior authentication
attempt during the single reactive 401 refresh. The standard static bearer and
header constructors attach an ephemeral, non-secret binding automatically.
Custom providers that need Responses continuation replay must attach a stable,
non-secret credential identity or generation with
`AuthenticationAttempt::with_binding`. If a reactive refresh changes that
binding, the adapter fails rather than sending the already encoded, binding-bound
body. Resilience is stored as `Option<ResilienceConfig>` and defaults to
`None`. Calling `.with_resilience(...)` opts into retries and timeouts; leaving
it as `None` preserves the existing single-attempt behavior for transient
transport/status/stream failures (the one permitted 401 refresh remains part
of authentication).

Responses retries reuse one clone-cheap serialized request body and a
body-bound idempotency key. Events stream as soon as they are decoded. A failed
attempt is replayed automatically only before its first event becomes visible.
After visible output, replay is disabled unless the upstream consumer explicitly
calls `SessionConfig::with_response_attempt_supersession()`. On a visible-attempt
retry, the adapter emits `ModelTurnEvent::ResponseAttemptSuperseded` after the
failed attempt's events and before replacement output; the loop forwards it as
`AgentEvent::ResponseAttemptSuperseded`. Consumers must discard every delta, tool
call, usage update, and reconstruction state from that preceding attempt.
Cancellation, the logical retry deadline, stream-idle timeout, absolute
per-attempt deadline (including stream reads), auth/refresh, and backoff remain
bounded. Responses default to a 32 MiB serialized request limit, 16 MiB per
attempt, and 64 MiB aggregate wire limit across retries. Requests and responses
also default to at most 100,000 collection items/indexes and 8 MiB per text,
reasoning, ciphertext, or media field. SSE and normalized event traffic is bounded
by the response byte limits rather than this collection-cardinality limit. Override
these with `OpenAIResponsesLimits` and
`.with_limits(...)`; for example, a host can set `max_items` to `10_000` and
`max_text_bytes` to `1024 * 1024` while retaining the default aggregate bounds.
Limits must be non-zero; the per-field bound must fit both request and attempt
bounds, and the per-attempt bound must fit the aggregate wire bound.

## Responses WebSocket transport

HTTP/SSE remains the default. Select a transport explicitly on the Responses
configuration (public API and ChatGPT private profiles both support selection):

```rust
use agentkit_provider_openai::{OpenAIResponsesConfig, OpenAIResponsesTransport};

let config = OpenAIResponsesConfig::new("token", "gpt-5")
    .with_transport(OpenAIResponsesTransport::Auto);
```

- `Http`: the existing HTTP/SSE path, including custom `Http` clients.
- `WebSocket`: require an HTTP/1 WebSocket upgrade; never silently use SSE.
- `Auto`: try WebSocket. An upgrade response of HTTP **426** switches this
  session to HTTP/SSE for all remaining turns. No fallback is attempted after a
  `response.create` has been sent, or for arbitrary authentication/protocol errors.

The configured `https://.../responses` endpoint is upgraded on that same path
(the wire equivalent of `wss://.../responses`); `http://` endpoints support local
mock servers. Authentication comes from the same refresh-capable provider used
by HTTP. The handshake sends `OpenAI-Beta: responses_websockets=2026-02-06` and
requests are JSON text messages with `type: "response.create"` plus the normal
Responses fields, excluding HTTP-only `stream` and `background` controls as
required by the [public WebSocket guide](https://developers.openai.com/api/docs/guides/websocket-mode). JSON events use the same semantic decoder as SSE, preserving
reasoning, tool, usage, image, and credential-bound continuation metadata.
No browser or separate authentication flow is required.

Each `ModelSession` owns an independent connection slot. A turn checks out that
slot and releases it **when `Finished` is delivered**, not when the owned turn
value is eventually dropped. Only one active turn is allowed in a WebSocket/Auto
session; a concurrent `begin_turn` is rejected. A validated terminal response
makes the socket reusable. Cancellation, dropping an unfinished turn, malformed
messages, missing completion, and terminal errors discard it. Buffered unsolicited
frames force reconnection before another request; known completed response IDs
are rejected if they reappear on a reused socket. Cancellation never
sends `response.cancel`. Changed authentication headers or credential binding
force a new connection; a 401 can refresh once only if the binding is unchanged.

WebSocket retries are intentionally more conservative than HTTP retries:

- Handshake status failures use the existing bounded retry policy and observer.
- A wrapped HTTP error before `response.created` (and before visible output)
  may reconnect and retry only on a fresh socket. On reused sockets, errors lack
  reliable request correlation and may belong to a previous turn, so generic
  errors never trigger automatic replay or authentication refresh. The narrowly
  scoped missing-continuation recovery is described below. An accepted response
  is never automatically replayed.
- An interrupted send, socket EOF, receive failure, or timeout after sending is
  **not replayed**: the server may already have accepted the request.
- Visible WebSocket output is never superseded/replayed, even when the consumer
  opts into HTTP response-attempt supersession. No WebSocket idempotency guarantee
  is assumed. Wrapped error statuses and allowlisted retry headers are retained
  in normal retry observations; raw provider error messages are not exposed.
- Dropping a pending `next_event` future during authentication refresh or a
  WebSocket opening operation makes a retained turn fail safely on its next
  poll. It does not resume with stale credentials or replay an uncertain send.

Request, per-attempt, aggregate wire, field, and item bounds remain in force.
Frame/message sizes are bounded before JSON decoding; binary messages are
rejected and consecutive control frames are bounded. Upgrade and send operations
have a 30-second ceiling; configured attempt, idle, logical retry-budget, and
cancellation bounds also apply. As with HTTP, configure `with_resilience` to set
stream idle and whole-turn deadlines.

The optional [live continuation suite](../../docs/live-responses-websocket.md)
verifies incremental wire requests and server acceptance using environment credentials.
It is ignored by default and makes billable requests when explicitly enabled.

### Incremental continuation and limitations

The **full credential-bound transcript remains authoritative**. After a successful
`response.completed`, the live connection retains a bounded, memory-only checkpoint:
the response ID, full encoded request, raw completed output items, and their normal
transcript replay encoding. When all non-input request fields match and the next
input begins with the previous input plus that replay output, the next
`response.create` sends `previous_response_id` and only the new input suffix.
This includes tool results and later user messages, including rounds containing
encrypted reasoning or generated images. No continuation state is persisted.

Compatibility uses the same credential-bound encoder as an ordinary full request.
In particular, function arguments are serialized from parsed JSON, message output
metadata such as status and annotations is not replayed, and reasoning replays its
ID and encrypted content with an empty summary. These are the adapter's transcript
semantics, not byte equality against the server's raw output. Changes to retained
text, call IDs/arguments, protected reasoning or image data fail the prefix proof.
Unreadable/unrepresentable or oversized checkpoints disable the optimization.
Changes to model, tools, instructions, reasoning settings, or compacted/edited
history send the full input without an ID. Reconnection, authentication changes,
cancellation and failures discard the connection checkpoint. Checkpoint request
and output buffers are bounded by the configured request/item limits and zeroized
on drop; the socket also bounds completed response-ID correlation history.

An explicit `previous_response_not_found` rejection before acceptance or visible
output may reconnect and retry once with the full request, **within the configured
resilience retry budget**. Recovery requires the documented 400
`invalid_request_error` message naming the exact predecessor sent by this request
(`Previous response with id '<id>' not found.`). Code-only errors, older IDs, changed
message formats, and malformed error events fail closed: they cannot safely
correlate a rejection on a reused socket. Set `with_resilience` with `max_retries >= 1` to allow
this recovery. The full retry has no previous ID, so this special recovery cannot
repeat. Accepted responses, visible output, and ambiguous sends/disconnects are
never replayed. Recovery uses the existing retry observations and deadline budget.

WebSocket upgrades use a dedicated reqwest HTTP/1 client with redirects and
implicit HTTP retries disabled, using the existing reqwest TLS stack. A custom
`Http` passed to `with_client` applies only to HTTP/SSE, **not** to WebSocket
upgrades; applications requiring custom transport middleware should keep `Http`.

Wire contract reference: OpenAI Codex commit
[`6824dabe0393337a38cb257d5fe75ae5ca168470`](https://github.com/openai/codex/tree/6824dabe0393337a38cb257d5fe75ae5ca168470),
`codex-rs/codex-api/src/endpoint/responses_websocket.rs` and `common.rs`.

## Retry observations and typed failures

Responses emits `agentkit_loop::ProviderRetryEvent` through
`AgentEvent::ProviderRetry` to registered loop observers. Initial HTTP retries
are visible while `begin_turn` is still pending; stream retries are visible
before their deferred backoff, including before an attempt-supersession marker.
Direct adapter consumers can call `ModelSession::set_retry_observer` before
`begin_turn`. Other adapters retain the default no-op implementation.

- `Scheduled(RetryProgress)` carries a canonical route, attempt reason,
  allowlisted upstream type/code, HTTP status when available, next delay, and
  cumulative accounting. The first snapshot is immediate; subsequent snapshots
  are limited to one per 250 ms per model request (`begin_turn` invocation). Suppressed updates do not
  discard accounting. There is no queue, heartbeat, or trailing-update timer.
- `Stopped(ProviderFailure)` reports explicit failure or cancellation once.
  `Succeeded { route, accounting }` clears activity once on successful model
  completion. These observations are not additional model results or tool-effects
  records. Polling a completed/failed turn again produces no duplicate terminal
  observation. Dropping a future/turn does not promise terminal delivery.
- Callbacks are synchronous, infallible at the interface, and run without
  holding decoder/session locks. They must not block or re-enter the session.
  As with loop observers, panics propagate; blocking/panicking observers prevent
  delivery guarantees.

`RetryAccounting::attempts` counts polled HTTP client executions, including a
resend after reactive authentication refresh, not policy retries plus one.
Initial authentication and preflight failures can have zero attempts. Malformed
endpoints, non-HTTP(S) endpoints, and invalid local attribution headers are
rejected as `InvalidRequest` before transport execution, without retrying.
`completed_backoff` sums the requested durations of fully completed waits;
interrupted waits contribute zero, while earlier completed waits remain counted.
`elapsed` measures monotonic time from before authentication and includes
preflight, callbacks, requests, and interrupted waits. Cancellation and logical
deadlines are checked before accepting a ready completion, so a cancelled or
budget-expired wait is not counted as completed.

Responses model-session failures use `LoopError::ProviderFailure`;
`LoopError::provider_failure()` exposes the typed payload without string parsing.
`reason` distinguishes retry-count exhaustion, logical budget expiry, disabled
retry, unsafe replay after output, authentication, and protocol/transport failure.
`last_attempt_reason` retains the failed attempt category, such as attempt/idle
timeout, across a local exhaustion/budget stop. The last source response's
`upstream` classification survives local stops and authentication refresh failures.
A later HTTP/SSE failure replaces it, even when its type/code are unknown.
Unknown, missing, or malformed provider type/code values become `Unknown`;
provider messages, bodies, headers, credentials, prompts, endpoint URLs, and
customer identifiers never enter these payloads, their Debug, or their Display.
Non-success HTTP bodies are not read to obtain classifications.

`LoopError::Cancelled` remains unchanged for cancellation-aware callers; its
accounting is available in the `Stopped` observation. The driver's explicit
post-event cancellation boundary calls `ModelTurn::on_cancelled` before dropping
the turn, so terminal accounting does not require another event poll. Direct
consumers that explicitly abandon a turn can use the same synchronous hook.
Correlate observations
through `ObservedEvent.session_id` and the current `TurnStarted` event. Direct
session consumers own that association. A host must add its own stable fatal
event reference when projecting this contract into parent-visible failures.

**Compatibility:** the new `LoopError` variant requires updating exhaustive
error matches. Existing serialized `AgentEvent` variants are unchanged, but old
readers can reject the new variant; coordinate reader updates and compatible
crate releases before sending these events across a versioned boundary. This
upstream API does not implement ACP parent delivery, fatal-record persistence,
or stable fatal-record correlation.

## Responses API

`OpenAIResponsesConfig::new(authentication, model)` deliberately matches
`OpenAIConfig::new(authentication, model)`. The profile-specific constructors use
`public(model, authentication)` and `chatgpt_private(model, authentication)`; use
those named forms when selecting a profile so the different argument order is
explicit.

```rust,no_run
use agentkit_provider_openai::{OpenAIResponsesAdapter, OpenAIResponsesConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenAIResponsesConfig::new("sk-...", "gpt-5");
let adapter = OpenAIResponsesAdapter::new(config)?;
# let _ = adapter;
# Ok(())
# }
```

For the private ChatGPT Codex-shaped endpoint, inject authentication explicitly:

```rust,no_run
use agentkit_http::Authentication;
use agentkit_provider_openai::{OpenAIResponsesAdapter, OpenAIResponsesConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let authentication = Authentication::bearer("short-lived-access-token");
let config = OpenAIResponsesConfig::chatgpt_private("gpt-5-codex", authentication)
    .with_endpoint("https://chatgpt.com/backend-api/codex/responses");
let adapter = OpenAIResponsesAdapter::new(config)?;
# let _ = adapter;
# Ok(())
# }
```

`with_headers` supplies ordinary non-authentication headers.
`with_user_agent` and `with_originator` provide explicit request attribution,
and `with_request_policy` can override public/private request-field differences.
The synchronous `encode_request` helper cannot resolve an authentication binding
and rejects bound continuation metadata; submit through `OpenAIResponsesAdapter`
to replay adapter-emitted function-call, reasoning, or generated-image state.
The public and private profiles request encrypted reasoning continuation data by
default. Context content is sent unchanged: the public profile keeps system and
Context items as system messages, while the private profile downgrades both to
developer messages,
defaults `parallel_tool_calls` to `true`,
omits unsupported `max_output_tokens`, sends `originator`/`session-id`, and
replays validated `x-codex-turn-state` only within one logical turn and its
retries. HTTP turn state is accepted only from a successful SSE response; the
equivalent `response.metadata` headers can update retry context, and all header,
metadata, and retry values must agree. It does not perform credential discovery
or model catalog lookups.

Continuation metadata is versioned and bound to the authentication binding,
session, provider item ID, and item kind. It records the originating model as
provenance without restricting replay to that model. Durable encrypted reasoning,
function-call, and generated-image continuation metadata is emitted and replayed
only when authentication supplies a binding. Valid metadata for another binding
is omitted safely; malformed metadata is a protocol error. The private profile accepts image/audio transcript inputs and
media-bearing tool outputs. The public profile continues to reject the private
audio shape.

## Examples

### Minimal chat agent

```rust,no_run
use agentkit_loop::{Agent, SessionConfig};
use agentkit_provider_openai::{OpenAIAdapter, OpenAIConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenAIConfig::new("sk-...", "gpt-4o");
let adapter = OpenAIAdapter::new(config)?;

let agent = Agent::builder()
    .model(adapter)
    .build()?;

let mut driver = agent
    .start(SessionConfig::new("demo"))
    .await?;

let step = driver.next().await?;
println!("{step:?}");
# Ok(())
# }
```

### With model parameters

```rust,no_run
use agentkit_provider_openai::{OpenAIAdapter, OpenAIConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenAIConfig::new("sk-...", "gpt-4o")
    .with_temperature(0.0)
    .with_max_completion_tokens(4096)
    .with_frequency_penalty(0.5);

let adapter = OpenAIAdapter::new(config)?;
# Ok(())
# }
```

### Environment-based configuration with overrides

```rust,no_run
use agentkit_provider_openai::{OpenAIAdapter, OpenAIConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenAIConfig::from_env()?
    .with_temperature(0.0)
    .with_max_completion_tokens(512);

let adapter = OpenAIAdapter::new(config)?;
# Ok(())
# }
```

### Custom base URL (Azure OpenAI, proxies, etc.)

```rust,no_run
use agentkit_provider_openai::{OpenAIAdapter, OpenAIConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenAIConfig::new("sk-...", "gpt-4o")
    .with_base_url("https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview");

let adapter = OpenAIAdapter::new(config)?;
# Ok(())
# }
```
