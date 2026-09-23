# Live Responses WebSocket continuation test

`crates/agentkit-provider-openai/tests/live_responses_websocket.rs` uses only public
AgentKit APIs. The network test is ignored by default. It makes three billable
model requests, or four with the optional fresh-session replay check. Run it only
with explicit authorization and a small inference budget.

## Environment interface

Choose one credential source:

- ChatGPT subscription: `AGENTKIT_LIVE_CHATGPT_BEARER` **and**
  `AGENTKIT_LIVE_CHATGPT_ACCOUNT_ID` (both required).
- Public API: `OPENAI_API_KEY`.

The complete ChatGPT pair takes precedence. A partial pair is an error, even if
an API key exists. Empty values count as absent. The test never reads `.env`, Kit
credential files, keychains, or refresh tokens, and never refreshes credentials.

Other settings:

| Variable | Meaning |
| --- | --- |
| `AGENTKIT_LIVE_REQUIRED=1` | Fail if credentials are absent. Otherwise print `SKIP` and return before opening any socket. Use this in live CI to prevent a false-green missing-credentials run. |
| `AGENTKIT_LIVE_MODEL` | Model override. Defaults to `gpt-5.4` for ChatGPT and `gpt-4.1-mini` for the public API. Model/account must support Responses WebSocket and tools. |
| `AGENTKIT_LIVE_RECONNECT=1` | Add a fourth request using a fresh session/socket and the full real transcript. |

With credentials already securely injected into the environment:

```sh
AGENTKIT_LIVE_REQUIRED=1 RUST_LOG=off RUST_BACKTRACE=0 \
  cargo test -p agentkit-provider-openai --test live_responses_websocket \
  -- --ignored --nocapture
```

Prefer compiling **before** injecting credentials, then invoking the compiled
integration-test binary directly. This keeps credentials out of cargo, build
scripts, and compiler subprocesses. Never put secrets in command-line arguments,
print environment contents, enable shell tracing, or enable HTTP/wire logging.
The test itself does not install a tracing subscriber. Disable core dumps when
running with real credentials. Use a trusted local machine: credentials and
payloads necessarily exist in process memory.

## What is proved

A loopback-only forwarding WebSocket proxy upgrades to the fixed HTTPS OpenAI
endpoint with real authentication. Authentication headers are added only to the
upstream request; the provider sees a nonsecret placeholder on loopback. Redirects,
environment proxies, and upstream HTTP retries are disabled. The provider uses
`OpenAIResponsesTransport::WebSocket`, **not** `Auto`: no SSE fallback.

The test passes the full application transcript to every ordinary
`ModelSession::begin_turn` call, consumes actual `ModelTurnEvent`s, and appends
`Finished.output_items` unchanged (including opaque provider metadata), as the
agent loop does. Only the local tool result and subsequent user messages are
constructed by the application. It does not invent continuation IDs or metadata.

The proxy forwards unmodified frames. It retains only request counts, connection
indices, input counts/types, and booleans indicating whether `previous_response_id`
matched the latest completed response. Response IDs are held transiently in memory
for equality checks, never logged or included in assertions. No transcript, header,
response body, token, account ID, or raw upstream error is printed or persisted.

Assertions require:

1. A full first request produces exactly the requested local tool call.
2. Tool output is accepted and appears in the answer. The outgoing request has the
   correct `previous_response_id` and exactly one input: the new tool output.
3. A subsequent user turn is accepted and answered. Its request has the correct
   `previous_response_id` and exactly one input: the new user message.
4. With the optional replay setting, a **fresh public session/socket** accepts the
   complete transcript without `previous_response_id`. This is session restart
   coverage, not a forced same-session network-failure/retry test. Its exact full
   input count is calculated from the first wire request plus replayable outputs
   observed in the three real completed response streams, plus the new tool/user inputs.
   `response.output_item.done` supplies output counts when the terminal completion
   carries an empty output array. Summary-only reasoning is excluded; encrypted reasoning is counted. This does
   not call the public request encoder on credential-bound metadata, strip that
   metadata, or synthesize a continuation.

The proxy refuses to forward beyond the three/four-request budget. Strict
request-count assertions also reject hidden retry/replay success. Each model
turn has a 90-second deadline; upstream connection/handshake limits are 20/30
seconds. Prompts request short literal answers. Public API responses are capped at
512 output tokens; the private profile does not support that field. Drop cancels
the forwarding task. Authentication failure is a failure, never a skip; error
details are deliberately suppressed.

## Safely using an existing Kit file credential (external runner only)

The suite must remain environment-only. A separately authorized, **uncommitted**
runner may read the user's selected Kit file store and pass only the access token
and account ID into the already-compiled test process environment.

Kit's `src/provider/openai_auth.rs` uses the `openai-subscription` namespace and
`subscription` identity. `src/credentials.rs` names the record
`BLAKE3(b"openai-subscription\0subscription").hex() + ".json"`, where `\0`
denotes one NUL byte. Resolve the store directory from the nonsecret
`credential_store` / `credential_dir` configuration fields; do not dump the config.
The record contains `access_token`, `account_id`, and Unix-seconds `expires_at`,
as well as other fields the runner must not expose or pass to the test.

Before launching, check expiry with a five-minute margin. If expired, stop and ask
the owner to refresh through Kit; do not independently rotate the refresh token.
Do not rewrite the credential record. Catch parse/read errors without printing
record contents or exception representations. Construct a minimal child
environment; omit inherited debug/logging/proxy/preload settings and all unrelated
credentials. Disable core dumps and impose an overall subprocess timeout. Do not
start the runner until the production implementation is ready and live inference
is explicitly approved.
