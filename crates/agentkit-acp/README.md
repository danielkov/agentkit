# agentkit-acp

Agent Client Protocol integration for agentkit hosts.

This crate re-exports ACP wire types from a pinned `agent-client-protocol` fork and
adds only agentkit-specific glue: session binding, observer routing, prompt
conversion, cancellation handles, and approval resolver abstractions.

Hosts can wire `AcpIntegration` into their own `AgentBuilder` as a
`LoopObserver`, bind ACP session ids to agentkit session ids, and drain
`AcpClientMessage`s into their ACP connection.

For standalone agents, `AcpHeadlessRuntime` serves ACP over stdio or an
upstream SDK transport. It handles initialize, session lifecycle, prompt
conversion, streaming updates, cancellation, and ACP permission requests for
agentkit approval interrupts.

## Opt-in ACP v2 foundation

The crate root, default features, and `wire` module remain ACP v1. To use the
experimental upstream ACP v2 protocol, enable the additive feature:

```toml
agentkit-acp = { version = "0.10.11", features = ["protocol-v2"] }
```

The feature maps directly to the pinned fork's
`agent-client-protocol/unstable_protocol_v2` feature. V2 APIs and v2 wire
types live only under `agentkit_acp::v2` (and
`agentkit_acp::v2::wire`). Enable `unstable-inject` instead to add the unstable
ACP v2 session-injection surface; it implies `protocol-v2`.

`v2::AcpHeadlessRuntime` supports ACP v2 initialize, new/list/resume
session, prompt, cancel, session updates, and close. Listing and resume cover
the runtime's active in-memory sessions; replay is not supported. Each session
owns a worker and loop driver, so work in one session does not block request
handling for another. A prompt is acknowledged before model work completes. `Running` and `Idle` are
derived from authoritative loop `TurnStarted` and `TurnFinished` events, not
from calls that merely poll the driver. The session worker also wakes for
autonomous task-manager updates, while prompt ownership, cancellation, and
logical-turn cleanup are serialized so a prompt finishes only once. Session
updates retain structured background content and acknowledged updates do not
overtake earlier notifications. User, visible-agent, and thought message IDs
are distinct and stable for the lifetime of a prompt.

With `unstable-inject`, the runtime advertises only `steer` delivery with
finish-current-stream behavior. `session/inject` returns an agent-owned message
ID after its response frame is enqueued, preserves every `ContentBlock`, and
delivers at the next safe model/tool boundary. Single and batch requests use the
same receipt-backed acceptance path. Once response commitment succeeds, request
or session cancellation cannot remove the reservation; a committed steer that
misses the cancelled turn carries into the next prompt. Closing the session may
discard it. `session/revoke_inject` is mandatory and is serialized with
acknowledged `UserMessage` forwarding. Queueing, stream interruption, and
replacement are not supported. Delivered IDs retain their `already_delivered`
classification for the session lifetime. To bound that history safely, each
session accepts at most 4,096 injections and rejects later accepts with a
lifetime-limit error.

This first v2 foundation streams text, reasoning, and tool lifecycle updates.
The v1 permission bridge is not exposed through v2 wire types. Unsupported
approval requests are resolved as denials while retaining a provider-valid
transcript, so the prompt ends with the custom `_error` stop reason rather than
`Refusal`. Already accepted steers remain pending and are delivered at the next
safe boundary. Because the SDK marks
protocol v2 unstable, all APIs in the `v2` namespace can evolve with the pinned
SDK fork. The workspace patch is intentionally unpublishable until these SDK
APIs are available in an upstream release.

Run the stable v1 in-memory end-to-end example with:

```sh
cargo run -p openrouter-acp-trio
```
