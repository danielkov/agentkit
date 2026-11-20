# agentkit-acp

Agent Client Protocol integration for agentkit hosts.

This crate re-exports ACP wire types from the upstream `agent-client-protocol` SDK
and adds agentkit-specific glue: session binding, observer routing, prompt
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

The feature maps directly to the SDK's
`agent-client-protocol/unstable_protocol_v2` feature. V2 APIs and v2 wire
types live only under `agentkit_acp::v2` (and
`agentkit_acp::v2::wire`). Enable `unstable-prompt-delivery` to add ACP v2 prompt-delivery
primitives; it implies `protocol-v2`.

`v2::AcpHeadlessRuntime` supports ACP v2 initialize, new/list/resume
session, prompt, cancel, session updates, and close. Listing and resume cover
the runtime's active in-memory sessions; transcript replay is not supported.
Prompt idempotency keys stay with an in-memory session, including across
`session/resume`. The headless runtime does not persist sessions across
connections. Each session
owns a worker and loop driver, so work in one session does not block request
handling for another. A prompt is acknowledged after the driver appends it to
the transcript, without waiting for model work to complete. The runtime uses
loop `TurnStarted` and `TurnFinished` events for `Running` and `Idle`; it also
reports `Idle` if an inserted prompt is cancelled before the loop starts. The session worker also wakes for
autonomous task-manager updates, while prompt ownership, cancellation, and
logical-turn cleanup are serialized so a prompt finishes only once. Session
updates retain structured background content and acknowledged updates do not
overtake earlier notifications. User, visible-agent, and thought message IDs
are distinct and stable for the lifetime of a prompt.

With `unstable-prompt-delivery`, a host-owned turn actor can use `v2::delivery::PendingPrompts`
through its session handle. Enqueue `session/prompt` requests without responding immediately.
The actor claims and inserts them at its model-loop safe points or at the start of a new turn,
and only then completes the original request with its `messageId`. Request cancellation can
withdraw pending work without stopping the active turn. Session cancellation drops pending
ordinary prompts while preserving explicit steer and queue requests; close rejects all
pending requests. Steers take precedence in receive order, and queued prompts each start a
separate turn. Pending content and inline media budgets apply to each session.

`v2::agent_capabilities()` reports the base headless capabilities. Hosts that connect the
primitives to their actor can advertise the modes they implement under `session.prompt.delivery`.

This first v2 foundation streams text, reasoning, and tool lifecycle updates.
The v1 permission bridge is not exposed through v2 wire types. Unsupported
approval requests are resolved as denials while retaining a provider-valid
transcript, so the prompt ends with the custom `_error` stop reason rather than
`Refusal`. Because the SDK marks protocol v2 unstable, all APIs in the `v2`
namespace can evolve with the upstream SDK.

Run the stable v1 in-memory end-to-end example with:

```sh
cargo run -p openrouter-acp-trio
```
