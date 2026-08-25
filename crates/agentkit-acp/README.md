# agentkit-acp

Agent Client Protocol integration for agentkit hosts.

This crate re-exports upstream ACP wire types from `agent-client-protocol` and
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
agentkit-acp = { version = "0.10.8", features = ["protocol-v2"] }
```

The feature maps directly to the official
`agent-client-protocol/unstable_protocol_v2` feature. V2 APIs and official v2
wire types live only under `agentkit_acp::v2` (and
`agentkit_acp::v2::wire`).

`v2::AcpHeadlessRuntime` supports ACP v2 initialize, new session, prompt,
cancel, session updates, and close. Each session owns a worker and loop driver,
so work in one session does not block request handling for another. A prompt is
accepted before model work completes, then the runtime emits ordered
`UserMessage`, `Running`, streamed output, and `Idle` updates. User and agent
message IDs are stable for the lifetime of a prompt.

This first v2 foundation streams text and reasoning output. An approval
interrupt currently ends the prompt with `Refusal`; the v1 permission bridge is
not exposed through v2 wire types. Because upstream marks protocol v2 unstable,
all APIs in the `v2` namespace can evolve with the official SDK.

Run the stable v1 in-memory end-to-end example with:

```sh
cargo run -p openrouter-acp-trio
```
