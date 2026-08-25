# ACP integration

The [Agent Client Protocol (ACP)](https://agentclientprotocol.com) standardizes communication between clients (code editors, IDEs, desktop apps) and coding agents. Where MCP connects an agent to external tools, ACP connects a client to the agent itself: session lifecycle, prompt turns, streamed updates, tool call reporting, and permission prompts all travel over JSON-RPC — usually with the agent running as an editor child process on stdio. This chapter covers [`agentkit-acp`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-acp): how an agentkit host becomes ACP-addressable, and how a standalone agent serves ACP directly.

## Built on the official SDK

Like `agentkit-mcp`, this crate does not define a parallel protocol vocabulary. It builds on the official Rust SDK, [`agent-client-protocol`](https://crates.io/crates/agent-client-protocol), and re-exports the stable v1 wire types (`SessionId`, `ContentBlock`, `SessionUpdate`, `ToolCallUpdate`, `StopReason`, …) at the crate root and under `agentkit_acp::wire`. The full upstream SDK is available as `agentkit_acp::sdk`. Agentkit owns only the host-facing glue: session binding, observer routing, prompt conversion, cancellation handles, and approval resolution.

- **Protocol docs:** [agentclientprotocol.com](https://agentclientprotocol.com/protocol/v1/overview)
- **Rust SDK:** [`agent-client-protocol` on crates.io](https://crates.io/crates/agent-client-protocol)

## Opt-in ACP v2 runtime

ACP v2 support is additive and disabled by default. Enable it explicitly:

```toml
agentkit-acp = { version = "0.10.8", features = ["protocol-v2"] }
```

`protocol-v2` enables the official upstream
`agent-client-protocol/unstable_protocol_v2` feature. The root API and
`agentkit_acp::wire` continue to expose stable v1 behavior. Experimental v2
runtime APIs and official v2 wire types are isolated under
`agentkit_acp::v2` and `agentkit_acp::v2::wire`; v1 wire types are not part of
that namespace.

Build a v2 server with `agentkit_acp::v2::AcpHeadlessRuntime`. Its factory is
called once for each `session/new` and receives a v2 session ID, an agentkit
session ID, the v2 output observer, and a cancellation handle. Install the
observer and cancellation handle on the returned agent loop in the same way as
the v1 factory.

The v2 prompt lifecycle differs from v1: `session/prompt` acknowledges
acceptance immediately instead of waiting for the turn to finish. The runtime
then emits, in order:

1. a `user_message` update with a generated stable message ID;
2. a `running` state update;
3. streamed agent message or thought chunks with distinct stable message IDs;
4. tool-call lifecycle updates when tools run;
5. an `idle` state update with the final stop reason.

Each session has its own worker and loop driver. Independent sessions can make
progress concurrently, while a second prompt for a running session is rejected.
`session/cancel` interrupts only the selected session and produces an idle
`cancelled` update after loop cleanup. `session/close` cooperatively cancels
work and drops the session worker. `session/list` and `session/resume` cover
active in-memory sessions; replay is not supported.

The initial v2 foundation routes text, reasoning, and tool lifecycle updates.
ACP v2 permission callbacks are intentionally deferred; an unsupported approval
interrupt retains the transcript and ends the prompt with the custom `_error`
stop reason rather than `refusal`. Upstream labels the v2 protocol unstable, so
opt-in callers should expect the `v2` namespace to track official SDK changes.

## Two integration shapes

`agentkit-acp` exposes the same functionality at two levels:

1. **Hybrid integration** — the primitive. `AcpIntegration` returns the loop-facing pieces a host wires into its own `AgentBuilder`: a shared `LoopObserver`, an input surface, cancellation handles, an approval resolver seam, and a session registry. Local UI, background work, and ACP clients can all feed and observe the same agent sessions.
2. **Headless runtime** — a convenience layer built from those same parts. `AcpHeadlessRuntime` owns the session table and serving loop for standalone agents that speak ACP over stdio or a custom transport.

## Hybrid integration

`AcpIntegration` is the shared integration object. It implements `LoopObserver`, so the host registers it on its own agent like any other observer:

```rust,ignore
use agentkit_acp::{AcpIntegration, AcpSessionBinding, ClientPermissionResolver};

let acp = Arc::new(
    AcpIntegration::builder()
        .name("my-agent")
        .version(env!("CARGO_PKG_VERSION"))
        .approval_resolver(ClientPermissionResolver::new())
        .build()?,
);

let session = acp.bind_session(AcpSessionBinding::new(
    acp_session_id,
    agentkit_session_id,
    client_handle,
))?;

let agent = Agent::builder()
    .model(adapter)
    .observer(Arc::clone(&acp))
    .cancellation(session.cancellation_handle())
    .build()?;
```

`bind_session` associates an ACP session id with an agentkit session id and a per-session `AcpClientHandle` — the send capability for that client connection. The integration routes observed events by session id into per-session ACP state, so one shared object serves any number of concurrent sessions.

The observer is intentionally not responsible for mutation or control:

- **Input** goes through the host's turn arbiter. `AcpIntegration::input_port()` returns an `AcpInputPort` that converts ACP prompt content blocks into agentkit `Item`s; the host decides when to submit them.
- **Cancellation** goes through a per-session `CancellationController`. ACP `session/cancel` maps to `AcpIntegration::interrupt_session(&acp_session_id)`.
- **Approvals** go through `LoopStep::Interrupt` — see the approval bridge below.
- **Lookups** go through `AcpIntegration::session_registry()`, which maps ids in both directions and exposes per-session workspace roots and metadata.

### Session addressing

ACP routing is possible because loop observation is session-addressed. `LoopObserver::handle_event` receives an `ObservedEvent` envelope — the `session_id` plus the `AgentEvent` — so one shared observer can fan events out to the right ACP session:

```rust,ignore
impl LoopObserver for AcpIntegration {
    fn handle_event(&self, event: ObservedEvent) {
        // Routes by event.session_id into per-session ACP state.
    }
}
```

`handle_event` is synchronous, so the integration bridges to async JSON-RPC sending through an unbounded channel per session. The host (or the headless runtime) drains that channel and sends ACP `session/update` notifications; `flush_session_updates` pushes any buffered updates for a session through its client handle.

### Event mapping

Observed agentkit events convert to ACP session updates:

| agentkit event                  | ACP update                              |
| ------------------------------- | --------------------------------------- |
| `ContentDelta` (text part)      | `SessionUpdate::AgentMessageChunk`      |
| `ContentDelta` (reasoning part) | `SessionUpdate::AgentThoughtChunk`      |
| `ToolCallRequested`             | `SessionUpdate::ToolCall`               |
| `ToolExecutionStarted`          | `ToolCallUpdate { status: InProgress }` |
| `ToolExecutionProgress`         | `ToolCallUpdate { status: InProgress }` |
| `ToolResultReceived`            | terminal `ToolCallUpdate`               |
| `TurnFinished`                  | `PromptResponse` stop reason            |

The observer is stateful: `Delta::AppendText` only carries a `part_id` and a chunk, so the integration remembers each `Delta::BeginPart { part_id, kind }` to decide whether a chunk is assistant message text or reasoning. Tool call ids are preserved — the agentkit `ToolCallId` becomes the ACP `ToolCallId`, with tool inputs exposed as `raw_input` and outputs as `raw_output` plus text content.

Final stop reasons map through `finish_reason_to_stop_reason`:

| `FinishReason` | ACP `StopReason`         |
| -------------- | ------------------------ |
| `Completed`    | `EndTurn`                |
| `MaxTokens`    | `MaxTokens`              |
| `Cancelled`    | `Cancelled`              |
| `Blocked`      | `Refusal`                |
| `Error`        | JSON-RPC error           |
| `ToolCall`     | not final — keep driving |

## The approval bridge

Agentkit approval interrupts (`LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(...))`) are loop-level blocking pauses. ACP permission prompts (`session/request_permission`) are client-facing JSON-RPC requests. The resolver layer connects the two:

```rust,ignore
#[async_trait]
pub trait AcpApprovalResolver: Send + Sync + 'static {
    async fn resolve(
        &self,
        ctx: AcpApprovalContext,
        client: AcpClientHandle,
    ) -> Result<AcpApprovalDecision, AcpRuntimeError>;
}
```

Three resolvers ship with the crate, and builders force an explicit choice — there is no silent default:

| Resolver                   | Behavior                                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `ClientPermissionResolver` | Converts the `ApprovalRequest` into an ACP `session/request_permission`, awaits the client's selected option |
| `AutoApproveResolver`      | Approves everything — tests and trusted non-interactive hosts                                                |
| `AutoDenyResolver`         | Denies everything — locked-down non-interactive hosts                                                        |

Decisions are `AllowOnce`, `AllowAlways`, `RejectOnce`, `RejectAlways`, or `PatchAndAllow { input }` (approve with modified tool input). An optional `AcpApprovalMemory` (built-ins: `NoopApprovalMemory`, `InMemoryApprovalMemory`) is checked before the resolver runs; `AllowAlways`/`RejectAlways` decisions are remembered after applying, one-shot decisions and patched input never are.

The resolver does not replace permission policy. `agentkit-tools-core` policy still decides whether an action is allowed, denied, or requires approval; the ACP resolver only decides how to answer an already-surfaced approval request. Approval waits are cancellation-aware: if ACP cancellation arrives while a permission prompt is pending, the runtime stops waiting, clears the pending approvals (`LoopDriver::cancel_pending_approvals` pairs each cleared approval with an error tool result so the transcript stays provider-valid), and finishes the prompt with `StopReason::Cancelled`.

## The headless runtime

For standalone agents that don't already own a serving loop, `AcpHeadlessRuntime` handles initialize, session lifecycle, prompt conversion, streaming updates, cancellation, and permission requests:

```rust,ignore
use agentkit_acp::{AcpHeadlessRuntime, AcpIntegration, ClientPermissionResolver};

let integration = AcpIntegration::builder()
    .name("agentkit")
    .approval_resolver(ClientPermissionResolver::new())
    .build()?;

AcpHeadlessRuntime::builder()
    .agent_factory(agent_factory)
    .integration(integration)
    .serve_stdio()
    .await?;
```

The runtime constructs one agent per ACP session through the `AcpAgentFactory` trait:

```rust,ignore
#[async_trait]
pub trait AcpAgentFactory<M: ModelAdapter>: Send + Sync + 'static {
    async fn start(
        &self,
        ctx: AcpAgentFactoryContext,
    ) -> Result<LoopDriver<M::Session>, AcpRuntimeError>;
}
```

`AcpAgentFactoryContext` carries everything the factory needs to wire an agent before `Agent::start`: the ACP and agentkit session ids, `cwd`, additional workspace roots, the shared `Arc<AcpIntegration>` (register it as the observer), the session's cancellation handle, and session metadata. The factory owns model adapter configuration, tools, context loading, and permissions — the runtime owns the protocol.

Protocol handling maps ACP session lifecycle onto loop drivers:

- `session/new` → invoke the factory, store the returned `LoopDriver`
- `session/prompt` → convert content blocks to `Item`s, drive the loop to `LoopStep::Finished`, return the mapped `StopReason`
- `session/cancel` → interrupt that session's `CancellationController`

Each session owns its own driver behind its own lock, so a slow or approval-blocked session never serializes unrelated sessions. Prompts containing media the conversion path cannot represent return a structured ACP error rather than silently dropping user input.

`serve_stdio()` (behind the crate's default `stdio` feature) serves an editor child process. `serve_transport(...)` accepts any upstream `ConnectTo<Agent>` transport — in-memory pipes for tests, or whatever the SDK ships next.

## Feature flags

On the umbrella crate, ACP is behind the `acp` feature (implies `loop`):

```toml
[dependencies]
agentkit = { version = "0.10.8", features = ["acp"] }
```

The `agentkit-acp` crate itself has a default `stdio` feature that gates `serve_stdio`, which uses the upstream SDK's built-in `Stdio` transport. Its `unstable-acp` feature forwards to the upstream SDK's unstable protocol surface.

## What the crate does not own

- the ACP schema or JSON-RPC framing — that's the upstream SDK
- session persistence — hosts extract state via `LoopDriver::snapshot`, as in [Session persistence](./session-persistence.md)
- durable approval trust stores — approval memory keys are policy-owned; persistent stores remain host concerns
- filesystem/terminal client callbacks and ACP auth — later phases, gated on stable upstream support

> **Example:** [`openrouter-acp-trio`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-acp-trio) runs three OpenRouter-backed agents (orchestrator, worker, reviewer) that call each other over in-memory ACP endpoints while a REPL drives the orchestrator through a persistent ACP session — session binding, streamed updates, tool call reporting, and agent-to-agent handoffs in one program.
>
> **Crate:** [`agentkit-acp`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-acp) — depends on [`agentkit-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-core), [`agentkit-loop`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-loop), [`agentkit-tools-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-tools-core), and [`agent-client-protocol`](https://crates.io/crates/agent-client-protocol). Design notes: [`docs/acp.md`](https://github.com/danielkov/agentkit/blob/main/docs/acp.md).
