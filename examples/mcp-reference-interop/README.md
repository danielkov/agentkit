# `mcp-reference-interop`

Live Streamable HTTP interop checks for `agentkit-mcp` against the official Rust
MCP SDK, `rmcp`.

This example does not mock the transport layer. It launches three `rmcp`-backed
reference servers and probes them through the real `agentkit-mcp` client:

- legacy (2025-11-25) stateful Streamable HTTP with SSE responses
- legacy (2025-11-25) stateless Streamable HTTP with JSON responses
- modern (2026-07-28) sessionless Streamable HTTP with acknowledged subscriptions

Each probe verifies:

- modern `server/discover` or legacy `initialize` and capability discovery
- tool listing and `tools/call`
- resource listing and `resources/read`
- prompt listing and `prompts/get`
- negotiated protocol header propagation
- stateful session header reuse and `DELETE` on close
- legacy interrupted SSE response resume via `Last-Event-ID`
- modern `subscriptions/listen` without initialization, GET streams, or session cleanup

## Run

Probe the stateful server:

```bash
cargo run -p mcp-reference-interop -- stateful
```

Probe the stateless JSON-response server:

```bash
cargo run -p mcp-reference-interop -- stateless
```

Probe the modern subscription server:

```bash
cargo run -p mcp-reference-interop -- modern
```

## Live Tests

The integration tests are self-contained and run with Cargo alone.

Run the stateful reference-implementation interop test:

```bash
cargo test -p mcp-reference-interop rust_sdk_stateful_streamable_http_interop -- --nocapture
```

Run the stateless JSON-response reference-implementation interop test:

```bash
cargo test -p mcp-reference-interop rust_sdk_stateless_json_streamable_http_interop -- --nocapture
```

Run the full transport/lifecycle coverage:

```bash
cargo test -p mcp-reference-interop -- --nocapture
```

## Upstream Targets

- Rust SDK crate: `rmcp = 3.5.1` (resolved by the workspace lockfile)
- Upstream repository: <https://github.com/modelcontextprotocol/rust-sdk>
