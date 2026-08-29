# agentkit-http

<p align="center">
  <a href="https://crates.io/crates/agentkit-http"><img src="https://img.shields.io/crates/v/agentkit-http.svg?logo=rust" alt="Crates.io" /></a>
  <a href="https://docs.rs/agentkit-http"><img src="https://img.shields.io/docsrs/agentkit-http?logo=docsdotrs" alt="Documentation" /></a>
  <a href="https://github.com/danielkov/agentkit/blob/main/LICENSE"><img src="https://img.shields.io/crates/l/agentkit-http.svg" alt="License" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/MSRV-1.92-blue?logo=rust" alt="MSRV" /></a>
</p>

HTTP client abstraction used across agentkit. The `HttpClient` trait
defines the single `execute(HttpRequest) -> HttpResponse` contract; the
`Http` handle wraps any `Arc<dyn HttpClient>` and exposes a
`reqwest`-style request builder (`HttpRequestBuilder`) for ergonomic call
sites.

A default `reqwest`-backed implementation ships behind the `reqwest-client`
feature, which is **enabled by default**. Disable it to compile trait-only
when you want to bring your own backend:

```toml
agentkit-http = { version = "0.10.5", default-features = false }
```

A second optional feature, `reqwest-middleware-client`, layers
[`reqwest-middleware`](https://crates.io/crates/reqwest-middleware) on top
for retry / tracing middleware stacks.

## Quick start

```rust,no_run
use agentkit_http::{Http, StatusCode};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let http = Http::new(reqwest::Client::new());

let response = http
    .post("https://example.test/echo")
    .bearer_auth("tok")
    .json(&serde_json::json!({ "name": "agentkit" }))
    .send()
    .await?;

assert_eq!(response.status(), StatusCode::OK);
let body: serde_json::Value = response.json().await?;
# Ok(())
# }
```

`Http` exposes the usual `get` / `post` / `put` / `patch` / `delete` /
`request` constructors and is cheap to clone — it holds an `Arc` over the
underlying client.

## Authentication and resilience

`Authentication` is a cloneable, type-erased handle over an async
`AuthenticationProvider`. Providers return sensitive headers plus opaque attempt
state, which lets an adapter perform one reactive 401 refresh without learning
how credentials are stored. Optional non-secret credential bindings let
replay-sensitive adapters bind continuation data to an account or credential
generation. Owned bearer and arbitrary-header credentials are redacted and
zeroized when dropped.

`ResilienceConfig` is opt-in. It combines a wall-clock retry budget, an optional
attempt cap and attempt/stream-idle timeouts with capped exponential full-jitter
backoff and validated `Retry-After`/rate-limit reset hints. Adapters retain
control of replay safety; configuring resilience on `Http` alone does not replay
requests.

## Bring your own client

Any type that implements `HttpClient` plugs in unchanged. This is the seam
agentkit uses for stub clients in tests, custom TLS / proxy stacks, signing
middleware, and non-`reqwest` backends:

```rust
use std::sync::Arc;
use agentkit_http::{Http, HttpClient, HttpError, HttpRequest, HttpResponse};
use async_trait::async_trait;

struct MyClient;

#[async_trait]
impl HttpClient for MyClient {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        // Drive `request.method`, `request.url`, `request.headers`,
        // `request.body` through your transport, then build an
        // `HttpResponse` from the streamed body.
        unimplemented!()
    }
}

let http = Http::from_arc(Arc::new(MyClient) as Arc<dyn HttpClient>);
```

Responses are surfaced as a streaming body (`BodyStream`), so SSE consumers
can pull chunks incrementally rather than buffering the entire payload.
