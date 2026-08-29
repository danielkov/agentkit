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

Create a config with `OpenAIConfig::new(api_key, model)` and chain `.with_*()` builders for optional parameters. Alternatively, `OpenAIConfig::from_env()` reads from environment variables:

| Variable          | Required | Default                                      |
| ----------------- | -------- | -------------------------------------------- |
| `OPENAI_API_KEY`  | yes      | --                                           |
| `OPENAI_MODEL`    | no       | `gpt-4o`                                     |
| `OPENAI_BASE_URL` | no       | `https://api.openai.com/v1/chat/completions` |

## Authentication and resilience

`OpenAIConfig::new(api_key, model)` and `OpenAIResponsesConfig::new(api_key,
model)` retain the simple bearer-token constructors. The chat adapter and the
Responses config also accept a custom `agentkit_http::Authentication` or
`AuthenticationProvider`; the provider receives the opaque prior authentication
attempt during the single reactive 401 refresh. The standard static bearer and
header constructors attach an ephemeral, non-secret binding automatically.
Custom providers that need Responses continuation replay must attach a stable,
non-secret credential identity or generation with
`AuthenticationAttempt::with_binding`. If a reactive refresh changes that
binding, the adapter fails rather than sending the already encoded, binding-bound
body. Retries and timeouts are opt-in with
`.with_resilience(ResilienceConfig)`. Without it,
transient transport/status/stream failures get one attempt (the one permitted
401 refresh remains part of authentication).

Responses retries reuse one clone-cheap serialized request body and a
body-bound idempotency key. Events stream as soon as they are decoded. A failed
attempt is replayed automatically only before its first event becomes visible.
After visible output, replay is disabled unless the upstream consumer explicitly
enables `agentkit_loop::response_attempt` replacement on `SessionConfig` and
handles its reserved marker by discarding the preceding attempt. Cancellation, the logical retry deadline, stream-idle timeout, absolute
per-attempt deadline (including stream reads), auth/refresh, and backoff remain
bounded. Responses default to a 32 MiB serialized request limit, 16 MiB per
attempt, and 64 MiB aggregate wire limit across retries. Requests and responses
also default to at most 100,000 items/events and 8 MiB per text, reasoning,
ciphertext, or media field. Override these with `OpenAIResponsesLimits` and
`.with_limits(...)`; for example, a host can set `max_items` to `10_000` and
`max_text_bytes` to `1024 * 1024` while retaining the default aggregate bounds.
Limits must be non-zero; the per-field bound must fit both request and attempt
bounds, and the per-attempt bound must fit the aggregate wire bound.

## Responses API

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
and `with_request_policy` can override public/private request-field differences. The
public and private profiles request encrypted reasoning continuation data by
default. The private profile downgrades system messages to developer messages,
defaults `parallel_tool_calls` to `true`,
omits unsupported `max_output_tokens`, sends `originator`/`session-id`, and
replays validated `x-codex-turn-state` only within one logical turn and its
retries. HTTP turn state is accepted only from a successful SSE response; the
equivalent `response.metadata` headers can update retry context, and all header,
metadata, and retry values must agree. It does not perform credential discovery
or model catalog lookups.

Continuation metadata is versioned and bound to the authentication binding,
model, session, provider item ID, and item kind. Durable encrypted reasoning,
function-call, and generated-image continuation metadata is emitted and replayed
only when authentication supplies a binding. Valid metadata for another binding
is omitted safely; malformed metadata is a protocol error. Private deployments
that need to resume Kit's legacy `openai.subscription.v1` schema can install a
credential-binding validator with
`with_legacy_subscription_continuation_authenticator`; AgentKit validates the
remaining schema/model/session fields and emits only current continuation
metadata. The private profile accepts image/audio transcript inputs and
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
