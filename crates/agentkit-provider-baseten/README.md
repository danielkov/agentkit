# agentkit-provider-baseten

Baseten Model API adapter for the agentkit agent loop. It uses Baseten's
OpenAI-compatible Chat Completions API and supports streaming and tool calls.

## Configuration

```rust,no_run
use agentkit_provider_baseten::{BasetenAdapter, BasetenConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = BasetenConfig::new(
    "YOUR_BASETEN_API_KEY",
    "openai/gpt-oss-120b",
)
.with_temperature(0.0)
.with_max_tokens(4096);
let adapter = BasetenAdapter::new(config)?;
# Ok(())
# }
```

`BasetenConfig::from_env()` reads:

| Variable | Required | Default |
| --- | --- | --- |
| `BASETEN_API_KEY` | yes | -- |
| `BASETEN_MODEL` | yes | -- |
| `BASETEN_BASE_URL` | no | `https://inference.baseten.co/v1/chat/completions` |

## Authentication and resilience

`BasetenConfig` stores credentials as a first-class
`agentkit_http::Authentication`. A bare string passed to `BasetenConfig::new`
or `.with_authentication(...)` is shorthand for bearer authentication. Use
`.with_authentication_provider(...)` for a custom refresh-capable
`AuthenticationProvider`.

Resilience is opt-in: `resilience` is an `Option<ResilienceConfig>` that
defaults to `None`. Calling `.with_resilience(...)` enables retries and
timeouts; leaving it as `None` preserves the existing single-attempt behavior.

To use an OpenAI-compatible dedicated deployment, set `BASETEN_BASE_URL` to
the full endpoint, for example:

```text
https://model-{model_id}.api.baseten.co/environments/production/sync/v1/chat/completions
```
