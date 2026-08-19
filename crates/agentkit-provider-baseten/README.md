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

To use an OpenAI-compatible dedicated deployment, set `BASETEN_BASE_URL` to
the full endpoint, for example:

```text
https://model-{model_id}.api.baseten.co/environments/production/sync/v1/chat/completions
```
