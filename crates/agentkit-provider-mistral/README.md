# agentkit-provider-mistral

Mistral model adapter for the agentkit agent loop.

This crate provides `MistralAdapter` and `MistralConfig` for connecting the
agent loop to the [Mistral AI](https://mistral.ai) chat completions API. It
handles request translation, response normalization, and usage reporting for
Mistral-backed sessions.

Note: Mistral uses `max_tokens` instead of the `max_completion_tokens` field
used by most other OpenAI-compatible APIs.

## Configuration

Create a config with `MistralConfig::new(api_key, model)` and chain `.with_*()` builders for optional parameters. Alternatively, `MistralConfig::from_env()` reads from environment variables:

| Variable           | Required | Default                                      |
| ------------------ | -------- | -------------------------------------------- |
| `MISTRAL_API_KEY`  | yes      | --                                           |
| `MISTRAL_MODEL`    | no       | `mistral-small-latest`                       |
| `MISTRAL_BASE_URL` | no       | `https://api.mistral.ai/v1/chat/completions` |

## Examples

### Minimal chat agent

```rust,no_run
use agentkit_loop::{Agent, SessionConfig};
use agentkit_provider_mistral::{MistralAdapter, MistralConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = MistralConfig::new("sk-...", "mistral-large-latest");
let adapter = MistralAdapter::new(config)?;

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
use agentkit_provider_mistral::{MistralAdapter, MistralConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = MistralConfig::new("sk-...", "mistral-large-latest")
    .with_temperature(0.0)
    .with_max_tokens(4096);

let adapter = MistralAdapter::new(config)?;
# Ok(())
# }
```

### Environment-based configuration with overrides

```rust,no_run
use agentkit_provider_mistral::{MistralAdapter, MistralConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = MistralConfig::from_env()?
    .with_temperature(0.0)
    .with_max_tokens(512);

let adapter = MistralAdapter::new(config)?;
# Ok(())
# }
```
