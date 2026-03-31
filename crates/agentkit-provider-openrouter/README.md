# agentkit-provider-openrouter

OpenRouter model adapter for `agentkit-loop`.

This crate translates between agentkit transcript primitives and OpenRouter chat completion requests, including:

- session and turn adapters
- tool declaration and tool-call decoding
- multimodal user content mapping (images, audio)
- usage and finish-reason normalization
- environment-based configuration helpers

Use it when OpenRouter is the backing model provider for your agent runtime.

## Configuration

Set the following environment variables before calling `OpenRouterConfig::from_env()`:

| Variable | Required | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | -- |
| `OPENROUTER_MODEL` | no | `openrouter/auto` |
| `OPENROUTER_BASE_URL` | no | `https://openrouter.ai/api/v1/chat/completions` |
| `OPENROUTER_APP_NAME` | no | -- |
| `OPENROUTER_SITE_URL` | no | -- |
| `OPENROUTER_MAX_COMPLETION_TOKENS` | no | -- |
| `OPENROUTER_TEMPERATURE` | no | -- |

## Examples

### Minimal chat agent

```rust,no_run
use agentkit_loop::{Agent, PromptCacheRequest, PromptCacheRetention, SessionConfig};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Load API key and model from environment variables.
let config = OpenRouterConfig::from_env()?;
let adapter = OpenRouterAdapter::new(config)?;

let agent = Agent::builder()
    .model(adapter)
    .build()?;

let mut driver = agent
    .start(
        SessionConfig::new("demo").with_cache(
            PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
        ),
    )
    .await?;

let step = driver.next().await?;
println!("{step:?}");
# Ok(())
# }
```

### Explicit configuration with model selection

```rust,no_run
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenRouterConfig::new("sk-or-v1-...", "anthropic/claude-sonnet-4")
    .with_temperature(0.0)
    .with_max_completion_tokens(4096)
    .with_app_name("my-agent")
    .with_site_url("https://example.com");

let adapter = OpenRouterAdapter::new(config)?;
# Ok(())
# }
```

### Environment-based configuration with overrides

```rust,no_run
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OpenRouterConfig::from_env()?
    .with_temperature(0.0)
    .with_max_completion_tokens(512)
    .with_extra_body_value("top_p", 0.95);

let adapter = OpenRouterAdapter::new(config)?;
# Ok(())
# }
```
