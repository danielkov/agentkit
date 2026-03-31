# agentkit-provider-groq

Groq model adapter for the agentkit agent loop.

This crate provides `GroqAdapter` and `GroqConfig` for connecting the agent
loop to the [Groq](https://groq.com) chat completions API, which serves
open-source models on custom LPU hardware. It handles request translation,
response normalization, and usage reporting for Groq-backed sessions.

## Configuration

Set the following environment variables before calling `GroqConfig::from_env()`:

| Variable        | Required | Default                                           |
| --------------- | -------- | ------------------------------------------------- |
| `GROQ_API_KEY`  | yes      | --                                                |
| `GROQ_MODEL`    | no       | `llama-3.1-8b-instant`                            |
| `GROQ_BASE_URL` | no       | `https://api.groq.com/openai/v1/chat/completions` |

## Examples

### Minimal chat agent

```rust,no_run
use agentkit_loop::{Agent, SessionConfig};
use agentkit_provider_groq::{GroqAdapter, GroqConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Load API key and model from environment variables.
let config = GroqConfig::from_env()?;
let adapter = GroqAdapter::new(config)?;

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

### Explicit configuration

```rust,no_run
use agentkit_provider_groq::{GroqAdapter, GroqConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = GroqConfig::new("gsk_...", "llama-3.3-70b-versatile")
    .with_temperature(0.0)
    .with_max_completion_tokens(4096);

let adapter = GroqAdapter::new(config)?;
# Ok(())
# }
```

### Environment-based configuration with overrides

```rust,no_run
use agentkit_provider_groq::{GroqAdapter, GroqConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = GroqConfig::from_env()?
    .with_temperature(0.0)
    .with_max_completion_tokens(512);

let adapter = GroqAdapter::new(config)?;
# Ok(())
# }
```
