# agentkit-provider-vllm

vLLM model adapter for the agentkit agent loop.

This crate provides `VllmAdapter` and `VllmConfig` for connecting the agent
loop to a [vLLM](https://docs.vllm.ai) server via its OpenAI-compatible chat
completions endpoint. It handles request translation and response normalization
for vLLM-backed sessions.

An API key is optional — vLLM servers can run with or without authentication
(controlled by the `--api-key` flag when starting the server).

## Configuration

Set the following environment variables before calling `VllmConfig::from_env()`:

| Variable        | Required | Default                                     |
| --------------- | -------- | ------------------------------------------- |
| `VLLM_MODEL`    | yes      | --                                          |
| `VLLM_BASE_URL` | no       | `http://localhost:8000/v1/chat/completions` |
| `VLLM_API_KEY`  | no       | --                                          |

## Examples

### Minimal chat agent

```rust,no_run
use agentkit_loop::{Agent, SessionConfig};
use agentkit_provider_vllm::{VllmAdapter, VllmConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = VllmConfig::new("meta-llama/Llama-3.1-8B-Instruct");
let adapter = VllmAdapter::new(config)?;

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

### Authenticated vLLM server

```rust,no_run
use agentkit_provider_vllm::{VllmAdapter, VllmConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = VllmConfig::new("meta-llama/Llama-3.1-8B-Instruct")
    .with_api_key("my-secret-key")
    .with_temperature(0.0)
    .with_max_completion_tokens(4096);

let adapter = VllmAdapter::new(config)?;
# Ok(())
# }
```

### Remote server with environment-based configuration

```rust,no_run
use agentkit_provider_vllm::{VllmAdapter, VllmConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = VllmConfig::from_env()?
    .with_base_url("http://gpu-server:8000/v1/chat/completions")
    .with_temperature(0.0);

let adapter = VllmAdapter::new(config)?;
# Ok(())
# }
```
