# agentkit-provider-ollama

Ollama model adapter for the agentkit agent loop.

This crate provides `OllamaAdapter` and `OllamaConfig` for connecting the
agent loop to a local [Ollama](https://ollama.ai) instance via its
OpenAI-compatible chat completions endpoint. It handles request translation and
response normalization for Ollama-backed sessions.

No API key is required — Ollama runs locally and does not authenticate requests
by default. You need a running Ollama server (e.g. `ollama serve`) with your
desired model pulled (e.g. `ollama pull llama3.1:8b`).

## Configuration

Set the following environment variables before calling `OllamaConfig::from_env()`:

| Variable          | Required | Default                                      |
| ----------------- | -------- | -------------------------------------------- |
| `OLLAMA_MODEL`    | yes      | --                                           |
| `OLLAMA_BASE_URL` | no       | `http://localhost:11434/v1/chat/completions` |

## Examples

### Minimal chat agent

```rust,no_run
use agentkit_loop::{Agent, SessionConfig};
use agentkit_provider_ollama::{OllamaAdapter, OllamaConfig};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
// Ollama must be running locally (e.g. `ollama serve`).
let config = OllamaConfig::new("llama3.1:8b");
let adapter = OllamaAdapter::new(config)?;

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

### Environment-based configuration

```rust,no_run
use agentkit_provider_ollama::{OllamaAdapter, OllamaConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OllamaConfig::from_env()?
    .with_temperature(0.0)
    .with_num_predict(4096);

let adapter = OllamaAdapter::new(config)?;
# Ok(())
# }
```

### Remote Ollama instance

```rust,no_run
use agentkit_provider_ollama::{OllamaAdapter, OllamaConfig};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let config = OllamaConfig::new("llama3.1:8b")
    .with_base_url("http://gpu-server:11434/v1/chat/completions");

let adapter = OllamaAdapter::new(config)?;
# Ok(())
# }
```
