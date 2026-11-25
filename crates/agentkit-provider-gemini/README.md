# agentkit-provider-gemini

Google Gemini API (`generateContent` / `streamGenerateContent`) `ModelAdapter` implementation for [`agentkit`](https://github.com/danielkov/agentkit).

## Features

- **Native Gemini REST protocol** — direct `ModelAdapter` implementation supporting both the Gemini Developer API (`generativelanguage.googleapis.com/v1beta`) and Google Cloud Vertex AI (`GeminiConfig::vertex`).
- **Streaming SSE by default** — incremental text and thought deltas over `:streamGenerateContent?alt=sse`, with a buffered JSON fallback (`with_streaming(false)`).
- **Thinking & thought signatures** — supports both Gemini 2.5 (`ThinkingConfig::Budget`) and Gemini 3+ (`ThinkingConfig::Level`) while losslessly round-tripping `thoughtSignature` across multi-turn tool calls.
- **Native function calling & server tools** — passes tool schemas via `parametersJsonSchema`, supports `toolConfig.functionCallingConfig`, and exposes built-in `GoogleSearch`, `CodeExecution`, and `UrlContext` tools.
- **Structured outputs & multimodal I/O** — supports `responseJsonSchema` plus inline/URI images, audio, video, and documents.
- **Full resilience & observability parity** — shared `agentkit-http` authentication (`x-goog-api-key`, OAuth2/ADC bearer tokens, reactive 401 refresh), `ResilienceConfig` retries honoring `Retry-After` and `google.rpc.RetryInfo`, pre-visible-output SSE stream replay, typed `RetryObserver` events (`ProviderRoute::Gemini`), and typed `UpstreamErrorKind::ContextLengthExceeded` recovery.

## Quick start

```rust,no_run
use agentkit_http::ResilienceConfig;
use agentkit_loop::{Agent, SessionConfig};
use agentkit_provider_gemini::{GeminiAdapter, GeminiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = GeminiConfig::from_env()?
        .with_resilience(ResilienceConfig::default());
    let adapter = GeminiAdapter::new(config)?;

    let agent = Agent::builder()
        .model(adapter)
        .build()?;

    let _driver = agent.start(SessionConfig::new("demo")).await?;
    Ok(())
}
```
