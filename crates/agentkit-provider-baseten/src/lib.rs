//! Baseten Model API adapter for the agentkit agent loop.
//!
//! This crate connects agentkit to Baseten's OpenAI-compatible chat
//! completions API. It supports the shared Model API endpoint by default and
//! dedicated OpenAI-compatible deployments through [`BasetenConfig::with_base_url`].

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "https://inference.baseten.co/v1/chat/completions";

/// Configuration for connecting to Baseten.
///
/// Use a Baseten Model API slug such as `"openai/gpt-oss-120b"`, or the
/// served model name from a dedicated deployment.
#[derive(Clone, Debug)]
pub struct BasetenConfig {
    /// Baseten workspace API key.
    pub api_key: String,
    /// Model API slug or dedicated deployment's served model name.
    pub model: String,
    /// Full chat completions endpoint URL.
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of generated tokens.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter. Support depends on the selected model.
    pub top_k: Option<u32>,
    /// Whether the model may emit multiple tool calls in one turn.
    pub parallel_tool_calls: Option<bool>,
    /// Request SSE streaming responses. Defaults to `true`.
    pub streaming: bool,
}

impl BasetenConfig {
    /// Creates a configuration for Baseten's shared Model API endpoint.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            parallel_tool_calls: None,
            streaming: true,
        }
    }

    /// Overrides the full chat completions endpoint URL.
    ///
    /// For a dedicated production deployment, use a URL like
    /// `https://model-{model_id}.api.baseten.co/environments/production/sync/v1/chat/completions`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, value: f32) -> Self {
        self.temperature = Some(value);
        self
    }

    /// Sets the maximum number of generated tokens.
    pub fn with_max_tokens(mut self, value: u32) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Sets the nucleus sampling parameter.
    pub fn with_top_p(mut self, value: f32) -> Self {
        self.top_p = Some(value);
        self
    }

    /// Sets the top-k sampling parameter.
    pub fn with_top_k(mut self, value: u32) -> Self {
        self.top_k = Some(value);
        self
    }

    /// Sets whether the model may emit multiple tool calls in one turn.
    pub fn with_parallel_tool_calls(mut self, value: bool) -> Self {
        self.parallel_tool_calls = Some(value);
        self
    }

    /// Toggles SSE streaming. Default: `true`.
    pub fn with_streaming(mut self, value: bool) -> Self {
        self.streaming = value;
        self
    }

    /// Builds a configuration from environment variables.
    ///
    /// | Variable | Required | Default |
    /// |---|---|---|
    /// | `BASETEN_API_KEY` | yes | -- |
    /// | `BASETEN_MODEL` | yes | -- |
    /// | `BASETEN_BASE_URL` | no | `https://inference.baseten.co/v1/chat/completions` |
    pub fn from_env() -> Result<Self, BasetenError> {
        let api_key = std::env::var("BASETEN_API_KEY")
            .map_err(|_| BasetenError::MissingEnv("BASETEN_API_KEY"))?;
        let model = std::env::var("BASETEN_MODEL")
            .map_err(|_| BasetenError::MissingEnv("BASETEN_MODEL"))?;

        let mut config = Self::new(api_key, model);
        if let Ok(url) = std::env::var("BASETEN_BASE_URL") {
            config = config.with_base_url(url);
        }
        Ok(config)
    }
}

/// Request parameters serialized into the Baseten request body.
#[derive(Clone, Debug, Serialize)]
pub struct BasetenRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Baseten implementation of [`CompletionsProvider`].
#[derive(Clone, Debug)]
pub struct BasetenProvider {
    api_key: String,
    base_url: String,
    streaming: bool,
    request_config: BasetenRequestConfig,
}

impl From<BasetenConfig> for BasetenProvider {
    fn from(config: BasetenConfig) -> Self {
        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            streaming: config.streaming,
            request_config: BasetenRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_tokens: config.max_tokens,
                top_p: config.top_p,
                top_k: config.top_k,
                parallel_tool_calls: config.parallel_tool_calls,
            },
        }
    }
}

impl CompletionsProvider for BasetenProvider {
    type Config = BasetenRequestConfig;

    fn provider_name(&self) -> &str {
        "Baseten"
    }

    fn endpoint_url(&self) -> &str {
        &self.base_url
    }

    fn config(&self) -> &BasetenRequestConfig {
        &self.request_config
    }

    fn preprocess_request(
        &self,
        builder: agentkit_http::HttpRequestBuilder,
    ) -> agentkit_http::HttpRequestBuilder {
        builder.bearer_auth(&self.api_key).header(
            "User-Agent",
            concat!("agentkit-provider-baseten/", env!("CARGO_PKG_VERSION")),
        )
    }

    fn streaming(&self) -> bool {
        self.streaming
    }
}

/// Model adapter that connects the agentkit loop to Baseten.
#[derive(Clone)]
pub struct BasetenAdapter(CompletionsAdapter<BasetenProvider>);

/// An active Baseten session.
pub type BasetenSession = CompletionsSession<BasetenProvider>;

/// A completed Baseten turn.
pub type BasetenTurn = CompletionsTurn;

impl BasetenAdapter {
    /// Creates a new adapter from the given configuration.
    pub fn new(config: BasetenConfig) -> Result<Self, BasetenError> {
        Ok(Self(CompletionsAdapter::new(BasetenProvider::from(
            config,
        ))?))
    }
}

#[async_trait]
impl ModelAdapter for BasetenAdapter {
    type Session = BasetenSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

/// Errors produced by the Baseten adapter.
#[derive(Debug, Error)]
pub enum BasetenError {
    /// A required environment variable is not set.
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    /// An error from the generic completions adapter.
    #[error(transparent)]
    Completions(#[from] CompletionsError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_model_api_with_streaming() {
        let provider = BasetenProvider::from(BasetenConfig::new("key", "model"));

        assert_eq!(provider.endpoint_url(), DEFAULT_ENDPOINT);
        assert!(provider.streaming());
        assert_eq!(provider.config().model, "model");
    }
}
