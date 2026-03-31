//! Mistral model adapter for the agentkit agent loop.
//!
//! Connects to the [Mistral AI](https://mistral.ai) chat completions API.
//!
//! ```rust,ignore
//! use agentkit_provider_mistral::{MistralAdapter, MistralConfig};
//!
//! let adapter = MistralAdapter::new(MistralConfig::from_env()?)?;
//! let agent = Agent::builder().model(adapter).build()?;
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "https://api.mistral.ai/v1/chat/completions";

/// Configuration for connecting to the Mistral API.
#[derive(Clone, Debug)]
pub struct MistralConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
}

impl MistralConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_temperature(mut self, v: f32) -> Self {
        self.temperature = Some(v);
        self
    }

    pub fn with_max_tokens(mut self, v: u32) -> Self {
        self.max_tokens = Some(v);
        self
    }

    pub fn with_top_p(mut self, v: f32) -> Self {
        self.top_p = Some(v);
        self
    }

    pub fn from_env() -> Result<Self, MistralError> {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| MistralError::MissingEnv("MISTRAL_API_KEY"))?;
        let model =
            std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-small-latest".into());

        let mut config = Self::new(api_key, model);

        if let Ok(url) = std::env::var("MISTRAL_BASE_URL") {
            config = config.with_base_url(url);
        }

        Ok(config)
    }
}

/// Mistral uses `max_tokens` instead of `max_completion_tokens`.
#[derive(Clone, Debug, Serialize)]
pub struct MistralRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct MistralProvider {
    api_key: String,
    base_url: String,
    request_config: MistralRequestConfig,
}

impl From<MistralConfig> for MistralProvider {
    fn from(config: MistralConfig) -> Self {
        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            request_config: MistralRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_tokens: config.max_tokens,
                top_p: config.top_p,
            },
        }
    }
}

impl CompletionsProvider for MistralProvider {
    type Config = MistralRequestConfig;

    fn provider_name(&self) -> &str {
        "Mistral"
    }
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }
    fn config(&self) -> &MistralRequestConfig {
        &self.request_config
    }

    fn preprocess_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
            .bearer_auth(&self.api_key)
            .header("User-Agent", "agentkit-provider-mistral/0.1.0")
    }
}

#[derive(Clone)]
pub struct MistralAdapter(CompletionsAdapter<MistralProvider>);

pub type MistralSession = CompletionsSession<MistralProvider>;
pub type MistralTurn = CompletionsTurn;

impl MistralAdapter {
    pub fn new(config: MistralConfig) -> Result<Self, MistralError> {
        let provider = MistralProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }
}

#[async_trait]
impl ModelAdapter for MistralAdapter {
    type Session = MistralSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

#[derive(Debug, Error)]
pub enum MistralError {
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    #[error(transparent)]
    Completions(#[from] CompletionsError),
}
