//! Groq model adapter for the agentkit agent loop.
//!
//! Connects to the [Groq](https://groq.com) chat completions API.
//!
//! ```rust,ignore
//! use agentkit_provider_groq::{GroqAdapter, GroqConfig};
//!
//! let adapter = GroqAdapter::new(GroqConfig::from_env()?)?;
//! let agent = Agent::builder().model(adapter).build()?;
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "https://api.groq.com/openai/v1/chat/completions";

/// Configuration for connecting to the Groq API.
#[derive(Clone, Debug)]
pub struct GroqConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub temperature: Option<f32>,
    pub max_completion_tokens: Option<u32>,
    pub top_p: Option<f32>,
}

impl GroqConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            temperature: None,
            max_completion_tokens: None,
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

    pub fn with_max_completion_tokens(mut self, v: u32) -> Self {
        self.max_completion_tokens = Some(v);
        self
    }

    pub fn with_top_p(mut self, v: f32) -> Self {
        self.top_p = Some(v);
        self
    }

    pub fn from_env() -> Result<Self, GroqError> {
        let api_key =
            std::env::var("GROQ_API_KEY").map_err(|_| GroqError::MissingEnv("GROQ_API_KEY"))?;
        let model = std::env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.1-8b-instant".into());

        let mut config = Self::new(api_key, model);

        if let Ok(url) = std::env::var("GROQ_BASE_URL") {
            config = config.with_base_url(url);
        }

        Ok(config)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GroqRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct GroqProvider {
    api_key: String,
    base_url: String,
    request_config: GroqRequestConfig,
}

impl From<GroqConfig> for GroqProvider {
    fn from(config: GroqConfig) -> Self {
        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            request_config: GroqRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_completion_tokens: config.max_completion_tokens,
                top_p: config.top_p,
            },
        }
    }
}

impl CompletionsProvider for GroqProvider {
    type Config = GroqRequestConfig;

    fn provider_name(&self) -> &str {
        "Groq"
    }
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }
    fn config(&self) -> &GroqRequestConfig {
        &self.request_config
    }

    fn preprocess_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
            .bearer_auth(&self.api_key)
            .header("User-Agent", "agentkit-provider-groq/0.1.0")
    }
}

#[derive(Clone)]
pub struct GroqAdapter(CompletionsAdapter<GroqProvider>);

pub type GroqSession = CompletionsSession<GroqProvider>;
pub type GroqTurn = CompletionsTurn;

impl GroqAdapter {
    pub fn new(config: GroqConfig) -> Result<Self, GroqError> {
        let provider = GroqProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }
}

#[async_trait]
impl ModelAdapter for GroqAdapter {
    type Session = GroqSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

#[derive(Debug, Error)]
pub enum GroqError {
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    #[error(transparent)]
    Completions(#[from] CompletionsError),
}
