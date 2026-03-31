//! Ollama model adapter for the agentkit agent loop.
//!
//! Connects to a local [Ollama](https://ollama.ai) instance via its
//! OpenAI-compatible chat completions endpoint.
//!
//! ```rust,ignore
//! use agentkit_provider_ollama::{OllamaAdapter, OllamaConfig};
//!
//! let adapter = OllamaAdapter::new(OllamaConfig::new("llama3.1:8b"))?;
//! let agent = Agent::builder().model(adapter).build()?;
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "http://localhost:11434/v1/chat/completions";

/// Configuration for connecting to a local Ollama instance.
#[derive(Clone, Debug)]
pub struct OllamaConfig {
    pub model: String,
    pub base_url: String,
    pub temperature: Option<f32>,
    pub num_predict: Option<u32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
}

impl OllamaConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            temperature: None,
            num_predict: None,
            top_k: None,
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

    pub fn with_num_predict(mut self, v: u32) -> Self {
        self.num_predict = Some(v);
        self
    }

    pub fn with_top_k(mut self, v: u32) -> Self {
        self.top_k = Some(v);
        self
    }

    pub fn with_top_p(mut self, v: f32) -> Self {
        self.top_p = Some(v);
        self
    }

    pub fn from_env() -> Result<Self, OllamaError> {
        let model =
            std::env::var("OLLAMA_MODEL").map_err(|_| OllamaError::MissingEnv("OLLAMA_MODEL"))?;

        let mut config = Self::new(model);

        if let Ok(url) = std::env::var("OLLAMA_BASE_URL") {
            config = config.with_base_url(url);
        }

        Ok(config)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OllamaRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct OllamaProvider {
    base_url: String,
    request_config: OllamaRequestConfig,
}

impl From<OllamaConfig> for OllamaProvider {
    fn from(config: OllamaConfig) -> Self {
        Self {
            base_url: config.base_url,
            request_config: OllamaRequestConfig {
                model: config.model,
                temperature: config.temperature,
                num_predict: config.num_predict,
                top_k: config.top_k,
                top_p: config.top_p,
            },
        }
    }
}

impl CompletionsProvider for OllamaProvider {
    type Config = OllamaRequestConfig;

    fn provider_name(&self) -> &str {
        "Ollama"
    }
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }
    fn config(&self) -> &OllamaRequestConfig {
        &self.request_config
    }

    fn preprocess_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.header("User-Agent", "agentkit-provider-ollama/0.1.0")
    }
}

#[derive(Clone)]
pub struct OllamaAdapter(CompletionsAdapter<OllamaProvider>);

pub type OllamaSession = CompletionsSession<OllamaProvider>;
pub type OllamaTurn = CompletionsTurn;

impl OllamaAdapter {
    pub fn new(config: OllamaConfig) -> Result<Self, OllamaError> {
        let provider = OllamaProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }
}

#[async_trait]
impl ModelAdapter for OllamaAdapter {
    type Session = OllamaSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    #[error(transparent)]
    Completions(#[from] CompletionsError),
}
