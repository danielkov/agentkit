//! vLLM model adapter for the agentkit agent loop.
//!
//! Connects to a [vLLM](https://docs.vllm.ai) server via its
//! OpenAI-compatible chat completions endpoint.
//!
//! ```rust,ignore
//! use agentkit_provider_vllm::{VllmAdapter, VllmConfig};
//!
//! let adapter = VllmAdapter::new(VllmConfig::new("meta-llama/Llama-3.1-8B-Instruct"))?;
//! let agent = Agent::builder().model(adapter).build()?;
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "http://localhost:8000/v1/chat/completions";

/// Configuration for connecting to a vLLM server.
#[derive(Clone, Debug)]
pub struct VllmConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub max_completion_tokens: Option<u32>,
    pub top_p: Option<f32>,
}

impl VllmConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            api_key: None,
            temperature: None,
            max_completion_tokens: None,
            top_p: None,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
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

    pub fn from_env() -> Result<Self, VllmError> {
        let model = std::env::var("VLLM_MODEL").map_err(|_| VllmError::MissingEnv("VLLM_MODEL"))?;

        let mut config = Self::new(model);

        if let Ok(url) = std::env::var("VLLM_BASE_URL") {
            config = config.with_base_url(url);
        }
        if let Ok(key) = std::env::var("VLLM_API_KEY") {
            config = config.with_api_key(key);
        }

        Ok(config)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VllmRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct VllmProvider {
    base_url: String,
    api_key: Option<String>,
    request_config: VllmRequestConfig,
}

impl From<VllmConfig> for VllmProvider {
    fn from(config: VllmConfig) -> Self {
        Self {
            base_url: config.base_url,
            api_key: config.api_key,
            request_config: VllmRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_completion_tokens: config.max_completion_tokens,
                top_p: config.top_p,
            },
        }
    }
}

impl CompletionsProvider for VllmProvider {
    type Config = VllmRequestConfig;

    fn provider_name(&self) -> &str {
        "vLLM"
    }
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }
    fn config(&self) -> &VllmRequestConfig {
        &self.request_config
    }

    fn preprocess_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let builder = builder.header("User-Agent", "agentkit-provider-vllm/0.1.0");
        match &self.api_key {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }
}

#[derive(Clone)]
pub struct VllmAdapter(CompletionsAdapter<VllmProvider>);

pub type VllmSession = CompletionsSession<VllmProvider>;
pub type VllmTurn = CompletionsTurn;

impl VllmAdapter {
    pub fn new(config: VllmConfig) -> Result<Self, VllmError> {
        let provider = VllmProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }
}

#[async_trait]
impl ModelAdapter for VllmAdapter {
    type Session = VllmSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

#[derive(Debug, Error)]
pub enum VllmError {
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    #[error(transparent)]
    Completions(#[from] CompletionsError),
}
