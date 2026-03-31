//! OpenAI model adapter for the agentkit agent loop.
//!
//! Connects to the [OpenAI](https://platform.openai.com) chat completions API.
//!
//! ```rust,ignore
//! use agentkit_provider_openai::{OpenAIAdapter, OpenAIConfig};
//!
//! let adapter = OpenAIAdapter::new(OpenAIConfig::from_env()?)?;
//! let agent = Agent::builder().model(adapter).build()?;
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_core::{MetadataMap, Usage};
use agentkit_loop::{
    LoopError, ModelAdapter, PromptCacheMode, PromptCacheRetention, PromptCacheStrategy,
    SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

/// Configuration for connecting to the OpenAI API.
#[derive(Clone, Debug)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub temperature: Option<f32>,
    pub max_completion_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl OpenAIConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            temperature: None,
            max_completion_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
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

    pub fn with_frequency_penalty(mut self, v: f32) -> Self {
        self.frequency_penalty = Some(v);
        self
    }

    pub fn with_presence_penalty(mut self, v: f32) -> Self {
        self.presence_penalty = Some(v);
        self
    }

    pub fn from_env() -> Result<Self, OpenAIError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| OpenAIError::MissingEnv("OPENAI_API_KEY"))?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".into());

        let mut config = Self::new(api_key, model);

        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            config = config.with_base_url(url);
        }

        Ok(config)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OpenAIRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: String,
    request_config: OpenAIRequestConfig,
}

impl From<OpenAIConfig> for OpenAIProvider {
    fn from(config: OpenAIConfig) -> Self {
        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            request_config: OpenAIRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_completion_tokens: config.max_completion_tokens,
                top_p: config.top_p,
                frequency_penalty: config.frequency_penalty,
                presence_penalty: config.presence_penalty,
            },
        }
    }
}

impl CompletionsProvider for OpenAIProvider {
    type Config = OpenAIRequestConfig;

    fn provider_name(&self) -> &str {
        "OpenAI"
    }
    fn endpoint_url(&self) -> &str {
        &self.base_url
    }
    fn config(&self) -> &OpenAIRequestConfig {
        &self.request_config
    }

    fn preprocess_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder
            .bearer_auth(&self.api_key)
            .header("User-Agent", "agentkit-provider-openai/0.1.0")
    }

    fn apply_prompt_cache(
        &self,
        body: &mut serde_json::Map<String, Value>,
        request: &TurnRequest,
    ) -> Result<(), LoopError> {
        let Some(cache) = &request.cache else {
            return Ok(());
        };
        if matches!(cache.mode, PromptCacheMode::Disabled) {
            return Ok(());
        }

        if let Some(key) = &cache.key {
            body.insert("prompt_cache_key".into(), Value::String(key.clone()));
        }

        if let Some(retention) = cache.retention {
            let value = match retention {
                PromptCacheRetention::Default | PromptCacheRetention::Short => "in_memory",
                PromptCacheRetention::Extended => "24h",
            };
            body.insert("prompt_cache_retention".into(), Value::String(value.into()));
        }

        if matches!(cache.strategy, PromptCacheStrategy::Explicit { .. })
            && matches!(cache.mode, PromptCacheMode::Required)
        {
            return Err(LoopError::Provider(
                "OpenAI chat completions does not support explicit cache breakpoints".into(),
            ));
        }

        Ok(())
    }

    fn postprocess_response(
        &self,
        _usage: &mut Option<Usage>,
        metadata: &mut MetadataMap,
        raw_response: &Value,
    ) {
        if let Some(model) = raw_response.get("model").and_then(Value::as_str) {
            metadata.insert("openai.model".into(), Value::String(model.into()));
        }
        if let Some(fingerprint) = raw_response
            .get("system_fingerprint")
            .and_then(Value::as_str)
        {
            metadata.insert(
                "openai.system_fingerprint".into(),
                Value::String(fingerprint.into()),
            );
        }
    }
}

#[derive(Clone)]
pub struct OpenAIAdapter(CompletionsAdapter<OpenAIProvider>);

pub type OpenAISession = CompletionsSession<OpenAIProvider>;
pub type OpenAITurn = CompletionsTurn;

impl OpenAIAdapter {
    pub fn new(config: OpenAIConfig) -> Result<Self, OpenAIError> {
        let provider = OpenAIProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }
}

#[async_trait]
impl ModelAdapter for OpenAIAdapter {
    type Session = OpenAISession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

#[derive(Debug, Error)]
pub enum OpenAIError {
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    #[error(transparent)]
    Completions(#[from] CompletionsError),
}

#[cfg(test)]
mod tests {
    use agentkit_core::{MetadataMap, SessionId, TurnId};

    use super::*;

    fn empty_turn_request(cache: Option<agentkit_loop::PromptCacheRequest>) -> TurnRequest {
        TurnRequest {
            session_id: SessionId::new("session"),
            turn_id: TurnId::new("turn-1"),
            transcript: Vec::new(),
            available_tools: Vec::new(),
            cache,
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn openai_maps_automatic_cache_request() {
        let provider = OpenAIProvider::from(OpenAIConfig::new("sk-test", "gpt-5.1"));
        let mut body = serde_json::Map::new();

        provider
            .apply_prompt_cache(
                &mut body,
                &empty_turn_request(Some(
                    agentkit_loop::PromptCacheRequest::best_effort(PromptCacheStrategy::Automatic)
                        .with_key("cache-key")
                        .with_retention(PromptCacheRetention::Extended),
                )),
            )
            .unwrap();

        assert_eq!(
            body.get("prompt_cache_key"),
            Some(&Value::String("cache-key".into()))
        );
        assert_eq!(
            body.get("prompt_cache_retention"),
            Some(&Value::String("24h".into()))
        );
    }

    #[test]
    fn openai_rejects_required_explicit_breakpoints() {
        let provider = OpenAIProvider::from(OpenAIConfig::new("sk-test", "gpt-5.1"));
        let mut body = serde_json::Map::new();

        let error = provider
            .apply_prompt_cache(
                &mut body,
                &empty_turn_request(Some(agentkit_loop::PromptCacheRequest::required(
                    PromptCacheStrategy::Explicit {
                        breakpoints: vec![
                            agentkit_loop::PromptCacheBreakpoint::TranscriptItemEnd { index: 0 },
                        ],
                    },
                ))),
            )
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not support explicit cache breakpoints")
        );
    }
}
