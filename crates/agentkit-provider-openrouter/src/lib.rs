//! OpenRouter model adapter for the agentkit agent loop.
//!
//! This crate provides [`OpenRouterAdapter`] and [`OpenRouterConfig`] for
//! connecting the agent loop to any model available through the
//! [OpenRouter](https://openrouter.ai) API. It is built on the generic
//! [`agentkit_adapter_completions`] crate.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use agentkit_core::SessionId;
//! use agentkit_loop::{Agent, SessionConfig};
//! use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = OpenRouterConfig::from_env()?;
//!     let adapter = OpenRouterAdapter::new(config)?;
//!
//!     let agent = Agent::builder()
//!         .model(adapter)
//!         .build()?;
//!
//!     let mut driver = agent
//!         .start(SessionConfig {
//!             session_id: SessionId::new("demo"),
//!             metadata: Default::default(),
//!         })
//!         .await?;
//!     Ok(())
//! }
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_core::{CostUsage, MetadataMap, Usage};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Configuration for connecting to the OpenRouter API.
///
/// Holds credentials, model selection, and optional request parameters.
/// Build one with [`OpenRouterConfig::new`] for explicit values, or
/// [`OpenRouterConfig::from_env`] to read from environment variables.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_provider_openrouter::OpenRouterConfig;
///
/// let config = OpenRouterConfig::new("sk-or-v1-...", "anthropic/claude-sonnet-4")
///     .with_temperature(0.0)
///     .with_max_completion_tokens(4096)
///     .with_app_name("my-agent");
/// ```
#[derive(Clone, Debug)]
pub struct OpenRouterConfig {
    /// OpenRouter API key (starts with `sk-or-`).
    pub api_key: String,
    /// Model identifier, e.g. `"anthropic/claude-sonnet-4"` or `"openrouter/auto"`.
    pub model: String,
    /// Chat completions endpoint URL. Defaults to the OpenRouter production URL.
    pub base_url: String,
    /// Optional application name sent as the `X-Title` header.
    pub app_name: Option<String>,
    /// Optional site URL sent as the `HTTP-Referer` header.
    pub site_url: Option<String>,
    /// Maximum number of completion tokens the model may generate.
    pub max_completion_tokens: Option<u32>,
    /// Sampling temperature (0.0 = deterministic, higher = more creative).
    pub temperature: Option<f32>,
    /// Arbitrary extra fields merged into the request body.
    pub extra_body: MetadataMap,
}

impl OpenRouterConfig {
    /// Creates a new configuration with the given API key and model identifier.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.into(),
            app_name: None,
            site_url: None,
            max_completion_tokens: None,
            temperature: None,
            extra_body: MetadataMap::new(),
        }
    }

    /// Overrides the default chat completions endpoint URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the application name sent via the `X-Title` HTTP header.
    pub fn with_app_name(mut self, app_name: impl Into<String>) -> Self {
        self.app_name = Some(app_name.into());
        self
    }

    /// Sets the site URL sent via the `HTTP-Referer` header.
    pub fn with_site_url(mut self, site_url: impl Into<String>) -> Self {
        self.site_url = Some(site_url.into());
        self
    }

    /// Sets the maximum number of tokens the model may generate per turn.
    pub fn with_max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    /// Sets the sampling temperature (0.0 for deterministic output).
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Inserts an arbitrary key-value pair into the request body.
    pub fn with_extra_body_value(
        mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) -> Self {
        self.extra_body.insert(key.into(), value.into());
        self
    }

    /// Builds a configuration from environment variables.
    ///
    /// Reads the following variables:
    ///
    /// | Variable | Required | Default |
    /// |---|---|---|
    /// | `OPENROUTER_API_KEY` | yes | -- |
    /// | `OPENROUTER_MODEL` | no | `openrouter/auto` |
    /// | `OPENROUTER_BASE_URL` | no | production URL |
    /// | `OPENROUTER_APP_NAME` | no | -- |
    /// | `OPENROUTER_SITE_URL` | no | -- |
    /// | `OPENROUTER_MAX_COMPLETION_TOKENS` | no | -- |
    /// | `OPENROUTER_TEMPERATURE` | no | -- |
    pub fn from_env() -> Result<Self, OpenRouterError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| OpenRouterError::MissingEnv("OPENROUTER_API_KEY"))?;
        let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openrouter/auto".into());

        let mut config = Self::new(api_key, model);

        if let Ok(app_name) = std::env::var("OPENROUTER_APP_NAME") {
            config = config.with_app_name(app_name);
        }
        if let Ok(site_url) = std::env::var("OPENROUTER_SITE_URL") {
            config = config.with_site_url(site_url);
        }
        if let Ok(base_url) = std::env::var("OPENROUTER_BASE_URL") {
            config = config.with_base_url(base_url);
        }
        if let Ok(value) = std::env::var("OPENROUTER_MAX_COMPLETION_TOKENS") {
            let parsed = value.parse::<u32>().map_err(|_| {
                OpenRouterError::InvalidConfig(format!("invalid max tokens: {value}"))
            })?;
            config = config.with_max_completion_tokens(parsed);
        }
        if let Ok(value) = std::env::var("OPENROUTER_TEMPERATURE") {
            let parsed = value.parse::<f32>().map_err(|_| {
                OpenRouterError::InvalidConfig(format!("invalid temperature: {value}"))
            })?;
            config = config.with_temperature(parsed);
        }

        Ok(config)
    }
}

// --- Request config (serialised into the request body) ---

/// Strongly-typed request parameters for OpenRouter.
#[derive(Clone, Debug, Serialize)]
pub struct OpenRouterRequestConfig {
    /// Model identifier.
    pub model: String,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Extra fields merged into the body.
    #[serde(flatten)]
    pub extra: MetadataMap,
}

// --- Provider implementation ---

/// The OpenRouter provider, implementing [`CompletionsProvider`].
#[derive(Clone, Debug)]
pub struct OpenRouterProvider {
    api_key: String,
    base_url: String,
    app_name: Option<String>,
    site_url: Option<String>,
    request_config: OpenRouterRequestConfig,
}

impl From<OpenRouterConfig> for OpenRouterProvider {
    fn from(config: OpenRouterConfig) -> Self {
        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            app_name: config.app_name,
            site_url: config.site_url,
            request_config: OpenRouterRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_completion_tokens: config.max_completion_tokens,
                extra: config.extra_body,
            },
        }
    }
}

impl CompletionsProvider for OpenRouterProvider {
    type Config = OpenRouterRequestConfig;

    fn provider_name(&self) -> &str {
        "OpenRouter"
    }

    fn endpoint_url(&self) -> &str {
        &self.base_url
    }

    fn config(&self) -> &OpenRouterRequestConfig {
        &self.request_config
    }

    fn preprocess_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder
            .bearer_auth(&self.api_key)
            .header("User-Agent", "agentkit-provider-openrouter/0.1.0");
        if let Some(app_name) = &self.app_name {
            builder = builder.header("X-Title", app_name);
        }
        if let Some(site_url) = &self.site_url {
            builder = builder.header("HTTP-Referer", site_url);
        }
        builder
    }

    fn preprocess_response(
        &self,
        _status: reqwest::StatusCode,
        body: &str,
    ) -> Result<(), LoopError> {
        #[derive(serde::Deserialize)]
        struct ErrResp {
            error: ErrBody,
        }
        #[derive(serde::Deserialize)]
        struct ErrBody {
            message: String,
            code: Value,
        }

        if let Ok(e) = serde_json::from_str::<ErrResp>(body) {
            return Err(LoopError::Provider(format!(
                "OpenRouter returned error (code {}): {}",
                e.error.code, e.error.message
            )));
        }
        Ok(())
    }

    fn postprocess_response(
        &self,
        usage: &mut Option<Usage>,
        metadata: &mut MetadataMap,
        raw_response: &Value,
    ) {
        if let Some(cost) = raw_response.pointer("/usage/cost").and_then(Value::as_f64) {
            if let Some(usage) = usage {
                usage.cost = Some(CostUsage {
                    amount: cost,
                    currency: "USD".into(),
                    provider_amount: None,
                });
            }
        }
        if let Some(model) = raw_response.get("model").and_then(Value::as_str) {
            metadata.insert("openrouter.model".into(), Value::String(model.into()));
        }
        if let Some(refusal) = raw_response
            .pointer("/choices/0/message/refusal")
            .and_then(Value::as_str)
        {
            metadata.insert("openrouter.refusal".into(), Value::String(refusal.into()));
        }
    }
}

// --- Adapter newtype (preserves the existing public API) ---

/// Model adapter that connects the agentkit agent loop to OpenRouter.
///
/// This is a thin wrapper around [`CompletionsAdapter`] parameterised with
/// [`OpenRouterProvider`]. It preserves the same public API as the previous
/// standalone implementation.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_loop::Agent;
/// use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let adapter = OpenRouterAdapter::new(
///     OpenRouterConfig::from_env()?
///         .with_temperature(0.0)
///         .with_max_completion_tokens(512),
/// )?;
///
/// let agent = Agent::builder()
///     .model(adapter)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenRouterAdapter(CompletionsAdapter<OpenRouterProvider>);

/// An active session with the OpenRouter API.
pub type OpenRouterSession = CompletionsSession<OpenRouterProvider>;

/// A completed turn from the OpenRouter API.
pub type OpenRouterTurn = CompletionsTurn;

impl OpenRouterAdapter {
    /// Creates a new adapter from the given configuration.
    pub fn new(config: OpenRouterConfig) -> Result<Self, OpenRouterError> {
        let provider = OpenRouterProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }
}

#[async_trait]
impl ModelAdapter for OpenRouterAdapter {
    type Session = OpenRouterSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

// --- Error type ---

/// Errors produced by the OpenRouter adapter.
#[derive(Debug, Error)]
pub enum OpenRouterError {
    /// A required environment variable is not set.
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    /// A configuration value could not be parsed or is otherwise invalid.
    #[error("invalid OpenRouter configuration: {0}")]
    InvalidConfig(String),

    /// An error from the generic completions adapter.
    #[error(transparent)]
    Completions(#[from] CompletionsError),
}
