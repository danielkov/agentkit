//! Groq model adapter for the agentkit agent loop.
//!
//! This crate provides [`GroqAdapter`] and [`GroqConfig`] for connecting
//! the agent loop to the [Groq](https://groq.com) chat completions API,
//! which serves open-source models on custom LPU hardware.
//! It is built on the generic [`agentkit_adapter_completions`] crate.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use agentkit_loop::{Agent, SessionConfig};
//! use agentkit_provider_groq::{GroqAdapter, GroqConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = GroqConfig::from_env()?;
//!     let adapter = GroqAdapter::new(config)?;
//!
//!     let agent = Agent::builder()
//!         .model(adapter)
//!         .build()?;
//!
//!     let mut driver = agent
//!         .start(SessionConfig::new("demo"))
//!         .await?;
//!     Ok(())
//! }
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_http::{Authentication, AuthenticationProvider, ResilienceConfig};
use agentkit_loop::{LoopError, ModelAdapter, SessionConfig};
use async_trait::async_trait;
use serde::Serialize;
use thiserror::Error;

const DEFAULT_ENDPOINT: &str = "https://api.groq.com/openai/v1/chat/completions";

/// Configuration for connecting to the Groq API.
///
/// Build one with [`GroqConfig::new`] for explicit values, or
/// [`GroqConfig::from_env`] to read from environment variables.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_provider_groq::GroqConfig;
///
/// let config = GroqConfig::new("gsk_...", "llama-3.3-70b-versatile")
///     .with_temperature(0.0)
///     .with_max_completion_tokens(4096);
/// ```
#[derive(Clone)]
pub struct GroqConfig {
    /// Authentication used for API requests. String values become bearer authentication.
    pub authentication: Authentication,
    /// Model identifier, e.g. `"llama-3.3-70b-versatile"` or `"llama-3.1-8b-instant"`.
    pub model: String,
    /// Chat completions endpoint URL. Defaults to the Groq production URL.
    pub base_url: String,
    /// Optional retry and timeout policy. `None` preserves single-attempt behavior.
    pub resilience: Option<ResilienceConfig>,
    /// Sampling temperature (0.0 = deterministic, higher = more creative).
    pub temperature: Option<f32>,
    /// Maximum number of completion tokens the model may generate.
    pub max_completion_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Whether the model is allowed to emit multiple tool calls in a
    /// single turn. Omitted from the request when `None` so Groq's
    /// per-model default applies.
    pub parallel_tool_calls: Option<bool>,
    /// Request SSE streaming responses. Defaults to `true`.
    pub streaming: bool,
}

impl std::fmt::Debug for GroqConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroqConfig")
            .field("authentication", &"<redacted>")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("resilience", &self.resilience)
            .field("temperature", &self.temperature)
            .field("max_completion_tokens", &self.max_completion_tokens)
            .field("top_p", &self.top_p)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("streaming", &self.streaming)
            .finish()
    }
}

impl GroqConfig {
    /// Creates a new configuration with the given authentication and model identifier.
    pub fn new(authentication: impl Into<Authentication>, model: impl Into<String>) -> Self {
        Self {
            authentication: authentication.into(),
            model: model.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            resilience: None,
            temperature: None,
            max_completion_tokens: None,
            top_p: None,
            parallel_tool_calls: None,
            streaming: true,
        }
    }

    /// Replaces request authentication. String values become bearer authentication.
    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        self.authentication = authentication.into();
        self
    }

    /// Compatibility builder for bearer API-key authentication.
    pub fn with_api_key(self, api_key: impl Into<Authentication>) -> Self {
        self.with_authentication(api_key)
    }

    /// Uses a custom refresh-capable authentication provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(mut self, provider: P) -> Self {
        self.authentication = Authentication::new(provider);
        self
    }

    /// Enables request retries and timeouts.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        self.resilience = Some(resilience);
        self
    }

    /// Overrides the default chat completions endpoint URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the sampling temperature (0.0 for deterministic output).
    pub fn with_temperature(mut self, v: f32) -> Self {
        self.temperature = Some(v);
        self
    }

    /// Sets the maximum number of tokens the model may generate per turn.
    pub fn with_max_completion_tokens(mut self, v: u32) -> Self {
        self.max_completion_tokens = Some(v);
        self
    }

    /// Sets the nucleus sampling parameter.
    pub fn with_top_p(mut self, v: f32) -> Self {
        self.top_p = Some(v);
        self
    }

    /// Sets whether the model may emit multiple tool calls in a single turn.
    pub fn with_parallel_tool_calls(mut self, flag: bool) -> Self {
        self.parallel_tool_calls = Some(flag);
        self
    }

    /// Toggles SSE streaming of model responses. Default: true.
    pub fn with_streaming(mut self, flag: bool) -> Self {
        self.streaming = flag;
        self
    }

    /// Builds a configuration from environment variables.
    ///
    /// | Variable | Required | Default |
    /// |---|---|---|
    /// | `GROQ_API_KEY` | yes | -- |
    /// | `GROQ_MODEL` | no | `llama-3.1-8b-instant` |
    /// | `GROQ_BASE_URL` | no | `https://api.groq.com/openai/v1/chat/completions` |
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

/// Request parameters serialized into the Groq request body.
#[derive(Clone, Debug, Serialize)]
pub struct GroqRequestConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// The Groq provider, implementing [`CompletionsProvider`].
#[derive(Clone, Debug)]
pub struct GroqProvider {
    authentication: Authentication,
    resilience: Option<ResilienceConfig>,
    base_url: String,
    streaming: bool,
    request_config: GroqRequestConfig,
}

impl From<GroqConfig> for GroqProvider {
    fn from(config: GroqConfig) -> Self {
        Self {
            authentication: config.authentication,
            resilience: config.resilience,
            base_url: config.base_url,
            streaming: config.streaming,
            request_config: GroqRequestConfig {
                model: config.model,
                temperature: config.temperature,
                max_completion_tokens: config.max_completion_tokens,
                top_p: config.top_p,
                parallel_tool_calls: config.parallel_tool_calls,
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

    fn preprocess_request(
        &self,
        builder: agentkit_http::HttpRequestBuilder,
    ) -> agentkit_http::HttpRequestBuilder {
        builder.header(
            "User-Agent",
            concat!("agentkit-provider-groq/", env!("CARGO_PKG_VERSION")),
        )
    }

    fn authentication(&self) -> Option<Authentication> {
        Some(self.authentication.clone())
    }

    fn resilience_config(&self) -> Option<ResilienceConfig> {
        self.resilience.clone()
    }

    fn streaming(&self) -> bool {
        self.streaming
    }
}

/// Model adapter that connects the agentkit agent loop to Groq.
///
/// # Example
///
/// ```rust,no_run
/// use agentkit_loop::Agent;
/// use agentkit_provider_groq::{GroqAdapter, GroqConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let adapter = GroqAdapter::new(GroqConfig::from_env()?)?;
///
/// let agent = Agent::builder()
///     .model(adapter)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct GroqAdapter(CompletionsAdapter<GroqProvider>);

/// An active session with the Groq API.
pub type GroqSession = CompletionsSession<GroqProvider>;

/// A completed turn from the Groq API.
pub type GroqTurn = CompletionsTurn;

impl GroqAdapter {
    /// Creates a new adapter from the given configuration.
    pub fn new(config: GroqConfig) -> Result<Self, GroqError> {
        let provider = GroqProvider::from(config);
        Ok(Self(CompletionsAdapter::new(provider)?))
    }

    /// Overrides the default API-key authentication.
    pub fn with_authentication(self, authentication: impl Into<Authentication>) -> Self {
        Self(self.0.with_authentication(authentication))
    }

    /// Overrides authentication with a custom refresh-capable provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(self, provider: P) -> Self {
        self.with_authentication(Authentication::new(provider))
    }

    /// Enables retry and timeout behavior.
    pub fn with_resilience(self, resilience: ResilienceConfig) -> Self {
        Self(self.0.with_resilience(resilience))
    }
}

#[async_trait]
impl ModelAdapter for GroqAdapter {
    type Session = GroqSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        self.0.start_session(config).await
    }
}

/// Errors produced by the Groq adapter.
#[derive(Debug, Error)]
pub enum GroqError {
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
    fn config_authentication_and_resilience_reach_provider() {
        let provider = GroqProvider::from(
            GroqConfig::new("secret", "model").with_resilience(ResilienceConfig::default()),
        );
        assert!(provider.authentication().is_some());
        assert!(provider.resilience_config().is_some());
    }

    #[test]
    fn config_debug_redacts_authentication() {
        let debug = format!("{:?}", GroqConfig::new("groq-secret", "debug-model"));
        assert!(!debug.contains("groq-secret"));
        assert!(debug.contains("<redacted>"));
        assert!(debug.contains("debug-model"));
    }
}
