use std::borrow::Cow;
use std::sync::Arc;

use agentkit_http::{Authentication, AuthenticationProvider, HeaderName, ResilienceConfig};
use serde_json::{Value, json};

use crate::error::AnthropicError;
use crate::server_tool::ServerToolHandle;

/// Default Messages API endpoint.
pub const DEFAULT_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
/// Default `anthropic-version` header.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic API-key authentication for the `x-api-key` header.
#[derive(Clone, Debug)]
pub struct AnthropicApiKey(Authentication);

impl AnthropicApiKey {
    /// Wraps an Anthropic API key without retaining it as a plain config string.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(Authentication::header(
            HeaderName::from_static("x-api-key"),
            api_key,
        ))
    }
}

impl From<String> for AnthropicApiKey {
    fn from(api_key: String) -> Self {
        Self::new(api_key)
    }
}

impl From<&str> for AnthropicApiKey {
    fn from(api_key: &str) -> Self {
        Self::new(api_key)
    }
}

impl From<&String> for AnthropicApiKey {
    fn from(api_key: &String) -> Self {
        Self::new(api_key)
    }
}

impl From<Box<str>> for AnthropicApiKey {
    fn from(api_key: Box<str>) -> Self {
        Self::new(api_key)
    }
}

impl From<Arc<str>> for AnthropicApiKey {
    fn from(api_key: Arc<str>) -> Self {
        Self::new(api_key.as_ref())
    }
}

impl<'a> From<Cow<'a, str>> for AnthropicApiKey {
    fn from(api_key: Cow<'a, str>) -> Self {
        Self::new(api_key)
    }
}

impl From<AnthropicApiKey> for Authentication {
    fn from(api_key: AnthropicApiKey) -> Self {
        api_key.0
    }
}

/// Anthropic bearer authentication for the `Authorization` header.
#[derive(Clone, Debug)]
pub struct AnthropicAuthToken(Authentication);

impl AnthropicAuthToken {
    /// Wraps an Anthropic auth token without retaining it as a plain config string.
    pub fn new(auth_token: impl Into<String>) -> Self {
        Self(Authentication::bearer(auth_token))
    }
}

impl From<String> for AnthropicAuthToken {
    fn from(auth_token: String) -> Self {
        Self::new(auth_token)
    }
}

impl From<&str> for AnthropicAuthToken {
    fn from(auth_token: &str) -> Self {
        Self::new(auth_token)
    }
}

impl From<&String> for AnthropicAuthToken {
    fn from(auth_token: &String) -> Self {
        Self::new(auth_token)
    }
}

impl From<Box<str>> for AnthropicAuthToken {
    fn from(auth_token: Box<str>) -> Self {
        Self::new(auth_token)
    }
}

impl From<Arc<str>> for AnthropicAuthToken {
    fn from(auth_token: Arc<str>) -> Self {
        Self::new(auth_token.as_ref())
    }
}

impl<'a> From<Cow<'a, str>> for AnthropicAuthToken {
    fn from(auth_token: Cow<'a, str>) -> Self {
        Self::new(auth_token)
    }
}

impl From<AnthropicAuthToken> for Authentication {
    fn from(auth_token: AnthropicAuthToken) -> Self {
        auth_token.0
    }
}

/// Extended thinking configuration.
#[derive(Clone, Debug)]
pub enum ThinkingConfig {
    /// Disable extended thinking explicitly.
    Disabled,
    /// Enable extended thinking with a fixed token budget.
    Enabled {
        /// Upper bound on thinking tokens for this turn.
        budget_tokens: u32,
    },
    /// Let the model decide how much to think (supported models only).
    Adaptive,
}

impl ThinkingConfig {
    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::Disabled => json!({ "type": "disabled" }),
            Self::Enabled { budget_tokens } => {
                json!({ "type": "enabled", "budget_tokens": budget_tokens })
            }
            Self::Adaptive => json!({ "type": "adaptive" }),
        }
    }
}

/// Priority/standard routing hint.
#[derive(Clone, Copy, Debug)]
pub enum ServiceTier {
    /// Use priority capacity if available, fall back to standard.
    Auto,
    /// Reject the request rather than fall back to standard capacity.
    StandardOnly,
}

impl ServiceTier {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::StandardOnly => "standard_only",
        }
    }
}

/// Constraint applied to the model's tool-choice behaviour.
#[derive(Clone, Debug)]
pub enum ToolChoice {
    /// Model decides freely whether to call a tool.
    Auto,
    /// Model MUST call exactly one tool.
    Any,
    /// Model MUST NOT call any tool.
    None,
    /// Model MUST call this specific tool.
    Tool {
        /// Name of the tool to force.
        name: String,
    },
}

impl ToolChoice {
    pub(crate) fn to_json(&self, disable_parallel: Option<bool>) -> Value {
        let mut obj = match self {
            Self::Auto => json!({ "type": "auto" }),
            Self::Any => json!({ "type": "any" }),
            Self::None => json!({ "type": "none" }),
            Self::Tool { name } => json!({ "type": "tool", "name": name }),
        };
        if let Some(flag) = disable_parallel
            && let Some(obj) = obj.as_object_mut()
        {
            obj.insert("disable_parallel_tool_use".into(), Value::Bool(flag));
        }
        obj
    }
}

/// Structured output format constraint.
#[derive(Clone, Debug)]
pub enum OutputFormat {
    /// Constrain output to a JSON Schema.
    JsonSchema {
        /// The JSON Schema the model must satisfy.
        schema: Value,
    },
}

impl OutputFormat {
    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::JsonSchema { schema } => json!({
                "type": "json_schema",
                "schema": schema,
            }),
        }
    }
}

/// Relative reasoning effort hint.
#[derive(Clone, Copy, Debug)]
pub enum OutputEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl OutputEffort {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// MCP server descriptor passed through to Anthropic's `mcp_servers` array.
///
/// Stored as opaque `Value` so that new MCP-server fields land without requiring
/// a provider-crate release.
#[derive(Clone, Debug)]
pub struct AnthropicMcpServer(pub Value);

/// Configuration for connecting to the Anthropic Messages API.
///
/// Build one with either [`AnthropicConfig::new`] (API key) or
/// [`AnthropicConfig::with_auth_token`] (bearer token), or via
/// [`AnthropicConfig::from_env`]. `max_tokens` is a required constructor
/// argument — the Messages API rejects requests without it.
#[derive(Clone)]
pub struct AnthropicConfig {
    /// Authentication applied to each request.
    pub authentication: Authentication,

    /// Endpoint URL. Defaults to the Anthropic production endpoint.
    pub base_url: String,
    /// Value for the `anthropic-version` header.
    pub anthropic_version: String,
    /// Additional `anthropic-beta` flags to enable.
    pub anthropic_beta: Vec<String>,

    /// Model identifier, e.g. `"claude-opus-4-7"`.
    pub model: String,
    /// Maximum number of tokens the model may generate (required by the API).
    pub max_tokens: u32,

    /// Sampling temperature (0.0–1.0).
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Custom stop sequences.
    pub stop_sequences: Option<Vec<String>>,

    /// Extended thinking configuration.
    pub thinking: Option<ThinkingConfig>,

    /// Priority vs standard capacity routing.
    pub service_tier: Option<ServiceTier>,
    /// Value for `metadata.user_id` in requests.
    pub metadata_user_id: Option<String>,

    /// Forces, restricts, or disables tool-choice behaviour.
    pub tool_choice: Option<ToolChoice>,
    /// If set, overrides the API's default of allowing parallel tool use.
    ///
    /// `Some(true)` disables parallel tool use. Folded into the `tool_choice`
    /// object rather than set as a top-level field.
    pub disable_parallel_tool_use: Option<bool>,
    /// Anthropic-run server tools (web search, code execution, etc.).
    pub server_tools: Vec<ServerToolHandle>,
    /// Pre-existing container identifier for code-execution sessions.
    pub container: Option<String>,

    /// Structured output shape.
    pub output_format: Option<OutputFormat>,
    /// Reasoning effort hint for structured output.
    pub output_effort: Option<OutputEffort>,

    /// MCP servers passed through to Anthropic's request body verbatim.
    pub mcp_servers: Vec<AnthropicMcpServer>,

    /// Whether to request a streaming SSE response from the Messages API.
    ///
    /// Defaults to `true`. Streaming yields incremental `ModelTurnEvent`s as
    /// the model generates, enabling responsive UIs; when disabled the adapter
    /// buffers the full JSON response before emitting any events. Opt out via
    /// [`AnthropicConfig::with_streaming`] for debugging or when an upstream
    /// proxy doesn't forward SSE bodies.
    pub streaming: bool,
    /// Optional retry and timeout policy. `None` preserves single-attempt behavior.
    pub resilience: Option<ResilienceConfig>,
}

impl std::fmt::Debug for AnthropicConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicConfig")
            .field("authentication", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("anthropic_version", &self.anthropic_version)
            .field("anthropic_beta", &self.anthropic_beta)
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("stop_sequences", &self.stop_sequences)
            .field("thinking", &self.thinking)
            .field("service_tier", &self.service_tier)
            .field("tool_choice", &self.tool_choice)
            .field("disable_parallel_tool_use", &self.disable_parallel_tool_use)
            .field("output_format", &self.output_format)
            .field("output_effort", &self.output_effort)
            .field("streaming", &self.streaming)
            .field("resilience", &self.resilience)
            .finish_non_exhaustive()
    }
}

impl AnthropicConfig {
    /// Creates a new configuration using `x-api-key` authentication.
    pub fn new(
        api_key: impl Into<AnthropicApiKey>,
        model: impl Into<String>,
        max_tokens: u32,
    ) -> Result<Self, AnthropicError> {
        Self::from_authentication(api_key.into(), model, max_tokens)
    }

    /// Creates a new configuration using a bearer auth token.
    pub fn with_auth_token(
        auth_token: impl Into<AnthropicAuthToken>,
        model: impl Into<String>,
        max_tokens: u32,
    ) -> Result<Self, AnthropicError> {
        Self::from_authentication(auth_token.into(), model, max_tokens)
    }

    /// Creates a new configuration using arbitrary authentication.
    pub fn from_authentication(
        authentication: impl Into<Authentication>,
        model: impl Into<String>,
        max_tokens: u32,
    ) -> Result<Self, AnthropicError> {
        if max_tokens == 0 {
            return Err(AnthropicError::InvalidMaxTokens);
        }
        Ok(Self {
            authentication: authentication.into(),
            base_url: DEFAULT_ENDPOINT.into(),
            anthropic_version: DEFAULT_ANTHROPIC_VERSION.into(),
            anthropic_beta: Vec::new(),
            model: model.into(),
            max_tokens,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            thinking: None,
            service_tier: None,
            metadata_user_id: None,
            tool_choice: None,
            disable_parallel_tool_use: None,
            server_tools: Vec::new(),
            container: None,
            output_format: None,
            output_effort: None,
            mcp_servers: Vec::new(),
            streaming: true,
            resilience: None,
        })
    }

    /// Builds a configuration from environment variables.
    ///
    /// | Variable | Required | Default |
    /// |---|---|---|
    /// | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` | one | — |
    /// | `ANTHROPIC_MODEL` | yes | — |
    /// | `ANTHROPIC_MAX_TOKENS` | yes | — |
    /// | `ANTHROPIC_BASE_URL` | no | `https://api.anthropic.com/v1/messages` |
    /// | `ANTHROPIC_VERSION` | no | `2023-06-01` |
    /// | `ANTHROPIC_BETA` | no | comma-separated flag list |
    pub fn from_env() -> Result<Self, AnthropicError> {
        let model = std::env::var("ANTHROPIC_MODEL")
            .map_err(|_| AnthropicError::MissingEnv("ANTHROPIC_MODEL"))?;
        let max_tokens: u32 = std::env::var("ANTHROPIC_MAX_TOKENS")
            .map_err(|_| AnthropicError::MissingEnv("ANTHROPIC_MAX_TOKENS"))?
            .parse()
            .map_err(|_| AnthropicError::MissingEnv("ANTHROPIC_MAX_TOKENS"))?;

        let mut config = match (
            std::env::var("ANTHROPIC_AUTH_TOKEN").ok(),
            std::env::var("ANTHROPIC_API_KEY").ok(),
        ) {
            (Some(token), _) => Self::with_auth_token(token, model, max_tokens)?,
            (None, Some(key)) => Self::new(key, model, max_tokens)?,
            (None, None) => return Err(AnthropicError::MissingCredentials),
        };

        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            config = config.with_base_url(url);
        }
        if let Ok(ver) = std::env::var("ANTHROPIC_VERSION") {
            config = config.with_anthropic_version(ver);
        }
        if let Ok(betas) = std::env::var("ANTHROPIC_BETA") {
            for flag in betas.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                config = config.with_beta(flag.to_string());
            }
        }

        Ok(config)
    }

    // --- Builder methods ---

    /// Replaces the configured authentication.
    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        self.authentication = authentication.into();
        self
    }

    /// Uses a custom refresh-capable authentication provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(self, provider: P) -> Self {
        self.with_authentication(Authentication::new(provider))
    }

    /// Enables request and pre-visible-output retries and timeouts.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        self.resilience = Some(resilience);
        self
    }

    /// Overrides the endpoint URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Overrides the `anthropic-version` header value.
    pub fn with_anthropic_version(mut self, v: impl Into<String>) -> Self {
        self.anthropic_version = v.into();
        self
    }

    /// Adds a single `anthropic-beta` flag.
    pub fn with_beta(mut self, flag: impl Into<String>) -> Self {
        self.anthropic_beta.push(flag.into());
        self
    }

    /// Sets the sampling temperature (0.0 = deterministic).
    pub fn with_temperature(mut self, v: f32) -> Self {
        self.temperature = Some(v);
        self
    }

    /// Sets the nucleus sampling parameter.
    pub fn with_top_p(mut self, v: f32) -> Self {
        self.top_p = Some(v);
        self
    }

    /// Sets the top-k sampling parameter.
    pub fn with_top_k(mut self, v: u32) -> Self {
        self.top_k = Some(v);
        self
    }

    /// Replaces the stop-sequence list.
    pub fn with_stop_sequences(mut self, sequences: impl IntoIterator<Item = String>) -> Self {
        self.stop_sequences = Some(sequences.into_iter().collect());
        self
    }

    /// Sets the extended-thinking configuration.
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Sets the priority/standard routing hint.
    pub fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Sets the `metadata.user_id` value.
    pub fn with_metadata_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.metadata_user_id = Some(user_id.into());
        self
    }

    /// Sets the tool-choice constraint.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Disables parallel tool use (API default is to allow it).
    pub fn disable_parallel_tool_use(mut self, flag: bool) -> Self {
        self.disable_parallel_tool_use = Some(flag);
        self
    }

    /// Appends a server tool to the configuration.
    pub fn with_server_tool(mut self, tool: ServerToolHandle) -> Self {
        self.server_tools.push(tool);
        self
    }

    /// Sets the container identifier for code-execution sessions.
    pub fn with_container(mut self, id: impl Into<String>) -> Self {
        self.container = Some(id.into());
        self
    }

    /// Sets the structured output format.
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Sets the reasoning-effort hint.
    pub fn with_output_effort(mut self, effort: OutputEffort) -> Self {
        self.output_effort = Some(effort);
        self
    }

    /// Appends an MCP server descriptor.
    pub fn with_mcp_server(mut self, server: AnthropicMcpServer) -> Self {
        self.mcp_servers.push(server);
        self
    }

    /// Toggles SSE streaming of model responses.
    ///
    /// Streaming is on by default; pass `false` to fall back to the buffered
    /// non-streaming path (the request body will be sent with `stream: false`
    /// and the full response parsed once the Messages API returns).
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }
}
