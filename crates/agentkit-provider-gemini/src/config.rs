use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use agentkit_http::{
    Authentication, AuthenticationProvider, HeaderMap, HeaderName, HeaderValue, ResilienceConfig,
};
use serde_json::{Value, json};

use crate::error::GeminiError;

/// Default Gemini Developer API base URL (without the `/models/{model}:...` suffix).
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_ATTEMPT_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_WIRE_BYTES: usize = 64 * 1024 * 1024;

/// Gemini API-key authentication for the `x-goog-api-key` header.
///
/// The underlying key bytes are zeroized on drop and marked sensitive on HTTP
/// headers; keys are never placed into URL query strings.
#[derive(Clone, Debug)]
pub struct GeminiApiKey(Authentication);

impl GeminiApiKey {
    /// Wraps a Gemini API key without retaining it as a plain config string.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(Authentication::header(
            HeaderName::from_static("x-goog-api-key"),
            api_key,
        ))
    }
}

impl From<String> for GeminiApiKey {
    fn from(api_key: String) -> Self {
        Self::new(api_key)
    }
}

impl From<&str> for GeminiApiKey {
    fn from(api_key: &str) -> Self {
        Self::new(api_key)
    }
}

impl From<&String> for GeminiApiKey {
    fn from(api_key: &String) -> Self {
        Self::new(api_key)
    }
}

impl From<Box<str>> for GeminiApiKey {
    fn from(api_key: Box<str>) -> Self {
        Self::new(api_key)
    }
}

impl From<Arc<str>> for GeminiApiKey {
    fn from(api_key: Arc<str>) -> Self {
        Self::new(api_key.as_ref())
    }
}

impl<'a> From<Cow<'a, str>> for GeminiApiKey {
    fn from(api_key: Cow<'a, str>) -> Self {
        Self::new(api_key)
    }
}

impl From<GeminiApiKey> for Authentication {
    fn from(api_key: GeminiApiKey) -> Self {
        api_key.0
    }
}

/// Gemini bearer authentication for the `Authorization` header (e.g. OAuth2 / ADC / Vertex AI).
#[derive(Clone, Debug)]
pub struct GeminiAuthToken(Authentication);

impl GeminiAuthToken {
    /// Wraps a bearer token without retaining it as a plain config string.
    pub fn new(auth_token: impl Into<String>) -> Self {
        Self(Authentication::bearer(auth_token))
    }
}

impl From<String> for GeminiAuthToken {
    fn from(auth_token: String) -> Self {
        Self::new(auth_token)
    }
}

impl From<&str> for GeminiAuthToken {
    fn from(auth_token: &str) -> Self {
        Self::new(auth_token)
    }
}

impl From<&String> for GeminiAuthToken {
    fn from(auth_token: &String) -> Self {
        Self::new(auth_token)
    }
}

impl From<Box<str>> for GeminiAuthToken {
    fn from(auth_token: Box<str>) -> Self {
        Self::new(auth_token)
    }
}

impl From<Arc<str>> for GeminiAuthToken {
    fn from(auth_token: Arc<str>) -> Self {
        Self::new(auth_token.as_ref())
    }
}

impl<'a> From<Cow<'a, str>> for GeminiAuthToken {
    fn from(auth_token: Cow<'a, str>) -> Self {
        Self::new(auth_token)
    }
}

impl From<GeminiAuthToken> for Authentication {
    fn from(auth_token: GeminiAuthToken) -> Self {
        auth_token.0
    }
}

/// Relative reasoning effort level for Gemini 3+ thinking models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

impl ThinkingLevel {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "MINIMAL",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }
}

/// Thinking configuration serialized into `generationConfig.thinkingConfig`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingConfig {
    /// Fixed or dynamic token budget for Gemini 2.5 models (`0` disables, `-1` dynamic).
    Budget {
        /// Token budget for internal reasoning.
        budget_tokens: i32,
        /// Whether to include thought summaries in the model response.
        include_thoughts: bool,
    },
    /// Relative reasoning level for Gemini 3+ models.
    Level {
        /// Reasoning depth level.
        level: ThinkingLevel,
        /// Whether to include thought summaries in the model response.
        include_thoughts: bool,
    },
}

impl ThinkingConfig {
    /// Convenience constructor for Gemini 2.5 token-budget thinking.
    pub fn budget(budget_tokens: i32) -> Self {
        Self::Budget {
            budget_tokens,
            include_thoughts: true,
        }
    }

    /// Convenience constructor for Gemini 3+ thinking levels.
    pub fn level(level: ThinkingLevel) -> Self {
        Self::Level {
            level,
            include_thoughts: true,
        }
    }

    pub(crate) fn to_json(self) -> Value {
        match self {
            Self::Budget {
                budget_tokens,
                include_thoughts,
            } => json!({
                "thinkingBudget": budget_tokens,
                "includeThoughts": include_thoughts,
            }),
            Self::Level {
                level,
                include_thoughts,
            } => json!({
                "thinkingLevel": level.as_str(),
                "includeThoughts": include_thoughts,
            }),
        }
    }
}

/// Constraint applied to Gemini's function-calling behaviour (`toolConfig.functionCallingConfig`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    /// Model decides freely whether to call a function or reply with text.
    Auto,
    /// Model MUST call one of the provided functions.
    Any,
    /// Model MUST NOT call any function.
    None,
    /// Model MUST call one of the specified function names (`mode: "ANY"` + `allowedFunctionNames`).
    AllowedFunctions(Vec<String>),
}

impl ToolChoice {
    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::Auto => json!({
                "functionCallingConfig": { "mode": "AUTO" }
            }),
            Self::Any => json!({
                "functionCallingConfig": { "mode": "ANY" }
            }),
            Self::None => json!({
                "functionCallingConfig": { "mode": "NONE" }
            }),
            Self::AllowedFunctions(names) => json!({
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": names,
                }
            }),
        }
    }
}

/// Structured output format constraint serialized into `generationConfig`.
#[derive(Clone, Debug, PartialEq)]
pub enum OutputFormat {
    /// Request JSON output without an explicit schema (`responseMimeType: "application/json"`).
    Json,
    /// Constrain output to a standard JSON Schema (`responseMimeType` + `responseJsonSchema`).
    JsonSchema {
        /// Standard JSON Schema the model output must satisfy.
        schema: Value,
    },
}

/// Safety filter threshold override for a harm category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetySetting {
    /// Gemini harm category, e.g. `"HARM_CATEGORY_HARASSMENT"`.
    pub category: String,
    /// Gemini block threshold, e.g. `"BLOCK_ONLY_HIGH"`.
    pub threshold: String,
}

impl SafetySetting {
    pub fn new(category: impl Into<String>, threshold: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            threshold: threshold.into(),
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "category": self.category,
            "threshold": self.threshold,
        })
    }
}

/// Built-in Gemini server-side tools appended to the `tools` array.
#[derive(Clone, Debug, PartialEq)]
pub enum GeminiServerTool {
    /// Grounding with Google Search (`{ "googleSearch": {} }`).
    GoogleSearch,
    /// Server-side sandboxed Python code execution (`{ "codeExecution": {} }`).
    CodeExecution,
    /// Server-side URL context retrieval (`{ "urlContext": {} }`).
    UrlContext,
    /// Arbitrary raw Gemini tool object for preview or custom server tools.
    Raw(Value),
}

impl GeminiServerTool {
    pub(crate) fn to_json(&self) -> Value {
        match self {
            Self::GoogleSearch => json!({ "googleSearch": {} }),
            Self::CodeExecution => json!({ "codeExecution": {} }),
            Self::UrlContext => json!({ "urlContext": {} }),
            Self::Raw(value) => value.clone(),
        }
    }
}

/// Bounds serialized requests and streamed response wire bytes.
///
/// Mirrors `OpenAIResponsesLimits`: imposes byte caps to prevent runaway memory
/// usage without imposing artificial cumulative SSE event-count limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeminiLimits {
    /// Maximum serialized JSON request body size in bytes.
    pub max_request_bytes: usize,
    /// Maximum wire bytes accepted from a single response attempt.
    pub max_attempt_bytes: usize,
    /// Maximum aggregate wire bytes accepted across all attempts in one logical turn.
    pub max_wire_bytes: usize,
}

impl Default for GeminiLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_attempt_bytes: DEFAULT_MAX_ATTEMPT_BYTES,
            max_wire_bytes: DEFAULT_MAX_WIRE_BYTES,
        }
    }
}

/// Configuration for connecting to Google's Gemini API (Developer API or Vertex AI).
#[derive(Clone)]
pub struct GeminiConfig {
    /// Authentication applied to each request (`x-goog-api-key` or bearer token).
    pub authentication: Authentication,
    /// Base URL without the `/models/{model}:generateContent` suffix.
    pub base_url: String,
    /// Model identifier, e.g. `"gemini-2.5-flash"` or `"models/gemini-2.5-pro"`.
    pub model: String,
    /// Maximum number of output tokens the model may generate.
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Custom stop sequences.
    pub stop_sequences: Option<Vec<String>>,
    /// Deterministic sampling seed.
    pub seed: Option<i64>,
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Thinking / reasoning configuration.
    pub thinking: Option<ThinkingConfig>,
    /// Function-calling tool choice constraint.
    pub tool_choice: Option<ToolChoice>,
    /// Server-side Gemini tools (`googleSearch`, `codeExecution`, `urlContext`, etc.).
    pub server_tools: Vec<GeminiServerTool>,
    /// Structured output format constraint.
    pub output_format: Option<OutputFormat>,
    /// Safety threshold overrides.
    pub safety_settings: Vec<SafetySetting>,
    /// Explicit cached content resource name (`cachedContents/{id}` or full Vertex path).
    pub cached_content: Option<String>,
    /// Function tool names already declared inside `cached_content`.
    ///
    /// When `cached_content` is active, registered tools must match this set
    /// unless [`Self::allow_unadvertised_cached_tools`] is enabled for
    /// dispatch-only tool sources (`ToolSource::unadvertised()`).
    pub cached_tool_names: BTreeSet<String>,
    /// Permits `cached_tool_names` to contain dispatch-only tools not advertised
    /// in `TurnRequest::available_tools` (for example when using
    /// `ToolSource::unadvertised()`). Defaults to `false`.
    pub allow_unadvertised_cached_tools: bool,
    /// System/developer/context instruction texts already embedded inside `cached_content`.
    ///
    /// When `cached_content` is active, transcript system parts must match
    /// these instructions exactly; matching instructions are omitted from the
    /// wire payload while new or modified instructions are rejected.
    pub cached_system_instructions: Vec<String>,
    /// Additional HTTP headers attached to every request.
    pub extra_headers: HeaderMap,
    /// Whether to request SSE streaming via `:streamGenerateContent?alt=sse`. Defaults to `true`.
    pub streaming: bool,
    /// Optional retry and timeout policy. `None` preserves single-attempt behavior.
    pub resilience: Option<ResilienceConfig>,
    /// Defensive request and wire-byte bounds.
    pub limits: GeminiLimits,
}

impl fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiConfig")
            .field("authentication", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("stop_sequences", &self.stop_sequences)
            .field("seed", &self.seed)
            .field("presence_penalty", &self.presence_penalty)
            .field("frequency_penalty", &self.frequency_penalty)
            .field("thinking", &self.thinking)
            .field("tool_choice", &self.tool_choice)
            .field("server_tools", &self.server_tools)
            .field("output_format", &self.output_format)
            .field("safety_settings", &self.safety_settings)
            .field("cached_content", &self.cached_content)
            .field("cached_tool_names", &self.cached_tool_names)
            .field(
                "allow_unadvertised_cached_tools",
                &self.allow_unadvertised_cached_tools,
            )
            .field(
                "cached_system_instructions_len",
                &self.cached_system_instructions.len(),
            )
            .field(
                "extra_header_names",
                &self.extra_headers.keys().collect::<Vec<_>>(),
            )
            .field("streaming", &self.streaming)
            .field("resilience", &self.resilience)
            .field("limits", &self.limits)
            .finish()
    }
}

impl GeminiConfig {
    /// Creates a new Developer API configuration using `x-goog-api-key` authentication.
    pub fn new(
        api_key: impl Into<GeminiApiKey>,
        model: impl Into<String>,
    ) -> Result<Self, GeminiError> {
        Self::from_authentication(api_key.into(), model)
    }

    /// Creates a new configuration using a bearer auth token (`Authorization: Bearer ...`).
    pub fn with_auth_token(
        auth_token: impl Into<GeminiAuthToken>,
        model: impl Into<String>,
    ) -> Result<Self, GeminiError> {
        Self::from_authentication(auth_token.into(), model)
    }

    /// Creates a configuration targeting Google Cloud Vertex AI for Gemini models.
    ///
    /// Constructs the regional Vertex AI base URL (or global endpoint when
    /// `location == "global"`):
    /// `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google`.
    /// Bare string credentials are treated as bearer OAuth2 tokens.
    pub fn vertex(
        project: impl AsRef<str>,
        location: impl AsRef<str>,
        model: impl Into<String>,
        authentication: impl Into<Authentication>,
    ) -> Result<Self, GeminiError> {
        let project = project.as_ref().trim();
        let location = location.as_ref().trim();
        let host = if location.eq_ignore_ascii_case("global") {
            "https://aiplatform.googleapis.com".to_string()
        } else {
            format!("https://{location}-aiplatform.googleapis.com")
        };
        let base_url =
            format!("{host}/v1/projects/{project}/locations/{location}/publishers/google");
        Ok(Self::from_authentication(authentication, model)?.with_base_url(base_url))
    }

    /// Creates a new configuration using arbitrary [`Authentication`].
    pub fn from_authentication(
        authentication: impl Into<Authentication>,
        model: impl Into<String>,
    ) -> Result<Self, GeminiError> {
        let model = model.into();
        if model.trim().is_empty() {
            return Err(GeminiError::InvalidModel);
        }
        Ok(Self {
            authentication: authentication.into(),
            base_url: DEFAULT_BASE_URL.into(),
            model,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            thinking: None,
            tool_choice: None,
            server_tools: Vec::new(),
            output_format: None,
            safety_settings: Vec::new(),
            cached_content: None,
            cached_tool_names: BTreeSet::new(),
            allow_unadvertised_cached_tools: false,
            cached_system_instructions: Vec::new(),
            extra_headers: HeaderMap::new(),
            streaming: true,
            resilience: None,
            limits: GeminiLimits::default(),
        })
    }

    /// Builds a configuration from environment variables.
    ///
    /// | Variable | Required | Default |
    /// |---|---|---|
    /// | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_AUTH_TOKEN` | one | — |
    /// | `GEMINI_MODEL` | yes | — |
    /// | `GEMINI_BASE_URL` | no | `https://generativelanguage.googleapis.com/v1beta` |
    /// | `GEMINI_MAX_OUTPUT_TOKENS` | no | model default |
    pub fn from_env() -> Result<Self, GeminiError> {
        let model =
            std::env::var("GEMINI_MODEL").map_err(|_| GeminiError::MissingEnv("GEMINI_MODEL"))?;

        let mut config = match (
            std::env::var("GEMINI_AUTH_TOKEN").ok(),
            std::env::var("GEMINI_API_KEY")
                .ok()
                .or_else(|| std::env::var("GOOGLE_API_KEY").ok()),
        ) {
            (Some(token), _) if !token.is_empty() => Self::with_auth_token(token, model)?,
            (_, Some(key)) if !key.is_empty() => Self::new(key, model)?,
            _ => return Err(GeminiError::MissingCredentials),
        };

        if let Ok(url) = std::env::var("GEMINI_BASE_URL")
            && !url.trim().is_empty()
        {
            config = config.with_base_url(url);
        }
        if let Ok(raw_max) = std::env::var("GEMINI_MAX_OUTPUT_TOKENS") {
            let max_tokens = raw_max
                .parse::<u32>()
                .map_err(|_| GeminiError::MissingEnv("GEMINI_MAX_OUTPUT_TOKENS"))?;
            config = config.with_max_output_tokens(max_tokens);
        }

        Ok(config)
    }

    /// Computes the full endpoint URL for the configured model and streaming mode.
    pub(crate) fn endpoint_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        let model = self
            .model
            .trim()
            .strip_prefix("models/")
            .unwrap_or(self.model.trim());
        if self.streaming {
            format!("{base}/models/{model}:streamGenerateContent?alt=sse")
        } else {
            format!("{base}/models/{model}:generateContent")
        }
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

    /// Overrides the defensive request and wire-byte limits.
    pub fn with_limits(mut self, limits: GeminiLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Overrides the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the maximum number of output tokens per turn.
    pub fn with_max_output_tokens(mut self, v: u32) -> Self {
        self.max_output_tokens = Some(v);
        self
    }

    /// Sets the sampling temperature.
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

    /// Sets the sampling seed.
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the presence penalty.
    pub fn with_presence_penalty(mut self, v: f32) -> Self {
        self.presence_penalty = Some(v);
        self
    }

    /// Sets the frequency penalty.
    pub fn with_frequency_penalty(mut self, v: f32) -> Self {
        self.frequency_penalty = Some(v);
        self
    }

    /// Sets the thinking / reasoning configuration.
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Sets the function-calling tool choice constraint.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Appends a built-in Gemini server tool.
    pub fn with_server_tool(mut self, tool: GeminiServerTool) -> Self {
        self.server_tools.push(tool);
        self
    }

    /// Sets the structured output format constraint.
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Appends a safety setting override.
    pub fn with_safety_setting(mut self, setting: SafetySetting) -> Self {
        self.safety_settings.push(setting);
        self
    }

    /// Sets an explicit cached content resource name.
    pub fn with_cached_content(mut self, resource_name: impl Into<String>) -> Self {
        self.cached_content = Some(resource_name.into());
        self
    }

    /// Declares the function tool names already embedded inside `cached_content`.
    pub fn with_cached_tool_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cached_tool_names = names.into_iter().map(Into::into).collect();
        self
    }

    /// Allows `cached_tool_names` to include dispatch-only tools omitted from
    /// `TurnRequest::available_tools` (e.g. `ToolSource::unadvertised()`).
    pub fn with_unadvertised_cached_tools(mut self, allow: bool) -> Self {
        self.allow_unadvertised_cached_tools = allow;
        self
    }

    /// Declares the exact system instruction texts already embedded inside `cached_content`.
    pub fn with_cached_system_instructions<I, S>(mut self, instructions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cached_system_instructions = instructions.into_iter().map(Into::into).collect();
        self
    }

    /// Adds a custom header sent with every request.
    pub fn with_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.extra_headers.insert(name, value);
        self
    }

    /// Toggles SSE streaming of model responses. Default: `true`.
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_url_normalizes_model_prefix_and_streaming() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        assert_eq!(
            cfg.endpoint_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );

        let cfg = GeminiConfig::new("k", "models/gemini-2.5-pro")
            .unwrap()
            .with_streaming(false);
        assert_eq!(
            cfg.endpoint_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        );
    }

    #[test]
    fn vertex_helper_constructs_regional_and_global_endpoints() {
        let regional =
            GeminiConfig::vertex("my-proj", "us-central1", "gemini-2.5-pro", "oauth-tok").unwrap();
        assert_eq!(
            regional.endpoint_url(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );

        let global = GeminiConfig::vertex("my-proj", "global", "gemini-2.5-flash", "oauth-tok")
            .unwrap()
            .with_streaming(false);
        assert_eq!(
            global.endpoint_url(),
            "https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global/publishers/google/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn debug_redacts_credentials() {
        let cfg = GeminiConfig::new("super-secret-api-key", "gemini-2.5-flash").unwrap();
        let debug = format!("{cfg:?}");
        assert!(!debug.contains("super-secret-api-key"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn rejects_empty_model() {
        assert!(matches!(
            GeminiConfig::new("key", "   "),
            Err(GeminiError::InvalidModel)
        ));
    }
}
