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
//! use agentkit_loop::{Agent, PromptCacheRequest, PromptCacheRetention, SessionConfig};
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
//!         .start(
//!             SessionConfig::new("demo").with_cache(
//!                 PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
//!             ),
//!         )
//!         .await?;
//!     Ok(())
//! }
//! ```

use agentkit_adapter_completions::{
    CompletionsAdapter, CompletionsError, CompletionsProvider, CompletionsSession, CompletionsTurn,
};
use agentkit_core::{CostUsage, Item, ItemKind, MetadataMap, Part, Usage};
use agentkit_loop::{
    LoopError, ModelAdapter, PromptCacheBreakpoint, PromptCacheMode, PromptCacheRequest,
    PromptCacheRetention, PromptCacheStrategy, SessionConfig, TurnRequest,
};
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

        match &cache.strategy {
            PromptCacheStrategy::Automatic => {
                if model_supports_openrouter_explicit_cache(&self.request_config.model) {
                    let breakpoints = plan_automatic_prompt_cache_breakpoints(request);
                    return apply_openrouter_explicit_cache_breakpoints(
                        body,
                        &request.transcript,
                        cache,
                        &breakpoints,
                    );
                }

                if matches!(cache.mode, PromptCacheMode::Required) {
                    return Err(LoopError::Provider(format!(
                        "OpenRouter model {} does not support automatic prompt caching controls",
                        self.request_config.model
                    )));
                }

                Ok(())
            }
            PromptCacheStrategy::Explicit { breakpoints } => {
                if !model_supports_openrouter_explicit_cache(&self.request_config.model) {
                    if matches!(cache.mode, PromptCacheMode::Required) {
                        return Err(LoopError::Provider(format!(
                            "OpenRouter model {} does not support explicit prompt cache breakpoints",
                            self.request_config.model
                        )));
                    }
                    return Ok(());
                }

                apply_openrouter_explicit_cache_breakpoints(
                    body,
                    &request.transcript,
                    cache,
                    breakpoints,
                )
            }
        }
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

fn model_supports_openrouter_explicit_cache(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.starts_with("anthropic/") || model.contains("claude")
}

fn openrouter_cache_control(retention: Option<PromptCacheRetention>) -> Value {
    match retention.unwrap_or(PromptCacheRetention::Default) {
        PromptCacheRetention::Default | PromptCacheRetention::Short => {
            serde_json::json!({ "type": "ephemeral" })
        }
        PromptCacheRetention::Extended => {
            serde_json::json!({ "type": "ephemeral", "ttl": "1h" })
        }
    }
}

fn plan_automatic_prompt_cache_breakpoints(request: &TurnRequest) -> Vec<PromptCacheBreakpoint> {
    let mut breakpoints = Vec::new();

    if !request.available_tools.is_empty() {
        breakpoints.push(PromptCacheBreakpoint::ToolsEnd);
    }

    if let Some(index) = request
        .transcript
        .iter()
        .enumerate()
        .rev()
        .find(|(_, item)| {
            matches!(
                item.kind,
                ItemKind::System | ItemKind::Developer | ItemKind::Context
            )
        })
        .map(|(index, _)| index)
    {
        breakpoints.push(PromptCacheBreakpoint::TranscriptItemEnd { index });
    }

    if let Some(index) = stable_history_prefix_end(&request.transcript) {
        breakpoints.push(PromptCacheBreakpoint::TranscriptItemEnd { index });
    }

    dedupe_breakpoints(breakpoints)
}

fn stable_history_prefix_end(transcript: &[Item]) -> Option<usize> {
    if transcript.is_empty() {
        return None;
    }

    let mut boundary = transcript.len();
    while boundary > 0
        && matches!(
            transcript[boundary - 1].kind,
            ItemKind::User | ItemKind::Tool
        )
    {
        boundary -= 1;
    }

    if boundary > 0 {
        Some(boundary - 1)
    } else {
        transcript.len().checked_sub(1)
    }
}

fn dedupe_breakpoints(breakpoints: Vec<PromptCacheBreakpoint>) -> Vec<PromptCacheBreakpoint> {
    let mut deduped = Vec::new();
    for breakpoint in breakpoints {
        if !deduped.contains(&breakpoint) {
            deduped.push(breakpoint);
        }
    }
    deduped
}

fn apply_openrouter_explicit_cache_breakpoints(
    body: &mut serde_json::Map<String, Value>,
    transcript: &[Item],
    cache: &PromptCacheRequest,
    breakpoints: &[PromptCacheBreakpoint],
) -> Result<(), LoopError> {
    let cache_control = openrouter_cache_control(cache.retention);
    let selected = if breakpoints.len() > 4 {
        &breakpoints[breakpoints.len() - 4..]
    } else {
        breakpoints
    };

    let mut unsupported = Vec::new();

    for breakpoint in selected {
        let applied = match breakpoint {
            PromptCacheBreakpoint::ToolsEnd => {
                apply_cache_control_to_last_tool(body, cache_control.clone())
            }
            PromptCacheBreakpoint::TranscriptItemEnd { index } => {
                resolve_item_message_end(transcript, *index).is_some_and(|message_index| {
                    apply_cache_control_to_message(
                        body,
                        transcript,
                        *index,
                        message_index,
                        None,
                        cache_control.clone(),
                    )
                })
            }
            PromptCacheBreakpoint::TranscriptPartEnd {
                item_index,
                part_index,
            } => resolve_part_message_target(transcript, *item_index, *part_index).is_some_and(
                |(message_index, content_part_index)| {
                    apply_cache_control_to_message(
                        body,
                        transcript,
                        *item_index,
                        message_index,
                        content_part_index,
                        cache_control.clone(),
                    )
                },
            ),
        };

        if !applied {
            unsupported.push(format!("{breakpoint:?}"));
        }
    }

    if !unsupported.is_empty() && matches!(cache.mode, PromptCacheMode::Required) {
        return Err(LoopError::Provider(format!(
            "OpenRouter could not apply required cache breakpoints: {}",
            unsupported.join(", ")
        )));
    }

    Ok(())
}

fn apply_cache_control_to_last_tool(
    body: &mut serde_json::Map<String, Value>,
    cache_control: Value,
) -> bool {
    let Some(Value::Array(tools)) = body.get_mut("tools") else {
        return false;
    };
    let Some(Value::Object(tool)) = tools.last_mut() else {
        return false;
    };
    tool.insert("cache_control".into(), cache_control);
    true
}

fn resolve_item_message_end(transcript: &[Item], target_index: usize) -> Option<usize> {
    let mut message_index = 0usize;

    for (item_index, item) in transcript.iter().enumerate() {
        if item.kind == ItemKind::Tool {
            let tool_result_count = item
                .parts
                .iter()
                .filter(|part| matches!(part, agentkit_core::Part::ToolResult(_)))
                .count();
            if item_index == target_index {
                return tool_result_count
                    .checked_sub(1)
                    .map(|offset| message_index + offset);
            }
            message_index += tool_result_count;
        } else {
            if item_index == target_index {
                return Some(message_index);
            }
            message_index += 1;
        }
    }

    None
}

fn resolve_part_message_target(
    transcript: &[Item],
    target_item_index: usize,
    target_part_index: usize,
) -> Option<(usize, Option<usize>)> {
    let mut message_index = 0usize;

    for (item_index, item) in transcript.iter().enumerate() {
        if item.kind == ItemKind::Tool {
            for (part_index, part) in item.parts.iter().enumerate() {
                if matches!(part, agentkit_core::Part::ToolResult(_)) {
                    if item_index == target_item_index && part_index == target_part_index {
                        return Some((message_index, Some(0)));
                    }
                    message_index += 1;
                }
            }
            continue;
        }

        if item_index == target_item_index {
            return Some((message_index, Some(target_part_index)));
        }

        message_index += 1;
    }

    None
}

fn apply_cache_control_to_message(
    body: &mut serde_json::Map<String, Value>,
    transcript: &[Item],
    transcript_item_index: usize,
    message_index: usize,
    content_part_index: Option<usize>,
    cache_control: Value,
) -> bool {
    let Some(Value::Array(messages)) = body.get_mut("messages") else {
        return false;
    };
    let Some(Value::Object(message)) = messages.get_mut(message_index) else {
        return false;
    };

    let original_content = match message.get("content").cloned() {
        Some(content) => content,
        None => return false,
    };

    let Some(item) = transcript.get(transcript_item_index) else {
        return false;
    };

    let (mut blocks, part_mapping) = match message_content_to_blocks(item, original_content.clone())
    {
        Some(value) => value,
        None => return false,
    };

    let target_block_index = match content_part_index {
        Some(part_index) => part_mapping.iter().position(|mapped| *mapped == part_index),
        None => blocks.len().checked_sub(1),
    };

    let Some(block_index) = target_block_index else {
        message.insert("content".into(), original_content);
        return false;
    };

    let Some(Value::Object(block)) = blocks.get_mut(block_index) else {
        message.insert("content".into(), original_content);
        return false;
    };
    block.insert("cache_control".into(), cache_control);
    message.insert("content".into(), Value::Array(blocks));
    true
}

fn message_content_to_blocks(item: &Item, content: Value) -> Option<(Vec<Value>, Vec<usize>)> {
    match content {
        Value::Array(blocks) => {
            let mapping = if item.kind == ItemKind::Tool {
                vec![0; blocks.len()]
            } else {
                cacheable_content_part_indices(item)
            };
            Some((blocks, mapping))
        }
        Value::String(text) => {
            if item.kind == ItemKind::Tool {
                return Some((
                    vec![serde_json::json!({
                        "type": "text",
                        "text": text,
                    })],
                    vec![0],
                ));
            }

            if item.parts.len() != 1 {
                return None;
            }

            Some((
                vec![serde_json::json!({
                    "type": "text",
                    "text": text,
                })],
                vec![0],
            ))
        }
        Value::Null => None,
        other => Some((vec![other], vec![0])),
    }
}

fn cacheable_content_part_indices(item: &Item) -> Vec<usize> {
    item.parts
        .iter()
        .enumerate()
        .filter_map(|(index, part)| match item.kind {
            ItemKind::System | ItemKind::Developer | ItemKind::Context => match part {
                Part::Text(_) | Part::Structured(_) => Some(index),
                Part::Reasoning(reasoning) if reasoning.summary.is_some() => Some(index),
                _ => None,
            },
            ItemKind::User => match part {
                Part::Text(_) | Part::Structured(_) | Part::Media(_) | Part::File(_) => Some(index),
                Part::Reasoning(reasoning) if reasoning.summary.is_some() => Some(index),
                _ => None,
            },
            ItemKind::Assistant => match part {
                Part::Text(_) | Part::Structured(_) => Some(index),
                Part::Reasoning(reasoning) if reasoning.summary.is_some() => Some(index),
                _ => None,
            },
            ItemKind::Tool => None,
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TextPart, TurnId};

    use super::*;

    fn empty_turn_request(transcript: Vec<Item>, cache: Option<PromptCacheRequest>) -> TurnRequest {
        TurnRequest {
            session_id: SessionId::new("session"),
            turn_id: TurnId::new("turn-1"),
            transcript,
            available_tools: Vec::new(),
            cache,
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn openrouter_maps_automatic_cache_request_for_claude_models() {
        let provider = OpenRouterProvider::from(OpenRouterConfig::new(
            "sk-or-test",
            "anthropic/claude-sonnet-4.6",
        ));
        let mut body = serde_json::Map::new();
        body.insert(
            "messages".into(),
            Value::Array(vec![
                serde_json::json!({
                    "role": "system",
                    "content": "system prompt"
                }),
                serde_json::json!({
                    "role": "user",
                    "content": "latest input"
                }),
            ]),
        );
        body.insert("tools".into(), Value::Array(Vec::new()));

        provider
            .apply_prompt_cache(
                &mut body,
                &empty_turn_request(
                    vec![
                        Item {
                            id: None,
                            kind: ItemKind::System,
                            parts: vec![Part::Text(TextPart {
                                text: "system prompt".into(),
                                metadata: MetadataMap::new(),
                            })],
                            metadata: MetadataMap::new(),
                        },
                        Item {
                            id: None,
                            kind: ItemKind::User,
                            parts: vec![Part::Text(TextPart {
                                text: "latest input".into(),
                                metadata: MetadataMap::new(),
                            })],
                            metadata: MetadataMap::new(),
                        },
                    ],
                    Some(
                        PromptCacheRequest::best_effort(PromptCacheStrategy::Automatic)
                            .with_retention(PromptCacheRetention::Extended),
                    ),
                ),
            )
            .unwrap();

        assert_eq!(
            body.get("messages"),
            Some(&Value::Array(vec![
                serde_json::json!({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "system prompt",
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            }
                        }
                    ]
                }),
                serde_json::json!({
                    "role": "user",
                    "content": "latest input"
                }),
            ]))
        );
    }

    #[test]
    fn openrouter_applies_explicit_breakpoint_to_message_content() {
        let provider = OpenRouterProvider::from(OpenRouterConfig::new(
            "sk-or-test",
            "anthropic/claude-sonnet-4.6",
        ));
        let mut body = serde_json::Map::new();
        body.insert(
            "messages".into(),
            Value::Array(vec![serde_json::json!({
                "role": "user",
                "content": "hello"
            })]),
        );
        body.insert("tools".into(), Value::Array(Vec::new()));

        provider
            .apply_prompt_cache(
                &mut body,
                &empty_turn_request(
                    vec![Item {
                        id: None,
                        kind: ItemKind::User,
                        parts: vec![Part::Text(TextPart {
                            text: "hello".into(),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                    }],
                    Some(PromptCacheRequest::required(
                        PromptCacheStrategy::Explicit {
                            breakpoints: vec![PromptCacheBreakpoint::TranscriptItemEnd {
                                index: 0,
                            }],
                        },
                    )),
                ),
            )
            .unwrap();

        assert_eq!(
            body.get("messages"),
            Some(&Value::Array(vec![serde_json::json!({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": { "type": "ephemeral" }
                    }
                ]
            })]))
        );
    }
}
