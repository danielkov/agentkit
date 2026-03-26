//! OpenRouter model adapter for the agentkit agent loop.
//!
//! This crate translates between agentkit transcript primitives and OpenRouter
//! chat completion requests. It handles tool declaration, tool-call decoding,
//! multimodal content mapping (images, audio), usage normalization, and
//! environment-based configuration.
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

use std::collections::VecDeque;
use std::sync::Arc;

use agentkit_core::{
    CostUsage, DataRef, Delta, FilePart, FinishReason, Item, ItemKind, MediaPart, MetadataMap,
    Modality, Part, PartKind, ReasoningPart, TextPart, TokenUsage, ToolCallPart, ToolOutput,
    TurnCancellation, Usage,
};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult,
    SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use base64::Engine;
use futures_util::future::{Either, select};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Configuration for connecting to the OpenRouter API.
///
/// Holds credentials, model selection, and optional request parameters.
/// Build one with [`OpenRouterConfig::new`] for explicit values, or
/// [`OpenRouterConfig::from_env`] to read from environment variables.
/// Chain builder methods to customise temperature, token limits, and
/// arbitrary extra body fields before passing the config to
/// [`OpenRouterAdapter::new`].
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
    ///
    /// All optional fields default to `None` and the base URL defaults to
    /// the OpenRouter production endpoint.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenRouter API key.
    /// * `model` - Model identifier, e.g. `"anthropic/claude-sonnet-4"`.
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
    ///
    /// Use this to pass provider-specific parameters (e.g. `top_p`, `top_k`)
    /// that are not covered by dedicated builder methods.
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
    ///
    /// # Errors
    ///
    /// Returns [`OpenRouterError::MissingEnv`] if `OPENROUTER_API_KEY` is unset,
    /// or [`OpenRouterError::InvalidConfig`] if a numeric variable cannot be
    /// parsed.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use agentkit_provider_openrouter::OpenRouterConfig;
    ///
    /// let config = OpenRouterConfig::from_env()?
    ///     .with_temperature(0.0);
    /// # Ok::<(), agentkit_provider_openrouter::OpenRouterError>(())
    /// ```
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

/// Model adapter that connects the agentkit agent loop to OpenRouter.
///
/// Implements [`ModelAdapter`] so it can be passed to
/// [`Agent::builder().model()`](agentkit_loop::Agent::builder). Each call to
/// [`start_session`](ModelAdapter::start_session) produces an
/// [`OpenRouterSession`] that manages HTTP requests for individual turns.
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
pub struct OpenRouterAdapter {
    client: Client,
    config: Arc<OpenRouterConfig>,
}

impl OpenRouterAdapter {
    /// Creates a new adapter from the given configuration.
    ///
    /// Initialises an HTTP client that will be reused for all sessions and
    /// turns created from this adapter.
    ///
    /// # Errors
    ///
    /// Returns [`OpenRouterError::HttpClient`] if the underlying HTTP client
    /// cannot be constructed.
    pub fn new(config: OpenRouterConfig) -> Result<Self, OpenRouterError> {
        let client = Client::builder()
            .user_agent("agentkit-provider-openrouter/0.1.0")
            .build()
            .map_err(OpenRouterError::HttpClient)?;

        Ok(Self {
            client,
            config: Arc::new(config),
        })
    }
}

/// An active session with the OpenRouter API.
///
/// Created by [`OpenRouterAdapter::start_session`](ModelAdapter::start_session).
/// Holds the HTTP client and shared configuration; each call to
/// [`begin_turn`](ModelSession::begin_turn) sends a single chat completion
/// request and returns an [`OpenRouterTurn`] with the buffered response events.
pub struct OpenRouterSession {
    client: Client,
    config: Arc<OpenRouterConfig>,
    _session_config: SessionConfig,
}

/// A completed turn holding buffered events from a single OpenRouter response.
///
/// Because OpenRouter responses are consumed non-streaming (the full JSON body
/// is read at once), all events are available immediately. Successive calls to
/// [`next_event`](ModelTurn::next_event) drain the internal queue until it is
/// empty.
pub struct OpenRouterTurn {
    events: VecDeque<ModelTurnEvent>,
}

#[async_trait]
impl ModelAdapter for OpenRouterAdapter {
    type Session = OpenRouterSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(OpenRouterSession {
            client: self.client.clone(),
            config: self.config.clone(),
            _session_config: config,
        })
    }
}

#[async_trait]
impl ModelSession for OpenRouterSession {
    type Turn = OpenRouterTurn;

    async fn begin_turn(
        &mut self,
        request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Self::Turn, LoopError> {
        let request_future = async {
            let body = build_request_body(&self.config, &request).map_err(as_loop_error)?;
            let mut http = self
                .client
                .post(&self.config.base_url)
                .bearer_auth(&self.config.api_key)
                .header("Content-Type", "application/json");

            if let Some(app_name) = &self.config.app_name {
                http = http.header("X-Title", app_name);
            }
            if let Some(site_url) = &self.config.site_url {
                http = http.header("HTTP-Referer", site_url);
            }

            let response = http.json(&body).send().await.map_err(|error| {
                LoopError::Provider(format!("OpenRouter request failed: {error}"))
            })?;

            let status = response.status();
            if !status.is_success() {
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "<unreadable response body>".into());
                return Err(LoopError::Provider(format!(
                    "OpenRouter request failed with status {status}: {body}"
                )));
            }

            let body = response.text().await.map_err(|error| {
                LoopError::Provider(format!("failed to read OpenRouter response body: {error}"))
            })?;

            // OpenRouter sometimes returns 200 with an error body instead of
            // a proper HTTP error status.  Check for that before attempting
            // to deserialize the normal completion shape.
            if let Ok(err_resp) = serde_json::from_str::<OpenRouterErrorResponse>(&body) {
                return Err(LoopError::Provider(format!(
                    "OpenRouter returned error (code {}): {}",
                    err_resp.error.code, err_resp.error.message,
                )));
            }

            let completion: ChatCompletionResponse =
                serde_json::from_str(&body).map_err(|error| {
                    LoopError::Provider(format!(
                        "invalid OpenRouter response: {error}\n\nRaw response body:\n{body}"
                    ))
                })?;

            build_turn_from_response(completion).map_err(as_loop_error)
        };

        if let Some(cancellation) = cancellation {
            futures_util::pin_mut!(request_future);
            let cancelled = cancellation.cancelled();
            futures_util::pin_mut!(cancelled);
            match select(request_future, cancelled).await {
                Either::Left((result, _)) => result,
                Either::Right((_, _)) => Err(LoopError::Cancelled),
            }
        } else {
            request_future.await
        }
    }
}

#[async_trait]
impl ModelTurn for OpenRouterTurn {
    async fn next_event(
        &mut self,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return Err(LoopError::Cancelled);
        }
        Ok(self.events.pop_front())
    }
}

fn build_request_body(
    config: &OpenRouterConfig,
    request: &TurnRequest,
) -> Result<Value, OpenRouterError> {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), Value::String(config.model.clone()));
    body.insert(
        "messages".into(),
        Value::Array(build_messages(&request.transcript)?),
    );
    body.insert("stream".into(), Value::Bool(false));
    body.insert(
        "tools".into(),
        Value::Array(build_tools(&request.available_tools)),
    );
    body.insert(
        "parallel_tool_calls".into(),
        Value::Bool(!request.available_tools.is_empty()),
    );
    body.insert("user".into(), Value::String(request.session_id.0.clone()));

    if let Some(max_completion_tokens) = config.max_completion_tokens {
        body.insert(
            "max_completion_tokens".into(),
            Value::Number(max_completion_tokens.into()),
        );
    }

    if let Some(temperature) = config.temperature {
        body.insert("temperature".into(), json!(temperature));
    }

    for (key, value) in &config.extra_body {
        body.insert(key.clone(), value.clone());
    }

    Ok(Value::Object(body))
}

fn build_tools(tool_specs: &[agentkit_tools_core::ToolSpec]) -> Vec<Value> {
    tool_specs
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name.0,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect()
}

fn build_messages(transcript: &[Item]) -> Result<Vec<Value>, OpenRouterError> {
    let mut messages = Vec::new();

    for item in transcript {
        match item.kind {
            ItemKind::Tool => {
                for part in &item.parts {
                    let Part::ToolResult(result) = part else {
                        return Err(OpenRouterError::UnsupportedPart {
                            role: item.kind,
                            part_kind: part.kind(),
                        });
                    };
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": result.call_id.0,
                        "content": tool_output_to_string(&result.output),
                    }));
                }
            }
            _ => messages.push(build_message(item)?),
        }
    }

    Ok(messages)
}

fn build_message(item: &Item) -> Result<Value, OpenRouterError> {
    match item.kind {
        ItemKind::System | ItemKind::Developer | ItemKind::Context => Ok(json!({
            "role": role_for_item_kind(item.kind),
            "content": stringify_parts(&item.parts, item.kind)?,
        })),
        ItemKind::User => Ok(json!({
            "role": "user",
            "content": build_user_content(&item.parts)?,
        })),
        ItemKind::Assistant => build_assistant_message(item),
        ItemKind::Tool => Err(OpenRouterError::InvalidTranscript(
            "tool items must be expanded at the transcript level".into(),
        )),
    }
}

fn build_assistant_message(item: &Item) -> Result<Value, OpenRouterError> {
    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();

    for part in &item.parts {
        match part {
            Part::Text(text) => content_parts.push(json!({
                "type": "text",
                "text": text.text,
            })),
            Part::Structured(structured) => content_parts.push(json!({
                "type": "text",
                "text": serde_json::to_string(&structured.value).map_err(OpenRouterError::Serialize)?,
            })),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    content_parts.push(json!({
                        "type": "text",
                        "text": summary,
                    }));
                }
            }
            Part::ToolCall(call) => tool_calls.push(json!({
                "id": call.id.0,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": serde_json::to_string(&call.input).map_err(OpenRouterError::Serialize)?,
                }
            })),
            Part::ToolResult(_) => {
                return Err(OpenRouterError::UnsupportedPart {
                    role: item.kind,
                    part_kind: PartKind::ToolResult,
                });
            }
            Part::Media(_) | Part::File(_) | Part::Custom(_) => {
                return Err(OpenRouterError::UnsupportedPart {
                    role: item.kind,
                    part_kind: part.kind(),
                });
            }
        }
    }

    let content = if content_parts.is_empty() {
        Value::Null
    } else if content_parts.len() == 1 && content_parts[0]["type"] == "text" {
        Value::String(
            content_parts[0]["text"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        )
    } else {
        Value::Array(content_parts)
    };

    Ok(json!({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }))
}

fn build_user_content(parts: &[Part]) -> Result<Value, OpenRouterError> {
    let mut content = Vec::new();

    for part in parts {
        match part {
            Part::Text(text) => content.push(json!({
                "type": "text",
                "text": text.text,
            })),
            Part::Structured(structured) => content.push(json!({
                "type": "text",
                "text": serde_json::to_string_pretty(&structured.value)
                    .map_err(OpenRouterError::Serialize)?,
            })),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    content.push(json!({
                        "type": "text",
                        "text": summary,
                    }));
                }
            }
            Part::Media(media) => content.push(media_to_openrouter_content(media)?),
            Part::File(file) => content.push(file_to_openrouter_content(file)?),
            Part::ToolCall(_) | Part::ToolResult(_) | Part::Custom(_) => {
                return Err(OpenRouterError::UnsupportedPart {
                    role: ItemKind::User,
                    part_kind: part.kind(),
                });
            }
        }
    }

    if content.len() == 1 && content[0]["type"] == "text" {
        Ok(Value::String(
            content[0]["text"].as_str().unwrap_or_default().to_string(),
        ))
    } else {
        Ok(Value::Array(content))
    }
}

fn media_to_openrouter_content(media: &MediaPart) -> Result<Value, OpenRouterError> {
    match media.modality {
        Modality::Image => Ok(json!({
            "type": "image_url",
            "image_url": {
                "url": data_ref_to_url_like(&media.data, &media.mime_type)?,
            }
        })),
        Modality::Audio => Ok(json!({
            "type": "input_audio",
            "input_audio": {
                "data": data_ref_to_base64(&media.data)?,
                "format": audio_format_from_mime(&media.mime_type),
            }
        })),
        Modality::Video | Modality::Binary => {
            Err(OpenRouterError::UnsupportedModality(media.modality))
        }
    }
}

fn file_to_openrouter_content(file: &FilePart) -> Result<Value, OpenRouterError> {
    match file.mime_type.as_deref() {
        Some(mime) if mime.starts_with("image/") => Ok(json!({
            "type": "image_url",
            "image_url": {
                "url": data_ref_to_url_like(&file.data, mime)?,
            }
        })),
        Some(mime) if mime.starts_with("audio/") => Ok(json!({
            "type": "input_audio",
            "input_audio": {
                "data": data_ref_to_base64(&file.data)?,
                "format": audio_format_from_mime(mime),
            }
        })),
        _ => Ok(json!({
            "type": "text",
            "text": format!(
                "Attached file{}{}",
                file.name
                    .as_ref()
                    .map(|name| format!(": {name}"))
                    .unwrap_or_default(),
                file.mime_type
                    .as_ref()
                    .map(|mime| format!(" ({mime})"))
                    .unwrap_or_default(),
            ),
        })),
    }
}

fn stringify_parts(parts: &[Part], role: ItemKind) -> Result<String, OpenRouterError> {
    let mut segments = Vec::new();

    for part in parts {
        match part {
            Part::Text(text) => segments.push(text.text.clone()),
            Part::Structured(structured) => segments.push(
                serde_json::to_string_pretty(&structured.value)
                    .map_err(OpenRouterError::Serialize)?,
            ),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    segments.push(summary.clone());
                }
            }
            _ => {
                return Err(OpenRouterError::UnsupportedPart {
                    role,
                    part_kind: part.kind(),
                });
            }
        }
    }

    Ok(segments.join("\n\n"))
}

fn role_for_item_kind(kind: ItemKind) -> &'static str {
    match kind {
        ItemKind::System | ItemKind::Context => "system",
        ItemKind::Developer => "developer",
        ItemKind::User => "user",
        ItemKind::Assistant => "assistant",
        ItemKind::Tool => "tool",
    }
}

fn tool_output_to_string(output: &ToolOutput) -> String {
    match output {
        ToolOutput::Text(text) => text.clone(),
        ToolOutput::Structured(value) => value.to_string(),
        ToolOutput::Parts(parts) => serde_json::to_string(parts).unwrap_or_else(|_| "[]".into()),
        ToolOutput::Files(files) => serde_json::to_string(files).unwrap_or_else(|_| "[]".into()),
    }
}

fn data_ref_to_url_like(data: &DataRef, mime_type: &str) -> Result<String, OpenRouterError> {
    match data {
        DataRef::Uri(uri) => Ok(uri.clone()),
        DataRef::InlineBytes(bytes) => Ok(format!(
            "data:{mime_type};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )),
        DataRef::InlineText(text) => {
            if text.starts_with("data:")
                || text.starts_with("http://")
                || text.starts_with("https://")
            {
                Ok(text.clone())
            } else {
                Err(OpenRouterError::UnsupportedDataRef(
                    "image inputs must be a URL, data URL, or inline bytes".into(),
                ))
            }
        }
        DataRef::Handle(handle) => Err(OpenRouterError::UnsupportedDataRef(format!(
            "artifact handle {} cannot be sent directly to OpenRouter",
            handle.0
        ))),
    }
}

fn data_ref_to_base64(data: &DataRef) -> Result<String, OpenRouterError> {
    match data {
        DataRef::InlineBytes(bytes) => Ok(base64::engine::general_purpose::STANDARD.encode(bytes)),
        DataRef::InlineText(text) => {
            if let Some((_, encoded)) = text.split_once(";base64,") {
                Ok(encoded.to_string())
            } else {
                Ok(base64::engine::general_purpose::STANDARD.encode(text.as_bytes()))
            }
        }
        DataRef::Uri(uri) => Err(OpenRouterError::UnsupportedDataRef(format!(
            "audio input URI {uri} must be loaded into bytes first"
        ))),
        DataRef::Handle(handle) => Err(OpenRouterError::UnsupportedDataRef(format!(
            "artifact handle {} cannot be sent directly to OpenRouter",
            handle.0
        ))),
    }
}

fn audio_format_from_mime(mime: &str) -> &'static str {
    match mime {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        _ => "wav",
    }
}

fn build_turn_from_response(
    response: ChatCompletionResponse,
) -> Result<OpenRouterTurn, OpenRouterError> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| OpenRouterError::Protocol("response contained no choices".into()))?;

    let mut events = VecDeque::new();

    if let Some(usage) = map_usage(response.usage) {
        events.push_back(ModelTurnEvent::Usage(usage));
    }

    let message = choice.message;
    let mut parts = message_to_parts(&message)?;
    let finish_reason = map_finish_reason(choice.finish_reason.as_deref());

    for part in &parts {
        if let Part::ToolCall(call) = part {
            events.push_back(ModelTurnEvent::ToolCall(call.clone()));
        }
    }

    if !parts.is_empty() {
        let assistant_item = Item {
            id: response.id.map(Into::into),
            kind: ItemKind::Assistant,
            parts: std::mem::take(&mut parts),
            metadata: response_metadata(&response.model, &message),
        };

        for part in &assistant_item.parts {
            events.push_back(ModelTurnEvent::Delta(Delta::CommitPart {
                part: part.clone(),
            }));
        }

        events.push_back(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason,
            output_items: vec![assistant_item],
            usage: map_usage(response.usage),
            metadata: MetadataMap::new(),
        }));
    } else {
        events.push_back(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason,
            output_items: Vec::new(),
            usage: map_usage(response.usage),
            metadata: MetadataMap::new(),
        }));
    }

    Ok(OpenRouterTurn { events })
}

fn message_to_parts(message: &ResponseMessage) -> Result<Vec<Part>, OpenRouterError> {
    let mut parts = Vec::new();

    if let Some(reasoning) = &message.reasoning {
        parts.push(Part::Reasoning(ReasoningPart {
            summary: Some(reasoning.clone()),
            data: None,
            redacted: false,
            metadata: MetadataMap::new(),
        }));
    }

    if let Some(reasoning_details) = &message.reasoning_details
        && !reasoning_details.is_null()
    {
        parts.push(Part::Reasoning(ReasoningPart {
            summary: None,
            data: None,
            redacted: false,
            metadata: MetadataMap::from([(
                "openrouter.reasoning_details".into(),
                reasoning_details.clone(),
            )]),
        }));
    }

    if let Some(content) = &message.content {
        parts.extend(content_to_parts(content)?);
    }

    for tool_call in &message.tool_calls {
        parts.push(Part::ToolCall(ToolCallPart {
            id: tool_call.id.clone().into(),
            name: tool_call.function.name.clone(),
            input: parse_tool_arguments(&tool_call.function.arguments)?,
            metadata: MetadataMap::new(),
        }));
    }

    Ok(parts)
}

fn content_to_parts(content: &ResponseContent) -> Result<Vec<Part>, OpenRouterError> {
    match content {
        ResponseContent::Text(text) => Ok(vec![Part::Text(TextPart {
            text: text.clone(),
            metadata: MetadataMap::new(),
        })]),
        ResponseContent::Parts(parts) => {
            let mut normalized = Vec::new();
            for part in parts {
                match part.kind.as_str() {
                    "text" => {
                        if let Some(text) = &part.text {
                            normalized.push(Part::Text(TextPart {
                                text: text.clone(),
                                metadata: MetadataMap::new(),
                            }));
                        }
                    }
                    "image_url" => {
                        if let Some(image_url) = &part.image_url {
                            normalized.push(Part::Media(MediaPart {
                                modality: Modality::Image,
                                mime_type: "image/*".into(),
                                data: DataRef::Uri(image_url.url.clone()),
                                metadata: MetadataMap::new(),
                            }));
                        }
                    }
                    "input_audio" => {
                        if let Some(audio) = &part.input_audio {
                            normalized.push(Part::Media(MediaPart {
                                modality: Modality::Audio,
                                mime_type: format!("audio/{}", audio.format),
                                data: DataRef::InlineText(format!(
                                    "data:audio/{};base64,{}",
                                    audio.format, audio.data
                                )),
                                metadata: MetadataMap::new(),
                            }));
                        }
                    }
                    other => {
                        normalized.push(Part::Custom(agentkit_core::CustomPart {
                            kind: format!("openrouter.content.{other}"),
                            data: None,
                            value: Some(
                                serde_json::to_value(part).map_err(OpenRouterError::Serialize)?,
                            ),
                            metadata: MetadataMap::new(),
                        }));
                    }
                }
            }
            Ok(normalized)
        }
    }
}

fn parse_tool_arguments(arguments: &str) -> Result<Value, OpenRouterError> {
    serde_json::from_str(arguments).map_err(|error| {
        OpenRouterError::Protocol(format!(
            "invalid tool arguments JSON {arguments:?}: {error}"
        ))
    })
}

fn map_usage(usage: Option<ResponseUsage>) -> Option<Usage> {
    usage.map(|usage| Usage {
        tokens: Some(TokenUsage {
            input_tokens: usage.prompt_tokens.unwrap_or_default(),
            output_tokens: usage.completion_tokens.unwrap_or_default(),
            reasoning_tokens: usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens),
            cached_input_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens),
        }),
        cost: usage.cost.map(|amount| CostUsage {
            amount,
            currency: "USD".into(),
            provider_amount: None,
        }),
        metadata: MetadataMap::new(),
    })
}

fn response_metadata(model: &Option<String>, message: &ResponseMessage) -> MetadataMap {
    let mut metadata = MetadataMap::new();
    if let Some(model) = model {
        metadata.insert("openrouter.model".into(), Value::String(model.clone()));
    }
    if let Some(refusal) = &message.refusal {
        metadata.insert("openrouter.refusal".into(), Value::String(refusal.clone()));
    }
    metadata
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Completed,
        Some("tool_calls") => FinishReason::ToolCall,
        Some("length") => FinishReason::MaxTokens,
        Some("content_filter") => FinishReason::Blocked,
        Some("cancelled") => FinishReason::Cancelled,
        Some(other) => FinishReason::Other(other.into()),
        None => FinishReason::Completed,
    }
}

fn as_loop_error(error: OpenRouterError) -> LoopError {
    LoopError::Provider(error.to_string())
}

/// Errors produced by the OpenRouter adapter.
///
/// Covers configuration problems, unsupported content types, HTTP failures,
/// and protocol-level issues in the OpenRouter response.
#[derive(Debug, Error)]
pub enum OpenRouterError {
    /// A required environment variable is not set.
    #[error("missing environment variable {0}")]
    MissingEnv(&'static str),

    /// A configuration value could not be parsed or is otherwise invalid.
    #[error("invalid OpenRouter configuration: {0}")]
    InvalidConfig(String),

    /// The underlying HTTP client could not be constructed.
    #[error("failed to build HTTP client: {0}")]
    HttpClient(reqwest::Error),

    /// A transcript part is not supported for the given message role.
    #[error("unsupported item part {part_kind:?} for role {role:?}")]
    UnsupportedPart {
        /// The role of the item that contained the unsupported part.
        role: ItemKind,
        /// The kind of part that is not supported.
        part_kind: PartKind,
    },

    /// A media modality (e.g. video) is not supported by the adapter.
    #[error("unsupported modality {0:?}")]
    UnsupportedModality(Modality),

    /// A data reference type cannot be sent to the OpenRouter API.
    #[error("unsupported data reference: {0}")]
    UnsupportedDataRef(String),

    /// The transcript structure is invalid for conversion to chat messages.
    #[error("invalid transcript: {0}")]
    InvalidTranscript(String),

    /// The OpenRouter response could not be interpreted.
    #[error("OpenRouter protocol error: {0}")]
    Protocol(String),

    /// A value could not be serialized to JSON for the request body.
    #[error("failed to serialize request data: {0}")]
    Serialize(serde_json::Error),
}

trait PartExt {
    fn kind(&self) -> PartKind;
}

impl PartExt for Part {
    fn kind(&self) -> PartKind {
        match self {
            Part::Text(_) => PartKind::Text,
            Part::Media(_) => PartKind::Media,
            Part::File(_) => PartKind::File,
            Part::Structured(_) => PartKind::Structured,
            Part::Reasoning(_) => PartKind::Reasoning,
            Part::ToolCall(_) => PartKind::ToolCall,
            Part::ToolResult(_) => PartKind::ToolResult,
            Part::Custom(_) => PartKind::Custom,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorResponse {
    error: OpenRouterErrorBody,
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorBody {
    message: String,
    code: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<ResponseChoice>,
    usage: Option<ResponseUsage>,
}

#[derive(Debug, Deserialize)]
struct ResponseChoice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<ResponseContent>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
    reasoning: Option<String>,
    reasoning_details: Option<Value>,
    refusal: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponseContent {
    Text(String),
    Parts(Vec<ResponseContentPart>),
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct ResponseContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    image_url: Option<ResponseImageUrl>,
    input_audio: Option<ResponseInputAudio>,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct ResponseImageUrl {
    url: String,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct ResponseInputAudio {
    data: String,
    format: String,
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    id: String,
    function: ResponseFunction,
}

#[derive(Debug, Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ResponseUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    cost: Option<f64>,
    prompt_tokens_details: Option<ResponsePromptTokenDetails>,
    completion_tokens_details: Option<ResponseCompletionTokenDetails>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ResponsePromptTokenDetails {
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct ResponseCompletionTokenDetails {
    reasoning_tokens: Option<u64>,
}
