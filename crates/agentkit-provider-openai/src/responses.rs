//! Transport-neutral OpenAI Responses API adapter.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

use agentkit_core::{
    DataRef, Delta, FinishReason, Item, ItemKind, MediaPart, MessageId, MetadataMap, Modality,
    Part, PartId, PartKind, ReasoningPart, TextPart, TokenUsage, ToolCallPart, ToolOutput,
    TurnCancellation, Usage,
};
use agentkit_http::{
    Authentication, AuthenticationAttempt, AuthenticationProvider, HeaderMap, HeaderValue, Http,
    HttpError, ResilienceConfig, StatusCode, TruncatedStreamDetector, is_retryable_status,
    next_body_chunk, sleep,
};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult,
    PromptCacheMode, PromptCacheStrategy, SessionConfig, TurnRequest, set_provider_finish_reasons,
};
use async_trait::async_trait;
use futures_util::future::{Either, select};
use serde_json::{Map, Value, json};
use thiserror::Error;
use zeroize::{Zeroize, Zeroizing};

const PUBLIC_ENDPOINT: &str = "https://api.openai.com/v1/responses";
const PRIVATE_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const DEFAULT_MAX_ATTEMPT_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_WIRE_BYTES: usize = 64 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;
const MAX_ITEMS: usize = 100_000;
const CONTINUATION_METADATA: &str = "openai.responses.continuation.v1";
const CONTINUATION_SCHEMA_VERSION: u64 = 3;
const LEGACY_CONTINUATION_METADATA: &str = "openai.subscription.v1";
const LEGACY_CONTINUATION_SCHEMA_VERSION: u64 = 1;
const GENERATED_IMAGE_METADATA: &str = "openai.subscription.generated_image.v1";
const X_CODEX_TURN_STATE: &str = "x-codex-turn-state";
const MAX_CACHE_KEY_BYTES: usize = 256;

/// Selects the public Responses API or ChatGPT's private Codex-shaped profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenAIResponsesProfile {
    /// `https://api.openai.com/v1/responses`, including public request fields.
    Public,
    /// `https://chatgpt.com/backend-api/codex/responses`, with its narrower field policy.
    ChatGptPrivate,
}

/// Bounds serialized requests and streamed response bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenAIResponsesLimits {
    /// Maximum serialized JSON request body size.
    pub max_request_bytes: usize,
    /// Maximum wire bytes accepted from one response attempt.
    pub max_attempt_bytes: usize,
    /// Maximum aggregate wire bytes accepted across all attempts for one logical turn.
    pub max_wire_bytes: usize,
}

impl Default for OpenAIResponsesLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_attempt_bytes: DEFAULT_MAX_ATTEMPT_BYTES,
            max_wire_bytes: DEFAULT_MAX_WIRE_BYTES,
        }
    }
}

type LegacyContinuationAuthenticator = Arc<dyn Fn(&Value, &str) -> bool + Send + Sync + 'static>;

/// Controls fields whose support differs between Responses deployments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAIResponsesRequestPolicy {
    /// Downgrade system/context items to developer messages. Required by the private profile.
    pub downgrade_system_to_developer: bool,
    /// Send `store`; both built-in profiles default to `false`.
    pub store: Option<bool>,
    /// Ask the service to return encrypted reasoning for safe continuation.
    pub include_encrypted_reasoning: bool,
    /// Send `max_output_tokens` when configured. The private profile rejects it.
    pub send_max_output_tokens: bool,
}

impl OpenAIResponsesRequestPolicy {
    pub fn public() -> Self {
        Self {
            downgrade_system_to_developer: false,
            store: Some(false),
            include_encrypted_reasoning: true,
            send_max_output_tokens: true,
        }
    }

    pub fn chatgpt_private() -> Self {
        Self {
            downgrade_system_to_developer: true,
            store: Some(false),
            include_encrypted_reasoning: true,
            send_max_output_tokens: false,
        }
    }
}

/// Configuration shared by public and private Responses endpoints.
#[derive(Clone)]
pub struct OpenAIResponsesConfig {
    pub model: String,
    pub endpoint: String,
    pub profile: OpenAIResponsesProfile,
    pub headers: HeaderMap,
    pub request_policy: OpenAIResponsesRequestPolicy,
    pub reasoning_effort: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub parallel_tool_calls: Option<bool>,
    authentication: Authentication,
    resilience: Option<ResilienceConfig>,
    limits: OpenAIResponsesLimits,
    user_agent: Option<String>,
    originator: Option<String>,
    legacy_continuation_authenticator: Option<LegacyContinuationAuthenticator>,
}

impl fmt::Debug for OpenAIResponsesConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAIResponsesConfig")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("profile", &self.profile)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("request_policy", &self.request_policy)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("authentication", &"<redacted>")
            .field("resilience", &self.resilience)
            .field("limits", &self.limits)
            .field("user_agent", &self.user_agent)
            .field("originator", &self.originator)
            .field(
                "legacy_continuation_authenticator",
                &self
                    .legacy_continuation_authenticator
                    .as_ref()
                    .map(|_| "<callback>"),
            )
            .finish()
    }
}

impl OpenAIResponsesConfig {
    /// Creates a public Responses configuration using bearer authentication.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::public(model, Authentication::bearer(api_key.into()))
    }

    /// Creates a public Responses configuration with injectable authentication.
    pub fn public(model: impl Into<String>, authentication: impl Into<Authentication>) -> Self {
        Self {
            model: model.into(),
            endpoint: PUBLIC_ENDPOINT.into(),
            profile: OpenAIResponsesProfile::Public,
            headers: HeaderMap::new(),
            request_policy: OpenAIResponsesRequestPolicy::public(),
            reasoning_effort: None,
            max_output_tokens: None,
            parallel_tool_calls: None,
            authentication: authentication.into(),
            resilience: None,
            limits: OpenAIResponsesLimits::default(),
            user_agent: None,
            originator: None,
            legacy_continuation_authenticator: None,
        }
    }

    /// Creates a private ChatGPT Codex Responses configuration.
    pub fn chatgpt_private(
        model: impl Into<String>,
        authentication: impl Into<Authentication>,
    ) -> Self {
        Self {
            model: model.into(),
            endpoint: PRIVATE_ENDPOINT.into(),
            profile: OpenAIResponsesProfile::ChatGptPrivate,
            headers: HeaderMap::new(),
            request_policy: OpenAIResponsesRequestPolicy::chatgpt_private(),
            reasoning_effort: None,
            max_output_tokens: None,
            parallel_tool_calls: Some(true),
            authentication: authentication.into(),
            resilience: None,
            limits: OpenAIResponsesLimits::default(),
            user_agent: None,
            originator: None,
            legacy_continuation_authenticator: None,
        }
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Replaces non-authentication request headers. Authentication headers are applied last.
    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        self.authentication = authentication.into();
        self
    }

    /// Uses a custom refresh-capable authentication provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(self, provider: P) -> Self {
        self.with_authentication(Authentication::new(provider))
    }

    pub fn with_request_policy(mut self, policy: OpenAIResponsesRequestPolicy) -> Self {
        self.request_policy = policy;
        self
    }

    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn with_max_output_tokens(mut self, value: u32) -> Self {
        self.max_output_tokens = Some(value);
        self
    }

    pub fn with_parallel_tool_calls(mut self, value: bool) -> Self {
        self.parallel_tool_calls = Some(value);
        self
    }

    /// Opts into retries and stream/attempt timeouts. `None` means no transient retry.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        self.resilience = Some(resilience);
        self
    }

    /// Overrides serialized request and response wire-byte limits.
    pub fn with_limits(mut self, limits: OpenAIResponsesLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Overrides the adapter's default HTTP user agent.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Sets or overrides the Responses `originator` header.
    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = Some(originator.into());
        self
    }

    /// Enables safe replay of Kit's legacy `openai.subscription.v1` metadata.
    ///
    /// The callback receives the legacy `account_binding` object and the current
    /// authentication binding. The adapter validates every other schema, model,
    /// session, item, and kind field itself and writes only current metadata.
    pub fn with_legacy_subscription_continuation_authenticator<F>(
        mut self,
        authenticator: F,
    ) -> Self
    where
        F: Fn(&Value, &str) -> bool + Send + Sync + 'static,
    {
        self.legacy_continuation_authenticator = Some(Arc::new(authenticator));
        self
    }

    /// Encodes only the transport-neutral JSON request.
    pub fn encode_request(&self, request: &TurnRequest) -> Result<Value, OpenAIResponsesError> {
        encode_request(self, request)
    }
}

/// Errors produced while configuring or encoding a Responses request.
#[derive(Debug, Error)]
pub enum OpenAIResponsesError {
    #[error("failed to create HTTP client: {0}")]
    HttpClient(#[source] HttpError),
    #[error("invalid Responses request: {0}")]
    InvalidRequest(String),
    #[error("Responses protocol error: {0}")]
    Protocol(String),
    #[error("Responses request serialization failed: {0}")]
    Serialize(#[source] serde_json::Error),
}

/// OpenAI Responses adapter with live SSE delivery and capability-gated attempt replacement.
#[derive(Clone)]
pub struct OpenAIResponsesAdapter {
    client: Http,
    config: Arc<OpenAIResponsesConfig>,
}

impl OpenAIResponsesAdapter {
    pub fn new(config: OpenAIResponsesConfig) -> Result<Self, OpenAIResponsesError> {
        let client = reqwest::Client::builder()
            .build()
            .map(Http::new)
            .map_err(|error| OpenAIResponsesError::HttpClient(HttpError::request(error)))?;
        Ok(Self::with_client(config, client))
    }

    /// Creates an adapter over an arbitrary AgentKit HTTP transport.
    pub fn with_client(config: OpenAIResponsesConfig, client: Http) -> Self {
        Self {
            client,
            config: Arc::new(config),
        }
    }

    /// Overrides authentication after constructing the adapter.
    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        Arc::make_mut(&mut self.config).authentication = authentication.into();
        self
    }

    /// Overrides authentication with a custom refresh-capable provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(self, provider: P) -> Self {
        self.with_authentication(Authentication::new(provider))
    }

    /// Enables retries and stream/attempt timeouts after constructing the adapter.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        Arc::make_mut(&mut self.config).resilience = Some(resilience);
        self
    }
}

#[async_trait]
impl ModelAdapter for OpenAIResponsesAdapter {
    type Session = OpenAIResponsesSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(OpenAIResponsesSession {
            client: self.client.clone(),
            config: self.config.clone(),
            session: config,
        })
    }

    fn provider_name(&self) -> Option<&str> {
        Some("openai")
    }
}

/// Active Responses session.
pub struct OpenAIResponsesSession {
    client: Http,
    config: Arc<OpenAIResponsesConfig>,
    session: SessionConfig,
}

#[async_trait]
impl ModelSession for OpenAIResponsesSession {
    type Turn = OpenAIResponsesTurn;

    async fn begin_turn(
        &mut self,
        request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Self::Turn, LoopError> {
        if cancelled(cancellation.as_ref()) {
            return Err(LoopError::Cancelled);
        }
        // One logical deadline starts before encoding and initial authentication.
        let deadline = self
            .config
            .resilience
            .as_ref()
            .map(|config| LogicalDeadline::new(config.retry_budget));
        deadline_remaining(deadline.as_ref()).map_err(http_loop_error)?;
        let auth_timeout = self
            .config
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout);
        let auth = cancellable(
            run_bounded_http(
                self.config.authentication.authenticate(None),
                auth_timeout,
                deadline.as_ref(),
                "OpenAI authentication",
            ),
            cancellation.as_ref(),
        )
        .await?
        .map_err(http_loop_error)?;
        let mut value = encode_request_bound(&self.config, &request, auth.binding())
            .map_err(|error| LoopError::Provider(error.to_string()))?;
        // Serialize once. Every status, transport, and stream retry reuses these exact bytes.
        let body = agentkit_http::Bytes::from_owner(Zeroizing::new(
            serde_json::to_vec(&value)
                .map_err(OpenAIResponsesError::Serialize)
                .map_err(|error| LoopError::Provider(error.to_string()))?,
        ));
        zeroize_encrypted_content(&mut value);
        let idempotency_key = stable_idempotency_key(
            &self.session.session_id.to_string(),
            &request.turn_id.to_string(),
            &body,
        );
        let replacement_enabled = agentkit_loop::response_attempt::enabled(&self.session);
        OpenAIResponsesTurn::open(
            ResponsesRequestContext {
                client: self.client.clone(),
                config: self.config.clone(),
                body,
                idempotency_key,
                session_id: request.session_id.to_string(),
                turn_state: Arc::new(Mutex::new(None)),
                auth,
                deadline,
                retries: 0,
                refreshed: false,
                wire_bytes: 0,
            },
            replacement_enabled,
            cancellation.as_ref(),
        )
        .await
    }

    fn model_name(&self) -> Option<&str> {
        Some(&self.config.model)
    }

    fn provider_name(&self) -> Option<&str> {
        Some("openai")
    }
}

/// Live-streaming Responses turn.
pub struct OpenAIResponsesTurn {
    context: ResponsesRequestContext,
    attempt: Option<LiveAttempt>,
    replacement_enabled: bool,
    attempt_output_emitted: bool,
    pending_reopen: bool,
    pending_delay: Duration,
    finished: bool,
}

#[async_trait]
impl ModelTurn for OpenAIResponsesTurn {
    async fn next_event(
        &mut self,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        self.next_event_inner(cancellation.as_ref()).await
    }
}

impl OpenAIResponsesTurn {
    async fn open(
        context: ResponsesRequestContext,
        replacement_enabled: bool,
        cancellation: Option<&TurnCancellation>,
    ) -> Result<Self, LoopError> {
        let mut turn = Self {
            context,
            attempt: None,
            replacement_enabled,
            attempt_output_emitted: false,
            pending_reopen: true,
            pending_delay: Duration::ZERO,
            finished: false,
        };
        turn.reopen(cancellation).await?;
        Ok(turn)
    }

    async fn reopen(&mut self, cancellation: Option<&TurnCancellation>) -> Result<(), LoopError> {
        if !self.pending_delay.is_zero() {
            let delay = self.pending_delay;
            self.pending_delay = Duration::ZERO;
            cancellable(
                run_bounded_http(
                    async {
                        sleep(delay).await;
                        Ok(())
                    },
                    None,
                    self.context.deadline.as_ref(),
                    "retry backoff",
                ),
                cancellation,
            )
            .await?
            .map_err(http_loop_error)?;
        }
        self.attempt = Some(open_live_attempt(&mut self.context, cancellation).await?);
        self.pending_reopen = false;
        Ok(())
    }

    async fn next_event_inner(
        &mut self,
        cancellation: Option<&TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        loop {
            if cancelled(cancellation) {
                return Err(LoopError::Cancelled);
            }
            if self.pending_reopen {
                self.reopen(cancellation).await?;
            }
            let expired_attempt_timeout = self
                .attempt
                .as_ref()
                .and_then(|attempt| attempt.deadline.as_ref())
                .filter(|deadline| deadline.started_at.elapsed() >= deadline.budget)
                .map(|deadline| deadline.budget);
            if expired_attempt_timeout.is_none()
                && let Some(event) = self.attempt.as_mut().and_then(LiveAttempt::pop_event)
            {
                self.attempt_output_emitted = true;
                if matches!(event, ModelTurnEvent::Finished(_)) {
                    self.finished = true;
                }
                return Ok(Some(event));
            }
            if self.finished {
                return Ok(None);
            }

            let pending = if let Some(timeout) = expired_attempt_timeout {
                Err(attempt_timeout_failure(timeout))
            } else {
                self.attempt
                    .as_mut()
                    .expect("live attempt is open")
                    .decoder
                    .process_pending()
            };
            let result = if let Err(failure) = pending {
                Err(failure)
            } else if self
                .attempt
                .as_ref()
                .expect("live attempt is open")
                .decoder
                .peek_event()
                .is_some_and(|event| !matches!(event, ModelTurnEvent::Finished(_)))
            {
                continue;
            } else if self.attempt.as_ref().expect("live attempt is open").eof {
                let attempt = self.attempt.as_mut().expect("live attempt is open");
                let finished = attempt.decoder.finish_live();
                if finished.is_ok() {
                    attempt.closed = true;
                }
                finished
            } else {
                let remaining =
                    deadline_remaining(self.context.deadline.as_ref()).map_err(http_loop_error)?;
                let idle_timeout = self
                    .context
                    .config
                    .resilience
                    .as_ref()
                    .and_then(|config| config.stream_idle_timeout);
                let attempt_remaining = self
                    .attempt
                    .as_ref()
                    .and_then(|attempt| attempt.deadline.as_ref())
                    .map(|deadline| {
                        deadline
                            .budget
                            .saturating_sub(deadline.started_at.elapsed())
                    });
                let timeout = [idle_timeout, remaining, attempt_remaining]
                    .into_iter()
                    .flatten()
                    .min();
                let chunk = {
                    let attempt = self.attempt.as_mut().expect("live attempt is open");
                    cancellable(next_body_chunk(&mut attempt.body, timeout), cancellation).await
                };
                match chunk {
                    Err(error) => Err(nonretryable(error)),
                    Ok(Err(error)) => {
                        if self.context.deadline.as_ref().is_some_and(|deadline| {
                            deadline.started_at.elapsed() >= deadline.budget
                        }) {
                            Err(nonretryable(http_loop_error(HttpError::Timeout {
                                operation: "logical request retry budget",
                                timeout: self
                                    .context
                                    .deadline
                                    .as_ref()
                                    .expect("checked logical deadline")
                                    .budget,
                            })))
                        } else if let Some(timeout) = self
                            .attempt
                            .as_ref()
                            .and_then(|attempt| attempt.deadline.as_ref())
                            .filter(|deadline| deadline.started_at.elapsed() >= deadline.budget)
                            .map(|deadline| deadline.budget)
                        {
                            Err(attempt_timeout_failure(timeout))
                        } else {
                            Err(transport_failure(error))
                        }
                    }
                    Ok(Ok(Some(chunk))) => {
                        if let Some(total) = self.context.wire_bytes.checked_add(chunk.len()) {
                            self.context.wire_bytes = total;
                            if total > self.context.config.limits.max_wire_bytes {
                                Err(protocol_failure(
                                    "Responses logical turn exceeds configured wire-byte limit",
                                ))
                            } else {
                                let attempt = self.attempt.as_mut().expect("live attempt is open");
                                attempt.truncated.observe(&chunk);
                                attempt.decoder.push(&chunk)
                            }
                        } else {
                            Err(protocol_failure("Responses wire-byte count overflowed"))
                        }
                    }
                    Ok(Ok(None)) => {
                        let attempt = self.attempt.as_mut().expect("live attempt is open");
                        attempt.eof = true;
                        attempt.truncated.finish().map_err(transport_failure)
                    }
                }
            };
            if let Err(failure) = result {
                if !failure.retryable
                    || self.context.retries
                        >= self
                            .context
                            .config
                            .resilience
                            .as_ref()
                            .map_or(0, |config| config.max_retries)
                {
                    return Err(*failure.error);
                }
                if self.attempt_output_emitted && !self.replacement_enabled {
                    return Err(*failure.error);
                }
                let delay = self
                    .context
                    .config
                    .resilience
                    .as_ref()
                    .expect("retry requires resilience")
                    .retry_delay(self.context.retries, failure.headers.as_ref());
                self.context.retries += 1;
                self.attempt = None;
                self.pending_reopen = true;
                self.pending_delay = delay;
                if self.attempt_output_emitted {
                    self.attempt_output_emitted = false;
                    return Ok(Some(agentkit_loop::response_attempt::marker_event()));
                }
            }
        }
    }
}

fn encode_request(
    config: &OpenAIResponsesConfig,
    request: &TurnRequest,
) -> Result<Value, OpenAIResponsesError> {
    encode_request_bound(config, request, None)
}

fn encode_request_bound(
    config: &OpenAIResponsesConfig,
    request: &TurnRequest,
    authentication_binding: Option<&str>,
) -> Result<Value, OpenAIResponsesError> {
    if request.transcript.len() > MAX_ITEMS || request.available_tools.len() > MAX_ITEMS {
        return Err(OpenAIResponsesError::InvalidRequest(
            "too many transcript items or tools".into(),
        ));
    }
    let session_id = request.session_id.to_string();
    let mut input = Vec::new();
    for item in &request.transcript {
        input.extend(encode_item(
            config,
            &session_id,
            authentication_binding,
            item,
        )?);
    }
    let tools = request
        .available_tools
        .iter()
        .map(|tool| {
            validate_tool_name(&tool.name.0)?;
            Ok(json!({
                "type": "function",
                "name": tool.name.0,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": false
            }))
        })
        .collect::<Result<Vec<_>, OpenAIResponsesError>>()?;
    let mut body = Map::new();
    body.insert("model".into(), Value::String(config.model.clone()));
    body.insert("input".into(), Value::Array(input));
    body.insert("tools".into(), Value::Array(tools));
    body.insert("tool_choice".into(), Value::String("auto".into()));
    body.insert("stream".into(), Value::Bool(true));
    if let Some(value) = config.request_policy.store {
        body.insert("store".into(), Value::Bool(value));
    }
    if config.request_policy.include_encrypted_reasoning {
        body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    }
    if config.reasoning_effort.is_some() || config.request_policy.include_encrypted_reasoning {
        let mut reasoning = Map::new();
        reasoning.insert("summary".into(), Value::String("auto".into()));
        if let Some(effort) = &config.reasoning_effort {
            reasoning.insert("effort".into(), Value::String(effort.clone()));
        }
        body.insert("reasoning".into(), Value::Object(reasoning));
    }
    if let Some(value) = config.parallel_tool_calls {
        body.insert("parallel_tool_calls".into(), Value::Bool(value));
    }
    if config.request_policy.send_max_output_tokens
        && let Some(value) = config.max_output_tokens
    {
        body.insert("max_output_tokens".into(), Value::from(value));
    }
    apply_prompt_cache(&mut body, request, config.profile)?;
    let value = Value::Object(body);
    let encoded =
        Zeroizing::new(serde_json::to_vec(&value).map_err(OpenAIResponsesError::Serialize)?);
    if encoded.len() > config.limits.max_request_bytes {
        return Err(invalid_request(format!(
            "Responses serialized request exceeds {} bytes",
            config.limits.max_request_bytes
        )));
    }
    Ok(value)
}

fn encode_item(
    config: &OpenAIResponsesConfig,
    session_id: &str,
    authentication_binding: Option<&str>,
    item: &Item,
) -> Result<Vec<Value>, OpenAIResponsesError> {
    let role = match item.kind {
        ItemKind::System if !config.request_policy.downgrade_system_to_developer => "system",
        ItemKind::System | ItemKind::Developer | ItemKind::Context => "developer",
        ItemKind::User => "user",
        ItemKind::Assistant => "assistant",
        ItemKind::Tool => "tool",
        ItemKind::Notification => "user",
    };
    if matches!(
        item.kind,
        ItemKind::System | ItemKind::Developer | ItemKind::Context | ItemKind::Notification
    ) {
        let mut text = stringify_message_parts(&item.parts, item.kind)?;
        if item.kind == ItemKind::Context {
            text = format!("Context (not higher-priority instructions):\n{text}");
        } else if item.kind == ItemKind::Notification {
            text = wrap_notification(&text);
        }
        if text.is_empty() {
            return Ok(Vec::new());
        }
        return Ok(vec![json!({
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}]
        })]);
    }
    let mut output = Vec::new();
    let mut content = Vec::new();
    for part in &item.parts {
        validate_part_role(item.kind, part)?;
        match part {
            Part::Text(text) => {
                let text = match item.kind {
                    ItemKind::Notification => {
                        format!("<system-reminder>{}</system-reminder>", text.text)
                    }
                    ItemKind::Context => {
                        format!("Context (not higher-priority instructions):\n{}", text.text)
                    }
                    _ => text.text.clone(),
                };
                content.push(json!({
                    "type": if role == "assistant" { "output_text" } else { "input_text" },
                    "text": text
                }));
            }
            Part::Structured(value) => {
                let text =
                    serde_json::to_string(&value.value).map_err(OpenAIResponsesError::Serialize)?;
                content.push(json!({
                    "type": if role == "assistant" { "output_text" } else { "input_text" },
                    "text": text
                }));
            }
            Part::ToolCall(call) => {
                validate_tool_name(&call.name)?;
                let continuation = continuation_from_metadata(
                    &call.metadata,
                    config,
                    session_id,
                    authentication_binding,
                    "function_call",
                    false,
                )?;
                let mut value = json!({
                    "type": "function_call",
                    "call_id": call.id.0,
                    "name": call.name,
                    "arguments": serde_json::to_string(&call.input).map_err(OpenAIResponsesError::Serialize)?
                });
                if let Some(continuation) = continuation {
                    value["id"] = Value::String(continuation.item_id.to_owned());
                }
                output.push(value);
            }
            Part::ToolResult(result) => output.push(json!({
                "type": "function_call_output",
                "call_id": result.call_id.0,
                "output": encode_tool_output(&result.output, config.profile)?
            })),
            Part::Reasoning(reasoning) => {
                if let Some(continuation) = continuation_from_metadata(
                    &reasoning.metadata,
                    config,
                    session_id,
                    authentication_binding,
                    "reasoning",
                    true,
                )? {
                    output.push(json!({
                        "id": continuation.item_id,
                        "type": "reasoning",
                        "summary": [],
                        "encrypted_content": continuation.encrypted_content.expect("validated encrypted reasoning")
                    }));
                }
                // A readable summary without encrypted continuation state is deliberately not
                // converted into ordinary assistant text.
            }
            Part::Media(media) => {
                if role == "assistant" {
                    let generated =
                        generated_image_item(media, config, session_id, authentication_binding)?;
                    if let Some(generated) = generated {
                        output.push(generated);
                    } else {
                        return Err(invalid_request(
                            "assistant media is not a persisted Responses generated image",
                        ));
                    }
                } else if role == "tool" {
                    return Err(invalid_request(
                        "tool item media must be carried by a tool result",
                    ));
                } else {
                    content.push(encode_media(media, config.profile)?);
                }
            }
            Part::File(_) | Part::Custom(_) => {
                return Err(invalid_request(
                    "Responses transcript contains unsupported content",
                ));
            }
        }
    }
    if !content.is_empty() && role != "tool" {
        output.insert(
            0,
            json!({"type": "message", "role": role, "content": content}),
        );
    }
    Ok(output)
}

fn validate_tool_name(name: &str) -> Result<(), OpenAIResponsesError> {
    if name.is_empty()
        || name.len() > 64
        || !name.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
    {
        return Err(invalid_request(format!(
            "invalid OpenAI tool name `{name}`"
        )));
    }
    Ok(())
}

fn validate_part_role(role: ItemKind, part: &Part) -> Result<(), OpenAIResponsesError> {
    let supported = match role {
        ItemKind::System | ItemKind::Developer | ItemKind::Context | ItemKind::Notification => {
            matches!(
                part,
                Part::Text(_) | Part::Structured(_) | Part::Reasoning(_)
            )
        }
        ItemKind::User => matches!(part, Part::Text(_) | Part::Structured(_) | Part::Media(_)),
        ItemKind::Assistant => matches!(
            part,
            Part::Text(_)
                | Part::Structured(_)
                | Part::ToolCall(_)
                | Part::Reasoning(_)
                | Part::Media(_)
        ),
        ItemKind::Tool => matches!(part, Part::ToolResult(_)),
    };
    if supported {
        Ok(())
    } else {
        Err(unsupported_part(role, part))
    }
}

fn unsupported_part(role: ItemKind, part: &Part) -> OpenAIResponsesError {
    invalid_request(format!(
        "Responses does not support {} content on {role:?} items",
        match part {
            Part::Text(_) => "text",
            Part::Media(_) => "media",
            Part::File(_) => "file",
            Part::Structured(_) => "structured",
            Part::Reasoning(_) => "reasoning",
            Part::ToolCall(_) => "tool-call",
            Part::ToolResult(_) => "tool-result",
            Part::Custom(_) => "custom",
        }
    ))
}

fn stringify_message_parts(parts: &[Part], role: ItemKind) -> Result<String, OpenAIResponsesError> {
    let mut segments = Vec::new();
    for part in parts {
        validate_part_role(role, part)?;
        match part {
            Part::Text(text) => segments.push(text.text.clone()),
            Part::Structured(value) => segments.push(
                serde_json::to_string_pretty(&value.value)
                    .map_err(OpenAIResponsesError::Serialize)?,
            ),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    segments.push(summary.clone());
                }
            }
            _ => unreachable!("validated string content"),
        }
    }
    Ok(segments.join("\n\n"))
}

fn wrap_notification(text: &str) -> String {
    format!("<system-reminder>\n{text}\n</system-reminder>")
}

fn apply_prompt_cache(
    body: &mut Map<String, Value>,
    request: &TurnRequest,
    profile: OpenAIResponsesProfile,
) -> Result<(), OpenAIResponsesError> {
    let Some(cache) = &request.cache else {
        return Ok(());
    };
    if matches!(cache.mode, PromptCacheMode::Disabled) {
        return Ok(());
    }
    if matches!(cache.strategy, PromptCacheStrategy::Explicit { .. })
        && matches!(cache.mode, PromptCacheMode::Required)
    {
        return Err(invalid_request(
            "Responses does not support required explicit cache breakpoints",
        ));
    }
    if let Some(retention) = cache.retention
        && profile == OpenAIResponsesProfile::Public
    {
        body.insert(
            "prompt_cache_retention".into(),
            Value::String(crate::prompt_cache_retention_value(retention).into()),
        );
    }
    if let Some(key) = &cache.key {
        if key.is_empty() || key.len() > MAX_CACHE_KEY_BYTES || !key.is_ascii() {
            return Err(invalid_request(
                "Responses prompt cache key is outside canonical bounds",
            ));
        }
        body.insert("prompt_cache_key".into(), Value::String(key.clone()));
    }
    Ok(())
}

fn encode_media(
    media: &MediaPart,
    profile: OpenAIResponsesProfile,
) -> Result<Value, OpenAIResponsesError> {
    let expected = match media.modality {
        Modality::Image => "image/",
        Modality::Audio if profile == OpenAIResponsesProfile::ChatGptPrivate => "audio/",
        Modality::Audio => {
            return Err(invalid_request(
                "public Responses audio input is not supported by this adapter",
            ));
        }
        Modality::Video | Modality::Binary => {
            return Err(invalid_request(
                "Responses supports only image and private-profile audio media input",
            ));
        }
    };
    if !media.mime_type.starts_with(expected) || media.mime_type.contains(['\r', '\n', ';', ',']) {
        return Err(invalid_request("Responses media has an invalid MIME type"));
    }
    let url = media_data_url(media)?;
    Ok(match media.modality {
        Modality::Image => {
            json!({"type": "input_image", "image_url": url, "detail": "high"})
        }
        Modality::Audio => json!({"type": "input_audio", "audio_url": url}),
        Modality::Video | Modality::Binary => unreachable!("rejected above"),
    })
}

fn media_data_url(media: &MediaPart) -> Result<String, OpenAIResponsesError> {
    match &media.data {
        DataRef::InlineBytes(bytes) => Ok(format!(
            "data:{};base64,{}",
            media.mime_type,
            STANDARD.encode(bytes)
        )),
        DataRef::InlineText(text) if text.starts_with("data:") => {
            validate_data_url(text, &media.mime_type)?;
            Ok(text.clone())
        }
        DataRef::InlineText(text) => {
            validate_base64(text, "inline media")?;
            Ok(format!("data:{};base64,{text}", media.mime_type))
        }
        DataRef::Uri(uri) if uri.starts_with("data:") => {
            validate_data_url(uri, &media.mime_type)?;
            Ok(uri.clone())
        }
        DataRef::Uri(uri)
            if media.modality == Modality::Image
                && uri.len() <= MAX_TEXT_BYTES
                && reqwest::Url::parse(uri)
                    .is_ok_and(|url| matches!(url.scheme(), "http" | "https")) =>
        {
            Ok(uri.clone())
        }
        DataRef::Uri(_) => Err(invalid_request(
            "Responses cannot read this media URI; use inline bytes or an HTTP(S) image URL",
        )),
        DataRef::Handle(_) => Err(invalid_request(
            "Responses cannot resolve media handles; use inline bytes",
        )),
    }
}

fn validate_data_url(value: &str, mime_type: &str) -> Result<(), OpenAIResponsesError> {
    let payload = value
        .strip_prefix(&format!("data:{mime_type};base64,"))
        .filter(|payload| !payload.is_empty())
        .ok_or_else(|| invalid_request("Responses media data URL is not canonical base64"))?;
    validate_base64(payload, "media data URL")
}

fn validate_base64(value: &str, field: &str) -> Result<(), OpenAIResponsesError> {
    let decoded = STANDARD
        .decode(value)
        .map_err(|_| invalid_request(format!("Responses {field} is not valid base64")))?;
    if decoded.is_empty() || STANDARD.encode(&decoded) != value {
        return Err(invalid_request(format!(
            "Responses {field} is not canonical base64"
        )));
    }
    Ok(())
}

fn generated_image_item(
    media: &MediaPart,
    config: &OpenAIResponsesConfig,
    session_id: &str,
    authentication_binding: Option<&str>,
) -> Result<Option<Value>, OpenAIResponsesError> {
    let Some(metadata) = media.metadata.get(GENERATED_IMAGE_METADATA) else {
        return Ok(None);
    };
    let metadata = metadata
        .as_object()
        .filter(|metadata| (2..=3).contains(&metadata.len()))
        .ok_or_else(|| protocol_error("generated image metadata is malformed"))?;
    let item_id = bounded_metadata_string(metadata.get("item_id"), "generated image item_id")?;
    if metadata.get("status").and_then(Value::as_str) != Some("completed")
        || media.modality != Modality::Image
        || media.mime_type != "image/png"
    {
        return Err(protocol_error("generated image metadata is invalid"));
    }
    let Some(continuation) = continuation_from_metadata(
        &media.metadata,
        config,
        session_id,
        authentication_binding,
        "image_generation_call",
        false,
    )?
    else {
        return Ok(None);
    };
    if continuation.item_id != item_id {
        return Err(protocol_error(
            "generated image continuation item binding is invalid",
        ));
    }
    let revised_prompt = metadata
        .get("revised_prompt")
        .filter(|value| !value.is_null())
        .map(|value| bounded_metadata_string(Some(value), "generated image revised prompt"))
        .transpose()?;
    let result = match &media.data {
        DataRef::InlineBytes(bytes) if !bytes.is_empty() => STANDARD.encode(bytes),
        DataRef::InlineText(text) if !text.starts_with("data:") => {
            validate_base64(text, "persisted generated image result")?;
            text.clone()
        }
        DataRef::InlineText(text) => {
            validate_data_url(text, "image/png")?;
            text.split_once(',')
                .map(|(_, payload)| payload.to_owned())
                .ok_or_else(|| protocol_error("persisted generated image data URL is malformed"))?
        }
        DataRef::Uri(_) | DataRef::Handle(_) | DataRef::InlineBytes(_) => {
            return Err(invalid_request(
                "Responses cannot replay a generated image without inline bytes",
            ));
        }
    };
    Ok(Some(json!({
        "id": item_id,
        "type": "image_generation_call",
        "status": "completed",
        "revised_prompt": revised_prompt,
        "result": result,
    })))
}

struct Continuation<'a> {
    item_id: &'a str,
    encrypted_content: Option<&'a str>,
}

fn continuation_from_metadata<'a>(
    metadata: &'a MetadataMap,
    config: &OpenAIResponsesConfig,
    session_id: &str,
    authentication_binding: Option<&str>,
    expected_kind: &str,
    encrypted_required: bool,
) -> Result<Option<Continuation<'a>>, OpenAIResponsesError> {
    let Some(raw) = metadata.get(CONTINUATION_METADATA) else {
        return legacy_continuation_from_metadata(
            metadata,
            config,
            session_id,
            authentication_binding,
            expected_kind,
            encrypted_required,
        );
    };
    let object = raw
        .as_object()
        .ok_or_else(|| protocol_error("Responses continuation metadata is not an object"))?;
    let expected_len = if encrypted_required { 7 } else { 6 };
    if object.len() != expected_len
        || object.get("schema_version").and_then(Value::as_u64) != Some(CONTINUATION_SCHEMA_VERSION)
        || object.get("kind").and_then(Value::as_str) != Some(expected_kind)
        || object.get("model").and_then(Value::as_str) != Some(&*config.model)
        || object.get("session_id").and_then(Value::as_str) != Some(session_id)
    {
        return Err(protocol_error(
            "Responses continuation metadata binding is invalid",
        ));
    }
    let metadata_binding = bounded_metadata_string(
        object.get("authentication_binding"),
        "authentication binding",
    )?;
    let item_id = bounded_metadata_string(object.get("item_id"), "item_id")?;
    let encrypted_content = object.get("encrypted_content").and_then(Value::as_str);
    if encrypted_required
        && encrypted_content.is_none_or(|value| value.is_empty() || value.len() > MAX_TEXT_BYTES)
    {
        return Err(protocol_error(
            "Responses encrypted continuation is invalid",
        ));
    }
    if authentication_binding != Some(metadata_binding) {
        return Ok(None);
    }
    Ok(Some(Continuation {
        item_id,
        encrypted_content,
    }))
}

fn legacy_continuation_from_metadata<'a>(
    metadata: &'a MetadataMap,
    config: &OpenAIResponsesConfig,
    session_id: &str,
    authentication_binding: Option<&str>,
    expected_kind: &str,
    encrypted_required: bool,
) -> Result<Option<Continuation<'a>>, OpenAIResponsesError> {
    let Some(raw) = metadata.get(LEGACY_CONTINUATION_METADATA) else {
        return Ok(None);
    };
    if config.profile != OpenAIResponsesProfile::ChatGptPrivate {
        return Ok(None);
    }
    let Some(authentication_binding) = authentication_binding else {
        return Ok(None);
    };
    let Some(authenticator) = &config.legacy_continuation_authenticator else {
        return Ok(None);
    };
    let object = raw
        .as_object()
        .ok_or_else(|| protocol_error("legacy Responses continuation metadata is not an object"))?;
    let expected_len = if encrypted_required { 9 } else { 8 };
    if object.len() != expected_len
        || object.get("schema_version").and_then(Value::as_u64)
            != Some(LEGACY_CONTINUATION_SCHEMA_VERSION)
        || object.get("kind").and_then(Value::as_str) != Some(expected_kind)
        || object.get("model").and_then(Value::as_str) != Some(&*config.model)
        || object.get("session_id").and_then(Value::as_str) != Some(session_id)
        || object.get("output_index").and_then(Value::as_u64).is_none()
    {
        return Err(protocol_error(
            "legacy Responses continuation metadata binding is invalid",
        ));
    }
    let account_binding = object
        .get("account_binding")
        .and_then(Value::as_object)
        .filter(|binding| binding.len() == 2)
        .ok_or_else(|| protocol_error("legacy Responses account binding is invalid"))?;
    let digest = bounded_metadata_string(
        account_binding.get("account_id_digest"),
        "legacy account digest",
    )?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(protocol_error("legacy Responses account digest is invalid"));
    }
    bounded_metadata_string(
        account_binding.get("login_generation"),
        "legacy login generation",
    )?;
    bounded_metadata_string(object.get("response_id"), "legacy response_id")?;
    let item_id = bounded_metadata_string(object.get("item_id"), "legacy item_id")?;
    let encrypted_content = object.get("encrypted_content").and_then(Value::as_str);
    if encrypted_required
        && encrypted_content.is_none_or(|value| value.is_empty() || value.len() > MAX_TEXT_BYTES)
    {
        return Err(protocol_error(
            "legacy Responses encrypted continuation is invalid",
        ));
    }
    if !authenticator(
        object
            .get("account_binding")
            .expect("validated legacy account binding"),
        authentication_binding,
    ) {
        return Ok(None);
    }
    Ok(Some(Continuation {
        item_id,
        encrypted_content,
    }))
}

fn bounded_metadata_string<'a>(
    value: Option<&'a Value>,
    name: &str,
) -> Result<&'a str, OpenAIResponsesError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 512)
        .ok_or_else(|| protocol_error(format!("Responses continuation {name} is invalid")))
}

#[allow(clippy::too_many_arguments)]
fn continuation_metadata(
    model: &str,
    session_id: &str,
    authentication_binding: Option<&str>,
    _response_id: &str,
    item_id: &str,
    _output_index: u64,
    kind: &str,
    encrypted_content: Option<&str>,
) -> MetadataMap {
    let Some(authentication_binding) = authentication_binding else {
        return MetadataMap::new();
    };
    let mut value = json!({
        "schema_version": CONTINUATION_SCHEMA_VERSION,
        "authentication_binding": authentication_binding,
        "model": model,
        "session_id": session_id,
        "item_id": item_id,
        "kind": kind,
    });
    if let Some(encrypted) = encrypted_content {
        value["encrypted_content"] = Value::String(encrypted.to_owned());
    }
    MetadataMap::from([(CONTINUATION_METADATA.into(), value)])
}

fn invalid_request(message: impl Into<String>) -> OpenAIResponsesError {
    OpenAIResponsesError::InvalidRequest(message.into())
}

fn protocol_error(message: impl Into<String>) -> OpenAIResponsesError {
    OpenAIResponsesError::Protocol(message.into())
}

fn encode_tool_output(
    output: &ToolOutput,
    profile: OpenAIResponsesProfile,
) -> Result<Value, OpenAIResponsesError> {
    if profile == OpenAIResponsesProfile::Public {
        return match output {
            ToolOutput::Text(value) => Ok(Value::String(value.clone())),
            ToolOutput::Structured(value) => serde_json::to_string(value)
                .map(Value::String)
                .map_err(OpenAIResponsesError::Serialize),
            ToolOutput::Parts(parts) => serde_json::to_string(parts)
                .map(Value::String)
                .map_err(OpenAIResponsesError::Serialize),
            ToolOutput::Files(files) => serde_json::to_string(files)
                .map(Value::String)
                .map_err(OpenAIResponsesError::Serialize),
        };
    }
    match output {
        ToolOutput::Text(value) => Ok(Value::String(value.clone())),
        ToolOutput::Structured(value) => serde_json::to_string(value)
            .map(Value::String)
            .map_err(OpenAIResponsesError::Serialize),
        ToolOutput::Parts(parts) if parts.iter().any(|part| matches!(part, Part::Media(_))) => {
            parts
                .iter()
                .map(|part| match part {
                    Part::Text(text) => Ok(json!({"type": "input_text", "text": text.text})),
                    Part::Structured(value) => Ok(json!({
                        "type": "input_text",
                        "text": serde_json::to_string(&value.value)
                            .map_err(OpenAIResponsesError::Serialize)?,
                    })),
                    Part::Media(media) => encode_media(media, profile),
                    _ => Err(invalid_request(
                        "private Responses tool output contains unsupported content",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        ToolOutput::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                Part::Text(text) => Ok(text.text.clone()),
                Part::Structured(value) => {
                    serde_json::to_string(&value.value).map_err(OpenAIResponsesError::Serialize)
                }
                _ => Err(invalid_request(
                    "private Responses tool output contains unsupported content",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| Value::String(parts.join("\n"))),
        ToolOutput::Files(_) => Err(invalid_request(
            "private Responses file tool output is unsupported",
        )),
    }
}

#[derive(Clone, Debug)]
struct LogicalDeadline {
    started_at: Instant,
    budget: Duration,
}

impl LogicalDeadline {
    fn new(budget: Duration) -> Self {
        Self {
            started_at: Instant::now(),
            budget,
        }
    }

    fn remaining(&self) -> Result<Duration, HttpError> {
        let elapsed = self.started_at.elapsed();
        if elapsed >= self.budget {
            Err(HttpError::Timeout {
                operation: "logical request retry budget",
                timeout: self.budget,
            })
        } else {
            Ok(self.budget - elapsed)
        }
    }
}

fn deadline_remaining(deadline: Option<&LogicalDeadline>) -> Result<Option<Duration>, HttpError> {
    deadline.map(LogicalDeadline::remaining).transpose()
}

async fn run_bounded_http<F, T>(
    future: F,
    operation_timeout: Option<Duration>,
    deadline: Option<&LogicalDeadline>,
    operation: &'static str,
) -> Result<T, HttpError>
where
    F: Future<Output = Result<T, HttpError>>,
{
    let remaining = deadline_remaining(deadline)?;
    let timeout = match (operation_timeout, remaining) {
        (Some(operation), Some(remaining)) => operation.min(remaining),
        (Some(operation), None) => operation,
        (None, Some(remaining)) => remaining,
        (None, None) => return future.await,
    };
    let budget_limited = remaining
        .is_some_and(|remaining| operation_timeout.is_none_or(|operation| remaining <= operation));
    futures_util::pin_mut!(future);
    let timer = sleep(timeout);
    futures_util::pin_mut!(timer);
    match select(future, timer).await {
        Either::Left((result, _)) => result,
        Either::Right((_, _)) if budget_limited => Err(HttpError::Timeout {
            operation: "logical request retry budget",
            timeout: deadline
                .expect("budget-limited operation has deadline")
                .budget,
        }),
        Either::Right((_, _)) => Err(HttpError::Timeout { operation, timeout }),
    }
}

fn http_loop_error(error: HttpError) -> LoopError {
    LoopError::Provider(format!("OpenAI Responses {error}"))
}

#[derive(Debug)]
struct AttemptFailure {
    error: Box<LoopError>,
    retryable: bool,
    headers: Option<HeaderMap>,
}

struct ResponsesRequestContext {
    client: Http,
    config: Arc<OpenAIResponsesConfig>,
    body: agentkit_http::Bytes,
    idempotency_key: String,
    session_id: String,
    turn_state: Arc<Mutex<Option<HeaderValue>>>,
    auth: AuthenticationAttempt,
    deadline: Option<LogicalDeadline>,
    retries: usize,
    refreshed: bool,
    wire_bytes: usize,
}

struct LiveAttempt {
    body: agentkit_http::BodyStream,
    truncated: TruncatedStreamDetector,
    decoder: ResponsesSseDecoder,
    deadline: Option<LogicalDeadline>,
    eof: bool,
    closed: bool,
}

impl LiveAttempt {
    fn pop_event(&mut self) -> Option<ModelTurnEvent> {
        if !self.closed && matches!(self.decoder.peek_event(), Some(ModelTurnEvent::Finished(_))) {
            return None;
        }
        self.decoder.pop_event()
    }
}

async fn open_live_attempt(
    context: &mut ResponsesRequestContext,
    cancellation: Option<&TurnCancellation>,
) -> Result<LiveAttempt, LoopError> {
    loop {
        let attempt_timeout = context
            .config
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout);
        let attempt_deadline = attempt_timeout.map(LogicalDeadline::new);
        let result = attempt_with_timeout(
            send_live_attempt(context, cancellation),
            attempt_timeout,
            context.deadline.as_ref(),
            cancellation,
        )
        .await;
        match result {
            Ok(mut attempt) => {
                attempt.deadline = attempt_deadline;
                return Ok(attempt);
            }
            Err(failure) if is_unauthorized(&failure.error) && !context.refreshed => {
                let binding = context.auth.binding().map(str::to_owned);
                let refreshed = cancellable(
                    run_bounded_http(
                        context
                            .config
                            .authentication
                            .authenticate(Some(&context.auth)),
                        context
                            .config
                            .resilience
                            .as_ref()
                            .and_then(|config| config.attempt_timeout),
                        context.deadline.as_ref(),
                        "OpenAI reauthentication",
                    ),
                    cancellation,
                )
                .await?
                .map_err(http_loop_error)?;
                if refreshed.binding() != binding.as_deref() {
                    return Err(LoopError::Provider(
                        "OpenAI authentication binding changed during reactive refresh".into(),
                    ));
                }
                context.auth = refreshed;
                context.refreshed = true;
            }
            Err(failure)
                if failure.retryable
                    && context.retries
                        < context
                            .config
                            .resilience
                            .as_ref()
                            .map_or(0, |config| config.max_retries) =>
            {
                let delay = context
                    .config
                    .resilience
                    .as_ref()
                    .expect("retry requires resilience")
                    .retry_delay(context.retries, failure.headers.as_ref());
                context.retries += 1;
                cancellable(
                    run_bounded_http(
                        async {
                            sleep(delay).await;
                            Ok(())
                        },
                        None,
                        context.deadline.as_ref(),
                        "retry backoff",
                    ),
                    cancellation,
                )
                .await?
                .map_err(http_loop_error)?;
            }
            Err(failure) => return Err(*failure.error),
        }
    }
}

async fn send_live_attempt(
    context: &ResponsesRequestContext,
    cancellation: Option<&TurnCancellation>,
) -> Result<LiveAttempt, AttemptFailure> {
    deadline_remaining(context.deadline.as_ref())
        .map_err(|error| nonretryable(http_loop_error(error)))?;
    let mut headers = context.config.headers.clone();
    headers.insert("accept", HeaderValue::from_static("text/event-stream"));
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    if let Some(user_agent) = &context.config.user_agent {
        headers.insert(
            "user-agent",
            HeaderValue::from_str(user_agent)
                .map_err(|_| protocol_failure("invalid user-agent header"))?,
        );
    } else {
        headers
            .entry("user-agent")
            .or_insert(HeaderValue::from_static(concat!(
                "agentkit-provider-openai/",
                env!("CARGO_PKG_VERSION")
            )));
    }
    headers.insert(
        "idempotency-key",
        HeaderValue::from_str(&context.idempotency_key)
            .map_err(|_| protocol_failure("invalid idempotency key"))?,
    );
    if let Some(originator) = &context.config.originator {
        headers.insert(
            "originator",
            HeaderValue::from_str(originator)
                .map_err(|_| protocol_failure("invalid originator header"))?,
        );
    }
    let sent_turn_state = if context.config.profile == OpenAIResponsesProfile::ChatGptPrivate {
        headers.remove(X_CODEX_TURN_STATE);
        headers
            .entry("originator")
            .or_insert(HeaderValue::from_static("agentkit"));
        headers.entry("session-id").or_insert(
            HeaderValue::from_str(&context.session_id)
                .map_err(|_| protocol_failure("invalid session ID header"))?,
        );
        let state = context
            .turn_state
            .lock()
            .map_err(|_| protocol_failure("turn-state lock poisoned"))?
            .clone();
        if let Some(value) = &state {
            headers.insert(X_CODEX_TURN_STATE, value.clone());
        }
        state
    } else {
        None
    };
    headers.extend(context.auth.headers().clone());
    let response = cancellable(
        context
            .client
            .post(&context.config.endpoint)
            .headers(headers)
            .body(context.body.clone())
            .send(),
        cancellation,
    )
    .await
    .map_err(nonretryable)?
    .map_err(transport_failure)?;

    let status = response.status();
    if status == StatusCode::UNAUTHORIZED {
        return Err(AttemptFailure {
            error: Box::new(LoopError::Provider(
                "OpenAI Responses returned 401 Unauthorized".into(),
            )),
            retryable: false,
            headers: None,
        });
    }
    if !status.is_success() {
        return Err(AttemptFailure {
            error: Box::new(LoopError::Provider(format!(
                "OpenAI Responses returned HTTP {status}"
            ))),
            retryable: is_retryable_status(status)
                || (context.config.profile == OpenAIResponsesProfile::ChatGptPrivate
                    && status.as_u16() == 529),
            headers: retry_headers(response.headers()),
        });
    }
    if response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.split(';').next().unwrap_or("").trim() != "text/event-stream")
    {
        return Err(nonretryable(LoopError::Provider(
            "OpenAI Responses response is not text/event-stream".into(),
        )));
    }
    if context.config.profile == OpenAIResponsesProfile::ChatGptPrivate
        && let Some(captured) = validated_turn_state_header(response.headers())?
    {
        if sent_turn_state
            .as_ref()
            .is_some_and(|expected| expected != captured)
        {
            return Err(protocol_failure("provider changed x-codex-turn-state"));
        }
        *context
            .turn_state
            .lock()
            .map_err(|_| protocol_failure("turn-state lock poisoned"))? = Some(captured);
    }
    let truncated = TruncatedStreamDetector::from_headers(response.headers());
    Ok(LiveAttempt {
        body: response.bytes_stream(),
        truncated,
        decoder: ResponsesSseDecoder::with_policy(
            &context.config.model,
            &context.session_id,
            context.config.profile,
            context.config.request_policy.include_encrypted_reasoning,
            context.auth.binding(),
            context.turn_state.clone(),
            context.config.limits.max_attempt_bytes,
        ),
        deadline: None,
        eof: false,
        closed: false,
    })
}

fn attempt_timeout_failure(timeout: Duration) -> AttemptFailure {
    AttemptFailure {
        error: Box::new(LoopError::Provider(format!(
            "Responses attempt timed out after {timeout:?}"
        ))),
        retryable: true,
        headers: None,
    }
}

async fn attempt_with_timeout<F, T>(
    future: F,
    timeout: Option<Duration>,
    deadline: Option<&LogicalDeadline>,
    cancellation: Option<&TurnCancellation>,
) -> Result<T, AttemptFailure>
where
    F: Future<Output = Result<T, AttemptFailure>>,
{
    let timed = async {
        let remaining = match deadline_remaining(deadline) {
            Ok(value) => value,
            Err(error) => return Err(nonretryable(http_loop_error(error))),
        };
        let timeout = match (timeout, remaining) {
            (Some(operation), Some(remaining)) => operation.min(remaining),
            (Some(operation), None) => operation,
            (None, Some(remaining)) => remaining,
            (None, None) => return future.await,
        };
        let budget_limited = remaining.is_some_and(|remaining| timeout == remaining);
        futures_util::pin_mut!(future);
        let timer = sleep(timeout);
        futures_util::pin_mut!(timer);
        match select(future, timer).await {
            Either::Left((result, _)) => result,
            Either::Right((_, _)) if budget_limited => {
                Err(nonretryable(http_loop_error(HttpError::Timeout {
                    operation: "logical request retry budget",
                    timeout: deadline.expect("budget timeout has deadline").budget,
                })))
            }
            Either::Right((_, _)) => Err(attempt_timeout_failure(timeout)),
        }
    };
    match cancellable(timed, cancellation).await {
        Ok(result) => result,
        Err(error) => Err(AttemptFailure {
            error: Box::new(error),
            retryable: false,
            headers: None,
        }),
    }
}

struct ResponsesSseDecoder {
    buffer: Zeroizing<Vec<u8>>,
    buffer_start: usize,
    received: usize,
    max_attempt_bytes: usize,
    state: ResponsesState,
}

impl ResponsesSseDecoder {
    #[cfg(test)]
    fn new(model: &str, session_id: &str) -> Self {
        Self::with_policy(
            model,
            session_id,
            OpenAIResponsesProfile::Public,
            true,
            Some("test-authentication-binding"),
            Arc::new(Mutex::new(None)),
            DEFAULT_MAX_ATTEMPT_BYTES,
        )
    }

    fn with_policy(
        model: &str,
        session_id: &str,
        profile: OpenAIResponsesProfile,
        require_encrypted_reasoning: bool,
        authentication_binding: Option<&str>,
        turn_state: Arc<Mutex<Option<HeaderValue>>>,
        max_attempt_bytes: usize,
    ) -> Self {
        Self {
            buffer: Zeroizing::new(Vec::new()),
            buffer_start: 0,
            received: 0,
            max_attempt_bytes,
            state: ResponsesState::new(
                model,
                session_id,
                profile,
                require_encrypted_reasoning,
                authentication_binding,
                turn_state,
                max_attempt_bytes,
            ),
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<(), AttemptFailure> {
        self.received = self.received.saturating_add(bytes.len());
        if self.received > self.max_attempt_bytes {
            return Err(protocol_failure(
                "Responses SSE attempt exceeds configured byte limit",
            ));
        }
        if self.buffer_start != 0 {
            self.buffer[..self.buffer_start].zeroize();
            self.buffer.drain(..self.buffer_start);
            self.buffer_start = 0;
        }
        zeroizing_extend(&mut self.buffer, bytes, self.max_attempt_bytes)?;
        self.process_pending()
    }

    fn process_pending(&mut self) -> Result<(), AttemptFailure> {
        while self.state.events.is_empty() {
            let Some((relative_end, delimiter)) = frame_end(&self.buffer[self.buffer_start..])
            else {
                if self.buffer.len().saturating_sub(self.buffer_start) > self.max_attempt_bytes {
                    return Err(protocol_failure(
                        "Responses SSE event exceeds configured byte limit",
                    ));
                }
                return Ok(());
            };
            let end = self.buffer_start + relative_end;
            if end - self.buffer_start > self.max_attempt_bytes {
                return Err(protocol_failure(
                    "Responses SSE event exceeds configured byte limit",
                ));
            }
            let mut frame = Zeroizing::new(self.buffer[self.buffer_start..end].to_vec());
            self.buffer_start = end + delimiter;
            self.consume_frame(&mut frame)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn process_all_pending(&mut self) -> Result<(), AttemptFailure> {
        while let Some((relative_end, delimiter)) = frame_end(&self.buffer[self.buffer_start..]) {
            let end = self.buffer_start + relative_end;
            if end - self.buffer_start > self.max_attempt_bytes {
                return Err(protocol_failure(
                    "Responses SSE event exceeds configured byte limit",
                ));
            }
            let mut frame = Zeroizing::new(self.buffer[self.buffer_start..end].to_vec());
            self.buffer_start = end + delimiter;
            self.consume_frame(&mut frame)?;
        }
        Ok(())
    }

    fn peek_event(&self) -> Option<&ModelTurnEvent> {
        self.state.events.front()
    }

    fn pop_event(&mut self) -> Option<ModelTurnEvent> {
        self.state.events.pop_front()
    }

    fn finish_live(&mut self) -> Result<(), AttemptFailure> {
        if self.buffer_start != self.buffer.len() {
            return Err(AttemptFailure {
                error: Box::new(LoopError::Provider(
                    "Responses SSE closed with a partial event".into(),
                )),
                retryable: true,
                headers: None,
            });
        }
        self.state.validate_finished()
    }

    #[cfg(test)]
    fn finish(mut self) -> Result<VecDeque<ModelTurnEvent>, AttemptFailure> {
        self.process_all_pending()?;
        if self.buffer_start != self.buffer.len() {
            return Err(AttemptFailure {
                error: Box::new(LoopError::Provider(
                    "Responses SSE closed with a partial event".into(),
                )),
                retryable: true,
                headers: None,
            });
        }
        self.state.finish()
    }

    fn consume_frame(&mut self, frame: &mut [u8]) -> Result<(), AttemptFailure> {
        let text = std::str::from_utf8(frame)
            .map_err(|_| protocol_failure("Responses SSE event is not UTF-8"))?;
        let mut event = None;
        let mut data = Vec::new();
        for raw in text.lines() {
            let line = raw.trim_end_matches('\r');
            if line.starts_with(':') || line.is_empty() {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" => event = Some(value),
                "data" => data.push(value),
                "id" | "retry" => {}
                _ => {}
            }
        }
        if data.is_empty() {
            return Ok(());
        }
        let data = Zeroizing::new(data.join("\n"));
        if data.as_str() == "[DONE]" {
            return Err(protocol_failure(
                "Responses SSE used an unsupported terminal marker",
            ));
        }
        let mut value: Value = serde_json::from_str(data.as_str())
            .map_err(|_| protocol_failure("Responses SSE data is malformed JSON"))?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| protocol_failure("Responses SSE event omitted type"))?
            .to_owned();
        if event.is_some_and(|name| name != kind) {
            zeroize_encrypted_content(&mut value);
            return Err(protocol_failure("Responses SSE event name/type mismatch"));
        }
        let result = self.state.consume(&kind, &value);
        zeroize_encrypted_content(&mut value);
        result
    }
}

struct PartAccumulator {
    id: PartId,
    text: String,
}

struct ResponsesState {
    requested_model: String,
    profile: OpenAIResponsesProfile,
    max_attempt_bytes: usize,
    session_id: String,
    authentication_binding: Option<String>,
    turn_state: Arc<Mutex<Option<HeaderValue>>>,
    require_encrypted_reasoning: bool,
    sequence: Option<u64>,
    created: bool,
    terminal: bool,
    response_id: Option<String>,
    response_model: Option<String>,
    events: VecDeque<ModelTurnEvent>,
    event_count: usize,
    output: BTreeMap<u64, Item>,
    item_indices: BTreeMap<String, u64>,
    item_types: BTreeMap<String, String>,
    seen_ids: BTreeSet<String>,
    seen_indices: BTreeSet<u64>,
    done_ids: BTreeSet<String>,
    seen_call_ids: BTreeSet<String>,
    content_added: BTreeSet<(String, u64)>,
    content_done: BTreeSet<(String, u64)>,
    summary_added: BTreeSet<(String, u64)>,
    summary_done: BTreeSet<(String, u64)>,
    argument_done: BTreeSet<String>,
    text: BTreeMap<(String, u64), PartAccumulator>,
    reasoning: BTreeMap<(String, u64), PartAccumulator>,
    function_arguments: BTreeMap<String, String>,
    usage: Option<Usage>,
    finish_reason: Option<FinishReason>,
    provider_finish_reason: Option<String>,
    tool_call: bool,
    next_media: usize,
}

impl ResponsesState {
    fn new(
        model: &str,
        session_id: &str,
        profile: OpenAIResponsesProfile,
        require_encrypted_reasoning: bool,
        authentication_binding: Option<&str>,
        turn_state: Arc<Mutex<Option<HeaderValue>>>,
        max_attempt_bytes: usize,
    ) -> Self {
        Self {
            requested_model: model.to_owned(),
            profile,
            max_attempt_bytes,
            session_id: session_id.to_owned(),
            authentication_binding: authentication_binding.map(str::to_owned),
            turn_state,
            require_encrypted_reasoning,
            sequence: None,
            created: false,
            terminal: false,
            response_id: None,
            response_model: None,
            events: VecDeque::new(),
            event_count: 0,
            output: BTreeMap::new(),
            item_indices: BTreeMap::new(),
            item_types: BTreeMap::new(),
            seen_ids: BTreeSet::new(),
            seen_indices: BTreeSet::new(),
            done_ids: BTreeSet::new(),
            seen_call_ids: BTreeSet::new(),
            content_added: BTreeSet::new(),
            content_done: BTreeSet::new(),
            summary_added: BTreeSet::new(),
            summary_done: BTreeSet::new(),
            argument_done: BTreeSet::new(),
            text: BTreeMap::new(),
            reasoning: BTreeMap::new(),
            function_arguments: BTreeMap::new(),
            usage: None,
            finish_reason: None,
            provider_finish_reason: None,
            tool_call: false,
            next_media: 0,
        }
    }

    fn consume(&mut self, kind: &str, value: &Value) -> Result<(), AttemptFailure> {
        if self.terminal {
            return Err(protocol_failure(
                "Responses event followed a terminal event",
            ));
        }
        if let Some(sequence) = value.get("sequence_number").and_then(Value::as_u64) {
            let expected = self
                .sequence
                .map_or(sequence, |last| last.saturating_add(1));
            if sequence != expected {
                return Err(protocol_failure(
                    "Responses SSE sequence is duplicate or out of order",
                ));
            }
            self.sequence = Some(sequence);
        }

        match kind {
            "response.created" => self.created(value),
            "response.in_progress" => {
                self.require_created()?;
                self.observe_response(value.get("response"), "response.in_progress")
            }
            "response.output_item.added" => self.output_item_added(value),
            "response.content_part.added" => self.section_added(value, false),
            "response.content_part.done" => self.section_done(value, false),
            "response.reasoning_summary_part.added" => self.section_added(value, true),
            "response.reasoning_summary_part.done" => self.section_done(value, true),
            "response.output_text.delta" | "response.refusal.delta" => {
                self.text_delta(value, false)
            }
            "response.reasoning_summary_text.delta" => self.text_delta(value, true),
            "response.output_text.done" | "response.refusal.done" => self.text_done(value, false),
            "response.reasoning_summary_text.done" => self.text_done(value, true),
            "response.function_call_arguments.delta" => self.arguments_delta(value),
            "response.function_call_arguments.done" => self.arguments_done(value),
            "response.output_item.done" => self.output_item_done(value),
            "response.completed" => self.complete(value, FinishReason::Completed),
            "response.incomplete" => self.incomplete(value),
            "response.failed" | "error" => {
                self.require_created_if_response_event(kind)?;
                if kind == "response.failed" {
                    if let Some(id) = value.pointer("/response/id") {
                        let id = bounded_id(Some(id))?;
                        if self.response_id.as_deref() != Some(id) {
                            return Err(protocol_failure(
                                "response.failed changed the response ID",
                            ));
                        }
                    }
                    if let Some(model) = value.pointer("/response/model") {
                        let model = bounded_id(Some(model))?;
                        if self
                            .response_model
                            .as_deref()
                            .is_some_and(|expected| expected != model)
                        {
                            return Err(protocol_failure(
                                "response.failed changed the response model",
                            ));
                        }
                    }
                }
                let code = value
                    .pointer("/response/error/code")
                    .or_else(|| value.pointer("/error/code"))
                    .or_else(|| value.get("code"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let retryable = stream_failure_retryable(self.profile, value, kind);
                Err(AttemptFailure {
                    error: Box::new(LoopError::Provider(format!(
                        "OpenAI Responses stream failed ({code})"
                    ))),
                    retryable,
                    headers: None,
                })
            }
            "keepalive" => Ok(()),
            "response.metadata" => {
                self.require_created()?;
                if let Some(response) = value.get("response") {
                    self.observe_response(Some(response), "response.metadata")?;
                }
                if let Some(captured) = responses_turn_state(value)? {
                    let mut turn_state = self
                        .turn_state
                        .lock()
                        .map_err(|_| protocol_failure("turn-state lock poisoned"))?;
                    if turn_state
                        .as_ref()
                        .is_some_and(|expected| expected != captured)
                    {
                        return Err(protocol_failure(
                            "response metadata changed x-codex-turn-state",
                        ));
                    }
                    *turn_state = Some(captured);
                }
                Ok(())
            }
            _ if self.profile == OpenAIResponsesProfile::ChatGptPrivate => Ok(()),
            _ => Err(protocol_failure("unsupported Responses SSE event kind")),
        }
    }

    fn created(&mut self, value: &Value) -> Result<(), AttemptFailure> {
        if self.created {
            return Err(protocol_failure("duplicate response.created"));
        }
        self.created = true;
        self.observe_response(value.get("response"), "response.created")?;
        if self.response_id.is_none() {
            return Err(protocol_failure("response.created omitted response ID"));
        }
        Ok(())
    }

    fn require_created(&self) -> Result<(), AttemptFailure> {
        if self.created {
            Ok(())
        } else {
            Err(protocol_failure(
                "Responses event preceded response.created",
            ))
        }
    }

    fn require_created_if_response_event(&self, kind: &str) -> Result<(), AttemptFailure> {
        if kind == "response.failed" {
            self.require_created()
        } else {
            Ok(())
        }
    }

    fn observe_response(
        &mut self,
        response: Option<&Value>,
        event: &str,
    ) -> Result<(), AttemptFailure> {
        let response = response
            .and_then(Value::as_object)
            .ok_or_else(|| protocol_failure("Responses lifecycle event omitted response"))?;
        let id = bounded_id(response.get("id"))?;
        if self
            .response_id
            .as_deref()
            .is_some_and(|expected| expected != id)
        {
            return Err(protocol_failure(
                "Responses lifecycle event changed response ID",
            ));
        }
        self.response_id.get_or_insert_with(|| id.to_owned());
        if let Some(model) = response.get("model") {
            let model = bounded_id(Some(model))?;
            if self
                .response_model
                .as_deref()
                .is_some_and(|expected| expected != model)
            {
                return Err(protocol_failure(
                    "Responses lifecycle event changed response model",
                ));
            }
            self.response_model.get_or_insert_with(|| model.to_owned());
        }
        let _ = event;
        Ok(())
    }

    fn output_item_added(&mut self, value: &Value) -> Result<(), AttemptFailure> {
        self.require_created()?;
        let item = value
            .get("item")
            .and_then(Value::as_object)
            .ok_or_else(|| protocol_failure("output_item.added omitted item"))?;
        let id = bounded_id(item.get("id"))?;
        let index = nonnegative(value, "output_index")?;
        let kind = item.get("type").and_then(Value::as_str);
        if kind.is_some_and(|kind| {
            !matches!(
                kind,
                "message" | "reasoning" | "function_call" | "image_generation_call"
            )
        }) {
            return Err(protocol_failure("unsupported Responses output item"));
        }
        if !self.seen_ids.insert(id.to_owned()) {
            return Err(protocol_failure("duplicate output item ID"));
        }
        if !self.seen_indices.insert(index) {
            return Err(protocol_failure("duplicate output item index"));
        }
        self.item_indices.insert(id.to_owned(), index);
        if let Some(kind) = kind {
            self.item_types.insert(id.to_owned(), kind.to_owned());
        }
        Ok(())
    }

    fn event_item(
        &self,
        value: &Value,
        index_field: &str,
    ) -> Result<(String, u64), AttemptFailure> {
        self.require_created()?;
        let id = bounded_id(value.get("item_id"))?;
        let output_index = nonnegative(value, "output_index")?;
        if self.item_indices.get(id).copied() != Some(output_index) {
            return Err(protocol_failure(
                "content event refers to an unknown or inconsistent output item",
            ));
        }
        Ok((id.to_owned(), nonnegative(value, index_field)?))
    }

    fn section_added(&mut self, value: &Value, reasoning: bool) -> Result<(), AttemptFailure> {
        let field = if reasoning {
            "summary_index"
        } else {
            "content_index"
        };
        let key = self.event_item(value, field)?;
        let expected_type = if reasoning {
            "summary_text"
        } else {
            "output_text"
        };
        let part = value
            .get("part")
            .and_then(Value::as_object)
            .ok_or_else(|| protocol_failure("content-part add omitted part"))?;
        let part_type = part.get("type").and_then(Value::as_str);
        let supported =
            part_type == Some(expected_type) || (!reasoning && part_type == Some("refusal"));
        if !supported {
            return Err(protocol_failure("unsupported Responses content part"));
        }
        let set = if reasoning {
            &mut self.summary_added
        } else {
            &mut self.content_added
        };
        if !set.insert(key) {
            return Err(protocol_failure("duplicate Responses content-part add"));
        }
        Ok(())
    }

    fn section_done(&mut self, value: &Value, reasoning: bool) -> Result<(), AttemptFailure> {
        let field = if reasoning {
            "summary_index"
        } else {
            "content_index"
        };
        let key = self.event_item(value, field)?;
        let expected_type = if reasoning {
            "summary_text"
        } else {
            "output_text"
        };
        let part = value
            .get("part")
            .and_then(Value::as_object)
            .ok_or_else(|| protocol_failure("content-part done omitted part"))?;
        let part_type = part.get("type").and_then(Value::as_str);
        let supported =
            part_type == Some(expected_type) || (!reasoning && part_type == Some("refusal"));
        if !supported {
            return Err(protocol_failure(
                "unsupported completed Responses content part",
            ));
        }
        let added = if reasoning {
            &self.summary_added
        } else {
            &self.content_added
        };
        let done = if reasoning {
            &mut self.summary_done
        } else {
            &mut self.content_done
        };
        if !added.contains(&key) || !done.insert(key) {
            return Err(protocol_failure(
                "Responses content part completed without add or twice",
            ));
        }
        Ok(())
    }

    fn text_delta(&mut self, value: &Value, reasoning: bool) -> Result<(), AttemptFailure> {
        let field = if reasoning {
            "summary_index"
        } else {
            "content_index"
        };
        let key = self.event_item(value, field)?;
        let added = if reasoning {
            &self.summary_added
        } else {
            &self.content_added
        };
        if !added.contains(&key) {
            return Err(protocol_failure(
                "Responses text delta preceded content-part add",
            ));
        }
        let delta = bounded_field(value, "delta")?;
        let target = if reasoning {
            &mut self.reasoning
        } else {
            &mut self.text
        };
        append_part(
            target,
            key,
            delta,
            if reasoning {
                PartKind::Reasoning
            } else {
                PartKind::Text
            },
            &mut self.events,
            &mut self.event_count,
        )
    }

    fn text_done(&mut self, value: &Value, reasoning: bool) -> Result<(), AttemptFailure> {
        let field = if reasoning {
            "summary_index"
        } else {
            "content_index"
        };
        let key = self.event_item(value, field)?;
        let added = if reasoning {
            &self.summary_added
        } else {
            &self.content_added
        };
        if !added.contains(&key) {
            return Err(protocol_failure(
                "Responses text done preceded content-part add",
            ));
        }
        if let Some(completed) = value
            .get("text")
            .or_else(|| value.get("refusal"))
            .and_then(Value::as_str)
        {
            let target = if reasoning {
                &self.reasoning
            } else {
                &self.text
            };
            if target
                .get(&key)
                .is_some_and(|streamed| streamed.text != completed)
            {
                return Err(protocol_failure(
                    "Responses text done changed streamed content",
                ));
            }
        }
        Ok(())
    }

    fn arguments_delta(&mut self, value: &Value) -> Result<(), AttemptFailure> {
        let (id, _) = self.event_item(value, "output_index")?;
        if self
            .item_types
            .get(&id)
            .is_some_and(|kind| kind != "function_call")
        {
            return Err(protocol_failure(
                "function arguments referred to a non-call item",
            ));
        }
        append_bounded(
            &mut self.function_arguments,
            &id,
            bounded_field(value, "delta")?,
        )
    }

    fn arguments_done(&mut self, value: &Value) -> Result<(), AttemptFailure> {
        let id = bounded_id(value.get("item_id"))?.to_owned();
        let output_index = nonnegative(value, "output_index")?;
        if self.item_indices.get(&id).copied() != Some(output_index)
            || !self.argument_done.insert(id.clone())
        {
            return Err(protocol_failure(
                "function arguments done is inconsistent or duplicate",
            ));
        }
        if let Some(arguments) = value.get("arguments").and_then(Value::as_str)
            && self
                .function_arguments
                .get(&id)
                .is_some_and(|streamed| streamed != arguments)
        {
            return Err(protocol_failure(
                "function arguments done changed streamed arguments",
            ));
        }
        Ok(())
    }

    fn output_item_done(&mut self, value: &Value) -> Result<(), AttemptFailure> {
        self.require_created()?;
        let item = value
            .get("item")
            .ok_or_else(|| protocol_failure("output item omitted item"))?;
        let id = bounded_id(item.get("id"))?;
        if !self.seen_ids.contains(id) || !self.done_ids.insert(id.to_owned()) {
            return Err(protocol_failure(
                "output item completed without add or completed twice",
            ));
        }
        let output_index = nonnegative(value, "output_index")?;
        if self.item_indices.get(id).copied() != Some(output_index) {
            return Err(protocol_failure("completed output item index changed"));
        }
        if self.item_types.get(id).is_some_and(|expected| {
            Some(expected.as_str()) != item.get("type").and_then(Value::as_str)
        }) {
            return Err(protocol_failure("completed output item type changed"));
        }
        self.output_item(output_index, item)
    }

    fn output_item(&mut self, output_index: u64, item: &Value) -> Result<(), AttemptFailure> {
        let item_id = bounded_id(item.get("id"))?;
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if item.get("role").and_then(Value::as_str) != Some("assistant") {
                    return Err(protocol_failure("output message role is not assistant"));
                }
                let content = item
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(|| protocol_failure("output message content is malformed"))?;
                let mut parts = Vec::new();
                for (index, raw) in content.iter().enumerate() {
                    let key = (item_id.to_owned(), index as u64);
                    let content_type = raw.get("type").and_then(Value::as_str);
                    if !matches!(content_type, Some("output_text" | "refusal"))
                        || !self.content_added.contains(&key)
                        || !self.content_done.contains(&key)
                    {
                        return Err(protocol_failure(
                            "output message content lifecycle is incomplete",
                        ));
                    }
                    let text = if content_type == Some("refusal") {
                        bounded_field(raw, "refusal")?
                    } else {
                        bounded_field(raw, "text")?
                    };
                    let streamed = self.text.remove(&key);
                    if streamed
                        .as_ref()
                        .is_some_and(|streamed| streamed.text != text)
                    {
                        return Err(protocol_failure(
                            "completed output changed streamed text content",
                        ));
                    }
                    let part = Part::Text(TextPart::new(text));
                    if streamed.is_some() {
                        push_event(
                            &mut self.events,
                            &mut self.event_count,
                            ModelTurnEvent::Delta(Delta::CommitPart { part: part.clone() }),
                        )?;
                    }
                    parts.push(part);
                }
                self.output
                    .insert(output_index, provider_item(item_id, parts));
            }
            Some("function_call") => {
                let call_id = bounded_nonempty_field(item, "call_id")?;
                if !self.seen_call_ids.insert(call_id.to_owned()) {
                    return Err(protocol_failure("duplicate function call ID"));
                }
                let name = bounded_nonempty_field(item, "name")?;
                validate_tool_name(name)
                    .map_err(|error| protocol_failure(&format!("provider returned {error}")))?;
                let arguments = bounded_field(item, "arguments")?;
                if let Some(streamed) = self.function_arguments.remove(item_id)
                    && (streamed != arguments || !self.argument_done.contains(item_id))
                {
                    return Err(protocol_failure(
                        "completed function arguments changed streamed arguments",
                    ));
                }
                let input: Value = serde_json::from_str(arguments)
                    .map_err(|_| protocol_failure("function-call arguments are not JSON"))?;
                if !input.is_object() {
                    return Err(protocol_failure(
                        "function-call arguments are not an object",
                    ));
                }
                let response_id = self
                    .response_id
                    .as_deref()
                    .expect("created response has ID");
                let call =
                    ToolCallPart::new(call_id, name, input).with_metadata(continuation_metadata(
                        &self.requested_model,
                        &self.session_id,
                        self.authentication_binding.as_deref(),
                        response_id,
                        item_id,
                        output_index,
                        "function_call",
                        None,
                    ));
                self.tool_call = true;
                self.output.insert(
                    output_index,
                    provider_item(item_id, vec![Part::ToolCall(call)]),
                );
            }
            Some("image_generation_call") => {
                let status = bounded_nonempty_field(item, "status")?;
                if status != "completed" {
                    return Err(protocol_failure("image generation did not complete"));
                }
                let result = item
                    .get("result")
                    .and_then(Value::as_str)
                    .filter(|result| !result.is_empty() && result.len() <= self.max_attempt_bytes)
                    .ok_or_else(|| {
                        protocol_failure("generated image result is outside configured byte bounds")
                    })?;
                let bytes = STANDARD
                    .decode(result)
                    .map_err(|_| protocol_failure("generated image result is not valid base64"))?;
                if bytes.is_empty() || STANDARD.encode(&bytes) != result {
                    return Err(protocol_failure(
                        "generated image result is not canonical base64",
                    ));
                }
                let revised_prompt = item
                    .get("revised_prompt")
                    .filter(|value| !value.is_null())
                    .map(|_| bounded_field(item, "revised_prompt"))
                    .transpose()?;
                let response_id = self
                    .response_id
                    .as_deref()
                    .expect("created response has ID");
                let mut metadata = continuation_metadata(
                    &self.requested_model,
                    &self.session_id,
                    self.authentication_binding.as_deref(),
                    response_id,
                    item_id,
                    output_index,
                    "image_generation_call",
                    None,
                );
                metadata.insert(
                    GENERATED_IMAGE_METADATA.to_owned(),
                    json!({
                        "item_id": item_id,
                        "status": status,
                        "revised_prompt": revised_prompt,
                    }),
                );
                let media =
                    MediaPart::new(Modality::Image, "image/png", DataRef::InlineBytes(bytes))
                        .with_metadata(metadata);
                self.next_media += 1;
                let placeholder_id = PartId::new(format!("generated-image-{output_index}"));
                push_event(
                    &mut self.events,
                    &mut self.event_count,
                    ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: placeholder_id.clone(),
                        kind: PartKind::Text,
                    }),
                )?;
                push_event(
                    &mut self.events,
                    &mut self.event_count,
                    ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: placeholder_id,
                        chunk: format!("[Image #{}]", self.next_media),
                    }),
                )?;
                self.output.insert(
                    output_index,
                    provider_item(item_id, vec![Part::Media(media)]),
                );
            }
            Some("reasoning") => {
                let summaries = item
                    .get("summary")
                    .and_then(Value::as_array)
                    .ok_or_else(|| protocol_failure("reasoning summary is malformed"))?;
                let encrypted = item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty() && value.len() <= MAX_TEXT_BYTES);
                if self.require_encrypted_reasoning && encrypted.is_none() {
                    return Err(protocol_failure(
                        "encrypted reasoning is missing or outside bounds",
                    ));
                }
                let mut summary_texts = Vec::new();
                for (index, raw) in summaries.iter().enumerate() {
                    let key = (item_id.to_owned(), index as u64);
                    if raw.get("type").and_then(Value::as_str) != Some("summary_text")
                        || !self.summary_added.contains(&key)
                        || !self.summary_done.contains(&key)
                    {
                        return Err(protocol_failure(
                            "reasoning summary lifecycle is incomplete",
                        ));
                    }
                    let text = bounded_field(raw, "text")?;
                    let streamed = self.reasoning.remove(&key);
                    if streamed
                        .as_ref()
                        .is_some_and(|streamed| streamed.text != text)
                    {
                        return Err(protocol_failure(
                            "completed reasoning changed streamed summary",
                        ));
                    }
                    if streamed.is_some() {
                        push_event(
                            &mut self.events,
                            &mut self.event_count,
                            ModelTurnEvent::Delta(Delta::CommitPart {
                                part: Part::Reasoning(ReasoningPart::summary(text)),
                            }),
                        )?;
                    }
                    summary_texts.push(text);
                }
                let response_id = self
                    .response_id
                    .as_deref()
                    .expect("created response has ID");
                let metadata = encrypted.map_or_else(MetadataMap::new, |encrypted| {
                    continuation_metadata(
                        &self.requested_model,
                        &self.session_id,
                        self.authentication_binding.as_deref(),
                        response_id,
                        item_id,
                        output_index,
                        "reasoning",
                        Some(encrypted),
                    )
                });
                let part = Part::Reasoning(ReasoningPart {
                    summary: (!summary_texts.is_empty()).then(|| summary_texts.join("\n\n")),
                    data: None,
                    redacted: encrypted.is_some(),
                    metadata,
                });
                self.output
                    .insert(output_index, provider_item(item_id, vec![part]));
            }
            _ => return Err(protocol_failure("unsupported Responses output item")),
        }
        Ok(())
    }

    fn complete(&mut self, value: &Value, default: FinishReason) -> Result<(), AttemptFailure> {
        self.require_created()?;
        self.observe_response(value.get("response"), "response.completed")?;
        if self.seen_ids != self.done_ids
            || self.content_added != self.content_done
            || self.summary_added != self.summary_done
            || !self.text.is_empty()
            || !self.reasoning.is_empty()
        {
            return Err(protocol_failure(
                "response.completed preceded complete output items",
            ));
        }
        let response = value.get("response").expect("observed response");
        if let Some(raw) = response.get("usage") {
            self.usage = Some(parse_usage(raw));
        }
        self.provider_finish_reason = response
            .get("status")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty() && reason.len() <= 128 && reason.is_ascii())
            .map(str::to_owned)
            .or_else(|| Some("completed".into()));
        self.finish_reason = Some(if self.tool_call {
            FinishReason::ToolCall
        } else {
            default
        });
        self.terminal = true;
        self.queue_terminal()?;
        Ok(())
    }

    fn incomplete(&mut self, value: &Value) -> Result<(), AttemptFailure> {
        self.require_created()?;
        self.observe_response(value.get("response"), "response.incomplete")?;
        let response = value.get("response").expect("observed response");
        let reason = response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty() && reason.len() <= 128 && reason.is_ascii())
            .ok_or_else(|| protocol_failure("response.incomplete omitted a valid reason"))?;
        self.provider_finish_reason = Some(reason.to_owned());
        self.finish_reason = Some(match reason {
            "max_output_tokens" => FinishReason::MaxTokens,
            "content_filter" => FinishReason::Blocked,
            other => FinishReason::Other(other.to_owned()),
        });
        if let Some(raw) = response.get("usage") {
            self.usage = Some(parse_usage(raw));
        }
        self.flush_partial_text()?;
        self.output
            .retain(|_, item| item.parts.iter().all(|part| matches!(part, Part::Text(_))));
        self.reasoning.clear();
        self.function_arguments.clear();
        self.terminal = true;
        self.queue_terminal()?;
        Ok(())
    }

    fn flush_partial_text(&mut self) -> Result<(), AttemptFailure> {
        let partial = std::mem::take(&mut self.text);
        for ((item_id, _), accumulator) in partial {
            let Some(index) = self.item_indices.get(&item_id).copied() else {
                continue;
            };
            if accumulator.text.is_empty() {
                continue;
            }
            let part = Part::Text(TextPart::new(accumulator.text));
            push_event(
                &mut self.events,
                &mut self.event_count,
                ModelTurnEvent::Delta(Delta::CommitPart { part: part.clone() }),
            )?;
            self.output
                .insert(index, provider_item(&item_id, vec![part]));
        }
        Ok(())
    }

    fn queue_terminal(&mut self) -> Result<(), AttemptFailure> {
        for call in self
            .output
            .values()
            .flat_map(|item| item.parts.iter())
            .filter_map(|part| match part {
                Part::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
        {
            push_event(
                &mut self.events,
                &mut self.event_count,
                ModelTurnEvent::Delta(Delta::CommitPart {
                    part: Part::ToolCall(call.clone()),
                }),
            )?;
            push_event(
                &mut self.events,
                &mut self.event_count,
                ModelTurnEvent::ToolCall(call),
            )?;
        }
        if let Some(usage) = self.usage.clone() {
            push_event(
                &mut self.events,
                &mut self.event_count,
                ModelTurnEvent::Usage(usage),
            )?;
        }
        let mut metadata = MetadataMap::from([(
            "openai.responses.profile".into(),
            Value::String("responses".into()),
        )]);
        set_provider_finish_reasons(&mut metadata, self.provider_finish_reason.iter().cloned());
        push_event(
            &mut self.events,
            &mut self.event_count,
            ModelTurnEvent::Finished(ModelTurnResult {
                finish_reason: self.finish_reason.clone().expect("terminal reason"),
                output_items: std::mem::take(&mut self.output).into_values().collect(),
                usage: self.usage.clone(),
                metadata,
                model: self.response_model.clone(),
                response_id: self.response_id.clone(),
            }),
        )?;
        Ok(())
    }

    fn validate_finished(&self) -> Result<(), AttemptFailure> {
        if !self.created {
            return Err(protocol_failure(
                "Responses SSE closed before response.created",
            ));
        }
        if !self.terminal {
            return Err(AttemptFailure {
                error: Box::new(LoopError::Provider(
                    "OpenAI Responses SSE stream closed before a terminal event".into(),
                )),
                retryable: true,
                headers: None,
            });
        }
        if self.events.len() > MAX_ITEMS {
            return Err(protocol_failure(
                "Responses attempt produced too many events",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn finish(mut self) -> Result<VecDeque<ModelTurnEvent>, AttemptFailure> {
        self.validate_finished()?;
        Ok(std::mem::take(&mut self.events))
    }
}

fn provider_item(item_id: &str, parts: Vec<Part>) -> Item {
    let mut item = Item::new(ItemKind::Assistant, parts);
    item.id = Some(MessageId::new(item_id));
    item
}

fn push_event(
    events: &mut VecDeque<ModelTurnEvent>,
    event_count: &mut usize,
    event: ModelTurnEvent,
) -> Result<(), AttemptFailure> {
    if *event_count >= MAX_ITEMS {
        return Err(protocol_failure(
            "Responses attempt produced too many events",
        ));
    }
    *event_count += 1;
    events.push_back(event);
    Ok(())
}

fn append_part(
    parts: &mut BTreeMap<(String, u64), PartAccumulator>,
    key: (String, u64),
    delta: &str,
    kind: PartKind,
    events: &mut VecDeque<ModelTurnEvent>,
    event_count: &mut usize,
) -> Result<(), AttemptFailure> {
    if !parts.contains_key(&key) {
        let id = PartId::new(format!(
            "openai-responses:{}:{}:{:?}:{}",
            key.0.len(),
            key.0,
            kind,
            key.1
        ));
        push_event(
            events,
            event_count,
            ModelTurnEvent::Delta(Delta::BeginPart {
                part_id: id.clone(),
                kind,
            }),
        )?;
        parts.insert(
            key.clone(),
            PartAccumulator {
                id,
                text: String::new(),
            },
        );
    }
    let part = parts.get_mut(&key).expect("inserted part accumulator");
    if part.text.len().saturating_add(delta.len()) > MAX_TEXT_BYTES {
        return Err(protocol_failure("Responses streamed content exceeds 8 MiB"));
    }
    part.text.push_str(delta);
    push_event(
        events,
        event_count,
        ModelTurnEvent::Delta(Delta::AppendText {
            part_id: part.id.clone(),
            chunk: delta.to_owned(),
        }),
    )
}

fn nonnegative(value: &Value, field: &str) -> Result<u64, AttemptFailure> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .filter(|value| *value < MAX_ITEMS as u64)
        .ok_or_else(|| protocol_failure("Responses index is missing or outside bounds"))
}

fn bounded_id(value: Option<&Value>) -> Result<&str, AttemptFailure> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 512)
        .ok_or_else(|| protocol_failure("Responses ID is missing or outside bounds"))
}

fn bounded_nonempty_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, AttemptFailure> {
    bounded_field(value, field).and_then(|value| {
        if value.is_empty() {
            Err(protocol_failure("Responses field is empty"))
        } else {
            Ok(value)
        }
    })
}

fn zeroizing_extend(
    buffer: &mut Zeroizing<Vec<u8>>,
    bytes: &[u8],
    limit: usize,
) -> Result<(), AttemptFailure> {
    let new_len = buffer
        .len()
        .checked_add(bytes.len())
        .filter(|length| *length <= limit)
        .ok_or_else(|| protocol_failure("Responses SSE buffer exceeds canonical bounds"))?;
    if new_len > buffer.capacity() {
        let capacity = buffer.capacity().saturating_mul(2).max(new_len).min(limit);
        let mut replacement = Vec::with_capacity(capacity);
        replacement.extend_from_slice(buffer);
        let mut previous = std::mem::replace(&mut **buffer, replacement);
        previous.zeroize();
    }
    buffer.extend_from_slice(bytes);
    Ok(())
}

fn zeroize_encrypted_content(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "encrypted_content" {
                    if let Value::String(secret) = value {
                        secret.zeroize();
                    }
                } else {
                    zeroize_encrypted_content(value);
                }
            }
        }
        Value::Array(values) => values.iter_mut().for_each(zeroize_encrypted_content),
        _ => {}
    }
}

fn responses_turn_state(value: &Value) -> Result<Option<HeaderValue>, AttemptFailure> {
    let mut state = None;
    for headers in [value.pointer("/response/headers"), value.get("headers")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        for raw in headers.iter().filter_map(|(name, value)| {
            name.eq_ignore_ascii_case(X_CODEX_TURN_STATE)
                .then_some(value)
        }) {
            let values: Vec<&str> = match raw {
                Value::String(value) => vec![value],
                Value::Array(values) if !values.is_empty() => values
                    .iter()
                    .map(Value::as_str)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| protocol_failure("response metadata turn state is invalid"))?,
                _ => {
                    return Err(protocol_failure("response metadata turn state is invalid"));
                }
            };
            for value in values {
                let value = HeaderValue::from_str(value)
                    .map_err(|_| protocol_failure("response metadata turn state is invalid"))?;
                if state.as_ref().is_some_and(|expected| expected != value) {
                    return Err(protocol_failure(
                        "response metadata changed x-codex-turn-state",
                    ));
                }
                state = Some(value);
            }
        }
    }
    Ok(state)
}

fn retry_headers(headers: &HeaderMap) -> Option<HeaderMap> {
    let mut retained = HeaderMap::new();
    for name in [
        "retry-after",
        "ratelimit-reset",
        "x-ratelimit-reset",
        "x-rate-limit-reset",
    ] {
        if let Some(value) = headers.get(name) {
            retained.insert(agentkit_http::HeaderName::from_static(name), value.clone());
        }
    }
    (!retained.is_empty()).then_some(retained)
}

fn validated_turn_state_header(headers: &HeaderMap) -> Result<Option<HeaderValue>, AttemptFailure> {
    let mut state: Option<HeaderValue> = None;
    for value in headers.get_all(X_CODEX_TURN_STATE) {
        if state.as_ref().is_some_and(|expected| expected != value) {
            return Err(protocol_failure(
                "provider returned conflicting x-codex-turn-state headers",
            ));
        }
        state = Some(value.clone());
    }
    Ok(state)
}

fn parse_usage(value: &Value) -> Usage {
    let input = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = value
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut tokens = TokenUsage::new(input, output);
    if let Some(reasoning) = value
        .pointer("/output_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64)
    {
        tokens = tokens.with_reasoning_tokens(reasoning);
    }
    if let Some(cached) = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
    {
        tokens = tokens.with_cached_input_tokens(cached);
    }
    Usage::new(tokens)
}

fn bounded_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, AttemptFailure> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| value.len() <= MAX_TEXT_BYTES)
        .ok_or_else(|| protocol_failure("Responses text field is missing or too large"))
}

fn append_bounded(
    target: &mut BTreeMap<String, String>,
    id: &str,
    delta: &str,
) -> Result<(), AttemptFailure> {
    let value = target.entry(id.to_owned()).or_default();
    if value.len().saturating_add(delta.len()) > MAX_TEXT_BYTES {
        return Err(protocol_failure("Responses output exceeds 8 MiB"));
    }
    value.push_str(delta);
    Ok(())
}

fn frame_end(buffer: &[u8]) -> Option<(usize, usize)> {
    [
        buffer
            .windows(4)
            .position(|value| value == b"\r\n\r\n")
            .map(|index| (index, 4)),
        buffer
            .windows(2)
            .position(|value| value == b"\n\n")
            .map(|index| (index, 2)),
        buffer
            .windows(2)
            .position(|value| value == b"\r\r")
            .map(|index| (index, 2)),
    ]
    .into_iter()
    .flatten()
    .min_by_key(|(index, _)| *index)
}

fn stable_idempotency_key(session: &str, turn: &str, body: &[u8]) -> String {
    let mut left = 0xcbf29ce484222325_u64;
    let mut right = 0x9e3779b97f4a7c15_u64;
    for byte in session
        .bytes()
        .chain([0])
        .chain(turn.bytes())
        .chain([0])
        .chain(body.iter().copied())
    {
        left ^= u64::from(byte);
        left = left.wrapping_mul(0x100000001b3);
        right ^= left.rotate_left(17).wrapping_add(u64::from(byte));
        right = right.wrapping_mul(0xff51afd7ed558ccd);
    }
    format!("agentkit-{left:016x}{right:016x}")
}

fn transport_failure(error: HttpError) -> AttemptFailure {
    let retryable = error.is_retryable_transport();
    AttemptFailure {
        error: Box::new(LoopError::Provider(format!(
            "OpenAI Responses transport failed: {error}"
        ))),
        retryable,
        headers: None,
    }
}

fn protocol_failure(message: &str) -> AttemptFailure {
    nonretryable(LoopError::Provider(message.into()))
}

fn stream_failure_retryable(profile: OpenAIResponsesProfile, value: &Value, kind: &str) -> bool {
    let error = if kind == "response.failed" {
        value.pointer("/response/error").unwrap_or(&Value::Null)
    } else {
        value.get("error").unwrap_or(value)
    };
    let code = error
        .get("code")
        .or_else(|| value.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if profile == OpenAIResponsesProfile::Public {
        return matches!(
            code,
            "server_error" | "rate_limit_exceeded" | "temporarily_unavailable"
        );
    }
    let error_type = error
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let status = error
        .get("status")
        .or_else(|| value.get("status"))
        .or_else(|| value.pointer("/response/status_code"))
        .and_then(Value::as_u64);
    let authentication = status.is_some_and(|status| status == 401 || status == 403)
        || [code, error_type].iter().any(|value| {
            matches!(
                *value,
                "authentication_error"
                    | "invalid_api_key"
                    | "invalid_authentication"
                    | "unauthorized"
            )
        });
    let permanent = status.is_some_and(|status| {
        matches!(
            status,
            400 | 402 | 403 | 404 | 405 | 406 | 410 | 413 | 415 | 422 | 501 | 505
        )
    }) || [code, error_type].iter().any(|value| {
        [
            "billing",
            "content_policy",
            "deactivated",
            "insufficient",
            "invalid",
            "not_found",
            "not_supported",
            "permission",
            "quota",
            "unsupported",
        ]
        .iter()
        .any(|marker| value.contains(marker))
    });
    !authentication
        && !permanent
        && status
            .is_none_or(|status| matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529))
}

fn nonretryable(error: LoopError) -> AttemptFailure {
    AttemptFailure {
        error: Box::new(error),
        retryable: false,
        headers: None,
    }
}

fn is_unauthorized(error: &LoopError) -> bool {
    matches!(error, LoopError::Provider(message) if message.contains("401 Unauthorized"))
}

fn cancelled(cancellation: Option<&TurnCancellation>) -> bool {
    cancellation.is_some_and(TurnCancellation::is_cancelled)
}

async fn cancellable<F, T>(
    future: F,
    cancellation: Option<&TurnCancellation>,
) -> Result<T, LoopError>
where
    F: Future<Output = T>,
{
    let Some(cancellation) = cancellation else {
        return Ok(future.await);
    };
    if cancellation.is_cancelled() {
        return Err(LoopError::Cancelled);
    }
    futures_util::pin_mut!(future);
    let cancelled = cancellation.cancelled();
    futures_util::pin_mut!(cancelled);
    match select(future, cancelled).await {
        Either::Left((result, _)) => Ok(result),
        Either::Right((_, _)) => Err(LoopError::Cancelled),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agentkit_core::{Item, SessionId, TurnId};
    use agentkit_http::{HeaderValue, HttpClient, HttpRequest, HttpResponse, StatusCode, header};
    use async_trait::async_trait;
    use futures_util::stream;

    use super::*;

    #[derive(Clone)]
    struct WireResponse {
        status: StatusCode,
        headers: HeaderMap,
        body: &'static str,
    }

    struct ScriptedClient {
        responses: Mutex<VecDeque<WireResponse>>,
        requests: Mutex<Vec<HttpRequest>>,
    }

    #[async_trait]
    impl HttpClient for ScriptedClient {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.requests.lock().unwrap().push(request.clone());
            let response = self.responses.lock().unwrap().pop_front().unwrap();
            let body = agentkit_http::Bytes::from_static(response.body.as_bytes());
            Ok(HttpResponse::new(
                response.status,
                response.headers,
                request.url,
                Box::pin(stream::once(async move { Ok(body) })),
            ))
        }
    }

    struct ActiveStreamClient {
        requests: AtomicUsize,
    }

    #[async_trait]
    impl HttpClient for ActiveStreamClient {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            let body = stream::unfold((), |_| async {
                sleep(Duration::from_millis(1)).await;
                Some((
                    Ok(agentkit_http::Bytes::from_static(b": keepalive\n\n")),
                    (),
                ))
            });
            Ok(HttpResponse::new(
                StatusCode::OK,
                sse_headers(),
                request.url,
                Box::pin(body),
            ))
        }
    }

    struct RefreshingAuth {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl agentkit_http::AuthenticationProvider for RefreshingAuth {
        async fn authenticate(
            &self,
            previous: Option<&AuthenticationAttempt>,
        ) -> Result<AuthenticationAttempt, HttpError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                assert!(previous.is_none());
            } else {
                assert_eq!(previous.and_then(|value| value.state::<usize>()), Some(&0));
            }
            let mut headers = HeaderMap::new();
            headers.insert(
                header::AUTHORIZATION,
                HeaderValue::from_str(if call == 0 {
                    "Bearer first"
                } else {
                    "Bearer refreshed"
                })
                .unwrap(),
            );
            Ok(AuthenticationAttempt::new(headers, call))
        }
    }

    struct BindingChangingAuth {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AuthenticationProvider for BindingChangingAuth {
        async fn authenticate(
            &self,
            _previous: Option<&AuthenticationAttempt>,
        ) -> Result<AuthenticationAttempt, HttpError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(AuthenticationAttempt::stateless(HeaderMap::new())
                .with_binding(format!("binding-{call}")))
        }
    }

    fn request() -> TurnRequest {
        TurnRequest {
            session_id: SessionId::new("session"),
            turn_id: TurnId::new("turn"),
            transcript: vec![Item::text(ItemKind::User, "hello")],
            available_tools: Vec::new(),
            cache: None,
            metadata: MetadataMap::new(),
        }
    }

    fn sse_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        headers
    }

    const SUCCESS: &str = r#"event: response.created
data: {"type":"response.created","sequence_number":1,"response":{"id":"resp-1","model":"gpt-test"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg-1","type":"message"}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"item_id":"msg-1","output_index":0,"content_index":0,"part":{"type":"output_text"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg-1","output_index":0,"content_index":0,"delta":"hello"}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":5,"item_id":"msg-1","output_index":0,"content_index":0,"text":"hello"}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":6,"item_id":"msg-1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"hello"}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"id":"msg-1","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":8,"output_index":1,"item":{"id":"reason-1","type":"reasoning"}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","sequence_number":9,"item_id":"reason-1","output_index":1,"summary_index":0,"part":{"type":"summary_text"}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":10,"item_id":"reason-1","output_index":1,"summary_index":0,"delta":"brief"}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","sequence_number":11,"item_id":"reason-1","output_index":1,"summary_index":0,"text":"brief"}

event: response.reasoning_summary_part.done
data: {"type":"response.reasoning_summary_part.done","sequence_number":12,"item_id":"reason-1","output_index":1,"summary_index":0,"part":{"type":"summary_text","text":"brief"}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":13,"output_index":1,"item":{"id":"reason-1","type":"reasoning","summary":[{"type":"summary_text","text":"brief"}],"encrypted_content":"opaque"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":14,"output_index":2,"item":{"id":"call-item","type":"function_call"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":15,"item_id":"call-item","output_index":2,"delta":"{\"q\":1}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":16,"item_id":"call-item","output_index":2,"arguments":"{\"q\":1}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":17,"output_index":2,"item":{"id":"call-item","type":"function_call","call_id":"call-1","name":"lookup","arguments":"{\"q\":1}"}}

event: response.completed
data: {"type":"response.completed","sequence_number":18,"response":{"id":"resp-1","model":"gpt-test","usage":{"input_tokens":3,"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2}}}}

"#;

    #[test]
    fn public_and_private_profiles_encode_distinct_fields() {
        let mut request = request();
        request.transcript[0] = Item::text(ItemKind::System, "rules");

        let public = OpenAIResponsesConfig::new("secret", "gpt-test")
            .with_max_output_tokens(99)
            .encode_request(&request)
            .unwrap();
        assert_eq!(public.pointer("/input/0/role"), Some(&json!("system")));
        assert_eq!(public["max_output_tokens"], 99);
        assert_eq!(public["include"], json!(["reasoning.encrypted_content"]));

        let private =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"))
                .with_max_output_tokens(99)
                .encode_request(&request)
                .unwrap();
        assert_eq!(private.pointer("/input/0/role"), Some(&json!("developer")));
        assert!(private.get("max_output_tokens").is_none());
        assert_eq!(private["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(private["parallel_tool_calls"], true);
    }

    #[tokio::test]
    async fn refreshes_once_and_replays_stable_body_and_idempotency_key_before_output() {
        let calls = Arc::new(AtomicUsize::new(0));
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::UNAUTHORIZED,
                    headers: HeaderMap::new(),
                    body: "",
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"discarded-response\",\"model\":\"gpt-test\"}}\n\n",
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: SUCCESS,
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config = OpenAIResponsesConfig::public(
            "gpt-test",
            Authentication::new(RefreshingAuth {
                calls: calls.clone(),
            }),
        )
        .with_resilience(ResilienceConfig {
            max_retries: 1,
            retry_budget: Duration::from_secs(1),
            attempt_timeout: None,
            stream_idle_timeout: None,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let mut events = Vec::new();
        while let Some(event) = turn.next_event(None).await.unwrap() {
            events.push(event);
        }

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let requests = client.requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].body, requests[1].body);
        assert_eq!(requests[1].body, requests[2].body);
        let keys: Vec<_> = requests
            .iter()
            .map(|value| value.headers["idempotency-key"].clone())
            .collect();
        assert!(keys.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(requests[0].headers[header::AUTHORIZATION], "Bearer first");
        assert_eq!(
            requests[1].headers[header::AUTHORIZATION],
            "Bearer refreshed"
        );
        assert_eq!(
            requests[2].headers[header::AUTHORIZATION],
            "Bearer refreshed"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTurnEvent::Delta(Delta::AppendText { chunk, .. }) if chunk == "hello"
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            ModelTurnEvent::Delta(Delta::AppendText { chunk, .. }) if chunk == "discarded"
        )));
        assert!(
            events.iter().any(
                |event| matches!(event, ModelTurnEvent::ToolCall(call) if call.name == "lookup")
            )
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTurnEvent::Delta(Delta::CommitPart { part: Part::ToolCall(call) })
                if call.name == "lookup"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            ModelTurnEvent::Usage(usage)
                if usage.tokens.as_ref().is_some_and(|tokens| tokens.reasoning_tokens == Some(2))
        )));
        assert!(
            matches!(events.last(), Some(ModelTurnEvent::Finished(result)) if result.finish_reason == FinishReason::ToolCall && result.response_id.as_deref() == Some("resp-1"))
        );
    }

    #[tokio::test]
    async fn reactive_refresh_rejects_changed_authentication_binding() {
        let calls = Arc::new(AtomicUsize::new(0));
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([WireResponse {
                status: StatusCode::UNAUTHORIZED,
                headers: HeaderMap::new(),
                body: "",
            }])),
            requests: Mutex::new(Vec::new()),
        });
        let config = OpenAIResponsesConfig::public(
            "gpt-test",
            Authentication::new(BindingChangingAuth {
                calls: calls.clone(),
            }),
        )
        .with_endpoint("https://example.test/responses");
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let error = match session.begin_turn(request(), None).await {
            Ok(_) => panic!("changed authentication binding unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("authentication binding changed"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(client.requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn retries_retryable_status_and_honors_none_as_no_retry() {
        for (resilience, expected_requests, succeeds) in [
            (
                Some(ResilienceConfig {
                    max_retries: 1,
                    retry_budget: Duration::from_secs(1),
                    attempt_timeout: None,
                    stream_idle_timeout: None,
                    initial_backoff: Duration::ZERO,
                    max_backoff: Duration::ZERO,
                }),
                2,
                true,
            ),
            (None, 1, false),
        ] {
            let client = Arc::new(ScriptedClient {
                responses: Mutex::new(VecDeque::from([
                    WireResponse {
                        status: StatusCode::SERVICE_UNAVAILABLE,
                        headers: HeaderMap::new(),
                        body: "",
                    },
                    WireResponse {
                        status: StatusCode::OK,
                        headers: sse_headers(),
                        body: SUCCESS,
                    },
                ])),
                requests: Mutex::new(Vec::new()),
            });
            let mut config = OpenAIResponsesConfig::new("secret", "gpt-test");
            config.resilience = resilience;
            let adapter =
                OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
            let mut session = adapter
                .start_session(SessionConfig::new("session"))
                .await
                .unwrap();
            let result = session.begin_turn(request(), None).await;
            assert_eq!(result.is_ok(), succeeds);
            assert_eq!(client.requests.lock().unwrap().len(), expected_requests);
        }
    }

    #[test]
    fn strict_lifecycle_rejects_invalid_order_sequence_and_terminal_events() {
        let before_create = b"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"id\":\"x\",\"type\":\"message\"}}\n\n";
        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        assert!(decoder.push(before_create).is_err());

        let duplicate_sequence = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"r\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1,\"response\":{\"id\":\"r\",\"model\":\"gpt-test\"}}\n\n",
        );
        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        assert!(decoder.push(duplicate_sequence.as_bytes()).is_err());

        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        decoder.push(SUCCESS.as_bytes()).unwrap();
        decoder.process_all_pending().unwrap();
        while decoder.pop_event().is_some() {}
        let trailing = b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":19,\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-test\"}}\n\n";
        assert!(decoder.push(trailing).is_err());
    }

    #[test]
    fn incomplete_is_an_explicit_terminal_with_normalized_finish_reason() {
        let wire = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"r\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"sequence_number\":2,\"response\":{\"id\":\"r\",\"model\":\"gpt-test\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
        );
        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        decoder.push(wire.as_bytes()).unwrap();
        let events = decoder.finish().unwrap();
        assert!(
            matches!(events.back(), Some(ModelTurnEvent::Finished(result)) if result.finish_reason == FinishReason::MaxTokens)
        );

        let unknown = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"r2\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"sequence_number\":2,\"response\":{\"id\":\"r2\",\"model\":\"gpt-test\",\"incomplete_details\":{\"reason\":\"provider_new_reason\"}}}\n\n",
        );
        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        decoder.push(unknown.as_bytes()).unwrap();
        let events = decoder.finish().unwrap();
        let Some(ModelTurnEvent::Finished(result)) = events.back() else {
            panic!("incomplete response must finish");
        };
        assert_eq!(
            result.finish_reason,
            FinishReason::Other("provider_new_reason".into())
        );
        assert_eq!(
            result.metadata[agentkit_loop::PROVIDER_FINISH_REASONS_METADATA_KEY],
            json!(["provider_new_reason"])
        );
    }

    #[test]
    fn continuation_is_versioned_bound_and_preserves_function_item_id() {
        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        decoder.push(SUCCESS.as_bytes()).unwrap();
        let events = decoder.finish().unwrap();
        let result = events
            .iter()
            .find_map(|event| match event {
                ModelTurnEvent::Finished(result) => Some(result),
                _ => None,
            })
            .unwrap();
        assert!(result.output_items.iter().all(|item| item.id.is_some()));
        let mut replay = request();
        replay.transcript = result.output_items.clone();
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"));
        let encoded =
            encode_request_bound(&config, &replay, Some("test-authentication-binding")).unwrap();
        let input = encoded["input"].as_array().unwrap();
        assert!(
            input
                .iter()
                .any(|item| item.get("type") == Some(&json!("function_call"))
                    && item.get("id") == Some(&json!("call-item")))
        );
        assert!(
            input
                .iter()
                .any(|item| item.get("type") == Some(&json!("reasoning"))
                    && item.get("id") == Some(&json!("reason-1"))
                    && item.get("encrypted_content") == Some(&json!("opaque")))
        );
        let metadata = result
            .output_items
            .iter()
            .flat_map(|item| &item.parts)
            .find_map(|part| match part {
                Part::Reasoning(reasoning) => reasoning.metadata.get(CONTINUATION_METADATA),
                _ => None,
            })
            .unwrap();
        assert_eq!(metadata["schema_version"], CONTINUATION_SCHEMA_VERSION);
        assert_eq!(
            metadata["authentication_binding"],
            "test-authentication-binding"
        );
        assert_eq!(metadata["model"], "gpt-test");
        assert_eq!(metadata["session_id"], "session");
        assert!(metadata.get("response_id").is_none());
        assert!(metadata.get("output_index").is_none());

        let mismatched = encode_request_bound(&config, &replay, Some("other-binding")).unwrap();
        let mismatched = mismatched["input"].as_array().unwrap();
        assert!(mismatched.iter().all(|item| {
            item.get("type") != Some(&json!("reasoning"))
                && item.get("id") != Some(&json!("call-item"))
        }));

        let mut malformed = replay.clone();
        let reasoning = malformed
            .transcript
            .iter_mut()
            .flat_map(|item| &mut item.parts)
            .find_map(|part| match part {
                Part::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .unwrap();
        reasoning
            .metadata
            .insert(CONTINUATION_METADATA.into(), json!("malformed"));
        assert!(matches!(
            encode_request_bound(&config, &malformed, Some("test-authentication-binding")),
            Err(OpenAIResponsesError::Protocol(_))
        ));
    }

    #[test]
    fn continuation_metadata_is_not_emitted_without_authentication_binding() {
        let mut decoder = ResponsesSseDecoder::with_policy(
            "gpt-test",
            "session",
            OpenAIResponsesProfile::Public,
            true,
            None,
            Arc::new(Mutex::new(None)),
            DEFAULT_MAX_ATTEMPT_BYTES,
        );
        decoder.push(SUCCESS.as_bytes()).unwrap();
        let events = decoder.finish().unwrap();
        let result = events
            .iter()
            .find_map(|event| match event {
                ModelTurnEvent::Finished(result) => Some(result),
                _ => None,
            })
            .unwrap();
        assert!(
            result
                .output_items
                .iter()
                .flat_map(|item| &item.parts)
                .all(|part| {
                    match part {
                        Part::Reasoning(reasoning) => reasoning.metadata.is_empty(),
                        Part::ToolCall(call) => call.metadata.is_empty(),
                        _ => true,
                    }
                })
        );
    }

    #[test]
    fn public_profile_keeps_unencrypted_reasoning_as_reasoning() {
        let wire = SUCCESS.replace(",\"encrypted_content\":\"opaque\"", "");
        let mut decoder = ResponsesSseDecoder::with_policy(
            "gpt-test",
            "session",
            OpenAIResponsesProfile::Public,
            false,
            Some("test-authentication-binding"),
            Arc::new(Mutex::new(None)),
            DEFAULT_MAX_ATTEMPT_BYTES,
        );
        decoder.push(wire.as_bytes()).unwrap();
        let events = decoder.finish().unwrap();
        let result = events
            .iter()
            .find_map(|event| match event {
                ModelTurnEvent::Finished(result) => Some(result),
                _ => None,
            })
            .unwrap();
        assert!(result.output_items.iter().flat_map(|item| &item.parts).any(|part| {
            matches!(part, Part::Reasoning(reasoning) if reasoning.summary.as_deref() == Some("brief") && !reasoning.redacted)
        }));
    }

    #[test]
    fn role_validation_and_media_encoding_are_strict() {
        let mut invalid = request();
        invalid.transcript = vec![Item::new(
            ItemKind::User,
            vec![Part::Reasoning(ReasoningPart::summary("do not expose"))],
        )];
        assert!(
            OpenAIResponsesConfig::new("secret", "gpt-test")
                .encode_request(&invalid)
                .is_err()
        );

        let mut request = request();
        request.transcript = vec![Item::new(
            ItemKind::User,
            vec![Part::media(
                Modality::Image,
                "image/png",
                DataRef::InlineBytes(vec![1, 2, 3]),
            )],
        )];
        let encoded = OpenAIResponsesConfig::new("secret", "gpt-test")
            .encode_request(&request)
            .unwrap();
        let content = encoded
            .pointer("/input/0/content")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "input_image");
        assert!(
            content[0]["image_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[tokio::test]
    async fn private_profile_scopes_turn_state_to_accepted_logical_turn() {
        let mut headers = sse_headers();
        headers.insert(X_CODEX_TURN_STATE, HeaderValue::from_static("state-1"));
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    headers: headers.clone(),
                    body: "",
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: headers.clone(),
                    body: SUCCESS,
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers,
                    body: SUCCESS,
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"))
                .with_resilience(ResilienceConfig {
                    max_retries: 1,
                    retry_budget: Duration::from_secs(1),
                    attempt_timeout: None,
                    stream_idle_timeout: None,
                    initial_backoff: Duration::ZERO,
                    max_backoff: Duration::ZERO,
                });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        session.begin_turn(request(), None).await.unwrap();
        let mut second = request();
        second.turn_id = TurnId::new("turn-2");
        session.begin_turn(second, None).await.unwrap();
        let requests = client.requests.lock().unwrap();
        assert_eq!(requests[0].headers["originator"], "agentkit");
        assert_eq!(requests[0].headers["session-id"], "session");
        assert!(!requests[0].headers.contains_key(X_CODEX_TURN_STATE));
        assert!(!requests[1].headers.contains_key(X_CODEX_TURN_STATE));
        assert!(!requests[2].headers.contains_key(X_CODEX_TURN_STATE));
        assert_eq!(requests[0].body, requests[1].body);
        assert_eq!(
            requests[0].headers["idempotency-key"],
            requests[1].headers["idempotency-key"]
        );
    }

    #[tokio::test]
    async fn response_metadata_turn_state_updates_same_turn_retry_context() {
        const FAILED: &str = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.metadata\ndata: {\"type\":\"response.metadata\",\"sequence_number\":2,\"headers\":{\"X-Codex-Turn-State\":[\"retry-state\"]}}\n\n",
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":3,\"response\":{\"id\":\"resp-1\",\"model\":\"gpt-test\",\"error\":{\"code\":\"server_error\"}}}\n\n",
        );
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: FAILED,
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: SUCCESS,
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"))
                .with_resilience(ResilienceConfig {
                    max_retries: 1,
                    retry_budget: Duration::from_secs(1),
                    attempt_timeout: None,
                    stream_idle_timeout: None,
                    initial_backoff: Duration::ZERO,
                    max_backoff: Duration::ZERO,
                });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        while turn.next_event(None).await.unwrap().is_some() {}
        let requests = client.requests.lock().unwrap();
        assert!(!requests[0].headers.contains_key(X_CODEX_TURN_STATE));
        assert_eq!(requests[1].headers[X_CODEX_TURN_STATE], "retry-state");
        assert_eq!(requests[0].body, requests[1].body);
    }

    struct SlowAuthentication;

    #[async_trait]
    impl AuthenticationProvider for SlowAuthentication {
        async fn authenticate(
            &self,
            _previous: Option<&AuthenticationAttempt>,
        ) -> Result<AuthenticationAttempt, HttpError> {
            sleep(Duration::from_millis(100)).await;
            Ok(AuthenticationAttempt::stateless(HeaderMap::new()))
        }
    }

    #[test]
    fn prompt_cache_modes_keys_and_redacted_debug_are_validated() {
        let config = OpenAIResponsesConfig::new("super-secret", "gpt-test");
        assert!(!format!("{config:?}").contains("super-secret"));

        let mut disabled = request();
        disabled.cache = Some(agentkit_loop::PromptCacheRequest::disabled().with_key("ignored"));
        let body = config.encode_request(&disabled).unwrap();
        assert!(body.get("prompt_cache_key").is_none());

        let mut explicit = request();
        explicit.cache = Some(agentkit_loop::PromptCacheRequest::explicit_required([]));
        assert!(config.encode_request(&explicit).is_err());

        let mut invalid_key = request();
        invalid_key.cache =
            Some(agentkit_loop::PromptCacheRequest::automatic().with_key("x".repeat(257)));
        assert!(config.encode_request(&invalid_key).is_err());
    }

    #[tokio::test]
    async fn logical_deadline_starts_before_initial_authentication() {
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        });
        let config =
            OpenAIResponsesConfig::public("gpt-test", Authentication::new(SlowAuthentication))
                .with_resilience(ResilienceConfig {
                    max_retries: 1,
                    retry_budget: Duration::from_millis(10),
                    attempt_timeout: Some(Duration::from_secs(1)),
                    stream_idle_timeout: Some(Duration::from_secs(1)),
                    initial_backoff: Duration::ZERO,
                    max_backoff: Duration::ZERO,
                });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let error = session.begin_turn(request(), None).await.err().unwrap();
        assert!(error.to_string().contains("logical request retry budget"));
        assert!(client.requests.lock().unwrap().is_empty());
    }

    struct SlowRefreshAuthentication(AtomicUsize);

    #[async_trait]
    impl AuthenticationProvider for SlowRefreshAuthentication {
        async fn authenticate(
            &self,
            previous: Option<&AuthenticationAttempt>,
        ) -> Result<AuthenticationAttempt, HttpError> {
            let call = self.0.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                assert!(previous.is_none());
            } else {
                assert!(previous.is_some());
                sleep(Duration::from_millis(100)).await;
            }
            Ok(AuthenticationAttempt::new(HeaderMap::new(), call))
        }
    }

    #[tokio::test]
    async fn logical_deadline_also_bounds_reactive_refresh() {
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([WireResponse {
                status: StatusCode::UNAUTHORIZED,
                headers: HeaderMap::new(),
                body: "",
            }])),
            requests: Mutex::new(Vec::new()),
        });
        let config = OpenAIResponsesConfig::public(
            "gpt-test",
            Authentication::new(SlowRefreshAuthentication(AtomicUsize::new(0))),
        )
        .with_resilience(ResilienceConfig {
            max_retries: 1,
            retry_budget: Duration::from_millis(10),
            attempt_timeout: Some(Duration::from_secs(1)),
            stream_idle_timeout: Some(Duration::from_secs(1)),
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let error = session.begin_turn(request(), None).await.err().unwrap();
        assert!(error.to_string().contains("logical request retry budget"));
    }

    #[test]
    fn request_helpers_preserve_boundaries_cache_and_idempotency() {
        assert_eq!(frame_end(b"data: a\n\nlater\r\n\r\n"), Some((7, 2)));
        assert_ne!(
            stable_idempotency_key("session", "turn", b"one"),
            stable_idempotency_key("session", "turn", b"two")
        );

        let mut request = request();
        request.cache = Some(
            agentkit_loop::PromptCacheRequest::automatic()
                .with_retention(agentkit_loop::PromptCacheRetention::Extended),
        );
        let public = OpenAIResponsesConfig::new("secret", "gpt-test")
            .encode_request(&request)
            .unwrap();
        assert_eq!(public["prompt_cache_retention"], "24h");
        let private =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"))
                .encode_request(&request)
                .unwrap();
        assert!(private.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn notifications_are_wrapped_once_and_invalid_tools_and_audio_are_rejected() {
        let mut notification = request();
        notification.transcript = vec![Item::new(
            ItemKind::Notification,
            vec![Part::text("notice"), Part::structured(json!({"code": 7}))],
        )];
        let encoded = OpenAIResponsesConfig::new("secret", "gpt-test")
            .encode_request(&notification)
            .unwrap();
        assert_eq!(
            encoded
                .pointer("/input/0/content/0/text")
                .and_then(Value::as_str),
            Some("<system-reminder>\nnotice\n\n{\n  \"code\": 7\n}\n</system-reminder>")
        );

        let mut invalid_tool = request();
        invalid_tool.transcript = vec![Item::new(
            ItemKind::Assistant,
            vec![Part::ToolCall(ToolCallPart::new(
                "call",
                "not a valid tool",
                json!({}),
            ))],
        )];
        assert!(
            OpenAIResponsesConfig::new("secret", "gpt-test")
                .encode_request(&invalid_tool)
                .is_err()
        );

        let mut audio = request();
        audio.transcript = vec![Item::new(
            ItemKind::User,
            vec![Part::media(
                Modality::Audio,
                "audio/wav",
                DataRef::InlineBytes(vec![1, 2, 3]),
            )],
        )];
        assert!(
            OpenAIResponsesConfig::new("secret", "gpt-test")
                .encode_request(&audio)
                .is_err()
        );
    }

    #[test]
    fn refusal_and_keepalive_decode_without_losing_provider_item_id() {
        let body = concat!(
            "event: keepalive\ndata: {\"type\":\"keepalive\",\"sequence_number\":0}\n\n",
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"resp-refusal\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"refusal-item\",\"type\":\"message\"}}\n\n",
            "event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":3,\"item_id\":\"refusal-item\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"refusal\"}}\n\n",
            "event: response.refusal.delta\ndata: {\"type\":\"response.refusal.delta\",\"sequence_number\":4,\"item_id\":\"refusal-item\",\"output_index\":0,\"content_index\":0,\"delta\":\"no\"}\n\n",
            "event: response.refusal.done\ndata: {\"type\":\"response.refusal.done\",\"sequence_number\":5,\"item_id\":\"refusal-item\",\"output_index\":0,\"content_index\":0,\"refusal\":\"no\"}\n\n",
            "event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":6,\"item_id\":\"refusal-item\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"refusal\",\"refusal\":\"no\"}}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":7,\"output_index\":0,\"item\":{\"id\":\"refusal-item\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"no\"}]}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":8,\"response\":{\"id\":\"resp-refusal\",\"model\":\"gpt-test\",\"status\":\"completed\"}}\n\n",
        );
        let mut decoder = ResponsesSseDecoder::new("gpt-test", "session");
        decoder.push(body.as_bytes()).unwrap();
        let result = decoder
            .finish()
            .unwrap()
            .into_iter()
            .find_map(|event| match event {
                ModelTurnEvent::Finished(result) => Some(result),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            result.output_items[0].id.as_ref().unwrap().0,
            "refusal-item"
        );
        assert!(matches!(
            &result.output_items[0].parts[0],
            Part::Text(text) if text.text == "no"
        ));
    }

    #[tokio::test]
    async fn static_bearer_authentication_has_a_stable_non_secret_binding() {
        let authentication = Authentication::bearer("secret");
        let first = authentication.authenticate(None).await.unwrap();
        let second = authentication.authenticate(Some(&first)).await.unwrap();
        assert!(first.binding().is_some());
        assert_eq!(first.binding(), second.binding());
        assert_ne!(first.binding(), Some("secret"));
    }

    #[tokio::test]
    async fn visible_stream_failure_is_fatal_without_replacement_capability() {
        const PARTIAL: &str = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"failed\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"failed-item\",\"type\":\"message\"}}\n\n",
            "event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":3,\"item_id\":\"failed-item\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\"}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":4,\"item_id\":\"failed-item\",\"output_index\":0,\"content_index\":0,\"delta\":\"visible\"}\n\n",
        );
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: PARTIAL,
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: SUCCESS,
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config =
            OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(ResilienceConfig {
                max_retries: 1,
                retry_budget: Duration::from_secs(1),
                attempt_timeout: None,
                stream_idle_timeout: None,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
            });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let mut saw_visible = false;
        let error = loop {
            match turn.next_event(None).await {
                Ok(Some(ModelTurnEvent::Delta(Delta::AppendText { chunk, .. }))) => {
                    saw_visible |= chunk == "visible";
                }
                Ok(Some(_)) => {}
                Ok(None) => panic!("truncated attempt unexpectedly completed"),
                Err(error) => break error,
            }
        };
        assert!(saw_visible);
        assert!(matches!(error, LoopError::Provider(_)));
        assert_eq!(client.requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn opted_in_visible_stream_failure_emits_marker_then_replays() {
        const PARTIAL: &str = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"failed\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"failed-item\",\"type\":\"message\"}}\n\n",
            "event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":3,\"item_id\":\"failed-item\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\"}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":4,\"item_id\":\"failed-item\",\"output_index\":0,\"content_index\":0,\"delta\":\"replace me\"}\n\n",
        );
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: PARTIAL,
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: SUCCESS,
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config =
            OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(ResilienceConfig {
                max_retries: 1,
                retry_budget: Duration::from_secs(1),
                attempt_timeout: None,
                stream_idle_timeout: None,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
            });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session_config = SessionConfig::new("session");
        agentkit_loop::response_attempt::enable(&mut session_config);
        let mut session = adapter.start_session(session_config).await.unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let mut saw_marker = false;
        let mut saw_replacement = false;
        while let Some(event) = turn.next_event(None).await.unwrap() {
            match event {
                ModelTurnEvent::Delta(ref delta)
                    if agentkit_loop::response_attempt::is_marker(delta) =>
                {
                    saw_marker = true;
                }
                ModelTurnEvent::Delta(Delta::AppendText { chunk, .. }) if chunk == "hello" => {
                    saw_replacement = true;
                }
                _ => {}
            }
        }
        assert!(saw_marker);
        assert!(saw_replacement);
        assert_eq!(client.requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn performs_only_one_reactive_refresh() {
        let calls = Arc::new(AtomicUsize::new(0));
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::UNAUTHORIZED,
                    headers: HeaderMap::new(),
                    body: "",
                },
                WireResponse {
                    status: StatusCode::UNAUTHORIZED,
                    headers: HeaderMap::new(),
                    body: "",
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config = OpenAIResponsesConfig::public(
            "gpt-test",
            Authentication::new(RefreshingAuth {
                calls: calls.clone(),
            }),
        );
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        assert!(session.begin_turn(request(), None).await.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(client.requests.lock().unwrap().len(), 2);
    }

    #[test]
    fn private_audio_and_media_tool_outputs_preserve_multimodal_content() {
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"));
        let mut audio = request();
        audio.transcript = vec![Item::new(
            ItemKind::User,
            vec![Part::media(
                Modality::Audio,
                "audio/wav",
                DataRef::InlineBytes(vec![1, 2, 3]),
            )],
        )];
        let encoded = config.encode_request(&audio).unwrap();
        assert_eq!(encoded["input"][0]["content"][0]["type"], "input_audio");
        assert_eq!(
            encoded["input"][0]["content"][0]["audio_url"],
            "data:audio/wav;base64,AQID"
        );

        let mut tool = request();
        tool.transcript = vec![Item::new(
            ItemKind::Tool,
            vec![Part::ToolResult(agentkit_core::ToolResultPart::success(
                "call-1",
                ToolOutput::parts(vec![
                    Part::text("screenshot"),
                    Part::media(
                        Modality::Image,
                        "image/png",
                        DataRef::InlineBytes(vec![1, 2, 3]),
                    ),
                ]),
            ))],
        )];
        let encoded = config.encode_request(&tool).unwrap();
        let output = encoded["input"][0]["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "input_text");
        assert_eq!(output[1]["type"], "input_image");
    }

    #[test]
    fn media_validation_rejects_noncanonical_base64_and_unsafe_uris() {
        let config = OpenAIResponsesConfig::new("secret", "gpt-test");
        for data in [
            DataRef::InlineText("%%%".into()),
            DataRef::Uri("file:///tmp/private.png".into()),
            DataRef::Uri("custom://artifact/image".into()),
            DataRef::Uri("data:image/jpeg;base64,AQID".into()),
        ] {
            let mut invalid = request();
            invalid.transcript = vec![Item::new(
                ItemKind::User,
                vec![Part::media(Modality::Image, "image/png", data)],
            )];
            assert!(config.encode_request(&invalid).is_err());
        }
        let mut valid = request();
        valid.transcript = vec![Item::new(
            ItemKind::User,
            vec![Part::media(
                Modality::Image,
                "image/png",
                DataRef::Uri("https://example.com/image.png".into()),
            )],
        )];
        assert!(config.encode_request(&valid).is_ok());
    }

    #[test]
    fn generated_image_output_is_persisted_and_replayed() {
        let wire = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":1,\"response\":{\"id\":\"resp-image\",\"model\":\"gpt-test\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"image-1\",\"type\":\"image_generation_call\"}}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":3,\"output_index\":0,\"item\":{\"id\":\"image-1\",\"type\":\"image_generation_call\",\"status\":\"completed\",\"revised_prompt\":\"safer prompt\",\"result\":\"AQID\"}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"resp-image\",\"model\":\"gpt-test\",\"status\":\"completed\"}}\n\n",
        );
        let mut decoder = ResponsesSseDecoder::with_policy(
            "gpt-test",
            "session",
            OpenAIResponsesProfile::ChatGptPrivate,
            true,
            Some("test-authentication-binding"),
            Arc::new(Mutex::new(None)),
            DEFAULT_MAX_ATTEMPT_BYTES,
        );
        decoder.push(wire.as_bytes()).unwrap();
        let events = decoder.finish().unwrap();
        let result = events
            .iter()
            .find_map(|event| match event {
                ModelTurnEvent::Finished(result) => Some(result),
                _ => None,
            })
            .unwrap();
        let Part::Media(media) = &result.output_items[0].parts[0] else {
            panic!("generated image was not persisted as media");
        };
        assert_eq!(media.data, DataRef::InlineBytes(vec![1, 2, 3]));
        assert!(media.metadata.contains_key(CONTINUATION_METADATA));
        assert!(!media.metadata.contains_key(LEGACY_CONTINUATION_METADATA));

        let replay = TurnRequest {
            transcript: result.output_items.clone(),
            ..request()
        };
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"));
        let encoded =
            encode_request_bound(&config, &replay, Some("test-authentication-binding")).unwrap();
        assert_eq!(encoded["input"][0]["type"], "image_generation_call");
        assert_eq!(encoded["input"][0]["result"], "AQID");
    }

    #[test]
    fn private_unknown_events_are_ignored_and_statusless_failures_retry_unless_permanent() {
        let unknown = b"event: response.future_hint\ndata: {\"type\":\"response.future_hint\",\"sequence_number\":1}\n\n";
        let mut private = ResponsesSseDecoder::with_policy(
            "gpt-test",
            "session",
            OpenAIResponsesProfile::ChatGptPrivate,
            true,
            Some("binding"),
            Arc::new(Mutex::new(None)),
            DEFAULT_MAX_ATTEMPT_BYTES,
        );
        assert!(private.push(unknown).is_ok());
        let mut public = ResponsesSseDecoder::new("gpt-test", "session");
        assert!(public.push(unknown).is_err());

        assert!(stream_failure_retryable(
            OpenAIResponsesProfile::ChatGptPrivate,
            &json!({"type": "error", "error": {"type": "brand_new_error", "code": "never_seen_before"}}),
            "error",
        ));
        assert!(!stream_failure_retryable(
            OpenAIResponsesProfile::ChatGptPrivate,
            &json!({"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_prompt"}}),
            "error",
        ));
    }

    #[test]
    fn legacy_subscription_continuation_requires_all_bindings() {
        let legacy = json!({
            "schema_version": 1,
            "account_binding": {
                "account_id_digest": "a".repeat(64),
                "login_generation": "generation-1",
            },
            "model": "gpt-test",
            "session_id": "session",
            "response_id": "resp-1",
            "item_id": "call-item-1",
            "output_index": 0,
            "kind": "function_call",
        });
        let call = ToolCallPart::new("call-1", "tool", json!({})).with_metadata(MetadataMap::from(
            [(LEGACY_CONTINUATION_METADATA.into(), legacy)],
        ));
        let replay = TurnRequest {
            transcript: vec![Item::new(ItemKind::Assistant, vec![Part::ToolCall(call)])],
            ..request()
        };
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"))
                .with_legacy_subscription_continuation_authenticator(|account, current| {
                    current == "current-binding" && account["login_generation"] == "generation-1"
                });
        let encoded = encode_request_bound(&config, &replay, Some("current-binding")).unwrap();
        assert_eq!(encoded["input"][0]["id"], "call-item-1");
        let not_bound = encode_request_bound(&config, &replay, Some("other-binding")).unwrap();
        assert!(not_bound["input"][0].get("id").is_none());

        let public = OpenAIResponsesConfig::new("secret", "gpt-test")
            .with_legacy_subscription_continuation_authenticator(|_, _| true);
        let public = encode_request_bound(&public, &replay, Some("current-binding")).unwrap();
        assert!(public["input"][0].get("id").is_none());
    }

    #[test]
    fn serialized_request_limit_is_configurable() {
        let config =
            OpenAIResponsesConfig::new("secret", "gpt-test").with_limits(OpenAIResponsesLimits {
                max_request_bytes: 16,
                ..OpenAIResponsesLimits::default()
            });
        assert!(config.encode_request(&request()).is_err());
    }

    #[tokio::test]
    async fn private_http_529_retries_and_custom_attribution_is_sent() {
        let status_529 = StatusCode::from_u16(529).unwrap();
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: status_529,
                    headers: HeaderMap::new(),
                    body: "",
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: SUCCESS,
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config =
            OpenAIResponsesConfig::chatgpt_private("gpt-test", Authentication::bearer("secret"))
                .with_user_agent("kit/test")
                .with_originator("kit")
                .with_resilience(ResilienceConfig {
                    max_retries: 1,
                    retry_budget: Duration::from_secs(1),
                    attempt_timeout: None,
                    stream_idle_timeout: None,
                    initial_backoff: Duration::ZERO,
                    max_backoff: Duration::ZERO,
                });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        session.begin_turn(request(), None).await.unwrap();
        let requests = client.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].headers["user-agent"], "kit/test");
        assert_eq!(requests[0].headers["originator"], "kit");
    }

    #[tokio::test]
    async fn attempt_deadline_bounds_continuously_active_stream() {
        let client = Arc::new(ActiveStreamClient {
            requests: AtomicUsize::new(0),
        });
        let config =
            OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(ResilienceConfig {
                max_retries: 0,
                retry_budget: Duration::from_secs(1),
                attempt_timeout: Some(Duration::from_millis(10)),
                stream_idle_timeout: Some(Duration::from_millis(100)),
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
            });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let error = turn.next_event(None).await.unwrap_err();
        assert!(error.to_string().contains("attempt timed out"));
        assert_eq!(client.requests.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn aggregate_wire_limit_spans_retries() {
        let client = Arc::new(ScriptedClient {
            responses: Mutex::new(VecDeque::from([
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: "1234567890",
                },
                WireResponse {
                    status: StatusCode::OK,
                    headers: sse_headers(),
                    body: "1234567890",
                },
            ])),
            requests: Mutex::new(Vec::new()),
        });
        let config = OpenAIResponsesConfig::new("secret", "gpt-test")
            .with_limits(OpenAIResponsesLimits {
                max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
                max_attempt_bytes: 16,
                max_wire_bytes: 15,
            })
            .with_resilience(ResilienceConfig {
                max_retries: 3,
                retry_budget: Duration::from_secs(1),
                attempt_timeout: None,
                stream_idle_timeout: None,
                initial_backoff: Duration::ZERO,
                max_backoff: Duration::ZERO,
            });
        let adapter = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let error = turn.next_event(None).await.unwrap_err();
        assert!(error.to_string().contains("wire-byte limit"));
        assert_eq!(client.requests.lock().unwrap().len(), 2);
    }
}
