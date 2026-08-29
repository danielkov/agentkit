//! Anthropic Messages API adapter for the agentkit agent loop.
//!
//! This crate implements the agentkit [`ModelAdapter`] directly against
//! Anthropic's `/v1/messages` endpoint. The API is not OpenAI-compatible
//! (different message shape, `system` is top-level, tool results live as
//! content blocks inside user messages, etc.), so the generic completions
//! adapter is not reused.
//!
//! Streaming is on by default: the adapter consumes Anthropic's SSE response
//! and yields `ModelTurnEvent`s as tokens arrive. Call
//! [`AnthropicConfig::with_streaming(false)`] to opt out in favour of a single
//! buffered request.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use agentkit_loop::{Agent, SessionConfig};
//! use agentkit_provider_anthropic::{AnthropicAdapter, AnthropicConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = AnthropicConfig::from_env()?;
//!     let adapter = AnthropicAdapter::new(config)?;
//!     let agent = Agent::builder().model(adapter).build()?;
//!     let _driver = agent.start(SessionConfig::new("demo")).await?;
//!     Ok(())
//! }
//! ```

mod config;
mod error;
mod media;
mod request;
mod response;
mod server_tool;
mod sse;
mod stream;

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use agentkit_core::TurnCancellation;
use agentkit_http::{
    Authentication, AuthenticationAttempt, AuthenticationProvider, BodyStream, Bytes, Http,
    HttpError, HttpResponse, LogicalDeadline, ResilienceConfig, StatusCode,
    TruncatedStreamDetector, is_retryable_body_read, is_retryable_status, next_body_chunk_bounded,
    run_bounded, sleep as resilience_sleep,
};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use futures_util::future::{BoxFuture, Either, select};

use crate::stream::{EventTranslator, SseDecoder};

pub use crate::config::{
    AnthropicApiKey, AnthropicAuthToken, AnthropicConfig, AnthropicMcpServer,
    DEFAULT_ANTHROPIC_VERSION, DEFAULT_ENDPOINT, OutputEffort, OutputFormat, ServiceTier,
    ThinkingConfig, ToolChoice,
};
pub use crate::error::AnthropicError;
pub use crate::server_tool::{
    BashCodeExecutionTool, CodeExecutionTool, DEFAULT_BASH_EXECUTION_VERSION,
    DEFAULT_CODE_EXECUTION_VERSION, DEFAULT_TEXT_EDITOR_EXECUTION_VERSION,
    DEFAULT_WEB_FETCH_VERSION, DEFAULT_WEB_SEARCH_VERSION, RawServerTool, ServerTool,
    ServerToolHandle, TextEditorCodeExecutionTool, WebFetchTool, WebSearchTool, boxed,
};

/// Model adapter that connects the agentkit agent loop to the Anthropic
/// Messages API.
#[derive(Clone)]
pub struct AnthropicAdapter {
    client: Http,
    config: Arc<AnthropicConfig>,
}

impl AnthropicAdapter {
    /// Creates a new adapter from the given configuration, building a default
    /// reqwest-backed HTTP client.
    pub fn new(config: AnthropicConfig) -> Result<Self, AnthropicError> {
        let client = reqwest::Client::builder()
            .build()
            .map(Http::new)
            .map_err(|error| AnthropicError::HttpClient(HttpError::request(error)))?;
        Ok(Self::with_client(config, client))
    }

    /// Creates a new adapter using a pre-configured [`Http`] client.
    pub fn with_client(config: AnthropicConfig, client: Http) -> Self {
        Self {
            client,
            config: Arc::new(config),
        }
    }

    /// Overrides the configured `x-api-key` or bearer-token authentication.
    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        Arc::make_mut(&mut self.config).authentication = authentication.into();
        self
    }

    /// Overrides authentication with a custom refresh-capable provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(self, provider: P) -> Self {
        self.with_authentication(Authentication::new(provider))
    }

    /// Enables request and pre-visible-output stream retries and timeouts.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        Arc::make_mut(&mut self.config).resilience = Some(resilience);
        self
    }
}

/// An active session with the Anthropic Messages API.
pub struct AnthropicSession {
    client: Http,
    config: Arc<AnthropicConfig>,
    authentication: Option<Authentication>,
    resilience: Option<ResilienceConfig>,
    _session_config: SessionConfig,
}

/// A turn in progress against the Messages API.
///
/// Either runs in buffered (full-JSON) or streaming (SSE) mode depending on
/// [`AnthropicConfig::streaming`]. The variant is private because the
/// streaming state carries opaque decoder/translator types.
pub struct AnthropicTurn {
    inner: TurnInner,
}

enum TurnInner {
    /// Buffered, non-streaming mode.
    Buffered { events: VecDeque<ModelTurnEvent> },
    /// Live SSE stream in progress. Boxed because [`EventTranslator`] carries
    /// a fairly large state machine and SSE responses are a small fraction of
    /// total turns; keeping the enum compact avoids a ~350B stack cost on the
    /// buffered path.
    Streaming(Box<StreamingState>),
}

type StreamReplay = Box<
    dyn FnMut() -> BoxFuture<'static, Result<(BodyStream, TruncatedStreamDetector), LoopError>>
        + Send,
>;

struct StreamingState {
    body: BodyStream,
    decoder: SseDecoder,
    translator: EventTranslator,
    pending: VecDeque<ModelTurnEvent>,
    eof: bool,
    visible_output: bool,
    detect_truncation: bool,
    idle_timeout: Option<Duration>,
    deadline: Option<LogicalDeadline>,
    integrity: TruncatedStreamDetector,
    replay: Option<StreamReplay>,
}

#[derive(Clone)]
struct ReplayRequest {
    client: Http,
    config: Arc<AnthropicConfig>,
    body: Bytes,
    authentication: Option<Authentication>,
    authentication_attempt: Arc<std::sync::Mutex<Option<AuthenticationAttempt>>>,
    reauthenticated: Arc<AtomicBool>,
    resilience: Option<ResilienceConfig>,
    deadline: Option<LogicalDeadline>,
    retries_used: Arc<AtomicUsize>,
}

impl ReplayRequest {
    async fn authentication_attempt(&self) -> Result<AuthenticationAttempt, LoopError> {
        let existing = self
            .authentication_attempt
            .lock()
            .map_err(|_| LoopError::Provider("authentication state lock poisoned".into()))?
            .clone();
        if let Some(existing) = existing {
            return Ok(existing);
        }
        let authentication = self
            .authentication
            .as_ref()
            .ok_or_else(|| LoopError::Provider(AnthropicError::MissingCredentials.to_string()))?;
        let attempt = run_bounded(
            authentication.authenticate(None),
            None,
            self.deadline.as_ref(),
            "authentication",
        )
        .await
        .map_err(|error| LoopError::Provider(format!("authentication failed: {error}")))?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| LoopError::Provider("authentication state lock poisoned".into()))? =
            Some(attempt.clone());
        Ok(attempt)
    }

    async fn reauthenticate(&self, previous: AuthenticationAttempt) -> Result<(), LoopError> {
        let authenticate = self
            .authentication
            .as_ref()
            .expect("401 refresh requires authentication")
            .authenticate(Some(&previous));
        let next = run_bounded(
            authenticate,
            None,
            self.deadline.as_ref(),
            "reauthentication",
        )
        .await
        .map_err(|error| LoopError::Provider(format!("reauthentication failed: {error}")))?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| LoopError::Provider("authentication state lock poisoned".into()))? =
            Some(next);
        Ok(())
    }

    fn reserve_retry(&self) -> Option<usize> {
        let maximum = self.resilience.as_ref()?.max_retries;
        self.retries_used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                (used < maximum).then_some(used + 1)
            })
            .ok()
    }

    fn retry_delay(
        &self,
        retry_number: usize,
        headers: Option<&agentkit_http::HeaderMap>,
    ) -> Duration {
        self.resilience
            .as_ref()
            .expect("reserved retry requires resilience")
            .retry_delay(retry_number, headers)
    }

    async fn wait_before_retry(
        &self,
        retry_number: usize,
        headers: Option<&agentkit_http::HeaderMap>,
    ) -> Result<(), LoopError> {
        self.wait_for_retry_delay(self.retry_delay(retry_number, headers))
            .await
    }

    async fn wait_for_retry_delay(&self, delay: Duration) -> Result<(), LoopError> {
        run_bounded(
            async {
                resilience_sleep(delay).await;
                Ok(())
            },
            None,
            self.deadline.as_ref(),
            "retry backoff",
        )
        .await
        .map_err(|error| LoopError::Provider(format!("retry backoff failed: {error}")))
    }

    async fn execute_attempt(
        &self,
        request: agentkit_http::HttpRequest,
    ) -> Result<HttpResponse, HttpError> {
        let timeout = self
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout);
        run_bounded(
            self.client.execute(request),
            timeout,
            self.deadline.as_ref(),
            "HTTP request attempt",
        )
        .await
    }

    async fn read_response_text(&self, response: HttpResponse) -> Result<String, HttpError> {
        let timeout = self
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout);
        run_bounded(
            response.text(),
            timeout,
            self.deadline.as_ref(),
            "HTTP response body",
        )
        .await
    }

    async fn open_response(&self) -> Result<HttpResponse, LoopError> {
        loop {
            let authentication = self.authentication_attempt().await?;
            let mut builder = self
                .client
                .post(&self.config.base_url)
                .header("Content-Type", "application/json")
                .header("anthropic-version", self.config.anthropic_version.as_str());

            let betas = collect_beta_flags(&self.config);
            if !betas.is_empty() {
                builder = builder.header(
                    "anthropic-beta",
                    betas.into_iter().collect::<Vec<_>>().join(","),
                );
            }
            builder = builder.header(
                "User-Agent",
                concat!("agentkit-provider-anthropic/", env!("CARGO_PKG_VERSION")),
            );
            if self.config.streaming {
                builder = builder.header("Accept", "text/event-stream");
            }
            builder = builder.headers(authentication.headers().clone());
            let request = builder.body(self.body.clone()).build().map_err(|error| {
                LoopError::Provider(format!("Anthropic request failed: {error}"))
            })?;

            match self.execute_attempt(request).await {
                Ok(response) => {
                    if response.status() == StatusCode::UNAUTHORIZED
                        && !self.reauthenticated.swap(true, Ordering::SeqCst)
                    {
                        drop(response);
                        self.reauthenticate(authentication).await?;
                        continue;
                    }
                    if is_retryable_status(response.status())
                        && let Some(retry_number) = self.reserve_retry()
                    {
                        let delay = self.retry_delay(retry_number, Some(response.headers()));
                        drop(response);
                        self.wait_for_retry_delay(delay).await?;
                        continue;
                    }
                    return Ok(response);
                }
                Err(error) if error.is_retryable_transport() => {
                    if let Some(retry_number) = self.reserve_retry() {
                        self.wait_before_retry(retry_number, None).await?;
                        continue;
                    }
                    return Err(LoopError::Provider(format!(
                        "Anthropic request failed: {error}"
                    )));
                }
                Err(error) => {
                    return Err(LoopError::Provider(format!(
                        "Anthropic request failed: {error}"
                    )));
                }
            }
        }
    }

    async fn replay_stream(&self) -> Result<(BodyStream, TruncatedStreamDetector), LoopError> {
        let retry_number = self.reserve_retry().ok_or_else(|| {
            LoopError::Provider(
                "Anthropic stream failed before output and retry budget is exhausted".into(),
            )
        })?;
        self.wait_before_retry(retry_number, None).await?;
        let response = self.open_response().await?;
        if !response.status().is_success() {
            return Err(LoopError::Provider(format!(
                "Anthropic stream replay failed with status {}",
                response.status()
            )));
        }
        let integrity = TruncatedStreamDetector::from_headers(response.headers());
        Ok((response.bytes_stream(), integrity))
    }
}

#[async_trait]
impl ModelAdapter for AnthropicAdapter {
    type Session = AnthropicSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(AnthropicSession {
            client: self.client.clone(),
            config: self.config.clone(),
            authentication: Some(self.config.authentication.clone()),
            resilience: self.config.resilience.clone(),
            _session_config: config,
        })
    }

    fn provider_name(&self) -> Option<&str> {
        Some("anthropic")
    }
}

#[async_trait]
impl ModelSession for AnthropicSession {
    type Turn = AnthropicTurn;

    async fn begin_turn(
        &mut self,
        turn_request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<AnthropicTurn, LoopError> {
        let config = self.config.clone();

        let request_future = async {
            let body = request::build_request_body(&config, &turn_request)
                .map_err(|e| LoopError::Provider(e.to_string()))?;
            let body = serde_json::to_vec(&body)
                .map(Bytes::from)
                .map_err(|e| LoopError::Provider(format!("failed to serialize request: {e}")))?;
            let deadline = self
                .resilience
                .as_ref()
                .map(|resilience| LogicalDeadline::new(resilience.retry_budget));
            let replay_request = ReplayRequest {
                client: self.client.clone(),
                config: config.clone(),
                body,
                authentication: self.authentication.clone(),
                authentication_attempt: Arc::new(std::sync::Mutex::new(None)),
                reauthenticated: Arc::new(AtomicBool::new(false)),
                resilience: self.resilience.clone(),
                deadline: deadline.clone(),
                retries_used: Arc::new(AtomicUsize::new(0)),
            };

            loop {
                let response = replay_request.open_response().await?;
                let status = response.status();
                if config.streaming && status.is_success() {
                    let integrity = TruncatedStreamDetector::from_headers(response.headers());
                    let idle_timeout = self
                        .resilience
                        .as_ref()
                        .and_then(|resilience| resilience.stream_idle_timeout);
                    let replay = self.resilience.as_ref().and_then(|resilience| {
                        (resilience.max_retries > 0).then(|| {
                            let replay_request = replay_request.clone();
                            Box::new(move || {
                                let replay_request = replay_request.clone();
                                Box::pin(async move { replay_request.replay_stream().await })
                                    as BoxFuture<'static, _>
                            }) as StreamReplay
                        })
                    });
                    return Ok(AnthropicTurn {
                        inner: TurnInner::Streaming(Box::new(StreamingState {
                            body: response.bytes_stream(),
                            decoder: SseDecoder::new(),
                            translator: EventTranslator::new(),
                            pending: VecDeque::new(),
                            eof: false,
                            visible_output: false,
                            detect_truncation: self.resilience.is_some(),
                            idle_timeout,
                            deadline: deadline.clone(),
                            integrity,
                            replay,
                        })),
                    });
                }

                let body_text = match replay_request.read_response_text(response).await {
                    Ok(body) => body,
                    Err(error) if is_retryable_body_read(status, &error) => {
                        if let Some(retry_number) = replay_request.reserve_retry() {
                            replay_request.wait_before_retry(retry_number, None).await?;
                            continue;
                        }
                        return Err(LoopError::Provider(format!(
                            "failed to read Anthropic response body: {error}"
                        )));
                    }
                    Err(error) => {
                        return Err(LoopError::Provider(format!(
                            "failed to read Anthropic response body: {error}"
                        )));
                    }
                };

                if !status.is_success() {
                    return Err(LoopError::Provider(format!(
                        "Anthropic request failed with status {status}: {body_text}"
                    )));
                }
                let events = response::build_turn_from_response(&body_text)
                    .map_err(|e| LoopError::Provider(e.to_string()))?;
                return Ok(AnthropicTurn {
                    inner: TurnInner::Buffered { events },
                });
            }
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

    fn model_name(&self) -> Option<&str> {
        Some(&self.config.model)
    }

    fn provider_name(&self) -> Option<&str> {
        Some("anthropic")
    }
}

#[async_trait]
impl ModelTurn for AnthropicTurn {
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
        match &mut self.inner {
            TurnInner::Buffered { events } => Ok(events.pop_front()),
            TurnInner::Streaming(state) => {
                let StreamingState {
                    body,
                    decoder,
                    translator,
                    pending,
                    eof,
                    visible_output,
                    detect_truncation,
                    idle_timeout,
                    deadline,
                    integrity,
                    replay,
                } = state.as_mut();
                next_streaming_event(
                    body,
                    decoder,
                    translator,
                    pending,
                    eof,
                    visible_output,
                    *detect_truncation,
                    *idle_timeout,
                    deadline.as_ref(),
                    integrity,
                    replay,
                    cancellation,
                )
                .await
            }
        }
    }
}

async fn next_stream_chunk(
    body: &mut BodyStream,
    idle_timeout: Option<Duration>,
    deadline: Option<&LogicalDeadline>,
    cancellation: Option<&TurnCancellation>,
) -> Result<Result<Option<Bytes>, HttpError>, LoopError> {
    let next = next_body_chunk_bounded(body, idle_timeout, deadline);
    futures_util::pin_mut!(next);
    if let Some(cancellation) = cancellation {
        let cancelled = cancellation.cancelled();
        futures_util::pin_mut!(cancelled);
        match select(next, cancelled).await {
            Either::Left((chunk, _)) => Ok(chunk),
            Either::Right((_, _)) => Err(LoopError::Cancelled),
        }
    } else {
        Ok(next.await)
    }
}

/// Pulls the next event from an active SSE stream, replaying only failures
/// which occur before any event has become visible to the caller.
#[allow(clippy::too_many_arguments)]
async fn next_streaming_event(
    body: &mut BodyStream,
    decoder: &mut SseDecoder,
    translator: &mut EventTranslator,
    pending: &mut VecDeque<ModelTurnEvent>,
    eof: &mut bool,
    visible_output: &mut bool,
    detect_truncation: bool,
    idle_timeout: Option<Duration>,
    deadline: Option<&LogicalDeadline>,
    integrity: &mut TruncatedStreamDetector,
    replay: &mut Option<StreamReplay>,
    cancellation: Option<TurnCancellation>,
) -> Result<Option<ModelTurnEvent>, LoopError> {
    loop {
        if let Some(event) = pending.pop_front() {
            *visible_output = true;
            return Ok(Some(event));
        }
        if *eof || translator.is_done() {
            return Ok(None);
        }

        let chunk = next_stream_chunk(body, idle_timeout, deadline, cancellation.as_ref()).await?;
        let failure = match chunk {
            Ok(Some(bytes)) => {
                integrity.observe(&bytes);
                let text = std::str::from_utf8(&bytes).map_err(|e| {
                    LoopError::Provider(format!("invalid UTF-8 in Anthropic stream: {e}"))
                })?;
                for sse in decoder.feed(text) {
                    for produced in translator.handle(&sse)? {
                        pending.push_back(produced);
                    }
                }
                None
            }
            Ok(None) if !detect_truncation => {
                *eof = true;
                None
            }
            Ok(None) => {
                *eof = true;
                let message = match integrity.finish() {
                    Ok(()) => "Anthropic stream ended before its terminal event".to_owned(),
                    Err(error) => format!("Anthropic stream body error: {error}"),
                };
                Some(LoopError::Provider(message))
            }
            Err(error) if !error.is_retryable_transport() => {
                return Err(LoopError::Provider(format!(
                    "Anthropic stream body error: {error}"
                )));
            }
            Err(error) => Some(LoopError::Provider(format!(
                "Anthropic stream body error: {error}"
            ))),
        };

        let Some(failure) = failure else {
            continue;
        };
        if *visible_output || replay.is_none() {
            return Err(failure);
        }

        let replay_future = replay.as_mut().expect("checked above")();
        futures_util::pin_mut!(replay_future);
        let replayed = if let Some(cancellation) = cancellation.as_ref() {
            let cancelled = cancellation.cancelled();
            futures_util::pin_mut!(cancelled);
            match select(replay_future, cancelled).await {
                Either::Left((result, _)) => result,
                Either::Right((_, _)) => return Err(LoopError::Cancelled),
            }
        } else {
            replay_future.await
        }?;
        *body = replayed.0;
        *integrity = replayed.1;
        *decoder = SseDecoder::new();
        *translator = EventTranslator::new();
        pending.clear();
        *eof = false;
    }
}

fn collect_beta_flags(config: &AnthropicConfig) -> BTreeSet<String> {
    let mut betas: BTreeSet<String> = config.anthropic_beta.iter().cloned().collect();
    for tool in &config.server_tools {
        for flag in tool.beta_flags() {
            betas.insert(flag);
        }
    }
    betas
}

#[cfg(test)]
mod tests {
    use agentkit_core::{CancellationController, FinishReason};
    use agentkit_http::{HeaderMap, HeaderValue, HttpClient, HttpError, HttpRequest, header};
    use bytes::Bytes;
    use futures_util::stream;

    use super::*;

    struct AlwaysUnauthorized {
        authorizations: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl HttpClient for AlwaysUnauthorized {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            let authorization = request.headers[header::AUTHORIZATION]
                .to_str()
                .unwrap()
                .to_owned();
            self.authorizations.lock().unwrap().push(authorization);
            Ok(HttpResponse::new(
                StatusCode::UNAUTHORIZED,
                HeaderMap::new(),
                request.url,
                Box::pin(stream::empty()),
            ))
        }
    }

    struct RefreshingAuthentication {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AuthenticationProvider for RefreshingAuthentication {
        async fn authenticate(
            &self,
            previous: Option<&AuthenticationAttempt>,
        ) -> Result<AuthenticationAttempt, HttpError> {
            let generation = self.calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(
                previous
                    .and_then(|attempt| attempt.state::<usize>())
                    .copied(),
                generation.checked_sub(1)
            );
            let mut headers = HeaderMap::new();
            headers.insert(
                header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer refreshed-{generation}")).unwrap(),
            );
            Ok(AuthenticationAttempt::new(headers, generation))
        }
    }

    #[test]
    fn config_debug_redacts_api_key_and_auth_token() {
        let api_key = AnthropicConfig::new("anthropic-secret", "debug-model", 1024).unwrap();
        assert_eq!(api_key.resilience, None);
        let auth_token =
            AnthropicConfig::with_auth_token("bearer-secret", "debug-model", 1024).unwrap();

        for debug in [format!("{api_key:?}"), format!("{auth_token:?}")] {
            assert!(!debug.contains("anthropic-secret"));
            assert!(!debug.contains("bearer-secret"));
            assert!(debug.contains("<redacted>"));
            assert!(debug.contains("debug-model"));
        }
    }

    #[test]
    fn rejects_zero_max_tokens() {
        match AnthropicConfig::new("k", "claude-opus-4-7", 0) {
            Err(AnthropicError::InvalidMaxTokens) => {}
            other => panic!("expected InvalidMaxTokens, got {:?}", other.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn replay_request_reauthenticates_exactly_once() {
        let client = Arc::new(AlwaysUnauthorized {
            authorizations: std::sync::Mutex::new(Vec::new()),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let config = Arc::new(
            AnthropicConfig::new("unused", "claude-opus-4-7", 1024)
                .unwrap()
                .with_streaming(false),
        );
        let request = ReplayRequest {
            client: Http::from_arc(client.clone()),
            config,
            body: Bytes::from_static(b"{}"),
            authentication: Some(Authentication::new(RefreshingAuthentication {
                calls: calls.clone(),
            })),
            authentication_attempt: Arc::new(std::sync::Mutex::new(None)),
            reauthenticated: Arc::new(AtomicBool::new(false)),
            resilience: None,
            deadline: None,
            retries_used: Arc::new(AtomicUsize::new(0)),
        };

        let response = request.open_response().await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            *client.authorizations.lock().unwrap(),
            ["Bearer refreshed-0", "Bearer refreshed-1"]
        );
    }

    #[tokio::test]
    async fn logical_deadline_starts_before_initial_authentication() {
        let client = Arc::new(AlwaysUnauthorized {
            authorizations: std::sync::Mutex::new(Vec::new()),
        });
        let resilience = ResilienceConfig {
            retry_budget: Duration::ZERO,
            ..ResilienceConfig::default()
        };
        let deadline = LogicalDeadline::new(resilience.retry_budget);
        let request = ReplayRequest {
            client: Http::from_arc(client.clone()),
            config: Arc::new(AnthropicConfig::new("unused", "claude-opus-4-7", 1024).unwrap()),
            body: Bytes::from_static(b"{}"),
            authentication: Some(Authentication::bearer("unused")),
            authentication_attempt: Arc::new(std::sync::Mutex::new(None)),
            reauthenticated: Arc::new(AtomicBool::new(false)),
            resilience: Some(resilience),
            deadline: Some(deadline),
            retries_used: Arc::new(AtomicUsize::new(0)),
        };

        let error = request.open_response().await.unwrap_err();

        assert!(error.to_string().contains("logical request retry budget"));
        assert!(client.authorizations.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn default_authentication_preserves_anthropic_header_schemes() {
        let borrowed_key = String::from("borrowed-key");
        let key_configs = [
            AnthropicConfig::new("secret-key", "claude-opus-4-7", 1024).unwrap(),
            AnthropicConfig::new(&borrowed_key, "claude-opus-4-7", 1024).unwrap(),
            AnthropicConfig::new(Box::<str>::from("boxed-key"), "claude-opus-4-7", 1024).unwrap(),
            AnthropicConfig::new(
                std::sync::Arc::<str>::from("arc-key"),
                "claude-opus-4-7",
                1024,
            )
            .unwrap(),
            AnthropicConfig::new(
                std::borrow::Cow::Borrowed("cow-key"),
                "claude-opus-4-7",
                1024,
            )
            .unwrap(),
        ];
        for key_config in key_configs {
            let key_attempt = key_config.authentication.authenticate(None).await.unwrap();
            assert!(key_attempt.headers().get("x-api-key").is_some());
            assert!(
                key_attempt
                    .headers()
                    .get(agentkit_http::header::AUTHORIZATION)
                    .is_none()
            );
            assert!(key_attempt.headers()["x-api-key"].is_sensitive());
        }

        let key_config = AnthropicConfig::new("secret-key", "claude-opus-4-7", 1024).unwrap();
        let key_attempt = key_config.authentication.authenticate(None).await.unwrap();
        assert_eq!(key_attempt.headers()["x-api-key"], "secret-key");
        assert!(
            key_attempt
                .headers()
                .get(agentkit_http::header::AUTHORIZATION)
                .is_none()
        );
        assert!(key_attempt.headers()["x-api-key"].is_sensitive());

        let token_config =
            AnthropicConfig::with_auth_token("secret-token", "claude-opus-4-7", 1024).unwrap();
        let token_attempt = token_config
            .authentication
            .authenticate(None)
            .await
            .unwrap();
        assert_eq!(
            token_attempt.headers()[agentkit_http::header::AUTHORIZATION],
            "Bearer secret-token"
        );
        assert!(token_attempt.headers().get("x-api-key").is_none());
        assert!(token_attempt.headers()[agentkit_http::header::AUTHORIZATION].is_sensitive());

        let borrowed_token = String::from("borrowed-token");
        for token_config in [
            AnthropicConfig::with_auth_token(&borrowed_token, "claude-opus-4-7", 1024).unwrap(),
            AnthropicConfig::with_auth_token(
                Box::<str>::from("boxed-token"),
                "claude-opus-4-7",
                1024,
            )
            .unwrap(),
            AnthropicConfig::with_auth_token(
                std::sync::Arc::<str>::from("arc-token"),
                "claude-opus-4-7",
                1024,
            )
            .unwrap(),
            AnthropicConfig::with_auth_token(
                std::borrow::Cow::Borrowed("cow-token"),
                "claude-opus-4-7",
                1024,
            )
            .unwrap(),
        ] {
            let token_attempt = token_config
                .authentication
                .authenticate(None)
                .await
                .unwrap();
            assert!(token_attempt.headers().get("x-api-key").is_none());
            assert!(token_attempt.headers()[agentkit_http::header::AUTHORIZATION].is_sensitive());
        }
    }

    #[test]
    fn beta_flags_union_includes_server_tool_requirements() {
        let cfg = AnthropicConfig::new("k", "claude-opus-4-7", 1024)
            .unwrap()
            .with_beta("extended-thinking-2025-05-07")
            .with_server_tool(boxed(
                RawServerTool::new(serde_json::json!({
                    "type": "future_tool_20271231",
                    "name": "future_tool",
                }))
                .with_beta("future-tool-2027-12-31"),
            ));
        let flags = collect_beta_flags(&cfg);
        assert!(flags.contains("extended-thinking-2025-05-07"));
        assert!(flags.contains("future-tool-2027-12-31"));
    }

    /// Builds an `AnthropicTurn::Streaming` backed by a canned byte stream so
    /// we can exercise the full decode -> translate -> yield pipeline without
    /// a live HTTP connection.
    fn streaming_turn_from(chunks: Vec<&'static str>) -> AnthropicTurn {
        let body: BodyStream = Box::pin(stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<_, HttpError>(Bytes::from_static(c.as_bytes()))),
        ));
        AnthropicTurn {
            inner: TurnInner::Streaming(Box::new(StreamingState {
                body,
                decoder: SseDecoder::new(),
                translator: EventTranslator::new(),
                pending: VecDeque::new(),
                eof: false,
                visible_output: false,
                detect_truncation: false,
                idle_timeout: None,
                deadline: None,
                integrity: TruncatedStreamDetector::from_headers(&agentkit_http::HeaderMap::new()),
                replay: None,
            })),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn streaming_turn_drains_to_finished() {
        let chunks = vec![
            "event: message_start\ndata: {\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: content_block_stop\ndata: {\"index\":0}\n\n",
            "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ];
        let mut turn = streaming_turn_from(chunks);

        let mut seen_finished = false;
        while let Some(event) = turn.next_event(None).await.expect("next_event") {
            if let ModelTurnEvent::Finished(result) = event {
                assert_eq!(result.finish_reason, FinishReason::Completed);
                seen_finished = true;
            }
        }
        assert!(seen_finished, "turn never emitted Finished");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn streaming_turn_never_replays_after_visible_output() {
        let chunk = Bytes::from_static(
            b"event: message_start\ndata: {\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        );
        let body: BodyStream = Box::pin(stream::iter(vec![
            Ok(chunk),
            Err(HttpError::Other("stream failed".into())),
        ]));
        let replay_count = Arc::new(AtomicUsize::new(0));
        let count = replay_count.clone();
        let replay: StreamReplay = Box::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err(LoopError::Provider("unexpected replay".into())) })
        });
        let mut turn = AnthropicTurn {
            inner: TurnInner::Streaming(Box::new(StreamingState {
                body,
                decoder: SseDecoder::new(),
                translator: EventTranslator::new(),
                pending: VecDeque::new(),
                eof: false,
                visible_output: false,
                detect_truncation: false,
                idle_timeout: None,
                deadline: None,
                integrity: TruncatedStreamDetector::from_headers(&agentkit_http::HeaderMap::new()),
                replay: Some(replay),
            })),
        };

        let mut visible_events = 0;
        loop {
            match turn.next_event(None).await {
                Ok(Some(_)) => visible_events += 1,
                Err(_) => break,
                Ok(None) => panic!("stream ended without surfacing the body error"),
            }
        }
        assert!(visible_events > 0);
        assert_eq!(replay_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn streaming_turn_does_not_replay_non_transient_body_errors() {
        let body: BodyStream = Box::pin(stream::iter([Err(HttpError::Other(
            "protocol failure".into(),
        ))]));
        let replay_count = Arc::new(AtomicUsize::new(0));
        let count = replay_count.clone();
        let replay: StreamReplay = Box::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { unreachable!("non-transient errors must not replay") })
        });
        let mut turn = AnthropicTurn {
            inner: TurnInner::Streaming(Box::new(StreamingState {
                body,
                decoder: SseDecoder::new(),
                translator: EventTranslator::new(),
                pending: VecDeque::new(),
                eof: false,
                visible_output: false,
                detect_truncation: true,
                idle_timeout: None,
                deadline: None,
                integrity: TruncatedStreamDetector::default(),
                replay: Some(replay),
            })),
        };

        let error = turn.next_event(None).await.unwrap_err();
        assert!(error.to_string().contains("protocol failure"));
        assert_eq!(replay_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn streaming_turn_respects_pre_fired_cancellation() {
        let chunks = vec![
            "event: message_start\ndata: {\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        ];
        let mut turn = streaming_turn_from(chunks);

        let controller = CancellationController::new();
        let checkpoint = TurnCancellation::new(controller.handle());
        // Fire cancellation before polling.
        controller.interrupt();

        let err = turn.next_event(Some(checkpoint)).await.unwrap_err();
        assert!(matches!(err, LoopError::Cancelled));
    }
}
