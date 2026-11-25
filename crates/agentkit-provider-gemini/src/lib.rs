//! Google Gemini API adapter for the agentkit agent loop.
//!
//! This crate implements the agentkit [`ModelAdapter`] directly against
//! Google's Gemini REST API (`generateContent` / `streamGenerateContent?alt=sse`),
//! supporting both the Gemini Developer API and Google Cloud Vertex AI.
//!
//! Streaming is enabled by default. Call [`GeminiConfig::with_streaming`]
//! with `false` to use the buffered JSON response path.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use agentkit_http::ResilienceConfig;
//! use agentkit_loop::{Agent, SessionConfig};
//! use agentkit_provider_gemini::{GeminiAdapter, GeminiConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = GeminiConfig::from_env()?
//!         .with_resilience(ResilienceConfig::default());
//!     let adapter = GeminiAdapter::new(config)?;
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
mod retry;
mod sse;
mod stream;
mod usage;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentkit_core::TurnCancellation;
use agentkit_http::{
    Authentication, AuthenticationAttempt, AuthenticationProvider, BodyStream, Bytes, Http,
    HttpError, HttpResponse, LogicalDeadline, ResilienceConfig, StatusCode,
    TruncatedStreamDetector, is_retryable_body_read, is_retryable_status, next_body_chunk_bounded,
    retry_hint, run_bounded, sleep as resilience_sleep,
};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, ProviderFailureReason,
    RetryObserver, SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use futures_util::future::{BoxFuture, Either, select};

use crate::retry::{
    RetryTracker, classify_http_body, context_length_error, gemini_retry_hint, local_error,
    provider_error,
};
use crate::stream::{EventTranslator, SseDecoder};

pub use crate::config::{
    DEFAULT_BASE_URL, GeminiApiKey, GeminiAuthToken, GeminiConfig, GeminiLimits, GeminiServerTool,
    OutputFormat, SafetySetting, ThinkingConfig, ThinkingLevel, ToolChoice,
};
pub use crate::error::GeminiError;

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Model adapter connecting the agentkit agent loop to Google's Gemini API.
#[derive(Clone)]
pub struct GeminiAdapter {
    client: Http,
    config: Arc<GeminiConfig>,
}

impl GeminiAdapter {
    /// Creates a new adapter from the given configuration with a default reqwest HTTP client.
    ///
    /// Redirects are disabled on the adapter-owned client so `x-goog-api-key`
    /// authentication headers cannot be forwarded across origins.
    pub fn new(config: GeminiConfig) -> Result<Self, GeminiError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map(Http::new)
            .map_err(|error| GeminiError::HttpClient(HttpError::request(error)))?;
        Ok(Self::with_client(config, client))
    }

    /// Creates a new adapter using a pre-configured [`Http`] client.
    ///
    /// Callers supplying a custom client should disable redirects or restrict
    /// them to the same origin so `x-goog-api-key` headers cannot leak across
    /// origins.
    pub fn with_client(config: GeminiConfig, client: Http) -> Self {
        Self {
            client,
            config: Arc::new(config),
        }
    }

    /// Overrides the configured `x-goog-api-key` or bearer-token authentication.
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

/// An active session with the Gemini API.
pub struct GeminiSession {
    client: Http,
    config: Arc<GeminiConfig>,
    authentication: Authentication,
    resilience: Option<ResilienceConfig>,
    retry_observer: Option<Arc<dyn RetryObserver>>,
    _session_config: SessionConfig,
}

/// A turn in progress against the Gemini API.
pub struct GeminiTurn {
    inner: TurnInner,
    tracker: Arc<Mutex<RetryTracker>>,
}

enum TurnInner {
    Buffered { events: VecDeque<ModelTurnEvent> },
    Streaming(Box<StreamingState>),
}

type StreamReplay = Box<
    dyn FnMut(
            LoopError,
        )
            -> BoxFuture<'static, Result<(BodyStream, TruncatedStreamDetector), LoopError>>
        + Send,
>;

struct StreamingState {
    body: BodyStream,
    decoder: SseDecoder,
    translator: EventTranslator,
    pending: VecDeque<ModelTurnEvent>,
    eof: bool,
    visible_output: bool,
    idle_timeout: Option<Duration>,
    deadline: Option<LogicalDeadline>,
    integrity: TruncatedStreamDetector,
    replay: Option<StreamReplay>,
    attempt_bytes: usize,
    wire_bytes: Arc<AtomicUsize>,
    limits: GeminiLimits,
    tracker: Arc<Mutex<RetryTracker>>,
}

#[derive(Clone)]
struct ReplayRequest {
    client: Http,
    config: Arc<GeminiConfig>,
    endpoint_url: String,
    body: Bytes,
    authentication: Authentication,
    authentication_attempt: Arc<Mutex<Option<AuthenticationAttempt>>>,
    reauthenticated: Arc<AtomicBool>,
    resilience: Option<ResilienceConfig>,
    deadline: Option<LogicalDeadline>,
    retries_used: Arc<AtomicUsize>,
    wire_bytes: Arc<AtomicUsize>,
    tracker: Arc<Mutex<RetryTracker>>,
}

impl ReplayRequest {
    async fn authentication_attempt(&self) -> Result<AuthenticationAttempt, LoopError> {
        let existing = self
            .authentication_attempt
            .lock()
            .map_err(|_| local_error(ProviderFailureReason::Authentication))?
            .clone();
        if let Some(existing) = existing {
            return Ok(existing);
        }
        let attempt = run_bounded(
            self.authentication.authenticate(None),
            None,
            self.deadline.as_ref(),
            "authentication",
        )
        .await
        .map_err(|error| map_auth_error(&error))?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| local_error(ProviderFailureReason::Authentication))? =
            Some(attempt.clone());
        Ok(attempt)
    }

    async fn reauthenticate(&self, previous: AuthenticationAttempt) -> Result<(), LoopError> {
        let next = run_bounded(
            self.authentication.authenticate(Some(&previous)),
            None,
            self.deadline.as_ref(),
            "reauthentication",
        )
        .await
        .map_err(|error| map_auth_error(&error))?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| local_error(ProviderFailureReason::Authentication))? = Some(next);
        Ok(())
    }

    fn reserve_retry(&self) -> Option<usize> {
        let maximum = self.resilience.as_ref()?.max_retries;
        let mut current = self.retries_used.load(Ordering::SeqCst);
        loop {
            if current >= maximum {
                return None;
            }
            match self.retries_used.compare_exchange(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(prev) => return Some(prev),
                Err(actual) => current = actual,
            }
        }
    }

    fn retry_delay(
        &self,
        retry_number: usize,
        headers: Option<&agentkit_http::HeaderMap>,
        body_hint: Option<Duration>,
    ) -> Duration {
        let resilience = self
            .resilience
            .as_ref()
            .expect("reserved retry requires resilience");
        if let Some(hint) = headers.and_then(retry_hint).or(body_hint) {
            return hint.min(resilience.retry_budget);
        }
        resilience.retry_delay(retry_number, None)
    }

    async fn wait_for_retry_delay(
        &self,
        error: &LoopError,
        delay: Duration,
    ) -> Result<(), LoopError> {
        if let Ok(mut tracker) = self.tracker.lock() {
            tracker.scheduled(error, delay);
        }
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
        .map_err(|error| map_http_error(&error, ProviderFailureReason::RetryBudget))?;
        if let Ok(mut tracker) = self.tracker.lock() {
            tracker.completed_wait(delay);
        }
        Ok(())
    }

    async fn execute_attempt(
        &self,
        request: agentkit_http::HttpRequest,
    ) -> Result<HttpResponse, HttpError> {
        if let Ok(mut tracker) = self.tracker.lock() {
            tracker.accounting.attempts += 1;
        }
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

    async fn read_bounded_body(
        &self,
        response: HttpResponse,
        max_bytes: usize,
    ) -> Result<Option<String>, LoopError> {
        let timeout = self
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout)
            .unwrap_or(Duration::from_secs(2))
            .min(Duration::from_secs(2));
        let max_attempt = self.config.limits.max_attempt_bytes;
        let max_wire = self.config.limits.max_wire_bytes;
        let wire_bytes = self.wire_bytes.clone();
        let inspect = async {
            let mut body = response.bytes_stream();
            let mut bytes = Vec::new();
            let mut attempt_bytes: usize = 0;
            while let Some(chunk) = next_body_chunk_bounded(&mut body, None, None).await? {
                attempt_bytes = attempt_bytes.saturating_add(chunk.len());
                let prev_wire = wire_bytes.fetch_add(chunk.len(), Ordering::SeqCst);
                if attempt_bytes > max_attempt || prev_wire.saturating_add(chunk.len()) > max_wire {
                    return Err(HttpError::Other(
                        "Gemini error response exceeded configured byte limit".into(),
                    ));
                }
                if bytes.len().saturating_add(chunk.len()) > max_bytes {
                    return Ok(None);
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(String::from_utf8(bytes).ok())
        };
        match run_bounded(
            inspect,
            Some(timeout),
            self.deadline.as_ref(),
            "HTTP error body",
        )
        .await
        {
            Ok(body) => Ok(body),
            Err(HttpError::Timeout {
                operation: "logical request retry budget",
                ..
            }) => Err(local_error(ProviderFailureReason::RetryBudget)),
            Err(HttpError::Other(_)) => Err(local_error(ProviderFailureReason::Protocol)),
            Err(_) => Ok(None),
        }
    }

    async fn read_response_text(&self, response: HttpResponse) -> Result<String, HttpError> {
        let timeout = self
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout);
        let max_attempt = self.config.limits.max_attempt_bytes;
        let max_wire = self.config.limits.max_wire_bytes;
        let wire_bytes = self.wire_bytes.clone();
        let read_all = async {
            let mut body = response.bytes_stream();
            let mut bytes = Vec::new();
            while let Some(chunk) = next_body_chunk_bounded(&mut body, None, None).await? {
                let next_attempt = bytes.len().saturating_add(chunk.len());
                let prev_wire = wire_bytes.fetch_add(chunk.len(), Ordering::SeqCst);
                if next_attempt > max_attempt || prev_wire.saturating_add(chunk.len()) > max_wire {
                    return Err(HttpError::Other(
                        "Gemini response exceeded configured byte limit".into(),
                    ));
                }
                bytes.extend_from_slice(&chunk);
            }
            String::from_utf8(bytes)
                .map_err(|e| HttpError::Other(format!("invalid UTF-8 in Gemini response: {e}")))
        };
        run_bounded(
            read_all,
            timeout,
            self.deadline.as_ref(),
            "HTTP response body",
        )
        .await
    }

    async fn rejection_error(&self, response: HttpResponse) -> LoopError {
        let status = response.status();
        let body = match self.read_bounded_body(response, MAX_ERROR_BODY_BYTES).await {
            Ok(body) => body,
            Err(err) => {
                if let Ok(mut tracker) = self.tracker.lock() {
                    tracker.note_failure(&provider_error(
                        ProviderFailureReason::HttpStatus,
                        classify_http_body(None, status.as_u16()),
                    ));
                }
                return err;
            }
        };
        if status == StatusCode::BAD_REQUEST
            && let Some(text) = body.as_deref()
            && let Some(ctx_err) = context_length_error(text, Some(400))
        {
            return ctx_err;
        }
        let classification = classify_http_body(body.as_deref(), status.as_u16());
        let reason = match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                ProviderFailureReason::Authentication
            }
            StatusCode::BAD_REQUEST => ProviderFailureReason::InvalidRequest,
            _ => ProviderFailureReason::HttpStatus,
        };
        provider_error(reason, classification)
    }

    async fn open_response(&self) -> Result<HttpResponse, LoopError> {
        loop {
            let authentication = self.authentication_attempt().await?;
            let mut builder = self
                .client
                .post(&self.endpoint_url)
                .header("Content-Type", "application/json")
                .header(
                    "User-Agent",
                    concat!("agentkit-provider-gemini/", env!("CARGO_PKG_VERSION")),
                );
            if self.config.streaming {
                builder = builder.header("Accept", "text/event-stream");
            }
            if !self.config.extra_headers.is_empty() {
                builder = builder.headers(self.config.extra_headers.clone());
            }
            builder = builder.headers(authentication.headers().clone());
            let request = builder
                .body(self.body.clone())
                .build()
                .map_err(|_| local_error(ProviderFailureReason::InvalidRequest))?;

            match self.execute_attempt(request).await {
                Ok(response) => {
                    let status = response.status();
                    if status == StatusCode::UNAUTHORIZED
                        && !self.reauthenticated.swap(true, Ordering::SeqCst)
                    {
                        let auth_401 = classify_http_body(None, 401);
                        drop(response);
                        if let Ok(mut tracker) = self.tracker.lock() {
                            tracker.scheduled_reauth(auth_401);
                        }
                        if let Err(refresh_err) = self.reauthenticate(authentication).await {
                            if let Ok(mut tracker) = self.tracker.lock() {
                                tracker.note_failure(&provider_error(
                                    ProviderFailureReason::Authentication,
                                    auth_401,
                                ));
                            }
                            return Err(refresh_err);
                        }
                        continue;
                    }
                    if is_retryable_status(status) {
                        let headers = response.headers().clone();
                        let body =
                            match self.read_bounded_body(response, MAX_ERROR_BODY_BYTES).await {
                                Ok(body) => body,
                                Err(err) => {
                                    if let Ok(mut tracker) = self.tracker.lock() {
                                        tracker.note_failure(&provider_error(
                                            ProviderFailureReason::HttpStatus,
                                            classify_http_body(None, status.as_u16()),
                                        ));
                                    }
                                    return Err(err);
                                }
                            };
                        let classification = classify_http_body(body.as_deref(), status.as_u16());
                        let attempt_err =
                            provider_error(ProviderFailureReason::HttpStatus, classification);
                        if let Some(retry_number) = self.reserve_retry() {
                            let body_hint = body.as_deref().and_then(gemini_retry_hint);
                            let delay = self.retry_delay(retry_number, Some(&headers), body_hint);
                            self.wait_for_retry_delay(&attempt_err, delay).await?;
                            continue;
                        }
                        if let Ok(mut tracker) = self.tracker.lock() {
                            tracker.note_failure(&attempt_err);
                        }
                        return Err(if self.resilience.is_some() {
                            local_error(ProviderFailureReason::RetryExhausted)
                        } else {
                            attempt_err
                        });
                    }
                    return Ok(response);
                }
                Err(error) if error.is_retryable_transport() => {
                    let attempt_err = map_http_error(&error, ProviderFailureReason::Transport);
                    if let Some(retry_number) = self.reserve_retry() {
                        let delay = self.retry_delay(retry_number, None, None);
                        self.wait_for_retry_delay(&attempt_err, delay).await?;
                        continue;
                    }
                    if let Ok(mut tracker) = self.tracker.lock() {
                        tracker.note_failure(&attempt_err);
                    }
                    return Err(if self.resilience.is_some() {
                        match attempt_err.provider_failure().map(|f| f.reason) {
                            Some(ProviderFailureReason::RetryBudget) => attempt_err,
                            _ => local_error(ProviderFailureReason::RetryExhausted),
                        }
                    } else {
                        attempt_err
                    });
                }
                Err(error) => {
                    return Err(map_http_error(&error, ProviderFailureReason::Protocol));
                }
            }
        }
    }

    async fn replay_stream(
        &self,
        cause: LoopError,
    ) -> Result<(BodyStream, TruncatedStreamDetector), LoopError> {
        let Some(retry_number) = self.reserve_retry() else {
            if let Ok(mut tracker) = self.tracker.lock() {
                tracker.note_failure(&cause);
            }
            return Err(local_error(ProviderFailureReason::RetryExhausted));
        };
        let delay = self.retry_delay(retry_number, None, None);
        self.wait_for_retry_delay(&cause, delay).await?;
        let response = self.open_response().await?;
        if !response.status().is_success() {
            return Err(self.rejection_error(response).await);
        }
        let integrity = TruncatedStreamDetector::from_headers(response.headers());
        Ok((response.bytes_stream(), integrity))
    }
}

fn map_auth_error(error: &HttpError) -> LoopError {
    match error {
        HttpError::Timeout {
            operation: "logical request retry budget",
            ..
        } => local_error(ProviderFailureReason::RetryBudget),
        HttpError::Timeout { .. } => local_error(ProviderFailureReason::AttemptTimeout),
        _ => local_error(ProviderFailureReason::Authentication),
    }
}

fn map_http_error(error: &HttpError, fallback: ProviderFailureReason) -> LoopError {
    match error {
        HttpError::Timeout {
            operation: "logical request retry budget",
            ..
        } => local_error(ProviderFailureReason::RetryBudget),
        HttpError::Timeout {
            operation: "response stream idle",
            ..
        } => local_error(ProviderFailureReason::IdleTimeout),
        HttpError::Timeout { .. } => local_error(ProviderFailureReason::AttemptTimeout),
        HttpError::Request(_) | HttpError::Body(_) | HttpError::TruncatedBody { .. } => {
            local_error(fallback)
        }
        _ => local_error(ProviderFailureReason::Protocol),
    }
}

#[async_trait]
impl ModelAdapter for GeminiAdapter {
    type Session = GeminiSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(GeminiSession {
            client: self.client.clone(),
            config: self.config.clone(),
            authentication: self.config.authentication.clone(),
            resilience: self.config.resilience.clone(),
            retry_observer: None,
            _session_config: config,
        })
    }

    fn provider_name(&self) -> Option<&str> {
        Some("gemini")
    }
}

#[async_trait]
impl ModelSession for GeminiSession {
    type Turn = GeminiTurn;

    fn set_retry_observer(&mut self, observer: Option<Arc<dyn RetryObserver>>) {
        self.retry_observer = observer;
    }

    async fn begin_turn(
        &mut self,
        turn_request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<GeminiTurn, LoopError> {
        let tracker = Arc::new(Mutex::new(RetryTracker::new(self.retry_observer.clone())));
        let config = self.config.clone();

        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            if let Ok(mut guard) = tracker.lock() {
                return Err(guard.finish(LoopError::Cancelled));
            }
            return Err(LoopError::Cancelled);
        }

        let request_future = async {
            let body = request::build_request_body(&config, &turn_request)?;
            let body = serde_json::to_vec(&body)
                .map(Bytes::from)
                .map_err(|e| LoopError::Provider(format!("failed to serialize request: {e}")))?;
            if body.len() > config.limits.max_request_bytes {
                return Err(local_error(ProviderFailureReason::InvalidRequest));
            }

            let deadline = self
                .resilience
                .as_ref()
                .map(|resilience| LogicalDeadline::new(resilience.retry_budget));
            let wire_bytes = Arc::new(AtomicUsize::new(0));
            let replay_request = ReplayRequest {
                client: self.client.clone(),
                config: config.clone(),
                endpoint_url: config.endpoint_url(),
                body,
                authentication: self.authentication.clone(),
                authentication_attempt: Arc::new(Mutex::new(None)),
                reauthenticated: Arc::new(AtomicBool::new(false)),
                resilience: self.resilience.clone(),
                deadline: deadline.clone(),
                retries_used: Arc::new(AtomicUsize::new(0)),
                wire_bytes: wire_bytes.clone(),
                tracker: tracker.clone(),
            };

            loop {
                let response = replay_request.open_response().await?;
                let status = response.status();
                if !status.is_success() {
                    return Err(replay_request.rejection_error(response).await);
                }

                if config.streaming {
                    let integrity = TruncatedStreamDetector::from_headers(response.headers());
                    let idle_timeout = self
                        .resilience
                        .as_ref()
                        .and_then(|resilience| resilience.stream_idle_timeout);
                    let replay = self.resilience.as_ref().and_then(|resilience| {
                        (resilience.max_retries > 0).then(|| {
                            let replay_request = replay_request.clone();
                            Box::new(move |cause: LoopError| {
                                let replay_request = replay_request.clone();
                                Box::pin(async move { replay_request.replay_stream(cause).await })
                                    as BoxFuture<'static, _>
                            }) as StreamReplay
                        })
                    });
                    return Ok(GeminiTurn {
                        inner: TurnInner::Streaming(Box::new(StreamingState {
                            body: response.bytes_stream(),
                            decoder: SseDecoder::new(),
                            translator: EventTranslator::new(),
                            pending: VecDeque::new(),
                            eof: false,
                            visible_output: false,
                            idle_timeout,
                            deadline: deadline.clone(),
                            integrity,
                            replay,
                            attempt_bytes: 0,
                            wire_bytes: wire_bytes.clone(),
                            limits: config.limits,
                            tracker: tracker.clone(),
                        })),
                        tracker: tracker.clone(),
                    });
                }

                let body_text = match replay_request.read_response_text(response).await {
                    Ok(body) => body,
                    Err(error) if is_retryable_body_read(status, &error) => {
                        let attempt_err = map_http_error(&error, ProviderFailureReason::Transport);
                        if matches!(
                            attempt_err.provider_failure().map(|f| f.reason),
                            Some(ProviderFailureReason::RetryBudget)
                        ) {
                            return Err(attempt_err);
                        }
                        if let Some(retry_number) = replay_request.reserve_retry() {
                            let delay = replay_request.retry_delay(retry_number, None, None);
                            replay_request
                                .wait_for_retry_delay(&attempt_err, delay)
                                .await?;
                            continue;
                        }
                        if let Ok(mut guard) = tracker.lock() {
                            guard.note_failure(&attempt_err);
                        }
                        return Err(if self.resilience.is_some() {
                            local_error(ProviderFailureReason::RetryExhausted)
                        } else {
                            attempt_err
                        });
                    }
                    Err(error) => {
                        return Err(map_http_error(&error, ProviderFailureReason::Protocol));
                    }
                };

                let events = response::build_turn_from_response(&body_text)?;
                if let Ok(mut guard) = tracker.lock() {
                    guard.succeed();
                }
                return Ok(GeminiTurn {
                    inner: TurnInner::Buffered { events },
                    tracker: tracker.clone(),
                });
            }
        };

        let result = if let Some(cancellation) = cancellation {
            futures_util::pin_mut!(request_future);
            let cancelled = cancellation.cancelled();
            futures_util::pin_mut!(cancelled);
            match select(request_future, cancelled).await {
                Either::Left((result, _)) => result,
                Either::Right((_, _)) => Err(LoopError::Cancelled),
            }
        } else {
            request_future.await
        };

        match result {
            Ok(turn) => Ok(turn),
            Err(error) => {
                if let Ok(mut guard) = tracker.lock() {
                    Err(guard.finish(error))
                } else {
                    Err(error)
                }
            }
        }
    }

    fn model_name(&self) -> Option<&str> {
        Some(&self.config.model)
    }

    fn provider_name(&self) -> Option<&str> {
        Some("gemini")
    }
}

#[async_trait]
impl ModelTurn for GeminiTurn {
    fn on_cancelled(&mut self) {
        if let Ok(mut guard) = self.tracker.lock() {
            let _ = guard.finish(LoopError::Cancelled);
        }
    }

    async fn next_event(
        &mut self,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            self.on_cancelled();
            return Err(LoopError::Cancelled);
        }
        match &mut self.inner {
            TurnInner::Buffered { events } => Ok(events.pop_front()),
            TurnInner::Streaming(state) => {
                let result = next_streaming_event(state.as_mut(), cancellation).await;
                match result {
                    Ok(Some(event)) => {
                        if matches!(event, ModelTurnEvent::Finished(_))
                            && let Ok(mut guard) = self.tracker.lock()
                        {
                            guard.succeed();
                        }
                        Ok(Some(event))
                    }
                    Ok(None) => {
                        if let Ok(mut guard) = self.tracker.lock() {
                            guard.succeed();
                        }
                        Ok(None)
                    }
                    Err(error) => {
                        if let Ok(mut guard) = self.tracker.lock() {
                            Err(guard.finish(error))
                        } else {
                            Err(error)
                        }
                    }
                }
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

async fn next_streaming_event(
    state: &mut StreamingState,
    cancellation: Option<TurnCancellation>,
) -> Result<Option<ModelTurnEvent>, LoopError> {
    loop {
        if let Some(event) = state.pending.pop_front() {
            state.visible_output = true;
            return Ok(Some(event));
        }
        if state.eof || state.translator.is_done() {
            return Ok(None);
        }

        let chunk = next_stream_chunk(
            &mut state.body,
            state.idle_timeout,
            state.deadline.as_ref(),
            cancellation.as_ref(),
        )
        .await?;

        let failure = match chunk {
            Ok(Some(bytes)) => {
                state.attempt_bytes = state.attempt_bytes.saturating_add(bytes.len());
                let prev_wire = state.wire_bytes.fetch_add(bytes.len(), Ordering::SeqCst);
                if state.attempt_bytes > state.limits.max_attempt_bytes
                    || prev_wire.saturating_add(bytes.len()) > state.limits.max_wire_bytes
                {
                    return Err(local_error(ProviderFailureReason::Protocol));
                }
                state.integrity.observe(&bytes);
                let sse_events = state
                    .decoder
                    .feed_bytes(&bytes)
                    .map_err(|_| local_error(ProviderFailureReason::Protocol))?;
                for sse in sse_events {
                    for produced in state.translator.handle(&sse)? {
                        state.pending.push_back(produced);
                    }
                }
                None
            }
            Ok(None) => {
                state.eof = true;
                let mut truncated_tail = false;
                let tail_events = match state.decoder.finish() {
                    Ok(events) => events,
                    Err(utf8_err) if utf8_err.error_len().is_none() => {
                        // Incomplete multi-byte UTF-8 sequence at EOF -> truncated stream
                        truncated_tail = true;
                        Vec::new()
                    }
                    Err(_) => return Err(local_error(ProviderFailureReason::Protocol)),
                };
                for sse in tail_events {
                    let trimmed = sse.data.trim();
                    if !trimmed.is_empty()
                        && serde_json::from_str::<serde_json::Value>(trimmed)
                            .is_err_and(|e| e.is_eof())
                    {
                        // Stream cut off mid-JSON frame -> treat as truncated stream
                        truncated_tail = true;
                        break;
                    }
                    for produced in state.translator.handle(&sse)? {
                        state.pending.push_back(produced);
                    }
                }

                let integrity_ok = state.integrity.finish();
                let complete = !truncated_tail
                    && state.translator.saw_terminal_signal()
                    && integrity_ok.is_ok();
                if complete {
                    for produced in state.translator.finalize() {
                        state.pending.push_back(produced);
                    }
                    None
                } else {
                    Some(local_error(ProviderFailureReason::Transport))
                }
            }
            Err(error) if !error.is_retryable_transport() => {
                return Err(map_http_error(&error, ProviderFailureReason::Protocol));
            }
            Err(error) => {
                let err = map_http_error(&error, ProviderFailureReason::Transport);
                if matches!(
                    err.provider_failure().map(|f| f.reason),
                    Some(ProviderFailureReason::RetryBudget)
                ) {
                    return Err(err);
                }
                Some(err)
            }
        };

        let Some(failure) = failure else {
            continue;
        };

        if state.visible_output {
            if let Ok(mut guard) = state.tracker.lock() {
                guard.note_failure(&failure);
            }
            return Err(if state.replay.is_some() {
                local_error(ProviderFailureReason::ReplayUnsafe)
            } else {
                failure
            });
        }
        let Some(replay) = state.replay.as_mut() else {
            return Err(failure);
        };

        if let Ok(mut guard) = state.tracker.lock() {
            guard.note_failure(&failure);
        }
        let replay_future = replay(failure);
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
        state.body = replayed.0;
        state.integrity = replayed.1;
        state.decoder = SseDecoder::new();
        state.translator = EventTranslator::new();
        state.pending.clear();
        state.attempt_bytes = 0;
        state.eof = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentkit_core::{CancellationController, FinishReason, Item, ItemKind, SessionId, TurnId};
    use agentkit_http::{HeaderMap, HeaderValue, HttpClient, HttpRequest};
    use agentkit_loop::{ProviderRetryEvent, ProviderRoute, UpstreamErrorKind};
    use futures_util::stream;
    use serde_json::json;

    fn sample_turn_request() -> TurnRequest {
        TurnRequest {
            session_id: SessionId::new("s1"),
            turn_id: TurnId::new("t1"),
            transcript: vec![Item::text(ItemKind::User, "Hello Gemini")],
            available_tools: vec![],
            cache: None,
            metadata: agentkit_core::MetadataMap::new(),
        }
    }

    fn fast_resilience(max_retries: usize) -> ResilienceConfig {
        ResilienceConfig {
            max_retries,
            retry_budget: Duration::from_secs(5),
            attempt_timeout: Some(Duration::from_secs(2)),
            stream_idle_timeout: Some(Duration::from_secs(2)),
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(5),
        }
    }

    struct SequenceClient {
        calls: AtomicUsize,
        responses: Mutex<VecDeque<Result<HttpResponse, HttpError>>>,
        captured_headers: Mutex<Vec<HeaderMap>>,
    }

    impl SequenceClient {
        fn new(responses: Vec<Result<HttpResponse, HttpError>>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                responses: Mutex::new(responses.into()),
                captured_headers: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl HttpClient for SequenceClient {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.captured_headers.lock().unwrap().push(request.headers);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected extra HTTP call")
        }
    }

    fn ok_response(body: &'static [u8]) -> Result<HttpResponse, HttpError> {
        Ok(HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse".into(),
            Box::pin(stream::once(async move { Ok(Bytes::from_static(body)) })),
        ))
    }

    fn status_response(status: StatusCode, body: &'static [u8]) -> Result<HttpResponse, HttpError> {
        Ok(HttpResponse::new(
            status,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::once(async move { Ok(Bytes::from_static(body)) })),
        ))
    }

    #[tokio::test]
    async fn context_length_rejection_is_typed_for_streaming_and_buffered() {
        let body = br#"{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"The input token count (2000000) exceeds the maximum number of tokens allowed (1048576). SECRET-PROMPT"}}"#;
        for streaming in [false, true] {
            let client = Arc::new(SequenceClient::new(vec![status_response(
                StatusCode::BAD_REQUEST,
                body,
            )]));
            let config = GeminiConfig::new("k", "gemini-2.5-flash")
                .unwrap()
                .with_streaming(streaming);
            let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
            let mut session = adapter
                .start_session(SessionConfig::new("s1"))
                .await
                .unwrap();
            let err = match session.begin_turn(sample_turn_request(), None).await {
                Err(e) => e,
                Ok(_) => panic!("expected context length error"),
            };
            let failure = err.provider_failure().unwrap();
            assert_eq!(failure.route, ProviderRoute::Gemini);
            assert_eq!(
                failure.upstream.code,
                UpstreamErrorKind::ContextLengthExceeded
            );
            assert!(!format!("{err:?} {err}").contains("SECRET-PROMPT"));
        }
    }

    #[tokio::test]
    async fn retries_429_with_retry_observer_and_succeeds() {
        let rate_limit_body = br#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Too many requests","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0.001s"}]}}"#;
        let ok_body = br#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}

"#;
        let client = Arc::new(SequenceClient::new(vec![
            status_response(StatusCode::TOO_MANY_REQUESTS, rate_limit_body),
            ok_response(ok_body),
        ]));
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = events.clone();

        let config = GeminiConfig::new("test-key", "gemini-2.5-flash")
            .unwrap()
            .with_resilience(fast_resilience(2));
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        session.set_retry_observer(Some(Arc::new(move |ev| {
            capture.lock().unwrap().push(ev);
        })));

        let mut turn = session
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        let mut finished = None;
        while let Some(ev) = turn.next_event(None).await.unwrap() {
            if let ModelTurnEvent::Finished(res) = ev {
                finished = Some(res);
            }
        }
        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
        assert_eq!(finished.unwrap().finish_reason, FinishReason::Completed);

        let observed = events.lock().unwrap();
        assert!(observed.iter().any(|e| matches!(
            e,
            ProviderRetryEvent::Scheduled(p) if p.route == ProviderRoute::Gemini && p.upstream.error_type == UpstreamErrorKind::RateLimitExceeded
        )));
        assert!(matches!(
            observed.last().unwrap(),
            ProviderRetryEvent::Succeeded { route: ProviderRoute::Gemini, accounting } if accounting.attempts == 2
        ));
    }

    #[tokio::test]
    async fn reauthenticates_once_on_401() {
        struct RefreshAuth(AtomicUsize);

        #[async_trait]
        impl AuthenticationProvider for RefreshAuth {
            async fn authenticate(
                &self,
                previous: Option<&AuthenticationAttempt>,
            ) -> Result<AuthenticationAttempt, HttpError> {
                let gen_id = self.0.fetch_add(1, Ordering::SeqCst);
                if gen_id == 1 {
                    assert_eq!(previous.and_then(|p| p.state::<usize>()), Some(&0));
                }
                let mut headers = HeaderMap::new();
                headers.insert(
                    agentkit_http::header::AUTHORIZATION,
                    HeaderValue::from_str(&format!("Bearer tok-{gen_id}")).unwrap(),
                );
                Ok(AuthenticationAttempt::new(headers, gen_id))
            }
        }

        let ok_body = br#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}"#;
        let client = Arc::new(SequenceClient::new(vec![
            status_response(
                StatusCode::UNAUTHORIZED,
                br#"{"error":{"code":401,"status":"UNAUTHENTICATED"}}"#,
            ),
            ok_response(ok_body),
        ]));

        let config = GeminiConfig::new("unused", "gemini-2.5-flash")
            .unwrap()
            .with_streaming(false)
            .with_authentication_provider(RefreshAuth(AtomicUsize::new(0)));
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let mut turn = session
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        assert!(turn.next_event(None).await.unwrap().is_some());

        let headers = client.captured_headers.lock().unwrap();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0]["authorization"], "Bearer tok-0");
        assert_eq!(headers[1]["authorization"], "Bearer tok-1");
    }

    #[tokio::test]
    async fn replays_truncated_stream_before_visible_output_and_stops_after_visible_output() {
        // First attempt: EOF before finishReason and before any visible event -> replayed!
        let truncated_empty = Ok(HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::empty()),
        ));
        let complete = ok_response(
            br#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"recovered"}]},"finishReason":"STOP"}]}

"#,
        );
        let client = Arc::new(SequenceClient::new(vec![truncated_empty, complete]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_resilience(fast_resilience(2));
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let mut turn = session
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        let mut got_finished = false;
        while let Some(ev) = turn.next_event(None).await.unwrap() {
            if matches!(ev, ModelTurnEvent::Finished(_)) {
                got_finished = true;
            }
        }
        assert!(got_finished);
        assert_eq!(client.calls.load(Ordering::SeqCst), 2);

        // Second scenario: visible output emitted, then transport error -> must NOT replay!
        let partial_then_err = Ok(HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::iter(vec![
                Ok(Bytes::from_static(
                    br#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}

"#,
                )),
                Err(HttpError::body(std::io::Error::other("reset"))),
            ])),
        ));
        let client2 = Arc::new(SequenceClient::new(vec![partial_then_err]));
        let config2 = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_resilience(fast_resilience(2));
        let adapter2 = GeminiAdapter::with_client(config2, Http::from_arc(client2.clone()));
        let mut session2 = adapter2
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let mut turn2 = session2
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        let mut err = None;
        loop {
            match turn2.next_event(None).await {
                Ok(Some(_)) => continue,
                Ok(None) => break,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        let failure = *err.unwrap().provider_failure().unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::ReplayUnsafe);
        assert_eq!(client2.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_finalizes_retry_observer() {
        let controller = CancellationController::new();
        let checkpoint = controller.handle().checkpoint();
        controller.interrupt();

        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = events.clone();

        let client = Arc::new(SequenceClient::new(vec![]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        session.set_retry_observer(Some(Arc::new(move |ev| {
            capture.lock().unwrap().push(ev);
        })));

        let err = match session
            .begin_turn(sample_turn_request(), Some(checkpoint))
            .await
        {
            Err(e) => e,
            Ok(_) => panic!("expected cancellation"),
        };
        assert!(matches!(err, LoopError::Cancelled));
        let observed = events.lock().unwrap();
        assert!(matches!(
            observed.last().unwrap(),
            ProviderRetryEvent::Stopped(f) if f.reason == ProviderFailureReason::Cancelled && f.route == ProviderRoute::Gemini
        ));
    }

    #[test]
    fn provider_and_model_identity_match_conventions() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-pro").unwrap();
        let adapter = GeminiAdapter::new(cfg).unwrap();
        assert_eq!(adapter.provider_name(), Some("gemini"));
        let _ = json!({});
    }

    #[tokio::test]
    async fn rejects_incomplete_sse_even_when_resilience_is_disabled() {
        let incomplete = Ok(HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::once(async {
                Ok(Bytes::from_static(
                    br#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"cut off"}]}}]}

"#,
                ))
            })),
        ));
        let client = Arc::new(SequenceClient::new(vec![incomplete]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let mut turn = session
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        loop {
            match turn.next_event(None).await {
                Ok(Some(ModelTurnEvent::Finished(_))) => {
                    panic!("incomplete stream must not emit Finished")
                }
                Ok(Some(_)) => continue,
                Ok(None) => panic!("incomplete stream must not succeed on EOF"),
                Err(_) => break,
            }
        }
    }

    #[tokio::test]
    async fn buffered_logical_deadline_preserves_retry_budget_reason() {
        struct SlowBodyClient;

        #[async_trait]
        impl HttpClient for SlowBodyClient {
            async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
                Ok(HttpResponse::new(
                    StatusCode::OK,
                    HeaderMap::new(),
                    request.url,
                    Box::pin(stream::once(async {
                        agentkit_http::sleep(Duration::from_millis(80)).await;
                        Ok(Bytes::from_static(b"{}"))
                    })),
                ))
            }
        }

        let config = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_streaming(false)
            .with_resilience(ResilienceConfig {
                max_retries: 0,
                retry_budget: Duration::from_millis(20),
                attempt_timeout: Some(Duration::from_secs(5)),
                stream_idle_timeout: Some(Duration::from_secs(5)),
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(5),
            });
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(Arc::new(SlowBodyClient)));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let err = match session.begin_turn(sample_turn_request(), None).await {
            Err(e) => e,
            Ok(_) => panic!("expected retry budget timeout"),
        };
        assert_eq!(
            err.provider_failure().unwrap().reason,
            ProviderFailureReason::RetryBudget
        );
    }

    #[tokio::test]
    async fn replays_pre_output_stream_cut_off_inside_sse_frame() {
        let mid_json = Ok(HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::once(async {
                Ok(Bytes::from_static(b"data: {\"candidates\":[{\"content\":"))
            })),
        ));
        let complete = ok_response(
            br#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}

"#,
        );
        let client = Arc::new(SequenceClient::new(vec![mid_json, complete]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_resilience(fast_resilience(2));
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client.clone()));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let mut turn = session
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        let mut finished = false;
        while let Some(ev) = turn.next_event(None).await.unwrap() {
            if matches!(ev, ModelTurnEvent::Finished(_)) {
                finished = true;
            }
        }
        assert!(finished);
        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn invalid_credential_header_maps_to_authentication_failure() {
        let client = Arc::new(SequenceClient::new(vec![]));
        let config = GeminiConfig::new("bad\nkey", "gemini-2.5-flash").unwrap();
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let err = match session.begin_turn(sample_turn_request(), None).await {
            Err(e) => e,
            Ok(_) => panic!("expected auth error"),
        };
        assert_eq!(
            err.provider_failure().unwrap().reason,
            ProviderFailureReason::Authentication
        );
    }

    #[tokio::test]
    async fn recovered_401_does_not_label_subsequent_transport_failure() {
        let client = Arc::new(SequenceClient::new(vec![
            status_response(
                StatusCode::UNAUTHORIZED,
                br#"{"error":{"code":401,"status":"UNAUTHENTICATED"}}"#,
            ),
            Err(HttpError::request(std::io::Error::other(
                "connection reset",
            ))),
        ]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_streaming(false);
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let err = match session.begin_turn(sample_turn_request(), None).await {
            Err(e) => e,
            Ok(_) => panic!("expected transport error"),
        };
        let failure = *err.provider_failure().unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::Transport);
        assert_eq!(failure.upstream.http_status, None);
        assert_eq!(failure.upstream.error_type, UpstreamErrorKind::Unknown);
    }

    #[tokio::test]
    async fn inspected_error_bodies_count_toward_wire_byte_limits() {
        let client = Arc::new(SequenceClient::new(vec![status_response(
            StatusCode::TOO_MANY_REQUESTS,
            b"0123456789abcdef0123456789abcdef",
        )]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_resilience(fast_resilience(2))
            .with_limits(GeminiLimits {
                max_request_bytes: 1024 * 1024,
                max_attempt_bytes: 16,
                max_wire_bytes: 16,
            });
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let err = match session.begin_turn(sample_turn_request(), None).await {
            Err(e) => e,
            Ok(_) => panic!("expected byte limit failure"),
        };
        assert_eq!(
            err.provider_failure().unwrap().reason,
            ProviderFailureReason::Protocol
        );
    }

    #[tokio::test]
    async fn failed_refresh_after_401_retains_http_401_status() {
        struct FailingRefresh(AtomicUsize);

        #[async_trait]
        impl AuthenticationProvider for FailingRefresh {
            async fn authenticate(
                &self,
                previous: Option<&AuthenticationAttempt>,
            ) -> Result<AuthenticationAttempt, HttpError> {
                if previous.is_some() {
                    return Err(HttpError::Other("refresh failed".into()));
                }
                let _ = self.0.fetch_add(1, Ordering::SeqCst);
                let mut headers = HeaderMap::new();
                headers.insert(
                    agentkit_http::header::AUTHORIZATION,
                    HeaderValue::from_static("Bearer initial"),
                );
                Ok(AuthenticationAttempt::stateless(headers))
            }
        }

        let client = Arc::new(SequenceClient::new(vec![status_response(
            StatusCode::UNAUTHORIZED,
            br#"{"error":{"code":401,"status":"UNAUTHENTICATED"}}"#,
        )]));
        let config = GeminiConfig::new("unused", "gemini-2.5-flash")
            .unwrap()
            .with_streaming(false)
            .with_authentication_provider(FailingRefresh(AtomicUsize::new(0)));
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let err = match session.begin_turn(sample_turn_request(), None).await {
            Err(e) => e,
            Ok(_) => panic!("expected auth failure"),
        };
        let failure = *err.provider_failure().unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::Authentication);
        assert_eq!(failure.upstream.http_status, Some(401));
    }

    #[tokio::test]
    async fn trailing_sse_cut_inside_utf8_after_finish_reason_is_rejected() {
        // Complete chunk with finishReason followed by an incomplete UTF-8 byte sequence at EOF
        let mut raw = br#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":""#.to_vec();
        raw.extend_from_slice(&[0xE3, 0x81]); // incomplete 3-byte UTF-8 char
        let response = Ok(HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::once(async move { Ok(Bytes::from(raw)) })),
        ));
        let client = Arc::new(SequenceClient::new(vec![response]));
        let config = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let mut turn = session
            .begin_turn(sample_turn_request(), None)
            .await
            .unwrap();
        loop {
            match turn.next_event(None).await {
                Ok(Some(ModelTurnEvent::Finished(_))) => {
                    panic!("truncated UTF-8 tail must not emit Finished")
                }
                Ok(Some(_)) => continue,
                Ok(None) => panic!("truncated UTF-8 tail must fail"),
                Err(_) => break,
            }
        }
    }

    #[tokio::test]
    async fn logical_deadline_during_error_body_inspection_preserves_retry_budget_and_status() {
        let response = HttpResponse::new(
            StatusCode::SERVICE_UNAVAILABLE,
            HeaderMap::new(),
            "https://example.test".into(),
            Box::pin(stream::pending()),
        );
        let client = Arc::new(SequenceClient::new(vec![Ok(response)]));
        let resilience = ResilienceConfig {
            retry_budget: Duration::from_millis(80),
            ..fast_resilience(0)
        };
        let config = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_resilience(resilience);
        let adapter = GeminiAdapter::with_client(config, Http::from_arc(client));
        let mut session = adapter
            .start_session(SessionConfig::new("s1"))
            .await
            .unwrap();
        let err = match session.begin_turn(sample_turn_request(), None).await {
            Err(e) => e,
            Ok(_) => panic!("expected retry budget timeout"),
        };
        let failure = *err.provider_failure().unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::RetryBudget);
        assert_eq!(failure.upstream.http_status, Some(503));
    }
}
