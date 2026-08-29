//! Cerebras Inference API adapter for the agentkit agent loop.
//!
//! This crate implements the agentkit [`ModelAdapter`] directly against
//! Cerebras' `/v1/chat/completions` endpoint.
//!
//! Streaming is on by default. Toggle via [`CerebrasConfig::with_streaming`].
//!
//! # Quick start
//!
//! ```rust,ignore
//! use agentkit_loop::{Agent, SessionConfig};
//! use agentkit_provider_cerebras::{CerebrasAdapter, CerebrasConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = CerebrasConfig::from_env()?;
//!     let adapter = CerebrasAdapter::new(config)?;
//!     let agent = Agent::builder().model(adapter).build()?;
//!     let _driver = agent.start(SessionConfig::new("demo")).await?;
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod error;
pub mod models;
pub mod rate_limit;
pub mod request;
pub mod response;
pub mod version;

#[cfg(feature = "batch")]
pub mod batch;
#[cfg(feature = "compression")]
pub mod compression;
#[cfg(feature = "batch")]
pub mod files;

mod sse;
mod stream;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentkit_core::TurnCancellation;
use agentkit_http::{
    Authentication, AuthenticationAttempt, AuthenticationProvider, BodyStream, Bytes, Http,
    HttpError, HttpRequestBuilder, HttpResponse, LogicalDeadline, ResilienceConfig, StatusCode,
    TruncatedStreamDetector, is_retryable_body_read, is_retryable_status, next_body_chunk_bounded,
    run_bounded, sleep as resilience_sleep,
};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use futures_util::future::{BoxFuture, Either, select};

pub use crate::config::{
    CerebrasConfig, DEFAULT_BASE_URL, DEFAULT_VERSION_PATCH, OutputFormat, PartKindName,
    ReasoningConfig, ReasoningEffort, ReasoningFormat, ToolChoice,
};
pub use crate::error::{BuildError, CerebrasError, ResponseError};
pub use crate::models::{ModelObject, ModelsClient};
pub use crate::rate_limit::RateLimitSnapshot;

#[cfg(feature = "predicted-outputs")]
pub use crate::config::Prediction;
#[cfg(feature = "compression")]
pub use crate::config::{CompressionConfig, RequestEncoding};
#[cfg(feature = "service-tiers")]
pub use crate::config::{QueueThreshold, ServiceTier};

#[cfg(feature = "batch")]
pub use crate::batch::{
    BatchClient, BatchItem, BatchJob, BatchOutcome, BatchRequestCounts, BatchStatus, ChatOverrides,
};
#[cfg(feature = "batch")]
pub use crate::files::{FileObject, FilePurpose, FilesClient};

use crate::stream::{EventTranslator, SseDecoder};

/// Model adapter that connects the agentkit agent loop to Cerebras'
/// `/v1/chat/completions` endpoint.
#[derive(Clone)]
pub struct CerebrasAdapter {
    client: Http,
    config: Arc<CerebrasConfig>,
    authentication: Authentication,
    resilience: Option<ResilienceConfig>,
    last_rate_limit: Arc<Mutex<Option<RateLimitSnapshot>>>,
}

impl CerebrasAdapter {
    /// Creates a new adapter from the given configuration, building a default
    /// reqwest-backed HTTP client.
    pub fn new(config: CerebrasConfig) -> Result<Self, CerebrasError> {
        config.validate()?;
        let client = reqwest::Client::builder()
            .build()
            .map(Http::new)
            .map_err(|error| CerebrasError::Http(HttpError::request(error)))?;
        Self::with_client(config, client)
    }

    /// Creates a new adapter using a pre-configured [`Http`] client.
    pub fn with_client(config: CerebrasConfig, client: Http) -> Result<Self, CerebrasError> {
        config.validate()?;
        let authentication = Authentication::bearer(config.api_key.clone());
        Ok(Self {
            client,
            config: Arc::new(config),
            authentication,
            resilience: None,
            last_rate_limit: Arc::new(Mutex::new(None)),
        })
    }

    /// Overrides the configured bearer-token authentication.
    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        self.authentication = authentication.into();
        self
    }

    /// Overrides authentication with a custom refresh-capable provider.
    pub fn with_authentication_provider<P: AuthenticationProvider>(self, provider: P) -> Self {
        self.with_authentication(Authentication::new(provider))
    }

    /// Enables request and pre-visible-output stream retries and timeouts.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        self.resilience = Some(resilience);
        self
    }

    /// Reads the latest rate-limit snapshot, if any response has been received.
    pub fn last_rate_limit(&self) -> Option<RateLimitSnapshot> {
        self.last_rate_limit.lock().ok()?.clone()
    }

    /// Returns a typed client over `/v1/models`.
    pub fn models(&self) -> ModelsClient<'_> {
        ModelsClient::new_with_policy(
            &self.client,
            self.config.clone(),
            self.authentication.clone(),
            self.resilience.clone(),
        )
    }

    /// Returns a typed client over the Batch API.
    #[cfg(feature = "batch")]
    pub fn batches(&self) -> BatchClient<'_> {
        BatchClient::new_with_policy(
            &self.client,
            self.config.clone(),
            self.authentication.clone(),
            self.resilience.clone(),
        )
    }

    /// Returns a typed client over the Files API.
    #[cfg(feature = "batch")]
    pub fn files(&self) -> FilesClient<'_> {
        FilesClient::new_with_policy(
            &self.client,
            self.config.clone(),
            self.authentication.clone(),
            self.resilience.clone(),
        )
    }
}

/// An active session against the Cerebras chat-completions endpoint.
pub struct CerebrasSession {
    client: Http,
    config: Arc<CerebrasConfig>,
    authentication: Authentication,
    resilience: Option<ResilienceConfig>,
    rate_limit_slot: Arc<Mutex<Option<RateLimitSnapshot>>>,
    _session_config: SessionConfig,
}

/// A single Cerebras chat-completions turn in progress.
pub struct CerebrasTurn {
    inner: TurnInner,
}

enum TurnInner {
    Buffered { events: VecDeque<ModelTurnEvent> },
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

#[cfg(any(feature = "batch", test))]
struct GuardedBodyState {
    body: BodyStream,
    idle_timeout: Option<Duration>,
    deadline: Option<LogicalDeadline>,
    integrity: TruncatedStreamDetector,
    done: bool,
}

pub(crate) struct BufferedResponse {
    status: StatusCode,
    body: Bytes,
}

impl BufferedResponse {
    pub(crate) fn status(&self) -> StatusCode {
        self.status
    }

    pub(crate) async fn text(self) -> Result<String, HttpError> {
        String::from_utf8(self.body.to_vec()).map_err(|error| HttpError::Body(Box::new(error)))
    }

    pub(crate) async fn json<T: serde::de::DeserializeOwned>(self) -> Result<T, HttpError> {
        serde_json::from_slice(&self.body).map_err(HttpError::Deserialize)
    }
}

/// Authentication and resilience state for one replayable logical request.
#[derive(Clone)]
pub(crate) struct RequestExecutor {
    client: Http,
    authentication: Authentication,
    authentication_attempt: Arc<Mutex<Option<AuthenticationAttempt>>>,
    reauthenticated: Arc<AtomicBool>,
    resilience: Option<ResilienceConfig>,
    deadline: Option<LogicalDeadline>,
    retries_used: Arc<AtomicUsize>,
}

impl RequestExecutor {
    pub(crate) fn new(
        client: &Http,
        authentication: Authentication,
        resilience: Option<ResilienceConfig>,
    ) -> Self {
        let deadline = resilience
            .as_ref()
            .map(|resilience| LogicalDeadline::new(resilience.retry_budget));
        Self {
            client: client.clone(),
            authentication,
            authentication_attempt: Arc::new(Mutex::new(None)),
            reauthenticated: Arc::new(AtomicBool::new(false)),
            resilience,
            deadline,
            retries_used: Arc::new(AtomicUsize::new(0)),
        }
    }

    async fn authentication_attempt(&self) -> Result<AuthenticationAttempt, HttpError> {
        let existing = self
            .authentication_attempt
            .lock()
            .map_err(|_| HttpError::Other("authentication state lock poisoned".into()))?
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
        .await?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| HttpError::Other("authentication state lock poisoned".into()))? =
            Some(attempt.clone());
        Ok(attempt)
    }

    async fn reauthenticate(&self, previous: AuthenticationAttempt) -> Result<(), HttpError> {
        let next = run_bounded(
            self.authentication.authenticate(Some(&previous)),
            None,
            self.deadline.as_ref(),
            "reauthentication",
        )
        .await?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| HttpError::Other("authentication state lock poisoned".into()))? =
            Some(next);
        Ok(())
    }

    pub(crate) fn reserve_retry(&self) -> Option<usize> {
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

    pub(crate) async fn wait_before_retry(
        &self,
        retry_number: usize,
        headers: Option<&agentkit_http::HeaderMap>,
    ) -> Result<(), HttpError> {
        self.wait_for_retry_delay(self.retry_delay(retry_number, headers))
            .await
    }

    async fn wait_for_retry_delay(&self, delay: Duration) -> Result<(), HttpError> {
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

    async fn read_response_bytes(&self, response: HttpResponse) -> Result<Bytes, HttpError> {
        let timeout = self
            .resilience
            .as_ref()
            .and_then(|config| config.attempt_timeout);
        run_bounded(
            response.bytes(),
            timeout,
            self.deadline.as_ref(),
            "HTTP response body",
        )
        .await
    }

    pub(crate) async fn read_response_text(
        &self,
        response: HttpResponse,
    ) -> Result<String, HttpError> {
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

    pub(crate) async fn execute<F>(&self, build: F) -> Result<HttpResponse, HttpError>
    where
        F: Fn() -> HttpRequestBuilder,
    {
        self.execute_inner(build, true).await
    }

    #[cfg(feature = "batch")]
    pub(crate) async fn execute_without_retries<F>(
        &self,
        build: F,
    ) -> Result<HttpResponse, HttpError>
    where
        F: Fn() -> HttpRequestBuilder,
    {
        self.execute_inner(build, false).await
    }

    async fn execute_inner<F>(&self, build: F, retry_safe: bool) -> Result<HttpResponse, HttpError>
    where
        F: Fn() -> HttpRequestBuilder,
    {
        loop {
            let authentication = self.authentication_attempt().await?;
            let request = build().headers(authentication.headers().clone()).build()?;
            match self.execute_attempt(request).await {
                Ok(response) => {
                    if response.status() == StatusCode::UNAUTHORIZED
                        && !self.reauthenticated.swap(true, Ordering::SeqCst)
                    {
                        drop(response);
                        self.reauthenticate(authentication).await?;
                        continue;
                    }
                    if retry_safe
                        && is_retryable_status(response.status())
                        && let Some(retry_number) = self.reserve_retry()
                    {
                        let delay = self.retry_delay(retry_number, Some(response.headers()));
                        drop(response);
                        self.wait_for_retry_delay(delay).await?;
                        continue;
                    }
                    return Ok(response);
                }
                Err(error) if retry_safe && error.is_retryable_transport() => {
                    if let Some(retry_number) = self.reserve_retry() {
                        self.wait_before_retry(retry_number, None).await?;
                        continue;
                    }
                    return Err(error);
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub(crate) async fn execute_buffered<F>(&self, build: F) -> Result<BufferedResponse, HttpError>
    where
        F: Fn() -> HttpRequestBuilder,
    {
        self.execute_buffered_inner(build, true).await
    }

    #[cfg(any(feature = "batch", test))]
    pub(crate) async fn execute_buffered_without_retries<F>(
        &self,
        build: F,
    ) -> Result<BufferedResponse, HttpError>
    where
        F: Fn() -> HttpRequestBuilder,
    {
        self.execute_buffered_inner(build, false).await
    }

    async fn execute_buffered_inner<F>(
        &self,
        build: F,
        retry_safe: bool,
    ) -> Result<BufferedResponse, HttpError>
    where
        F: Fn() -> HttpRequestBuilder,
    {
        loop {
            let response = self.execute_inner(&build, retry_safe).await?;
            let status = response.status();
            match self.read_response_bytes(response).await {
                Ok(body) => return Ok(BufferedResponse { status, body }),
                Err(error) if retry_safe && is_retryable_body_read(status, &error) => {
                    if let Some(retry_number) = self.reserve_retry() {
                        self.wait_before_retry(retry_number, None).await?;
                        continue;
                    }
                    return Err(error);
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Applies the configured logical deadline, idle timeout, and content-length
    /// truncation check without changing the public `BodyStream` type. The
    /// stream reports failures but never replays after it has been returned.
    #[cfg(any(feature = "batch", test))]
    pub(crate) fn guard_body_stream(&self, response: HttpResponse) -> BodyStream {
        if self.resilience.is_none() {
            return response.bytes_stream();
        }
        let state = GuardedBodyState {
            idle_timeout: self
                .resilience
                .as_ref()
                .and_then(|resilience| resilience.stream_idle_timeout),
            deadline: self.deadline.clone(),
            integrity: TruncatedStreamDetector::from_headers(response.headers()),
            body: response.bytes_stream(),
            done: false,
        };
        Box::pin(futures_util::stream::unfold(
            state,
            |mut state| async move {
                if state.done {
                    return None;
                }
                let next = next_body_chunk_bounded(
                    &mut state.body,
                    state.idle_timeout,
                    state.deadline.as_ref(),
                )
                .await;
                match next {
                    Ok(Some(bytes)) => {
                        state.integrity.observe(&bytes);
                        Some((Ok(bytes), state))
                    }
                    Ok(None) => match state.integrity.finish() {
                        Ok(()) => None,
                        Err(error) => {
                            state.done = true;
                            Some((Err(error), state))
                        }
                    },
                    Err(error) => {
                        state.done = true;
                        Some((Err(error), state))
                    }
                }
            },
        ))
    }
}

#[derive(Clone)]
struct InferenceRequest {
    executor: RequestExecutor,
    config: Arc<CerebrasConfig>,
    body: Bytes,
    content_type: &'static str,
    content_encoding: Option<&'static str>,
    extra_headers: Vec<(&'static str, String)>,
    rate_limit_slot: Arc<Mutex<Option<RateLimitSnapshot>>>,
}

impl InferenceRequest {
    async fn open_response(&self) -> Result<HttpResponse, LoopError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let response = self
            .executor
            .execute(|| {
                let mut builder = self
                    .executor
                    .client
                    .post(&url)
                    .header("Content-Type", self.content_type);
                if let Some(encoding) = self.content_encoding {
                    builder = builder.header("Content-Encoding", encoding);
                }
                if let Some(patch) = self.config.version_patch {
                    builder = builder.header(
                        crate::version::VERSION_PATCH_HEADER,
                        crate::version::format_version_patch(patch),
                    );
                }
                for (name, value) in &self.extra_headers {
                    builder = builder.header(*name, value.clone());
                }
                builder = builder.header(
                    "User-Agent",
                    concat!("agentkit-provider-cerebras/", env!("CARGO_PKG_VERSION")),
                );
                if self.config.streaming {
                    builder = builder.header("Accept", "text/event-stream");
                }
                for (name, value) in &self.config.extra_headers {
                    builder = builder.header(name.as_str(), value.as_str());
                }
                builder.body(self.body.clone())
            })
            .await
            .map_err(|error| LoopError::Provider(format!("Cerebras request failed: {error}")))?;
        let snapshot = RateLimitSnapshot::from_headers(response.headers());
        if let Ok(mut slot) = self.rate_limit_slot.lock() {
            *slot = Some(snapshot);
        }
        Ok(response)
    }

    async fn replay_stream(&self) -> Result<(BodyStream, TruncatedStreamDetector), LoopError> {
        let retry_number = self.executor.reserve_retry().ok_or_else(|| {
            LoopError::Provider(
                "Cerebras stream failed before output and retry budget is exhausted".into(),
            )
        })?;
        self.executor
            .wait_before_retry(retry_number, None)
            .await
            .map_err(|error| LoopError::Provider(format!("retry backoff failed: {error}")))?;
        let response = self.open_response().await?;
        if !response.status().is_success() {
            return Err(LoopError::Provider(format!(
                "Cerebras stream replay failed with status {}",
                response.status()
            )));
        }
        let integrity = TruncatedStreamDetector::from_headers(response.headers());
        Ok((response.bytes_stream(), integrity))
    }
}

#[async_trait]
impl ModelAdapter for CerebrasAdapter {
    type Session = CerebrasSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(CerebrasSession {
            client: self.client.clone(),
            config: self.config.clone(),
            authentication: self.authentication.clone(),
            resilience: self.resilience.clone(),
            rate_limit_slot: self.last_rate_limit.clone(),
            _session_config: config,
        })
    }

    fn provider_name(&self) -> Option<&str> {
        Some("cerebras")
    }
}

#[async_trait]
impl ModelSession for CerebrasSession {
    type Turn = CerebrasTurn;

    async fn begin_turn(
        &mut self,
        turn_request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<CerebrasTurn, LoopError> {
        let config = self.config.clone();
        let rate_limit_slot = self.rate_limit_slot.clone();

        let request_future = async {
            let built = request::build_chat_body(&config, &turn_request)
                .map_err(|e| LoopError::Provider(e.to_string()))?;

            #[cfg(feature = "compression")]
            let (body, content_type, content_encoding) = match &config.compression {
                Some(cfg) => {
                    let encoded = crate::compression::encode_body(&built.body, cfg)
                        .map_err(LoopError::Provider)?;
                    (
                        Bytes::from(encoded.body),
                        encoded.content_type,
                        encoded.content_encoding,
                    )
                }
                None => (
                    Bytes::from(
                        serde_json::to_vec(&built.body)
                            .map_err(|e| LoopError::Provider(format!("json serialize: {e}")))?,
                    ),
                    "application/json",
                    None,
                ),
            };
            #[cfg(not(feature = "compression"))]
            let (body, content_type, content_encoding) = (
                Bytes::from(
                    serde_json::to_vec(&built.body)
                        .map_err(|e| LoopError::Provider(format!("json serialize: {e}")))?,
                ),
                "application/json",
                None::<&'static str>,
            );

            let inference_request = InferenceRequest {
                executor: RequestExecutor::new(
                    &self.client,
                    self.authentication.clone(),
                    self.resilience.clone(),
                ),
                config: config.clone(),
                body,
                content_type,
                content_encoding,
                extra_headers: built.extra_headers,
                rate_limit_slot,
            };

            loop {
                let response = inference_request.open_response().await?;
                let status = response.status();
                if config.streaming && status.is_success() {
                    let integrity = TruncatedStreamDetector::from_headers(response.headers());
                    let idle_timeout = self
                        .resilience
                        .as_ref()
                        .and_then(|resilience| resilience.stream_idle_timeout);
                    let replay = self.resilience.as_ref().and_then(|resilience| {
                        (resilience.max_retries > 0).then(|| {
                            let inference_request = inference_request.clone();
                            Box::new(move || {
                                let inference_request = inference_request.clone();
                                Box::pin(async move { inference_request.replay_stream().await })
                                    as BoxFuture<'static, _>
                            }) as StreamReplay
                        })
                    });
                    return Ok(CerebrasTurn {
                        inner: TurnInner::Streaming(Box::new(StreamingState {
                            body: response.bytes_stream(),
                            decoder: SseDecoder::new(),
                            translator: EventTranslator::new(),
                            pending: VecDeque::new(),
                            eof: false,
                            visible_output: false,
                            detect_truncation: self.resilience.is_some(),
                            idle_timeout,
                            deadline: inference_request.executor.deadline.clone(),
                            integrity,
                            replay,
                        })),
                    });
                }

                let body_text = match inference_request
                    .executor
                    .read_response_text(response)
                    .await
                {
                    Ok(body) => body,
                    Err(error) if is_retryable_body_read(status, &error) => {
                        if let Some(retry_number) = inference_request.executor.reserve_retry() {
                            inference_request
                                .executor
                                .wait_before_retry(retry_number, None)
                                .await
                                .map_err(|error| {
                                    LoopError::Provider(format!("retry backoff failed: {error}"))
                                })?;
                            continue;
                        }
                        return Err(LoopError::Provider(format!(
                            "failed to read Cerebras response body: {error}"
                        )));
                    }
                    Err(error) => {
                        return Err(LoopError::Provider(format!(
                            "failed to read Cerebras response body: {error}"
                        )));
                    }
                };

                if !status.is_success() {
                    return Err(LoopError::Provider(format!(
                        "Cerebras request failed with status {status}: {body_text}"
                    )));
                }
                let events = response::build_turn_from_response(&body_text)
                    .map_err(|e| LoopError::Provider(e.to_string()))?;
                return Ok(CerebrasTurn {
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
        Some("cerebras")
    }
}

#[async_trait]
impl ModelTurn for CerebrasTurn {
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
                    LoopError::Provider(format!("invalid UTF-8 in Cerebras stream: {e}"))
                })?;
                for sse in decoder.feed(text) {
                    for event in translator
                        .handle(&sse)
                        .map_err(|e| LoopError::Provider(e.to_string()))?
                    {
                        pending.push_back(event);
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
                    Ok(()) => "Cerebras stream ended before its terminal event".to_owned(),
                    Err(error) => format!("Cerebras stream body error: {error}"),
                };
                Some(LoopError::Provider(message))
            }
            Err(error) if !error.is_retryable_transport() => {
                return Err(LoopError::Provider(format!(
                    "Cerebras stream body error: {error}"
                )));
            }
            Err(error) => Some(LoopError::Provider(format!(
                "Cerebras stream body error: {error}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use agentkit_core::{CancellationController, FinishReason};
    use agentkit_http::{HeaderMap, HeaderValue, HttpClient, HttpError, HttpRequest, header};
    use bytes::Bytes;
    use futures_util::{StreamExt, stream};

    struct AlwaysUnauthorized {
        authorizations: Mutex<Vec<String>>,
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

    struct FixedStatus {
        calls: AtomicUsize,
        status: StatusCode,
    }

    #[async_trait]
    impl HttpClient for FixedStatus {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(HttpResponse::new(
                self.status,
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

    #[tokio::test]
    async fn request_executor_replays_unauthorized_exactly_once() {
        let client = Arc::new(AlwaysUnauthorized {
            authorizations: Mutex::new(Vec::new()),
        });
        let calls = Arc::new(AtomicUsize::new(0));
        let http = Http::from_arc(client.clone());
        let executor = RequestExecutor::new(
            &http,
            Authentication::new(RefreshingAuthentication {
                calls: calls.clone(),
            }),
            None,
        );

        let response = executor
            .execute(|| http.get("https://example.test/models"))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            *client.authorizations.lock().unwrap(),
            ["Bearer refreshed-0", "Bearer refreshed-1"]
        );
    }

    #[tokio::test]
    async fn logical_deadline_starts_before_auxiliary_authentication() {
        let client = Arc::new(FixedStatus {
            calls: AtomicUsize::new(0),
            status: StatusCode::OK,
        });
        let http = Http::from_arc(client.clone());
        let resilience = ResilienceConfig {
            retry_budget: Duration::ZERO,
            ..ResilienceConfig::default()
        };
        let executor =
            RequestExecutor::new(&http, Authentication::bearer("unused"), Some(resilience));

        let error = executor
            .execute(|| http.get("https://example.test/models"))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("logical request retry budget"));
        assert_eq!(client.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unsafe_post_policy_does_not_retry_retryable_statuses() {
        let client = Arc::new(FixedStatus {
            calls: AtomicUsize::new(0),
            status: StatusCode::SERVICE_UNAVAILABLE,
        });
        let http = Http::from_arc(client.clone());
        let resilience = ResilienceConfig {
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
            ..ResilienceConfig::default()
        };
        let executor =
            RequestExecutor::new(&http, Authentication::bearer("unused"), Some(resilience));

        let response = executor
            .execute_buffered_without_retries(|| http.post("https://example.test/files"))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn guarded_file_stream_reports_content_length_truncation() {
        let http = Http::from_arc(Arc::new(FixedStatus {
            calls: AtomicUsize::new(0),
            status: StatusCode::OK,
        }));
        let executor = RequestExecutor::new(
            &http,
            Authentication::bearer("unused"),
            Some(ResilienceConfig::default()),
        );
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_LENGTH, HeaderValue::from_static("5"));
        let response = HttpResponse::new(
            StatusCode::OK,
            headers,
            "https://example.test/files/1/content".into(),
            Box::pin(stream::once(async {
                Ok::<_, HttpError>(Bytes::from_static(b"123"))
            })),
        );
        let mut body = executor.guard_body_stream(response);

        assert_eq!(
            body.next().await.unwrap().unwrap(),
            Bytes::from_static(b"123")
        );
        assert!(matches!(
            body.next().await.unwrap(),
            Err(HttpError::TruncatedBody {
                expected: 5,
                received: 3
            })
        ));
        assert!(body.next().await.is_none());
    }

    fn streaming_turn_from(chunks: Vec<&'static str>) -> CerebrasTurn {
        let body: BodyStream = Box::pin(stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<_, HttpError>(Bytes::from_static(c.as_bytes()))),
        ));
        CerebrasTurn {
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
            "data: {\"id\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"done\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
        ];
        let mut turn = streaming_turn_from(chunks);
        let mut saw_finished = false;
        while let Some(event) = turn.next_event(None).await.expect("next_event") {
            if let ModelTurnEvent::Finished(result) = event {
                assert_eq!(result.finish_reason, FinishReason::Completed);
                saw_finished = true;
            }
        }
        assert!(saw_finished);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn streaming_turn_never_replays_after_visible_output() {
        let body: BodyStream = Box::pin(stream::iter(vec![
            Ok(Bytes::from_static(
                b"data: {\"id\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            )),
            Err(HttpError::Other("stream failed".into())),
        ]));
        let replay_count = Arc::new(AtomicUsize::new(0));
        let count = replay_count.clone();
        let replay: StreamReplay = Box::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err(LoopError::Provider("unexpected replay".into())) })
        });
        let mut turn = CerebrasTurn {
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
        let mut turn = CerebrasTurn {
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
        let chunks = vec!["data: {\"id\":\"m\",\"choices\":[]}\n\n"];
        let mut turn = streaming_turn_from(chunks);
        let controller = CancellationController::new();
        let checkpoint = TurnCancellation::new(controller.handle());
        controller.interrupt();
        let err = turn.next_event(Some(checkpoint)).await.unwrap_err();
        assert!(matches!(err, LoopError::Cancelled));
    }
}
