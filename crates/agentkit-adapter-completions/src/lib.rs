//! Generic OpenAI-compatible chat completions adapter for agentkit.
//!
//! This crate provides the [`CompletionsProvider`] trait and a generic
//! [`CompletionsAdapter`] that handles all common chat completions logic:
//! transcript conversion, request building, response parsing, tool call
//! extraction, usage mapping, cancellation, and multimodal content.
//!
//! Provider crates (OpenRouter, OpenAI, Ollama, etc.) implement
//! [`CompletionsProvider`] to supply authentication, endpoint URLs, and
//! provider-specific hooks. The adapter does the rest.
//!
//! # Example
//!
//! ```rust,ignore
//! use agentkit_adapter_completions::{CompletionsAdapter, CompletionsProvider};
//!
//! let adapter = CompletionsAdapter::new(my_provider)?;
//! let agent = Agent::builder().model(adapter).build()?;
//! ```

mod error;
mod media;
mod request;
mod response;
mod sse;
mod stream;

#[cfg(test)]
mod resilience_tests;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use agentkit_core::{MetadataMap, TurnCancellation, Usage};
use agentkit_http::{
    Authentication, AuthenticationAttempt, BodyStream, Bytes, Http, HttpError, HttpRequestBuilder,
    HttpResponse, LogicalDeadline, ResilienceConfig, StatusCode, TruncatedStreamDetector,
    is_retryable_body_read, is_retryable_status, next_body_chunk_bounded, run_bounded,
    sleep as resilience_sleep,
};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use futures_util::future::{BoxFuture, Either, select};
use serde::Serialize;
use serde_json::Value;

pub use crate::error::CompletionsError;
use crate::stream::{EventTranslator, PostprocessResponse, SseDecoder};

/// Trait implemented by each provider to customise the generic chat completions adapter.
///
/// The associated [`Config`](CompletionsProvider::Config) type allows each provider
/// to define a strongly-typed struct with the exact request parameters it supports.
/// The adapter serialises it and merges it into the request body.
///
/// # Required methods
///
/// - [`provider_name`](CompletionsProvider::provider_name) — for error messages
/// - [`endpoint_url`](CompletionsProvider::endpoint_url) — the chat completions URL
/// - [`config`](CompletionsProvider::config) — returns the request configuration
///
/// # Hooks
///
/// All have default implementations that pass through unchanged:
///
/// - [`preprocess_request`](CompletionsProvider::preprocess_request) — add auth headers, custom user-agent, etc.
/// - [`apply_prompt_cache`](CompletionsProvider::apply_prompt_cache) — map normalized cache requests into provider request fields
/// - [`preprocess_response`](CompletionsProvider::preprocess_response) — inspect/reject raw response before parsing
/// - [`postprocess_response`](CompletionsProvider::postprocess_response) — enrich parsed usage/metadata from raw response
pub trait CompletionsProvider: Send + Sync + Clone {
    /// Strongly-typed request configuration (model, temperature, top_p, etc.).
    ///
    /// Serialised via `serde_json::to_value` and merged into the request body.
    /// Use `#[serde(skip_serializing_if = "Option::is_none")]` on optional fields
    /// to avoid sending `null` values.
    type Config: Serialize + Clone + Send + Sync;

    /// Provider name for error messages (e.g. "OpenRouter", "Ollama").
    fn provider_name(&self) -> &str;

    /// The chat completions endpoint URL.
    fn endpoint_url(&self) -> &str;

    /// Returns the request configuration to merge into the body.
    fn config(&self) -> &Self::Config;

    /// Hook to modify the HTTP request before it is sent.
    ///
    /// Use this to add authentication headers, set a custom user-agent,
    /// or apply any other request-level customisation.
    ///
    /// The default implementation passes the builder through unchanged.
    fn preprocess_request(&self, builder: HttpRequestBuilder) -> HttpRequestBuilder {
        builder
    }

    /// Optional generic authentication. Existing provider request hooks continue
    /// to work when this returns `None`.
    fn authentication(&self) -> Option<Authentication> {
        None
    }

    /// Optional retry and timeout policy. `None` preserves the original single-
    /// attempt, no-timeout behavior.
    fn resilience_config(&self) -> Option<ResilienceConfig> {
        None
    }

    /// Hook to map a normalized prompt cache request into the provider's JSON
    /// request body.
    ///
    /// Called after the adapter has constructed the standard chat-completions
    /// payload. Providers can inspect [`TurnRequest::cache`] and mutate the
    /// request body accordingly.
    fn apply_prompt_cache(
        &self,
        _body: &mut serde_json::Map<String, Value>,
        _request: &TurnRequest,
    ) -> Result<(), LoopError> {
        Ok(())
    }

    /// Whether to request an SSE streaming response. Defaults to `true`.
    fn streaming(&self) -> bool {
        true
    }

    /// Hook to add provider-specific streaming options to the JSON request.
    ///
    /// Providers that support terminal usage frames can insert fields such as
    /// `stream_options`; the default leaves the request unchanged.
    fn apply_stream_options(
        &self,
        _body: &mut serde_json::Map<String, Value>,
    ) -> Result<(), LoopError> {
        Ok(())
    }

    /// Whether the upstream chat template enforces strict
    /// `user`/`assistant` role alternation.
    ///
    /// When `true`, the adapter merges adjacent `user`-role messages
    /// (including notifications and tool-result follow-ups that come back
    /// as user messages) into a single message before sending. Required
    /// for vLLM-served Mistral templates and the Mistral hosted API; see
    /// <https://github.com/vllm-project/vllm/issues/6862>.
    ///
    /// Defaults to `false`. Providers that target strictly-alternating
    /// upstreams should override.
    fn requires_alternating_roles(&self) -> bool {
        false
    }

    /// Hook to inspect the raw HTTP response before deserialisation.
    ///
    /// Called after the response body is read but before it is parsed into
    /// the chat completion response struct. Return `Err` to reject the
    /// response (e.g. for providers that return HTTP 200 with an error payload).
    ///
    /// The default implementation does nothing.
    fn preprocess_response(&self, _status: StatusCode, _body: &str) -> Result<(), LoopError> {
        Ok(())
    }

    /// Hook to enrich parsed response data with provider-specific fields.
    ///
    /// Called after the standard response parsing is complete. The provider
    /// can read extra fields from the raw JSON (e.g. `cost` in the usage
    /// object, `model` or `refusal` in the response) and fold them into
    /// the `Usage` and `MetadataMap` that will be attached to the output items.
    ///
    /// The default implementation does nothing.
    fn postprocess_response(
        &self,
        _usage: &mut Option<Usage>,
        _metadata: &mut MetadataMap,
        _raw_response: &Value,
    ) {
    }
}

/// Generic chat completions adapter, parameterised by a [`CompletionsProvider`].
///
/// Implements [`ModelAdapter`] so it can be passed to
/// [`Agent::builder().model()`](agentkit_loop::Agent::builder).
#[derive(Clone)]
pub struct CompletionsAdapter<P: CompletionsProvider> {
    client: Http,
    provider: Arc<P>,
    /// Lowercase provider name stamped onto telemetry spans as the
    /// `gen_ai.provider.name` attribute from the OTel GenAI semantic
    /// conventions.
    provider_label: String,
    authentication: Option<Authentication>,
    resilience: Option<ResilienceConfig>,
}

impl<P: CompletionsProvider> CompletionsAdapter<P> {
    /// Creates a new adapter from the given provider.
    ///
    /// Builds a default reqwest-backed HTTP client reused for all sessions and turns.
    pub fn new(provider: P) -> Result<Self, CompletionsError> {
        let client = reqwest::Client::builder()
            .build()
            .map(Http::new)
            .map_err(|error| CompletionsError::HttpClient(HttpError::request(error)))?;

        let authentication = provider.authentication();
        let resilience = provider.resilience_config();
        Ok(Self {
            client,
            provider_label: provider.provider_name().to_lowercase(),
            provider: Arc::new(provider),
            authentication,
            resilience,
        })
    }

    /// Creates a new adapter with a pre-configured [`Http`] client. Use this to
    /// attach auth headers via `default_headers`, supply custom TLS/proxies,
    /// or plug in a non-reqwest backend.
    pub fn with_client(provider: P, client: Http) -> Self {
        let authentication = provider.authentication();
        let resilience = provider.resilience_config();
        Self {
            client,
            provider_label: provider.provider_name().to_lowercase(),
            provider: Arc::new(provider),
            authentication,
            resilience,
        }
    }

    /// Overrides the provider's generic authentication configuration.
    pub fn with_authentication(mut self, authentication: impl Into<Authentication>) -> Self {
        self.authentication = Some(authentication.into());
        self
    }

    /// Enables request and pre-visible-output stream retries and timeouts.
    pub fn with_resilience(mut self, resilience: ResilienceConfig) -> Self {
        self.resilience = Some(resilience);
        self
    }
}

/// An active session with a chat completions provider.
///
/// Created by [`CompletionsAdapter::start_session`](ModelAdapter::start_session).
pub struct CompletionsSession<P: CompletionsProvider> {
    client: Http,
    provider: Arc<P>,
    model: Option<String>,
    provider_label: String,
    authentication: Option<Authentication>,
    resilience: Option<ResilienceConfig>,
    _session_config: SessionConfig,
}

/// A turn from a chat completion response.
pub struct CompletionsTurn {
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
    postprocess: PostprocessResponse,
}

impl CompletionsTurn {
    fn buffered(events: VecDeque<ModelTurnEvent>) -> Self {
        Self {
            inner: TurnInner::Buffered { events },
        }
    }

    fn streaming(
        body: BodyStream,
        postprocess: PostprocessResponse,
        idle_timeout: Option<Duration>,
        detect_truncation: bool,
        deadline: Option<LogicalDeadline>,
        integrity: TruncatedStreamDetector,
        replay: Option<StreamReplay>,
    ) -> Self {
        Self {
            inner: TurnInner::Streaming(Box::new(StreamingState {
                body,
                decoder: SseDecoder::new(),
                translator: EventTranslator::new(),
                pending: VecDeque::new(),
                eof: false,
                visible_output: false,
                detect_truncation,
                idle_timeout,
                deadline,
                integrity,
                replay,
                postprocess,
            })),
        }
    }
}

#[derive(Clone)]
struct ReplayRequest<P: CompletionsProvider> {
    client: Http,
    provider: Arc<P>,
    body: Bytes,
    authentication: Option<Authentication>,
    authentication_attempt: Arc<std::sync::Mutex<Option<AuthenticationAttempt>>>,
    reauthenticated: Arc<AtomicBool>,
    resilience: Option<ResilienceConfig>,
    deadline: Option<LogicalDeadline>,
    retries_used: Arc<AtomicUsize>,
}

impl<P: CompletionsProvider + 'static> ReplayRequest<P> {
    async fn authentication_attempt(&self) -> Result<Option<AuthenticationAttempt>, LoopError> {
        let existing = self
            .authentication_attempt
            .lock()
            .map_err(|_| LoopError::Provider("authentication state lock poisoned".into()))?
            .clone();
        if existing.is_some() || self.authentication.is_none() {
            return Ok(existing);
        }
        let authenticate = self
            .authentication
            .as_ref()
            .expect("checked above")
            .authenticate(None);
        let attempt = run_bounded(authenticate, None, self.deadline.as_ref(), "authentication")
            .await
            .map_err(|error| LoopError::Provider(format!("authentication failed: {error}")))?;
        *self
            .authentication_attempt
            .lock()
            .map_err(|_| LoopError::Provider("authentication state lock poisoned".into()))? =
            Some(attempt.clone());
        Ok(Some(attempt))
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
            .expect("a reserved retry has resilience")
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
        let provider_name = self.provider.provider_name().to_owned();
        loop {
            let authentication = self.authentication_attempt().await?;
            let mut builder = self
                .client
                .post(self.provider.endpoint_url())
                .header("Content-Type", "application/json");
            builder = self.provider.preprocess_request(builder);
            if self.provider.streaming() {
                builder = builder.header("Accept", "text/event-stream");
            }
            if let Some(authentication) = authentication.as_ref() {
                builder = builder.headers(authentication.headers().clone());
            }
            let request = builder.body(self.body.clone()).build().map_err(|error| {
                LoopError::Provider(format!("{provider_name} request failed: {error}"))
            })?;

            match self.execute_attempt(request).await {
                Ok(response) => {
                    if response.status() == StatusCode::UNAUTHORIZED
                        && let Some(authentication) = authentication.as_ref()
                        && !self.reauthenticated.swap(true, Ordering::SeqCst)
                    {
                        drop(response);
                        self.reauthenticate(authentication.clone()).await?;
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
                        "{provider_name} request failed: {error}"
                    )));
                }
                Err(error) => {
                    return Err(LoopError::Provider(format!(
                        "{provider_name} request failed: {error}"
                    )));
                }
            }
        }
    }

    async fn replay_stream(&self) -> Result<(BodyStream, TruncatedStreamDetector), LoopError> {
        let retry_number = self.reserve_retry().ok_or_else(|| {
            LoopError::Provider(
                "completions stream failed before output and retry budget is exhausted".into(),
            )
        })?;
        self.wait_before_retry(retry_number, None).await?;
        let response = self.open_response().await?;
        if !response.status().is_success() {
            return Err(LoopError::Provider(format!(
                "{} stream replay failed with status {}",
                self.provider.provider_name(),
                response.status()
            )));
        }
        let integrity = TruncatedStreamDetector::from_headers(response.headers());
        Ok((response.bytes_stream(), integrity))
    }
}

#[async_trait]
impl<P: CompletionsProvider + 'static> ModelAdapter for CompletionsAdapter<P> {
    type Session = CompletionsSession<P>;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        // The provider's typed request config is opaque to the adapter; the
        // serialized "model" key is the chat-completions contract, so pull
        // the telemetry model name from there.
        let model = serde_json::to_value(self.provider.config())
            .ok()
            .and_then(|config| {
                config
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        Ok(CompletionsSession {
            client: self.client.clone(),
            provider: self.provider.clone(),
            model,
            provider_label: self.provider_label.clone(),
            authentication: self.authentication.clone(),
            resilience: self.resilience.clone(),
            _session_config: config,
        })
    }

    fn provider_name(&self) -> Option<&str> {
        Some(&self.provider_label)
    }
}

#[async_trait]
impl<P: CompletionsProvider + 'static> ModelSession for CompletionsSession<P> {
    type Turn = CompletionsTurn;

    async fn begin_turn(
        &mut self,
        turn_request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<CompletionsTurn, LoopError> {
        let provider = self.provider.clone();
        let provider_name = provider.provider_name().to_owned();
        let deadline = self
            .resilience
            .as_ref()
            .map(|config| LogicalDeadline::new(config.retry_budget));

        let request_future = async {
            let body = request::build_request_body(provider.as_ref(), &turn_request)
                .map_err(|e| LoopError::Provider(e.to_string()))?;
            let body = serde_json::to_vec(&body)
                .map(Bytes::from)
                .map_err(|e| LoopError::Provider(format!("failed to serialize request: {e}")))?;
            let replay_request = ReplayRequest {
                client: self.client.clone(),
                provider: provider.clone(),
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
                if provider.streaming() && status.is_success() {
                    let provider_for_postprocess = provider.clone();
                    let postprocess: PostprocessResponse = Arc::new(move |usage, metadata, raw| {
                        provider_for_postprocess.postprocess_response(usage, metadata, raw);
                    });
                    let integrity = TruncatedStreamDetector::from_headers(response.headers());
                    let idle_timeout = self
                        .resilience
                        .as_ref()
                        .and_then(|config| config.stream_idle_timeout);
                    let replay = self.resilience.as_ref().and_then(|config| {
                        (config.max_retries > 0).then(|| {
                            let replay_request = replay_request.clone();
                            Box::new(move || {
                                let replay_request = replay_request.clone();
                                Box::pin(async move { replay_request.replay_stream().await })
                                    as BoxFuture<'static, _>
                            }) as StreamReplay
                        })
                    });
                    return Ok(CompletionsTurn::streaming(
                        response.bytes_stream(),
                        postprocess,
                        idle_timeout,
                        self.resilience.is_some(),
                        deadline.clone(),
                        integrity,
                        replay,
                    ));
                }

                let body_result = replay_request.read_response_text(response).await;
                let body = match body_result {
                    Ok(body) => body,
                    Err(error) if is_retryable_body_read(status, &error) => {
                        if let Some(retry_number) = replay_request.reserve_retry() {
                            replay_request.wait_before_retry(retry_number, None).await?;
                            continue;
                        }
                        return Err(LoopError::Provider(format!(
                            "failed to read {provider_name} response body: {error}"
                        )));
                    }
                    Err(error) => {
                        return Err(LoopError::Provider(format!(
                            "failed to read {provider_name} response body: {error}"
                        )));
                    }
                };

                provider.preprocess_response(status, &body)?;
                if !status.is_success() {
                    return Err(LoopError::Provider(format!(
                        "{provider_name} request failed with status {status}: {body}"
                    )));
                }
                let (events, _raw) = response::build_turn_from_response(provider.as_ref(), &body)
                    .map_err(|e| LoopError::Provider(e.to_string()))?;
                return Ok(CompletionsTurn::buffered(events));
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
        self.model.as_deref()
    }

    fn provider_name(&self) -> Option<&str> {
        Some(&self.provider_label)
    }
}

#[async_trait]
impl ModelTurn for CompletionsTurn {
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
                    postprocess,
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
                    postprocess,
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
    postprocess: &PostprocessResponse,
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
                    LoopError::Provider(format!("invalid UTF-8 in completions stream: {e}"))
                })?;
                for sse in decoder.feed(text) {
                    for event in translator
                        .handle(&sse, postprocess)
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
                    Ok(()) => "completions stream ended before its terminal event".to_owned(),
                    Err(error) => format!("completions stream body error: {error}"),
                };
                Some(LoopError::Provider(message))
            }
            Err(error) if !error.is_retryable_transport() => {
                return Err(LoopError::Provider(format!(
                    "completions stream body error: {error}"
                )));
            }
            Err(error) => Some(LoopError::Provider(format!(
                "completions stream body error: {error}"
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
