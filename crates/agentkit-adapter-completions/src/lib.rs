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

use std::collections::VecDeque;
use std::sync::Arc;

use agentkit_core::{MetadataMap, TurnCancellation, Usage};
use agentkit_http::{Http, HttpError, HttpRequestBuilder, StatusCode};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use futures_util::future::{Either, select};
use serde::Serialize;
use serde_json::Value;

pub use crate::error::CompletionsError;

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

        Ok(Self {
            client,
            provider: Arc::new(provider),
        })
    }

    /// Creates a new adapter with a pre-configured [`Http`] client. Use this to
    /// attach auth headers via `default_headers`, supply custom TLS/proxies,
    /// or plug in a non-reqwest backend.
    pub fn with_client(provider: P, client: Http) -> Self {
        Self {
            client,
            provider: Arc::new(provider),
        }
    }
}

/// An active session with a chat completions provider.
///
/// Created by [`CompletionsAdapter::start_session`](ModelAdapter::start_session).
pub struct CompletionsSession<P: CompletionsProvider> {
    client: Http,
    provider: Arc<P>,
    _session_config: SessionConfig,
}

/// A completed turn holding buffered events from a chat completion response.
///
/// All events are available immediately — successive calls to
/// [`next_event`](ModelTurn::next_event) drain the internal queue.
pub struct CompletionsTurn {
    events: VecDeque<ModelTurnEvent>,
}

#[async_trait]
impl<P: CompletionsProvider + 'static> ModelAdapter for CompletionsAdapter<P> {
    type Session = CompletionsSession<P>;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(CompletionsSession {
            client: self.client.clone(),
            provider: self.provider.clone(),
            _session_config: config,
        })
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
        let provider = &self.provider;
        let provider_name = provider.provider_name().to_owned();

        let request_future = async {
            let body = request::build_request_body(provider.as_ref(), &turn_request)
                .map_err(|e| LoopError::Provider(e.to_string()))?;

            let http = self
                .client
                .post(provider.endpoint_url())
                .header("Content-Type", "application/json");

            let http = provider.preprocess_request(http);

            let response = http.json(&body).send().await.map_err(|error| {
                LoopError::Provider(format!("{provider_name} request failed: {error}"))
            })?;

            let status = response.status();
            let body = response.text().await.map_err(|error| {
                LoopError::Provider(format!(
                    "failed to read {provider_name} response body: {error}"
                ))
            })?;

            provider.preprocess_response(status, &body)?;

            if !status.is_success() {
                return Err(LoopError::Provider(format!(
                    "{provider_name} request failed with status {status}: {body}"
                )));
            }

            let (events, _raw) = response::build_turn_from_response(provider.as_ref(), &body)
                .map_err(|e| LoopError::Provider(e.to_string()))?;

            Ok(CompletionsTurn { events })
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
        Ok(self.events.pop_front())
    }
}
