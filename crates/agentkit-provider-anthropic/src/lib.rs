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

use agentkit_core::TurnCancellation;
use agentkit_http::{BodyStream, Http, HttpError, HttpRequestBuilder};
use agentkit_loop::{
    LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, SessionConfig, TurnRequest,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::future::{Either, select};

use crate::stream::{EventTranslator, SseDecoder};

pub use crate::config::{
    AnthropicConfig, AnthropicMcpServer, DEFAULT_ANTHROPIC_VERSION, DEFAULT_ENDPOINT, OutputEffort,
    OutputFormat, ServiceTier, ThinkingConfig, ToolChoice,
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
        Ok(Self {
            client,
            config: Arc::new(config),
        })
    }

    /// Creates a new adapter using a pre-configured [`Http`] client.
    pub fn with_client(config: AnthropicConfig, client: Http) -> Self {
        Self {
            client,
            config: Arc::new(config),
        }
    }
}

/// An active session with the Anthropic Messages API.
pub struct AnthropicSession {
    client: Http,
    config: Arc<AnthropicConfig>,
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

struct StreamingState {
    body: BodyStream,
    decoder: SseDecoder,
    translator: EventTranslator,
    pending: VecDeque<ModelTurnEvent>,
    eof: bool,
}

#[async_trait]
impl ModelAdapter for AnthropicAdapter {
    type Session = AnthropicSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(AnthropicSession {
            client: self.client.clone(),
            config: self.config.clone(),
            _session_config: config,
        })
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

        let request_future = async move {
            let body = request::build_request_body(&config, &turn_request)
                .map_err(|e| LoopError::Provider(e.to_string()))?;

            let betas = collect_beta_flags(&config);

            let mut http = self
                .client
                .post(&config.base_url)
                .header("Content-Type", "application/json")
                .header("anthropic-version", config.anthropic_version.as_str());

            http = attach_auth(http, &config)?;

            if !betas.is_empty() {
                let joined = betas.into_iter().collect::<Vec<_>>().join(",");
                http = http.header("anthropic-beta", joined);
            }

            http = http.header(
                "User-Agent",
                concat!("agentkit-provider-anthropic/", env!("CARGO_PKG_VERSION")),
            );

            if config.streaming {
                http = http.header("Accept", "text/event-stream");
            }

            let response = http.json(&body).send().await.map_err(|error| {
                LoopError::Provider(format!("Anthropic request failed: {error}"))
            })?;

            let status = response.status();

            if !status.is_success() {
                // Drain the body for the error message, regardless of mode —
                // the server typically returns JSON error details here.
                let body_text = response.text().await.unwrap_or_default();
                return Err(LoopError::Provider(format!(
                    "Anthropic request failed with status {status}: {body_text}"
                )));
            }

            if config.streaming {
                Ok(AnthropicTurn {
                    inner: TurnInner::Streaming(Box::new(StreamingState {
                        body: response.bytes_stream(),
                        decoder: SseDecoder::new(),
                        translator: EventTranslator::new(),
                        pending: VecDeque::new(),
                        eof: false,
                    })),
                })
            } else {
                let body_text = response.text().await.map_err(|error| {
                    LoopError::Provider(format!("failed to read Anthropic response body: {error}"))
                })?;
                let events = response::build_turn_from_response(&body_text)
                    .map_err(|e| LoopError::Provider(e.to_string()))?;
                Ok(AnthropicTurn {
                    inner: TurnInner::Buffered { events },
                })
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
                } = state.as_mut();
                next_streaming_event(body, decoder, translator, pending, eof, cancellation).await
            }
        }
    }
}

/// Pulls the next event from an active SSE stream, decoding more bytes as
/// needed. Returns `Ok(None)` once the translator has emitted `Finished` and
/// the pending queue is empty.
async fn next_streaming_event(
    body: &mut BodyStream,
    decoder: &mut SseDecoder,
    translator: &mut EventTranslator,
    pending: &mut VecDeque<ModelTurnEvent>,
    eof: &mut bool,
    cancellation: Option<TurnCancellation>,
) -> Result<Option<ModelTurnEvent>, LoopError> {
    loop {
        if let Some(event) = pending.pop_front() {
            return Ok(Some(event));
        }
        if *eof || translator.is_done() {
            return Ok(None);
        }

        // Await the next chunk, racing against cancellation so long-lived
        // streams can be interrupted mid-response.
        let chunk = if let Some(cancellation) = cancellation.as_ref() {
            let next = body.next();
            futures_util::pin_mut!(next);
            let cancelled = cancellation.cancelled();
            futures_util::pin_mut!(cancelled);
            match select(next, cancelled).await {
                Either::Left((chunk, _)) => chunk,
                Either::Right((_, _)) => return Err(LoopError::Cancelled),
            }
        } else {
            body.next().await
        };

        match chunk {
            Some(Ok(bytes)) => {
                let text = std::str::from_utf8(&bytes).map_err(|e| {
                    LoopError::Provider(format!("invalid UTF-8 in Anthropic stream: {e}"))
                })?;
                for sse in decoder.feed(text) {
                    for produced in translator.handle(&sse)? {
                        pending.push_back(produced);
                    }
                }
            }
            Some(Err(e)) => {
                return Err(LoopError::Provider(format!(
                    "Anthropic stream body error: {e}"
                )));
            }
            None => {
                *eof = true;
            }
        }
    }
}

fn attach_auth(
    builder: HttpRequestBuilder,
    config: &AnthropicConfig,
) -> Result<HttpRequestBuilder, LoopError> {
    if let Some(token) = &config.auth_token {
        return Ok(builder.bearer_auth(token));
    }
    if let Some(key) = &config.api_key {
        return Ok(builder.header("x-api-key", key.as_str()));
    }
    Err(LoopError::Provider(
        AnthropicError::MissingCredentials.to_string(),
    ))
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
    use agentkit_http::HttpError;
    use bytes::Bytes;
    use futures_util::stream;

    use super::*;

    #[test]
    fn rejects_zero_max_tokens() {
        match AnthropicConfig::new("k", "claude-opus-4-7", 0) {
            Err(AnthropicError::InvalidMaxTokens) => {}
            other => panic!("expected InvalidMaxTokens, got {:?}", other.map(|_| ())),
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
