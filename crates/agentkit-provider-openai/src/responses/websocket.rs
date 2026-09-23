//! A session owns at most one leased socket. No lock is held across I/O.
use super::*;
use futures_util::{FutureExt, SinkExt, StreamExt};
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        Message,
        handshake::{client::generate_key, derive_accept_key},
        protocol::{Role, WebSocketConfig},
    },
};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Default)]
pub(super) struct Session {
    busy: bool,
    http_only: bool,
    idle: Option<Box<Connection>>,
}

/// Writers: checkout, sticky fallback, completion and Drop. Checkout commits the
/// busy bit and takes the socket under one guard. Only that lease can return it.
/// Failure/cancellation/unwind drops the private socket; Drop clears the claim.
/// Poison is fail-closed, never recovered. No callbacks/I/O/destructors run under
/// the guard. A completed owned turn releases its claim before returning Finished.
pub(super) struct Lease {
    session: Arc<Mutex<Session>>,
    connection: Option<Box<Connection>>,
    reusable: bool,
    http_only: bool,
}

impl Lease {
    pub(super) fn checkout(session: Arc<Mutex<Session>>) -> Result<Self, LoopError> {
        let (connection, http_only) = {
            let mut state = session
                .lock()
                .map_err(|_| local_error(ProviderFailureReason::Protocol))?;
            if state.busy {
                return Err(LoopError::Provider(
                    "Responses WebSocket session already has an active turn".into(),
                ));
            }
            state.busy = true;
            (state.idle.take(), state.http_only)
        };
        Ok(Self {
            session,
            connection,
            reusable: false,
            http_only,
        })
    }

    pub(super) fn http_only(&self) -> bool {
        self.http_only
    }

    fn fallback(&mut self) -> Result<(), AttemptFailure> {
        self.session
            .lock()
            .map_err(|_| protocol_failure("WebSocket session lock poisoned"))?
            .http_only = true;
        self.http_only = true;
        Ok(())
    }

    pub(super) fn complete(&mut self, mut connection: Box<Connection>, response_id: Option<&str>) {
        // Keep bounded replay correlation, not a server-side continuation cache.
        if let Some(id) = response_id
            && connection.completed_ids.len() < 1024
        {
            connection.completed_ids.insert(id.to_owned());
            self.connection = Some(connection);
            self.reusable = true;
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let returned = if self.reusable {
            self.connection.take()
        } else {
            None
        };
        // No poison recovery: a poisoned session rejects all future checkouts.
        let old = if let Ok(mut state) = self.session.lock() {
            state.busy = false;
            std::mem::replace(&mut state.idle, returned)
        } else {
            returned
        };
        drop(old);
    }
}

pub(super) struct Connection {
    stream: WebSocketStream<reqwest::Upgraded>,
    auth_headers: HeaderMap,
    binding: Option<String>,
    completed_ids: BTreeSet<String>,
}

impl Connection {
    pub(super) fn can_retry_rejection(&self) -> bool {
        // Wrapped errors have no reliable request correlation. On a reused
        // socket they may belong to a previous turn, even after a pending read.
        self.completed_ids.is_empty()
    }

    pub(super) async fn recv(&mut self) -> Result<Option<agentkit_http::Bytes>, HttpError> {
        // Bound control traffic as well as data, including peers that only ping.
        for _ in 0..128 {
            match self.stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    if !self.completed_ids.is_empty() {
                        let mut value: Value = serde_json::from_str(&text)
                            .map_err(|_| HttpError::Other("invalid WebSocket JSON".into()))?;
                        let stale = value
                            .pointer("/response/id")
                            .or_else(|| value.get("response_id"))
                            .and_then(Value::as_str)
                            .is_some_and(|id| self.completed_ids.contains(id));
                        zeroize_encrypted_content(&mut value);
                        if stale {
                            return Err(HttpError::Other(
                                "stale Responses WebSocket response".into(),
                            ));
                        }
                    }
                    return Ok(Some(agentkit_http::Bytes::copy_from_slice(text.as_bytes())));
                }
                Some(Ok(Message::Ping(_))) => self
                    .stream
                    .flush()
                    .await
                    .map_err(|_| HttpError::Other("WebSocket pong failed".into()))?,
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) | None => return Ok(None),
                _ => {
                    return Err(HttpError::Other(
                        "invalid or failed Responses WebSocket message".into(),
                    ));
                }
            }
        }
        Err(HttpError::Other(
            "excessive WebSocket control frames".into(),
        ))
    }
}

pub(super) async fn send(
    context: &mut ResponsesRequestContext,
    mut headers: HeaderMap,
) -> Result<Option<LiveAttempt>, AttemptFailure> {
    let previous = context
        .websocket
        .as_mut()
        .expect("WebSocket lease")
        .connection
        .take();
    let previous = previous.and_then(|mut connection| {
        // Polling mutates the framed reader, so Option::filter cannot be used.
        // Unsolicited data/close/error at a boundary forces a fresh connection.
        let clean = connection.stream.next().now_or_never().is_none();
        clean.then_some(connection)
    });
    let mut connection = if let Some(connection) = previous.filter(|connection| {
        connection.auth_headers == *context.auth.headers()
            && connection.binding.as_deref() == context.auth.binding()
    }) {
        context.tracker.accounting.attempts = context.tracker.accounting.attempts.saturating_add(1);
        connection
    } else {
        headers.remove("idempotency-key"); // WebSocket response.create has no idempotency contract.
        headers.remove("accept");
        headers.remove("content-type");
        headers.insert(
            "openai-beta",
            HeaderValue::from_static("responses_websockets=2026-02-06"),
        );
        let key = generate_key();
        headers.insert("connection", HeaderValue::from_static("Upgrade"));
        headers.insert("upgrade", HeaderValue::from_static("websocket"));
        headers.insert("sec-websocket-version", HeaderValue::from_static("13"));
        headers.insert(
            "sec-websocket-key",
            HeaderValue::from_str(&key).map_err(|_| protocol_failure("invalid WebSocket nonce"))?,
        );
        let client = reqwest::Client::builder()
            .http1_only()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(HANDSHAKE_TIMEOUT)
            .timeout(HANDSHAKE_TIMEOUT)
            .build()
            .map_err(|_| protocol_failure("could not build WebSocket upgrade client"))?;
        context.tracker.accounting.attempts = context.tracker.accounting.attempts.saturating_add(1);
        // HTTP/1 upgrade on https uses reqwest's existing rustls trust/proxy stack;
        // it is the same wire operation as connecting to the corresponding wss URL.
        let response = client
            .get(&context.config.endpoint)
            .headers(headers)
            .send()
            .await
            .map_err(|error| transport_failure(HttpError::request(error)))?;
        let status = response.status();
        if status == StatusCode::UPGRADE_REQUIRED
            && context.config.transport == OpenAIResponsesTransport::Auto
        {
            context
                .websocket
                .as_mut()
                .expect("WebSocket lease")
                .fallback()?;
            return Ok(None);
        }
        if status != StatusCode::SWITCHING_PROTOCOLS {
            return Err(AttemptFailure {
                error: Box::new(provider_error(
                    ProviderFailureReason::HttpStatus,
                    ProviderClassification {
                        http_status: Some(status.as_u16()),
                        ..ProviderClassification::default()
                    },
                )),
                retryable: retryable_response_status(status, context.config.profile),
                headers: retry_headers(response.headers()),
            });
        }
        validate_handshake(response.headers(), &key)?;
        let turn_state = if context.config.profile == OpenAIResponsesProfile::ChatGptPrivate {
            validated_turn_state_header(response.headers())?
        } else {
            None
        };
        if let Some(captured) = turn_state {
            let mut state = context
                .turn_state
                .lock()
                .map_err(|_| protocol_failure("turn-state lock poisoned"))?;
            if state.as_ref().is_some_and(|expected| expected != captured) {
                return Err(protocol_failure("provider changed x-codex-turn-state"));
            }
            *state = Some(captured);
        }
        let socket = response
            .upgrade()
            .await
            .map_err(|_| protocol_failure("WebSocket upgrade failed"))?;
        let limit = context.config.limits.max_attempt_bytes;
        let config = WebSocketConfig::default()
            .max_message_size(Some(limit))
            .max_frame_size(Some(limit))
            .write_buffer_size(0);
        Box::new(Connection {
            stream: WebSocketStream::from_raw_socket(socket, Role::Client, Some(config)).await,
            auth_headers: context.auth.headers().clone(),
            binding: context.auth.binding().map(str::to_owned),
            completed_ids: BTreeSet::new(),
        })
    };
    // Always send the authoritative, credential-bound full transcript. Incremental
    // previous_response_id is intentionally not used without a lossless prefix proof.
    let mut value: Value = serde_json::from_slice(&context.body)
        .map_err(|_| protocol_failure("invalid encoded Responses request"))?;
    let fields = value
        .as_object_mut()
        .ok_or_else(|| protocol_failure("invalid Responses request object"))?;
    // Public Responses WebSocket mode excludes HTTP transport controls.
    // https://developers.openai.com/api/docs/guides/websocket-mode
    fields.remove("stream");
    fields.remove("background");
    fields.insert("type".into(), json!("response.create"));
    let serialized = serde_json::to_string(&value);
    zeroize_encrypted_content(&mut value);
    let request = Zeroizing::new(
        serialized.map_err(|_| protocol_failure("could not serialize WebSocket request"))?,
    );
    if request.len() > context.config.limits.max_request_bytes {
        return Err(protocol_failure(
            "Responses WebSocket request exceeds byte limit",
        ));
    }
    // Once send is polled, delivery is ambiguous. Never retry send/timeout errors.
    context.websocket_sent = true;
    run_bounded_http(
        async {
            connection
                .stream
                .send(Message::Text(request.as_str().into()))
                .await
                .map_err(|_| HttpError::Other("WebSocket request send failed".into()))
        },
        Some(HANDSHAKE_TIMEOUT),
        context.deadline.as_ref(),
        "WebSocket send",
    )
    .await
    .map_err(|e| nonretryable(http_loop_error(e)))?;
    Ok(Some(LiveAttempt {
        body: LiveBody::WebSocket(connection),
        truncated: TruncatedStreamDetector::from_headers(&HeaderMap::new()),
        decoder: ResponsesSseDecoder::with_policy(
            &context.config.model,
            &context.session_id,
            context.config.profile,
            context.config.request_policy.include_encrypted_reasoning,
            context.auth.binding(),
            context.turn_state.clone(),
            context.config.limits,
        ),
        deadline: None,
        eof: false,
        closed: false,
    }))
}

fn validate_handshake(headers: &HeaderMap, key: &str) -> Result<(), AttemptFailure> {
    let has_token = |name: &str, token: &str| {
        headers
            .get_all(name)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(','))
            .any(|v| v.trim().eq_ignore_ascii_case(token))
    };
    let accepts: Vec<_> = headers.get_all("sec-websocket-accept").iter().collect();
    if !has_token("connection", "upgrade")
        || !has_token("upgrade", "websocket")
        || accepts.len() != 1
        || accepts[0].as_bytes() != derive_accept_key(key.as_bytes()).as_bytes()
        || headers.contains_key("sec-websocket-extensions")
        || headers.contains_key("sec-websocket-protocol")
    {
        return Err(protocol_failure("invalid Responses WebSocket handshake"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
