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
        // Only successful response.completed checkpoints may survive the lease.
        if let Some(id) = response_id
            && id.len() <= 1024
            && connection.completed_ids.len() < 1024
        {
            connection.completed_ids.insert(id.to_owned());
            if !connection.checkpoint.as_ref().is_some_and(|c| c.completed) {
                connection.checkpoint = None;
            } else if let Some(checkpoint) = &mut connection.checkpoint {
                checkpoint.response_id = id.to_owned();
            }
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
    checkpoint: Option<Checkpoint>,
    previous_response_id: Option<String>,
}

// All fields are private to the leased connection: no new lock or persistent
// state. Serialized buffers zeroize encrypted reasoning on every exit path.
struct Checkpoint {
    request: Zeroizing<String>,
    response_id: String,
    raw_output: BTreeMap<u64, Zeroizing<String>>,
    replay_output: BTreeMap<u64, Zeroizing<String>>,
    bytes: usize,
    completed: bool,
}

impl Checkpoint {
    fn suffix(&self, current: &Value) -> Option<Vec<Value>> {
        if !self.completed || self.response_id.is_empty() {
            return None;
        }
        let mut previous: Value = serde_json::from_str(&self.request).ok()?;
        let result = (|| {
            let old = previous.as_object()?;
            let new = current.as_object()?;
            if old.len() != new.len()
                || old
                    .iter()
                    .any(|(key, value)| key != "input" && new.get(key) != Some(value))
            {
                return None;
            }
            let old_input = old.get("input")?.as_array()?;
            let input = new.get("input")?.as_array()?;
            if !input.starts_with(old_input) {
                return None;
            }
            let mut offset = old_input.len();
            for encoded in self.replay_output.values() {
                let mut replay: Value = serde_json::from_str(encoded).ok()?;
                let matches = replay.as_array().is_some_and(|items| {
                    let end = offset + items.len();
                    let matches = input.get(offset..end) == Some(items.as_slice());
                    offset = end;
                    matches
                });
                zeroize_encrypted_content(&mut replay);
                if !matches {
                    return None;
                }
            }
            Some(input[offset..].to_vec())
        })();
        zeroize_encrypted_content(&mut previous);
        result
    }
}

impl Connection {
    pub(super) fn missing_previous(&self, decoder: &ResponsesSseDecoder) -> bool {
        // Successful-completion-only, single-flight reuse and unique response IDs
        // ensure this predecessor was never used by an earlier request. A delayed
        // rejection naming an older predecessor cannot authorize a replay here.
        self.previous_response_id.as_ref().is_some_and(|id| {
            decoder.previous_response_missing.as_ref() == Some(id) && !decoder.state.created
        })
    }

    pub(super) fn observe(
        &mut self,
        chunk: &[u8],
        state: &ResponsesState,
        context: &ResponsesRequestContext,
    ) {
        let Some(checkpoint) = &mut self.checkpoint else {
            return;
        };
        let Ok(mut event) = serde_json::from_slice::<Value>(chunk) else {
            self.checkpoint = None;
            return;
        };
        let valid = (|| {
            match event.get("type").and_then(Value::as_str) {
                Some("response.completed") => {
                    let response = event.get("response")?;
                    if response
                        .get("status")
                        .is_some_and(|status| status != "completed")
                    {
                        return None;
                    }
                    // Streamed output_item.done items are the indexed source of
                    // truth used by Finished and the checkpoint. Live WS completion
                    // envelopes can omit output or send output:[] despite those
                    // items. An empty terminal array must not erase them. Validate
                    // any nonempty terminal copy against the observed raw items.
                    if let Some(output) = response.get("output") {
                        let output = output.as_array()?;
                        if !output.is_empty() && output.len() != checkpoint.raw_output.len() {
                            return None;
                        }
                        for (item, raw) in output.iter().zip(checkpoint.raw_output.values()) {
                            let mut observed: Value = serde_json::from_str(raw).ok()?;
                            let same = *item == observed;
                            zeroize_encrypted_content(&mut observed);
                            if !same {
                                return None;
                            }
                        }
                    }
                    checkpoint.completed = true;
                }
                Some("response.output_item.done") => {
                    let index = event.get("output_index")?.as_u64()?;
                    // The same representation the next real transcript produces:
                    // status/annotations and readable reasoning summaries are not
                    // request continuation fields in this adapter.
                    let item = state.output.get(&index)?;
                    let mut replay = Value::Array(
                        encode_item(
                            &context.config,
                            &context.session_id,
                            context.auth.binding(),
                            item,
                        )
                        .ok()?,
                    );
                    let encoded = serde_json::to_string(&replay).ok().map(Zeroizing::new);
                    let empty = replay.as_array().is_none_or(Vec::is_empty);
                    zeroize_encrypted_content(&mut replay);
                    if empty {
                        return None;
                    }
                    let encoded = encoded?;
                    let raw = Zeroizing::new(serde_json::to_string(event.get("item")?).ok()?);
                    checkpoint.bytes = checkpoint
                        .bytes
                        .checked_add(raw.len())?
                        .checked_add(encoded.len())?;
                    if checkpoint.raw_output.len() >= context.config.limits.max_items
                        || checkpoint.bytes > context.config.limits.max_request_bytes
                    {
                        return None;
                    }
                    checkpoint.raw_output.insert(index, raw);
                    checkpoint.replay_output.insert(index, encoded);
                }
                _ => {}
            }
            Some(())
        })()
        .is_some();
        zeroize_encrypted_content(&mut event);
        if !valid {
            self.checkpoint = None;
        }
    }

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
            checkpoint: None,
            previous_response_id: None,
        })
    };
    // Encode the complete credential-bound transcript first. The connection-local
    // checkpoint is only an optimization, never an alternative source of history.
    let mut value: Value = serde_json::from_slice(&context.body)
        .map_err(|_| protocol_failure("invalid encoded Responses request"))?;
    let fields = value
        .as_object_mut()
        .ok_or_else(|| protocol_failure("invalid Responses request object"))?;
    fields.remove("stream");
    fields.remove("background");
    fields.insert("type".into(), json!("response.create"));
    let full = Zeroizing::new(
        serde_json::to_string(&value)
            .map_err(|_| protocol_failure("could not serialize WebSocket request"))?,
    );
    connection.previous_response_id = None;
    if let Some(checkpoint) = connection.checkpoint.take()
        && let Some(suffix) = checkpoint.suffix(&value)
    {
        connection.previous_response_id = Some(checkpoint.response_id.clone());
        zeroize_encrypted_content(&mut value["input"]);
        value["input"] = Value::Array(suffix);
        value["previous_response_id"] = json!(checkpoint.response_id);
    }
    connection.checkpoint = Some(Checkpoint {
        request: full,
        response_id: String::new(),
        raw_output: BTreeMap::new(),
        replay_output: BTreeMap::new(),
        bytes: 0,
        completed: false,
    });
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
