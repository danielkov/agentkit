//! Loopback-only transport tests; no live inference.
use super::*;
use agentkit_core::{SessionId, TurnId};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use tokio_tungstenite::tungstenite::{self, WebSocket};

// Same text/tool/reasoning/usage events as the HTTP decoder tests. Live WS
// completion envelopes can contain output:[] even after output_item.done; the
// streamed items, not this empty terminal field, supply the real transcript.
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
data: {"type":"response.completed","sequence_number":18,"response":{"id":"resp-1","model":"gpt-test","output":[],"usage":{"input_tokens":3,"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2}}}}

"#;

const WAIT: Duration = Duration::from_secs(5);
type Socket = WebSocket<TcpStream>;

fn server<F>(run: F) -> (String, thread::JoinHandle<()>)
where
    F: FnOnce(TcpListener) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1/responses", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    (endpoint, thread::spawn(move || run(listener)))
}
fn accept(listener: &TcpListener) -> TcpStream {
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(WAIT)).unwrap();
                stream.set_write_timeout(Some(WAIT)).unwrap();
                return stream;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(start.elapsed() < WAIT, "missing expected connection");
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => panic!("accept: {e}"),
        }
    }
}
fn socket(listener: &TcpListener) -> Socket {
    tungstenite::accept(accept(listener)).unwrap()
}
fn receive(ws: &mut Socket) -> Value {
    let value = receive_continuation(ws);
    assert!(value.get("previous_response_id").is_none());
    value
}
fn receive_continuation(ws: &mut Socket) -> Value {
    let message = ws.read().unwrap();
    let value: Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
    assert_eq!(value["type"], "response.create");
    assert!(
        value.get("stream").is_none(),
        "HTTP stream field on WS wire"
    );
    assert!(
        value.get("background").is_none(),
        "HTTP background field on WS wire"
    );
    assert!(value.get("model").is_some_and(Value::is_string));
    assert!(value.get("input").is_some_and(Value::is_array));
    value
}
fn success(ws: &mut Socket) {
    success_id(ws, "resp-1");
}
fn success_id(ws: &mut Socket, id: &str) {
    for line in SUCCESS
        .replace("resp-1", id)
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        ws.send(Message::Text(line.into())).unwrap();
    }
}
// Read one request without over-reading into the next. Close each HTTP response.
fn http(listener: &TcpListener, status: &str, body: &str) -> (String, Value) {
    let mut stream = accept(listener);
    let mut raw = Vec::new();
    while !raw.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        stream.read_exact(&mut byte).unwrap();
        raw.push(byte[0]);
        assert!(raw.len() < 64 * 1024);
    }
    let headers = String::from_utf8(raw).unwrap();
    let length = headers
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    let mut request = vec![0; length];
    stream.read_exact(&mut request).unwrap();
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    (
        headers,
        serde_json::from_slice(&request).unwrap_or(Value::Null),
    )
}
fn config(endpoint: &str, transport: OpenAIResponsesTransport) -> OpenAIResponsesConfig {
    OpenAIResponsesConfig::new("loopback-test", "gpt-test")
        .with_endpoint(endpoint)
        .with_transport(transport)
        .with_resilience(ResilienceConfig {
            max_retries: 0,
            retry_budget: WAIT,
            attempt_timeout: Some(Duration::from_secs(2)),
            stream_idle_timeout: Some(Duration::from_secs(2)),
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        })
}
fn request() -> TurnRequest {
    TurnRequest {
        session_id: SessionId::new("session"),
        turn_id: TurnId::new("turn"),
        transcript: vec![Item::text(ItemKind::User, "hello")],
        available_tools: vec![],
        cache: None,
        metadata: MetadataMap::new(),
    }
}
async fn drain(turn: &mut OpenAIResponsesTurn) -> Result<Vec<ModelTurnEvent>, LoopError> {
    tokio::time::timeout(WAIT, async {
        let mut events = vec![];
        while let Some(event) = turn.next_event(None).await? {
            events.push(event);
        }
        Ok(events)
    })
    .await
    .expect("turn hung")
}
fn finished(events: &[ModelTurnEvent]) {
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelTurnEvent::Finished(_)))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(event,
        ModelTurnEvent::Delta(Delta::AppendText {chunk, ..}) if chunk == "hello")));
}
#[tokio::test]
async fn reuses_completed_socket_releases_claim_and_isolates_sessions() {
    let (endpoint, peer) = server(|listener| {
        let mut first = socket(&listener);
        let one = receive(&mut first);
        success(&mut first);
        let two = receive(&mut first);
        assert_eq!(one["input"], two["input"]);
        success_id(&mut first, "resp-2");
        let mut second = socket(&listener);
        receive(&mut second);
        success(&mut second);
    });
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
            .unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    assert!(session.begin_turn(request(), None).await.is_err());
    finished(&drain(&mut turn).await.unwrap());
    // Finished, not Drop, releases the claim: keep the completed turn alive.
    let mut next = session.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut next).await.unwrap());
    let mut other = adapter
        .start_session(SessionConfig::new("other"))
        .await
        .unwrap();
    let mut third = other.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut third).await.unwrap());
    peer.join().unwrap();
}
#[tokio::test]
async fn http_and_websocket_have_identical_text_tool_reasoning_usage_events() {
    let (endpoint, peer) = server(|listener| {
        let mut ws = socket(&listener);
        let ws_request = receive(&mut ws);
        success(&mut ws);
        let (headers, http_request) = http(&listener, "200 OK", SUCCESS);
        assert!(headers.starts_with("POST /v1/responses"));
        assert_eq!(http_request["stream"], true);
        assert!(http_request.get("type").is_none());
        // Compare semantic request fields, not transport-specific wire envelopes.
        for (key, value) in http_request.as_object().unwrap() {
            if key != "stream" && key != "background" {
                assert_eq!(ws_request.get(key), Some(value), "field {key}");
            }
        }
        for key in ws_request.as_object().unwrap().keys() {
            assert!(key == "type" || http_request.get(key).is_some());
        }
    });
    let mut outputs = vec![];
    let base = config(&endpoint, OpenAIResponsesTransport::WebSocket);
    for transport in [
        OpenAIResponsesTransport::WebSocket,
        OpenAIResponsesTransport::Http,
    ] {
        let adapter = OpenAIResponsesAdapter::new(base.clone().with_transport(transport)).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let events = drain(&mut turn).await.unwrap();
        finished(&events);
        outputs.push(format!("{events:?}"));
    }
    assert_eq!(outputs[0], outputs[1]);
    peer.join().unwrap();
}
#[tokio::test]
async fn auto_426_fallback_is_sticky_but_explicit_websocket_fails() {
    let (endpoint, peer) = server(|listener| {
        let (headers, _) = http(&listener, "426 Upgrade Required", "");
        assert!(headers.starts_with("GET /v1/responses"));
        for _ in 0..2 {
            let (headers, _) = http(&listener, "200 OK", SUCCESS);
            assert!(
                headers.starts_with("POST /v1/responses"),
                "fallback must remain sticky"
            );
        }
        let (headers, _) = http(&listener, "426 Upgrade Required", "");
        assert!(headers.starts_with("GET /v1/responses"));
    });
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::Auto)).unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    for _ in 0..2 {
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        finished(&drain(&mut turn).await.unwrap());
    }
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
            .unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    assert!(session.begin_turn(request(), None).await.is_err());
    peer.join().unwrap();
}
#[tokio::test]
async fn eof_malformed_binary_and_oversized_frames_never_complete() {
    for frame in [
        None,
        Some(Message::Text("not json".into())),
        Some(Message::Binary(vec![0, 1].into())),
        Some(Message::Text("x".repeat(1025).into())),
    ] {
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            receive(&mut ws);
            if let Some(frame) = frame {
                ws.send(frame).unwrap();
            }
            let _ = ws.close(None);
            let mut fresh = socket(&listener);
            receive(&mut fresh);
            fresh.send(Message::Text(r#"{"type":"response.created","sequence_number":1,"response":{"id":"fresh","model":"gpt-test"}}"#.into())).unwrap();
            fresh.send(Message::Text(r#"{"type":"response.completed","sequence_number":2,"response":{"id":"fresh","model":"gpt-test","status":"completed","output":[]}}"#.into())).unwrap();
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        cfg.limits.max_attempt_bytes = 1024;
        cfg.limits.max_text_bytes = 1024;
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        assert!(drain(&mut turn).await.is_err());
        let mut fresh = session.begin_turn(request(), None).await.unwrap();
        let events = drain(&mut fresh).await.unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ModelTurnEvent::Finished(_)))
        );
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn dropped_and_cancelled_turns_discard_late_frames_and_release_claim() {
    for cancel in [false, true] {
        let (release, released) = std::sync::mpsc::channel();
        let (endpoint, peer) = server(move |listener| {
            let mut old = socket(&listener);
            receive(&mut old);
            released.recv_timeout(WAIT).unwrap();
            // This old response must never satisfy the next turn, even if the
            // kernel accepts the write after the client has dropped its socket.
            let late = SUCCESS
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            let _ = old.send(Message::Text(late.replace("resp-1", "late").into()));
            let mut fresh = socket(&listener);
            receive(&mut fresh);
            success_id(&mut fresh, "fresh");
        });
        let adapter =
            OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
                .unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        if cancel {
            let handle = agentkit_core::CancellationController::new();
            let cancellation = handle.handle().checkpoint();
            handle.interrupt();
            assert!(turn.next_event(Some(cancellation)).await.is_err());
        } else {
            drop(turn);
        }
        release.send(()).unwrap();
        let mut fresh = session.begin_turn(request(), None).await.unwrap();
        let events = drain(&mut fresh).await.unwrap();
        finished(&events);
        assert!(events.iter().any(|event| matches!(event,
            ModelTurnEvent::Finished(result) if result.response_id.as_deref() == Some("fresh"))));
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn attempt_and_logical_deadlines_bound_silent_socket() {
    for logical in [false, true] {
        let (release, released) = std::sync::mpsc::channel();
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            receive(&mut ws);
            released.recv_timeout(WAIT).unwrap();
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        cfg.resilience = Some(ResilienceConfig {
            max_retries: 0,
            retry_budget: if logical {
                Duration::from_millis(100)
            } else {
                WAIT
            },
            attempt_timeout: if logical {
                None
            } else {
                Some(Duration::from_millis(100))
            },
            stream_idle_timeout: None,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        });
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let start = Instant::now();
        let error = drain(&mut turn).await.unwrap_err();
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(error.provider_failure().is_some());
        release.send(()).unwrap();
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn wrapped_retryable_error_reconnects_before_acceptance_but_not_after() {
    for mode in 0..3 {
        let accepted = mode != 0;
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            let original = receive(&mut ws);
            if accepted {
                ws.send(Message::Text(r#"{"type":"response.created","sequence_number":1,"response":{"id":"accepted","model":"gpt-test"}}"#.into())).unwrap();
            }
            if mode == 2 {
                ws.send(Message::Text(r#"{"type":"response.failed","sequence_number":2,"response":{"id":"accepted","error":{"code":"server_error","type":"server_error"}}}"#.into())).unwrap();
            } else {
                ws.send(Message::Text(r#"{"type":"error","status":429,"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"retry later"}}"#.into())).unwrap();
            }
            if !accepted {
                let mut retry = socket(&listener);
                assert_eq!(original, receive(&mut retry));
                success(&mut retry);
            }
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        cfg.resilience.as_mut().unwrap().max_retries = 1;
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let result = drain(&mut turn).await;
        if accepted {
            let error = result.unwrap_err();
            assert_eq!(error.provider_failure().unwrap().accounting.attempts, 1);
        } else {
            finished(&result.unwrap());
        }
        peer.join().unwrap();
    }
}

#[test]
fn poisoned_session_is_fail_closed() {
    let session = Arc::new(Mutex::new(Session::default()));
    let poisoned = session.clone();
    assert!(
        std::panic::catch_unwind(move || {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the session");
        })
        .is_err()
    );
    assert!(Lease::checkout(session.clone()).is_err());
    assert!(Lease::checkout(session).is_err());
}

#[tokio::test]
async fn observer_unwind_does_not_poison_session_or_strand_lease() {
    use futures_util::FutureExt;
    let (endpoint, peer) = server(|listener| {
        let mut first = socket(&listener);
        receive(&mut first);
        first.send(Message::Text(r#"{"type":"error","status":429,"error":{"type":"rate_limit_error","code":"rate_limit_exceeded"}}"#.into())).unwrap();
        let mut fresh = socket(&listener);
        receive(&mut fresh);
        success(&mut fresh);
    });
    let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
    cfg.resilience.as_mut().unwrap().max_retries = 1;
    let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let callback_armed = armed.clone();
    session.set_retry_observer(Some(Arc::new(move |_: ProviderRetryEvent| {
        if callback_armed.load(std::sync::atomic::Ordering::SeqCst) {
            panic!("observer panic");
        }
    })));
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(
        std::panic::AssertUnwindSafe(turn.next_event(None))
            .catch_unwind()
            .await
            .is_err()
    );
    drop(turn);
    session.set_retry_observer(None);
    let mut fresh = session.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut fresh).await.unwrap());
    peer.join().unwrap();
}

#[tokio::test]
async fn stale_completed_response_id_is_rejected_on_reused_socket() {
    let (endpoint, peer) = server(|listener| {
        let mut ws = socket(&listener);
        receive(&mut ws);
        success(&mut ws);
        receive(&mut ws);
        ws.send(Message::Text(r#"{"type":"response.created","sequence_number":1,"response":{"id":"resp-1","model":"gpt-test"}}"#.into())).unwrap();
    });
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
            .unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut first = session.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut first).await.unwrap());
    let mut second = session.begin_turn(request(), None).await.unwrap();
    assert!(drain(&mut second).await.is_err());
    peer.join().unwrap();
}

struct RefreshAuth;
#[async_trait]
impl AuthenticationProvider for RefreshAuth {
    async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_static(if previous.is_some() {
                "Bearer refreshed"
            } else {
                "Bearer initial"
            }),
        );
        Ok(AuthenticationAttempt::stateless(headers).with_binding("loopback-identity"))
    }
}

#[tokio::test]
#[allow(
    clippy::result_large_err,
    reason = "tungstenite fixes the handshake callback error type"
)]
async fn handshake_and_wrapped_401_refresh_credentials_on_fresh_connection() {
    for wrapped in [false, true] {
        let (endpoint, peer) = server(move |listener| {
            let original = if wrapped {
                let mut first = socket(&listener);
                let request = receive(&mut first);
                first.send(Message::Text(r#"{"type":"error","status":401,"error":{"type":"authentication_error","code":"invalid_api_key"}}"#.into())).unwrap();
                Some(request)
            } else {
                let (headers, _) = http(&listener, "401 Unauthorized", "");
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer initial")
                );
                None
            };
            let mut refreshed = tungstenite::accept_hdr(
                accept(&listener),
                |request: &tungstenite::handshake::server::Request,
                 response: tungstenite::handshake::server::Response| {
                    assert_eq!(request.headers()["authorization"], "Bearer refreshed");
                    Ok(response)
                },
            )
            .unwrap();
            let replay = receive(&mut refreshed);
            if let Some(original) = original {
                assert_eq!(original, replay);
            }
            success(&mut refreshed);
        });
        let cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket)
            .with_authentication_provider(RefreshAuth);
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        finished(&drain(&mut turn).await.unwrap());
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn unsolicited_buffered_frames_force_reconnect_before_next_send() {
    let (endpoint, peer) = server(|listener| {
        let mut ws = socket(&listener);
        receive(&mut ws);
        for line in SUCCESS
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
        {
            ws.write(Message::Text(line.into())).unwrap();
        }
        ws.write(Message::Text(
            r#"{"type":"error","status":429,"error":{"code":"rate_limit_exceeded"}}"#.into(),
        ))
        .unwrap();
        ws.flush().unwrap();
        let mut fresh = socket(&listener);
        receive(&mut fresh);
        success_id(&mut fresh, "fresh-response");
    });
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
            .unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut first = session.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut first).await.unwrap());
    let mut second = session.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut second).await.unwrap());
    peer.join().unwrap();
}

#[tokio::test]
async fn partial_stale_error_on_reused_socket_never_replays_new_request() {
    let (partial_tx, partial_rx) = tokio::sync::oneshot::channel();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let (endpoint, peer) = server(move |listener| {
        let mut ws = socket(&listener);
        receive(&mut ws);
        success(&mut ws);
        let error = br#"{"type":"error","status":429,"error":{"code":"rate_limit_exceeded"}}"#;
        assert!(error.len() < 126);
        // An unmasked text frame, split before the next request is sent.
        ws.get_mut().write_all(&[0x81, error.len() as u8]).unwrap();
        let split = error.len() / 2;
        ws.get_mut().write_all(&error[..split]).unwrap();
        partial_tx.send(()).unwrap();
        receive(&mut ws);
        ws.get_mut().write_all(&error[split..]).unwrap();
        let start = Instant::now();
        loop {
            if done_rx.try_recv().is_ok() {
                assert!(
                    matches!(listener.accept(), Err(e) if e.kind() == std::io::ErrorKind::WouldBlock)
                );
                break;
            }
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    stream.set_read_timeout(Some(WAIT)).unwrap();
                    stream.set_write_timeout(Some(WAIT)).unwrap();
                    let mut replay = tungstenite::accept(stream).unwrap();
                    receive(&mut replay);
                    success_id(&mut replay, "unexpected-replay");
                    panic!("replayed a request after an uncorrelated stale error");
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("accept: {e}"),
            }
            assert!(start.elapsed() < WAIT);
            thread::sleep(Duration::from_millis(2));
        }
    });
    let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
    cfg.resilience.as_mut().unwrap().max_retries = 1;
    let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut first = session.begin_turn(request(), None).await.unwrap();
    finished(&drain(&mut first).await.unwrap());
    partial_rx.await.unwrap();
    let mut second = session.begin_turn(request(), None).await.unwrap();
    let result = drain(&mut second).await;
    done_tx.send(()).ok();
    peer.join().unwrap();
    let error = result.unwrap_err();
    assert_eq!(
        error.provider_failure().unwrap().upstream.http_status,
        Some(429)
    );
}

struct SuspendedRefresh {
    started: Arc<std::sync::atomic::AtomicBool>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl AuthenticationProvider for SuspendedRefresh {
    async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        if previous.is_some() {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
            futures_util::future::pending().await
        } else {
            let mut headers = HeaderMap::new();
            headers.insert("authorization", HeaderValue::from_static("Bearer initial"));
            Ok(AuthenticationAttempt::stateless(headers).with_binding("loopback-identity"))
        }
    }
}

#[tokio::test]
async fn dropped_refresh_future_repolls_safely_without_replaying_stale_credentials() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::Poll;
    for handshake_refresh in [false, true] {
        let (check_tx, check_rx) = std::sync::mpsc::channel();
        let (checked_tx, checked_rx) = std::sync::mpsc::channel();
        let (endpoint, peer) = server(move |listener| {
            let mut first = socket(&listener);
            receive(&mut first);
            let status = if handshake_refresh { 429 } else { 401 };
            first
                .send(Message::Text(
                    json!({"type":"error", "status": status,
                "error":{"code":"rate_limit_exceeded"}})
                    .to_string()
                    .into(),
                ))
                .unwrap();
            if handshake_refresh {
                http(&listener, "401 Unauthorized", "");
            }
            // The rejected socket is discarded before entering authentication.
            assert!(first.read().is_err());
            check_rx.recv_timeout(WAIT).unwrap();
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock,
                "repoll must not reconnect with stale credentials"
            );
            checked_tx.send(()).unwrap();
            let mut fresh = socket(&listener);
            receive(&mut fresh);
            success(&mut fresh);
        });
        let started = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket)
            .with_authentication_provider(SuspendedRefresh {
                started: started.clone(),
                calls: calls.clone(),
            });
        cfg.resilience.as_mut().unwrap().max_retries = 1;
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        let mut pending = Box::pin(turn.next_event(None));
        tokio::time::timeout(
            WAIT,
            futures_util::future::poll_fn(|cx| {
                assert!(pending.as_mut().poll(cx).is_pending());
                if started.load(Ordering::SeqCst) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }),
        )
        .await
        .unwrap();
        drop(pending);
        let error = turn.next_event(None).await.unwrap_err();
        assert_eq!(
            error.provider_failure().unwrap().reason,
            if handshake_refresh {
                ProviderFailureReason::ReplayUnsafe
            } else {
                ProviderFailureReason::Authentication
            }
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(turn.next_event(None).await.unwrap().is_none());
        check_tx.send(()).unwrap();
        checked_rx.recv_timeout(WAIT).unwrap();
        // Terminal failure released the claim even though the owned turn remains alive.
        let mut fresh = session.begin_turn(request(), None).await.unwrap();
        finished(&drain(&mut fresh).await.unwrap());
        peer.join().unwrap();
    }
}

// Like the loop, append Finished's canonical items rather than synthesizing
// continuation metadata from raw wire output or incremental text deltas.
async fn append_completed(turn: &mut OpenAIResponsesTurn, request: &mut TurnRequest) {
    let events = drain(turn).await.unwrap();
    finished(&events);
    let result = events
        .into_iter()
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
            .any(|part| matches!(part, Part::Reasoning(_)))
    );
    assert!(
        result
            .output_items
            .iter()
            .flat_map(|item| &item.parts)
            .any(|part| matches!(part, Part::ToolCall(_)))
    );
    request.transcript.extend(result.output_items);
}
fn append_tool_result(request: &mut TurnRequest) {
    request.transcript.push(Item::new(
        ItemKind::Tool,
        vec![Part::ToolResult(agentkit_core::ToolResultPart::success(
            "call-1",
            ToolOutput::text("found"),
        ))],
    ));
}
fn missing_previous(ws: &mut Socket) {
    ws.send(Message::Text(r#"{"type":"error","status":400,"error":{"type":"invalid_request_error","code":"previous_response_not_found","message":"Previous response with id 'resp-1' not found."}}"#.into())).unwrap();
}

#[tokio::test]
async fn continuation_reduces_real_tool_reasoning_rounds_and_new_user_turn() {
    let (endpoint, peer) = server(|listener| {
        let mut ws = socket(&listener);
        receive(&mut ws);
        normalized_success(&mut ws);
        for (previous, next) in [("resp-1", "resp-2"), ("resp-2", "resp-3")] {
            let wire = receive_continuation(&mut ws);
            assert_eq!(wire["previous_response_id"], previous);
            assert_eq!(
                wire["input"],
                json!([{
                    "type": "function_call_output", "call_id": "call-1", "output": "found"
                }])
            );
            success_id(&mut ws, next);
        }
        let wire = receive_continuation(&mut ws);
        assert_eq!(wire["previous_response_id"], "resp-3");
        assert_eq!(wire["input"].as_array().unwrap().len(), 2);
        assert_eq!(wire["input"][0]["type"], "function_call_output");
        assert_eq!(wire["input"][1]["role"], "user");
        assert_eq!(wire["input"][1]["content"][0]["text"], "next question");
        success_id(&mut ws, "resp-4");
    });
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
            .unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut request = request();
    for _ in 0..3 {
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        append_tool_result(&mut request);
    }
    request.turn_id = TurnId::new("next-user-turn");
    request
        .transcript
        .push(Item::text(ItemKind::User, "next question"));
    let mut turn = session.begin_turn(request, None).await.unwrap();
    finished(&drain(&mut turn).await.unwrap());
    peer.join().unwrap();
}

#[tokio::test]
async fn continuation_falls_back_to_full_input_for_mutated_history_or_settings() {
    for mutation in ["user", "tools", "system", "compaction", "call", "reasoning"] {
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            let initial = receive(&mut ws);
            decorated_success(&mut ws);
            let full = receive(&mut ws);
            assert!(
                full["input"].as_array().unwrap().len()
                    > initial["input"].as_array().unwrap().len()
            );
            assert!(
                full["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["type"] == "reasoning")
            );
            assert!(
                full["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|item| item["type"] == "function_call")
            );
            if mutation == "tools" {
                assert_ne!(full["tools"], initial["tools"]);
            } else if matches!(mutation, "system" | "compaction") {
                assert_ne!(full["input"][0], initial["input"][0]);
            }
            assert_canonical_output(
                &full,
                if mutation == "reasoning" {
                    "changed-opaque"
                } else {
                    "opaque"
                },
                if mutation == "call" {
                    json!({"q":2})
                } else {
                    json!({"q":1})
                },
            );
            success_id(&mut ws, "resp-2");
        });
        let adapter =
            OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
                .unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut request = request();
        request
            .transcript
            .insert(0, Item::text(ItemKind::System, "original instructions"));
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        append_tool_result(&mut request);
        if mutation == "tools" {
            request.available_tools.push(
                serde_json::from_value(json!({
                    "name": "lookup", "description": "changed tools",
                    "input_schema": {"type": "object"}, "annotations": {"read_only_hint": false, "destructive_hint": false,
                    "idempotent_hint": false, "needs_approval_hint": false,
                    "supports_streaming_hint": false}, "metadata": {}
                }))
                .unwrap(),
            );
        } else {
            match mutation {
                "user" => request.transcript[1] = Item::text(ItemKind::User, "edited history"),
                "system" => {
                    request.transcript[0] = Item::text(ItemKind::System, "new instructions")
                }
                "compaction" => {
                    request.transcript.drain(..2);
                    request
                        .transcript
                        .insert(0, Item::text(ItemKind::User, "compacted history"));
                }
                "call" | "reasoning" => {
                    for part in request
                        .transcript
                        .iter_mut()
                        .flat_map(|item| &mut item.parts)
                    {
                        match part {
                            Part::ToolCall(call) if mutation == "call" => {
                                call.input = json!({"q":2})
                            }
                            Part::Reasoning(reasoning) if mutation == "reasoning" => {
                                reasoning.metadata.get_mut(CONTINUATION_METADATA).unwrap()["encrypted_content"] =
                                    json!("changed-opaque");
                            }
                            _ => {}
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        let mut next = session.begin_turn(request, None).await.unwrap();
        finished(&drain(&mut next).await.unwrap());
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn continuation_missing_previous_recovers_once_on_fresh_socket() {
    for reject_full_retry in [false, true] {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (endpoint, peer) =
            server(move |listener| {
                let mut ws = socket(&listener);
                receive(&mut ws);
                success(&mut ws);
                let suffix = receive_continuation(&mut ws);
                assert_eq!(suffix["previous_response_id"], "resp-1");
                assert_eq!(suffix["input"].as_array().unwrap().len(), 1);
                missing_previous(&mut ws);
                assert!(ws.read().is_err(), "rejected socket must be discarded");
                let mut fresh = socket(&listener);
                let full = receive(&mut fresh);
                let input = full["input"].as_array().unwrap();
                assert_eq!(input.len(), 5); // user, text, reasoning, call, result
                assert_eq!(input.last(), suffix["input"].as_array().unwrap().last());
                assert_eq!(input[0]["role"], "user");
                assert!(input.iter().any(
                    |item| item["type"] == "reasoning" && item["encrypted_content"] == "opaque"
                ));
                if reject_full_retry {
                    missing_previous(&mut fresh);
                } else {
                    success_id(&mut fresh, "resp-recovered");
                }
                done_rx.recv_timeout(WAIT).unwrap();
                assert_eq!(
                    listener.accept().unwrap_err().kind(),
                    std::io::ErrorKind::WouldBlock
                );
            });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        cfg.resilience.as_mut().unwrap().max_retries = 3;
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut request = request();
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        append_tool_result(&mut request);
        let mut next = session.begin_turn(request, None).await.unwrap();
        let result = drain(&mut next).await;
        if reject_full_retry {
            assert!(result.is_err());
            assert!(next.next_event(None).await.unwrap().is_none());
        } else {
            finished(&result.unwrap());
        }
        done_tx.send(()).unwrap();
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn continuation_missing_previous_after_created_or_output_never_replays() {
    for case in ["created", "output", "code_only", "stale_id", "out_of_order"] {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            receive(&mut ws);
            success(&mut ws);
            assert_eq!(
                receive_continuation(&mut ws)["previous_response_id"],
                "resp-1"
            );
            for line in SUCCESS
                .replace("resp-1", "accepted")
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .take(match case {
                    "output" => 4,
                    "created" => 1,
                    _ => 0,
                })
            {
                ws.send(Message::Text(line.into())).unwrap();
            }
            if matches!(case, "created" | "output") {
                missing_previous(&mut ws);
            } else {
                let mut error = json!({"type":"error", "status":400, "error":{
                    "type":"invalid_request_error", "code":"previous_response_not_found"
                }});
                if case == "stale_id" {
                    error["error"]["message"] =
                        json!("Previous response with id 'resp-stale' not found.");
                } else if case == "out_of_order" {
                    ws.send(Message::Text(
                        json!({"type":"keepalive", "sequence_number":1})
                            .to_string()
                            .into(),
                    ))
                    .unwrap();
                    error["sequence_number"] = json!(1);
                    error["error"]["message"] =
                        json!("Previous response with id 'resp-1' not found.");
                }
                ws.send(Message::Text(error.to_string().into())).unwrap();
            }
            assert!(ws.read().is_err(), "failed socket must be discarded");
            done_rx.recv_timeout(WAIT).unwrap();
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        cfg.resilience.as_mut().unwrap().max_retries = 3;
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut request = request();
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        append_tool_result(&mut request);
        let mut next = session.begin_turn(request, None).await.unwrap();
        assert!(drain(&mut next).await.is_err());
        assert!(next.next_event(None).await.unwrap().is_none());
        done_tx.send(()).unwrap();
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn continuation_raw_output_is_bounded_and_overflow_uses_full_request() {
    for overflow in [false, true] {
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            receive(&mut ws);
            for line in SUCCESS
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
            {
                let mut event: Value = serde_json::from_str(line).unwrap();
                if event["type"] == "response.output_item.done" && event["output_index"] == 0 {
                    // Real output may contain provider-only fields absent from the
                    // canonical replay representation. They still consume the bound.
                    event["item"]["provider_padding"] =
                        json!("x".repeat(if overflow { 8192 } else { 16 }));
                }
                ws.send(Message::Text(event.to_string().into())).unwrap();
            }
            let wire = if overflow {
                receive(&mut ws)
            } else {
                receive_continuation(&mut ws)
            };
            if overflow {
                assert_eq!(wire["input"].as_array().unwrap().len(), 5);
            } else {
                assert_eq!(wire["previous_response_id"], "resp-1");
                assert_eq!(wire["input"].as_array().unwrap().len(), 1);
            }
            success_id(&mut ws, "resp-2");
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        cfg.limits.max_request_bytes = 4096;
        cfg.limits.max_text_bytes = 4096;
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut request = request();
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        {
            let state = session.websocket.lock().unwrap();
            let connection = state.idle.as_ref().unwrap();
            if overflow {
                assert!(connection.checkpoint.is_none());
            } else {
                let checkpoint = connection.checkpoint.as_ref().unwrap();
                assert_eq!(checkpoint.raw_output.len(), 3);
                assert!(checkpoint.bytes <= 4096);
                let raw: Value = serde_json::from_str(&checkpoint.raw_output[&0]).unwrap();
                assert_eq!(raw["provider_padding"], "x".repeat(16));
            }
        }
        append_tool_result(&mut request);
        let mut next = session.begin_turn(request, None).await.unwrap();
        finished(&drain(&mut next).await.unwrap());
        peer.join().unwrap();
    }
}

fn decorated_success(ws: &mut Socket) {
    let fixture = SUCCESS.replace(r#"{\"q\":1}"#, r#"{ \"q\" : 1 }"#);
    for line in fixture
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        let mut event: Value = serde_json::from_str(line).unwrap();
        if event["type"] == "response.output_item.done" {
            event["item"]["status"] = json!("completed");
            event["item"]["annotations"] = json!([{"type":"provider_only"}]);
        }
        ws.send(Message::Text(event.to_string().into())).unwrap();
    }
}
fn assert_canonical_output(wire: &Value, encrypted: &str, arguments: Value) {
    let input = wire["input"].as_array().unwrap();
    assert_eq!(
        input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .unwrap(),
        &json!({
            "id":"reason-1", "type":"reasoning", "summary":[], "encrypted_content":encrypted
        })
    );
    assert_eq!(
        input
            .iter()
            .find(|item| item["type"] == "function_call")
            .unwrap(),
        &json!({
            "id":"call-item", "type":"function_call", "call_id":"call-1", "name":"lookup",
            "arguments":arguments.to_string()
        })
    );
}

#[tokio::test]
async fn continuation_generated_image_reduces_into_real_transcript() {
    let (endpoint, peer) = server(|listener| {
        let mut ws = socket(&listener);
        receive(&mut ws);
        let item = json!({"id":"image-1", "type":"image_generation_call", "status":"completed",
            "revised_prompt":"safer prompt", "result":"AQID"});
        for event in [
            json!({"type":"response.created", "sequence_number":1, "response":{"id":"resp-image", "model":"gpt-test"}}),
            json!({"type":"response.output_item.added", "sequence_number":2, "output_index":0, "item":{"id":"image-1", "type":"image_generation_call"}}),
            json!({"type":"response.output_item.done", "sequence_number":3, "output_index":0, "item":item}),
            json!({"type":"response.completed", "sequence_number":4, "response":{"id":"resp-image", "model":"gpt-test", "status":"completed", "output":[item]}}),
        ] {
            ws.send(Message::Text(event.to_string().into())).unwrap();
        }
        let wire = receive_continuation(&mut ws);
        assert_eq!(wire["previous_response_id"], "resp-image");
        assert_eq!(
            wire["input"],
            json!([{"type":"message", "role":"user", "content":[{"type":"input_text", "text":"describe image"}]}])
        );
        success_id(&mut ws, "resp-after-image");
    });
    let adapter =
        OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
            .unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut request = request();
    let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
    let events = drain(&mut turn).await.unwrap();
    let result = events
        .into_iter()
        .find_map(|event| match event {
            ModelTurnEvent::Finished(result) => Some(result),
            _ => None,
        })
        .unwrap();
    let Part::Media(media) = &result.output_items[0].parts[0] else {
        panic!("missing generated image")
    };
    assert_eq!(media.data, DataRef::InlineBytes(vec![1, 2, 3]));
    request.transcript.extend(result.output_items);
    request
        .transcript
        .push(Item::text(ItemKind::User, "describe image"));
    let mut next = session.begin_turn(request, None).await.unwrap();
    finished(&drain(&mut next).await.unwrap());
    peer.join().unwrap();
}

#[test]
fn continuation_checkpoint_requires_identical_model_and_reasoning_settings() {
    let previous = json!({"type":"response.create", "model":"gpt-test", "reasoning":{"effort":"low", "summary":"auto"}, "input":[{"role":"user","content":"hello"}]});
    let output = json!({"type":"reasoning","id":"r", "summary":[], "encrypted_content":"opaque"});
    let checkpoint = Checkpoint {
        request: Zeroizing::new(previous.to_string()),
        response_id: "resp-1".into(),
        raw_output: BTreeMap::new(),
        replay_output: BTreeMap::from([(0, Zeroizing::new(json!([output]).to_string()))]),
        bytes: 0,
        completed: true,
    };
    let mut current = previous.clone();
    current["input"]
        .as_array_mut()
        .unwrap()
        .extend([output, json!({"role":"user","content":"next"})]);
    assert_eq!(
        checkpoint.suffix(&current),
        Some(vec![json!({"role":"user","content":"next"})])
    );
    for (key, value) in [
        ("model", json!("other-model")),
        ("reasoning", json!({"effort":"high","summary":"auto"})),
    ] {
        let mut changed = current.clone();
        changed[key] = value;
        assert!(checkpoint.suffix(&changed).is_none(), "changed {key}");
    }
}

#[tokio::test]
async fn continuation_terminal_output_preserves_streamed_items_or_discards_conflicts() {
    for variant in [
        "malformed",
        "missing_item",
        "mismatch",
        "status",
        "matching",
        "empty",
    ] {
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            receive(&mut ws);
            let mut output = vec![];
            for line in SUCCESS
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
            {
                let mut event: Value = serde_json::from_str(line).unwrap();
                if event["type"] == "response.output_item.done" {
                    output.push(event["item"].clone());
                }
                if event["type"] == "response.completed" {
                    event["response"]["status"] = json!("completed");
                    event["response"]["output"] = json!(output);
                    match variant {
                        "malformed" => event["response"]["output"] = json!({"unexpected":"object"}),
                        "missing_item" => {
                            event["response"]["output"].as_array_mut().unwrap().pop();
                        }
                        "mismatch" => {
                            event["response"]["output"][1]["encrypted_content"] = json!("different")
                        }
                        "status" => event["response"]["status"] = json!("in_progress"),
                        "matching" => {}
                        "empty" => event["response"]["output"] = json!([]),
                        _ => unreachable!(),
                    }
                }
                ws.send(Message::Text(event.to_string().into())).unwrap();
            }
            if matches!(variant, "matching" | "empty") {
                let wire = receive_continuation(&mut ws);
                assert_eq!(wire["previous_response_id"], "resp-1");
                assert_eq!(wire["input"].as_array().unwrap().len(), 1);
            } else {
                let full = receive(&mut ws);
                assert_eq!(full["input"].as_array().unwrap().len(), 5);
                assert_canonical_output(&full, "opaque", json!({"q":1}));
            }
            success_id(&mut ws, "resp-2");
        });
        let adapter =
            OpenAIResponsesAdapter::new(config(&endpoint, OpenAIResponsesTransport::WebSocket))
                .unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut request = request();
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        append_tool_result(&mut request);
        let mut next = session.begin_turn(request, None).await.unwrap();
        finished(&drain(&mut next).await.unwrap());
        peer.join().unwrap();
    }
}

// Change every arguments-bearing frame, not just output_item.done: the real
// decoder observes identical values with noncanonical whitespace and key order.
fn normalized_success(ws: &mut Socket) {
    let fixture = SUCCESS.replace(r#"{\"q\":1}"#, r#"{ \"z\" : 2, \"q\" : 1 }"#);
    assert_ne!(fixture, SUCCESS);
    for line in fixture
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        let mut event: Value = serde_json::from_str(line).unwrap();
        if event["type"] == "response.output_item.done" {
            event["item"]["status"] = json!("completed");
            if event["item"]["type"] == "message" {
                event["item"]["content"][0]["annotations"] = json!([]);
            }
        }
        ws.send(Message::Text(event.to_string().into())).unwrap();
    }
}

struct RotatingInitialAuth {
    calls: std::sync::atomic::AtomicUsize,
    change_binding: bool,
}
#[async_trait]
impl AuthenticationProvider for RotatingInitialAuth {
    async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        assert!(
            previous.is_none(),
            "this test must not refresh a rejected attempt"
        );
        let generation = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer loopback-{generation}")).unwrap(),
        );
        Ok(
            AuthenticationAttempt::stateless(headers).with_binding(if self.change_binding {
                format!("loopback-identity-{generation}")
            } else {
                "loopback-identity".to_owned()
            }),
        )
    }
}

#[tokio::test]
async fn continuation_reconnect_or_rotated_headers_sends_full_reduced_history() {
    for rotate_headers in [false, true] {
        let (close_tx, close_rx) = tokio::sync::oneshot::channel();
        let (endpoint, peer) = server(move |listener| {
            let mut first = socket(&listener);
            receive(&mut first);
            success(&mut first);
            if !rotate_headers {
                first.close(None).unwrap();
                close_tx.send(()).unwrap();
                assert!(matches!(first.read(), Ok(Message::Close(_))));
                drop(first);
            }
            let mut fresh = socket(&listener);
            let full = receive(&mut fresh);
            assert_eq!(full["input"].as_array().unwrap().len(), 5);
            assert_canonical_output(&full, "opaque", json!({"q":1}));
            assert_eq!(full["input"][0]["role"], "user");
            assert_eq!(
                full["input"][4],
                json!({"type":"function_call_output", "call_id":"call-1", "output":"found"})
            );
            success_id(&mut fresh, "resp-fresh");
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        if rotate_headers {
            cfg = cfg.with_authentication_provider(RotatingInitialAuth {
                calls: std::sync::atomic::AtomicUsize::new(0),
                change_binding: false,
            });
        }
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut request = request();
        let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
        append_completed(&mut turn, &mut request).await;
        append_tool_result(&mut request);
        if !rotate_headers {
            tokio::time::timeout(WAIT, close_rx).await.unwrap().unwrap();
            // Synchronize receipt of the peer's Close, not a sleep or a race
            // against send's nonblocking boundary check. Leave the completed
            // checkpoint on the closed connection for normal checkout to reject.
            let mut connection = session.websocket.lock().unwrap().idle.take().unwrap();
            let frame = tokio::time::timeout(WAIT, connection.stream.next())
                .await
                .unwrap();
            assert!(matches!(frame, Some(Ok(Message::Close(_)))));
            connection.stream.flush().await.unwrap();
            assert!(
                tokio::time::timeout(WAIT, connection.stream.next())
                    .await
                    .unwrap()
                    .is_none()
            );
            assert!(connection.checkpoint.as_ref().unwrap().completed);
            session.websocket.lock().unwrap().idle = Some(connection);
        }
        let mut next = session.begin_turn(request, None).await.unwrap();
        finished(&drain(&mut next).await.unwrap());
        peer.join().unwrap();
    }
}

#[tokio::test]
async fn continuation_changed_binding_reconnects_with_full_text_history() {
    let (endpoint, peer) = server(|listener| {
        let mut first = socket(&listener);
        receive(&mut first);
        // Text-only output has no authentication-bound protected metadata.
        for line in SUCCESS
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .take(7)
        {
            first.send(Message::Text(line.into())).unwrap();
        }
        first
            .send(Message::Text(
                json!({"type":"response.completed", "sequence_number":8,
            "response":{"id":"resp-1", "model":"gpt-test", "status":"completed"}})
                .to_string()
                .into(),
            ))
            .unwrap();
        let mut fresh = socket(&listener);
        let full = receive(&mut fresh);
        assert_eq!(
            full["input"],
            json!([
                {"type":"message", "role":"user", "content":[{"type":"input_text", "text":"hello"}]},
                {"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"hello"}]},
                {"type":"message", "role":"user", "content":[{"type":"input_text", "text":"next"}]}
            ])
        );
        success_id(&mut fresh, "resp-2");
    });
    let cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket).with_authentication_provider(
        RotatingInitialAuth {
            calls: std::sync::atomic::AtomicUsize::new(0),
            change_binding: true,
        },
    );
    let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let mut request = request();
    let mut turn = session.begin_turn(request.clone(), None).await.unwrap();
    let events = drain(&mut turn).await.unwrap();
    finished(&events);
    for event in events {
        if let ModelTurnEvent::Finished(result) = event {
            request.transcript.extend(result.output_items);
        }
    }
    request.transcript.push(Item::text(ItemKind::User, "next"));
    let mut next = session.begin_turn(request, None).await.unwrap();
    finished(&drain(&mut next).await.unwrap());
    peer.join().unwrap();
}

#[tokio::test]
async fn interrupted_missing_previous_backoff_releases_session_without_replay() {
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Poll;

    for cancel in [false, true] {
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let (checked_tx, checked_rx) = tokio::sync::oneshot::channel();
        let (endpoint, peer) = server(move |listener| {
            let mut ws = socket(&listener);
            receive(&mut ws);
            success(&mut ws);
            let incremental = receive_continuation(&mut ws);
            assert_eq!(incremental["previous_response_id"], "resp-1");
            assert_eq!(incremental["input"].as_array().unwrap().len(), 1);
            missing_previous(&mut ws);
            assert!(ws.read().is_err(), "rejected socket must be discarded");
            // The client signals only after interrupting the suspended backoff.
            // No recovery connection may have been opened in the meantime.
            resume_rx.recv_timeout(WAIT).unwrap();
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
            checked_tx.send(()).unwrap();
            let mut fresh = socket(&listener);
            let full = receive(&mut fresh);
            let input = full["input"].as_array().unwrap();
            assert_eq!(input.len(), 6);
            assert_eq!(input[0]["content"][0]["text"], "hello");
            assert_eq!(input[2]["type"], "reasoning");
            assert_eq!(input[2]["encrypted_content"], "opaque");
            assert_eq!(input[4], incremental["input"][0]);
            assert_eq!(input[5]["content"][0]["text"], "after interrupted recovery");
            success_id(&mut fresh, "fresh-after-interruption");
        });
        let mut cfg = config(&endpoint, OpenAIResponsesTransport::WebSocket);
        let resilience = cfg.resilience.as_mut().unwrap();
        resilience.max_retries = 1;
        // Never wait for this duration: an observed Scheduled event followed by
        // Poll::Pending proves suspension, without sleep-based timing assertions.
        resilience.initial_backoff = Duration::from_secs(3600);
        resilience.max_backoff = Duration::from_secs(3600);
        resilience.retry_budget = Duration::from_secs(7200);
        let adapter = OpenAIResponsesAdapter::new(cfg).unwrap();
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let mut history = request();
        let mut first = session.begin_turn(history.clone(), None).await.unwrap();
        append_completed(&mut first, &mut history).await;
        append_tool_result(&mut history);

        let scheduled = Arc::new(AtomicBool::new(false));
        let observed = scheduled.clone();
        session.set_retry_observer(Some(Arc::new(move |event: ProviderRetryEvent| {
            if let ProviderRetryEvent::Scheduled(progress) = event {
                assert_eq!(progress.accounting.attempts, 1);
                assert_eq!(progress.accounting.completed_backoff, Duration::ZERO);
                assert!(!progress.next_delay.is_zero());
                assert!(!observed.swap(true, Ordering::SeqCst));
            }
        })));
        let controller = agentkit_core::CancellationController::new();
        let cancellation = controller.handle().checkpoint();
        let mut interrupted = session.begin_turn(history.clone(), None).await.unwrap();
        let mut pending = Box::pin(interrupted.next_event(Some(cancellation.clone())));
        tokio::time::timeout(
            WAIT,
            futures_util::future::poll_fn(|cx| {
                assert!(pending.as_mut().poll(cx).is_pending());
                if scheduled.load(Ordering::SeqCst) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }),
        )
        .await
        .unwrap();
        if cancel {
            controller.interrupt();
            assert!(matches!(pending.await, Err(LoopError::Cancelled)));
            // Cancellation finalizes the lease even while the owned turn lives.
            assert!(interrupted.next_event(None).await.unwrap().is_none());
        } else {
            drop(pending);
            drop(interrupted);
        }
        resume_tx.send(()).unwrap();
        // Do not open the legitimate next socket until the server has completed
        // its no-replay assertion; channel ordering, not elapsed time, controls it.
        tokio::time::timeout(WAIT, checked_rx)
            .await
            .unwrap()
            .unwrap();
        history
            .transcript
            .push(Item::text(ItemKind::User, "after interrupted recovery"));
        let mut fresh = session.begin_turn(history, None).await.unwrap();
        let events = drain(&mut fresh).await.unwrap();
        finished(&events);
        assert!(events.iter().any(|event| matches!(event,
            ModelTurnEvent::Finished(result)
                if result.response_id.as_deref() == Some("fresh-after-interruption"))));
        peer.join().unwrap();
    }
}
