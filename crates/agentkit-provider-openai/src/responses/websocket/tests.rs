//! Loopback-only transport tests; no live inference.
use super::*;
use agentkit_core::{SessionId, TurnId};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use tokio_tungstenite::tungstenite::{self, WebSocket};

// Same text/tool/reasoning/usage fixture as the HTTP decoder tests.
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
    let message = ws.read().unwrap();
    let value: Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
    assert_eq!(value["type"], "response.create");
    assert!(value.get("previous_response_id").is_none());
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
