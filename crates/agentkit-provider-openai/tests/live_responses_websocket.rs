//! Opt-in, billable network tests. See docs/live-responses-websocket.md.
//! Never reads credential files or logs headers, payloads, IDs, or provider errors.
use agentkit_core::{
    Item, ItemKind, MetadataMap, Part, SessionId, ToolOutput, ToolResultPart, TurnId,
};
use agentkit_loop::{
    ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, ModelTurnResult, SessionConfig,
    TurnRequest,
};
use agentkit_provider_openai::{
    OpenAIResponsesAdapter, OpenAIResponsesConfig, OpenAIResponsesTransport,
};
use futures_util::{SinkExt, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::{
    env,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        Message, handshake::client::generate_key, handshake::derive_accept_key, protocol::Role,
    },
};

// Never format external errors: they can contain credentials or echoed payloads.
fn checked<T, E>(result: Result<T, E>, label: &'static str) -> T {
    result.unwrap_or_else(|_| panic!("{label} (details suppressed)"))
}
struct Credentials {
    headers: HeaderMap,
    private: bool,
    model: String,
}
impl Credentials {
    fn load() -> Option<Self> {
        let token = env::var("AGENTKIT_LIVE_CHATGPT_BEARER")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let account = env::var("AGENTKIT_LIVE_CHATGPT_ACCOUNT_ID")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let key = env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let (secret, account, private) = match (token, account, key) {
            (Some(token), Some(account), _) => (token, Some(account), true),
            (None, None, Some(key)) => (key, None, false),
            (None, None, None) => {
                assert!(
                    env::var("AGENTKIT_LIVE_REQUIRED").as_deref() != Ok("1"),
                    "live credentials required but absent"
                );
                eprintln!(
                    "SKIP: set AGENTKIT_LIVE_CHATGPT_BEARER + AGENTKIT_LIVE_CHATGPT_ACCOUNT_ID, or OPENAI_API_KEY; AGENTKIT_LIVE_REQUIRED=1 makes absence fail"
                );
                return None;
            }
            _ => panic!("incomplete ChatGPT credential environment"),
        };
        let mut headers = HeaderMap::new();
        let mut bearer = checked(
            HeaderValue::from_str(&format!("Bearer {secret}")),
            "invalid bearer header",
        );
        bearer.set_sensitive(true);
        headers.insert("authorization", bearer);
        if let Some(account) = account {
            let mut value = checked(HeaderValue::from_str(&account), "invalid account header");
            value.set_sensitive(true);
            headers.insert("chatgpt-account-id", value);
        }
        let model = env::var("AGENTKIT_LIVE_MODEL")
            .unwrap_or_else(|_| if private { "gpt-5.4" } else { "gpt-4.1-mini" }.into());
        Some(Self {
            headers,
            private,
            model,
        })
    }
}
#[derive(Clone, Debug)]
struct Shape {
    connection: usize,
    previous: bool,
    previous_matches_completed: bool,
    inputs: usize,
    user_messages: usize,
    tool_outputs: usize,
    completed_replay_inputs: Option<usize>,
}
struct Proxy {
    endpoint: String,
    shapes: Arc<Mutex<Vec<Shape>>>,
    task: JoinHandle<()>,
}
impl Drop for Proxy {
    fn drop(&mut self) {
        self.task.abort();
        if std::thread::panicking() {
            // Diagnostics only: a proof assertion can poison this read-only capture
            // lock. Recover its safe counts without recovering production state.
            let shapes = self
                .shapes
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            eprintln!("sanitized request shapes: {shapes:?}");
        }
    }
}
impl Proxy {
    async fn start(credentials: Credentials) -> Self {
        let listener = checked(TcpListener::bind("127.0.0.1:0").await, "bind proxy");
        let endpoint = format!(
            "http://{}/responses",
            checked(listener.local_addr(), "proxy address")
        );
        let shapes = Arc::new(Mutex::new(Vec::new()));
        let captured = shapes.clone();
        let budget = if env::var("AGENTKIT_LIVE_RECONNECT").as_deref() == Ok("1") {
            4
        } else {
            3
        };
        let task = tokio::spawn(async move {
            // No detached children: dropping Proxy cancels all networking.
            for connection in 0..2 {
                let (socket, _) = checked(listener.accept().await, "accept local websocket");
                let mut forwarded = HeaderMap::new();
                // Tungstenite fixes the callback error type; this closure never rejects.
                #[allow(clippy::result_large_err)]
                let mut local = checked(
                    tokio_tungstenite::accept_hdr_async(
                        socket,
                        |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
                         response| {
                            for name in [
                                "originator",
                                "user-agent",
                                "session_id",
                                "x-codex-turn-state",
                            ] {
                                if let Some(value) = request.headers().get(name) {
                                    forwarded.insert(name, value.clone());
                                }
                            }
                            Ok(response)
                        },
                    )
                    .await,
                    "local websocket handshake",
                );
                let key = generate_key();
                let mut headers = credentials.headers.clone();
                headers.extend(forwarded);
                headers.insert("connection", HeaderValue::from_static("Upgrade"));
                headers.insert("upgrade", HeaderValue::from_static("websocket"));
                headers.insert("sec-websocket-version", HeaderValue::from_static("13"));
                headers.insert(
                    "sec-websocket-key",
                    checked(HeaderValue::from_str(&key), "websocket key"),
                );
                headers.insert(
                    "openai-beta",
                    HeaderValue::from_static("responses_websockets=2026-02-06"),
                );
                let client = checked(
                    reqwest::Client::builder()
                        .http1_only()
                        .no_proxy()
                        .redirect(reqwest::redirect::Policy::none())
                        .retry(reqwest::retry::never())
                        .connect_timeout(Duration::from_secs(20))
                        .timeout(Duration::from_secs(30))
                        .build(),
                    "upstream client",
                );
                let endpoint = if credentials.private {
                    "https://chatgpt.com/backend-api/codex/responses"
                } else {
                    "https://api.openai.com/v1/responses"
                };
                let response = checked(
                    client.get(endpoint).headers(headers).send().await,
                    "upstream handshake",
                );
                assert!(
                    response.status() == reqwest::StatusCode::SWITCHING_PROTOCOLS,
                    "upstream did not accept websocket (status/body suppressed; check credentials and access)"
                );
                assert!(
                    response
                        .headers()
                        .get("sec-websocket-accept")
                        .and_then(|v| v.to_str().ok())
                        == Some(derive_accept_key(key.as_bytes()).as_str()),
                    "invalid upstream websocket accept"
                );
                let upgraded = checked(response.upgrade().await, "upstream upgrade");
                let mut upstream =
                    WebSocketStream::from_raw_socket(upgraded, Role::Client, None).await;
                let mut completed_id: Option<String> = None;
                let mut streamed_replay_inputs = 0;
                loop {
                    tokio::select! {
                        message = local.next() => {
                            let Some(Ok(message)) = message else { break };
                            if message.is_close() { break; }
                            if let Message::Text(text) = &message {
                                let value: Value = checked(serde_json::from_str(text), "outbound JSON");
                                assert!(value["type"] == "response.create", "unexpected outbound message type");
                                let input = value["input"].as_array().unwrap_or_else(|| panic!("input array missing"));
                                let previous = value["previous_response_id"].as_str();
                                assert!(checked(captured.lock(), "capture lock").len() < budget, "live request budget exceeded; refusing to forward");
                                checked(captured.lock(), "capture lock").push(Shape {
                                    connection, previous: previous.is_some(), completed_replay_inputs: None,
                                    previous_matches_completed: previous.is_some() && previous == completed_id.as_deref(),
                                    inputs: input.len(),
                                    user_messages: input.iter().filter(|v| v["role"] == "user").count(),
                                    tool_outputs: input.iter().filter(|v| v["type"] == "function_call_output").count(),
                                });
                            }
                            checked(upstream.send(message).await, "forward request");
                        }
                        message = upstream.next() => {
                            let Some(Ok(message)) = message else { break };
                            if let Message::Text(text) = &message {
                                let value: Value = checked(serde_json::from_str(text), "inbound JSON");
                                if value["type"] == "error" || value["type"] == "response.failed" {
                                    let error = value.get("error").or_else(|| value.pointer("/response/error"));
                                    let message = error.and_then(|e| e["message"].as_str()).unwrap_or("").to_ascii_lowercase();
                                    eprintln!("sanitized upstream error: instructions={} model={} unsupported={} invalid_request={} auth={}",
                                        message.contains("instructions"), message.contains("model"), message.contains("unsupported"),
                                        error.is_some_and(|e| e["type"] == "invalid_request_error"),
                                        message.contains("auth") || message.contains("token"));
                                }
                                if value["type"] == "response.output_item.done" {
                                    streamed_replay_inputs += completed_replay_inputs(&json!({"response":{"output":[value["item"].clone()]}}));
                                }
                                if value["type"] == "response.completed" {
                                    completed_id = value.pointer("/response/id").and_then(Value::as_str).map(str::to_owned);
                                    // Some real endpoints send an empty terminal output array:
                                    // item.done remains the authoritative streamed output.
                                    let count = if streamed_replay_inputs > 0 { streamed_replay_inputs } else { completed_replay_inputs(&value) };
                                    streamed_replay_inputs = 0;
                                    let mut shapes = checked(captured.lock(), "capture lock");
                                    let shape = shapes.last_mut().unwrap_or_else(|| panic!("completion without request"));
                                    assert!(shape.completed_replay_inputs.is_none(), "duplicate completion");
                                    shape.completed_replay_inputs = Some(count);
                                }
                            }
                            if local.send(message).await.is_err() { break; }
                        }
                    }
                }
            }
        });
        Self {
            endpoint,
            shapes,
            task,
        }
    }
}

// Count replayable wire outputs, independently of AgentKit's credential-bound
// metadata encoding. Restrict this tiny fixture to text/tool/reasoning outputs.
// Summary-only reasoning has no replayable encrypted state and is omitted.
fn completed_replay_inputs(event: &Value) -> usize {
    event
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("completed response missing output array"))
        .iter()
        .map(|item| match item["type"].as_str() {
            Some("message") => usize::from(
                item["content"]
                    .as_array()
                    .is_some_and(|parts| !parts.is_empty()),
            ),
            Some("function_call") => 1,
            Some("reasoning") => usize::from(
                item["encrypted_content"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty()),
            ),
            _ => panic!("unexpected output type for live fixture"),
        })
        .sum()
}

fn request(transcript: &[Item], turn: &str) -> TurnRequest {
    TurnRequest {
        session_id: SessionId::new("live-ws"),
        turn_id: TurnId::new(turn),
        transcript: transcript.to_vec(),
        available_tools: vec![checked(
            serde_json::from_value(
                json!({"name":"read_probe", "description":"Read the probe value. Call exactly once when asked to read it.", "metadata":{},"annotations":{"read_only_hint":true,"destructive_hint":false,"idempotent_hint":true,"needs_approval_hint":false,"supports_streaming_hint":false},"input_schema":{"type":"object","properties":{},"required":[],"additionalProperties":false}}),
            ),
            "probe tool specification",
        )],
        cache: None,
        metadata: MetadataMap::new(),
    }
}
async fn run_turn(
    session: &mut impl ModelSession,
    transcript: &mut Vec<Item>,
    turn: &str,
) -> ModelTurnResult {
    let future = async {
        let mut stream = checked(
            session.begin_turn(request(transcript, turn), None).await,
            "begin model turn",
        );
        loop {
            match checked(stream.next_event(None).await, "read model event") {
                Some(ModelTurnEvent::Finished(result)) => {
                    // Same authoritative reduction as LoopDriver: preserve all output items and
                    // opaque continuation metadata, not a hand-built assistant/tool-call replay.
                    transcript.extend(result.output_items.clone());
                    return result;
                }
                Some(_) => {}
                None => panic!("model stream ended before Finished"),
            }
        }
    };
    checked(
        tokio::time::timeout(Duration::from_secs(90), future).await,
        "model turn timed out",
    )
}
fn answer_contains(result: &ModelTurnResult, expected: &str) -> bool {
    result
        .output_items
        .iter()
        .flat_map(|item| &item.parts)
        .any(|part| matches!(part, Part::Text(text) if text.text.contains(expected)))
}
#[tokio::test]
#[ignore = "billable live WebSocket test; explicit credentials required; see docs/live-responses-websocket.md"]
async fn incremental_tool_and_user_roundtrip() {
    let Some(credentials) = Credentials::load() else {
        return;
    };
    let config = if credentials.private {
        OpenAIResponsesConfig::chatgpt_private(credentials.model.clone(), "local-proxy-placeholder")
            .with_originator("codex_cli_rs")
            .with_user_agent("agentkit-live-websocket-test")
    } else {
        OpenAIResponsesConfig::public(credentials.model.clone(), "local-proxy-placeholder")
            .with_max_output_tokens(512)
    };
    let proxy = Proxy::start(credentials).await;
    let config = config
        .with_endpoint(&proxy.endpoint)
        .with_transport(OpenAIResponsesTransport::WebSocket)
        .with_parallel_tool_calls(false);
    let adapter = checked(OpenAIResponsesAdapter::new(config), "create adapter");
    let mut session = checked(
        adapter.start_session(SessionConfig::new("live-ws")).await,
        "start session",
    );
    let mut transcript = vec![
        Item::text(
            ItemKind::System,
            "Follow these instructions exactly. When asked to read the probe, call read_probe exactly once with no arguments. After the tool result, answer only with its value. For later user messages reply only with the literal text requested; do not call tools again.",
        ),
        Item::text(ItemKind::User, "Read the probe now."),
    ];
    let first = run_turn(&mut session, &mut transcript, "tool").await;
    let calls: Vec<_> = first
        .output_items
        .iter()
        .flat_map(|i| &i.parts)
        .filter_map(|p| {
            if let Part::ToolCall(call) = p {
                Some(call)
            } else {
                None
            }
        })
        .collect();
    assert!(
        calls.len() == 1 && calls[0].name.as_str() == "read_probe",
        "expected exactly the probe tool call"
    );
    transcript.push(Item::new(
        ItemKind::Tool,
        vec![Part::ToolResult(ToolResultPart::success(
            calls[0].id.clone(),
            ToolOutput::text("PROBE_731"),
        ))],
    ));
    let second = run_turn(&mut session, &mut transcript, "tool-output").await;
    assert!(
        answer_contains(&second, "PROBE_731"),
        "tool output was not reflected in answer"
    );
    transcript.push(Item::text(ItemKind::User, "Reply only with USER_927."));
    let third = run_turn(&mut session, &mut transcript, "next-user").await;
    assert!(
        answer_contains(&third, "USER_927"),
        "subsequent user turn not answered"
    );
    {
        let shapes = checked(proxy.shapes.lock(), "capture lock");
        assert_eq!(shapes.len(), 3, "unexpected retries or extra requests");
        assert!(
            !shapes[0].previous && shapes[0].user_messages == 1,
            "first request must be full input"
        );
        for shape in &shapes[1..] {
            assert!(
                shape.connection == 0 && shape.previous && shape.previous_matches_completed,
                "continuation must reference the completed response on the same connection"
            );
            assert_eq!(
                shape.inputs, 1,
                "continuation must send only one new input, never replay"
            );
        }
        assert_eq!(shapes[1].tool_outputs, 1);
        assert_eq!(shapes[1].user_messages, 0);
        assert_eq!(shapes[2].tool_outputs, 0);
        assert_eq!(shapes[2].user_messages, 1);
        eprintln!(
            "PASS: three accepted requests; continuation inputs: tool_output=1, user=1; prior IDs matched completed responses"
        );
    }
    // Optional fourth billable request. A new public session forces a new socket;
    // replay real transcript items without synthesizing provider metadata.
    if env::var("AGENTKIT_LIVE_RECONNECT").as_deref() == Ok("1") {
        drop(session);
        transcript.push(Item::text(ItemKind::User, "Reply only with RECONNECT_419."));
        // Public encode_request deliberately cannot validate credential-bound output
        // metadata. Use independent, sanitized wire accounting instead; never strip
        // metadata or manufacture a continuation to make the public encoder work.
        let expected_inputs = {
            let shapes = checked(proxy.shapes.lock(), "capture lock");
            shapes[0].inputs
                + shapes
                    .iter()
                    .map(|shape| {
                        shape
                            .completed_replay_inputs
                            .unwrap_or_else(|| panic!("missing completed output accounting"))
                    })
                    .sum::<usize>()
                + 3 // local tool output, second user turn, reconnect user turn
        };
        let mut session = checked(
            adapter.start_session(SessionConfig::new("live-ws")).await,
            "restart session",
        );
        let fourth = run_turn(&mut session, &mut transcript, "reconnect").await;
        assert!(
            answer_contains(&fourth, "RECONNECT_419"),
            "reconnect response missing"
        );
        let shapes = checked(proxy.shapes.lock(), "capture lock");
        assert_eq!(shapes.len(), 4);
        assert!(
            shapes[3].connection == 1 && !shapes[3].previous,
            "new socket must not use previous_response_id"
        );
        assert_eq!(
            shapes[3].inputs, expected_inputs,
            "new socket must replay full input"
        );
        assert!(
            shapes[3].inputs > 1 && shapes[3].tool_outputs == 1 && shapes[3].user_messages == 3
        );
        eprintln!("PASS: new session/socket accepted full transcript without previous_response_id");
    }
}

#[test]
fn fixture_encodes_without_credentials_or_network() {
    let config = OpenAIResponsesConfig::chatgpt_private("unused", "unused");
    let value = checked(
        config.encode_request(&request(&[Item::text(ItemKind::User, "probe")], "fixture")),
        "fixture encoding",
    );
    assert_eq!(value["tools"][0]["name"], "read_probe");
}

#[test]
fn completed_wire_accounting_omits_summary_only_reasoning() {
    let event = json!({"response":{"output":[
        {"type":"function_call"},
        {"type":"message","content":[{"type":"output_text","text":"fixture"}]},
        {"type":"reasoning","summary":[]},
        {"type":"reasoning","encrypted_content":"fixture-only"}
    ]}});
    assert_eq!(completed_replay_inputs(&event), 3);
}
