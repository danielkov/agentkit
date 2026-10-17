//! Built-in Auto lifecycle through an actual HTTP JSON-RPC and POST SSE boundary.
use std::{
    convert::Infallible,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use agentkit_mcp::{
    AuthOperation, AuthRequest, AuthResolution, McpConnection, McpServerConfig, McpServerEvent,
    McpServerId, McpServerManager, McpSubscriptionStatus, McpTransportBinding,
    StreamableHttpTransportConfig,
};
use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use rmcp::model::{DiscoverResult, ProtocolVersion, ServerCapabilities};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("HTTP MCP operation timed out")
}

struct Listen {
    id: Value,
    filter: Value,
    headers: HeaderMap,
    frames: mpsc::UnboundedSender<Bytes>,
}
impl Listen {
    fn acknowledge(&self, filter: Value) {
        self.notify(
            "notifications/subscriptions/acknowledged",
            json!({"notifications":filter}),
        );
    }

    fn fail(&self) {
        self.frames
            .send(frame(json!({"jsonrpc":"2.0", "id":self.id,
            "error":{"code":-32603,"message":"listen failed; ordinary RPC remains healthy"}})))
            .unwrap();
    }

    fn notify(&self, method: &str, params: Value) {
        let mut params = params;
        params["_meta"] = json!({"io.modelcontextprotocol/subscriptionId":self.id});
        self.frames
            .send(frame(
                json!({"jsonrpc":"2.0", "method":method, "params":params}),
            ))
            .unwrap();
    }
}
fn frame(message: Value) -> Bytes {
    Bytes::from(format!("event: message\ndata: {message}\n\n"))
}

#[derive(Clone)]
struct Endpoint {
    hold_ack: Arc<AtomicBool>,
    tool_version: Arc<AtomicU64>,
    listens: mpsc::UnboundedSender<Listen>,
    requests: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
}

async fn rpc(
    State(state): State<Endpoint>,
    headers: HeaderMap,
    Json(message): Json<Value>,
) -> Response {
    state
        .requests
        .lock()
        .await
        .push((headers.clone(), message.clone()));
    let Some(id) = message.get("id") else {
        return StatusCode::ACCEPTED.into_response();
    };
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    // Validate the version before dispatch, as legacy endpoints can do even
    // for methods they do not recognize.
    let supported = if token == "Bearer no-common" {
        vec!["2099-01-01"]
    } else if let Some(version) = token.strip_prefix("Bearer legacy-negotiate-") {
        vec![version]
    } else if token == "Bearer legacy-negotiate" {
        // Deliberately ascending: the client's preference, not server order,
        // must choose the newest mutual version.
        vec!["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
    } else {
        vec![]
    };
    if !supported.is_empty()
        && message["method"] == "server/discover"
        && !headers
            .get("mcp-protocol-version")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|version| supported.contains(&version))
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "jsonrpc":"2.0", "id":id, "error":{
                    "code":-32022, "message":"Unsupported protocol version",
                    "data":{"supported":supported}
                }
            })),
        )
            .into_response();
    }
    let result = match message["method"].as_str().unwrap_or_default() {
        "server/discover" => {
            if token.starts_with("Bearer legacy") {
                return Json(json!({"jsonrpc":"2.0", "id":id, "error":{
                    "code":-32601, "message":"legacy server has no discovery"
                }}))
                .into_response();
            }
            if token == "Bearer fail" {
                return Json(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32603,"message":"test discovery failure"}})).into_response();
            }
            let capabilities = ServerCapabilities::builder()
                .enable_tools()
                .enable_tool_list_changed()
                .enable_resources()
                .enable_resources_list_changed()
                .enable_resources_subscribe();
            let capabilities = if token == "Bearer replacement" {
                capabilities
                    .enable_prompts()
                    .enable_prompts_list_changed()
                    .build()
            } else {
                capabilities.build()
            };
            serde_json::to_value(DiscoverResult::new(
                vec![ProtocolVersion::V_2026_07_28],
                capabilities,
            ))
            .unwrap()
        }
        "subscriptions/listen" => {
            let (frames, receiver) = mpsc::unbounded_channel();
            let filter = message["params"]["notifications"].clone();
            if token != "Bearer hold" && !state.hold_ack.load(Ordering::SeqCst) {
                frames.send(frame(json!({"jsonrpc":"2.0", "method":"notifications/subscriptions/acknowledged",
                    "params":{"notifications":filter,"_meta":{"io.modelcontextprotocol/subscriptionId":id}}}))).unwrap();
            }
            state
                .listens
                .send(Listen {
                    id: id.clone(),
                    filter,
                    headers,
                    frames,
                })
                .unwrap_or_else(|_| panic!("fixture receiver alive"));
            let stream = futures_util::stream::unfold(receiver, |mut receiver| async move {
                receiver
                    .recv()
                    .await
                    .map(|bytes| (Ok::<_, Infallible>(bytes), receiver))
            });
            return (
                [("content-type", "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response();
        }
        "initialize" => {
            if token == "Bearer fail" {
                return Json(json!({"jsonrpc":"2.0", "id":id, "error":{
                    "code":-32603, "message":"test initialization failure"
                }}))
                .into_response();
            }
            json!({
                "protocolVersion":"2024-11-05",
                "capabilities":{"tools":{}, "resources":{"subscribe":true}},
                "serverInfo":{"name":"legacy-http-test", "version":"1"}
            })
        }
        "resources/subscribe" => {
            if token == "Bearer legacy-fail" {
                return Json(json!({"jsonrpc":"2.0", "id":id, "error":{
                    "code":-32603, "message":"test subscription replay failure"
                }}))
                .into_response();
            }
            json!({})
        }
        "resources/unsubscribe" => json!({}),
        "tools/list" => {
            json!({"tools":[{"name":format!("tool{}", state.tool_version.load(Ordering::SeqCst)),"inputSchema":{"type":"object"}}]})
        }
        "resources/list" => json!({"resources":[]}),
        "prompts/list" => json!({"prompts":[]}),
        _ => return Json(
            json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"unexpected method"}}),
        )
        .into_response(),
    };
    Json(json!({"jsonrpc":"2.0", "id":id, "result":result})).into_response()
}

struct Fixture {
    hold_ack: Arc<AtomicBool>,
    tool_version: Arc<AtomicU64>,
    config: McpServerConfig,
    listens: mpsc::UnboundedReceiver<Listen>,
    requests: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    async fn start() -> Self {
        let (tx, listens) = mpsc::unbounded_channel();
        let hold_ack = Arc::new(AtomicBool::new(false));
        let tool_version = Arc::new(AtomicU64::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let app = Router::new().route("/mcp", post(rpc)).with_state(Endpoint {
            hold_ack: hold_ack.clone(),
            tool_version: tool_version.clone(),
            listens: tx,
            requests: requests.clone(),
        });
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let config = McpServerConfig::new(
            "http-modern",
            McpTransportBinding::StreamableHttp(
                StreamableHttpTransportConfig::new(url).with_bearer_token("initial"),
            ),
        );
        Self {
            hold_ack,
            tool_version,
            config,
            listens,
            requests,
            task,
        }
    }
    async fn listen(&mut self) -> Listen {
        bounded(self.listens.recv()).await.unwrap()
    }
}

fn auth(token: &str) -> AuthResolution {
    AuthResolution::Provided {
        request: AuthRequest {
            id: "test".into(),
            provider: "test".into(),
            operation: AuthOperation::McpConnect {
                server_id: "http-modern".into(),
                metadata: Default::default(),
            },
            challenge: Default::default(),
        },
        credentials: [("bearer_token".into(), json!(token))]
            .into_iter()
            .collect(),
    }
}

#[tokio::test]
async fn auto_http_discovery_stream_metadata_and_auth_replacement_restore_resources() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let connection = McpConnection::connect(&fixture.config).await.unwrap();
        let initial = fixture.listen().await;
        assert_eq!(connection.protocol_version(), Some(ProtocolVersion::V_2026_07_28));
        assert!(matches!(connection.subscription_status().await, Some(McpSubscriptionStatus::Active { .. })));
        assert!(connection.capabilities().prompts.is_none());
        connection.discover().await.unwrap();
        let requests = fixture.requests.lock().await;
        assert_eq!(requests[0].1["method"], "server/discover");
        assert!(requests.iter().all(|(_, message)| message["method"] != "initialize"));
        let (headers, request) = requests.iter().find(|(_, message)| message["method"] == "subscriptions/listen").unwrap();
        assert_eq!(headers.get("mcp-protocol-version").unwrap(), "2026-07-28");
        assert_eq!(headers.get("mcp-method").unwrap(), "subscriptions/listen");
        let meta = &request["params"]["_meta"];
        assert_eq!(meta["io.modelcontextprotocol/protocolVersion"], "2026-07-28");
        assert!(meta["io.modelcontextprotocol/clientInfo"].is_object());
        assert!(meta["io.modelcontextprotocol/clientCapabilities"].is_object());
        drop(requests);
        connection.subscribe_resource("memo:keep").await.unwrap();
        let subscribed = fixture.listen().await;
        bounded(initial.frames.closed()).await;
        let epoch = connection.connection_epoch();
        let revision = connection.catalog_revision();
        connection.resolve_auth(auth("replacement")).await.unwrap();
        let replacement = fixture.listen().await;
        assert_eq!(replacement.headers.get("authorization").unwrap(), "Bearer replacement");
        assert_eq!(replacement.filter["resourceSubscriptions"], json!(["memo:keep"]));
        assert_eq!(replacement.filter["promptsListChanged"], true);
        assert!(connection.capabilities().prompts.is_some());
        assert!(connection.connection_epoch() > epoch);
        assert!(connection.catalog_revision() > revision);
        bounded(subscribed.frames.closed()).await;
        let mut events = connection.subscribe_events();
        replacement.notify("notifications/resources/updated", json!({"uri":"memo:keep"}));
        assert!(matches!(events.recv().await.unwrap(), McpServerEvent::ResourceUpdated(params) if params.uri == "memo:keep"));
        connection.close().await.unwrap();
        bounded(replacement.frames.closed()).await;
    }).await;
}

#[tokio::test]
async fn failed_and_cancelled_auth_reconnect_leave_old_http_stream_usable() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let connection = Arc::new(McpConnection::connect(&fixture.config).await.unwrap());
        let active = fixture.listen().await;
        let epoch = connection.connection_epoch();
        let capabilities = connection.capabilities();
        let status = connection.subscription_status().await;
        assert!(connection.resolve_auth(auth("fail")).await.is_err());
        assert_eq!(connection.connection_epoch(), epoch);
        assert_eq!(connection.capabilities(), capabilities);
        assert_eq!(connection.subscription_status().await, status);
        let reconnect = {
            let connection = connection.clone();
            tokio::spawn(async move { connection.resolve_auth(auth("hold")).await })
        };
        let pending = fixture.listen().await;
        reconnect.abort();
        assert!(reconnect.await.unwrap_err().is_cancelled());
        bounded(pending.frames.closed()).await;
        assert_eq!(connection.connection_epoch(), epoch);
        assert_eq!(connection.capabilities(), capabilities);
        assert_eq!(connection.subscription_status().await, status);
        let mut events = connection.subscribe_events();
        active.notify("notifications/tools/list_changed", json!({}));
        assert!(matches!(
            events.recv().await.unwrap(),
            McpServerEvent::ToolListChanged
        ));
        connection.discover().await.unwrap();
        connection.close().await.unwrap();
        bounded(active.frames.closed()).await;
    })
    .await;
}

#[tokio::test]
async fn modern_to_legacy_and_legacy_to_legacy_replay_committed_resource_uris() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let connection = McpConnection::connect(&fixture.config).await.unwrap();
        let initial = fixture.listen().await;
        connection.subscribe_resource("memo:keep").await.unwrap();
        let subscribed = fixture.listen().await;
        bounded(initial.frames.closed()).await;
        let mut epoch = connection.connection_epoch();
        for token in ["legacy-first", "legacy-second"] {
            connection.resolve_auth(auth(token)).await.unwrap();
            assert_eq!(
                connection.protocol_version(),
                Some(ProtocolVersion::V_2024_11_05)
            );
            assert!(connection.subscription_status().await.is_none());
            assert!(connection.connection_epoch() > epoch);
            epoch = connection.connection_epoch();
            connection.discover().await.unwrap();
            let requests = fixture.requests.lock().await;
            let authorization = format!("Bearer {token}");
            let candidate: Vec<_> = requests
                .iter()
                .filter(|(headers, _)| {
                    headers
                        .get("authorization")
                        .and_then(|value| value.to_str().ok())
                        == Some(authorization.as_str())
                })
                .map(|(_, request)| request)
                .collect();
            let initialized = candidate
                .iter()
                .position(|request| request["method"] == "initialize")
                .unwrap();
            let replayed = candidate
                .iter()
                .position(|request| request["method"] == "resources/subscribe")
                .unwrap();
            assert!(initialized < replayed);
            assert_eq!(candidate[replayed]["params"]["uri"], "memo:keep");
            assert!(
                candidate
                    .iter()
                    .all(|request| request["method"] != "subscriptions/listen")
            );
        }
        bounded(subscribed.frames.closed()).await;
        // A replay rejection during legacy-to-legacy reconnect must not retire
        // the second legacy transport or lose its committed resource set.
        assert!(connection.resolve_auth(auth("legacy-fail")).await.is_err());
        assert_eq!(connection.connection_epoch(), epoch);
        assert_eq!(
            connection.protocol_version(),
            Some(ProtocolVersion::V_2024_11_05)
        );
        connection.list_resources().await.unwrap();
        {
            let requests = fixture.requests.lock().await;
            let (headers, request) = requests.last().unwrap();
            assert_eq!(request["method"], "resources/list");
            assert_eq!(
                headers.get("authorization").unwrap(),
                "Bearer legacy-second"
            );
        }
        // Crossing back also proves the committed URI set survived both legacy commits.
        connection.resolve_auth(auth("replacement")).await.unwrap();
        let modern = fixture.listen().await;
        assert_eq!(modern.filter["resourceSubscriptions"], json!(["memo:keep"]));
        connection.close().await.unwrap();
        bounded(modern.frames.closed()).await;
    })
    .await;
}

#[tokio::test]
async fn rejected_legacy_resource_replay_keeps_authoritative_modern_connection() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let connection = McpConnection::connect(&fixture.config).await.unwrap();
        let initial = fixture.listen().await;
        connection.subscribe_resource("memo:keep").await.unwrap();
        let active = fixture.listen().await;
        bounded(initial.frames.closed()).await;
        let epoch = connection.connection_epoch();
        let capabilities = connection.capabilities();
        let status = connection.subscription_status().await;
        assert!(connection.resolve_auth(auth("legacy-fail")).await.is_err());
        assert_eq!(connection.connection_epoch(), epoch);
        assert_eq!(connection.protocol_version(), Some(ProtocolVersion::V_2026_07_28));
        assert_eq!(connection.capabilities(), capabilities);
        assert_eq!(connection.subscription_status().await, status);
        let requests = fixture.requests.lock().await;
        assert!(requests.iter().any(|(headers, request)| {
            headers.get("authorization").is_some_and(|value| value == "Bearer legacy-fail")
                && request["method"] == "resources/subscribe"
                && request["params"]["uri"] == "memo:keep"
        }));
        drop(requests);
        let mut events = connection.subscribe_events();
        active.notify("notifications/resources/updated", json!({"uri":"memo:keep"}));
        assert!(matches!(events.recv().await.unwrap(), McpServerEvent::ResourceUpdated(params) if params.uri == "memo:keep"));
        connection.discover().await.unwrap();
        connection.close().await.unwrap();
        bounded(active.frames.closed()).await;
    }).await;
}

#[tokio::test]
async fn cancelling_initiated_close_terminates_http_stream_with_connection_retained() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let connection = McpConnection::connect(&fixture.config).await.unwrap();
        let active = fixture.listen().await;
        let epoch = connection.connection_epoch();
        let mut closing = Box::pin(connection.close());
        // One poll initiates shutdown with uncontended guards, then yields on worker cleanup.
        // Drop precisely there rather than racing a spawned task against an arbitrary sleep.
        assert!(futures_util::poll!(closing.as_mut()).is_pending());
        assert!(
            connection.connection_epoch() > epoch,
            "shutdown initiated before cancellation"
        );
        drop(closing);
        bounded(active.frames.closed()).await;
        // Keep the connection alive throughout: stream disposal cannot be attributed to Drop.
        assert!(connection.discover().await.is_err());
    })
    .await;
}

#[tokio::test]
async fn auth_reconnect_partial_ack_preserves_committed_http_subscription() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let connection = Arc::new(McpConnection::connect(&fixture.config).await.unwrap());
        let initial = fixture.listen().await;
        connection.subscribe_resource("memo:first").await.unwrap();
        let first = fixture.listen().await;
        bounded(initial.frames.closed()).await;
        connection.subscribe_resource("memo:second").await.unwrap();
        let active = fixture.listen().await;
        bounded(first.frames.closed()).await;
        let epoch = connection.connection_epoch();
        let capabilities = connection.capabilities();
        let status = connection.subscription_status().await;
        let mut events = connection.subscribe_events();

        // Gate only the candidate's actual POST SSE acknowledgement. Discovery
        // advertises different capabilities, so neither discovery nor a partial
        // acknowledgement may commit any part of this authentication change.
        fixture.hold_ack.store(true, Ordering::SeqCst);
        let reconnect = {
            let connection = connection.clone();
            tokio::spawn(async move { connection.resolve_auth(auth("replacement")).await })
        };
        let candidate = fixture.listen().await;
        assert_eq!(candidate.headers.get("authorization").unwrap(), "Bearer replacement");
        assert_eq!(candidate.filter["resourceSubscriptions"], json!(["memo:first", "memo:second"]));
        let mut partial = candidate.filter.clone();
        partial["resourceSubscriptions"] = json!(["memo:first"]);
        candidate.acknowledge(partial);
        assert!(bounded(reconnect).await.unwrap().is_err());
        bounded(candidate.frames.closed()).await;
        assert_eq!(connection.connection_epoch(), epoch);
        assert_eq!(connection.capabilities(), capabilities);
        assert_eq!(connection.subscription_status().await, status);
        assert!(!active.frames.is_closed());
        for uri in ["memo:first", "memo:second"] {
            active.notify("notifications/resources/updated", json!({"uri":uri}));
            assert!(matches!(events.recv().await.unwrap(), McpServerEvent::ResourceUpdated(params) if params.uri == uri));
        }
        connection.list_resources().await.unwrap();
        let requests = fixture.requests.lock().await;
        let (headers, request) = requests.last().unwrap();
        assert_eq!(request["method"], "resources/list");
        assert_eq!(headers.get("authorization").unwrap(), "Bearer initial");
        drop(requests);
        connection.close().await.unwrap();
        bounded(active.frames.closed()).await;
    }).await;
}

#[tokio::test]
async fn recovered_http_listen_redirties_catalog_after_refresh_during_gap() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        let id = McpServerId::new("http-modern");
        let mut manager = McpServerManager::new().with_server(fixture.config.clone());
        manager.connect_server(&id).await.unwrap();
        let active = fixture.listen().await;
        let connection = manager.connected_server(&id).unwrap().connection();
        manager.refresh_changed_catalogs().await.unwrap();
        assert!(manager.prepare_changed_catalogs().is_empty());
        assert_eq!(
            manager.connected_server(&id).unwrap().snapshot().tools[0].name,
            "tool0"
        );

        // A JSON-RPC error terminates the listen, not the HTTP endpoint. Hold
        // recovery at the next POST SSE ACK while ordinary catalog RPCs work.
        fixture.hold_ack.store(true, Ordering::SeqCst);
        active.fail();
        let recovery = fixture.listen().await;
        assert_eq!(
            connection.subscription_status().await,
            Some(McpSubscriptionStatus::Pending)
        );
        manager.refresh_changed_catalogs().await.unwrap();
        assert!(manager.prepare_changed_catalogs().is_empty());
        assert_eq!(
            manager.connected_server(&id).unwrap().snapshot().tools[0].name,
            "tool0"
        );
        assert_eq!(
            connection.subscription_status().await,
            Some(McpSubscriptionStatus::Pending)
        );

        // Change the remote catalog after the host has consumed the first dirty
        // mark. Send no list-changed notification, only the recovery ACK.
        fixture.tool_version.store(1, Ordering::SeqCst);
        recovery.acknowledge(recovery.filter.clone());
        while !matches!(
            connection.subscription_status().await,
            Some(McpSubscriptionStatus::Active { .. })
        ) {
            tokio::task::yield_now().await;
        }
        assert_eq!(manager.prepare_changed_catalogs().len(), 1);
        manager.refresh_changed_catalogs().await.unwrap();
        assert_eq!(
            manager.connected_server(&id).unwrap().snapshot().tools[0].name,
            "tool1"
        );
        assert!(manager.prepare_changed_catalogs().is_empty());
        manager.disconnect_server(&id).await.unwrap();
        bounded(recovery.frames.closed()).await;
    })
    .await;
}

#[tokio::test]
async fn unsupported_listen_preserves_rpc_but_other_errors_fail_connection() {
    bounded(async {
        for code in [-32601, -32603, -32001] {
            let mut fixture = Fixture::start().await;
            fixture.hold_ack.store(true, Ordering::SeqCst);
            let config = fixture.config.clone();
            let connecting = tokio::spawn(async move { McpConnection::connect(&config).await });
            let listen = fixture.listen().await;
            listen
                .frames
                .send(frame(json!({"jsonrpc":"2.0","id":listen.id,
                "error":{"code":code,"message":"subscription rejected"}})))
                .unwrap();
            let result = connecting.await.unwrap();
            if code == -32601 {
                let connection = result.unwrap();
                assert!(matches!(
                    connection.subscription_status().await,
                    Some(McpSubscriptionStatus::Closed {
                        graceful: false,
                        error: Some(_)
                    })
                ));
                connection.discover().await.unwrap();
                assert!(
                    connection
                        .subscribe_resource("memo:unsupported")
                        .await
                        .is_err()
                );
            } else {
                assert!(result.is_err(), "unrelated subscription failure was hidden");
            }
        }
    })
    .await;
}

#[tokio::test]
async fn auto_http_version_rejection_retries_before_legacy_initialization() {
    bounded(async {
        for (token, expected_discovery_versions) in [
            ("legacy-negotiate", vec!["2026-07-28", "2025-11-25"]),
            (
                "legacy-negotiate-2025-11-25",
                vec!["2026-07-28", "2025-11-25"],
            ),
            (
                "legacy-negotiate-2025-06-18",
                vec!["2026-07-28", "2025-06-18"],
            ),
            (
                "legacy-negotiate-2025-03-26",
                vec!["2026-07-28", "2025-03-26"],
            ),
            (
                "legacy-negotiate-2024-11-05",
                vec!["2026-07-28", "2024-11-05"],
            ),
            ("legacy-direct", vec!["2026-07-28"]),
        ] {
            let mut fixture = Fixture::start().await;
            if let McpTransportBinding::StreamableHttp(binding) = &mut fixture.config.transport {
                *binding = binding.clone().with_bearer_token(token);
            }
            let connection = McpConnection::connect(&fixture.config).await.unwrap();
            // The initialize result, not the retry candidate, determines the
            // version. Legacy connections must not open modern streams.
            assert_eq!(
                connection.protocol_version(),
                Some(ProtocolVersion::V_2024_11_05)
            );
            connection.discover().await.unwrap();
            let requests = fixture.requests.lock().await;
            let discoveries: Vec<_> = requests
                .iter()
                .filter(|(_, m)| m["method"] == "server/discover")
                .collect();
            assert_eq!(discoveries.len(), expected_discovery_versions.len());
            for ((headers, message), version) in discoveries.iter().zip(expected_discovery_versions)
            {
                assert_eq!(headers.get("mcp-protocol-version").unwrap(), version);
                assert_eq!(
                    message["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"],
                    version
                );
            }
            let methods: Vec<_> = requests
                .iter()
                .map(|(_, m)| m["method"].as_str().unwrap())
                .collect();
            let initialize = methods.iter().position(|m| *m == "initialize").unwrap();
            assert_eq!(initialize, discoveries.len());
            assert_eq!(
                requests[initialize].1["params"]["protocolVersion"],
                "2025-11-25"
            );
            assert_eq!(methods[initialize + 1], "notifications/initialized");
            assert!(methods.contains(&"tools/list"));
            assert!(!methods.contains(&"subscriptions/listen"));
        }
    })
    .await;
}

#[tokio::test]
async fn auto_http_no_common_version_does_not_initialize() {
    bounded(async {
        let mut fixture = Fixture::start().await;
        if let McpTransportBinding::StreamableHttp(binding) = &mut fixture.config.transport {
            *binding = binding.clone().with_bearer_token("no-common");
        }
        let error = McpConnection::connect(&fixture.config)
            .await
            .err()
            .expect("must reject incompatible server");
        assert!(
            error.to_string().contains("no compatible protocol version"),
            "{error}"
        );
        let requests = fixture.requests.lock().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].1["method"], "server/discover");
    })
    .await;
}
