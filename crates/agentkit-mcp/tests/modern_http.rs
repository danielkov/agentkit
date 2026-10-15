//! Built-in Auto lifecycle through an actual HTTP JSON-RPC and POST SSE boundary.
use std::{convert::Infallible, future::Future, sync::Arc, time::Duration};

use agentkit_mcp::{
    AuthOperation, AuthRequest, AuthResolution, McpConnection, McpServerConfig, McpServerEvent,
    McpSubscriptionStatus, McpTransportBinding, StreamableHttpTransportConfig,
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
            if token != "Bearer hold" {
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
        "tools/list" => json!({"tools":[]}),
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
        let requests = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let app = Router::new().route("/mcp", post(rpc)).with_state(Endpoint {
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
