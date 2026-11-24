//! Integration test for [`agentkit_mcp::McpRequestDecorator`].
//!
//! The decorator puts a marker in each `tools/call` request's extensions; a
//! custom [`McpHttpClient`] copies it into an HTTP header; the mock server
//! records the header per call. This proves the extension travels with the
//! message to the transport, that each call carries its own value, and that
//! the decorator runs inside the `mcp.call_tool` span.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agentkit_mcp::{
    ClientJsonRpcMessage, McpCallToolRequest, McpConnection, McpHandlerConfig, McpHttpClient,
    McpServerConfig, McpSseStream, McpStreamableHttpError, McpStreamableHttpPostResponse,
    McpTransportBinding, StreamableHttpTransportConfig,
};
use async_trait::async_trait;
use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use rmcp::model::ClientRequest;
use rmcp::transport::streamable_http_client::StreamableHttpClient as RmcpStreamableHttpClient;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

const MARKER_HEADER: &str = "x-call-marker";

/// What the decorator stores in the request extensions.
#[derive(Clone)]
struct CallMarker(String);

#[derive(Clone, Default)]
struct MockState {
    /// `(method, marker header)` for every POST the server received.
    seen: Arc<Mutex<Vec<(String, Option<String>)>>>,
}

async fn spawn_mock(state: MockState) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = Router::new()
        .route("/mcp", post(handle_mcp))
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, router).await.ok();
    });
    addr
}

async fn handle_mcp(State(state): State<MockState>, headers: HeaderMap, body: String) -> Response {
    let message: Value = serde_json::from_str(&body).expect("valid json from rmcp");
    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let marker = headers
        .get(MARKER_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    state.seen.lock().await.push((method.clone(), marker));

    let result = match method.as_str() {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "mock", "version": "0.0.0" }
        }),
        "notifications/initialized" => return StatusCode::ACCEPTED.into_response(),
        "tools/list" => json!({
            "tools": [{
                "name": "ping",
                "description": "replies with pong",
                "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
            }]
        }),
        "tools/call" => json!({
            "content": [{ "type": "text", "text": "pong" }],
            "isError": false
        }),
        "resources/list" => json!({ "resources": [] }),
        "prompts/list" => json!({ "prompts": [] }),
        _ => {
            return json_ok(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("unknown method: {method}") }
            }));
        }
    };
    json_ok(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

fn json_ok(value: Value) -> Response {
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        serde_json::to_string(&value).unwrap(),
    )
        .into_response()
}

/// Copies a [`CallMarker`] extension into [`MARKER_HEADER`].
struct MarkerHeaderClient {
    inner: reqwest::Client,
}

#[async_trait]
impl McpHttpClient for MarkerHeaderClient {
    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        mut custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<McpStreamableHttpPostResponse, McpStreamableHttpError<reqwest::Error>> {
        if let ClientJsonRpcMessage::Request(request) = &message
            && let ClientRequest::CallToolRequest(call) = &request.request
            && let Some(CallMarker(marker)) = call.extensions.get::<CallMarker>()
        {
            custom_headers.insert(
                HeaderName::from_static(MARKER_HEADER),
                HeaderValue::from_str(marker).unwrap(),
            );
        }
        RmcpStreamableHttpClient::post_message(
            &self.inner,
            uri,
            message,
            session_id,
            auth_header,
            custom_headers,
        )
        .await
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), McpStreamableHttpError<reqwest::Error>> {
        RmcpStreamableHttpClient::delete_session(
            &self.inner,
            uri,
            session_id,
            auth_header,
            custom_headers,
        )
        .await
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<McpSseStream, McpStreamableHttpError<reqwest::Error>> {
        RmcpStreamableHttpClient::get_stream(
            &self.inner,
            uri,
            session_id,
            last_event_id,
            auth_header,
            custom_headers,
        )
        .await
    }
}

#[tokio::test]
async fn decorated_extensions_reach_the_transport_per_call() {
    // A registry subscriber so span metadata is available to the decorator.
    let _subscriber = tracing::subscriber::set_default(tracing_subscriber::registry());

    let mock = MockState::default();
    let addr = spawn_mock(mock.clone()).await;

    let counter = Arc::new(AtomicU64::new(0));
    let decorator_counter = counter.clone();
    let decorator = move |request: &mut McpCallToolRequest| {
        let span = tracing::Span::current()
            .metadata()
            .map(|metadata| metadata.name())
            .unwrap_or("none");
        let n = decorator_counter.fetch_add(1, Ordering::SeqCst);
        request
            .extensions
            .insert(CallMarker(format!("{span}:{}:{n}", request.params.name)));
    };

    let connection = McpConnection::connect_with_handler(
        &McpServerConfig::new(
            "decorated",
            McpTransportBinding::StreamableHttp(
                StreamableHttpTransportConfig::new(format!("http://{addr}/mcp")).with_http_client(
                    Arc::new(MarkerHeaderClient {
                        inner: reqwest::Client::new(),
                    }),
                ),
            ),
        ),
        McpHandlerConfig::new().with_request_decorator(Arc::new(decorator)),
    )
    .await
    .expect("connect succeeds");

    for _ in 0..3 {
        connection
            .call_tool("ping", json!({}))
            .await
            .expect("tool call succeeds");
    }

    let seen = mock.seen.lock().await.clone();
    let markers: Vec<Option<String>> = seen
        .iter()
        .filter(|(method, _)| method == "tools/call")
        .map(|(_, marker)| marker.clone())
        .collect();
    assert_eq!(
        markers,
        vec![
            Some("mcp.call_tool:ping:0".to_string()),
            Some("mcp.call_tool:ping:1".to_string()),
            Some("mcp.call_tool:ping:2".to_string()),
        ],
        "each tools/call carries its own marker, set inside the mcp.call_tool span"
    );
    // Only tools/call requests are decorated.
    assert!(
        seen.iter()
            .filter(|(method, _)| method != "tools/call")
            .all(|(_, marker)| marker.is_none()),
        "unexpected marker on a non-call request: {seen:?}"
    );
}
