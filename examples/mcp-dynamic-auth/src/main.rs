//! End-to-end demo: spins up a tiny in-process mock MCP endpoint that
//! validates `Authorization: Bearer <token>` against an expected value, then
//! drives it through an MCP client whose HTTP layer picks the token up fresh
//! from an in-memory registry on every request.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agentkit_core::MetadataMap;
use agentkit_mcp::{
    McpConnection, McpServerConfig, McpTransportBinding, StreamableHttpTransportConfig,
};
use agentkit_tools_core::{AuthOperation, AuthRequest, AuthResolution};
use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use mcp_dynamic_auth::TokenRegistry;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[derive(Clone, Default)]
struct MockState {
    expected_token: Arc<Mutex<String>>,
    observed_tokens: Arc<Mutex<Vec<String>>>,
    request_count: Arc<AtomicU64>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mock = MockState::default();
    *mock.expected_token.lock().await = "token-v1".into();

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr: SocketAddr = listener.local_addr()?;
    let server = {
        let router = Router::new()
            .route("/mcp", post(handle_mcp))
            .with_state(mock.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.ok();
        })
    };

    let registry = TokenRegistry::new("token-v1");

    let connection = McpConnection::connect(&McpServerConfig::new(
        "dynamic-auth",
        McpTransportBinding::StreamableHttp(
            StreamableHttpTransportConfig::new(format!("http://{addr}/mcp"))
                .with_bearer_token(registry.current().await),
        ),
    ))
    .await?;

    let first = connection.discover().await?;
    println!(
        "[first discover] tools={:?} (registry = {})",
        first
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        registry.current().await,
    );

    registry.rotate("token-v2").await;
    *mock.expected_token.lock().await = "token-v2".into();

    // Trigger a reconnect that picks up the rotated token via stored auth
    // credentials. (Phase-4 work will replace this reconnect with a live
    // header rotation on the rmcp transport.)
    let mut credentials = MetadataMap::new();
    credentials.insert(
        "bearer_token".into(),
        Value::String(registry.current().await),
    );
    connection
        .resolve_auth(AuthResolution::Provided {
            request: AuthRequest::new(
                "rotate-token",
                "mcp.dynamic-auth",
                AuthOperation::McpConnect {
                    server_id: "dynamic-auth".into(),
                    metadata: MetadataMap::new(),
                },
            ),
            credentials,
        })
        .await?;

    let second = connection.discover().await?;
    println!(
        "[second discover] tools={:?} (registry = {})",
        second
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        registry.current().await,
    );

    connection.close().await?;
    server.abort();

    let observed = mock.observed_tokens.lock().await.clone();
    println!(
        "[server] total requests = {}",
        mock.request_count.load(Ordering::SeqCst)
    );
    println!("[server] unique tokens seen = {:?}", unique(observed));

    Ok(())
}

async fn handle_mcp(State(state): State<MockState>, headers: HeaderMap, body: String) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);

    let presented = headers
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::to_owned);

    match &presented {
        Some(token) => state.observed_tokens.lock().await.push(token.clone()),
        None => return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response(),
    }

    let expected = state.expected_token.lock().await.clone();
    if presented.as_deref() != Some(expected.as_str()) {
        return (StatusCode::UNAUTHORIZED, "stale token").into_response();
    }

    let message: Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid json").into_response(),
    };

    let id = message.get("id").cloned().unwrap_or(Value::Null);
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");

    let result = match method {
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
        "resources/list" => json!({ "resources": [] }),
        "prompts/list" => json!({ "prompts": [] }),
        _ => {
            let payload = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("unknown method: {method}") }
            });
            return json_ok(payload);
        }
    };

    json_ok(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

fn json_ok(value: Value) -> Response {
    (
        StatusCode::OK,
        [(http::header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&value).unwrap(),
    )
        .into_response()
}

fn unique(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values
}
