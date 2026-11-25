use std::io::{self, BufRead, Write};
use std::time::Duration;

use agentkit_mcp::{
    AuthOperation, AuthRequest, AuthResolution, McpConnection, McpError, McpServerConfig,
    McpSubscriptionStatus, McpTransportBinding, StdioLineTooLong, StdioTransportConfig,
};
use rmcp::model::{DiscoverResult, ProtocolVersion, ServerCapabilities};
use serde_json::{Value, json};
use tokio::time::timeout;

const FRAME_LENGTH: usize = 20 * 1024;
const WATCHDOG: Duration = Duration::from_secs(5);

// Re-execute this test binary as a stdio peer, rather than requiring an external runtime.
#[test]
#[ignore = "stdio fixture invoked by the transport tests"]
fn stdio_peer() {
    let Ok(mode) = std::env::var("AGENTKIT_STDIO_TEST_PEER") else {
        return;
    };
    let mut stdout = io::stdout().lock();
    let mut first_call = None;
    for line in io::stdin().lock().lines() {
        let request: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let id = &request["id"];
        let method = request["method"].as_str().unwrap();
        if mode == "startup-eof" {
            return;
        }
        let mut response = match method {
            "server/discover" if mode.starts_with("modern-") => json!({
                "jsonrpc": "2.0", "id": id,
                "result": DiscoverResult::new(
                    vec![ProtocolVersion::V_2026_07_28],
                    ServerCapabilities::builder().enable_tools().enable_tool_list_changed().build(),
                )
            }),
            "subscriptions/listen" if mode == "modern-unsupported" => json!({
                "jsonrpc": "2.0", "id": id,
                "error": {"code": -32601, "message": "Method not found"}
            }),
            "subscriptions/listen" => json!({
                "jsonrpc": "2.0", "method": "notifications/subscriptions/acknowledged",
                "params": {
                    "notifications": request["params"]["notifications"],
                    "_meta": {"io.modelcontextprotocol/subscriptionId": id}
                }
            }),
            "server/discover" => json!({
                "jsonrpc": "2.0", "id": id,
                "error": {"code": -32601, "message": "Method not found"}
            }),
            "initialize" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stdio-peer", "version": "1.0"}
                }
            }),
            "notifications/initialized" | "notifications/cancelled" => continue,
            "tools/call" if request["params"]["name"] == "eof" => return,
            "tools/call" if request["params"]["name"] == "error" => json!({
                "jsonrpc": "2.0", "id": id,
                "error": {"code": -32603, "message": "fixture RPC error"}
            }),
            "tools/call" if request["params"]["name"] == "hang" => continue,
            "tools/call" => {
                let Some(first_id) = first_call.replace(id.clone()) else {
                    continue;
                };
                let mut large = json!({
                    "jsonrpc": "2.0", "id": first_id,
                    "result": {"content": [{"type": "text", "text": ""}]}
                });
                let empty_frame_length = large.to_string().len() + 1;
                large["result"]["content"][0]["text"] =
                    Value::String("x".repeat(FRAME_LENGTH - empty_frame_length));
                if mode == "unterminated" {
                    write!(stdout, "{large}").unwrap();
                    stdout.flush().unwrap();
                } else {
                    writeln!(stdout, "{large}").unwrap();
                }
                // Leave the second call pending if no subsequent request arrives.
                continue;
            }
            method => panic!("unexpected method: {method}"),
        };
        let pad_path = match (mode.as_str(), method) {
            ("discover-oversize", "server/discover") => Some("/error/message"),
            ("initialize-oversize", "initialize") => Some("/result/serverInfo/name"),
            _ => None,
        };
        if let Some(path) = pad_path {
            *response.pointer_mut(path).unwrap() = Value::String("x".repeat(FRAME_LENGTH));
        }
        writeln!(stdout, "{response}").unwrap();
    }
}

fn config(limit: usize, mode: &str) -> McpServerConfig {
    let mut stdio = StdioTransportConfig::new(std::env::current_exe().unwrap().to_str().unwrap())
        .with_env("AGENTKIT_STDIO_TEST_PEER", mode)
        .with_max_line_length(limit);
    stdio.args = ["--quiet", "--ignored", "--exact", "stdio_peer"]
        .map(String::from)
        .to_vec();
    McpServerConfig::new("test-stdio", McpTransportBinding::Stdio(stdio))
}

async fn connect(limit: usize, mode: &str) -> McpConnection {
    timeout(WATCHDOG, McpConnection::connect(&config(limit, mode)))
        .await
        .expect("stdio handshake timed out")
        .expect("stdio handshake failed")
}

#[tokio::test]
async fn oversize_closes_stdio_and_fails_all_pending_calls() {
    assert_pending_calls_fail("complete").await;
}

#[tokio::test]
async fn exhausting_the_line_budget_does_not_wait_for_a_delimiter() {
    assert_pending_calls_fail("unterminated").await;
}

async fn assert_pending_calls_fail(mode: &str) {
    let limit = FRAME_LENGTH - 1;
    let connection = connect(limit, mode).await;
    break_with_oversize(&connection, limit).await;
    assert_oversize(connection.list_tools().await.unwrap_err(), limit);
    connection.close().await.unwrap();
}

async fn break_with_oversize(connection: &McpConnection, limit: usize) {
    let (first, second) = timeout(WATCHDOG, async {
        tokio::join!(
            connection.call_tool("first", json!({})),
            connection.call_tool("second", json!({})),
        )
    })
    .await
    .expect("oversized reply left calls pending");
    assert_oversize(first.unwrap_err(), limit);
    assert_oversize(second.unwrap_err(), limit);
}

#[tokio::test]
async fn queued_subscription_update_reports_stdout_oversize() {
    let limit = 4096;
    let connection = connect(limit, "modern-subscription").await;
    let mut status = connection.subscription_changes().await.unwrap();
    assert!(matches!(
        *status.borrow(),
        McpSubscriptionStatus::Active { .. }
    ));

    break_with_oversize(&connection, limit).await;
    timeout(
        WATCHDOG,
        status.wait_for(|s| matches!(s, McpSubscriptionStatus::Pending)),
    )
    .await
    .expect("subscription did not enter recovery")
    .unwrap();

    let update = connection.subscribe_resource("file:///queued");
    tokio::pin!(update);
    assert!(futures_util::poll!(&mut update).is_pending());
    let error = timeout(WATCHDOG, update)
        .await
        .expect("queued update did not complete")
        .unwrap_err();
    assert_oversize(error, limit);
    assert_oversize(
        connection
            .unsubscribe_resource("file:///queued")
            .await
            .unwrap_err(),
        limit,
    );
}

#[tokio::test]
async fn unrelated_subscription_closure_is_not_reclassified_after_stdout_oversize() {
    let limit = 4096;
    let connection = connect(limit, "modern-unsupported").await;
    assert!(matches!(
        connection.subscription_status().await,
        Some(McpSubscriptionStatus::Closed { .. })
    ));
    break_with_oversize(&connection, limit).await;
    assert!(matches!(
        connection.subscribe_resource("file:///unavailable").await,
        Err(McpError::Transport(message)) if message == "subscription controller closed"
    ));
}

#[tokio::test]
async fn raised_limit_accepts_a_reply_at_the_byte_boundary() {
    let connection = connect(FRAME_LENGTH, "complete").await;
    // The peer waits for both requests before replying to the first request it sees.
    let first_reply = timeout(WATCHDOG, async {
        tokio::select! {
            result = connection.call_tool("first", json!({})) => result,
            result = connection.call_tool("second", json!({})) => result,
        }
    })
    .await
    .expect("reply at the byte limit was dropped")
    .unwrap();
    assert!(first_reply.content[0].as_text().unwrap().text.len() > 16 * 1024);
    connection.close().await.unwrap();
}

fn assert_oversize(error: McpError, limit: usize) {
    assert_eq!(
        error.stdio_line_too_long(),
        Some(&StdioLineTooLong { limit }),
        "{error:?}",
    );
    assert!(matches!(error, McpError::Io(ref error) if error.kind() == io::ErrorKind::InvalidData));
}

#[tokio::test]
async fn startup_reports_stdout_oversize_in_discovery_and_legacy_fallback() {
    let limit = 1024;
    for mode in ["discover-oversize", "initialize-oversize"] {
        let result = timeout(WATCHDOG, McpConnection::connect(&config(limit, mode)))
            .await
            .expect("oversized handshake left connect pending");
        let Err(error) = result else {
            panic!("oversized handshake succeeded in {mode}");
        };
        assert_oversize(error, limit);
    }
}

#[tokio::test]
async fn ordinary_startup_eof_is_still_a_transport_error() {
    let result = timeout(
        WATCHDOG,
        McpConnection::connect(&config(1024, "startup-eof")),
    )
    .await
    .expect("EOF left connect pending");
    assert!(
        matches!(result, Err(McpError::Transport(message)) if message.contains("connection closed"))
    );
}

#[tokio::test]
async fn replacing_stdio_after_oversize_does_not_taint_rpc_errors_timeouts_or_eof() {
    timeout(WATCHDOG, async {
        let limit = FRAME_LENGTH - 1;
        let connection = connect(limit, "complete").await;
        let pending_old_call = connection.call_tool("hang", json!({}));
        tokio::pin!(pending_old_call);
        assert!(futures_util::poll!(&mut pending_old_call).is_pending());
        break_with_oversize(&connection, limit).await;
        let old_epoch = connection.connection_epoch();
        connection
            .resolve_auth(AuthResolution::Provided {
                request: AuthRequest {
                    id: "test".into(),
                    provider: "test".into(),
                    operation: AuthOperation::McpConnect {
                        server_id: "test-stdio".into(),
                        metadata: Default::default(),
                    },
                    challenge: Default::default(),
                },
                credentials: Default::default(),
            })
            .await
            .unwrap();
        assert!(connection.connection_epoch() > old_epoch);
        assert_oversize(pending_old_call.await.unwrap_err(), limit);
        let rpc_error = connection.call_tool("error", json!({})).await;
        assert!(matches!(rpc_error, Err(McpError::Invocation(_))), "{rpc_error:?}");
        let duration = Duration::from_millis(10);
        let timed_out = connection
            .call_tool_with_timeout("hang", json!({}), duration)
            .await;
        assert!(matches!(timed_out, Err(McpError::Timeout { operation: "tools/call", duration: actual }) if actual == duration));
        let eof = connection.call_tool("eof", json!({})).await;
        assert!(matches!(eof, Err(McpError::Transport(_))), "{eof:?}");
        connection.close().await.unwrap();
    })
    .await
    .expect("connection lifecycle stalled");
}
