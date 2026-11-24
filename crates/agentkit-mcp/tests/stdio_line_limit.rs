use std::io::{self, BufRead, Write};
use std::time::Duration;

use agentkit_mcp::{
    McpConnection, McpError, McpServerConfig, McpTransportBinding, StdioTransportConfig,
};
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
        let response = match request["method"].as_str().unwrap() {
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
            "notifications/initialized" => continue,
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
        writeln!(stdout, "{response}").unwrap();
    }
}

async fn connect(limit: usize, mode: &str) -> McpConnection {
    let mut stdio = StdioTransportConfig::new(std::env::current_exe().unwrap().to_str().unwrap())
        .with_env("AGENTKIT_STDIO_TEST_PEER", mode)
        .with_max_line_length(limit);
    stdio.args = ["--quiet", "--ignored", "--exact", "stdio_peer"]
        .map(String::from)
        .to_vec();
    let config = McpServerConfig::new("test-stdio", McpTransportBinding::Stdio(stdio));
    timeout(WATCHDOG, McpConnection::connect(&config))
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
    let connection = connect(FRAME_LENGTH - 1, mode).await;
    let (first, second) = timeout(WATCHDOG, async {
        tokio::join!(
            connection.call_tool("first", json!({})),
            connection.call_tool("second", json!({})),
        )
    })
    .await
    .expect("oversized reply left calls pending");
    assert!(matches!(first, Err(McpError::Transport(_))), "{first:?}");
    assert!(matches!(second, Err(McpError::Transport(_))), "{second:?}");
    connection.close().await.unwrap();
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
