//! End-to-end test: agent driven by a scripted [`MockAdapter`], with a
//! static [`ToolRegistry`] (one mock recording tool) federated alongside a
//! [`CatalogReader`] backed by a live in-memory rmcp server.
//!
//! Asserts:
//!
//! - The first [`TurnRequest`] handed to the model lists *both* tool
//!   sources.
//! - When the model emits a tool call against the static tool, the loop
//!   executes it locally; the recording tool captures the request and the
//!   result lands in the next turn's transcript.
//! - When the model emits a tool call against an MCP tool, the loop routes
//!   it through the rmcp client to the in-memory server, the server's call
//!   log records the invocation, and the product comes back in the next
//!   turn's transcript.
//! - The final assistant message is reflected in `result.items`.

use std::sync::{Arc, Mutex};

use agentkit_core::{Item, ItemKind, Part, ToolCallId, ToolCallPart, ToolOutput};
use agentkit_integration_tests::mcp_server::spawn_in_memory;
use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
use agentkit_integration_tests::mock_tool::RecordingTool;
use agentkit_loop::{Agent, AgentEvent, LoopInterrupt, LoopObserver, LoopStep, SessionConfig};
use agentkit_mcp::{McpToolAdapter, McpToolNamespace};
use agentkit_tools_core::{ToolRegistry, ToolSource, ToolSpec, dynamic_catalog};
use serde_json::json;

#[derive(Clone, Default)]
struct EventCapture {
    events: Arc<Mutex<Vec<AgentEvent>>>,
}

impl EventCapture {
    fn snapshot(&self) -> Vec<AgentEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl LoopObserver for EventCapture {
    fn handle_event(&mut self, event: AgentEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[tokio::test]
async fn agent_drives_static_and_mcp_tools_in_one_session() {
    // 1. In-memory MCP server → CatalogWriter populated with McpToolAdapters,
    //    paired with a CatalogReader handed to the agent.
    let server = spawn_in_memory("demo").await;
    let connection = Arc::new(server.connection);
    let snapshot = connection.discover().await.expect("discover succeeds");
    assert_eq!(snapshot.tools.len(), 2, "demo server exposes two tools");

    let (writer, reader) = dynamic_catalog("mcp:demo");
    for tool in snapshot.tools.iter().cloned() {
        let adapter = McpToolAdapter::with_namespace(
            connection.server_id(),
            connection.clone(),
            tool,
            &McpToolNamespace::Default,
        );
        writer.upsert(Arc::new(adapter));
    }
    // Drain bootstrap events so the first agent turn doesn't see a
    // synthetic "two tools just appeared" notification.
    let _ = reader.drain_catalog_events();

    // 2. Static source: a recording mock tool that echoes the input.
    let echo_local = RecordingTool::new(
        ToolSpec::new(
            "echo_local",
            "Echo the supplied text back as a local in-process result.",
            json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
                "additionalProperties": false,
            }),
        ),
        |req| {
            let text = req
                .input
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            Ok(ToolOutput::text(format!("local:{text}")))
        },
    );
    let static_registry = ToolRegistry::new().with(echo_local.clone());

    // 3. Script: turn 1 calls echo_local, turn 2 calls mcp_demo_multiply,
    //    turn 3 finishes with a text answer.
    let mock = MockAdapter::new();
    mock.enqueue(TurnScript::tool_call(ToolCallPart::new(
        ToolCallId::new("call-echo"),
        "echo_local",
        json!({ "text": "ping" }),
    )));
    mock.enqueue(TurnScript::tool_call(ToolCallPart::new(
        ToolCallId::new("call-multiply"),
        "mcp_demo_multiply",
        json!({ "a": 6, "b": 7 }),
    )));
    mock.enqueue(TurnScript::text("done: local:ping and 42"));

    let observer = EventCapture::default();
    let agent = Agent::builder()
        .model(mock.clone())
        .add_tool_source(static_registry)
        .add_tool_source(reader)
        .observer(observer.clone())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new("federation-e2e"),
            vec![Item::text(ItemKind::User, "go")],
        )
        .await
        .unwrap();

    let final_turn = loop {
        match driver.next().await.unwrap() {
            LoopStep::Finished(result) => break result,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                panic!("model script ran out before the final assistant text")
            }
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                panic!("unexpected approval interrupt: {}", pending.request.summary)
            }
        }
    };

    // 4. All three scripted turns consumed.
    assert_eq!(mock.pending_scripts(), 0);

    // 5. The model saw both tool sources on turn 1; subsequent turns
    //    received the previous turn's tool results in the transcript.
    let observed = mock.observed();
    assert_eq!(observed.len(), 3);

    let mut tools_t1 = observed[0].tool_names.clone();
    tools_t1.sort();
    assert_eq!(
        tools_t1,
        vec![
            "echo_local".to_string(),
            "mcp_demo_echo".to_string(),
            "mcp_demo_multiply".to_string(),
        ]
    );
    assert_eq!(observed[0].transcript.len(), 1);

    let local_results = collect_tool_result_text(&observed[1].transcript);
    assert!(
        local_results.iter().any(|t| t == "local:ping"),
        "expected static tool result in turn-2 transcript, saw {local_results:?}",
    );

    let mcp_results = collect_tool_result_text(&observed[2].transcript);
    assert!(
        mcp_results.iter().any(|t| t == "42"),
        "expected MCP multiply result (42) in turn-3 transcript, saw {mcp_results:?}",
    );

    // 6. Mock tools self-report — assert against the tool, not the model.
    let local_invocations = echo_local.invocations();
    assert_eq!(local_invocations.len(), 1);
    assert_eq!(
        local_invocations[0].input,
        json!({ "text": "ping" }),
        "static tool received the scripted input",
    );

    let mcp_invocations = server.server.call_log.lock().unwrap().clone();
    assert_eq!(mcp_invocations, vec![(6, 7)]);

    // 7. Final assistant text is in `result.items`.
    let final_text = collect_assistant_text(&final_turn.items);
    assert_eq!(final_text, "done: local:ping and 42");

    // 8. Observer saw a tool call event per scripted call and a tool
    //    result per executed tool.
    let events = observer.snapshot();
    let tool_call_names: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolCallRequested(call) => Some(call.name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_call_names,
        vec!["echo_local".to_string(), "mcp_demo_multiply".to_string()]
    );
    let tool_result_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ToolResultReceived(_)))
        .count();
    assert_eq!(tool_result_count, 2);
}

fn collect_tool_result_text(transcript: &[Item]) -> Vec<String> {
    let mut out = Vec::new();
    for item in transcript {
        for part in &item.parts {
            if let Part::ToolResult(result) = part
                && let ToolOutput::Text(text) = &result.output
            {
                out.push(text.clone());
            }
        }
    }
    out
}

fn collect_assistant_text(items: &[Item]) -> String {
    let mut out = String::new();
    for item in items {
        if item.kind != ItemKind::Assistant {
            continue;
        }
        for part in &item.parts {
            if let Part::Text(text) = part {
                out.push_str(&text.text);
            }
        }
    }
    out
}
