//! End-to-end coverage for [`McpServerManager`] mutation paths, framed
//! through the LLM. Each test stands up one or more real HTTP MCP
//! servers, points an [`McpServerManager`] at them, and exposes its
//! source to a model-driven agent. The snapshot's per-turn `tools` list
//! is the LLM's view of the catalog: a passing snapshot proves that the
//! manager's connect / disconnect / refresh paths actually surface to
//! the model as an available tool catalog change between turns. Where
//! relevant, the model also calls a tool so the snapshot transcript
//! pins down the actual round-trip output.

use std::time::Duration;

use agentkit_core::{Item, ItemKind};
use agentkit_integration_tests::http_mcp_server::{simple_tool, spawn_http_mcp};
use agentkit_integration_tests::snapshot::{
    SessionRecording, SnapshotAdapter, assert_recording, snapshot_path,
};
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_mcp::{McpCatalogEvent, McpServerConfig, McpServerId, McpServerManager};
use agentkit_tools_core::ToolSource;

#[tokio::test]
async fn connect_server_populates_catalog() {
    let server = spawn_http_mcp(vec![
        simple_tool("echo", "Echoes input."),
        simple_tool("multiply", "Multiplies two numbers."),
    ])
    .await;

    let mut manager =
        McpServerManager::new().with_server(McpServerConfig::streamable_http("demo", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");

    let path = snapshot_path("mcp_connect.ron");
    let recording = SessionRecording::load(&path);
    let adapter = SnapshotAdapter::from_recording(&recording);

    let agent = Agent::builder()
        .model(adapter.clone())
        .add_tool_source(manager.source())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new(recording.session_id.clone()),
            recording.initial_items.clone(),
        )
        .await
        .unwrap();

    drive_until_finished(&mut driver).await;

    let observed = adapter.into_recording(&recording, driver.snapshot().transcript.clone());
    assert_recording(&observed, &path);
}

#[tokio::test]
async fn disconnect_server_isolates_per_server_tools() {
    let alpha = spawn_http_mcp(vec![simple_tool("only_alpha", "alpha-only tool.")]).await;
    let beta = spawn_http_mcp(vec![simple_tool("only_beta", "beta-only tool.")]).await;

    let mut manager = McpServerManager::new()
        .with_server(McpServerConfig::streamable_http("alpha", &alpha.url))
        .with_server(McpServerConfig::streamable_http("beta", &beta.url));
    manager.connect_all().await.expect("connect_all succeeds");

    let path = snapshot_path("mcp_disconnect.ron");
    let recording = SessionRecording::load(&path);
    let adapter = SnapshotAdapter::from_recording(&recording);

    let agent = Agent::builder()
        .model(adapter.clone())
        .add_tool_source(manager.source())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new(recording.session_id.clone()),
            recording.initial_items.clone(),
        )
        .await
        .unwrap();

    // Turn 1: catalog shows both alpha and beta tools.
    drive_until_finished(&mut driver).await;

    manager
        .disconnect_server(&McpServerId::new("alpha"))
        .await
        .expect("disconnect succeeds");

    let pending = await_input_request(&mut driver).await;
    pending
        .submit(
            &mut driver,
            vec![Item::text(ItemKind::User, "anything else?")],
        )
        .unwrap();
    // Turn 2: alpha disconnected — catalog only shows beta.
    drive_until_finished(&mut driver).await;

    let observed = adapter.into_recording(&recording, driver.snapshot().transcript.clone());
    assert_recording(&observed, &path);
}

#[tokio::test]
async fn refresh_server_picks_up_added_tool() {
    let server = spawn_http_mcp(vec![simple_tool("first", "Original tool.")]).await;

    let mut manager =
        McpServerManager::new().with_server(McpServerConfig::streamable_http("dyn", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");

    let path = snapshot_path("mcp_refresh_added.ron");
    let recording = SessionRecording::load(&path);
    let adapter = SnapshotAdapter::from_recording(&recording);

    let agent = Agent::builder()
        .model(adapter.clone())
        .add_tool_source(manager.source())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new(recording.session_id.clone()),
            recording.initial_items.clone(),
        )
        .await
        .unwrap();

    // Turn 1: only "first" visible.
    drive_until_finished(&mut driver).await;

    server.add_tool(simple_tool("second", "Newly-added tool."));
    manager
        .refresh_server(&McpServerId::new("dyn"))
        .await
        .expect("refresh succeeds");

    let pending = await_input_request(&mut driver).await;
    pending
        .submit(
            &mut driver,
            vec![Item::text(ItemKind::User, "anything new?")],
        )
        .unwrap();
    // Turn 2: both "first" and "second" visible.
    drive_until_finished(&mut driver).await;

    let observed = adapter.into_recording(&recording, driver.snapshot().transcript.clone());
    assert_recording(&observed, &path);
}

#[tokio::test]
async fn refresh_server_picks_up_removed_tool() {
    let server = spawn_http_mcp(vec![
        simple_tool("keeper", "Stays."),
        simple_tool("goner", "Leaves."),
    ])
    .await;

    let mut manager =
        McpServerManager::new().with_server(McpServerConfig::streamable_http("dyn", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");

    let path = snapshot_path("mcp_refresh_removed.ron");
    let recording = SessionRecording::load(&path);
    let adapter = SnapshotAdapter::from_recording(&recording);

    let agent = Agent::builder()
        .model(adapter.clone())
        .add_tool_source(manager.source())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new(recording.session_id.clone()),
            recording.initial_items.clone(),
        )
        .await
        .unwrap();

    // Turn 1: both keeper and goner visible.
    drive_until_finished(&mut driver).await;

    assert!(server.remove_tool("goner"));
    manager
        .refresh_server(&McpServerId::new("dyn"))
        .await
        .expect("refresh succeeds");

    let pending = await_input_request(&mut driver).await;
    pending
        .submit(&mut driver, vec![Item::text(ItemKind::User, "again?")])
        .unwrap();
    // Turn 2: goner removed — only keeper visible.
    drive_until_finished(&mut driver).await;

    let observed = adapter.into_recording(&recording, driver.snapshot().transcript.clone());
    assert_recording(&observed, &path);
}

#[tokio::test]
async fn reconnect_server_removes_stale_tools() {
    let server = spawn_http_mcp(vec![
        simple_tool("keeper", "Stays."),
        simple_tool("stale", "Removed before reconnect."),
    ])
    .await;

    let mut manager =
        McpServerManager::new().with_server(McpServerConfig::streamable_http("dyn", &server.url));
    let source = manager.source();

    manager
        .connect_server(&McpServerId::new("dyn"))
        .await
        .expect("initial connect succeeds");
    assert_tool_names(&source, &["mcp_dyn_keeper", "mcp_dyn_stale"]);

    assert!(server.remove_tool("stale"));
    manager
        .connect_server(&McpServerId::new("dyn"))
        .await
        .expect("reconnect succeeds");

    assert_tool_names(&source, &["mcp_dyn_keeper"]);
}

#[tokio::test]
async fn refresh_changed_catalogs_reacts_to_list_changed_notification() {
    let server = spawn_http_mcp(vec![simple_tool("orig", "Initial tool.")]).await;

    let mut manager =
        McpServerManager::new().with_server(McpServerConfig::streamable_http("live", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");

    let path = snapshot_path("mcp_list_changed.ron");
    let recording = SessionRecording::load(&path);
    let adapter = SnapshotAdapter::from_recording(&recording);

    let agent = Agent::builder()
        .model(adapter.clone())
        .add_tool_source(manager.source())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new(recording.session_id.clone()),
            recording.initial_items.clone(),
        )
        .await
        .unwrap();

    // Turn 1: only "orig" visible.
    drive_until_finished(&mut driver).await;

    // Mutate server-side then push a list_changed notification.
    server.add_tool(simple_tool("hot_added", "Pushed via list_changed."));
    server
        .notify_tool_list_changed()
        .await
        .expect("server notifies list_changed");

    // Drain the SSE notification — poll refresh_changed_catalogs until
    // the manager picks it up and re-discovers the new tool.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        let events = manager
            .refresh_changed_catalogs()
            .await
            .expect("refresh_changed_catalogs succeeds");
        if events.iter().any(|event| {
            matches!(
                event,
                McpCatalogEvent::ToolsChanged { added, .. }
                    if added.contains(&"hot_added".to_string())
            )
        }) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let pending = await_input_request(&mut driver).await;
    pending
        .submit(
            &mut driver,
            vec![Item::text(ItemKind::User, "anything new?")],
        )
        .unwrap();
    // Turn 2: hot_added visible alongside orig.
    drive_until_finished(&mut driver).await;

    let observed = adapter.into_recording(&recording, driver.snapshot().transcript.clone());
    assert_recording(&observed, &path);
}

fn assert_tool_names(source: &impl ToolSource, expected: &[&str]) {
    let mut actual = source
        .specs()
        .into_iter()
        .map(|spec| spec.name.0)
        .collect::<Vec<_>>();
    actual.sort();
    assert_eq!(actual, expected);
}

async fn drive_until_finished<S>(driver: &mut agentkit_loop::LoopDriver<S>)
where
    S: agentkit_loop::ModelSession,
{
    loop {
        match driver.next().await.unwrap() {
            LoopStep::Finished(_) => return,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                panic!("model script ran out before reaching Finished")
            }
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                panic!("unexpected approval interrupt: {}", pending.request.summary)
            }
        }
    }
}

async fn await_input_request<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> agentkit_loop::InputRequest
where
    S: agentkit_loop::ModelSession,
{
    match driver.next().await.unwrap() {
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(req)) => req,
        other => panic!("expected AwaitingInput, got {other:?}"),
    }
}
