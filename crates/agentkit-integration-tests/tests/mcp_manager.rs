//! End-to-end coverage for [`McpServerManager`] mutation paths.
//!
//! Each test spins up one or more real HTTP MCP servers (bound to random
//! local ports), points an [`McpServerManager`] at them, and exercises the
//! full agentkit ↔ rmcp ↔ HTTP path for connect / disconnect / refresh.
//! Assertions check both the catalog reader (`manager.source().specs()`)
//! and the lifecycle event stream (`manager.subscribe_catalog_events()`).

use std::time::Duration;

use agentkit_integration_tests::http_mcp_server::{simple_tool, spawn_http_mcp};
use agentkit_mcp::{McpCatalogEvent, McpServerConfig, McpServerId, McpServerManager};
use agentkit_tools_core::{ToolName, ToolSource};
use tokio::sync::broadcast;

/// Drains every event currently buffered on the receiver.
fn drain_events(rx: &mut broadcast::Receiver<McpCatalogEvent>) -> Vec<McpCatalogEvent> {
    let mut out = Vec::new();
    while let Ok(event) = rx.try_recv() {
        out.push(event);
    }
    out
}

fn sorted_tool_names(source: &impl ToolSource) -> Vec<String> {
    let mut names: Vec<String> = source
        .specs()
        .into_iter()
        .map(|spec| spec.name.0)
        .collect();
    names.sort();
    names
}

#[tokio::test]
async fn connect_server_populates_catalog() {
    let server = spawn_http_mcp(vec![
        simple_tool("echo", "Echoes input."),
        simple_tool("multiply", "Multiplies two numbers."),
    ])
    .await;

    let mut manager = McpServerManager::new()
        .with_server(McpServerConfig::streamable_http("demo", &server.url));
    let mut events = manager.subscribe_catalog_events();

    let handles = manager.connect_all().await.expect("connect_all succeeds");
    assert_eq!(handles.len(), 1);

    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec![
            "mcp_demo_echo".to_string(),
            "mcp_demo_multiply".to_string(),
        ],
    );

    let lifecycle = drain_events(&mut events);
    assert!(
        lifecycle
            .iter()
            .any(|event| matches!(event, McpCatalogEvent::ServerConnected { server_id } if server_id == &McpServerId::new("demo"))),
        "expected ServerConnected for demo, saw {lifecycle:?}",
    );
}

#[tokio::test]
async fn disconnect_server_isolates_per_server_tools() {
    let alpha = spawn_http_mcp(vec![simple_tool("only_alpha", "alpha-only tool.")]).await;
    let beta = spawn_http_mcp(vec![simple_tool("only_beta", "beta-only tool.")]).await;

    let mut manager = McpServerManager::new()
        .with_server(McpServerConfig::streamable_http("alpha", &alpha.url))
        .with_server(McpServerConfig::streamable_http("beta", &beta.url));

    manager.connect_all().await.expect("connect_all succeeds");
    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec![
            "mcp_alpha_only_alpha".to_string(),
            "mcp_beta_only_beta".to_string(),
        ],
    );

    let mut events = manager.subscribe_catalog_events();
    manager
        .disconnect_server(&McpServerId::new("alpha"))
        .await
        .expect("disconnect succeeds");

    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_beta_only_beta".to_string()],
        "disconnect must remove only the disconnected server's tools",
    );

    let lifecycle = drain_events(&mut events);
    assert!(
        lifecycle
            .iter()
            .any(|event| matches!(event, McpCatalogEvent::ServerDisconnected { server_id } if server_id == &McpServerId::new("alpha"))),
        "expected ServerDisconnected for alpha, saw {lifecycle:?}",
    );
}

#[tokio::test]
async fn refresh_server_picks_up_added_tool() {
    let server = spawn_http_mcp(vec![simple_tool("first", "Original tool.")]).await;

    let mut manager = McpServerManager::new()
        .with_server(McpServerConfig::streamable_http("dyn", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");
    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_dyn_first".to_string()],
    );

    server.add_tool(simple_tool("second", "Newly-added tool."));

    let mut events = manager.subscribe_catalog_events();
    manager
        .refresh_server(&McpServerId::new("dyn"))
        .await
        .expect("refresh succeeds");

    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_dyn_first".to_string(), "mcp_dyn_second".to_string()],
    );

    let lifecycle = drain_events(&mut events);
    assert!(
        lifecycle.iter().any(|event| matches!(
            event,
            McpCatalogEvent::ToolsChanged { server_id, added, removed, changed }
                if server_id == &McpServerId::new("dyn")
                    && added == &vec!["second".to_string()]
                    && removed.is_empty()
                    && changed.is_empty(),
        )),
        "expected ToolsChanged{{added:[second]}}, saw {lifecycle:?}",
    );
}

#[tokio::test]
async fn refresh_server_picks_up_removed_tool() {
    let server = spawn_http_mcp(vec![
        simple_tool("keeper", "Stays."),
        simple_tool("goner", "Leaves."),
    ])
    .await;

    let mut manager = McpServerManager::new()
        .with_server(McpServerConfig::streamable_http("dyn", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");
    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_dyn_goner".to_string(), "mcp_dyn_keeper".to_string()],
    );

    assert!(server.remove_tool("goner"));

    let mut events = manager.subscribe_catalog_events();
    manager
        .refresh_server(&McpServerId::new("dyn"))
        .await
        .expect("refresh succeeds");

    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_dyn_keeper".to_string()],
    );
    assert!(
        manager.source().get(&ToolName::new("mcp_dyn_goner")).is_none(),
        "removed tool must be gone from the live catalog reader",
    );

    let lifecycle = drain_events(&mut events);
    assert!(
        lifecycle.iter().any(|event| matches!(
            event,
            McpCatalogEvent::ToolsChanged { server_id, added, removed, changed }
                if server_id == &McpServerId::new("dyn")
                    && added.is_empty()
                    && removed == &vec!["goner".to_string()]
                    && changed.is_empty(),
        )),
        "expected ToolsChanged{{removed:[goner]}}, saw {lifecycle:?}",
    );
}

#[tokio::test]
async fn refresh_changed_catalogs_reacts_to_list_changed_notification() {
    let server = spawn_http_mcp(vec![simple_tool("orig", "Initial tool.")]).await;

    let mut manager = McpServerManager::new()
        .with_server(McpServerConfig::streamable_http("live", &server.url));
    manager.connect_all().await.expect("connect_all succeeds");
    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_live_orig".to_string()],
    );

    // Mutate the server-side tool list, then push a notification. The
    // McpConnection's notification queue should receive
    // `notifications/tools/list_changed`, which `refresh_changed_catalogs`
    // consumes to re-discover.
    server.add_tool(simple_tool("hot_added", "Pushed via list_changed."));
    server
        .notify_tool_list_changed()
        .await
        .expect("server notifies list_changed");

    // The notification arrives over the SSE stream asynchronously; poll
    // refresh_changed_catalogs until it sees the change land.
    let mut emitted = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        let new_events = manager
            .refresh_changed_catalogs()
            .await
            .expect("refresh_changed_catalogs succeeds");
        emitted.extend(new_events);
        if emitted.iter().any(|event| {
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

    assert!(
        emitted.iter().any(|event| matches!(
            event,
            McpCatalogEvent::ToolsChanged { server_id, added, .. }
                if server_id == &McpServerId::new("live")
                    && added.contains(&"hot_added".to_string()),
        )),
        "list_changed should produce ToolsChanged{{added:[hot_added]}}, saw {emitted:?}",
    );

    assert_eq!(
        sorted_tool_names(&manager.source()),
        vec!["mcp_live_hot_added".to_string(), "mcp_live_orig".to_string()],
    );
}
