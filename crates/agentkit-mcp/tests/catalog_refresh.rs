//! Catalog publication through the public manager API and a JSON-RPC HTTP boundary.
use agentkit_mcp::{
    AuthOperation, AuthRequest, AuthResolution, McpCatalogEvent, McpServerConfig, McpServerId,
    McpServerManager, McpTransportBinding, StreamableHttpTransportConfig,
};
use agentkit_tools_core::ToolSource;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

#[derive(Clone, Default)]
struct Endpoint {
    fail: Arc<AtomicBool>,
    version: Arc<AtomicU64>,
    lists: Arc<AtomicU64>,
    authenticated: Arc<AtomicBool>,
    resource_entered: Arc<tokio::sync::Notify>,
    resource_release: Arc<tokio::sync::Notify>,
}

async fn rpc(
    State(state): State<Endpoint>,
    headers: HeaderMap,
    Json(message): Json<Value>,
) -> Response {
    if headers
        .get("authorization")
        .is_some_and(|value| value == "Bearer refreshed-token")
    {
        state.authenticated.store(true, Ordering::SeqCst);
    }
    let Some(id) = message.get("id") else {
        return StatusCode::ACCEPTED.into_response();
    };
    let result = match message["method"].as_str().unwrap_or_default() {
        "server/discover" => return (StatusCode::BAD_REQUEST, Json(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"legacy endpoint"}}))).into_response(),
        "initialize" => {
            json!({"protocolVersion":"2024-11-05", "capabilities":{"tools":{}}, "serverInfo":{"name":"catalog-test","version":"1"}})
        }
        "tools/list" => {
            state.lists.fetch_add(1, Ordering::SeqCst);
            if state.fail.load(Ordering::SeqCst) {
                return Json(json!({"jsonrpc":"2.0", "id":id, "error":{"code":-32603,"message":"discovery failed"}})).into_response();
            }
            json!({"tools":[{"name":format!("tool{}",state.version.load(Ordering::SeqCst)),"inputSchema":{"type":"object"}}]})
        }
        "resources/subscribe" => {
            state.resource_entered.notify_one();
            state.resource_release.notified().await;
            json!({})
        }
        "resources/list" => json!({"resources":[]}),
        "prompts/list" => json!({"prompts":[]}),
        _ => json!({}),
    };
    Json(json!({"jsonrpc":"2.0","id":id,"result":result})).into_response()
}

async fn endpoint(name: &str) -> (Endpoint, McpServerConfig, tokio::task::JoinHandle<()>) {
    let state = Endpoint::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/mcp", post(rpc))
        .with_state(state.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (
        state,
        McpServerConfig::new(
            name,
            McpTransportBinding::StreamableHttp(StreamableHttpTransportConfig::new(url)),
        ),
        task,
    )
}

#[tokio::test]
async fn cancelled_and_superseded_work_stays_dirty_and_coalesces() {
    let (state, config, task) = endpoint("catalog").await;
    let id = McpServerId::new("catalog");
    let mut manager = McpServerManager::new().with_server(config);
    manager.connect_server(&id).await.unwrap();
    manager.refresh_changed_catalogs().await.unwrap();
    assert!(manager.prepare_changed_catalogs().is_empty());

    let mut wake = manager.catalog_refresh_wake();
    drop(manager.prepare_catalog_refresh(&id).unwrap());
    tokio::time::timeout(Duration::from_secs(2), wake.changed())
        .await
        .unwrap();
    assert_eq!(manager.prepare_changed_catalogs().len(), 1);
    let stale = manager
        .prepare_changed_catalogs()
        .pop()
        .unwrap()
        .fetch()
        .await;
    state.version.store(1, Ordering::SeqCst);
    // A new invalidation after fetch fences the completed old discovery.
    drop(manager.prepare_catalog_refresh(&id).unwrap());
    assert!(manager.publish_catalog_refresh(stale).unwrap().is_empty());
    assert_eq!(
        manager.connected_server(&id).unwrap().snapshot().tools[0].name,
        "tool0"
    );
    let before = state.lists.load(Ordering::SeqCst);
    drop(manager.prepare_catalog_refresh(&id).unwrap());
    drop(manager.prepare_catalog_refresh(&id).unwrap());
    manager.refresh_changed_catalogs().await.unwrap();
    assert_eq!(state.lists.load(Ordering::SeqCst), before + 1);
    assert_eq!(
        manager.connected_server(&id).unwrap().snapshot().tools[0].name,
        "tool1"
    );
    assert!(manager.prepare_changed_catalogs().is_empty());
    manager.disconnect_server(&id).await.unwrap();
    task.abort();
}

#[tokio::test]
async fn failed_server_does_not_block_other_publications() {
    let (bad, bad_config, bad_task) = endpoint("a-bad").await;
    let (good, good_config, good_task) = endpoint("z-good").await;
    let mut manager = McpServerManager::new()
        .with_server(bad_config)
        .with_server(good_config);
    manager.connect_all().await.unwrap();
    bad.fail.store(true, Ordering::SeqCst);
    good.version.store(2, Ordering::SeqCst);
    assert!(manager.refresh_changed_catalogs().await.is_err());
    let good_id = McpServerId::new("z-good");
    assert_eq!(
        manager.connected_server(&good_id).unwrap().snapshot().tools[0].name,
        "tool2"
    );
    let pending = manager.prepare_changed_catalogs();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].server_id(), &McpServerId::new("a-bad"));
    bad.fail.store(false, Ordering::SeqCst);
    manager.refresh_changed_catalogs().await.unwrap();
    assert!(manager.prepare_changed_catalogs().is_empty());
    bad_task.abort();
    good_task.abort();
}

#[tokio::test]
async fn replacement_rejects_old_connection_results() {
    let (state, config, task) = endpoint("replace").await;
    let id = McpServerId::new("replace");
    let mut manager = McpServerManager::new().with_server(config);
    manager.connect_server(&id).await.unwrap();
    let old = manager.prepare_catalog_refresh(&id).unwrap().fetch().await;
    state.version.store(3, Ordering::SeqCst);
    manager.connect_server(&id).await.unwrap();
    assert!(manager.publish_catalog_refresh(old).unwrap().is_empty());
    assert_eq!(
        manager.connected_server(&id).unwrap().snapshot().tools[0].name,
        "tool3"
    );
    manager.disconnect_server(&id).await.unwrap();
    task.abort();
}

#[tokio::test]
async fn auth_reconnect_fences_discovery_on_same_connection_arc() {
    let (state, config, task) = endpoint("auth-reconnect").await;
    let id = McpServerId::new("auth-reconnect");
    let mut manager = McpServerManager::new().with_server(config);
    let handle = manager.connect_server(&id).await.unwrap();
    let connection = handle.connection();
    let old_epoch = connection.connection_epoch();
    let old = manager.prepare_catalog_refresh(&id).unwrap().fetch().await;
    state.version.store(4, Ordering::SeqCst);
    let request = AuthRequest {
        id: "refresh-auth".into(),
        provider: "test".into(),
        operation: AuthOperation::McpConnect {
            server_id: id.to_string(),
            metadata: Default::default(),
        },
        challenge: Default::default(),
    };
    let mut credentials = agentkit_core::MetadataMap::new();
    credentials.insert("bearer_token".into(), json!("refreshed-token"));
    tokio::time::timeout(
        Duration::from_secs(5),
        connection.resolve_auth(AuthResolution::provided(request, credentials)),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(state.authenticated.load(Ordering::SeqCst));
    assert!(Arc::ptr_eq(
        &connection,
        &manager.connected_server(&id).unwrap().connection()
    ));
    assert!(connection.connection_epoch() > old_epoch);
    assert!(manager.publish_catalog_refresh(old).unwrap().is_empty());
    assert_eq!(
        manager.connected_server(&id).unwrap().snapshot().tools[0].name,
        "tool0"
    );
    assert_eq!(manager.prepare_changed_catalogs().len(), 1);
    manager.refresh_changed_catalogs().await.unwrap();
    assert_eq!(
        manager.connected_server(&id).unwrap().snapshot().tools[0].name,
        "tool4"
    );
    assert!(manager.prepare_changed_catalogs().is_empty());
    manager.disconnect_server(&id).await.unwrap();
    task.abort();
}

// The legacy resources/subscribe request holds the connection lifecycle guard
// until this HTTP boundary replies. It establishes an actual cleanup await,
// rather than relying on a timer to cancel before or after the manager commit.
async fn hold_connection_lifecycle(
    state: &Endpoint,
    manager: &McpServerManager,
    id: &McpServerId,
) -> tokio::task::JoinHandle<()> {
    let connection = manager.connected_server(id).unwrap().connection();
    let pending = tokio::spawn(async move {
        connection.subscribe_resource("memo:held").await.unwrap();
    });
    tokio::time::timeout(Duration::from_secs(5), state.resource_entered.notified())
        .await
        .unwrap();
    pending
}

#[tokio::test]
async fn cancelled_disconnect_has_already_removed_catalog_and_handle() {
    let (state, config, task) = endpoint("disconnect-cancel").await;
    let id = McpServerId::new("disconnect-cancel");
    let mut manager = McpServerManager::new().with_server(config);
    manager.connect_server(&id).await.unwrap();
    let source = manager.source();
    assert_eq!(source.specs().len(), 1);
    let pending = hold_connection_lifecycle(&state, &manager, &id).await;
    let mut events = manager.subscribe_catalog_events();
    let mut disconnect = Box::pin(manager.disconnect_server(&id));
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            result = &mut disconnect => panic!("cleanup unexpectedly completed: {result:?}"),
            event = events.recv() => assert!(matches!(event.unwrap(), McpCatalogEvent::ServerDisconnected { .. })),
        }
    }).await.unwrap();
    drop(disconnect);
    assert!(manager.connected_server(&id).is_none());
    assert!(source.specs().is_empty());
    state.resource_release.notify_one();
    pending.await.unwrap();
    task.abort();
}

#[tokio::test]
async fn cancelled_replacement_has_already_installed_new_catalog_and_handle() {
    let (state, config, task) = endpoint("replace-cancel").await;
    let id = McpServerId::new("replace-cancel");
    let mut manager = McpServerManager::new().with_server(config);
    manager.connect_server(&id).await.unwrap();
    let source = manager.source();
    let old_connection = manager.connected_server(&id).unwrap().connection();
    let pending = hold_connection_lifecycle(&state, &manager, &id).await;
    state.version.store(9, Ordering::SeqCst);
    let mut events = manager.subscribe_catalog_events();
    let mut replace = Box::pin(manager.connect_server(&id));
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            _ = &mut replace => panic!("cleanup unexpectedly completed"),
            event = events.recv() => assert!(matches!(event.unwrap(), McpCatalogEvent::ServerConnected { .. })),
        }
    }).await.unwrap();
    drop(replace);
    assert!(!Arc::ptr_eq(
        &old_connection,
        &manager.connected_server(&id).unwrap().connection()
    ));
    assert_eq!(
        manager.connected_server(&id).unwrap().snapshot().tools[0].name,
        "tool9"
    );
    let names: Vec<_> = source.specs().into_iter().map(|spec| spec.name.0).collect();
    assert_eq!(names, ["mcp_replace-cancel_tool9"]);
    state.resource_release.notify_one();
    pending.await.unwrap();
    old_connection.close().await.unwrap();
    manager.disconnect_server(&id).await.unwrap();
    task.abort();
}
