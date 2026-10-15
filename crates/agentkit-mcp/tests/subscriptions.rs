//! Real modern RMCP peers: discovery, acknowledged streams, and public invalidations.
use std::{future::Future, time::Duration};

use agentkit_mcp::{
    McpConnection, McpHandlerConfig, McpServerEvent, McpServerId, McpSubscriptionStatus,
};
use rmcp::model::{ProtocolVersion, ServerCapabilities, ServerConfig, SubscriptionFilter};
use rmcp::service::{SubscriptionContext, SubscriptionSink};
use rmcp::{ClientLifecycleMode, ClientServiceExt, ServerHandler, ServiceExt};
use tokio::sync::{mpsc, oneshot};

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(15), future)
        .await
        .expect("MCP operation timed out")
}

struct Stream {
    sink: SubscriptionSink,
    finish: oneshot::Sender<()>,
    ended: oneshot::Receiver<()>,
}

#[derive(Clone)]
struct Server {
    streams: mpsc::UnboundedSender<Stream>,
    partial: bool,
}

impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_tool_list_changed()
                .enable_resources()
                .enable_resources_list_changed()
                .enable_resources_subscribe()
                .enable_prompts()
                .enable_prompts_list_changed()
                .build(),
        )
    }

    fn accepted_subscription_filter(
        &self,
        requested: &SubscriptionFilter,
    ) -> Option<SubscriptionFilter> {
        let mut accepted = requested.clone();
        if self.partial {
            accepted.prompts_list_changed = None;
        }
        if let Some(resources) = &mut accepted.resource_subscriptions {
            resources.retain(|uri| uri != "memo:rejected");
        }
        Some(accepted)
    }

    async fn listen(&self, context: SubscriptionContext) -> Result<(), rmcp::ErrorData> {
        let (finish, finished) = oneshot::channel();
        let (ended_tx, ended) = oneshot::channel();
        let _ = self.streams.send(Stream {
            sink: context.sink().clone(),
            finish,
            ended,
        });
        tokio::select! {
            _ = context.cancelled() => {},
            _ = finished => {},
        }
        let _ = ended_tx.send(());
        Ok(())
    }
}

async fn connect(partial: bool, auto: bool) -> (McpConnection, mpsc::UnboundedReceiver<Stream>) {
    let (streams, incoming) = mpsc::unbounded_channel();
    let (server_io, client_io) = tokio::io::duplex(16 * 1024);
    tokio::spawn(async move {
        let running = Server { streams, partial }.serve(server_io).await.unwrap();
        let _ = running.waiting().await;
    });
    let (handler, channels) = McpHandlerConfig::new().build();
    let versions = vec![ProtocolVersion::V_2026_07_28];
    let lifecycle = if auto {
        ClientLifecycleMode::Auto {
            preferred_versions: versions,
            legacy_version: None,
        }
    } else {
        ClientLifecycleMode::Discover {
            preferred_versions: versions,
        }
    };
    let service = bounded(handler.serve_with_lifecycle(client_io, lifecycle))
        .await
        .unwrap();
    let connection = McpConnection::from_running_service_with_events(
        McpServerId::new("modern"),
        service,
        channels.notifications,
        channels.events,
    );
    (connection, incoming)
}

#[tokio::test]
async fn auto_and_discover_establish_acknowledged_streams_and_bridge_catalog_events() {
    bounded(async {
        for auto in [false, true] {
            let (connection, mut streams) = connect(false, auto).await;
            assert_eq!(
                connection.protocol_version(),
                Some(ProtocolVersion::V_2026_07_28)
            );
            assert!(connection.capabilities().tools.is_some());
            assert!(
                connection.subscription_status().await.is_none(),
                "adoption is synchronous"
            );
            connection.discover().await.unwrap();
            let stream = streams.recv().await.unwrap();
            let Some(McpSubscriptionStatus::Active {
                requested,
                acknowledged,
            }) = connection.subscription_status().await
            else {
                panic!("discovery must await acknowledgment");
            };
            assert_eq!(requested, acknowledged);
            assert_eq!(acknowledged.tools_list_changed, Some(true));
            assert_eq!(acknowledged.resources_list_changed, Some(true));
            assert_eq!(acknowledged.prompts_list_changed, Some(true));
            let mut events = connection.subscribe_events();
            let mut dirty = connection.catalog_changes();
            let revision = connection.catalog_revision();
            stream.sink.notify_tool_list_changed().await.unwrap();
            assert!(matches!(
                events.recv().await.unwrap(),
                McpServerEvent::ToolListChanged
            ));
            stream.sink.notify_resource_list_changed().await.unwrap();
            assert!(matches!(
                events.recv().await.unwrap(),
                McpServerEvent::ResourceListChanged
            ));
            stream.sink.notify_prompt_list_changed().await.unwrap();
            assert!(matches!(
                events.recv().await.unwrap(),
                McpServerEvent::PromptListChanged
            ));
            dirty.changed().await.unwrap();
            assert!(connection.catalog_revision() >= revision + 3);
            let epoch = connection.connection_epoch();
            connection.close().await.unwrap();
            // RMCP drains server handler responses for five seconds after transport
            // EOF before dropping RunningService and canceling its request contexts.
            // Client transport cancellation need not deliver a wire-level request
            // cancellation first, so keep a separate bound above that SDK drain.
            tokio::time::timeout(Duration::from_secs(7), stream.ended)
                .await
                .expect("server subscription handler remained alive after transport EOF")
                .expect("server subscription handler must observe cancellation");
            assert!(connection.connection_epoch() > epoch);
        }
    })
    .await;
}

#[tokio::test]
async fn resource_replacement_rejection_preserves_active_filter_and_stream() {
    bounded(async {
        let (connection, mut streams) = connect(false, false).await;
        connection.discover().await.unwrap();
        let _initial = streams.recv().await.unwrap();
        connection.subscribe_resource("memo:accepted").await.unwrap();
        let accepted = streams.recv().await.unwrap();
        let before = connection.subscription_status().await.unwrap();
        assert!(connection.subscribe_resource("memo:rejected").await.is_err());
        let _rejected = streams.recv().await.unwrap();
        assert_eq!(connection.subscription_status().await.unwrap(), before);
        let mut events = connection.subscribe_events();
        accepted.sink.notify_resource_updated("memo:accepted").await.unwrap();
        assert!(matches!(events.recv().await.unwrap(), McpServerEvent::ResourceUpdated(params) if params.uri == "memo:accepted"));
        connection.unsubscribe_resource("memo:accepted").await.unwrap();
        let _unsubscribed = streams.recv().await.unwrap();
        let Some(McpSubscriptionStatus::Active { requested, acknowledged }) = connection.subscription_status().await else { panic!("active replacement"); };
        assert_eq!(requested, acknowledged);
        assert!(requested.resource_subscriptions.as_ref().is_none_or(Vec::is_empty));
        connection.close().await.unwrap();
    }).await;
}

#[tokio::test]
async fn partial_catalog_acknowledgment_and_graceful_completion_are_observable() {
    bounded(async {
        let (connection, mut streams) = connect(true, false).await;
        connection.discover().await.unwrap();
        let stream = streams.recv().await.unwrap();
        let Some(McpSubscriptionStatus::Active {
            requested,
            acknowledged,
        }) = connection.subscription_status().await
        else {
            panic!("active stream");
        };
        assert_eq!(requested.prompts_list_changed, Some(true));
        assert_eq!(acknowledged.prompts_list_changed, None);
        let mut status = connection.subscription_changes().await.unwrap();
        stream.finish.send(()).unwrap();
        loop {
            status.changed().await.unwrap();
            if matches!(
                *status.borrow_and_update(),
                McpSubscriptionStatus::Closed { graceful: true, .. }
            ) {
                break;
            }
        }
        assert!(
            streams.try_recv().is_err(),
            "graceful completion must not restart"
        );
        connection.close().await.unwrap();
        // A fresh connection starts clean instead of inheriting the closed stream.
        let (next, mut next_streams) = connect(false, true).await;
        next.discover().await.unwrap();
        let _stream = next_streams.recv().await.unwrap();
        assert!(matches!(
            next.subscription_status().await,
            Some(McpSubscriptionStatus::Active { .. })
        ));
        next.close().await.unwrap();
    })
    .await;
}
