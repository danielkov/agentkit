//! Connection-owned subscriptions. Only the worker commits desired state.
use std::{collections::BTreeSet, time::Duration};

use rmcp::{
    model::SubscriptionFilter,
    service::{Peer, RoleClient, Subscription, SubscriptionEnd},
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
};

use crate::{McpError, McpNotificationSink};

const ESTABLISHMENT_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_RETRIES: u32 = 5;

/// Current state of the connection's modern MCP notification subscription.
#[derive(Clone, Debug, PartialEq)]
pub enum McpSubscriptionStatus {
    /// Establishing a fresh stream; catalog contents may need reconciliation.
    Pending,
    /// The server acknowledged this stream. Unaccepted catalog flags remain visible.
    Active {
        /// Complete filter sent to the server.
        requested: SubscriptionFilter,
        /// Filter accepted by the server.
        acknowledged: SubscriptionFilter,
    },
    /// The worker has stopped. A graceful server completion is not retried.
    Closed {
        /// Whether the server returned the final listen result.
        graceful: bool,
        /// Failure diagnostic, if this was not an intentional shutdown.
        error: Option<String>,
    },
}

enum ResourceChange {
    Barrier,
    Add(String),
    Remove(String),
}

struct Update {
    change: ResourceChange,
    reply: oneshot::Sender<Result<(), McpError>>,
}

/// Owns a stream and the only task allowed to replace its committed filter.
pub(super) struct SubscriptionController {
    commands: mpsc::Sender<Update>,
    status: watch::Receiver<McpSubscriptionStatus>,
    requested: watch::Receiver<BTreeSet<String>>,
    stop: watch::Sender<bool>,
    worker: Option<JoinHandle<()>>,
}

impl SubscriptionController {
    /// Return only after the initial stream has been acknowledged.
    pub(super) async fn start(
        peer: Peer<RoleClient>,
        base_filter: SubscriptionFilter,
        sink: McpNotificationSink,
    ) -> Result<Self, McpError> {
        let subscription = establish(&peer, base_filter.clone()).await?;
        let initial = active_status(&base_filter, &subscription);
        let (status_tx, status) = watch::channel(initial);
        let (requested_tx, requested) = watch::channel(resource_set(&base_filter));
        let (commands, updates) = mpsc::channel(32);
        let (stop, mut stopped) = watch::channel(false);
        let worker = tokio::spawn(async move {
            // Dropping run drops every in-flight RMCP request/stream, which
            // unregisters it and sends best-effort cancellation in RMCP.
            tokio::select! {
                result = run(peer, base_filter, subscription, sink, updates, &status_tx, &requested_tx) => {
                    status_tx.send_replace(result);
                }
                _ = stopped.changed() => {
                    status_tx.send_replace(McpSubscriptionStatus::Closed {
                        graceful: false, error: None,
                    });
                }
            }
        });
        Ok(Self {
            commands,
            status,
            requested,
            stop,
            worker: Some(worker),
        })
    }

    pub(super) fn status(&self) -> McpSubscriptionStatus {
        self.status.borrow().clone()
    }

    pub(super) fn subscribe_status(&self) -> watch::Receiver<McpSubscriptionStatus> {
        self.status.clone()
    }

    // Lifecycle replacement waits behind updates already owned by this worker.
    // A canceled API caller may have released its connection guard while its
    // command still awaits acknowledgment.
    pub(super) async fn settled_resources(&self) -> BTreeSet<String> {
        let _ = self.change_resources(ResourceChange::Barrier).await;
        self.requested_resources()
    }

    pub(super) fn requested_resources(&self) -> BTreeSet<String> {
        self.requested.borrow().clone()
    }

    pub(super) async fn subscribe_resource(&self, uri: String) -> Result<(), McpError> {
        self.change_resources(ResourceChange::Add(uri)).await
    }

    pub(super) async fn unsubscribe_resource(&self, uri: String) -> Result<(), McpError> {
        self.change_resources(ResourceChange::Remove(uri)).await
    }

    /// Once enqueued, the worker owns the change despite caller cancellation.
    async fn change_resources(&self, change: ResourceChange) -> Result<(), McpError> {
        let (reply, response) = oneshot::channel();
        self.commands
            .send(Update { change, reply })
            .await
            .map_err(|_| closed_error())?;
        response.await.map_err(|_| closed_error())?
    }

    /// Cancel all outstanding establishment/receive work and join the task.
    pub(super) async fn shutdown(&mut self) {
        self.stop.send_replace(true);
        // Keep ownership until await completes: cancellation of shutdown still
        // leaves Drop able to abort the worker.
        if let Some(worker) = self.worker.as_mut() {
            let _ = worker.await;
        }
        self.worker = None;
    }
}

impl Drop for SubscriptionController {
    fn drop(&mut self) {
        self.stop.send_replace(true);
        if let Some(worker) = self.worker.take() {
            worker.abort();
        }
    }
}

// Non-security jitter avoids synchronized reconnect bursts without adding an RNG
// dependency. The bounded retry count keeps the shift and total delay finite.
fn retry_delay(attempt: u32) -> Duration {
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64
        % 100;
    Duration::from_millis(100 * (1 << attempt) + jitter)
}

fn resource_set(filter: &SubscriptionFilter) -> BTreeSet<String> {
    filter
        .resource_subscriptions
        .iter()
        .flatten()
        .cloned()
        .collect()
}

fn closed_error() -> McpError {
    McpError::Transport("subscription controller closed".into())
}

async fn establish(
    peer: &Peer<RoleClient>,
    filter: SubscriptionFilter,
) -> Result<Subscription, McpError> {
    tokio::time::timeout(ESTABLISHMENT_TIMEOUT, peer.listen(filter))
        .await
        .map_err(|_| McpError::Timeout {
            operation: "subscriptions/listen acknowledgment",
            duration: ESTABLISHMENT_TIMEOUT,
        })?
        .map_err(|error| McpError::Transport(error.to_string()))
}

fn verify_resources(
    requested: &SubscriptionFilter,
    accepted: &SubscriptionFilter,
) -> Result<(), McpError> {
    let accepted = accepted
        .resource_subscriptions
        .as_deref()
        .unwrap_or_default();
    if requested
        .resource_subscriptions
        .as_deref()
        .unwrap_or_default()
        .iter()
        .all(|uri| accepted.contains(uri))
    {
        Ok(())
    } else {
        Err(McpError::Protocol(
            "server did not acknowledge every requested resource URI".into(),
        ))
    }
}

fn active_status(
    filter: &SubscriptionFilter,
    subscription: &Subscription,
) -> McpSubscriptionStatus {
    McpSubscriptionStatus::Active {
        requested: filter.clone(),
        acknowledged: subscription.acknowledged().clone(),
    }
}

fn terminal(subscription: &Subscription) -> Option<McpSubscriptionStatus> {
    match subscription.end() {
        Some(SubscriptionEnd::Graceful(_)) => Some(McpSubscriptionStatus::Closed {
            graceful: true,
            error: None,
        }),
        Some(SubscriptionEnd::Cancelled) => Some(McpSubscriptionStatus::Closed {
            graceful: false,
            error: None,
        }),
        _ => None,
    }
}

async fn run(
    peer: Peer<RoleClient>,
    mut filter: SubscriptionFilter,
    mut active: Subscription,
    sink: McpNotificationSink,
    mut commands: mpsc::Receiver<Update>,
    status: &watch::Sender<McpSubscriptionStatus>,
    requested: &watch::Sender<BTreeSet<String>>,
) -> McpSubscriptionStatus {
    loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else {
                    return McpSubscriptionStatus::Closed { graceful: false, error: None };
                };
                let mut resources = resource_set(&filter);
                match command.change {
                    ResourceChange::Barrier => {
                        let _ = command.reply.send(Ok(()));
                        continue;
                    }
                    ResourceChange::Add(uri) => { resources.insert(uri); }
                    ResourceChange::Remove(uri) => { resources.remove(&uri); }
                }
                let mut proposed = filter.clone();
                proposed.resource_subscriptions = if resources.is_empty() {
                    None
                } else {
                    Some(resources.iter().cloned().collect())
                };
                // Drain the original stream during establishment: it remains
                // authoritative until the replacement ACK has been validated.
                let replacement = establish(&peer, proposed.clone());
                tokio::pin!(replacement);
                let mut original_ended = false;
                let result = loop {
                    tokio::select! {
                        result = &mut replacement => break result,
                        notification = active.next(), if !original_ended => {
                            match notification {
                                Ok(Some(notification)) => sink.dispatch(notification),
                                _ => {
                                    original_ended = true;
                                    sink.mark_catalog_dirty();
                                }
                            }
                        }
                    }
                };
                let result = result.and_then(|replacement| {
                    verify_resources(&proposed, replacement.acknowledged())?;
                    // No awaits or callbacks in this commit. Dropping the old
                    // RMCP stream cancels it without touching the new stream.
                    active = replacement;
                    filter = proposed;
                    requested.send_replace(resources);
                    status.send_replace(active_status(&filter, &active));
                    Ok(())
                });
                let _ = command.reply.send(result);
                // If the original ended during a rejected update, process its
                // terminal state/recovery on the next next() call.
            }
            notification = active.next() => {
                match notification {
                    Ok(Some(notification)) => sink.dispatch(notification),
                    outcome => {
                        if let Some(terminal) = terminal(&active) {
                            return terminal;
                        }
                        sink.mark_catalog_dirty();
                        status.send_replace(McpSubscriptionStatus::Pending);
                        let mut last_error = match outcome {
                            Err(error) => error.to_string(),
                            _ => "subscription ended abruptly".into(),
                        };
                        let mut recovered = None;
                        for attempt in 0..MAX_RETRIES {
                            tokio::time::sleep(retry_delay(attempt)).await;
                            match establish(&peer, filter.clone()).await.and_then(|stream| {
                                verify_resources(&filter, stream.acknowledged())?;
                                Ok(stream)
                            }) {
                                Ok(stream) => { recovered = Some(stream); break; }
                                Err(error) => last_error = error.to_string(),
                            }
                        }
                        let Some(stream) = recovered else {
                            return McpSubscriptionStatus::Closed { graceful: false, error: Some(last_error) };
                        };
                        active = stream;
                        status.send_replace(active_status(&filter, &active));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_ack_does_not_mutate_requested_filter() {
        let mut requested = SubscriptionFilter::default();
        requested.resource_subscriptions = Some(vec!["resource:a".into(), "resource:b".into()]);
        let mut accepted = SubscriptionFilter::default();
        accepted.resource_subscriptions = Some(vec!["resource:a".into()]);
        assert!(verify_resources(&requested, &accepted).is_err());
        assert_eq!(requested.resource_subscriptions.as_ref().unwrap().len(), 2);
        assert!(verify_resources(&requested, &requested).is_ok());
    }

    #[test]
    fn empty_resource_set_accepts_absent_resource_filter() {
        assert!(
            verify_resources(
                &SubscriptionFilter::default(),
                &SubscriptionFilter::default()
            )
            .is_ok()
        );
    }

    struct ResourceServer;

    impl rmcp::ServerHandler for ResourceServer {
        fn get_info(&self) -> rmcp::model::ServerConfig {
            rmcp::model::ServerConfig::new(
                rmcp::model::ServerCapabilities::builder()
                    .enable_resources()
                    .enable_resources_subscribe()
                    .build(),
            )
        }

        fn accepted_subscription_filter(
            &self,
            requested: &SubscriptionFilter,
        ) -> Option<SubscriptionFilter> {
            let mut accepted = requested.clone();
            if let Some(uris) = accepted.resource_subscriptions.as_mut() {
                uris.retain(|uri| uri != "rejected");
            }
            Some(accepted)
        }

        async fn listen(
            &self,
            _context: rmcp::service::SubscriptionContext,
        ) -> Result<(), rmcp::ErrorData> {
            std::future::pending().await
        }
    }

    async fn controller() -> (
        SubscriptionController,
        rmcp::service::RunningService<RoleClient, ()>,
    ) {
        use rmcp::{ClientServiceExt, ServiceExt};
        let (server_transport, client_transport) = tokio::io::duplex(16 * 1024);
        tokio::spawn(async move {
            let server = ResourceServer.serve(server_transport).await.unwrap();
            let _ = server.waiting().await;
        });
        let client = ()
            .serve_with_lifecycle(
                client_transport,
                rmcp::ClientLifecycleMode::Discover {
                    preferred_versions: vec![rmcp::model::ProtocolVersion::V_2026_07_28],
                },
            )
            .await
            .unwrap();
        let (revision, _) = watch::channel(0);
        let (events, _) = tokio::sync::broadcast::channel(16);
        let sink = McpNotificationSink {
            active: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            revision,
            events,
        };
        let controller = SubscriptionController::start(
            client.peer().clone(),
            SubscriptionFilter::default(),
            sink,
        )
        .await
        .unwrap();
        (controller, client)
    }

    #[tokio::test]
    async fn rejected_replacement_preserves_committed_stream_and_resources() {
        let (mut controller, client) = controller().await;
        controller
            .subscribe_resource("accepted".into())
            .await
            .unwrap();
        let before = controller.status();
        assert!(
            controller
                .subscribe_resource("rejected".into())
                .await
                .is_err()
        );
        assert_eq!(controller.status(), before);
        assert_eq!(
            controller.requested_resources(),
            BTreeSet::from(["accepted".into()])
        );
        controller
            .subscribe_resource("another".into())
            .await
            .unwrap();
        assert_eq!(controller.requested_resources().len(), 2);
        controller.shutdown().await;
        assert!(matches!(
            controller.status(),
            McpSubscriptionStatus::Closed { error: None, .. }
        ));
        let _ = client.cancel().await;
    }

    #[tokio::test]
    async fn cancelled_caller_does_not_lose_enqueued_incremental_change() {
        let (mut controller, client) = controller().await;
        {
            let update = controller.subscribe_resource("first".into());
            tokio::pin!(update);
            assert!(futures_util::poll!(&mut update).is_pending());
        }
        let settled = tokio::time::timeout(Duration::from_secs(2), controller.settled_resources())
            .await
            .unwrap();
        assert_eq!(settled, BTreeSet::from(["first".into()]));
        controller
            .subscribe_resource("second".into())
            .await
            .unwrap();
        assert_eq!(
            controller.requested_resources(),
            BTreeSet::from(["first".into(), "second".into()])
        );
        controller
            .unsubscribe_resource("first".into())
            .await
            .unwrap();
        assert_eq!(
            controller.requested_resources(),
            BTreeSet::from(["second".into()])
        );
        controller.shutdown().await;
        let _ = client.cancel().await;
    }
}
