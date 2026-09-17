//! Response-wait timeouts, not hard transport deadlines. Channel handshakes
//! establish the ordering under test; the outer watchdogs only bound failures.
// The roots probe deliberately exercises RMCP's legacy request association API.
#![allow(deprecated)]
use std::{sync::Arc, time::Duration};

use agentkit_mcp::{McpConnection, McpError, McpHandlerConfig, McpServerId};
use rmcp::transport::{Transport, async_rw::AsyncRwTransport};
use rmcp::{
    ServerHandler,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, CancelledNotificationParam,
        ClientInfo, ClientNotification, ContentBlock, ErrorCode, ErrorData, JsonRpcMessage,
        ListRootsResult, ProtocolVersion, RequestId, ServerCapabilities, ServerInfo,
    },
    service::{
        RequestContext, RoleClient, RoleServer, RunningService, RxJsonRpcMessage, ServiceError,
        TxJsonRpcMessage, serve_directly,
    },
};
use serde_json::json;
use tokio::sync::{Notify, mpsc, oneshot};

const WATCHDOG: Duration = Duration::from_secs(5);
const RESPONSE_TIMEOUT: Duration = Duration::from_millis(30);

enum Command {
    Reply(Result<CallToolResult, ErrorData>),
    Probe(oneshot::Sender<Result<ListRootsResult, ServiceError>>),
}

struct PendingCall {
    id: RequestId,
    name: String,
    commands: mpsc::UnboundedSender<Command>,
    finished: oneshot::Receiver<()>,
}

#[derive(Clone)]
struct ControlledServer {
    calls: mpsc::UnboundedSender<PendingCall>,
}

impl ServerHandler for ControlledServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        // SEP-2260 lets roots/list probe the client's private pending map on
        // this byte stream (Unknown association, not explicitly Unassociated).
        info.protocol_version = ProtocolVersion::V_2026_07_28;
        info
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let (commands, mut rx) = mpsc::unbounded_channel();
        let (finished, finished_rx) = oneshot::channel();
        self.calls
            .send(PendingCall {
                id: context.id,
                name: request.name.to_string(),
                commands,
                finished: finished_rx,
            })
            .unwrap();
        while let Some(command) = rx.recv().await {
            match command {
                Command::Probe(result) => {
                    // Run inside the original handler scope so the server's
                    // outbound association check allows this request.
                    let _ = result.send(context.peer.list_roots().await);
                }
                Command::Reply(result) => {
                    let _ = finished.send(());
                    // Deliberately ignore context.ct: cancellation is not rollback.
                    return result.map(Into::into);
                }
            }
        }
        Err(ErrorData::internal_error("test caller dropped", None))
    }
}

// Model a server that receives cancellation but deliberately ignores it. RMCP's
// ordinary server loop suppresses late responses after cancellation, which would
// otherwise prevent these tests from exercising late frames at the client.
struct UncooperativeTransport<T> {
    inner: T,
    cancellations: mpsc::UnboundedSender<CancelledNotificationParam>,
    responses: mpsc::UnboundedSender<RequestId>,
}

impl<T: Transport<RoleServer, Error = std::io::Error>> Transport<RoleServer>
    for UncooperativeTransport<T>
{
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleServer>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let id = match &item {
            JsonRpcMessage::Response(response) => Some(response.id.clone()),
            JsonRpcMessage::Error(error) => error.id.clone(),
            _ => None,
        };
        let send = self.inner.send(item);
        let responses = self.responses.clone();
        async move {
            send.await?;
            if let Some(id) = id {
                let _ = responses.send(id);
            }
            Ok(())
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleServer>> {
        loop {
            let item = self.inner.receive().await?;
            if let JsonRpcMessage::Notification(ref notification) = item
                && let ClientNotification::CancelledNotification(ref cancelled) =
                    notification.notification
            {
                let _ = self.cancellations.send(cancelled.params.clone());
                continue;
            }
            return Some(item);
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.inner.close().await
    }
}

struct CancelGate {
    started: Notify,
    release: Notify,
    fail: bool,
}

struct GatedTransport<T> {
    inner: T,
    gate: Option<Arc<CancelGate>>,
}

impl<T: Transport<RoleClient, Error = std::io::Error>> Transport<RoleClient> for GatedTransport<T> {
    type Error = std::io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let gate = if matches!(&item, JsonRpcMessage::Notification(notification)
            if matches!(notification.notification, ClientNotification::CancelledNotification(_)))
        {
            self.gate.clone()
        } else {
            None
        };
        let send = self.inner.send(item);
        async move {
            if let Some(gate) = gate {
                gate.started.notify_one();
                gate.release.notified().await;
                if gate.fail {
                    return Err(std::io::Error::other("cancellation send failed"));
                }
            }
            send.await
        }
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        self.inner.receive().await
    }
    async fn close(&mut self) -> Result<(), Self::Error> {
        self.inner.close().await
    }
}

struct Fixture {
    connection: Arc<McpConnection>,
    server: RunningService<RoleServer, ControlledServer>,
    calls: mpsc::UnboundedReceiver<PendingCall>,
    cancellations: mpsc::UnboundedReceiver<CancelledNotificationParam>,
    responses: mpsc::UnboundedReceiver<RequestId>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_cancel_gate(None)
    }

    fn with_cancel_gate(gate: Option<Arc<CancelGate>>) -> Self {
        let (calls_tx, calls) = mpsc::unbounded_channel();
        let (cancellations_tx, cancellations) = mpsc::unbounded_channel();
        let handler = ControlledServer { calls: calls_tx };
        let (responses_tx, responses) = mpsc::unbounded_channel();
        let server_info = handler.get_info();
        let mut client_info = ClientInfo::default();
        client_info.protocol_version = ProtocolVersion::V_2026_07_28;
        let (server_io, client_io) = tokio::io::duplex(8192);
        let (read, write) = tokio::io::split(server_io);
        let server = serve_directly(
            handler,
            UncooperativeTransport {
                inner: AsyncRwTransport::new(read, write),
                cancellations: cancellations_tx,
                responses: responses_tx,
            },
            Some(client_info),
        );
        let (handler, channels) = McpHandlerConfig::new().build();
        let (read, write) = tokio::io::split(client_io);
        let client = serve_directly(
            handler,
            GatedTransport {
                inner: AsyncRwTransport::new(read, write),
                gate,
            },
            Some(server_info.into()),
        );
        let connection = Arc::new(McpConnection::from_running_service(
            McpServerId::new("timeout-test"),
            client,
            channels.notifications,
        ));
        Self {
            connection,
            server,
            calls,
            cancellations,
            responses,
        }
    }

    fn call(
        &self,
        name: &'static str,
        timeout: Option<Duration>,
    ) -> tokio::task::JoinHandle<Result<CallToolResult, McpError>> {
        let connection = self.connection.clone();
        tokio::spawn(async move {
            match timeout {
                Some(timeout) => {
                    connection
                        .call_tool_with_timeout(name, json!({}), timeout)
                        .await
                }
                None => connection.call_tool(name, json!({})).await,
            }
        })
    }

    async fn next_call(&mut self) -> PendingCall {
        tokio::time::timeout(WATCHDOG, self.calls.recv())
            .await
            .unwrap()
            .unwrap()
    }

    async fn cancelled(&mut self, id: RequestId) {
        let cancellation = tokio::time::timeout(WATCHDOG, self.cancellations.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cancellation.request_id, Some(id));
        assert_eq!(cancellation.reason.as_deref(), Some("request timeout"));
    }
}

async fn probe(call: &PendingCall) -> Result<ListRootsResult, ServiceError> {
    let (tx, rx) = oneshot::channel();
    call.commands.send(Command::Probe(tx)).unwrap();
    tokio::time::timeout(WATCHDOG, rx).await.unwrap().unwrap()
}

async fn outcome(
    call: tokio::task::JoinHandle<Result<CallToolResult, McpError>>,
) -> Result<CallToolResult, McpError> {
    tokio::time::timeout(WATCHDOG, call).await.unwrap().unwrap()
}

fn assert_timeout(result: Result<CallToolResult, McpError>) {
    assert!(
        matches!(result, Err(McpError::Timeout { operation: "tools/call", duration }) if duration == RESPONSE_TIMEOUT)
    );
}

#[tokio::test]
async fn success_before_expiry_preserves_results_and_unlimited_api() {
    let mut fixture = Fixture::new();
    for timeout in [None, Some(WATCHDOG)] {
        let call = fixture.call("success", timeout);
        let pending = fixture.next_call().await;
        let mut result = CallToolResult::success(vec![ContentBlock::text("hello")]);
        result.structured_content = Some(json!({"value": 42}));
        result.is_error = Some(true);
        pending
            .commands
            .send(Command::Reply(Ok(result.clone())))
            .unwrap();
        assert_eq!(outcome(call).await.unwrap(), result);
    }
    assert!(fixture.cancellations.try_recv().is_err());
    fixture.connection.close().await.unwrap();
}

#[tokio::test]
async fn expiry_cleans_pending_state_before_late_success_or_error() {
    let mut fixture = Fixture::new();
    // Positive control: this same roots request is accepted while a call is
    // pending. After each timeout, rejection proves the pending map is empty,
    // rather than merely proving that a cancellation notification was received.
    let control = fixture.call("control", None);
    let pending = fixture.next_call().await;
    assert!(probe(&pending).await.is_ok());
    pending
        .commands
        .send(Command::Reply(Ok(CallToolResult::success(vec![]))))
        .unwrap();
    outcome(control).await.unwrap();
    assert_eq!(fixture.responses.recv().await.unwrap(), pending.id);

    for late in [
        Ok(CallToolResult::success(vec![])),
        Err(ErrorData::internal_error("late", None)),
    ] {
        let call = fixture.call("timeout", Some(RESPONSE_TIMEOUT));
        let pending = fixture.next_call().await;
        assert_timeout(outcome(call).await);
        fixture.cancelled(pending.id.clone()).await;
        let error = probe(&pending).await.unwrap_err();
        assert!(
            matches!(error, ServiceError::McpError(data) if data.code == ErrorCode::INVALID_PARAMS && data.message.contains("SEP-2260"))
        );
        pending.commands.send(Command::Reply(late)).unwrap();
        tokio::time::timeout(WATCHDOG, pending.finished)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            tokio::time::timeout(WATCHDOG, fixture.responses.recv())
                .await
                .unwrap()
                .unwrap(),
            pending.id
        );
        // The write acknowledgement alone does not prove the client read the
        // late frame. A server-initiated ping follows it on the same stream,
        // without adding a client-side outbound pending request.
        tokio::time::timeout(
            WATCHDOG,
            fixture
                .server
                .peer()
                .send_request(rmcp::model::ServerRequest::PingRequest(Default::default())),
        )
        .await
        .unwrap()
        .unwrap();
    }
    assert!(fixture.cancellations.try_recv().is_err());
    fixture.connection.close().await.unwrap();
}

#[tokio::test]
async fn timeout_does_not_cancel_another_pending_call() {
    let mut fixture = Fixture::new();
    let slow = fixture.call("slow", Some(RESPONSE_TIMEOUT));
    let slow_pending = fixture.next_call().await;
    let other = fixture.call("other", None);
    let other_pending = fixture.next_call().await;
    assert_eq!(other_pending.name, "other");
    assert_timeout(outcome(slow).await);
    fixture.cancelled(slow_pending.id).await;
    other_pending
        .commands
        .send(Command::Reply(Ok(CallToolResult::success(vec![]))))
        .unwrap();
    outcome(other).await.unwrap();
    assert!(fixture.cancellations.try_recv().is_err());
    fixture.connection.close().await.unwrap();
}

#[tokio::test]
async fn validation_protocol_error_and_disconnect_are_not_timeouts() {
    let mut fixture = Fixture::new();
    assert!(matches!(
        fixture
            .connection
            .call_tool_with_timeout("invalid", json!([]), WATCHDOG)
            .await,
        Err(McpError::Protocol(_))
    ));
    assert!(fixture.calls.try_recv().is_err());
    let call = fixture.call("protocol", Some(WATCHDOG));
    fixture
        .next_call()
        .await
        .commands
        .send(Command::Reply(Err(ErrorData::invalid_params(
            "invalid", None,
        ))))
        .unwrap();
    assert!(matches!(outcome(call).await, Err(McpError::Invocation(_))));
    let call = fixture.call("disconnect", Some(WATCHDOG));
    let _pending = fixture.next_call().await;
    fixture.server.cancel().await.unwrap();
    assert!(matches!(outcome(call).await, Err(McpError::Transport(_))));
}

#[tokio::test]
async fn cancellation_send_completion_gates_timeout_and_cleanup_even_on_send_error() {
    for fail in [false, true] {
        let gate = Arc::new(CancelGate {
            started: Notify::new(),
            release: Notify::new(),
            fail,
        });
        let mut fixture = Fixture::with_cancel_gate(Some(gate.clone()));
        let call = fixture.call("stalled-cancellation", Some(RESPONSE_TIMEOUT));
        let pending = fixture.next_call().await;
        tokio::time::timeout(WATCHDOG, gate.started.notified())
            .await
            .unwrap();
        // This is the RMCP 3.1.2 limitation, NOT a hard-deadline guarantee:
        // after expiry, cancellation I/O still holds up local abandonment.
        assert!(probe(&pending).await.is_ok());
        assert!(!call.is_finished());
        gate.release.notify_one();
        assert_timeout(outcome(call).await);
        assert!(
            matches!(probe(&pending).await, Err(ServiceError::McpError(data)) if data.code == ErrorCode::INVALID_PARAMS && data.message.contains("SEP-2260"))
        );
        if fail {
            assert!(fixture.cancellations.try_recv().is_err());
        } else {
            fixture.cancelled(pending.id).await;
        }
        fixture.connection.close().await.unwrap();
    }
}
