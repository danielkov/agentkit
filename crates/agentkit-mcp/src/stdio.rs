use std::io;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};
use std::time::Duration;

use rmcp::service::{RoleClient, RxJsonRpcMessage, TxJsonRpcMessage};
use rmcp::transport::{Transport, async_rw::AsyncRwTransport};
use tokio::io::{AsyncRead, ReadBuf};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

use crate::StdioTransportConfig;

pub(crate) struct StdioTransport {
    io: AsyncRwTransport<RoleClient, BoundedStdout, ChildStdin>,
    child: Child,
    oversized: Arc<AtomicBool>,
}

impl StdioTransport {
    pub(crate) fn spawn(config: &StdioTransportConfig) -> io::Result<Self> {
        let limit = config.max_line_length;
        if limit == 0 || limit == usize::MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "stdio max_line_length must be positive and less than usize::MAX",
            ));
        }
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .envs(config.env.iter().map(|(key, value)| (key, value)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().expect("piped child stdin");
        let stdout = child.stdout.take().expect("piped child stdout");
        let oversized = Arc::new(AtomicBool::new(false));
        let read = BoundedStdout {
            inner: stdout,
            limit,
            remaining: limit,
            oversized: oversized.clone(),
        };
        Ok(Self {
            io: AsyncRwTransport::new_client(read, stdin).with_max_line_length(limit),
            child,
            oversized,
        })
    }
}

impl Transport<RoleClient> for StdioTransport {
    type Error = io::Error;

    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl Future<Output = io::Result<()>> + Send + 'static {
        self.io.send(item)
    }

    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        let message = self.io.receive().await;
        if self.oversized.load(Ordering::Relaxed) {
            // Unblock any pending writes before rmcp drains the service. `close`
            // reaps the child; kill-on-drop also covers initialization failure.
            if let Err(error) = self.child.start_kill() {
                tracing::warn!(%error, "failed to terminate oversized stdio peer");
            }
        }
        message
    }

    async fn close(&mut self) -> io::Result<()> {
        if self.oversized.load(Ordering::Relaxed) {
            // Do not await rmcp's stdin writer lock after a fatal read: an
            // in-flight write may still be blocked. Dropping the service drops it.
            return self.child.kill().await;
        }
        self.io.close().await?;
        // Give the server a chance to exit after its stdin has been closed.
        match timeout(Duration::from_secs(3), self.child.wait()).await {
            Ok(status) => status.map(|_| ()),
            Err(_) => self.child.kill().await,
        }
    }
}

// Keep rmcp's JSON-RPC framing and compatibility handling, but turn a full
// line budget without a delimiter into an I/O error rather than a dropped line.
struct BoundedStdout {
    inner: ChildStdout,
    limit: usize,
    remaining: usize,
    oversized: Arc<AtomicBool>,
}

impl AsyncRead for BoundedStdout {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        output: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if output.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        let this = &mut *self;
        if this.remaining == 0 {
            this.oversized.store(true, Ordering::Relaxed);
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("stdio line exceeded the {}-byte limit", this.limit),
            )));
        }
        let capacity = output.remaining().min(this.remaining);
        let mut limited = ReadBuf::new(output.initialize_unfilled_to(capacity));
        ready!(Pin::new(&mut this.inner).poll_read(cx, &mut limited))?;
        let bytes = limited.filled();
        let count = bytes.len();
        this.remaining = match bytes.iter().rposition(|&byte| byte == b'\n') {
            Some(newline) => this.limit - (count - newline - 1),
            None => this.remaining - count,
        };
        output.advance(count);
        Poll::Ready(Ok(()))
    }
}
