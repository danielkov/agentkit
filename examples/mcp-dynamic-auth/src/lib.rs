//! In-memory token registry shared between the mock MCP server and the
//! agentkit-mcp client used by the binary.
//!
//! After Phase 1 of the rmcp migration this crate no longer wires a
//! `reqwest_middleware` layer into the agentkit-mcp transport — rmcp's
//! Streamable HTTP transport accepts a static bearer at construction time
//! only. The binary demonstrates rotation by triggering an
//! [`agentkit_mcp::McpConnection::resolve_auth`], which currently rebuilds
//! the underlying rmcp service against the new credentials. Live header
//! rotation lands in Phase 4.

use std::sync::Arc;

use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct TokenRegistry {
    inner: Arc<RwLock<String>>,
}

impl TokenRegistry {
    pub fn new(initial: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(initial.into())),
        }
    }

    pub async fn current(&self) -> String {
        self.inner.read().await.clone()
    }

    pub async fn rotate(&self, next: impl Into<String>) {
        let mut slot = self.inner.write().await;
        *slot = next.into();
    }
}
