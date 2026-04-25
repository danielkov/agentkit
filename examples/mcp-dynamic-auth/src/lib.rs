//! In-memory token registry shared between the mock MCP server and the
//! agentkit-mcp client used by the binary.
//!
//! rmcp's Streamable HTTP transport accepts a static bearer at construction
//! time only, so this example rotates credentials by triggering
//! [`agentkit_mcp::McpConnection::resolve_auth`], which rebuilds the
//! underlying rmcp service against the new bearer.

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
