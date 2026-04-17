//! Dynamic bearer-token auth for the MCP streamable-HTTP transport.
//!
//! Wires a short-lived token kept in a small in-memory registry into every
//! outgoing request via a [`reqwest_middleware`] layer, then hands the
//! resulting client to [`agentkit_mcp`] as an [`agentkit_http::Http`].

use std::sync::Arc;
use std::time::Duration;

use agentkit_http::Http;
use async_trait::async_trait;
use reqwest_middleware::{ClientBuilder, Middleware, Next};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

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

pub struct DynamicBearerAuth {
    registry: TokenRegistry,
}

impl DynamicBearerAuth {
    pub fn new(registry: TokenRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Middleware for DynamicBearerAuth {
    async fn handle(
        &self,
        mut req: reqwest::Request,
        extensions: &mut http::Extensions,
        next: Next<'_>,
    ) -> reqwest_middleware::Result<reqwest::Response> {
        let token = self.registry.current().await;
        let value = http::HeaderValue::try_from(format!("Bearer {token}"))
            .map_err(|error| reqwest_middleware::Error::Middleware(error.into()))?;
        req.headers_mut().insert(http::header::AUTHORIZATION, value);
        next.run(req, extensions).await
    }
}

/// Build an [`Http`] whose every request is stamped with the current token
/// from `registry`.
pub fn build_http(client: reqwest::Client, registry: TokenRegistry) -> Http {
    let client = ClientBuilder::new(client)
        .with(DynamicBearerAuth::new(registry))
        .build();
    Http::new(client)
}

/// Rotate the registry's token at a fixed interval. Returns the task handle so
/// the caller controls shutdown; drop it to stop rotating.
pub fn spawn_rotator(
    registry: TokenRegistry,
    interval: Duration,
    mut mint: impl FnMut() -> String + Send + 'static,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.tick().await;
        loop {
            ticker.tick().await;
            registry.rotate(mint()).await;
        }
    })
}
