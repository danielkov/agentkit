use std::error::Error as StdError;
use std::time::Duration;

use thiserror::Error;

pub type BoxError = Box<dyn StdError + Send + Sync>;

#[derive(Debug, Error)]
pub enum HttpError {
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    #[error("invalid header: {0}")]
    InvalidHeader(String),

    #[error("request body serialization failed: {0}")]
    Serialize(#[source] serde_json::Error),

    #[error("response body deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),

    #[error("request failed: {0}")]
    Request(#[source] BoxError),

    #[error("response body read failed: {0}")]
    Body(#[source] BoxError),

    #[error("{operation} timed out after {timeout:?}")]
    Timeout {
        operation: &'static str,
        timeout: Duration,
    },

    #[error("response body was truncated (expected {expected} bytes, received {received})")]
    TruncatedBody { expected: u64, received: u64 },

    #[error("{0}")]
    Other(String),
}

impl HttpError {
    pub fn request<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Request(Box::new(err))
    }

    pub fn body<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Body(Box::new(err))
    }

    /// Whether the failure is transient at the HTTP transport layer.
    /// The adapter still decides whether replaying its request is safe.
    pub fn is_retryable_transport(&self) -> bool {
        matches!(
            self,
            Self::Request(_) | Self::Body(_) | Self::Timeout { .. } | Self::TruncatedBody { .. }
        )
    }
}
