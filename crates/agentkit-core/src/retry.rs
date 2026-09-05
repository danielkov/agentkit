//! Sanitized, payload-free model retry observations. These are not effects provenance.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Static provider route; never an endpoint URL or account identifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProviderRoute {
    #[default]
    Unknown,
    OpenAiResponses,
    OpenAiChatGptResponses,
}

/// Allowlisted provider type/code values. Unknown strings are never retained.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UpstreamErrorKind {
    ServiceUnavailableError,
    ServerIsOverloaded,
    ServerError,
    RateLimitError,
    RateLimitExceeded,
    TemporarilyUnavailable,
    AuthenticationError,
    InvalidApiKey,
    InvalidAuthentication,
    Unauthorized,
    InvalidRequestError,
    PermissionDenied,
    InsufficientQuota,
    ContentPolicyViolation,
    #[default]
    #[serde(other)]
    Unknown,
}

/// Sanitized source classification, kept separate from the local stopping reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderClassification {
    pub error_type: UpstreamErrorKind,
    pub code: UpstreamErrorKind,
    /// Source HTTP status, if present. No headers or response body are retained.
    pub http_status: Option<u16>,
}

/// Local reason for a failed attempt or logical request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProviderFailureReason {
    HttpStatus,
    Transport,
    ResponseFailed,
    Protocol,
    InvalidRequest,
    Authentication,
    AttemptTimeout,
    IdleTimeout,
    RetryExhausted,
    RetryBudget,
    RetryDisabled,
    ReplayUnsafe,
    Cancelled,
}

/// Per-logical-request accounting, independent of policy retry count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryAccounting {
    /// Actual HTTP sends started, including a resend after authentication refresh.
    /// Preflight/authentication failures can have zero attempts.
    pub attempts: u64,
    /// Sum of requested durations of fully completed backoff waits. Interrupted
    /// waits contribute zero, even when they consumed wall-clock time.
    pub completed_backoff: Duration,
    /// Monotonic elapsed time since before initial authentication/preflight.
    pub elapsed: Duration,
}

/// A nonterminal snapshot emitted before a retry wait or reactive refresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryProgress {
    pub route: ProviderRoute,
    pub reason: ProviderFailureReason,
    pub upstream: ProviderClassification,
    /// Attempts already started. The planned next send is `attempts + 1`, but
    /// cancellation/preflight failure can prevent it from ever starting.
    pub accounting: RetryAccounting,
    pub next_delay: Duration,
}

/// Payload-free terminal model failure. Display and Debug contain only typed data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("provider request failed ({reason:?}; attempts: {attempts})", attempts = .accounting.attempts)]
pub struct ProviderFailure {
    pub route: ProviderRoute,
    pub reason: ProviderFailureReason,
    /// Last failed request-attempt category, retained across local budget/limit stops.
    /// None when no request attempt failed (for example initial authentication).
    pub last_attempt_reason: Option<ProviderFailureReason>,
    pub upstream: ProviderClassification,
    pub accounting: RetryAccounting,
}

/// Observational lifecycle; never a second model result or a tool-effects record.
///
/// Correlate through the enclosing `ObservedEvent.session_id` and current
/// `AgentEvent::TurnStarted`. Direct session consumers own that association.
/// Stable fatal event IDs belong to the host, not to this payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProviderRetryEvent {
    Scheduled(RetryProgress),
    /// Emitted once on explicit failure/cancellation, including zero-send failures.
    Stopped(ProviderFailure),
    /// Clears retry activity without introducing another successful model result.
    Succeeded {
        route: ProviderRoute,
        accounting: RetryAccounting,
    },
}
