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

/// Synchronous, queue-free observer installed before `begin_turn`.
///
/// Implementations must not block or re-enter the session. Panics propagate, just
/// like loop observers; delivery cannot be guaranteed if observers panic/block.
/// Dropping a future or turn is not explicit cancellation and does not promise a
/// terminal observation. Implementations should rate-limit Scheduled snapshots
/// per logical turn, retaining exact accounting and unsuppressed terminal events.
pub trait RetryObserver: Send + Sync {
    fn on_retry_event(&self, event: ProviderRetryEvent);
}

impl<F: Fn(ProviderRetryEvent) + Send + Sync> RetryObserver for F {
    fn on_retry_event(&self, event: ProviderRetryEvent) {
        self(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentEvent;

    #[test]
    fn event_roundtrip_and_legacy_shape_remain_compatible() {
        let old = r#"{"RunStarted":{"session_id":"session"}}"#;
        let event: AgentEvent = serde_json::from_str(old).unwrap();
        assert_eq!(serde_json::to_string(&event).unwrap(), old);
        let current = AgentEvent::ProviderRetry(ProviderRetryEvent::Stopped(ProviderFailure {
            route: ProviderRoute::OpenAiChatGptResponses,
            reason: ProviderFailureReason::RetryExhausted,
            last_attempt_reason: Some(ProviderFailureReason::ResponseFailed),
            upstream: ProviderClassification {
                error_type: UpstreamErrorKind::ServiceUnavailableError,
                code: UpstreamErrorKind::ServerIsOverloaded,
                http_status: None,
            },
            accounting: RetryAccounting {
                attempts: 3,
                completed_backoff: Duration::from_millis(125),
                elapsed: Duration::from_millis(250),
            },
        }));
        let encoded = serde_json::to_string(&current).unwrap();
        assert_eq!(
            serde_json::from_str::<AgentEvent>(&encoded).unwrap(),
            current
        );
        assert!(encoded.contains("service_unavailable_error"));
        assert!(encoded.contains("server_is_overloaded"));
        assert!(serde_json::from_str::<AgentEvent>(r#"{"ProviderRetry":{"Stopped":{}}}"#).is_err());
        assert_eq!(
            serde_json::from_str::<UpstreamErrorKind>(r#""future-private-value""#).unwrap(),
            UpstreamErrorKind::Unknown
        );
    }
}

#[cfg(test)]
mod cancellation_hook_tests {
    use crate::{
        Agent, LoopError, ModelAdapter, ModelSession, ModelTurn, ModelTurnEvent, SessionConfig,
        TurnRequest,
    };
    use agentkit_core::{CancellationController, Item, ItemKind, TurnCancellation, Usage};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Clone)]
    struct CancellingModel {
        controller: Arc<CancellationController>,
        notified: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ModelAdapter for CancellingModel {
        type Session = Self;
        async fn start_session(&self, _: SessionConfig) -> Result<Self, LoopError> {
            Ok(self.clone())
        }
    }

    #[async_trait::async_trait]
    impl ModelSession for CancellingModel {
        type Turn = Self;
        async fn begin_turn(
            &mut self,
            _: TurnRequest,
            _: Option<TurnCancellation>,
        ) -> Result<Self, LoopError> {
            Ok(self.clone())
        }
    }

    #[async_trait::async_trait]
    impl ModelTurn for CancellingModel {
        fn on_cancelled(&mut self) {
            self.notified.fetch_add(1, Ordering::SeqCst);
        }
        async fn next_event(
            &mut self,
            _: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            self.controller.interrupt();
            Ok(Some(ModelTurnEvent::Usage(Usage::default())))
        }
    }

    #[tokio::test]
    async fn driver_notifies_before_dropping_a_turn_cancelled_between_events() {
        let controller = Arc::new(CancellationController::new());
        let notified = Arc::new(AtomicUsize::new(0));
        let agent = Agent::builder()
            .model(CancellingModel {
                controller: controller.clone(),
                notified: notified.clone(),
            })
            .cancellation(controller.handle())
            .input(vec![Item::text(ItemKind::User, "hello")])
            .build()
            .unwrap();
        let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
        driver.next().await.unwrap();
        assert_eq!(notified.load(Ordering::SeqCst), 1);
    }
}
