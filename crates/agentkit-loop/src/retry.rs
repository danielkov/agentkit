//! Loop-owned observer for canonical core retry values.

pub use agentkit_core::retry::*;

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
    use std::time::Duration;

    #[test]
    fn core_and_loop_retry_values_have_identical_types() {
        let accounting: agentkit_core::retry::RetryAccounting = RetryAccounting::default();
        let _: crate::RetryAccounting = accounting;
        let event: agentkit_core::retry::ProviderRetryEvent = ProviderRetryEvent::Succeeded {
            route: ProviderRoute::Unknown,
            accounting,
        };
        let _: crate::ProviderRetryEvent = event;
    }

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
