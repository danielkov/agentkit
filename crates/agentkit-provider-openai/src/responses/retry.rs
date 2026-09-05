use super::*;

/// One accumulator per logical request; no event queue or background task.
pub(super) struct RetryTracker {
    pub(super) started_at: Instant,
    pub(super) accounting: RetryAccounting,
    route: ProviderRoute,
    upstream: ProviderClassification,
    last_attempt_reason: Option<ProviderFailureReason>,
    observer: Option<Arc<dyn RetryObserver>>,
    last_progress: Option<Duration>,
    finalized: bool,
}

impl RetryTracker {
    pub(super) fn new(
        profile: OpenAIResponsesProfile,
        observer: Option<Arc<dyn RetryObserver>>,
    ) -> Self {
        Self {
            started_at: Instant::now(),
            accounting: RetryAccounting::default(),
            route: match profile {
                OpenAIResponsesProfile::Public => ProviderRoute::OpenAiResponses,
                OpenAIResponsesProfile::ChatGptPrivate => ProviderRoute::OpenAiChatGptResponses,
            },
            upstream: ProviderClassification::default(),
            last_attempt_reason: None,
            observer,
            last_progress: None,
            finalized: false,
        }
    }

    pub(super) fn snapshot(&self) -> RetryAccounting {
        self.snapshot_at(self.started_at.elapsed())
    }

    fn snapshot_at(&self, elapsed: Duration) -> RetryAccounting {
        RetryAccounting {
            elapsed,
            ..self.accounting
        }
    }

    pub(super) fn note_failure(&mut self, error: &LoopError) {
        let reason = failure_reason(error);
        if self.accounting.attempts > 0
            && matches!(
                reason,
                ProviderFailureReason::HttpStatus
                    | ProviderFailureReason::Transport
                    | ProviderFailureReason::ResponseFailed
                    | ProviderFailureReason::Protocol
                    | ProviderFailureReason::AttemptTimeout
                    | ProviderFailureReason::IdleTimeout
            )
        {
            self.last_attempt_reason = Some(reason);
        }
        if let Some(failure) = error.provider_failure() {
            // Local timeout/budget/auth stops retain the last provider classification.
            // A new source response (even Unknown) replaces it.
            if matches!(
                failure.reason,
                ProviderFailureReason::HttpStatus | ProviderFailureReason::ResponseFailed
            ) {
                self.upstream = failure.upstream;
            }
        }
    }

    pub(super) fn scheduled(&mut self, error: &LoopError, delay: Duration) {
        self.note_failure(error);
        self.scheduled_at(error, delay, self.started_at.elapsed());
    }

    fn scheduled_at(&mut self, error: &LoopError, delay: Duration, elapsed: Duration) {
        if self
            .last_progress
            .is_some_and(|last| elapsed.saturating_sub(last) < Duration::from_millis(250))
        {
            return;
        }
        self.last_progress = Some(elapsed);
        self.emit(ProviderRetryEvent::Scheduled(RetryProgress {
            route: self.route,
            reason: failure_reason(error),
            upstream: self.upstream,
            accounting: self.snapshot_at(elapsed),
            next_delay: delay,
        }));
    }

    pub(super) fn completed_wait(&mut self, delay: Duration) {
        self.accounting.completed_backoff = self.accounting.completed_backoff.saturating_add(delay);
    }

    pub(super) fn finish(&mut self, error: LoopError) -> LoopError {
        self.note_failure(&error);
        let failure = ProviderFailure {
            route: self.route,
            reason: failure_reason(&error),
            last_attempt_reason: self.last_attempt_reason,
            upstream: self.upstream,
            accounting: self.snapshot(),
        };
        if !self.finalized {
            self.finalized = true;
            self.emit(ProviderRetryEvent::Stopped(failure));
        }
        if matches!(error, LoopError::Cancelled) {
            error
        } else {
            LoopError::ProviderFailure(Box::new(failure))
        }
    }

    pub(super) fn succeed(&mut self) {
        if !self.finalized {
            self.finalized = true;
            self.emit(ProviderRetryEvent::Succeeded {
                route: self.route,
                accounting: self.snapshot(),
            });
        }
    }

    fn emit(&self, event: ProviderRetryEvent) {
        if let Some(observer) = &self.observer {
            observer.on_retry_event(event);
        }
    }
}

pub(super) fn failure_reason(error: &LoopError) -> ProviderFailureReason {
    match error {
        LoopError::Cancelled => ProviderFailureReason::Cancelled,
        LoopError::ProviderFailure(failure) => failure.reason,
        _ => ProviderFailureReason::Protocol,
    }
}

pub(super) fn provider_error(
    reason: ProviderFailureReason,
    upstream: ProviderClassification,
) -> LoopError {
    LoopError::ProviderFailure(Box::new(ProviderFailure {
        route: ProviderRoute::Unknown,
        reason,
        last_attempt_reason: None,
        upstream,
        accounting: RetryAccounting::default(),
    }))
}

pub(super) fn local_error(reason: ProviderFailureReason) -> LoopError {
    provider_error(reason, ProviderClassification::default())
}

pub(super) fn upstream_kind(value: Option<&Value>) -> UpstreamErrorKind {
    match value.and_then(Value::as_str) {
        Some("service_unavailable_error") => UpstreamErrorKind::ServiceUnavailableError,
        Some("server_is_overloaded") => UpstreamErrorKind::ServerIsOverloaded,
        Some("server_error") => UpstreamErrorKind::ServerError,
        Some("rate_limit_error") => UpstreamErrorKind::RateLimitError,
        Some("rate_limit_exceeded") => UpstreamErrorKind::RateLimitExceeded,
        Some("temporarily_unavailable") => UpstreamErrorKind::TemporarilyUnavailable,
        Some("authentication_error") => UpstreamErrorKind::AuthenticationError,
        Some("invalid_api_key") => UpstreamErrorKind::InvalidApiKey,
        Some("invalid_authentication") => UpstreamErrorKind::InvalidAuthentication,
        Some("unauthorized") => UpstreamErrorKind::Unauthorized,
        Some("invalid_request_error") => UpstreamErrorKind::InvalidRequestError,
        Some("permission_denied") => UpstreamErrorKind::PermissionDenied,
        Some("insufficient_quota") => UpstreamErrorKind::InsufficientQuota,
        Some("content_policy_violation") => UpstreamErrorKind::ContentPolicyViolation,
        _ => UpstreamErrorKind::Unknown,
    }
}

pub(super) fn stream_classification(value: &Value, kind: &str) -> ProviderClassification {
    let error = if kind == "response.failed" {
        value.pointer("/response/error").unwrap_or(&Value::Null)
    } else {
        value.get("error").unwrap_or(value)
    };
    ProviderClassification {
        error_type: upstream_kind(error.get("type")),
        code: upstream_kind(error.get("code").or_else(|| value.get("code"))),
        http_status: error
            .get("status")
            .or_else(|| value.get("status"))
            .or_else(|| value.pointer("/response/status_code"))
            .and_then(Value::as_u64)
            .filter(|status| (100..=599).contains(status))
            .map(|status| status as u16),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttling_keeps_exact_accounting_and_never_suppresses_terminal() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = events.clone();
        let mut tracker = RetryTracker::new(
            OpenAIResponsesProfile::Public,
            Some(Arc::new(move |event| capture.lock().unwrap().push(event))),
        );
        let error = local_error(ProviderFailureReason::Transport);
        for millis in [0, 249, 250, 499, 500] {
            tracker.accounting.attempts += 1;
            tracker.completed_wait(Duration::from_millis(15));
            tracker.scheduled_at(
                &error,
                Duration::from_secs(2),
                Duration::from_millis(millis),
            );
        }
        assert_eq!(
            tracker.snapshot_at(Duration::from_secs(1)),
            RetryAccounting {
                attempts: 5,
                completed_backoff: Duration::from_millis(75),
                elapsed: Duration::from_secs(1),
            }
        );
        let error = tracker.finish(local_error(ProviderFailureReason::RetryBudget));
        assert_eq!(error.provider_failure().unwrap().accounting.attempts, 5);
        assert_eq!(
            error
                .provider_failure()
                .unwrap()
                .accounting
                .completed_backoff,
            Duration::from_millis(75)
        );
        tracker.succeed(); // finalization is idempotent, even across dispositions
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4);
        for (index, millis) in [0, 250, 500].into_iter().enumerate() {
            let ProviderRetryEvent::Scheduled(progress) = events[index] else {
                panic!("missing schedule")
            };
            assert_eq!(progress.accounting.elapsed, Duration::from_millis(millis));
        }
        assert!(matches!(events[3], ProviderRetryEvent::Stopped(_)));
    }

    #[test]
    fn unknown_classification_never_retains_payload_and_local_stops_retain_last_source() {
        let secret = "secret-customer-credential-prompt".repeat(4096);
        for value in [
            Value::Null,
            json!({"error": {"type": secret, "code": secret, "message": secret}}),
            json!({"error": {"type": 42, "code": ["secret"], "status": 900}}),
        ] {
            let classification = stream_classification(&value, "error");
            assert_eq!(classification, ProviderClassification::default());
            assert!(serde_json::to_string(&classification).unwrap().len() < 100);
        }
        let source = stream_classification(
            &json!({"response": {"error": {"type": "service_unavailable_error", "code": "server_is_overloaded"}}}),
            "response.failed",
        );
        for reason in [
            ProviderFailureReason::RetryBudget,
            ProviderFailureReason::Authentication,
            ProviderFailureReason::RetryExhausted,
        ] {
            let mut tracker = RetryTracker::new(OpenAIResponsesProfile::ChatGptPrivate, None);
            tracker.accounting.attempts = 1;
            tracker.note_failure(&provider_error(
                ProviderFailureReason::ResponseFailed,
                source,
            ));
            let error = tracker.finish(local_error(reason));
            let failure = error.provider_failure().unwrap();
            assert_eq!(failure.reason, reason);
            assert_eq!(failure.upstream, source);
            assert_eq!(
                failure.last_attempt_reason,
                Some(ProviderFailureReason::ResponseFailed)
            );
        }
    }
}
