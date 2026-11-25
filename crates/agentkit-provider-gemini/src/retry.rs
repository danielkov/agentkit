//! Sanitized retry observability, Google API error classification, and
//! Gemini `RetryInfo` delay hint parsing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use agentkit_loop::{
    LoopError, ProviderClassification, ProviderFailure, ProviderFailureReason, ProviderRetryEvent,
    ProviderRoute, RetryAccounting, RetryObserver, RetryProgress, UpstreamErrorKind,
};
use serde_json::Value;

/// One accumulator per logical turn; no background task or unbounded queue.
pub(crate) struct RetryTracker {
    pub(crate) started_at: Instant,
    pub(crate) accounting: RetryAccounting,
    upstream: ProviderClassification,
    last_attempt_reason: Option<ProviderFailureReason>,
    observer: Option<Arc<dyn RetryObserver>>,
    last_progress: Option<Duration>,
    finalized: bool,
}

impl RetryTracker {
    pub(crate) fn new(observer: Option<Arc<dyn RetryObserver>>) -> Self {
        Self {
            started_at: Instant::now(),
            accounting: RetryAccounting::default(),
            upstream: ProviderClassification::default(),
            last_attempt_reason: None,
            observer,
            last_progress: None,
            finalized: false,
        }
    }

    pub(crate) fn snapshot(&self) -> RetryAccounting {
        self.snapshot_at(self.started_at.elapsed())
    }

    fn snapshot_at(&self, elapsed: Duration) -> RetryAccounting {
        RetryAccounting {
            elapsed,
            ..self.accounting
        }
    }

    pub(crate) fn note_failure(&mut self, error: &LoopError) {
        let reason = failure_reason(error);
        if self.accounting.attempts > 0
            && (matches!(
                reason,
                ProviderFailureReason::HttpStatus
                    | ProviderFailureReason::Transport
                    | ProviderFailureReason::ResponseFailed
                    | ProviderFailureReason::Protocol
                    | ProviderFailureReason::AttemptTimeout
                    | ProviderFailureReason::IdleTimeout
                    | ProviderFailureReason::InvalidRequest
            ) || (reason == ProviderFailureReason::Authentication
                && error
                    .provider_failure()
                    .is_some_and(|failure| failure.upstream.http_status.is_some())))
        {
            self.last_attempt_reason = Some(reason);
        }
        if let Some(failure) = error.provider_failure()
            && (matches!(
                failure.reason,
                ProviderFailureReason::HttpStatus
                    | ProviderFailureReason::ResponseFailed
                    | ProviderFailureReason::InvalidRequest
            ) || (failure.reason == ProviderFailureReason::Authentication
                && failure.upstream.http_status.is_some()))
        {
            self.upstream = failure.upstream;
        }
    }

    pub(crate) fn scheduled(&mut self, error: &LoopError, delay: Duration) {
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
            route: ProviderRoute::Gemini,
            reason: failure_reason(error),
            upstream: self.upstream,
            accounting: self.snapshot_at(elapsed),
            next_delay: delay,
        }));
    }

    pub(crate) fn completed_wait(&mut self, delay: Duration) {
        self.accounting.completed_backoff = self.accounting.completed_backoff.saturating_add(delay);
    }

    pub(crate) fn finish(&mut self, error: LoopError) -> LoopError {
        self.note_failure(&error);
        let failure = ProviderFailure {
            route: ProviderRoute::Gemini,
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

    pub(crate) fn succeed(&mut self) {
        if !self.finalized {
            self.finalized = true;
            self.emit(ProviderRetryEvent::Succeeded {
                route: ProviderRoute::Gemini,
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

pub(crate) fn failure_reason(error: &LoopError) -> ProviderFailureReason {
    match error {
        LoopError::Cancelled => ProviderFailureReason::Cancelled,
        LoopError::ProviderFailure(failure) => failure.reason,
        _ => ProviderFailureReason::Protocol,
    }
}

pub(crate) fn provider_error(
    reason: ProviderFailureReason,
    upstream: ProviderClassification,
) -> LoopError {
    LoopError::ProviderFailure(Box::new(ProviderFailure {
        route: ProviderRoute::Gemini,
        reason,
        last_attempt_reason: None,
        upstream,
        accounting: RetryAccounting::default(),
    }))
}

pub(crate) fn local_error(reason: ProviderFailureReason) -> LoopError {
    provider_error(reason, ProviderClassification::default())
}

/// Maps Google API canonical `error.status` strings to allowlisted [`UpstreamErrorKind`]s.
fn google_status_kind(status: Option<&str>) -> UpstreamErrorKind {
    match status {
        Some("RESOURCE_EXHAUSTED") => UpstreamErrorKind::RateLimitExceeded,
        Some("UNAVAILABLE") => UpstreamErrorKind::ServiceUnavailableError,
        Some("INTERNAL") => UpstreamErrorKind::ServerError,
        Some("DEADLINE_EXCEEDED") => UpstreamErrorKind::TemporarilyUnavailable,
        Some("UNAUTHENTICATED") => UpstreamErrorKind::Unauthorized,
        Some("PERMISSION_DENIED") => UpstreamErrorKind::PermissionDenied,
        Some("INVALID_ARGUMENT") => UpstreamErrorKind::InvalidRequestError,
        _ => UpstreamErrorKind::Unknown,
    }
}

/// Produces a sanitized, payload-free [`ProviderClassification`] from a parsed Google API error JSON.
pub(crate) fn classify_value(value: &Value, status: Option<u16>) -> ProviderClassification {
    let error = value.get("error").unwrap_or(&Value::Null);
    let raw_status = error.get("status").and_then(Value::as_str);
    let error_type = google_status_kind(raw_status);
    let mut code = error_type;

    if raw_status == Some("RESOURCE_EXHAUSTED") && is_quota_failure(error) {
        code = UpstreamErrorKind::InsufficientQuota;
    }

    let http_status = status.or_else(|| {
        error
            .get("code")
            .and_then(Value::as_u64)
            .filter(|s| (100..=599).contains(s))
            .map(|s| s as u16)
    });

    if http_status == Some(400)
        && raw_status == Some("INVALID_ARGUMENT")
        && error
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(is_context_length_message)
    {
        code = UpstreamErrorKind::ContextLengthExceeded;
    }

    ProviderClassification {
        error_type,
        code,
        http_status,
    }
}

pub(crate) fn classify_http_body(body: Option<&str>, status: u16) -> ProviderClassification {
    if let Some(body) = body
        && let Ok(value) = serde_json::from_str::<Value>(body)
    {
        return classify_value(&value, Some(status));
    }
    ProviderClassification {
        error_type: match status {
            401 => UpstreamErrorKind::Unauthorized,
            403 => UpstreamErrorKind::PermissionDenied,
            429 => UpstreamErrorKind::RateLimitExceeded,
            500 => UpstreamErrorKind::ServerError,
            503 => UpstreamErrorKind::ServiceUnavailableError,
            _ => UpstreamErrorKind::Unknown,
        },
        code: UpstreamErrorKind::Unknown,
        http_status: Some(status),
    }
}

/// Recognizes only Gemini's structured `400 INVALID_ARGUMENT` context-length rejections.
/// Message text is inspected here but never retained in the typed failure.
pub(crate) fn context_length_error(body: &str, status: Option<u16>) -> Option<LoopError> {
    if status.is_some_and(|s| s != 400) {
        return None;
    }
    let value: Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    if error.get("status")?.as_str()? != "INVALID_ARGUMENT" {
        return None;
    }
    if let Some(code) = error.get("code").and_then(Value::as_u64)
        && code != 400
    {
        return None;
    }
    let message = error.get("message")?.as_str()?;
    if !is_context_length_message(message) {
        return None;
    }
    Some(LoopError::ProviderFailure(Box::new(ProviderFailure {
        route: ProviderRoute::Gemini,
        reason: ProviderFailureReason::InvalidRequest,
        last_attempt_reason: Some(ProviderFailureReason::InvalidRequest),
        upstream: ProviderClassification {
            error_type: UpstreamErrorKind::InvalidRequestError,
            code: UpstreamErrorKind::ContextLengthExceeded,
            http_status: status.or(Some(400)),
        },
        accounting: RetryAccounting::default(),
    })))
}

fn is_context_length_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    (lower.contains("input token count") && lower.contains("exceeds the maximum number of tokens"))
        || lower.contains("exceeds the maximum number of tokens allowed")
        || lower.contains("request payload size exceeds the limit")
}

fn is_quota_failure(error: &Value) -> bool {
    error
        .get("details")
        .and_then(Value::as_array)
        .is_some_and(|details| {
            details.iter().any(|detail| {
                detail.get("@type").and_then(Value::as_str)
                    == Some("type.googleapis.com/google.rpc.QuotaFailure")
            })
        })
}

/// Extracts `google.rpc.RetryInfo.retryDelay` (e.g. `"15s"` or `"2.5s"`) from a
/// Gemini JSON error envelope when HTTP headers do not carry `Retry-After`.
pub(crate) fn gemini_retry_hint(body: &str) -> Option<Duration> {
    let value: Value = serde_json::from_str(body).ok()?;
    let details = value.pointer("/error/details")?.as_array()?;
    for detail in details {
        if detail.get("@type").and_then(Value::as_str)
            == Some("type.googleapis.com/google.rpc.RetryInfo")
            && let Some(raw) = detail.get("retryDelay").and_then(Value::as_str)
            && let Some(delay) = parse_protobuf_duration(raw)
        {
            return Some(delay);
        }
    }
    None
}

fn parse_protobuf_duration(raw: &str) -> Option<Duration> {
    let seconds_str = raw.trim().strip_suffix('s')?;
    let seconds: f64 = seconds_str.parse().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    if seconds >= Duration::MAX.as_secs_f64() {
        return Some(Duration::MAX);
    }
    Duration::try_from_secs_f64(seconds).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_only_structured_400_context_length_without_retaining_payload() {
        let body = json!({
            "error": {
                "code": 400,
                "message": "The input token count (1632254) exceeds the maximum number of tokens allowed (1048576). SECRET-PROMPT",
                "status": "INVALID_ARGUMENT"
            }
        })
        .to_string();

        let err = context_length_error(&body, Some(400)).unwrap();
        let failure = err.provider_failure().unwrap();
        assert_eq!(failure.route, ProviderRoute::Gemini);
        assert_eq!(
            failure.upstream.code,
            UpstreamErrorKind::ContextLengthExceeded
        );
        assert!(!format!("{err:?} {err}").contains("SECRET-PROMPT"));

        for status in [401, 403, 429, 500] {
            assert!(context_length_error(&body, Some(status)).is_none());
        }
        for bad in [
            "The input token count exceeds the maximum number of tokens allowed",
            "{}",
            r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"The input token count exceeds the maximum number of tokens allowed"}}"#,
            r#"{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"Invalid JSON payload"}}"#,
        ] {
            assert!(context_length_error(bad, Some(400)).is_none());
        }
    }

    #[test]
    fn parses_google_rpc_retry_info_delay() {
        let body = json!({
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "12.5s"
                    }
                ]
            }
        })
        .to_string();
        assert_eq!(
            gemini_retry_hint(&body),
            Some(Duration::from_millis(12_500))
        );
    }

    #[test]
    fn retry_tracker_throttles_scheduled_and_emits_idempotent_terminal() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let capture = events.clone();
        let mut tracker = RetryTracker::new(Some(Arc::new(move |event| {
            capture.lock().unwrap().push(event);
        })));
        let err = local_error(ProviderFailureReason::Transport);
        for ms in [0, 100, 250, 400, 500] {
            tracker.accounting.attempts += 1;
            tracker.completed_wait(Duration::from_millis(10));
            tracker.scheduled_at(&err, Duration::from_secs(1), Duration::from_millis(ms));
        }
        let terminal = tracker.finish(local_error(ProviderFailureReason::RetryBudget));
        assert_eq!(terminal.provider_failure().unwrap().accounting.attempts, 5);
        assert_eq!(
            terminal.provider_failure().unwrap().route,
            ProviderRoute::Gemini
        );
        tracker.succeed(); // idempotent
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4); // 3 scheduled (0, 250, 500) + 1 stopped
        assert!(matches!(events[3], ProviderRetryEvent::Stopped(_)));
    }

    #[test]
    fn terminal_http_auth_rejection_replaces_prior_upstream_status() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let capture = events.clone();
        let mut tracker = RetryTracker::new(Some(Arc::new(move |event| {
            capture.lock().unwrap().push(event);
        })));
        tracker.accounting.attempts = 1;
        let prior = provider_error(
            ProviderFailureReason::HttpStatus,
            classify_http_body(None, 429),
        );
        tracker.scheduled(&prior, Duration::from_secs(1));
        tracker.accounting.attempts = 2;
        let terminal = tracker.finish(provider_error(
            ProviderFailureReason::Authentication,
            classify_http_body(None, 403),
        ));
        assert_eq!(
            terminal.provider_failure().unwrap().upstream.http_status,
            Some(403)
        );
        assert!(matches!(
            events.lock().unwrap().last(),
            Some(ProviderRetryEvent::Stopped(f)) if f.upstream.http_status == Some(403)
        ));
    }
}
