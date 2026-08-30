use std::future::Future;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_timer::Delay;
use futures_util::StreamExt;
use futures_util::future::{Either, select};
use http::{HeaderMap, StatusCode};

use crate::{BodyStream, HttpError};

/// Opt-in retry and timeout policy shared by HTTP-backed adapters.
/// `retry_budget` is the primary wall-clock budget for a logical request.
/// `max_retries` is an optional safety cap in addition to the initial try; the
/// default leaves the wall-clock budget in control.
/// `retry_budget` bounds the complete logical request, including authentication,
/// attempts, response reads, stream reads, and backoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResilienceConfig {
    pub max_retries: usize,
    pub retry_budget: Duration,
    pub attempt_timeout: Option<Duration>,
    pub stream_idle_timeout: Option<Duration>,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            max_retries: usize::MAX,
            retry_budget: Duration::from_secs(60),
            attempt_timeout: Some(Duration::from_secs(30)),
            stream_idle_timeout: Some(Duration::from_secs(30)),
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(10),
        }
    }
}

impl ResilienceConfig {
    pub fn no_retries() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Computes capped exponential full-jitter backoff, honoring server hints.
    /// `retry_number` starts at zero for the first retry.
    pub fn retry_delay(&self, retry_number: usize, headers: Option<&HeaderMap>) -> Duration {
        if let Some(hint) = headers.and_then(retry_hint) {
            // Server-directed waits are not exponential backoff. The logical
            // request deadline remains authoritative and bounds this delay.
            return hint.min(self.retry_budget);
        }
        let shift = retry_number.min(31) as u32;
        let cap = self
            .initial_backoff
            .saturating_mul(1_u32 << shift)
            .min(self.max_backoff);
        full_jitter(cap)
    }
}

/// HTTP statuses which are normally safe to retry when the adapter can replay.
pub fn is_retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

/// Whether a body-read failure can safely replay a successful response.
pub fn is_retryable_body_read(status: StatusCode, error: &HttpError) -> bool {
    status.is_success() && error.is_retryable_transport()
}

/// Parses Retry-After and common rate-limit reset headers into a delay.
pub fn retry_hint(headers: &HeaderMap) -> Option<Duration> {
    if let Some(value) = headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
    {
        if let Ok(seconds) = value.trim().parse::<f64>()
            && let Some(delay) = duration_from_seconds(seconds)
        {
            return Some(delay);
        }
        if let Ok(timestamp) = httpdate::parse_http_date(value) {
            return Some(
                timestamp
                    .duration_since(SystemTime::now())
                    .unwrap_or_default(),
            );
        }
    }
    for name in [
        "ratelimit-reset",
        "x-ratelimit-reset",
        "x-rate-limit-reset",
        "x-ratelimit-reset-requests-day",
        "x-ratelimit-reset-tokens-minute",
    ] {
        if let Some(value) = header_number(headers, name) {
            if value.is_nan() {
                continue;
            }
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            let seconds = if value > now - 60.0 {
                value - now
            } else {
                value
            };
            if let Some(delay) = duration_from_seconds(seconds) {
                return Some(delay);
            }
        }
    }
    None
}

fn duration_from_seconds(seconds: f64) -> Option<Duration> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    if seconds >= Duration::MAX.as_secs_f64() {
        return Some(Duration::MAX);
    }
    Duration::try_from_secs_f64(seconds).ok()
}

fn header_number(headers: &HeaderMap, name: &str) -> Option<f64> {
    headers
        .get(name)?
        .to_str()
        .ok()?
        .trim()
        .trim_end_matches('s')
        .parse()
        .ok()
}

fn full_jitter(cap: Duration) -> Duration {
    let cap_nanos = cap.as_nanos().min(u64::MAX as u128) as u64;
    Duration::from_nanos(fastrand::u64(..=cap_nanos))
}

/// Wall-clock deadline shared by all work for one logical HTTP request.
#[derive(Clone, Debug)]
pub struct LogicalDeadline {
    started_at: Instant,
    budget: Duration,
}

impl LogicalDeadline {
    pub fn new(budget: Duration) -> Self {
        Self {
            started_at: Instant::now(),
            budget,
        }
    }

    fn remaining(&self) -> Result<Duration, HttpError> {
        let elapsed = self.started_at.elapsed();
        if elapsed >= self.budget {
            Err(self.timeout_error())
        } else {
            Ok(self.budget - elapsed)
        }
    }

    fn timeout_error(&self) -> HttpError {
        HttpError::Timeout {
            operation: "logical request retry budget",
            timeout: self.budget,
        }
    }
}

/// Runs an HTTP operation within its own timeout and the logical request deadline.
pub async fn run_bounded<F, T>(
    future: F,
    operation_timeout: Option<Duration>,
    deadline: Option<&LogicalDeadline>,
    operation: &'static str,
) -> Result<T, HttpError>
where
    F: Future<Output = Result<T, HttpError>>,
{
    let remaining = deadline.map(LogicalDeadline::remaining).transpose()?;
    let timeout = match (operation_timeout, remaining) {
        (Some(operation), Some(remaining)) => operation.min(remaining),
        (Some(operation), None) => operation,
        (None, Some(remaining)) => remaining,
        (None, None) => return future.await,
    };
    let budget_limited = remaining
        .is_some_and(|remaining| operation_timeout.is_none_or(|operation| remaining <= operation));

    futures_util::pin_mut!(future);
    let timeout_wait = sleep(timeout);
    futures_util::pin_mut!(timeout_wait);
    match select(future, timeout_wait).await {
        Either::Left((result, _)) => result,
        Either::Right((_, _)) if budget_limited => Err(deadline
            .expect("budget-limited timeout has deadline")
            .timeout_error()),
        Either::Right((_, _)) => Err(HttpError::Timeout { operation, timeout }),
    }
}

/// Sleeps without tying the HTTP abstraction to a particular async runtime.
/// Dropping the returned future cancels the timer registration.
pub async fn sleep(duration: Duration) {
    Delay::new(duration).await;
}

/// Reads one body chunk and fails if the stream stays idle for `timeout`.
/// Dropping this future cancels the body poll and the caller's wait.
pub async fn next_body_chunk(
    body: &mut BodyStream,
    timeout: Option<Duration>,
) -> Result<Option<Bytes>, HttpError> {
    next_body_chunk_bounded(body, timeout, None).await
}

/// Reads one body chunk using a single timer for the idle and logical deadlines.
pub async fn next_body_chunk_bounded(
    body: &mut BodyStream,
    idle_timeout: Option<Duration>,
    deadline: Option<&LogicalDeadline>,
) -> Result<Option<Bytes>, HttpError> {
    let remaining = deadline.map(LogicalDeadline::remaining).transpose()?;
    let timeout = match (idle_timeout, remaining) {
        (Some(idle), Some(remaining)) => idle.min(remaining),
        (Some(idle), None) => idle,
        (None, Some(remaining)) => remaining,
        (None, None) => return body.next().await.transpose(),
    };
    let budget_limited =
        remaining.is_some_and(|remaining| idle_timeout.is_none_or(|idle| remaining <= idle));

    let next = body.next();
    futures_util::pin_mut!(next);
    let timeout_wait = sleep(timeout);
    futures_util::pin_mut!(timeout_wait);
    match select(next, timeout_wait).await {
        Either::Left((chunk, _)) => chunk.transpose(),
        Either::Right((_, _)) if budget_limited => Err(deadline
            .expect("budget-limited timeout has deadline")
            .timeout_error()),
        Either::Right((_, _)) => Err(HttpError::Timeout {
            operation: "response stream idle",
            timeout,
        }),
    }
}

/// Byte-count hook for detecting a body shorter than Content-Length.
#[derive(Clone, Debug, Default)]
pub struct TruncatedStreamDetector {
    expected: Option<u64>,
    received: u64,
}

impl TruncatedStreamDetector {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let expected = headers
            .get(http::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse().ok());
        Self {
            expected,
            received: 0,
        }
    }
    pub fn observe(&mut self, bytes: &Bytes) {
        self.received = self.received.saturating_add(bytes.len() as u64);
    }
    pub fn finish(&self) -> Result<(), HttpError> {
        if self
            .expected
            .is_some_and(|expected| self.received < expected)
        {
            return Err(HttpError::TruncatedBody {
                expected: self.expected.unwrap_or_default(),
                received: self.received,
            });
        }
        Ok(())
    }
}
