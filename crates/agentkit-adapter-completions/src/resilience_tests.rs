use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use agentkit_http::{
    Authentication, AuthenticationAttempt, AuthenticationProvider, Bytes, HeaderMap, HeaderValue,
    HttpClient, HttpError, HttpRequest, HttpResponse, ResilienceConfig, StatusCode,
    TruncatedStreamDetector, header,
};
use agentkit_loop::{ModelTurn, ModelTurnEvent};
use async_trait::async_trait;
use futures_util::stream;
use serde::Serialize;

use super::*;

fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWake(std::thread::Thread);
    impl Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[derive(Clone, Serialize)]
struct Config {
    model: &'static str,
}

#[derive(Clone)]
struct Provider;

impl CompletionsProvider for Provider {
    type Config = Config;

    fn provider_name(&self) -> &str {
        "test"
    }
    fn endpoint_url(&self) -> &str {
        "https://example.test/chat"
    }
    fn config(&self) -> &Self::Config {
        static CONFIG: Config = Config { model: "test" };
        &CONFIG
    }
    fn streaming(&self) -> bool {
        false
    }
}

struct SequenceClient {
    statuses: std::sync::Mutex<VecDeque<StatusCode>>,
    bodies: std::sync::Mutex<Vec<Bytes>>,
}

#[async_trait]
impl HttpClient for SequenceClient {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.bodies
            .lock()
            .unwrap()
            .push(request.body.unwrap_or_default());
        let status = self.statuses.lock().unwrap().pop_front().unwrap();
        Ok(HttpResponse::new(
            status,
            HeaderMap::new(),
            request.url,
            Box::pin(stream::empty()),
        ))
    }
}

struct RefreshingAuth {
    calls: Arc<AtomicUsize>,
}

struct NeverAuth;

#[async_trait]
impl AuthenticationProvider for NeverAuth {
    async fn authenticate(
        &self,
        _previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        futures_util::future::pending().await
    }
}

struct HangingClient;

#[async_trait]
impl HttpClient for HangingClient {
    async fn execute(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
        futures_util::future::pending().await
    }
}

#[async_trait]
impl AuthenticationProvider for RefreshingAuth {
    async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        let generation = self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            previous
                .and_then(|attempt| attempt.state::<usize>())
                .copied(),
            generation.checked_sub(1)
        );
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer redacted"),
        );
        Ok(AuthenticationAttempt::new(headers, generation))
    }
}

fn replay_request(
    client: Arc<SequenceClient>,
    authentication: Option<Authentication>,
    resilience: Option<ResilienceConfig>,
) -> ReplayRequest<Provider> {
    let deadline = resilience
        .as_ref()
        .map(|config| LogicalDeadline::new(config.retry_budget));
    ReplayRequest {
        client: Http::from_arc(client),
        provider: Arc::new(Provider),
        body: Bytes::from_static(b"stable request bytes"),
        authentication,
        authentication_attempt: Arc::new(std::sync::Mutex::new(None)),
        reauthenticated: Arc::new(AtomicBool::new(false)),
        resilience,
        deadline,
        retries_used: Arc::new(AtomicUsize::new(0)),
    }
}

#[test]
fn performs_only_one_reactive_401_reauthentication() {
    block_on(async {
        let client = Arc::new(SequenceClient {
            statuses: std::sync::Mutex::new(VecDeque::from([
                StatusCode::UNAUTHORIZED,
                StatusCode::UNAUTHORIZED,
                StatusCode::OK,
            ])),
            bodies: std::sync::Mutex::new(Vec::new()),
        });
        let auth_calls = Arc::new(AtomicUsize::new(0));
        let request = replay_request(
            client.clone(),
            Some(Authentication::new(RefreshingAuth {
                calls: auth_calls.clone(),
            })),
            None,
        );

        let response = request.open_response().await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(auth_calls.load(Ordering::SeqCst), 2);
        assert_eq!(client.bodies.lock().unwrap().len(), 2);
    });
}

#[test]
fn retries_retryable_status_with_identical_body_bytes() {
    block_on(async {
        let client = Arc::new(SequenceClient {
            statuses: std::sync::Mutex::new(VecDeque::from([
                StatusCode::SERVICE_UNAVAILABLE,
                StatusCode::OK,
            ])),
            bodies: std::sync::Mutex::new(Vec::new()),
        });
        let resilience = ResilienceConfig {
            max_retries: 1,
            retry_budget: Duration::from_secs(1),
            attempt_timeout: None,
            stream_idle_timeout: None,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        };
        let request = replay_request(client.clone(), None, Some(resilience));

        assert_eq!(
            request.open_response().await.unwrap().status(),
            StatusCode::OK
        );
        let bodies = client.bodies.lock().unwrap();
        assert_eq!(
            bodies.as_slice(),
            [
                Bytes::from_static(b"stable request bytes"),
                Bytes::from_static(b"stable request bytes")
            ]
        );
    });
}

#[test]
fn logical_retry_budget_bounds_authentication_attempt_body_backoff_and_stream() {
    block_on(async {
        let short = || ResilienceConfig {
            max_retries: 1,
            retry_budget: Duration::from_millis(20),
            attempt_timeout: Some(Duration::from_secs(10)),
            stream_idle_timeout: None,
            initial_backoff: Duration::from_secs(10),
            max_backoff: Duration::from_secs(10),
        };
        let assert_budget = |error: LoopError| {
            assert!(
                error.to_string().contains("logical request retry budget"),
                "unexpected error: {error}"
            );
        };

        let auth_client = Arc::new(SequenceClient {
            statuses: std::sync::Mutex::new(VecDeque::from([StatusCode::OK])),
            bodies: std::sync::Mutex::new(Vec::new()),
        });
        let auth_request = replay_request(
            auth_client,
            Some(Authentication::new(NeverAuth)),
            Some(short()),
        );
        assert_budget(auth_request.open_response().await.unwrap_err());

        let attempt_config = short();
        let attempt_request = ReplayRequest {
            client: Http::new(HangingClient),
            provider: Arc::new(Provider),
            body: Bytes::from_static(b"body"),
            authentication: None,
            authentication_attempt: Arc::new(std::sync::Mutex::new(None)),
            reauthenticated: Arc::new(AtomicBool::new(false)),
            deadline: Some(LogicalDeadline::new(attempt_config.retry_budget)),
            resilience: Some(attempt_config),
            retries_used: Arc::new(AtomicUsize::new(0)),
        };
        assert_budget(attempt_request.open_response().await.unwrap_err());

        let body_request = replay_request(
            Arc::new(SequenceClient {
                statuses: std::sync::Mutex::new(VecDeque::new()),
                bodies: std::sync::Mutex::new(Vec::new()),
            }),
            None,
            Some(short()),
        );
        let pending_body: BodyStream = Box::pin(stream::pending());
        let response = HttpResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            "https://example.test".into(),
            pending_body,
        );
        let body_error = body_request.read_response_text(response).await.unwrap_err();
        assert!(
            body_error
                .to_string()
                .contains("logical request retry budget")
        );

        let backoff_client = Arc::new(SequenceClient {
            statuses: std::sync::Mutex::new(VecDeque::from([
                StatusCode::SERVICE_UNAVAILABLE,
                StatusCode::OK,
            ])),
            bodies: std::sync::Mutex::new(Vec::new()),
        });
        let backoff_request = replay_request(backoff_client.clone(), None, Some(short()));
        assert_budget(backoff_request.open_response().await.unwrap_err());
        assert_eq!(backoff_client.bodies.lock().unwrap().len(), 1);

        let stream_deadline = LogicalDeadline::new(Duration::from_millis(20));
        let mut turn = CompletionsTurn::streaming(
            Box::pin(stream::pending()),
            Arc::new(|_, _, _| {}),
            None,
            true,
            Some(stream_deadline),
            TruncatedStreamDetector::default(),
            None,
        );
        assert_budget(turn.next_event(None).await.unwrap_err());
    });
}

#[test]
fn no_resilience_preserves_legacy_non_terminal_eof() {
    block_on(async {
        let mut turn = CompletionsTurn::streaming(
            Box::pin(stream::empty()),
            Arc::new(|_, _, _| {}),
            None,
            false,
            None,
            TruncatedStreamDetector::default(),
            None,
        );
        assert!(turn.next_event(None).await.unwrap().is_none());
    });
}

#[test]
fn stream_does_not_retry_non_transient_body_errors() {
    block_on(async {
        let replay_calls = Arc::new(AtomicUsize::new(0));
        let replay_counter = replay_calls.clone();
        let replay: StreamReplay = Box::new(move || {
            replay_counter.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { unreachable!("non-transient errors must not replay") })
        });
        let initial = stream::iter([Err(HttpError::Other("protocol failure".into()))]);
        let mut turn = CompletionsTurn::streaming(
            Box::pin(initial),
            Arc::new(|_, _, _| {}),
            None,
            true,
            Some(LogicalDeadline::new(Duration::from_secs(1))),
            TruncatedStreamDetector::default(),
            Some(replay),
        );

        let error = turn.next_event(None).await.unwrap_err();
        assert!(error.to_string().contains("protocol failure"));
        assert_eq!(replay_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn stream_retries_before_output_but_never_after_output() {
    block_on(async {
        let replay_calls = Arc::new(AtomicUsize::new(0));
        let replay_counter = replay_calls.clone();
        let replay: StreamReplay = Box::new(move || {
            replay_counter.fetch_add(1, Ordering::SeqCst);
            let body: BodyStream =
                Box::pin(stream::iter([Ok::<_, HttpError>(Bytes::from_static(
                    b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
                ))]));
            Box::pin(async move { Ok((body, TruncatedStreamDetector::default())) })
        });
        let initial = stream::iter([Err(HttpError::body(std::io::Error::other("disconnected")))]);
        let mut turn = CompletionsTurn::streaming(
            Box::pin(initial),
            Arc::new(|_, _, _| {}),
            None,
            true,
            Some(LogicalDeadline::new(Duration::from_secs(1))),
            TruncatedStreamDetector::default(),
            Some(replay),
        );

        assert!(matches!(
            turn.next_event(None).await.unwrap(),
            Some(ModelTurnEvent::Delta(_))
        ));
        assert_eq!(replay_calls.load(Ordering::SeqCst), 1);
        let mut failure = None;
        for _ in 0..4 {
            match turn.next_event(None).await {
                Err(error) => {
                    failure = Some(error);
                    break;
                }
                Ok(Some(_)) => {}
                Ok(None) => panic!("truncated stream was accepted"),
            }
        }
        assert!(failure.unwrap().to_string().contains("terminal event"));
        assert_eq!(replay_calls.load(Ordering::SeqCst), 1);
    });
}
