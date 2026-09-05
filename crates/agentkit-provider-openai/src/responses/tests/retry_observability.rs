use super::*;
use agentkit_core::CancellationController;

const OVERLOADED: &str = r#"event: response.created
data: {"type":"response.created","sequence_number":1,"response":{"id":"private-id","model":"private-model"}}

event: response.failed
data: {"type":"response.failed","sequence_number":2,"response":{"error":{"type":"service_unavailable_error","code":"server_is_overloaded","message":"SECRET-PROMPT-CREDENTIAL"}}}

"#;

fn policy(retries: usize) -> ResilienceConfig {
    ResilienceConfig {
        max_retries: retries,
        retry_budget: Duration::from_secs(120),
        attempt_timeout: None,
        stream_idle_timeout: None,
        initial_backoff: Duration::ZERO,
        max_backoff: Duration::ZERO,
    }
}

fn wire(status: StatusCode, body: &'static str) -> WireResponse {
    WireResponse {
        status,
        headers: sse_headers(),
        body,
    }
}

async fn observed(
    config: OpenAIResponsesConfig,
    responses: Vec<WireResponse>,
) -> (
    OpenAIResponsesSession,
    Arc<Mutex<Vec<ProviderRetryEvent>>>,
    Arc<ScriptedClient>,
) {
    let client = Arc::new(ScriptedClient {
        responses: Mutex::new(responses.into()),
        requests: Mutex::new(Vec::new()),
    });
    let mut session = OpenAIResponsesAdapter::with_client(config, Http::from_arc(client.clone()))
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    session.set_retry_observer(Some(Arc::new(move |event| {
        capture.lock().unwrap().push(event)
    })));
    (session, events, client)
}

async fn finish(turn: &mut OpenAIResponsesTurn) -> Result<usize, LoopError> {
    let mut results = 0;
    while let Some(event) = turn.next_event(None).await? {
        if matches!(event, ModelTurnEvent::Finished(_)) {
            results += 1;
        }
    }
    Ok(results)
}

#[tokio::test]
async fn initial_retries_are_live_before_begin_turn_returns_and_cancelled_wait_is_not_completed() {
    let mut overloaded = wire(StatusCode::SERVICE_UNAVAILABLE, "SECRET-BODY");
    overloaded
        .headers
        .insert("retry-after", HeaderValue::from_static("60"));
    let (mut session, events, client) = observed(
        OpenAIResponsesConfig::new("SECRET-CREDENTIAL", "private-model").with_resilience(policy(1)),
        vec![overloaded],
    )
    .await;
    let controller = CancellationController::new();
    let cancellation = TurnCancellation::new(controller.handle());
    let future = session.begin_turn(request(), Some(cancellation));
    futures_util::pin_mut!(future);
    assert!(futures_util::poll!(future.as_mut()).is_pending());
    {
        let events = events.lock().unwrap();
        let ProviderRetryEvent::Scheduled(progress) = events[0] else {
            panic!("missing live retry")
        };
        assert_eq!(progress.accounting.attempts, 1);
        assert_eq!(progress.next_delay, Duration::from_secs(60));
        assert_eq!(progress.accounting.completed_backoff, Duration::ZERO);
        assert_eq!(progress.upstream.http_status, Some(503));
    }
    controller.interrupt();
    assert!(matches!(future.await, Err(LoopError::Cancelled)));
    let events = events.lock().unwrap();
    let ProviderRetryEvent::Stopped(failure) = events[1] else {
        panic!("missing cancellation summary")
    };
    assert_eq!(failure.reason, ProviderFailureReason::Cancelled);
    assert_eq!(failure.accounting.attempts, 1);
    assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
    assert_eq!(failure.upstream.http_status, Some(503));
    assert_eq!(events.len(), 2);
    assert_eq!(client.requests.lock().unwrap().len(), 1);
    assert!(!serde_json::to_string(&*events).unwrap().contains("SECRET"));
}

#[tokio::test]
async fn stream_exhaustion_preserves_two_classification_layers_and_is_finalized_once() {
    let (mut session, events, client) = observed(
        OpenAIResponsesConfig::chatgpt_private("private-model", "SECRET")
            .with_resilience(policy(3)),
        vec![wire(StatusCode::OK, OVERLOADED); 4],
    )
    .await;
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    let error = finish(&mut turn).await.unwrap_err();
    let failure = error.provider_failure().unwrap();
    assert_eq!(failure.reason, ProviderFailureReason::RetryExhausted);
    assert_eq!(
        failure.last_attempt_reason,
        Some(ProviderFailureReason::ResponseFailed)
    );
    assert_eq!(
        failure.upstream.error_type,
        UpstreamErrorKind::ServiceUnavailableError
    );
    assert_eq!(failure.upstream.code, UpstreamErrorKind::ServerIsOverloaded);
    assert_eq!(failure.accounting.attempts, 4);
    assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
    assert!(turn.next_event(None).await.unwrap().is_none());
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ProviderRetryEvent::Stopped(_)))
            .count(),
        1
    );
    assert_eq!(events.last(), Some(&ProviderRetryEvent::Stopped(*failure)));
    assert_eq!(client.requests.lock().unwrap().len(), 4);
    let rendered = format!(
        "{error:?} {error} {}",
        serde_json::to_string(&*events).unwrap()
    );
    for secret in ["SECRET", "private-id", "private-model", "message"] {
        assert!(!rendered.contains(secret));
    }
}

#[tokio::test]
async fn successful_retries_and_refresh_count_sends_and_clear_progress_once() {
    for status in [StatusCode::SERVICE_UNAVAILABLE, StatusCode::UNAUTHORIZED] {
        let (mut session, events, client) = observed(
            OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(1)),
            vec![wire(status, ""), wire(StatusCode::OK, SUCCESS)],
        )
        .await;
        let mut turn = session.begin_turn(request(), None).await.unwrap();
        assert_eq!(finish(&mut turn).await.unwrap(), 1);
        assert!(turn.next_event(None).await.unwrap().is_none());
        let events = events.lock().unwrap();
        assert!(matches!(
            events.first(),
            Some(ProviderRetryEvent::Scheduled(_))
        ));
        let Some(ProviderRetryEvent::Succeeded { accounting, .. }) = events.last() else {
            panic!("missing success")
        };
        assert_eq!(accounting.attempts, 2);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ProviderRetryEvent::Succeeded { .. }))
                .count(),
            1
        );
        assert_eq!(client.requests.lock().unwrap().len(), 2);
    }
}

#[tokio::test]
async fn already_cancelled_and_invalid_preflight_have_zero_sends() {
    for cancel in [false, true] {
        let (mut session, events, client) = observed(
            OpenAIResponsesConfig::new("secret", "gpt-test")
                .with_user_agent("invalid header value\n"),
            vec![],
        )
        .await;
        let controller = CancellationController::new();
        let cancellation = TurnCancellation::new(controller.handle());
        if cancel {
            controller.interrupt();
        }
        let result = session.begin_turn(request(), Some(cancellation)).await;
        assert!(result.is_err());
        let events = events.lock().unwrap();
        let ProviderRetryEvent::Stopped(failure) = events[0] else {
            panic!("missing stop")
        };
        assert_eq!(
            failure.reason,
            if cancel {
                ProviderFailureReason::Cancelled
            } else {
                ProviderFailureReason::InvalidRequest
            }
        );
        assert_eq!(failure.accounting.attempts, 0);
        assert_eq!(events.len(), 1);
        assert!(client.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn ready_cancellation_wins_without_polling_request() {
    let controller = CancellationController::new();
    let cancellation = TurnCancellation::new(controller.handle());
    controller.interrupt();
    let polled = AtomicUsize::new(0);
    let result = cancellable(
        async {
            polled.fetch_add(1, Ordering::SeqCst);
        },
        Some(&cancellation),
    )
    .await;
    assert!(matches!(result, Err(LoopError::Cancelled)));
    assert_eq!(polled.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancellation_of_deferred_stream_retry_preserves_supersession_order() {
    let partial = SUCCESS
        .split_once("event: response.output_text.done")
        .unwrap()
        .0;
    let (mut session, _, client) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(1)),
        vec![wire(StatusCode::OK, partial)],
    )
    .await;
    session.session = SessionConfig::new("session").with_response_attempt_supersession();
    let controller = Arc::new(CancellationController::new());
    let cancellation = TurnCancellation::new(controller.handle());
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    let interrupt = controller.clone();
    session.set_retry_observer(Some(Arc::new(move |event| {
        capture.lock().unwrap().push(event);
        if matches!(event, ProviderRetryEvent::Scheduled(_)) {
            interrupt.interrupt();
        }
    })));
    let mut turn = session
        .begin_turn(request(), Some(cancellation.clone()))
        .await
        .unwrap();
    loop {
        match turn.next_event(Some(cancellation.clone())).await {
            Err(LoopError::Cancelled) => break,
            Ok(Some(ModelTurnEvent::ResponseAttemptSuperseded)) => {
                panic!("cancelled retry must not escape as an event")
            }
            Ok(Some(_)) => {}
            _ => panic!("expected cancellation"),
        }
    }
    assert!(turn.next_event(None).await.unwrap().is_none());
    let events = events.lock().unwrap();
    let ProviderRetryEvent::Stopped(failure) = events[1] else {
        panic!("missing stop")
    };
    assert_eq!(failure.accounting.attempts, 1);
    assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
    assert_eq!(client.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn cancellation_wins_when_request_and_cancellation_become_ready_together() {
    use std::task::Poll;
    let controller = CancellationController::new();
    let cancellation = TurnCancellation::new(controller.handle());
    let mut polls = 0;
    let request = futures_util::future::poll_fn(|cx| {
        polls += 1;
        if polls == 1 {
            controller.interrupt();
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    });
    assert!(matches!(
        cancellable(request, Some(&cancellation)).await,
        Err(LoopError::Cancelled)
    ));
    assert_eq!(polls, 1);
}

#[derive(Clone)]
struct Capture(Arc<Mutex<Vec<agentkit_loop::ObservedEvent>>>);

impl agentkit_loop::LoopObserver for Capture {
    fn handle_event(&self, event: agentkit_loop::ObservedEvent) {
        self.0.lock().unwrap().push(event);
    }
}

#[tokio::test]
async fn loop_fans_out_initial_retries_with_isolated_session_accounting() {
    use agentkit_loop::{Agent, AgentEvent};
    let client = Arc::new(ScriptedClient {
        responses: Mutex::new(VecDeque::from([
            wire(StatusCode::SERVICE_UNAVAILABLE, ""),
            wire(StatusCode::BAD_REQUEST, ""),
            wire(StatusCode::SERVICE_UNAVAILABLE, ""),
            wire(StatusCode::BAD_REQUEST, ""),
        ])),
        requests: Mutex::new(Vec::new()),
    });
    let adapter = OpenAIResponsesAdapter::with_client(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(1)),
        Http::from_arc(client),
    );
    let first = Arc::new(Mutex::new(Vec::new()));
    let second = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder()
        .model(adapter)
        .input(vec![Item::text(ItemKind::User, "SECRET-PROMPT")])
        .observer(Capture(first.clone()))
        .observer(Capture(second.clone()))
        .build()
        .unwrap();
    for id in ["session-a", "session-b"] {
        let mut driver = agent.start(SessionConfig::new(id)).await.unwrap();
        let error = driver.next().await.unwrap_err();
        assert_eq!(error.provider_failure().unwrap().accounting.attempts, 2);
    }
    let first = first.lock().unwrap();
    assert_eq!(*first, *second.lock().unwrap());
    for id in ["session-a", "session-b"] {
        let retries: Vec<_> = first
            .iter()
            .filter(|event| event.session_id.0.as_str() == id)
            .filter_map(|event| {
                if let AgentEvent::ProviderRetry(event) = event.event {
                    Some(event)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(retries.len(), 2);
        let ProviderRetryEvent::Scheduled(progress) = retries[0] else {
            panic!("missing schedule")
        };
        assert_eq!(progress.accounting.attempts, 1);
        assert!(matches!(retries[1], ProviderRetryEvent::Stopped(_)));
    }
}

struct FailingAuthentication;

#[async_trait]
impl AuthenticationProvider for FailingAuthentication {
    async fn authenticate(
        &self,
        _: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        Err(HttpError::Other("SECRET-AUTH-DETAIL".into()))
    }
}

#[tokio::test]
async fn initial_auth_failure_has_typed_zero_send_summary_without_raw_source() {
    let (mut session, events, client) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test")
            .with_authentication_provider(FailingAuthentication),
        vec![],
    )
    .await;
    let error = match session.begin_turn(request(), None).await {
        Err(error) => error,
        Ok(_) => panic!("unexpected success"),
    };
    let failure = error.provider_failure().unwrap();
    assert_eq!(failure.reason, ProviderFailureReason::Authentication);
    assert_eq!(failure.accounting.attempts, 0);
    assert_eq!(failure.last_attempt_reason, None);
    assert_eq!(
        *events.lock().unwrap(),
        vec![ProviderRetryEvent::Stopped(*failure)]
    );
    assert!(client.requests.lock().unwrap().is_empty());
    assert!(!format!("{error:?} {error}").contains("SECRET"));
}

#[tokio::test]
async fn budget_interrupts_backoff_without_erasing_upstream_or_counting_planned_wait() {
    let mut response = wire(StatusCode::SERVICE_UNAVAILABLE, "SECRET");
    response
        .headers
        .insert("retry-after", HeaderValue::from_static("60"));
    let mut resilience = policy(2);
    resilience.retry_budget = Duration::from_millis(10);
    let (mut session, events, _) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(resilience),
        vec![response],
    )
    .await;
    let error = match session.begin_turn(request(), None).await {
        Err(error) => error,
        Ok(_) => panic!("unexpected success"),
    };
    let failure = error.provider_failure().unwrap();
    assert_eq!(failure.reason, ProviderFailureReason::RetryBudget);
    assert_eq!(failure.upstream.http_status, Some(503));
    assert_eq!(
        failure.last_attempt_reason,
        Some(ProviderFailureReason::HttpStatus)
    );
    assert_eq!(failure.accounting.attempts, 1);
    assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
    assert!(failure.accounting.elapsed >= Duration::from_millis(10));
    assert_eq!(
        events.lock().unwrap().last(),
        Some(&ProviderRetryEvent::Stopped(*failure))
    );
}

#[tokio::test]
async fn buffered_events_cannot_outrun_the_logical_budget() {
    let (mut session, events, _) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(1)),
        vec![wire(StatusCode::OK, SUCCESS)],
    )
    .await;
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    assert!(turn.next_event(None).await.unwrap().is_some());
    let deadline = turn.context.deadline.as_mut().unwrap();
    deadline.started_at = Instant::now() - deadline.budget;
    let error = turn.next_event(None).await.unwrap_err();
    assert_eq!(
        error.provider_failure().unwrap().reason,
        ProviderFailureReason::RetryBudget
    );
    assert!(matches!(
        events.lock().unwrap().as_slice(),
        [ProviderRetryEvent::Stopped(_)]
    ));
}

#[derive(Clone)]
struct InterruptRetry {
    events: Arc<Mutex<Vec<ProviderRetryEvent>>>,
    controller: Arc<CancellationController>,
}

impl agentkit_loop::LoopObserver for InterruptRetry {
    fn handle_event(&self, event: agentkit_loop::ObservedEvent) {
        if let agentkit_loop::AgentEvent::ProviderRetry(event) = event.event {
            self.events.lock().unwrap().push(event);
            if matches!(event, ProviderRetryEvent::Scheduled(_)) {
                self.controller.interrupt();
            }
        }
    }
}

#[tokio::test]
async fn driver_delivers_terminal_accounting_when_retry_observer_cancels() {
    let partial = SUCCESS
        .split_once("event: response.output_text.done")
        .unwrap()
        .0;
    let client = Arc::new(ScriptedClient {
        responses: Mutex::new(VecDeque::from([wire(StatusCode::OK, partial)])),
        requests: Mutex::new(Vec::new()),
    });
    let controller = Arc::new(CancellationController::new());
    let events = Arc::new(Mutex::new(Vec::new()));
    let adapter = OpenAIResponsesAdapter::with_client(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(1)),
        Http::from_arc(client.clone()),
    );
    let agent = agentkit_loop::Agent::builder()
        .model(adapter)
        .input(vec![Item::text(ItemKind::User, "hello")])
        .cancellation(controller.handle())
        .observer(InterruptRetry {
            events: events.clone(),
            controller,
        })
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("session").with_response_attempt_supersession())
        .await
        .unwrap();
    driver.next().await.unwrap();
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], ProviderRetryEvent::Scheduled(_)));
    let ProviderRetryEvent::Stopped(failure) = events[1] else {
        panic!("missing terminal accounting")
    };
    assert_eq!(failure.reason, ProviderFailureReason::Cancelled);
    assert_eq!(failure.accounting.attempts, 1);
    assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
    assert_eq!(client.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn explicit_driver_drop_hook_finalizes_without_another_poll() {
    let (mut session, events, _) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test"),
        vec![wire(StatusCode::OK, SUCCESS)],
    )
    .await;
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    turn.on_cancelled();
    turn.on_cancelled();
    assert!(turn.next_event(None).await.unwrap().is_none());
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1);
    let ProviderRetryEvent::Stopped(failure) = events[0] else {
        panic!("missing cancellation")
    };
    assert_eq!(failure.reason, ProviderFailureReason::Cancelled);
    assert_eq!(failure.accounting.attempts, 1);
}

#[tokio::test]
async fn completed_http_wait_survives_a_later_interrupted_wait() {
    let mut first = wire(StatusCode::SERVICE_UNAVAILABLE, "");
    first
        .headers
        .insert("retry-after", HeaderValue::from_static("0.001"));
    let mut second = first.clone();
    second
        .headers
        .insert("retry-after", HeaderValue::from_static("60"));
    let (mut session, events, client) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(2)),
        vec![first, second],
    )
    .await;
    let controller = CancellationController::new();
    let future = session.begin_turn(request(), Some(TurnCancellation::new(controller.handle())));
    futures_util::pin_mut!(future);
    // Poll the real HTTP retry path until the second wait starts. The only real
    // completed sleep is a server-directed 1ms; no elapsed-time equality is used.
    tokio::time::timeout(
        Duration::from_secs(2),
        futures_util::future::poll_fn(|cx| {
            assert!(future.as_mut().poll(cx).is_pending());
            if client.requests.lock().unwrap().len() == 2 {
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        }),
    )
    .await
    .unwrap();
    controller.interrupt();
    assert!(matches!(future.await, Err(LoopError::Cancelled)));
    let events = events.lock().unwrap();
    let Some(ProviderRetryEvent::Stopped(failure)) = events.last() else {
        panic!("missing stop")
    };
    assert_eq!(failure.accounting.attempts, 2);
    assert_eq!(
        failure.accounting.completed_backoff,
        Duration::from_millis(1)
    );
    assert!(failure.accounting.elapsed >= Duration::from_millis(1));
}

async fn superseded(turn: &mut OpenAIResponsesTurn) {
    loop {
        match turn.next_event(None).await.unwrap() {
            Some(ModelTurnEvent::ResponseAttemptSuperseded) => return,
            Some(_) => {}
            None => panic!("expected retry"),
        }
    }
}

#[tokio::test]
async fn completed_deferred_stream_wait_survives_a_later_interrupted_wait() {
    let partial = SUCCESS
        .split_once("event: response.output_text.done")
        .unwrap()
        .0;
    let (mut session, events, client) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(2)),
        vec![wire(StatusCode::OK, partial); 2],
    )
    .await;
    session.session = SessionConfig::new("session").with_response_attempt_supersession();
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    superseded(&mut turn).await;
    // Inject fixed delays at the existing deferred-wait seam rather than relying
    // on random full jitter. Both waits still use the real reopen implementation.
    turn.pending_delay = Duration::from_millis(1);
    superseded(&mut turn).await;
    turn.pending_delay = Duration::from_secs(60);
    let controller = CancellationController::new();
    let future = turn.next_event(Some(TurnCancellation::new(controller.handle())));
    futures_util::pin_mut!(future);
    assert!(futures_util::poll!(future.as_mut()).is_pending());
    controller.interrupt();
    assert!(matches!(future.await, Err(LoopError::Cancelled)));
    let events = events.lock().unwrap();
    let Some(ProviderRetryEvent::Stopped(failure)) = events.last() else {
        panic!("missing stop")
    };
    assert_eq!(failure.accounting.attempts, 2);
    assert_eq!(
        failure.accounting.completed_backoff,
        Duration::from_millis(1)
    );
    assert_eq!(client.requests.lock().unwrap().len(), 2);
}

struct PendingAuthentication {
    initial: bool,
}

#[async_trait]
impl AuthenticationProvider for PendingAuthentication {
    async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        if self.initial || previous.is_some() {
            futures_util::future::pending().await
        } else {
            Ok(AuthenticationAttempt::stateless(HeaderMap::new()))
        }
    }
}

#[tokio::test]
async fn cancellation_during_initial_auth_and_refresh_preserves_phase_accounting() {
    for initial in [true, false] {
        let (mut session, events, client) = observed(
            OpenAIResponsesConfig::new("secret", "gpt-test")
                .with_authentication_provider(PendingAuthentication { initial }),
            if initial {
                vec![]
            } else {
                vec![wire(StatusCode::UNAUTHORIZED, "")]
            },
        )
        .await;
        let controller = CancellationController::new();
        let future =
            session.begin_turn(request(), Some(TurnCancellation::new(controller.handle())));
        futures_util::pin_mut!(future);
        assert!(futures_util::poll!(future.as_mut()).is_pending());
        controller.interrupt();
        assert!(matches!(future.await, Err(LoopError::Cancelled)));
        let events = events.lock().unwrap();
        let Some(ProviderRetryEvent::Stopped(failure)) = events.last() else {
            panic!("missing stop")
        };
        assert_eq!(failure.accounting.attempts, if initial { 0 } else { 1 });
        assert_eq!(
            failure.upstream.http_status,
            if initial { None } else { Some(401) }
        );
        assert_eq!(
            client.requests.lock().unwrap().len(),
            if initial { 0 } else { 1 }
        );
    }
}

struct PendingClient(Arc<AtomicUsize>);

#[async_trait]
impl HttpClient for PendingClient {
    async fn execute(&self, _: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        futures_util::future::pending().await
    }
}

#[tokio::test]
async fn cancellation_during_send_counts_the_started_attempt() {
    let sends = Arc::new(AtomicUsize::new(0));
    let adapter = OpenAIResponsesAdapter::with_client(
        OpenAIResponsesConfig::new("secret", "gpt-test"),
        Http::new(PendingClient(sends.clone())),
    );
    let mut session = adapter
        .start_session(SessionConfig::new("session"))
        .await
        .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    session.set_retry_observer(Some(Arc::new(move |event| {
        capture.lock().unwrap().push(event)
    })));
    let controller = CancellationController::new();
    let future = session.begin_turn(request(), Some(TurnCancellation::new(controller.handle())));
    futures_util::pin_mut!(future);
    assert!(futures_util::poll!(future.as_mut()).is_pending());
    controller.interrupt();
    assert!(matches!(future.await, Err(LoopError::Cancelled)));
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 1);
    let ProviderRetryEvent::Stopped(failure) = events[0] else {
        panic!("missing stop")
    };
    assert_eq!(failure.accounting.attempts, 1);
    assert_eq!(sends.load(Ordering::SeqCst), 1);
}

#[test]
fn timeout_and_transport_source_classification_is_typed_and_sanitized() {
    for (error, expected) in [
        (
            HttpError::Timeout {
                operation: "response stream idle",
                timeout: Duration::ZERO,
            },
            ProviderFailureReason::IdleTimeout,
        ),
        (
            HttpError::Timeout {
                operation: "secret-transport-operation",
                timeout: Duration::ZERO,
            },
            ProviderFailureReason::AttemptTimeout,
        ),
        (
            HttpError::Other("SECRET-TRANSPORT-BODY".into()),
            ProviderFailureReason::Transport,
        ),
    ] {
        let failure = transport_failure(error);
        assert_eq!(failure.error.provider_failure().unwrap().reason, expected);
        assert!(!format!("{failure:?}").to_lowercase().contains("secret"));
    }
}

#[tokio::test]
async fn ready_finished_event_loses_to_an_expired_logical_deadline() {
    let (mut session, events, _) = observed(
        OpenAIResponsesConfig::new("secret", "gpt-test").with_resilience(policy(1)),
        vec![wire(StatusCode::OK, SUCCESS)],
    )
    .await;
    let mut turn = session.begin_turn(request(), None).await.unwrap();
    let attempt = turn.attempt.as_mut().unwrap();
    attempt.decoder.push(SUCCESS.as_bytes()).unwrap();
    attempt.decoder.process_all_pending().unwrap();
    while !matches!(
        attempt.decoder.peek_event(),
        Some(ModelTurnEvent::Finished(_))
    ) {
        assert!(attempt.decoder.pop_event().is_some());
    }
    attempt.closed = true; // EOF and final result are already ready.
    let deadline = turn.context.deadline.as_mut().unwrap();
    deadline.started_at = Instant::now() - deadline.budget;
    let error = turn.next_event(None).await.unwrap_err();
    assert_eq!(
        error.provider_failure().unwrap().reason,
        ProviderFailureReason::RetryBudget
    );
    assert!(matches!(
        events.lock().unwrap().as_slice(),
        [ProviderRetryEvent::Stopped(_)]
    ));
}

struct CountedReqwest {
    client: reqwest::Client,
    executions: Arc<AtomicUsize>,
}

#[async_trait]
impl HttpClient for CountedReqwest {
    async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        HttpClient::execute(&self.client, request).await
    }
}

#[tokio::test]
async fn invalid_endpoint_preflight_never_executes_or_retries_real_transport() {
    // Each endpoint is also rejected locally by reqwest if adapter preflight
    // regresses. The real transport cannot dispatch a network request here.
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    for endpoint in [
        "SECRET-ENDPOINT not a URL",
        "https://",
        "http://127.0.0.1:SECRET-ENDPOINT",
        "file:///SECRET-ENDPOINT",
        "ftp://127.0.0.1/SECRET-ENDPOINT",
    ] {
        let executions = Arc::new(AtomicUsize::new(0));
        let adapter = OpenAIResponsesAdapter::with_client(
            OpenAIResponsesConfig::new("SECRET-CREDENTIAL", "gpt-test")
                .with_endpoint(endpoint)
                .with_resilience(policy(2)),
            Http::new(CountedReqwest {
                client: client.clone(),
                executions: executions.clone(),
            }),
        );
        let mut session = adapter
            .start_session(SessionConfig::new("session"))
            .await
            .unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = events.clone();
        session.set_retry_observer(Some(Arc::new(move |event| {
            capture.lock().unwrap().push(event)
        })));
        let error = match session.begin_turn(request(), None).await {
            Err(error) => error,
            Ok(_) => panic!("invalid endpoint accepted"),
        };
        let failure = error.provider_failure().unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::InvalidRequest);
        assert_eq!(failure.last_attempt_reason, None);
        assert_eq!(failure.upstream, ProviderClassification::default());
        assert_eq!(failure.accounting.attempts, 0);
        assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
        assert_eq!(executions.load(Ordering::SeqCst), 0);
        let events = events.lock().unwrap();
        assert_eq!(*events, vec![ProviderRetryEvent::Stopped(*failure)]);
        let rendered = format!(
            "{error:?} {error} {}",
            serde_json::to_string(&*events).unwrap()
        );
        assert!(!rendered.contains("SECRET"));
    }
}

#[tokio::test]
async fn invalid_local_headers_preflight_have_zero_attempts_and_no_retry() {
    for invalid_session in [false, true] {
        let invalid = format!("SECRET-HEADER{}", char::from(10));
        let mut config = OpenAIResponsesConfig::chatgpt_private("gpt-test", "SECRET-CREDENTIAL")
            .with_resilience(policy(2));
        let mut request = request();
        if invalid_session {
            request.session_id = SessionId::new(invalid);
        } else {
            config = config.with_originator(invalid);
        }
        let (mut session, events, client) = observed(config, vec![]).await;
        let error = match session.begin_turn(request, None).await {
            Err(error) => error,
            Ok(_) => panic!("invalid local header accepted"),
        };
        let failure = error.provider_failure().unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::InvalidRequest);
        assert_eq!(failure.accounting.attempts, 0);
        assert_eq!(failure.accounting.completed_backoff, Duration::ZERO);
        assert_eq!(failure.last_attempt_reason, None);
        assert_eq!(
            *events.lock().unwrap(),
            vec![ProviderRetryEvent::Stopped(*failure)]
        );
        assert!(client.requests.lock().unwrap().is_empty());
        assert!(!format!("{error:?} {error}").contains("SECRET"));
    }
}
