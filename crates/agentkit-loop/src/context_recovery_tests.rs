mod context_recovery {
    use super::*;

    #[derive(Clone, Copy)]
    enum FailureSite {
        Begin,
        EarlyStream,
        AfterDelta,
        AfterSuperseded,
        AfterTool,
        NonContext,
        Always,
        AfterTools,
        SlowRetry,
        SlowStream,
        SlowInitial,
    }

    struct RecoveryAdapter {
        executions: StdArc<AtomicUsize>,
        site: FailureSite,
        requests: StdArc<StdMutex<Vec<TurnRequest>>>,
    }
    struct RecoverySession {
        executions: StdArc<AtomicUsize>,
        site: FailureSite,
        requests: StdArc<StdMutex<Vec<TurnRequest>>>,
    }
    struct RecoveryTurn {
        hold_when_empty: bool,
        events: VecDeque<Result<ModelTurnEvent, LoopError>>,
    }

    fn rejection() -> LoopError {
        LoopError::ProviderFailure(Box::new(ProviderFailure {
            route: ProviderRoute::Unknown,
            reason: ProviderFailureReason::InvalidRequest,
            last_attempt_reason: None,
            upstream: ProviderClassification {
                code: UpstreamErrorKind::ContextLengthExceeded,
                ..Default::default()
            },
            accounting: RetryAccounting::default(),
        }))
    }
    fn finished() -> ModelTurnEvent {
        ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason: FinishReason::Completed,
            output_items: vec![Item::text(ItemKind::Assistant, "ok")],
            usage: None,
            metadata: MetadataMap::new(),
            model: None,
            response_id: None,
        })
    }
    #[async_trait]
    impl ModelAdapter for RecoveryAdapter {
        type Session = RecoverySession;
        async fn start_session(&self, _: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(RecoverySession {
                executions: self.executions.clone(),
                site: self.site,
                requests: self.requests.clone(),
            })
        }
    }
    #[async_trait]
    impl ModelSession for RecoverySession {
        type Turn = RecoveryTurn;
        async fn begin_turn(
            &mut self,
            request: TurnRequest,
            cancellation: Option<TurnCancellation>,
        ) -> Result<Self::Turn, LoopError> {
            let attempt = {
                let mut requests = self.requests.lock().unwrap();
                requests.push(request.clone());
                requests.len()
            };
            if matches!(self.site, FailureSite::AfterTools) {
                if attempt == 2 {
                    return Err(rejection());
                }
                let mut turn = FakeSession.begin_turn(request, cancellation).await?;
                let mut events = VecDeque::new();
                while let Some(event) = turn.next_event(None).await? {
                    events.push_back(Ok(event));
                }
                return Ok(RecoveryTurn {
                    events,
                    hold_when_empty: false,
                });
            }
            if attempt > 1 && !matches!(self.site, FailureSite::Always) {
                if matches!(self.site, FailureSite::SlowRetry) {
                    std::future::pending::<()>().await;
                }
                return Ok(RecoveryTurn {
                    hold_when_empty: matches!(self.site, FailureSite::SlowStream),
                    events: if matches!(self.site, FailureSite::SlowStream) {
                        VecDeque::from([Ok(ModelTurnEvent::Usage(Usage::default()))])
                    } else {
                        VecDeque::from([Ok(finished())])
                    },
                });
            }
            if matches!(self.site, FailureSite::SlowInitial) {
                tokio::time::sleep(Duration::from_millis(40)).await;
            }
            let events = match self.site {
                FailureSite::Begin
                | FailureSite::Always
                | FailureSite::SlowRetry
                | FailureSite::SlowStream
                | FailureSite::SlowInitial => {
                    return Err(rejection());
                }
                FailureSite::NonContext => {
                    return Err(LoopError::Provider("context_length_exceeded".into()));
                }
                FailureSite::EarlyStream => VecDeque::from([
                    Ok(ModelTurnEvent::Usage(Usage::default())),
                    Err(rejection()),
                ]),
                FailureSite::AfterDelta => VecDeque::from([
                    Ok(ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: PartId::new("text"),
                        chunk: "visible".into(),
                    })),
                    Err(rejection()),
                ]),
                FailureSite::AfterSuperseded => VecDeque::from([
                    Ok(ModelTurnEvent::Delta(Delta::AppendText {
                        part_id: PartId::new("text"),
                        chunk: "visible".into(),
                    })),
                    Ok(ModelTurnEvent::ResponseAttemptSuperseded),
                    Err(rejection()),
                ]),
                FailureSite::AfterTool => VecDeque::from([
                    Ok(ModelTurnEvent::ToolCall(ToolCallPart {
                        id: ToolCallId::new("unsafe"),
                        name: "echo".into(),
                        input: json!({}),
                        metadata: MetadataMap::new(),
                    })),
                    Err(rejection()),
                ]),
                FailureSite::AfterTools => unreachable!(),
            };
            Ok(RecoveryTurn {
                events,
                hold_when_empty: false,
            })
        }
    }
    #[async_trait]
    impl ModelTurn for RecoveryTurn {
        async fn next_event(
            &mut self,
            _: Option<TurnCancellation>,
        ) -> Result<Option<ModelTurnEvent>, LoopError> {
            if self.hold_when_empty && self.events.is_empty() {
                std::future::pending::<()>().await;
            }
            self.events.pop_front().transpose()
        }
    }

    #[derive(Clone, Copy)]
    enum Edit {
        Change,
        Noop,
        Hang,
        Invalid,
        Fail,
    }
    struct RecoveryMutator {
        deadline: StdMutex<Option<std::time::Instant>>,
        edit: Edit,
        calls: StdArc<AtomicUsize>,
    }
    #[async_trait]
    impl LoopMutator for RecoveryMutator {
        async fn mutate(
            &self,
            cursor: &mut TranscriptCursor<'_>,
            ctx: LoopCtx<'_>,
        ) -> Result<(), LoopError> {
            if ctx.point != MutationPoint::ContextLengthRejected {
                assert!(ctx.recovery.is_none());
                return Ok(());
            }
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            let info = ctx.recovery.unwrap();
            assert_eq!(info.attempt as usize, attempt);
            {
                let mut deadline = self.deadline.lock().unwrap();
                match *deadline {
                    Some(previous) => assert_eq!(
                        info.deadline, previous,
                        "all recovery attempts share one absolute deadline"
                    ),
                    None => *deadline = Some(info.deadline),
                }
            }
            if matches!(self.edit, Edit::Noop) {
                let copy = cursor.to_vec();
                **cursor = copy;
                return Ok(());
            }
            cursor.insert(
                0,
                Item::text(ItemKind::System, format!("recovered {attempt}")),
            );
            match self.edit {
                Edit::Hang => std::future::pending::<()>().await,
                Edit::Fail => return Err(LoopError::Mutator("failed".into())),
                Edit::Invalid => cursor.push(Item {
                    parts: vec![Part::ToolResult(ToolResultPart {
                        call_id: ToolCallId::new("orphan"),
                        output: ToolOutput::Text("bad".into()),
                        is_error: false,
                        metadata: MetadataMap::new(),
                    })],
                    ..Item::text(ItemKind::Tool, "")
                }),
                _ => {}
            }
            Ok(())
        }
    }
    struct CountingEcho {
        inner: EchoTool,
        count: StdArc<AtomicUsize>,
    }
    #[async_trait]
    impl Tool for CountingEcho {
        fn spec(&self) -> &ToolSpec {
            self.inner.spec()
        }
        async fn invoke(
            &self,
            request: ToolRequest,
            ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            self.count.fetch_add(1, Ordering::SeqCst);
            self.inner.invoke(request, ctx).await
        }
    }
    struct Replacements(StdArc<StdMutex<Vec<Vec<Item>>>>);
    impl TranscriptObserver for Replacements {
        fn on_transcript_event(&self, _: TranscriptEvent<'_>) {}
        fn on_transcript_replaced(&self, _: &SessionId, items: &[Item]) {
            self.0.lock().unwrap().push(items.to_vec());
        }
    }
    async fn driver(
        site: FailureSite,
        edit: Edit,
        attempts: u32,
        budget: Duration,
    ) -> (
        LoopDriver<RecoverySession>,
        StdArc<StdMutex<Vec<TurnRequest>>>,
        StdArc<AtomicUsize>,
        StdArc<StdMutex<Vec<Vec<Item>>>>,
        CancellationController,
    ) {
        let requests = StdArc::new(StdMutex::new(Vec::new()));
        let calls = StdArc::new(AtomicUsize::new(0));
        let replacements = StdArc::new(StdMutex::new(Vec::new()));
        let controller = CancellationController::default();
        let mut tools = ToolRegistry::default();
        let executions = StdArc::new(AtomicUsize::new(0));
        tools.register(CountingEcho {
            inner: EchoTool::default(),
            count: executions.clone(),
        });
        let agent = Agent::builder()
            .model(RecoveryAdapter {
                executions,
                site,
                requests: requests.clone(),
            })
            .add_tool_source(tools)
            .input(vec![Item::text(ItemKind::User, "hello")])
            .context_recovery(ContextRecoveryPolicy {
                max_attempts: attempts,
                budget,
            })
            .mutator(RecoveryMutator {
                deadline: StdMutex::new(None),
                edit,
                calls: calls.clone(),
            })
            .transcript_observer(Replacements(replacements.clone()))
            .cancellation(controller.handle())
            .build()
            .unwrap();
        (
            agent
                .start(SessionConfig::new("recovery").with_cache(PromptCacheRequest::automatic()))
                .await
                .unwrap(),
            requests,
            calls,
            replacements,
            controller,
        )
    }

    #[tokio::test]
    async fn begin_and_early_stream_recover_canonical_request() {
        for site in [FailureSite::Begin, FailureSite::EarlyStream] {
            let (mut driver, requests, calls, replacements, _) =
                driver(site, Edit::Change, 2, Duration::from_secs(1)).await;
            assert!(matches!(
                driver.next().await.unwrap(),
                LoopStep::Finished(_)
            ));
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(requests[0].turn_id, requests[1].turn_id);
            assert_eq!(requests[0].cache, requests[1].cache);
            assert_eq!(requests[0].available_tools, requests[1].available_tools);
            assert_eq!(replacements.lock().unwrap()[0], requests[1].transcript);
            assert!(driver.transcript.starts_with(&requests[1].transcript));
        }
    }
    #[tokio::test]
    async fn refuses_noop_noncontext_and_visible_output() {
        for (site, edit, attempts, expected_calls) in [
            (FailureSite::Begin, Edit::Noop, 2, 1),
            (FailureSite::NonContext, Edit::Change, 2, 0),
            (FailureSite::AfterDelta, Edit::Change, 2, 0),
            (FailureSite::AfterSuperseded, Edit::Change, 2, 0),
            (FailureSite::AfterTool, Edit::Change, 2, 0),
            (FailureSite::Begin, Edit::Change, 0, 0),
        ] {
            let (mut driver, requests, calls, replacements, _) =
                driver(site, edit, attempts, Duration::from_secs(1)).await;
            assert!(
                timeout(Duration::from_secs(2), driver.next())
                    .await
                    .expect("recovery hung")
                    .is_err()
            );
            assert_eq!(requests.lock().unwrap().len(), 1);
            assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            assert!(replacements.lock().unwrap().is_empty());
        }
    }
    #[tokio::test]
    async fn bounded_exhaustion_and_transactional_failures() {
        let (mut exhausted, requests, calls, _, _) =
            driver(FailureSite::Always, Edit::Change, 2, Duration::from_secs(1)).await;
        assert!(
            exhausted
                .next()
                .await
                .unwrap_err()
                .is_context_length_exceeded()
        );
        assert_eq!(requests.lock().unwrap().len(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        for edit in [Edit::Invalid, Edit::Fail, Edit::Hang] {
            let (mut driver, requests, _, replacements, _) =
                driver(FailureSite::Begin, edit, 2, Duration::from_millis(20)).await;
            assert!(
                timeout(Duration::from_secs(2), driver.next())
                    .await
                    .expect("recovery hung")
                    .is_err()
            );
            assert_eq!(driver.transcript, requests.lock().unwrap()[0].transcript);
            assert!(replacements.lock().unwrap().is_empty());
        }
    }
    #[tokio::test]
    async fn deadline_bounds_replacement_generation() {
        for site in [FailureSite::SlowRetry, FailureSite::SlowStream] {
            let (mut driver, requests, _, _, _) =
                driver(site, Edit::Change, 2, Duration::from_millis(20)).await;
            assert!(matches!(
                timeout(Duration::from_secs(2), driver.next())
                    .await
                    .expect("replacement generation hung"),
                Err(LoopError::ContextRecoveryDeadlineExceeded)
            ));
            assert_eq!(requests.lock().unwrap().len(), 2);
        }
    }
    #[tokio::test]
    async fn cancellation_interrupts_mutation_without_commit() {
        let (mut driver, requests, calls, replacements, controller) =
            driver(FailureSite::Begin, Edit::Hang, 2, Duration::from_secs(5)).await;
        let cancel = async {
            while calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
            controller.interrupt();
        };
        let (result, _) = timeout(Duration::from_secs(2), async {
            tokio::join!(driver.next(), cancel)
        })
        .await
        .expect("cancellation hung");
        assert!(matches!(
            result,
            Err(LoopError::Cancelled) | Ok(LoopStep::Finished(_))
        ));
        assert!(
            driver
                .transcript
                .starts_with(&requests.lock().unwrap()[0].transcript)
        );
        assert!(
            !driver
                .transcript
                .iter()
                .any(|item| item.kind == ItemKind::System)
        );
        assert!(replacements.lock().unwrap().is_empty());
    }
    #[tokio::test]
    async fn recovery_after_tools_does_not_replay_execution() {
        let (mut driver, requests, calls, _, _) = driver(
            FailureSite::AfterTools,
            Edit::Change,
            2,
            Duration::from_secs(1),
        )
        .await;
        let _ = run_until_finished(&mut driver).await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            driver
                .session
                .as_ref()
                .unwrap()
                .executions
                .load(Ordering::SeqCst),
            1
        );
        for request in &requests[1..] {
            assert_eq!(
                request
                    .transcript
                    .iter()
                    .filter(|item| item.kind == ItemKind::Tool)
                    .count(),
                1
            );
        }
        assert_eq!(
            driver
                .transcript
                .iter()
                .filter(|item| item.kind == ItemKind::Tool)
                .count(),
            1
        );
    }
    #[tokio::test]
    async fn budget_starts_at_rejection_and_default_remains_disabled() {
        let (mut delayed, _, _, _, _) = driver(
            FailureSite::SlowInitial,
            Edit::Change,
            2,
            Duration::from_millis(20),
        )
        .await;
        assert!(matches!(
            delayed.next().await.unwrap(),
            LoopStep::Finished(_)
        ));
        let (mut disabled, requests, calls, _, _) =
            driver(FailureSite::Begin, Edit::Change, 2, Duration::from_secs(1)).await;
        disabled.context_recovery = None;
        assert!(
            disabled
                .next()
                .await
                .unwrap_err()
                .is_context_length_exceeded()
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn classifier_accepts_only_typed_context_evidence() {
        assert!(rejection().is_context_length_exceeded());
        let mut failure = *rejection().provider_failure().unwrap();
        failure.upstream.code = UpstreamErrorKind::Unknown;
        failure.upstream.error_type = UpstreamErrorKind::ContextLengthExceeded;
        assert!(!LoopError::ProviderFailure(Box::new(failure)).is_context_length_exceeded());
        failure.upstream.error_type = UpstreamErrorKind::InvalidRequestError;
        failure.upstream.http_status = Some(400);
        assert!(!LoopError::ProviderFailure(Box::new(failure)).is_context_length_exceeded());
        assert!(
            !LoopError::Provider("maximum context length exceeded".into())
                .is_context_length_exceeded()
        );
        failure.upstream.code = UpstreamErrorKind::ContextLengthExceeded;
        for reason in [
            ProviderFailureReason::Cancelled,
            ProviderFailureReason::ReplayUnsafe,
        ] {
            failure.reason = reason;
            assert!(!LoopError::ProviderFailure(Box::new(failure)).is_context_length_exceeded());
        }
        assert_eq!(
            serde_json::to_string(&MutationPoint::ContextLengthRejected).unwrap(),
            "\"ContextLengthRejected\""
        );
    }
}
