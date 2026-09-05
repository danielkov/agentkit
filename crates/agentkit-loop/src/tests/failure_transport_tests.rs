use super::*;
use agentkit_core::TurnId;
use agentkit_core::failure::{
    FailureCode, FailureMetadataV1, FatalStorage, HostFatalReceipt, HostReceiptId,
    ObservationSource, PossibleEffects,
};
use agentkit_tools_core::{DiagnosticFailureKind, DiagnosticToolFailure};

struct WinnerExecutor {
    mode: &'static str,
    entered: StdArc<AtomicBool>,
}
#[async_trait]
impl ToolExecutor for WinnerExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        vec![ToolSpec::new(
            "echo",
            "controlled terminal",
            json!({"type":"object"}),
        )]
    }
    async fn execute(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        let publisher = ctx.failure_observer().unwrap();
        let mut effects = PossibleEffects::default();
        effects.source = ObservationSource::LocalSession;
        effects.tool_execution_start_reported = true;
        publisher.publish_effects(effects).unwrap();
        publisher
            .publish_receipt(HostFatalReceipt {
                session_id: HostReceiptId::new("local-session").unwrap(),
                event_id: HostReceiptId::new("local-event").unwrap(),
                storage: FatalStorage::Unavailable,
            })
            .unwrap();
        publisher
            .publish_retry(ProviderFailure {
                route: ProviderRoute::Unknown,
                reason: ProviderFailureReason::Cancelled,
                last_attempt_reason: None,
                upstream: ProviderClassification::default(),
                accounting: RetryAccounting::default(),
            })
            .unwrap();
        self.entered.store(true, Ordering::SeqCst);
        if self.mode == "handle_cancel" {
            std::future::pending::<()>().await;
        }
        if self.mode == "approval" {
            return ToolExecutionOutcome::Interrupted(
                agentkit_tools_core::ToolInterruption::ApprovalRequired(ApprovalRequest {
                    task_id: None,
                    call_id: Some(request.call_id),
                    id: "approval:winner".into(),
                    request_kind: "tool.test".into(),
                    reason: agentkit_tools_core::ApprovalReason::SensitivePath,
                    summary: "approve".into(),
                    metadata: MetadataMap::new(),
                }),
            );
        }
        if self.mode == "success" {
            return ToolExecutionOutcome::Completed(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::Text("selected success".into()),
            )));
        }
        let mut child_effects = PossibleEffects::default();
        child_effects.source = ObservationSource::AcpNotifications;
        child_effects.assistant_output_observed = true;
        ToolExecutionOutcome::Failed(ToolError::diagnostic(DiagnosticToolFailure {
            kind: if self.mode == "native_cancel" {
                DiagnosticFailureKind::Cancelled
            } else {
                DiagnosticFailureKind::ExecutionFailed
            },
            metadata: FailureMetadataV1::new(FailureCode::ChildFailed).with_effects(child_effects),
        }))
    }
}

/// This wrapper does not return from start_task until a real terminal event is
/// enqueued, then cancels the turn before LoopDriver can call wait_for_turn.
struct CancelAfterSelection {
    inner: AsyncTaskManager,
    controller: CancellationController,
    selected: StdArc<StdMutex<Option<TaskEvent>>>,
    entered: StdArc<AtomicBool>,
    handle_cancel: bool,
}
#[async_trait]
impl TaskManager for CancelAfterSelection {
    async fn start_task(
        &self,
        request: TaskLaunchRequest,
        ctx: TaskStartContext,
    ) -> Result<TaskStartOutcome, TaskManagerError> {
        let outcome = self.inner.start_task(request, ctx).await?;
        let handle = self.inner.handle();
        let TaskEvent::Started(snapshot) = wait_for_task_event(&handle).await else {
            panic!()
        };
        if self.handle_cancel {
            wait_until_entered(&self.entered).await;
            handle.cancel(snapshot.id).await?;
        }
        *self.selected.lock().unwrap() = Some(wait_for_task_event(&handle).await);
        self.controller.interrupt();
        Ok(outcome)
    }
    async fn wait_for_turn(
        &self,
        turn: &TurnId,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, TaskManagerError> {
        self.inner.wait_for_turn(turn, cancellation).await
    }
    async fn take_pending_loop_updates(&self) -> Result<PendingLoopUpdates, TaskManagerError> {
        self.inner.take_pending_loop_updates().await
    }
    async fn on_turn_interrupted(&self, turn: &TurnId) -> Result<(), TaskManagerError> {
        self.inner.on_turn_interrupted(turn).await
    }
    fn take_interrupted_task_updates(
        &self,
        session: &SessionId,
        calls: &[ToolCallId],
    ) -> Vec<TurnTaskUpdate> {
        self.inner.take_interrupted_task_updates(session, calls)
    }
    fn handle(&self) -> TaskManagerHandle {
        self.inner.handle()
    }
}

#[tokio::test]
async fn real_loop_turn_cancel_keeps_already_selected_success_failure_and_cancellation() {
    for mode in ["success", "failed", "native_cancel", "handle_cancel"] {
        let controller = CancellationController::new();
        let selected = StdArc::new(StdMutex::new(None));
        let events = StdArc::new(StdMutex::new(Vec::new()));
        let entered = StdArc::new(AtomicBool::new(false));
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tool_executor(WinnerExecutor {
                mode,
                entered: entered.clone(),
            })
            .task_manager(CancelAfterSelection {
                inner: AsyncTaskManager::new(),
                controller: controller.clone(),
                selected: selected.clone(),
                entered,
                handle_cancel: mode == "handle_cancel",
            })
            .cancellation(controller.handle())
            .observer(RecordingObserver {
                events: events.clone(),
            })
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("winner-session"))
            .await
            .unwrap();
        driver
            .submit_input(vec![Item::text(ItemKind::User, "ping")])
            .unwrap();
        assert!(
            matches!(run_until_finished(&mut driver).await, LoopStep::Finished(turn) if turn.finish_reason == FinishReason::Cancelled)
        );
        let snapshot = driver.snapshot();
        validate_transcript_invariants(&snapshot.transcript).unwrap();
        let results: Vec<_> = snapshot
            .transcript
            .iter()
            .flat_map(|item| &item.parts)
            .filter_map(|part| match part {
                Part::ToolResult(result) => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 1, "{mode}");
        let event = selected.lock().unwrap().clone().unwrap();
        let task = match event {
            TaskEvent::Completed(task, selected) => {
                assert_eq!(mode, "success");
                assert_eq!(results[0], &selected);
                task
            }
            TaskEvent::Failed(task, _) => {
                assert_eq!(mode, "failed");
                task
            }
            TaskEvent::Cancelled(task) => {
                assert!(mode == "native_cancel" || mode == "handle_cancel");
                task
            }
            _ => panic!("not terminal"),
        };
        assert_eq!(
            agentkit_task_manager::tool_failure_info(results[0]).unwrap(),
            task.failure
        );
        assert_eq!(
            agentkit_task_manager::task_failure_observations(results[0]).unwrap(),
            task.failure_observations
        );
        if mode != "success" {
            let observations = task.failure_observations.unwrap();
            assert!(observations.receipt().is_some());
            assert!(observations.retry().is_some());
            assert!(
                observations
                    .effects()
                    .unwrap()
                    .tool_execution_start_reported
            );
        }
        assert_eq!(
            events
                .lock()
                .unwrap()
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolResultReceived(_)))
                .count(),
            1
        );
        assert!(
            driver
                .task_manager
                .take_interrupted_task_updates(&snapshot.session_id, &[ToolCallId::new("call-1")])
                .is_empty()
        );
    }
}

#[tokio::test]
async fn real_loop_pending_approval_cancel_prefers_frozen_facts_over_synthetic_result() {
    let controller = CancellationController::new();
    let events = StdArc::new(StdMutex::new(Vec::new()));
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let agent = Agent::builder()
        .model(FakeAdapter)
        .tool_executor(WinnerExecutor {
            mode: "approval",
            entered: StdArc::new(AtomicBool::new(false)),
        })
        .task_manager(manager)
        .cancellation(controller.handle())
        .observer(RecordingObserver {
            events: events.clone(),
        })
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("approval-session"))
        .await
        .unwrap();
    driver
        .submit_input(vec![Item::text(ItemKind::User, "ping")])
        .unwrap();
    assert!(matches!(
        driver.next().await.unwrap(),
        LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_))
    ));
    assert_eq!(handle.list_suspended().await.len(), 1);
    controller.interrupt();
    assert!(
        matches!(driver.next().await.unwrap(), LoopStep::Finished(turn) if turn.finish_reason == FinishReason::Cancelled)
    );
    let transcript = driver.snapshot().transcript;
    validate_transcript_invariants(&transcript).unwrap();
    let results: Vec<_> = transcript
        .iter()
        .flat_map(|item| &item.parts)
        .filter_map(|part| match part {
            Part::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 1);
    assert!(
        agentkit_task_manager::task_failure_observations(results[0])
            .unwrap()
            .unwrap()
            .effects()
            .unwrap()
            .tool_execution_start_reported
    );
    assert_eq!(
        agentkit_task_manager::tool_failure_info(results[0])
            .unwrap()
            .unwrap()
            .kind,
        agentkit_tools_core::ToolFailureKind::Cancelled
    );
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolResultReceived(_)))
            .count(),
        1
    );
}

#[tokio::test]
async fn individual_approval_closure_terminalizes_manager_and_preserves_facts() {
    for inline in [false, true] {
        for mode in ["cancel", "deny", "cancel_won", "approve_after_cancel"] {
            let manager: Arc<dyn TaskManager> = if inline {
                Arc::new(agentkit_task_manager::SimpleTaskManager::new())
            } else {
                Arc::new(AsyncTaskManager::new())
            };
            let handle = manager.handle();
            let mut builder = Agent::builder()
                .model(FakeAdapter)
                .tool_executor(WinnerExecutor {
                    mode: "approval",
                    entered: StdArc::new(AtomicBool::new(false)),
                });
            builder.task_manager = Some(manager);
            let agent = builder.build().unwrap();
            let mut driver = agent
                .start(SessionConfig::new("closure-session"))
                .await
                .unwrap();
            driver
                .submit_input(vec![Item::text(ItemKind::User, "ping")])
                .unwrap();
            let LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_)) =
                driver.next().await.unwrap()
            else {
                panic!()
            };
            let suspended = handle.list_suspended().await;
            assert_eq!(suspended.len(), 1);
            assert!(
                driver
                    .task_manager
                    .take_terminal_task_result(&suspended[0].id)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(handle.list_suspended().await.len(), 1);

            if mode == "cancel_won" || mode == "approve_after_cancel" {
                handle.cancel(suspended[0].id.clone()).await.unwrap();
            }
            if mode == "cancel" {
                driver
                    .cancel_pending_approval_for("call-1".into())
                    .await
                    .unwrap();
            } else {
                driver
                    .resolve_approval_for(
                        "call-1".into(),
                        if mode == "approve_after_cancel" {
                            ApprovalDecision::Approve
                        } else {
                            ApprovalDecision::Deny {
                                reason: Some("denied".into()),
                            }
                        },
                    )
                    .unwrap();
                driver.next().await.unwrap();
            }
            assert!(handle.list_suspended().await.is_empty());
            assert_eq!(handle.list_completed().await.len(), 1);
            assert!(
                driver
                    .task_manager
                    .take_terminal_task_result(&suspended[0].id)
                    .await
                    .unwrap()
                    .is_none()
            );

            let transcript = driver.snapshot().transcript;
            validate_transcript_invariants(&transcript).unwrap();
            let results: Vec<_> = transcript
                .iter()
                .flat_map(|item| &item.parts)
                .filter_map(|part| match part {
                    Part::ToolResult(result) => Some(result),
                    _ => None,
                })
                .collect();
            assert!(matches!(
                wait_for_task_event(&handle).await,
                TaskEvent::Started(_)
            ));
            let terminal = wait_for_task_event(&handle).await;
            let task = match &terminal {
                TaskEvent::Cancelled(task) | TaskEvent::Failed(task, _) => task,
                _ => panic!("expected terminal"),
            };
            assert_eq!(
                task.failure_observations,
                agentkit_task_manager::task_failure_observations(results[0]).unwrap()
            );
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(10), handle.next_event())
                    .await
                    .is_err()
            );
            assert_eq!(results.len(), 1);
            assert!(
                agentkit_task_manager::task_failure_observations(results[0])
                    .unwrap()
                    .is_some()
            );
            assert_eq!(
                agentkit_task_manager::tool_failure_info(results[0])
                    .unwrap()
                    .unwrap()
                    .kind,
                if mode == "deny" {
                    agentkit_tools_core::ToolFailureKind::ExecutionFailed
                } else {
                    agentkit_tools_core::ToolFailureKind::Cancelled
                }
            );
        }
    }
}

#[test]
fn synthetic_approval_metadata_rejects_reserved_request_facts() {
    let input = [
        (
            agentkit_task_manager::TOOL_RESULT_FAILURE_METADATA_KEY.into(),
            json!({"kind":"cancelled"}),
        ),
        (
            agentkit_task_manager::TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY.into(),
            json!({"receipt":"forged"}),
        ),
        ("application".into(), json!(1)),
    ]
    .into();
    assert_eq!(
        untrusted_call_metadata(input),
        [("application".into(), json!(1))].into()
    );
}
