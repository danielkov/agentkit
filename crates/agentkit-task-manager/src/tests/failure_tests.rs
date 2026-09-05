use super::*;
use agentkit_core::failure::{FailureCode, FailureMetadataV1, ObservationSource, PossibleEffects};
use agentkit_tools_core::{
    DiagnosticFailureKind, DiagnosticToolFailure, ObservationPublishError, ToolFailureKind,
};

fn native(cancelled: bool) -> ToolError {
    let mut effects = PossibleEffects::default();
    effects.source = ObservationSource::AcpNotifications;
    effects.assistant_output_observed = true;
    ToolError::diagnostic(DiagnosticToolFailure {
        kind: if cancelled {
            DiagnosticFailureKind::Cancelled
        } else {
            DiagnosticFailureKind::ExecutionFailed
        },
        metadata: FailureMetadataV1::new(FailureCode::ChildFailed).with_effects(effects),
    })
}
fn result(resolution: TaskResolution) -> ToolResultPart {
    let TaskResolution::Item(item) = resolution else {
        panic!("expected item")
    };
    item.parts
        .into_iter()
        .find_map(|part| match part {
            Part::ToolResult(result) => Some(result),
            _ => None,
        })
        .unwrap()
}
fn item_result(item: Item) -> ToolResultPart {
    result(TaskResolution::Item(item))
}

#[test]
fn failed_and_preinvocation_projections_replace_reserved_caller_keys() {
    for before in [false, true] {
        for error in [
            native(false),
            native(true),
            ToolError::Unavailable("legacy".into()),
        ] {
            let mut request = make_request("test", "turn", "call");
            request.metadata = [
                (
                    TOOL_RESULT_FAILURE_METADATA_KEY.into(),
                    json!({"kind":"permission_denied"}),
                ),
                (
                    TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY.into(),
                    json!({"effects":{"source":"local_session"}}),
                ),
                (
                    TOOL_RESULT_FAILURE_KIND_METADATA_KEY.into(),
                    json!("permission_denied"),
                ),
                (TOOL_RESULT_NOT_STARTED_METADATA_KEY.into(), json!(true)),
                ("application".into(), json!(42)),
            ]
            .into();
            let outcome = if before {
                ToolExecutionOutcome::FailedBeforeInvocation(error.clone())
            } else {
                ToolExecutionOutcome::Failed(error.clone())
            };
            let result = result(map_outcome_to_resolution(None, request, outcome));
            assert!(result.is_error);
            assert_eq!(
                tool_failure_info(&result).unwrap(),
                Some(error.failure_info())
            );
            assert_eq!(
                result.metadata.get(TOOL_RESULT_NOT_STARTED_METADATA_KEY),
                before.then_some(&json!(true))
            );
            assert!(
                !result
                    .metadata
                    .contains_key(TOOL_RESULT_FAILURE_KIND_METADATA_KEY)
            );
            assert!(task_failure_observations(&result).unwrap().is_none());
            assert_eq!(result.metadata["application"], json!(42));
        }
    }
}

#[tokio::test]
async fn failure_cancellation_and_abort_preserve_isolated_frozen_facts_in_all_delivery_modes() {
    for (background, manual) in [(false, false), (true, false), (true, true)] {
        for terminal in ["failed", "cancelled", "abort"] {
            let manager = AsyncTaskManager::new().routing(move |_: &ToolRequest| {
                if background {
                    RoutingDecision::Background
                } else {
                    RoutingDecision::Foreground
                }
            });
            let handle = manager.handle();
            let entered = StdArc::new(AtomicBool::new(false));
            let release = StdArc::new(Notify::new());
            let publisher = StdArc::new(std::sync::Mutex::new(None));
            let error = native(terminal == "cancelled");
            let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
                "observed",
                TestBehavior::ObserveBlock {
                    entered: entered.clone(),
                    release: release.clone(),
                    publisher: publisher.clone(),
                    outcome: ToolExecutionOutcome::Failed(error.clone()),
                },
            )]));
            let mut request = make_request("observed", "turn", "call");
            request
                .metadata
                .insert(TOOL_RESULT_FAILURE_METADATA_KEY.into(), json!("forged"));
            let template_slot = agentkit_tools_core::FailureObservationSlot::new();
            let mut context = make_context(executor, &request.turn_id, None);
            context.tool_context.failure_observer = Some(template_slot.publisher());
            let start = manager
                .start_task(TaskLaunchRequest::plain(None, request.clone()), context)
                .await
                .unwrap();
            let TaskStartOutcome::Pending { task_id, .. } = start else {
                panic!()
            };
            assert!(
                matches!(next_event(&handle).await, TaskEvent::Started(snapshot) if snapshot.failure.is_none() && !snapshot.metadata.contains_key(TOOL_RESULT_FAILURE_METADATA_KEY))
            );
            if manual {
                handle
                    .set_delivery_mode(task_id.clone(), DeliveryMode::Manual)
                    .await
                    .unwrap();
            }
            wait_until_entered(&entered).await;
            assert!(
                template_slot.snapshot().is_none(),
                "manager must replace caller publisher"
            );
            if terminal == "abort" {
                handle.cancel(task_id.clone()).await.unwrap();
            } else {
                release.notify_one();
            }
            let event = next_event(&handle).await;
            let snapshot = match event {
                TaskEvent::Failed(snapshot, actual) => {
                    assert_eq!(terminal, "failed");
                    assert_eq!(actual, error);
                    snapshot
                }
                TaskEvent::Cancelled(snapshot) => {
                    assert_ne!(terminal, "failed");
                    snapshot
                }
                other => panic!("wrong terminal {other:?}"),
            };
            let frozen = snapshot.failure_observations.clone().unwrap();
            assert_eq!(
                frozen.effects().unwrap().source,
                ObservationSource::LocalSession
            );
            assert!(frozen.effects().unwrap().tool_execution_start_reported);
            assert!(frozen.effects().unwrap().observation_incomplete());
            assert!(frozen.receipt().is_none());
            assert!(frozen.retry().is_none());
            if terminal == "abort" {
                assert_eq!(
                    snapshot.failure.as_ref().unwrap().kind,
                    ToolFailureKind::Cancelled
                );
                assert!(snapshot.failure.as_ref().unwrap().metadata.is_none());
            } else {
                assert_eq!(snapshot.failure, Some(error.failure_info()));
                assert_eq!(
                    snapshot
                        .failure
                        .as_ref()
                        .unwrap()
                        .metadata
                        .as_ref()
                        .unwrap()
                        .effects()
                        .unwrap()
                        .source,
                    ObservationSource::AcpNotifications
                );
            }
            let projected = if !background {
                let Some(TurnTaskUpdate::Resolution(resolution)) =
                    manager.wait_for_turn(&request.turn_id, None).await.unwrap()
                else {
                    panic!()
                };
                result(*resolution)
            } else if manual {
                let mut items = handle.drain_ready_items().await;
                assert_eq!(items.len(), 1);
                item_result(items.remove(0))
            } else {
                let mut updates = manager.take_pending_loop_updates().await.unwrap();
                assert_eq!(updates.resolutions.len(), 1);
                result(updates.resolutions.pop_front().unwrap())
            };
            assert_eq!(tool_failure_info(&projected).unwrap(), snapshot.failure);
            assert_eq!(task_failure_observations(&projected).unwrap(), Some(frozen));
            assert_eq!(
                publisher
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .publish_effects(PossibleEffects::default()),
                Err(ObservationPublishError::Sealed)
            );
            handle.cancel(task_id.clone()).await.unwrap();
            handle.cancel(task_id.clone()).await.unwrap();
            release.notify_one();
            assert!(
                timeout(Duration::from_millis(20), handle.next_event())
                    .await
                    .is_err()
            );
            assert!(handle.list_running().await.is_empty());
            assert_eq!(handle.list_completed().await.len(), 1);
        }
    }
}

#[tokio::test]
async fn success_after_observations_strips_failure_only_metadata_and_cannot_be_recancelled() {
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let entered = StdArc::new(AtomicBool::new(false));
    let release = StdArc::new(Notify::new());
    let publisher = StdArc::new(std::sync::Mutex::new(None));
    let request = make_request("observed", "turn", "call");
    let forged: MetadataMap = [
        TOOL_RESULT_FAILURE_METADATA_KEY,
        TOOL_RESULT_FAILURE_KIND_METADATA_KEY,
        TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY,
        TOOL_RESULT_NOT_STARTED_METADATA_KEY,
    ]
    .map(|key| (key.into(), json!(true)))
    .into();
    let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
        "observed",
        TestBehavior::ObserveBlock {
            entered: entered.clone(),
            release: release.clone(),
            publisher,
            outcome: ToolExecutionOutcome::Completed(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id.clone(),
                    output: ToolOutput::Text("ok".into()),
                    is_error: false,
                    metadata: forged.clone(),
                },
                duration: None,
                metadata: forged,
            }),
        },
    )]));
    manager
        .start_task(
            TaskLaunchRequest::plain(None, request.clone()),
            make_context(executor, &request.turn_id, None),
        )
        .await
        .unwrap();
    let TaskEvent::Started(started) = next_event(&handle).await else {
        panic!()
    };
    wait_until_entered(&entered).await;
    release.notify_one();
    let TaskEvent::Completed(snapshot, result) = next_event(&handle).await else {
        panic!()
    };
    assert!(snapshot.failure.is_none());
    assert!(snapshot.failure_observations.is_none());
    assert!(result.metadata.is_empty());
    handle.cancel(started.id).await.unwrap();
    assert!(
        timeout(Duration::from_millis(20), handle.next_event())
            .await
            .is_err()
    );
    let Some(TurnTaskUpdate::Resolution(resolution)) =
        manager.wait_for_turn(&request.turn_id, None).await.unwrap()
    else {
        panic!()
    };
    let TaskResolution::Item(item) = *resolution else {
        panic!()
    };
    assert!(item.metadata.is_empty());
}

#[tokio::test]
async fn interruption_exposes_frozen_real_result_once_for_loop_synthesis() {
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let entered = StdArc::new(AtomicBool::new(false));
    let release = StdArc::new(Notify::new());
    let publisher = StdArc::new(std::sync::Mutex::new(None));
    let request = make_request("observed", "turn", "call");
    let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
        "observed",
        TestBehavior::ObserveBlock {
            entered: entered.clone(),
            release,
            publisher,
            outcome: ToolExecutionOutcome::Failed(native(false)),
        },
    )]));
    manager
        .start_task(
            TaskLaunchRequest::plain(None, request.clone()),
            make_context(executor, &request.turn_id, None),
        )
        .await
        .unwrap();
    next_event(&handle).await;
    wait_until_entered(&entered).await;
    manager.on_turn_interrupted(&request.turn_id).await.unwrap();
    assert!(matches!(next_event(&handle).await, TaskEvent::Cancelled(_)));
    let mut updates =
        manager.take_interrupted_task_updates(&request.session_id, &[request.call_id.clone()]);
    let mut items: Vec<_> = updates
        .drain(..)
        .filter_map(|update| match update {
            TurnTaskUpdate::Resolution(resolution) => match *resolution {
                TaskResolution::Item(item) => Some(item),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(items.len(), 1);
    assert!(
        task_failure_observations(&item_result(items.remove(0)))
            .unwrap()
            .is_some()
    );
    assert!(
        manager
            .take_interrupted_task_updates(&request.session_id, &[request.call_id.clone()])
            .is_empty()
    );
    assert!(
        manager
            .wait_for_turn(&request.turn_id, None)
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn approved_generations_seal_old_publishers_and_reject_stale_detach() {
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let entered = StdArc::new(AtomicBool::new(false));
    let release = StdArc::new(Notify::new());
    let publisher = StdArc::new(std::sync::Mutex::new(None));
    let request = make_request("observed", "turn", "call");
    let approval = ApprovalRequest {
        task_id: None,
        call_id: Some(request.call_id.clone()),
        id: "approval:observed".into(),
        request_kind: "tool.test".into(),
        reason: ApprovalReason::SensitivePath,
        summary: "approve".into(),
        metadata: MetadataMap::new(),
    };
    let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
        "observed",
        TestBehavior::ObserveBlock {
            entered: entered.clone(),
            release: release.clone(),
            publisher: publisher.clone(),
            outcome: ToolExecutionOutcome::Interrupted(ToolInterruption::ApprovalRequired(
                approval.clone(),
            )),
        },
    )]));
    let context = make_context(executor, &request.turn_id, None);
    let TaskStartOutcome::Pending { task_id, .. } = manager
        .start_task(
            TaskLaunchRequest::plain(None, request.clone()),
            context.clone(),
        )
        .await
        .unwrap()
    else {
        panic!()
    };
    next_event(&handle).await;
    wait_until_entered(&entered).await;
    assert!(matches!(
        manager
            .start_task(
                TaskLaunchRequest::plain(Some(task_id.clone()), request.clone()),
                context.clone()
            )
            .await,
        Err(TaskManagerError::AlreadyRunning(_))
    ));
    let old_generation = manager.inner.state.lock().await.tasks[&task_id]
        .generation
        .clone();
    let old_publisher = publisher.lock().unwrap().clone().unwrap();
    release.notify_one();
    assert!(
        matches!(manager.wait_for_turn(&request.turn_id, None).await.unwrap(), Some(TurnTaskUpdate::Resolution(resolution)) if matches!(*resolution, TaskResolution::Approval(_)))
    );
    assert!(
        handle.list_completed().await.is_empty(),
        "approval is not terminal completion"
    );
    assert_eq!(
        old_publisher.publish_effects(PossibleEffects::default()),
        Err(ObservationPublishError::Sealed)
    );
    entered.store(false, AtomicOrdering::SeqCst);
    manager
        .start_task(
            TaskLaunchRequest::approved(Some(task_id.clone()), request.clone(), approval),
            context,
        )
        .await
        .unwrap();
    next_event(&handle).await;
    wait_until_entered(&entered).await;
    assert!(matches!(
        manager
            .inner
            .detach_running_foreground(&task_id, Some(&old_generation))
            .await,
        Err(TaskManagerError::NotRunning(_))
    ));
    handle.cancel(task_id).await.unwrap();
    assert!(matches!(next_event(&handle).await, TaskEvent::Cancelled(_)));
}

#[tokio::test]
async fn cancellation_before_executor_admission_marks_not_started_and_freezes_empty_slot() {
    // Deterministically exercise the state between record insertion and start-gate
    // admission; no spawned executor has permission to run in this state.
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let id = TaskId::new("pending-start");
    let slot = agentkit_tools_core::FailureObservationSlot::new();
    let snapshot = TaskSnapshot {
        id: id.clone(),
        turn_id: TurnId::new("turn"),
        call_id: ToolCallId::new("call"),
        tool_name: "test".into(),
        kind: TaskKind::Foreground,
        metadata: MetadataMap::new(),
        failure: None,
        failure_observations: None,
    };
    {
        let mut state = manager.inner.state.lock().await;
        state.per_turn_running.insert(snapshot.turn_id.clone(), 1);
        state.tasks.insert(
            id.clone(),
            TaskRecord {
                cancel_signal: None,
                suspended: None,
                session_id: SessionId::new("session"),
                generation: Arc::new(()),
                observations: slot.clone(),
                snapshot,
                continue_policy: ContinuePolicy::NotifyOnly,
                delivery_mode: DeliveryMode::ToLoop,
                running: true,
                invocation_admitted: false,
                completed: false,
                join: None,
            },
        );
    }
    handle.cancel(id).await.unwrap();
    let TaskEvent::Cancelled(snapshot) = next_event(&handle).await else {
        panic!()
    };
    assert!(snapshot.failure_observations.is_none());
    let Some(TurnTaskUpdate::Resolution(resolution)) = manager
        .wait_for_turn(&snapshot.turn_id, None)
        .await
        .unwrap()
    else {
        panic!()
    };
    let result = result(*resolution);
    assert_eq!(
        result.metadata[TOOL_RESULT_NOT_STARTED_METADATA_KEY],
        json!(true)
    );
    assert_eq!(
        slot.publisher().publish_effects(PossibleEffects::default()),
        Err(ObservationPublishError::Sealed)
    );
}

#[tokio::test]
async fn inline_drop_seals_publisher_and_preserves_one_cancellation_projection() {
    let manager = SimpleTaskManager::new();
    let handle = manager.handle();
    let entered = StdArc::new(AtomicBool::new(false));
    let release = StdArc::new(Notify::new());
    let publisher = StdArc::new(std::sync::Mutex::new(None));
    let request = make_request("observed", "turn", "inline-call");
    let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
        "observed",
        TestBehavior::ObserveBlock {
            entered: entered.clone(),
            release,
            publisher: publisher.clone(),
            outcome: ToolExecutionOutcome::Failed(native(false)),
        },
    )]));
    {
        let start = manager.start_task(
            TaskLaunchRequest::plain(None, request.clone()),
            make_context(executor, &request.turn_id, None),
        );
        tokio::pin!(start);
        tokio::select! { _ = wait_until_entered(&entered) => {}, result = &mut start => panic!("completed early: {result:?}") }
    }
    assert!(matches!(next_event(&handle).await, TaskEvent::Started(_)));
    let TaskEvent::Cancelled(snapshot) = next_event(&handle).await else {
        panic!()
    };
    assert!(
        snapshot
            .failure_observations
            .unwrap()
            .effects()
            .unwrap()
            .tool_execution_start_reported
    );
    assert_eq!(
        publisher
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .publish_effects(PossibleEffects::default()),
        Err(ObservationPublishError::Sealed)
    );
    assert!(handle.list_running().await.is_empty());
    assert!(
        manager
            .take_interrupted_task_updates(
                &SessionId::new("different-session"),
                &[request.call_id.clone()]
            )
            .is_empty()
    );
    assert_eq!(
        manager
            .take_interrupted_task_updates(&request.session_id, &[request.call_id.clone()])
            .len(),
        1
    );
    assert!(
        manager
            .take_interrupted_task_updates(&request.session_id, &[request.call_id])
            .is_empty()
    );
}

#[tokio::test]
async fn inline_approval_retains_facts_into_fresh_generation_and_suspended_cancel() {
    for resume in [false, true] {
        let manager = SimpleTaskManager::new();
        let handle = manager.handle();
        let entered = StdArc::new(AtomicBool::new(false));
        let release = StdArc::new(Notify::new());
        let publisher = StdArc::new(std::sync::Mutex::new(None));
        let request = make_request("observed", "turn", "inline-approval");
        let approval = ApprovalRequest {
            task_id: None,
            call_id: Some(request.call_id.clone()),
            id: "approval:inline".into(),
            request_kind: "tool.test".into(),
            reason: ApprovalReason::SensitivePath,
            summary: "approve".into(),
            metadata: MetadataMap::new(),
        };
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "observed",
            TestBehavior::ObserveBlock {
                entered: entered.clone(),
                release: release.clone(),
                publisher: publisher.clone(),
                outcome: ToolExecutionOutcome::Interrupted(ToolInterruption::ApprovalRequired(
                    approval,
                )),
            },
        )]));
        release.notify_one();
        let TaskStartOutcome::Ready(resolution) = manager
            .start_task(
                TaskLaunchRequest::plain(None, request.clone()),
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap()
        else {
            panic!()
        };
        let TaskResolution::Approval(task) = *resolution else {
            panic!()
        };
        next_event(&handle).await;
        assert!(handle.list_completed().await.is_empty());
        assert_eq!(handle.list_suspended().await.len(), 1);
        let old = publisher.lock().unwrap().clone().unwrap();
        assert_eq!(
            old.publish_effects(PossibleEffects::default()),
            Err(ObservationPublishError::Sealed)
        );
        if resume {
            entered.store(false, AtomicOrdering::SeqCst);
            let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
                "observed",
                TestBehavior::Block {
                    entered: entered.clone(),
                    release: StdArc::new(Notify::new()),
                    output: "not reached",
                },
            )]));
            let context = make_context(executor, &request.turn_id, None);
            let mut wrong = request.clone();
            wrong.session_id = SessionId::new("other");
            assert!(matches!(
                manager
                    .start_task(
                        TaskLaunchRequest::approved(
                            Some(task.task_id.clone()),
                            wrong,
                            task.approval.clone()
                        ),
                        context.clone()
                    )
                    .await,
                Err(TaskManagerError::InvalidContinuation(_))
            ));
            {
                let start = manager.start_task(
                    TaskLaunchRequest::approved(
                        Some(task.task_id.clone()),
                        request.clone(),
                        task.approval,
                    ),
                    context,
                );
                tokio::pin!(start);
                tokio::select! { _ = wait_until_entered(&entered) => {}, result = &mut start => panic!("completed early: {result:?}") }
            }
            assert!(matches!(next_event(&handle).await, TaskEvent::Started(_)));
        } else {
            handle.cancel(task.task_id.clone()).await.unwrap();
        }
        let TaskEvent::Cancelled(snapshot) = next_event(&handle).await else {
            panic!()
        };
        assert!(
            snapshot
                .failure_observations
                .unwrap()
                .effects()
                .unwrap()
                .tool_execution_start_reported
        );
        assert_eq!(
            manager
                .take_interrupted_task_updates(&request.session_id, &[request.call_id])
                .len(),
            1
        );
        handle.cancel(task.task_id).await.unwrap();
        assert!(
            timeout(Duration::from_millis(10), handle.next_event())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn dropping_launch_before_registration_cannot_strand_a_task() {
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let state = manager.inner.state.lock().await;
    let request = make_request("unknown", "turn", "pre-register");
    let executor: Arc<dyn ToolExecutor> =
        Arc::new(TestExecutor::new(Vec::<(String, TestBehavior)>::new()));
    {
        let start = manager.start_task(
            TaskLaunchRequest::plain(Some(TaskId::new("explicit")), request.clone()),
            make_context(executor, &request.turn_id, None),
        );
        tokio::pin!(start);
        std::future::poll_fn(|cx| {
            assert!(std::future::Future::poll(start.as_mut(), cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
    }
    assert!(state.tasks.is_empty());
    drop(state);
    assert!(handle.list_running().await.is_empty());
    assert!(
        timeout(Duration::from_millis(10), handle.next_event())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn concurrent_tasks_from_one_context_template_keep_distinct_receipts() {
    let manager = AsyncTaskManager::new();
    let handle = manager.handle();
    let entries: Vec<_> = ["first", "second"]
        .into_iter()
        .map(|name| {
            (
                name,
                StdArc::new(AtomicBool::new(false)),
                StdArc::new(std::sync::Mutex::new(None)),
            )
        })
        .collect();
    let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new(entries.iter().map(
        |(name, entered, publisher)| {
            (
                *name,
                TestBehavior::ObserveBlock {
                    entered: entered.clone(),
                    release: StdArc::new(Notify::new()),
                    publisher: publisher.clone(),
                    outcome: ToolExecutionOutcome::Failed(native(false)),
                },
            )
        },
    )));
    let context = make_context(executor, &TurnId::new("shared-turn"), None);
    let mut ids = Vec::new();
    for (name, _, _) in &entries {
        let request = make_request(name, "shared-turn", name);
        let TaskStartOutcome::Pending { task_id, .. } = manager
            .start_task(TaskLaunchRequest::plain(None, request), context.clone())
            .await
            .unwrap()
        else {
            panic!()
        };
        ids.push(task_id);
    }
    for (name, entered, publisher) in &entries {
        wait_until_entered(entered).await;
        publisher
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .publish_receipt(agentkit_core::failure::HostFatalReceipt {
                session_id: agentkit_core::failure::HostReceiptId::new("session").unwrap(),
                event_id: agentkit_core::failure::HostReceiptId::new(name).unwrap(),
                storage: agentkit_core::failure::FatalStorage::Unavailable,
            })
            .unwrap();
    }
    assert!(matches!(next_event(&handle).await, TaskEvent::Started(_)));
    assert!(matches!(next_event(&handle).await, TaskEvent::Started(_)));
    for id in ids {
        handle.cancel(id).await.unwrap();
    }
    for _ in 0..2 {
        let TaskEvent::Cancelled(snapshot) = next_event(&handle).await else {
            panic!()
        };
        assert_eq!(
            snapshot
                .failure_observations
                .unwrap()
                .receipt()
                .unwrap()
                .event_id
                .as_str(),
            snapshot.call_id.0
        );
    }
}
