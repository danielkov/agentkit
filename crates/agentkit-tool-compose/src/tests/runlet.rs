use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use tokio::sync::Barrier;

use agentkit_core::{MetadataMap, SessionId, ToolCallId, ToolOutput, ToolResultPart, TurnId};
use agentkit_tools_core::{
    AllowAllPermissions, BasicToolExecutor, PermissionChecker, Tool, ToolContext, ToolError,
    ToolExecutionOutcome, ToolExecutor, ToolInterruption, ToolName, ToolRegistry, ToolRequest,
    ToolResult, ToolSpec,
};
use serde_json::{Value, json};

use super::{ApprovalEchoTool, EchoTool, RequireApproval, owned_context};
use crate::{COMPOSE_TOOL_NAME, ComposeConfig, ComposeTool, RunletBackend};

fn request(script: &str, input: Value) -> ToolRequest {
    ToolRequest {
        call_id: ToolCallId::new("compose-call"),
        tool_name: ToolName::new(COMPOSE_TOOL_NAME),
        input: json!({ "script": script, "input": input }),
        session_id: SessionId::new("session"),
        turn_id: TurnId::new("turn"),
        metadata: MetadataMap::new(),
    }
}

async fn execute_compose(
    config: ComposeConfig,
    child: impl Tool + 'static,
    req: ToolRequest,
) -> ToolExecutionOutcome {
    let compose = ComposeTool::new(config).with_backend(RunletBackend);
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose).with(child),
    ));
    let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
    let mut ctx = owned.borrowed();
    executor.execute(req, &mut ctx).await
}

#[derive(Clone)]
struct OrderingProbeTool {
    spec: ToolSpec,
    parallel_barrier: Arc<Barrier>,
    events: Arc<StdMutex<Vec<String>>>,
}

impl OrderingProbeTool {
    fn new() -> Self {
        Self {
            // Default annotations intentionally make this an effectful,
            // at-most-once Runlet call rather than a pure read.
            spec: ToolSpec::new(
                "ordering_probe",
                "record compose scheduling",
                json!({"type": "object"}),
            ),
            parallel_barrier: Arc::new(Barrier::new(2)),
            events: Arc::new(StdMutex::new(Vec::new())),
        }
    }
}

#[async_trait::async_trait]
impl Tool for OrderingProbeTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let kind = request.input["kind"].as_str().unwrap_or_default();
        match kind {
            "parallel_a" | "parallel_b" => {
                tokio::time::timeout(Duration::from_secs(5), self.parallel_barrier.wait())
                    .await
                    .map_err(|_| {
                        ToolError::ExecutionFailed(
                            "independent effectful calls did not overlap".into(),
                        )
                    })?;
            }
            "prerequisite" => {
                self.events
                    .lock()
                    .expect("events lock")
                    .push("prerequisite:start".into());
                for _ in 0..100 {
                    tokio::task::yield_now().await;
                }
                self.events
                    .lock()
                    .expect("events lock")
                    .push("prerequisite:finish".into());
            }
            "after" => self
                .events
                .lock()
                .expect("events lock")
                .push("after:start".into()),
            _ => {}
        }

        Ok(ToolResult::new(ToolResultPart::success(
            request.call_id,
            ToolOutput::structured(request.input),
        )))
    }
}

#[tokio::test]
async fn effectful_calls_run_concurrently_and_after_orders_without_data_flow() {
    let child = OrderingProbeTool::new();
    let events = child.events.clone();
    let outcome = execute_compose(
        ComposeConfig::default(),
        child,
        request(
            r#"a = ordering_probe({ kind: "parallel_a" })
b = ordering_probe({ kind: "parallel_b" })
earlier = ordering_probe({ kind: "prerequisite" })
later = after earlier {
    return ordering_probe({ kind: "after" })
}
return [a.kind, b.kind, later.kind]"#,
            Value::Null,
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => assert_eq!(
            result.result.output,
            ToolOutput::structured(json!(["parallel_a", "parallel_b", "after"]))
        ),
        other => panic!("unexpected outcome: {other:?}"),
    }
    assert_eq!(
        *events.lock().expect("events lock"),
        vec!["prerequisite:start", "prerequisite:finish", "after:start"]
    );
}

#[tokio::test]
async fn converts_runlet_result_to_structured_json() {
    let outcome = execute_compose(
        ComposeConfig::default(),
        EchoTool::new(),
        request(
            "return { count: input.count + 1, label: \"ok\" }",
            json!({ "count": 2 }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "count": 3, "label": "ok" }))
            );
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn tool_call_dispatches_child_tool() {
    let child = EchoTool::new();
    let calls = child.calls.clone();
    let outcome = execute_compose(
        ComposeConfig::default(),
        child,
        request(
            "out = echo({ value: input.value })\nreturn out",
            json!({ "value": 7 }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "value": 7 }))
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn discard_eagerly_executes_a_pure_call() {
    let child = EchoTool::new();
    let calls = child.calls.clone();
    let outcome = execute_compose(
        ComposeConfig::default(),
        child,
        request("_ = echo({ value: 7 })\nreturn true", Value::Null),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(result.result.output, ToolOutput::structured(json!(true)));
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn loop_fans_out_and_preserves_order() {
    let child = EchoTool::new();
    let calls = child.calls.clone();
    let outcome = execute_compose(
        ComposeConfig::default(),
        child,
        request(
            "results = for item in input.items {\n\
                 detail = echo({ value: item })\n\
                 return detail.value\n\
             }\n\
             return results",
            json!({ "items": [1, 2, 3, 4, 5, 6] }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(
                result.result.output,
                ToolOutput::structured(json!([1, 2, 3, 4, 5, 6]))
            );
            assert_eq!(calls.load(Ordering::SeqCst), 6);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn fold_break_stops_with_the_final_accumulator() {
    let outcome = execute_compose(
        ComposeConfig::default(),
        EchoTool::new(),
        request(
            "result = fold state = { total: 0, count: 0 } for item in input.items {\n\
                 next = { total: state.total + item, count: state.count + 1 }\n\
                 break next if next.total >= input.limit\n\
                 return next\n\
             }\n\
             return result",
            json!({ "items": [2, 3, 5, 8], "limit": 5 }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => assert_eq!(
            result.result.output,
            ToolOutput::structured(json!({ "total": 5, "count": 2 }))
        ),
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn compile_diagnostics_surface_as_invalid_input() {
    let outcome = execute_compose(
        ComposeConfig::default(),
        EchoTool::new(),
        request("return missing_tool({ id: 1 })", Value::Null),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Failed(error) => {
            let message = error.to_string();
            assert!(
                message.contains("runlet program rejected"),
                "diagnostics should be model-repairable: {message}"
            );
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn nested_tool_call_limit_fails() {
    let outcome = execute_compose(
        ComposeConfig::default().with_max_nested_tool_calls(0),
        EchoTool::new(),
        request("return echo({ value: 1 })", Value::Null),
    )
    .await;

    assert!(matches!(outcome, ToolExecutionOutcome::Failed(_)));
}

#[tokio::test]
async fn compose_is_not_callable_from_runlet() {
    let outcome = execute_compose(
        ComposeConfig::default(),
        EchoTool::new(),
        request("return compose({ script: \"return 1\" })", Value::Null),
    )
    .await;

    // Compose filters itself out of the visible catalog, so the program
    // fails analysis with an unknown-tool diagnostic.
    assert!(matches!(outcome, ToolExecutionOutcome::Failed(_)));
}

#[tokio::test]
async fn nested_approval_interrupts_and_resumes_with_replay() {
    let (backend, mut progress_receiver) = progress_backend(1024);
    let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
    let first = EchoTool::new();
    let gated = ApprovalEchoTool::new();
    let first_calls = first.calls.clone();
    let gated_calls = gated.calls.clone();
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose).with(first).with(gated),
    ));
    let permissions: Arc<dyn PermissionChecker> = Arc::new(RequireApproval);
    let req = request(
        "a = echo({ value: 1 })\n\
         b = boundary { return approval_echo({ value: a.value + 1 }) } catch err { return { value: 0 } }\n\
         return b",
        Value::Null,
    );

    let owned = owned_context(executor.clone(), permissions.clone());
    let mut ctx = owned.borrowed();
    let first_outcome = executor.execute(req.clone(), &mut ctx).await;
    let approval = match first_outcome {
        ToolExecutionOutcome::Interrupted(ToolInterruption::ApprovalRequired(approval)) => approval,
        other => panic!("unexpected first outcome: {other:?}"),
    };
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gated_calls.load(Ordering::SeqCst), 0);

    let owned = owned_context(executor.clone(), permissions);
    let outcome = executor.execute_approved_owned(req, &approval, owned).await;
    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "value": 2 }))
            );
        }
        other => panic!("unexpected approved outcome: {other:?}"),
    }
    // The already-completed echo call replays from the record instead of
    // dispatching again; only the approved call executes.
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gated_calls.load(Ordering::SeqCst), 1);
    let mut first = progress_receiver.try_recv().unwrap();
    let mut replay = progress_receiver.try_recv().unwrap();
    assert_eq!(first.parent_call_id, replay.parent_call_id);
    assert_ne!(first.incarnation, replay.incarnation);
    assert_eq!(
        collect_progress(&mut first).1,
        crate::RunletProgressEnd::Interrupted
    );
    assert_eq!(
        collect_progress(&mut replay).1,
        crate::RunletProgressEnd::Succeeded
    );
}

#[tokio::test]
async fn computed_keys_and_object_merge_group_tool_output_into_maps() {
    let outcome = execute_compose(
        ComposeConfig::default(),
        EchoTool::new(),
        request(
            "r = echo({ value: input.rows })\n\
             by_team = fold acc = {} for p in r.value {\n\
                 return acc + { [p.team]: (acc[p.team] if p.team in acc else []) + [p.name] }\n\
             }\n\
             return by_team",
            json!({ "rows": [
                { "team": "eng", "name": "ada" },
                { "team": "ops", "name": "bo" },
                { "team": "eng", "name": "cy" }
            ] }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "eng": ["ada", "cy"], "ops": ["bo"] }))
            );
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn statement_form_if_auto_heals_and_reports_the_repair() {
    let child = EchoTool::new();
    let calls = child.calls.clone();
    // Python/JS muscle memory: statement-form if with a return-less body.
    // The healing pre-pass repairs it, the write dispatches, and the result
    // arrives wrapped with the repair notes.
    let outcome = execute_compose(
        ComposeConfig::default(),
        child,
        request(
            "flag = input.n > 1\n\
             if flag {\n\
                 r = echo({ value: input.n })\n\
             }\n\
             return flag",
            json!({ "n": 5 }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            let ToolOutput::Structured(value) = &result.result.output else {
                panic!("expected structured output: {:?}", result.result.output);
            };
            assert_eq!(value["value"], json!(true));
            let notes = value["compose_warnings"]["auto_repaired"]
                .as_array()
                .expect("repair notes present");
            assert!(!notes.is_empty());
            assert_eq!(calls.load(Ordering::SeqCst), 1, "the healed write ran");
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

#[tokio::test]
async fn prelude_intrinsics_and_folds_run_locally_without_consuming_call_budget() {
    let child = EchoTool::new();
    let calls = child.calls.clone();
    // Budget of 1: the single echo call fits; the intrinsic calls and the
    // fold reductions must not count against it.
    let outcome = execute_compose(
        ComposeConfig::default().with_max_nested_tool_calls(1),
        child,
        request(
            "r = echo({ value: input.values })\n\
             flat = fold acc = [] for page in r.value { return acc + page }\n\
             assert(list.length(flat) == 3, \"unexpected flattened length\")\n\
             return {\n\
                 total: fold t = 0 for x in flat { return t + x },\n\
                 count: fold n = 0 for x in flat { return n + 1 },\n\
                 label: text.upper(text.join([\"a\", \"b\"], \"-\"))\n\
             }",
            json!({ "values": [[1, 2], [3]] }),
        ),
    )
    .await;

    match outcome {
        ToolExecutionOutcome::Completed(result) => {
            assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "total": 6, "count": 3, "label": "A-B" }))
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
        other => panic!("unexpected outcome: {other:?}"),
    }
}

struct ProgressBackend {
    sink: tokio::sync::mpsc::Sender<crate::RunletProgress>,
    capacity: std::num::NonZeroUsize,
}

#[async_trait::async_trait]
impl crate::ComposeBackend for ProgressBackend {
    fn name(&self) -> &'static str {
        RunletBackend.name()
    }
    fn description(&self, catalog: Option<&[ToolSpec]>) -> String {
        RunletBackend.description(catalog)
    }
    fn script_description(&self) -> &'static str {
        RunletBackend.script_description()
    }
    async fn execute(&self, run: crate::BackendRun) -> Result<Value, crate::ComposeOutcome> {
        RunletBackend
            .execute_with_progress(run, self.sink.clone(), self.capacity)
            .await
    }
}

fn progress_backend(
    capacity: usize,
) -> (
    ProgressBackend,
    tokio::sync::mpsc::Receiver<crate::RunletProgress>,
) {
    let (sink, receiver) = tokio::sync::mpsc::channel(4);
    (
        ProgressBackend {
            sink,
            capacity: std::num::NonZeroUsize::new(capacity).unwrap(),
        },
        receiver,
    )
}

fn collect_progress(
    progress: &mut crate::RunletProgress,
) -> (Vec<runlet::ProgressEvent>, crate::RunletProgressEnd) {
    let mut events = Vec::new();
    loop {
        match progress.try_recv() {
            Ok(Some(event)) => events.push(event),
            Ok(None) => panic!("execution was already awaited"),
            Err(end) => return (events, end),
        }
    }
}

#[tokio::test]
async fn progress_namespaces_concurrent_parents_and_orders_repeated_calls() {
    let (backend, mut receiver) = progress_backend(1024);
    let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose).with(EchoTool::new()),
    ));
    let script = "a = echo({ value: 1 })\nb = echo({ value: a.value + 1 })\nreturn b";
    let run = |id: &'static str| {
        let executor = executor.clone();
        async move {
            let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
            let mut ctx = owned.borrowed();
            let mut req = request(script, Value::Null);
            req.call_id = ToolCallId::new(id);
            assert!(matches!(
                executor.execute(req, &mut ctx).await,
                ToolExecutionOutcome::Completed(_)
            ));
        }
    };
    tokio::join!(run("parent-a"), run("parent-b"));
    let mut a = receiver.try_recv().unwrap();
    let mut b = receiver.try_recv().unwrap();
    assert_ne!(a.parent_call_id, b.parent_call_id);
    assert!(
        (a.parent_call_id == ToolCallId::new("parent-a")
            && b.parent_call_id == ToolCallId::new("parent-b"))
            || (a.parent_call_id == ToolCallId::new("parent-b")
                && b.parent_call_id == ToolCallId::new("parent-a"))
    );
    assert_ne!(a.incarnation, b.incarnation);
    assert_eq!(a.source_digest, b.source_digest);
    assert!(!a.healed);
    for progress in [&mut a, &mut b] {
        let (events, end) = collect_progress(progress);
        assert_eq!(end, crate::RunletProgressEnd::Succeeded);
        let mut first_succeeded = None;
        let mut second_running = None;
        for event in events {
            if let runlet::ProgressChange::NodeUpdated(node) = event.change {
                if node.span.start == script.find("echo").unwrap()
                    && node.state == runlet::ProgressState::Succeeded
                {
                    first_succeeded = Some(event.sequence);
                }
                if node.span.start == script.rfind("echo").unwrap()
                    && node.state == runlet::ProgressState::Running
                {
                    second_running = Some(event.sequence);
                }
            }
        }
        assert!(first_succeeded.unwrap() < second_running.unwrap());
    }
}

#[tokio::test]
async fn progress_overflow_and_dropped_sink_do_not_fail_execution() {
    for drop_sink in [false, true] {
        let (backend, mut receiver) = progress_backend(1);
        if drop_sink {
            receiver.close();
        }
        let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose),
        ));
        let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
        let mut ctx = owned.borrowed();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            executor.execute(request("return 1 + 2", Value::Null), &mut ctx),
        )
        .await
        .unwrap();
        assert!(matches!(result, ToolExecutionOutcome::Completed(_)));
        if !drop_sink {
            let mut progress = receiver.try_recv().unwrap();
            assert_eq!(
                collect_progress(&mut progress).1,
                crate::RunletProgressEnd::Lagged
            );
        }
    }
}

#[tokio::test]
async fn progress_healing_identifies_compiled_not_submitted_source() {
    let (backend, mut receiver) = progress_backend(1024);
    let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose),
    ));
    let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
    let mut ctx = owned.borrowed();
    let script = "if true { x = 1 }\nreturn 2";
    assert!(matches!(
        executor
            .execute(request(script, Value::Null), &mut ctx)
            .await,
        ToolExecutionOutcome::Completed(_)
    ));
    let mut progress = receiver.try_recv().unwrap();
    assert!(progress.healed);
    let healed = runlet::heal(script).unwrap();
    let runtime = runlet::Runtime::builder().with_prelude().build().unwrap();
    assert_eq!(
        progress.source_digest,
        runtime.compile(&healed.source).unwrap().source_digest
    );
    assert_eq!(
        collect_progress(&mut progress).1,
        crate::RunletProgressEnd::Succeeded
    );
}

#[derive(Clone)]
struct ProgressGate {
    spec: ToolSpec,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    finished: Arc<tokio::sync::Notify>,
}

impl ProgressGate {
    fn new() -> Self {
        Self {
            spec: ToolSpec::new(
                "progress_gate",
                "hold at a real child boundary",
                json!({"type":"object"}),
            ),
            entered: Arc::new(tokio::sync::Notify::new()),
            release: Arc::new(tokio::sync::Notify::new()),
            finished: Arc::new(tokio::sync::Notify::new()),
        }
    }
}

#[async_trait::async_trait]
impl Tool for ProgressGate {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }
    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        self.entered.notify_one();
        self.release.notified().await;
        self.finished.notify_one();
        Ok(ToolResult::new(ToolResultPart::success(
            request.call_id,
            ToolOutput::structured(json!(null)),
        )))
    }
}

#[tokio::test]
async fn progress_future_drop_invalidates_and_consumer_drop_is_harmless() {
    for disposition in 0..3 {
        let abort = disposition == 2;
        let (backend, mut receiver) = progress_backend(1024);
        let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
        let gate = ProgressGate::new();
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose).with(gate.clone()),
        ));
        let task = tokio::spawn(async move {
            let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
            let mut ctx = owned.borrowed();
            executor
                .execute(request("return progress_gate({})", Value::Null), &mut ctx)
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
            .await
            .unwrap();
        let mut progress = receiver.try_recv().unwrap();
        assert!(matches!(progress.try_recv(), Ok(Some(_))));
        if abort {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            assert_eq!(
                progress.try_recv(),
                Err(crate::RunletProgressEnd::Incomplete)
            );
            gate.release.notify_one();
            // Explicitly release existing blocking execution, not an observer worker.
            tokio::time::timeout(Duration::from_secs(5), gate.finished.notified())
                .await
                .unwrap();
        } else {
            if disposition == 1 {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        let _consumer = progress;
                        panic!("consumer failure outside the executor");
                    }))
                    .is_err()
                );
            } else {
                drop(progress);
            }
            gate.release.notify_one();
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(5), task)
                    .await
                    .unwrap()
                    .unwrap(),
                ToolExecutionOutcome::Completed(_)
            ));
        }
    }
}

#[tokio::test]
async fn progress_external_cancellation_invalidates_before_runtime_finishes() {
    let (backend, mut receiver) = progress_backend(1024);
    let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
    let gate = ProgressGate::new();
    let controller = agentkit_core::CancellationController::new();
    let cancellation = controller.handle().checkpoint();
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose).with(gate.clone()),
    ));
    let task = tokio::spawn(async move {
        let mut owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
        owned.cancellation = Some(cancellation);
        let mut ctx = owned.borrowed();
        executor
            .execute(request("return progress_gate({})", Value::Null), &mut ctx)
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
        .await
        .unwrap();
    let mut progress = receiver.try_recv().unwrap();
    controller.interrupt();
    assert_eq!(
        progress.try_recv(),
        Err(crate::RunletProgressEnd::Incomplete)
    );
    gate.release.notify_one();
    let _ = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        progress.try_recv(),
        Err(crate::RunletProgressEnd::Incomplete)
    );
}

#[tokio::test]
async fn progress_full_host_sink_is_unobserved_and_events_are_payload_free() {
    let (backend, mut receiver) = progress_backend(1024);
    let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose).with(EchoTool::new()),
    ));
    for i in 0..5 {
        let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
        let mut ctx = owned.borrowed();
        let mut req = request(
            "return echo({ secret: input.secret })",
            json!({"secret":"private-input-output-marker"}),
        );
        req.call_id = ToolCallId::new(format!("bounded-parent-{i}"));
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(5), executor.execute(req, &mut ctx))
                .await
                .unwrap(),
            ToolExecutionOutcome::Completed(_)
        ));
    }
    for _ in 0..4 {
        let mut progress = receiver.try_recv().unwrap();
        let (events, end) = collect_progress(&mut progress);
        assert_eq!(end, crate::RunletProgressEnd::Succeeded);
        let serialized = serde_json::to_string(&events).unwrap();
        assert!(!serialized.contains("private-input-output-marker"));
        assert!(!serialized.contains("secret"));
        assert!(!serialized.contains("echo"));
    }
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn progress_failure_omits_runtime_error_text() {
    let (backend, mut receiver) = progress_backend(1024);
    let compose = ComposeTool::new(ComposeConfig::default()).with_backend(backend);
    let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
        ToolRegistry::new().with(compose),
    ));
    let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
    let mut ctx = owned.borrowed();
    let outcome = executor
        .execute(
            request(
                "return fail(\"PRIVATE_CODE\", \"private-error-marker\")",
                Value::Null,
            ),
            &mut ctx,
        )
        .await;
    assert!(!matches!(outcome, ToolExecutionOutcome::Completed(_)));
    let mut progress = receiver.try_recv().unwrap();
    let (events, end) = collect_progress(&mut progress);
    assert_eq!(end, crate::RunletProgressEnd::Failed);
    let serialized = serde_json::to_string(&events).unwrap();
    assert!(!serialized.contains("PRIVATE_CODE"));
    assert!(!serialized.contains("private-error-marker"));
}
