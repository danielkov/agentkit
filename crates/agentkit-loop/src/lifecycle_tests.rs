use super::*;
use agentkit_tools_core::{ToolContext, ToolExecutionOutcome, ToolResult};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy)]
enum Script {
    Reply,
    Tool,
    BeginError,
    StreamError,
    MissingFinished,
    Cancelled,
}

#[derive(Clone)]
struct ProbeModel {
    scripts: Arc<Mutex<VecDeque<Script>>>,
    requests: Arc<Mutex<Vec<TurnRequest>>>,
    configs: Arc<Mutex<Vec<SessionConfig>>>,
}
impl ProbeModel {
    fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Arc::new(Mutex::new(scripts.into_iter().collect())),
            requests: Arc::default(),
            configs: Arc::default(),
        }
    }
}
struct ProbeSession(ProbeModel);
struct ProbeTurn(VecDeque<Result<ModelTurnEvent, LoopError>>);
#[async_trait]
impl ModelAdapter for ProbeModel {
    type Session = ProbeSession;
    async fn start_session(&self, config: SessionConfig) -> Result<ProbeSession, LoopError> {
        self.configs.lock().unwrap().push(config);
        Ok(ProbeSession(self.clone()))
    }
}
#[async_trait]
impl ModelSession for ProbeSession {
    type Turn = ProbeTurn;
    async fn begin_turn(
        &mut self,
        request: TurnRequest,
        _: Option<TurnCancellation>,
    ) -> Result<ProbeTurn, LoopError> {
        self.0.requests.lock().unwrap().push(request);
        let script = self
            .0
            .scripts
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected inference");
        let item = match script {
            Script::BeginError => return Err(LoopError::Provider("original begin error".into())),
            Script::Cancelled => return Err(LoopError::Cancelled),
            Script::StreamError => {
                return Ok(ProbeTurn(VecDeque::from([Err(LoopError::Provider(
                    "original stream error".into(),
                ))])));
            }
            Script::MissingFinished => return Ok(ProbeTurn(VecDeque::new())),
            Script::Reply => Item::text(ItemKind::Assistant, "reply").with_id("message-reply"),
            Script::Tool => Item::new(
                ItemKind::Assistant,
                vec![Part::ToolCall(ToolCallPart::new(
                    "call-probe",
                    "probe",
                    serde_json::json!({"value": 1}),
                ))],
            )
            .with_id("message-tool"),
        };
        let finish_reason = if matches!(script, Script::Tool) {
            FinishReason::ToolCall
        } else {
            FinishReason::Completed
        };
        let mut events = VecDeque::new();
        if let Some(Part::ToolCall(call)) = item.parts.first() {
            events.push_back(Ok(ModelTurnEvent::ToolCall(call.clone())));
        }
        events.push_back(Ok(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason,
            output_items: vec![item],
            usage: None,
            metadata: MetadataMap::new(),
            model: Some("probe-model".into()),
            response_id: Some("response-probe".into()),
        })));
        Ok(ProbeTurn(events))
    }
}
#[async_trait]
impl ModelTurn for ProbeTurn {
    async fn next_event(
        &mut self,
        _: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        self.0.pop_front().transpose()
    }
}

struct ProbeExecutor;
#[async_trait]
impl ToolExecutor for ProbeExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        Vec::new()
    }
    async fn execute(&self, request: ToolRequest, _: &mut ToolContext<'_>) -> ToolExecutionOutcome {
        ToolExecutionOutcome::Completed(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Text("tool-result".into()),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: None,
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone)]
struct ProbeHook {
    label: &'static str,
    events: Arc<Mutex<Vec<String>>>,
    fail: Option<&'static str>,
    failed: Arc<AtomicBool>,
    break_links: bool,
}
impl ProbeHook {
    fn new(label: &'static str, events: &Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            label,
            events: events.clone(),
            fail: None,
            failed: Arc::new(AtomicBool::new(false)),
            break_links: false,
        }
    }
    fn event(&self, name: &str, ctx: &LifecycleCtx<'_>) -> Result<(), LoopError> {
        self.events.lock().unwrap().push(format!(
            "{}:{name}:{}:{}",
            self.label,
            ctx.turn_id.map(|id| id.0.as_str()).unwrap_or("-"),
            ctx.model_call_index
                .map(|i| i.to_string())
                .unwrap_or_else(|| "-".into())
        ));
        if self.fail == Some(name) && !self.failed.swap(true, Ordering::SeqCst) {
            Err(LoopError::Mutator(format!("{} failed {name}", self.label)))
        } else {
            Ok(())
        }
    }
    fn rewrite(&self, output: &mut [OutputItemPayload<'_>], suffix: &str) {
        for item in output {
            for part in item.parts.iter_mut() {
                if let Part::Text(text) = part {
                    text.text.push_str(&format!(":{suffix}{}", self.label));
                }
            }
        }
    }
}
#[async_trait]
impl LoopMutator for ProbeHook {
    async fn on_session_start(
        &self,
        payload: SessionStartPayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        payload.metadata.insert(self.label.into(), true.into());
        self.event("session_start", &ctx)
    }
    async fn on_session_end(
        &self,
        _: SessionEndPayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        self.event("session_end", &ctx)
    }
    async fn on_input(
        &self,
        input: &mut Vec<Item>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        for item in input {
            for part in &mut item.parts {
                if let Part::Text(text) = part {
                    text.text.push_str(self.label);
                }
            }
        }
        self.event("input", &ctx)
    }
    async fn on_turn_start(
        &self,
        _: &mut TranscriptCursor<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        self.event("turn_start", &ctx)
    }
    async fn on_turn_finish(
        &self,
        mut payload: TurnFinishPayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        if let Some(output) = &mut payload.output {
            self.rewrite(output, "f");
        }
        if matches!(
            payload.result.finish_reason,
            FinishReason::Error | FinishReason::Cancelled
        ) {
            assert!(payload.output.is_none());
        }
        self.event("turn_finish", &ctx)
    }
    async fn on_turn_end(&self, _: &TurnResult, ctx: LifecycleCtx<'_>) -> Result<(), LoopError> {
        self.event("turn_end", &ctx)
    }
    async fn on_model_request(
        &self,
        payload: ModelRequestPayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        payload.metadata.insert(self.label.into(), true.into());
        self.event("request", &ctx)
    }
    async fn on_model_response(
        &self,
        mut payload: ModelResponsePayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        self.rewrite(&mut payload.output, "m");
        if self.break_links {
            for item in payload.output {
                for part in item.parts.iter_mut() {
                    if let Part::ToolCall(call) = part {
                        call.id = ToolCallId::new("forged");
                    }
                }
            }
        }
        self.event("response", &ctx)
    }
    async fn on_model_progress(
        &self,
        _: &ModelTurnEvent,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        self.event("progress", &ctx)
    }
    async fn on_model_error(&self, _: &LoopError, ctx: LifecycleCtx<'_>) -> Result<(), LoopError> {
        self.event("model_error", &ctx)
    }
    async fn on_tool_batch(
        &self,
        payload: ToolBatchPayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        assert_eq!(payload.calls.len(), 1);
        assert_eq!(payload.results.len(), 1);
        assert_eq!(payload.calls[0].id, payload.results[0].call_id);
        self.event("batch", &ctx)
    }
}
struct Capture(Arc<Mutex<Vec<AgentEvent>>>);
impl LoopObserver for Capture {
    fn handle_event(&self, event: ObservedEvent) {
        self.0.lock().unwrap().push(event.event);
    }
}
fn count(events: &Arc<Mutex<Vec<String>>>, name: &str) -> usize {
    events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.contains(&format!(":{name}:")))
        .count()
}
fn text(item: &Item) -> &str {
    match &item.parts[0] {
        Part::Text(text) => &text.text,
        _ => panic!("expected text"),
    }
}
fn finished(step: LoopStep) -> TurnResult {
    match step {
        LoopStep::Finished(result) => result,
        other => panic!("expected finished, got {other:?}"),
    }
}

#[tokio::test]
async fn lifecycle_sequential_payloads_and_committed_final_output() {
    let events = Arc::default();
    let observed: Arc<Mutex<Vec<AgentEvent>>> = Arc::default();
    let model = ProbeModel::new([Script::Reply]);
    let agent = Agent::builder()
        .model(model.clone())
        .input(vec![Item::text(ItemKind::User, "input")])
        .mutator(ProbeHook::new("1", &events))
        .mutator(ProbeHook::new("2", &events))
        .observer(Capture(observed.clone()))
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("lifecycle")).await.unwrap();
    assert_eq!(model.configs.lock().unwrap()[0].metadata.len(), 2);
    let result = finished(driver.next().await.unwrap());
    let request = &model.requests.lock().unwrap()[0];
    assert_eq!(request.session_id, SessionId::new("lifecycle"));
    assert_eq!(text(&request.transcript[0]), "input12");
    assert_eq!(request.metadata.len(), 2);
    assert_eq!(text(&result.items[0]), "reply:m1:m2:f1:f2");
    assert_eq!(driver.snapshot().transcript.last(), result.items.last());
    assert!(
        observed
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, AgentEvent::TurnFinished(turn) if turn == &result))
    );
    let calls = events.lock().unwrap();
    for pair in calls.chunks_exact(2) {
        assert!(pair[0].starts_with("1:"));
        assert_eq!(&pair[0][1..], &pair[1][1..]);
    }
    drop(calls);
    assert_eq!(count(&events, "turn_start"), 2);
    assert_eq!(count(&events, "turn_finish"), 2);
    assert_eq!(count(&events, "turn_end"), 2);
    assert_eq!(count(&events, "session_end"), 0);
    driver.close().await.unwrap();
    driver.close().await.unwrap();
    assert_eq!(count(&events, "session_end"), 2);
    assert!(matches!(
        driver.next().await,
        Err(LoopError::InvalidState(_))
    ));
    assert!(matches!(
        driver.submit_input(vec![]),
        Err(LoopError::InvalidState(_))
    ));
    assert!(matches!(
        driver.submit_input_async(vec![]).await,
        Err(LoopError::InvalidState(_))
    ));
}

#[tokio::test]
async fn sync_admission_is_pre_dispatch_and_async_admission_is_pre_acceptance() {
    let events = Arc::default();
    let observed: Arc<Mutex<Vec<AgentEvent>>> = Arc::default();
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Reply, Script::Reply]))
        .mutator(ProbeHook::new("1", &events))
        .observer(Capture(observed.clone()))
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("input")).await.unwrap();
    driver
        .submit_input(vec![Item::text(ItemKind::User, "sync")])
        .unwrap();
    assert_eq!(count(&events, "input"), 0);
    assert_eq!(driver.snapshot().pending_input.len(), 1);
    assert!(
        !observed
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, AgentEvent::InputAccepted { .. }))
    );
    finished(driver.next().await.unwrap());
    assert_eq!(count(&events, "input"), 1);
    driver
        .submit_input_async(vec![Item::text(ItemKind::User, "async")])
        .await
        .unwrap();
    assert_eq!(count(&events, "input"), 2);
    finished(driver.next().await.unwrap());
    assert_eq!(count(&events, "input"), 2);
}

#[tokio::test]
async fn input_failure_drops_admission_without_starting_a_turn_and_can_recover() {
    for awaited in [false, true] {
        let events = Arc::default();
        let model = ProbeModel::new([Script::Reply]);
        let mut hook = ProbeHook::new("1", &events);
        hook.fail = Some("input");
        let agent = Agent::builder()
            .model(model.clone())
            .mutator(hook)
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("input-failure"))
            .await
            .unwrap();
        let input = vec![Item::text(ItemKind::User, "rejected")];
        if awaited {
            assert!(driver.submit_input_async(input).await.is_err());
        } else {
            driver.submit_input(input).unwrap();
            assert!(driver.next().await.is_err());
        }
        assert!(driver.snapshot().transcript.is_empty());
        assert!(driver.snapshot().pending_input.is_empty());
        assert_eq!(count(&events, "turn_start"), 0);
        assert!(model.requests.lock().unwrap().is_empty());
        driver
            .submit_input_async(vec![Item::text(ItemKind::User, "accepted")])
            .await
            .unwrap();
        finished(driver.next().await.unwrap());
    }
}

#[tokio::test]
async fn tool_continuation_keeps_one_logical_turn_and_reports_real_batch() {
    let events = Arc::default();
    let model = ProbeModel::new([Script::Tool, Script::Reply]);
    let agent = Agent::builder()
        .model(model.clone())
        .tool_executor(ProbeExecutor)
        .mutator(ProbeHook::new("1", &events))
        .input(vec![Item::text(ItemKind::User, "tools")])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("tools")).await.unwrap();
    let turn_id = match driver.next().await.unwrap() {
        LoopStep::Interrupt(LoopInterrupt::AfterToolResult(info)) => info.turn_id,
        other => panic!("{other:?}"),
    };
    assert_eq!(count(&events, "batch"), 1);
    assert_eq!(count(&events, "turn_end"), 0);
    driver
        .submit_input(vec![Item::text(ItemKind::User, "interject")])
        .unwrap();
    let result = finished(driver.next().await.unwrap());
    assert_eq!(result.turn_id, turn_id);
    assert_eq!(count(&events, "turn_start"), 1);
    assert_eq!(count(&events, "turn_finish"), 1);
    assert_eq!(count(&events, "turn_end"), 1);
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].turn_id, requests[1].turn_id);
    let calls = events.lock().unwrap();
    assert!(
        calls
            .iter()
            .any(|event| event.ends_with(":1") && event.contains(":request:"))
    );
    assert!(
        calls
            .iter()
            .any(|event| event.ends_with(":2") && event.contains(":request:"))
    );
}

#[tokio::test]
async fn output_hook_failure_is_atomic_and_finishes_once() {
    for point in [
        "response",
        "turn_finish",
        "progress",
        "request",
        "turn_start",
    ] {
        let events = Arc::default();
        let observed: Arc<Mutex<Vec<AgentEvent>>> = Arc::default();
        let mut hook = ProbeHook::new("1", &events);
        hook.fail = Some(point);
        let agent = Agent::builder()
            .model(ProbeModel::new([Script::Reply]))
            .mutator(hook)
            .input(vec![Item::text(ItemKind::User, "input")])
            .observer(Capture(observed.clone()))
            .build()
            .unwrap();
        let mut driver = agent.start(SessionConfig::new("atomic")).await.unwrap();
        assert!(matches!(driver.next().await, Err(LoopError::Mutator(_))));
        assert!(
            !driver
                .snapshot()
                .transcript
                .iter()
                .any(|item| item.kind == ItemKind::Assistant)
        );
        assert_eq!(count(&events, "turn_finish"), 1, "{point}");
        assert_eq!(count(&events, "turn_end"), 1, "{point}");
        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
        assert_eq!(count(&events, "turn_end"), 1);
        assert_eq!(
            observed
                .lock()
                .unwrap()
                .iter()
                .filter(|event| matches!(event, AgentEvent::TurnFinished(_)))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn model_failures_preserve_original_error_despite_terminal_hook_failures() {
    for script in [
        Script::BeginError,
        Script::StreamError,
        Script::MissingFinished,
    ] {
        let events = Arc::default();
        let mut hook = ProbeHook::new("1", &events);
        hook.fail = Some("model_error");
        let mut terminal = ProbeHook::new("2", &events);
        terminal.fail = Some("turn_finish");
        let mut end = ProbeHook::new("3", &events);
        end.fail = Some("turn_end");
        let agent = Agent::builder()
            .model(ProbeModel::new([script]))
            .mutator(hook)
            .mutator(terminal)
            .mutator(end)
            .input(vec![Item::text(ItemKind::User, "input")])
            .build()
            .unwrap();
        let mut driver = agent.start(SessionConfig::new("failure")).await.unwrap();
        assert!(matches!(driver.next().await, Err(LoopError::Provider(_))));
        assert_eq!(count(&events, "model_error"), 3);
        assert_eq!(count(&events, "turn_finish"), 3);
        assert_eq!(count(&events, "turn_end"), 3);
        driver.close().await.unwrap();
        assert_eq!(count(&events, "turn_end"), 3);
    }
}

#[tokio::test]
async fn cancellation_and_explicit_retirement_run_readonly_finish_and_end() {
    let events = Arc::default();
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Cancelled]))
        .mutator(ProbeHook::new("1", &events))
        .input(vec![Item::text(ItemKind::User, "input")])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("cancelled")).await.unwrap();
    let result = finished(driver.next().await.unwrap());
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    assert_eq!(driver.snapshot().transcript.last(), result.items.last());
    assert_eq!(count(&events, "turn_finish"), 1);
    assert_eq!(count(&events, "turn_end"), 1);
    assert_eq!(count(&events, "model_error"), 1);
    assert!(driver.retire_interrupted_turn().await.unwrap().is_none());
    driver.close().await.unwrap();
    assert_eq!(count(&events, "turn_end"), 1);

    let events = Arc::default();
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Tool]))
        .tool_executor(ProbeExecutor)
        .mutator(ProbeHook::new("1", &events))
        .input(vec![Item::text(ItemKind::User, "input")])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("retire")).await.unwrap();
    driver.next().await.unwrap();
    driver.close().await.unwrap();
    assert_eq!(count(&events, "turn_finish"), 1);
    assert_eq!(count(&events, "turn_end"), 1);
    assert_eq!(count(&events, "session_end"), 1);
}

#[tokio::test]
async fn forged_tool_linkage_is_rejected_before_commit_or_execution() {
    let events = Arc::default();
    let mut hook = ProbeHook::new("1", &events);
    hook.break_links = true;
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Tool]))
        .tool_executor(ProbeExecutor)
        .mutator(hook)
        .input(vec![Item::text(ItemKind::User, "input")])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("identity")).await.unwrap();
    assert!(matches!(driver.next().await, Err(LoopError::Mutator(_))));
    assert_eq!(driver.snapshot().transcript.len(), 1);
    assert_eq!(count(&events, "batch"), 0);
    assert_eq!(count(&events, "turn_end"), 1);
}

#[tokio::test]
async fn session_end_failure_does_not_skip_later_hooks_or_reopen_session() {
    let events = Arc::default();
    let mut hook = ProbeHook::new("1", &events);
    hook.fail = Some("session_end");
    let agent = Agent::builder()
        .model(ProbeModel::new([]))
        .mutator(hook)
        .mutator(ProbeHook::new("2", &events))
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("close-error"))
        .await
        .unwrap();
    assert!(driver.close().await.is_err());
    assert_eq!(count(&events, "session_end"), 2);
    driver.close().await.unwrap();
    assert_eq!(count(&events, "session_end"), 2);
    assert!(driver.next().await.is_err());
}

struct PromptHook {
    fail_start: bool,
    fail_request: bool,
}
#[async_trait]
impl LoopMutator for PromptHook {
    async fn on_turn_start(
        &self,
        cursor: &mut TranscriptCursor<'_>,
        _: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        for item in cursor.iter_mut() {
            if item.kind == ItemKind::User {
                item.parts = vec![Part::text("logical prompt")];
            }
        }
        cursor.insert(0, Item::text(ItemKind::System, "persistent context"));
        if self.fail_start {
            Err(LoopError::Mutator("start prompt failed".into()))
        } else {
            Ok(())
        }
    }
    async fn on_model_request(
        &self,
        payload: ModelRequestPayload<'_>,
        _: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        for item in payload.transcript.iter_mut() {
            if item.kind == ItemKind::User {
                item.parts = vec![Part::text("inference prompt")];
            }
        }
        payload
            .transcript
            .insert(0, Item::text(ItemKind::System, "inference-only context"));
        if self.fail_request {
            Err(LoopError::Mutator("request prompt failed".into()))
        } else {
            Ok(())
        }
    }
}

#[tokio::test]
async fn logical_prompt_edits_persist_but_inference_prompt_edits_do_not() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = Agent::builder()
        .model(model.clone())
        .mutator(PromptHook {
            fail_start: false,
            fail_request: false,
        })
        .input(vec![
            Item::text(ItemKind::User, "initial").with_id("input-id"),
        ])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("prompts")).await.unwrap();
    finished(driver.next().await.unwrap());
    let requests = model.requests.lock().unwrap();
    assert_eq!(text(&requests[0].transcript[2]), "inference prompt");
    assert_eq!(
        requests[0].transcript[2].id,
        Some(agentkit_core::MessageId::new("input-id"))
    );
    assert_eq!(text(&requests[0].transcript[0]), "inference-only context");
    let transcript = driver.snapshot().transcript;
    assert_eq!(text(&transcript[1]), "logical prompt");
    assert_eq!(text(&transcript[0]), "persistent context");
    assert!(!transcript.iter().any(|item| {
        item.parts
            .iter()
            .any(|part| matches!(part, Part::Text(text) if text.text.contains("inference")))
    }));
}

#[tokio::test]
async fn failed_logical_prompt_edits_never_commit_and_request_edits_never_persist() {
    for fail_start in [false, true] {
        let model = ProbeModel::new([Script::Reply]);
        let agent = Agent::builder()
            .model(model.clone())
            .mutator(PromptHook {
                fail_start,
                fail_request: !fail_start,
            })
            .input(vec![Item::text(ItemKind::User, "initial")])
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("prompt-failure"))
            .await
            .unwrap();
        assert!(matches!(driver.next().await, Err(LoopError::Mutator(_))));
        let transcript = driver.snapshot().transcript;
        if fail_start {
            assert!(transcript.is_empty());
        } else {
            assert_eq!(transcript.len(), 2);
            assert_eq!(text(&transcript[1]), "logical prompt");
        }
        assert!(model.requests.lock().unwrap().is_empty());
        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_))
        ));
    }
}

#[derive(Default)]
struct StoredTranscript {
    items: Mutex<Vec<Item>>,
    appends: Mutex<Vec<Item>>,
    rewrites: std::sync::atomic::AtomicUsize,
}
impl TranscriptObserver for Arc<StoredTranscript> {
    fn on_transcript_event(&self, event: TranscriptEvent<'_>) {
        self.items.lock().unwrap().push(event.item.clone());
        self.appends.lock().unwrap().push(event.item.clone());
    }
    fn on_transcript_rewrite(&self, event: TranscriptRewriteEvent<'_>) {
        *self.items.lock().unwrap() = event.items.to_vec();
        self.rewrites.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn second_turn_prompt_insertion_emits_rewrite_not_historic_appends() {
    let stored = Arc::new(StoredTranscript::default());
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Reply, Script::Reply]))
        .mutator(PromptHook {
            fail_start: false,
            fail_request: false,
        })
        .transcript_observer(stored.clone())
        .input(vec![Item::text(ItemKind::User, "first")])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("rewrite")).await.unwrap();
    finished(driver.next().await.unwrap());
    assert_eq!(*stored.items.lock().unwrap(), driver.snapshot().transcript);
    let before = stored.appends.lock().unwrap().len();
    driver
        .submit_input_async(vec![Item::text(ItemKind::User, "second")])
        .await
        .unwrap();
    finished(driver.next().await.unwrap());
    assert_eq!(*stored.items.lock().unwrap(), driver.snapshot().transcript);
    assert_eq!(stored.rewrites.load(Ordering::SeqCst), 1);
    assert_eq!(
        stored.appends.lock().unwrap().len(),
        before + 1,
        "only the final output was appended; history/input arrived in the rewrite"
    );
}

struct RejectInput;
#[async_trait]
impl LoopMutator for RejectInput {
    async fn on_input(&self, input: &mut Vec<Item>, _: LifecycleCtx<'_>) -> Result<(), LoopError> {
        if input.iter().any(|item| text(item) == "reject") {
            Err(LoopError::Mutator("rejected batch".into()))
        } else {
            Ok(())
        }
    }
}

#[tokio::test]
async fn mixed_submission_rejection_preserves_earlier_valid_batch_and_order() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = Agent::builder()
        .model(model.clone())
        .mutator(RejectInput)
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("mixed-input"))
        .await
        .unwrap();
    driver
        .submit_input(vec![Item::text(ItemKind::User, "A")])
        .unwrap();
    assert!(
        driver
            .submit_input_async(vec![Item::text(ItemKind::User, "reject")])
            .await
            .is_err()
    );
    assert_eq!(
        driver
            .snapshot()
            .pending_input
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["A"]
    );
    driver
        .submit_input(vec![Item::text(ItemKind::User, "B")])
        .unwrap();
    driver
        .submit_input_async(vec![Item::text(ItemKind::User, "C")])
        .await
        .unwrap();
    assert_eq!(
        driver
            .snapshot()
            .pending_input
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["A", "B", "C"]
    );
    finished(driver.next().await.unwrap());
    assert_eq!(
        model.requests.lock().unwrap()[0]
            .transcript
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["A", "B", "C"]
    );
}

#[tokio::test]
async fn queued_rejection_preserves_prior_accepted_and_later_unrelated_batches() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = Agent::builder()
        .model(model.clone())
        .mutator(RejectInput)
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("queued-input"))
        .await
        .unwrap();
    for value in ["A", "reject", "C"] {
        driver
            .submit_input(vec![Item::text(ItemKind::User, value)])
            .unwrap();
    }
    assert!(driver.next().await.is_err());
    assert_eq!(
        driver
            .snapshot()
            .pending_input
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["A", "C"]
    );
    finished(driver.next().await.unwrap());
    assert_eq!(
        model.requests.lock().unwrap()[0]
            .transcript
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["A", "C"]
    );
}

#[derive(Clone)]
struct Gate {
    open: Arc<AtomicBool>,
    attempts: Arc<std::sync::atomic::AtomicUsize>,
    completed: Arc<std::sync::atomic::AtomicUsize>,
}
impl Gate {
    fn new() -> Self {
        Self {
            open: Arc::new(AtomicBool::new(false)),
            attempts: Arc::default(),
            completed: Arc::default(),
        }
    }
    async fn wait(&self) {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        if !self.open.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        self.completed.fetch_add(1, Ordering::SeqCst);
    }
}
struct BlockingHook {
    phase: &'static str,
    gate: Gate,
}
#[async_trait]
impl LoopMutator for BlockingHook {
    async fn on_turn_finish(
        &self,
        _: TurnFinishPayload<'_>,
        _: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        if self.phase == "finish" {
            self.gate.wait().await;
        }
        Ok(())
    }
    async fn on_turn_end(&self, _: &TurnResult, _: LifecycleCtx<'_>) -> Result<(), LoopError> {
        if self.phase == "end" {
            self.gate.wait().await;
        }
        Ok(())
    }
    async fn on_session_end(
        &self,
        _: SessionEndPayload<'_>,
        _: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        if self.phase == "session" {
            self.gate.wait().await;
        }
        Ok(())
    }
    async fn on_tool_batch(
        &self,
        _: ToolBatchPayload<'_>,
        _: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        if self.phase == "batch" {
            self.gate.wait().await;
        }
        Ok(())
    }
}
struct CompactCompletedTools;
#[async_trait]
impl LoopMutator for CompactCompletedTools {
    async fn mutate(
        &self,
        cursor: &mut TranscriptCursor<'_>,
        ctx: LoopCtx<'_>,
    ) -> Result<(), LoopError> {
        if ctx.point == MutationPoint::AfterToolResult {
            cursor.retain(|item| {
                !item
                    .parts
                    .iter()
                    .any(|part| matches!(part, Part::ToolCall(_) | Part::ToolResult(_)))
            });
        }
        Ok(())
    }
}

#[tokio::test]
async fn batch_facts_and_completed_callbacks_survive_compaction_and_dropped_delivery() {
    let events = Arc::default();
    let gate = Gate::new();
    let stored = Arc::new(StoredTranscript::default());
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Tool, Script::Reply]))
        .tool_executor(ProbeExecutor)
        .mutator(ProbeHook::new("first", &events))
        .mutator(BlockingHook {
            phase: "batch",
            gate: gate.clone(),
        })
        .mutator(ProbeHook::new("last", &events))
        .mutator(CompactCompletedTools)
        .transcript_observer(stored.clone())
        .input(vec![Item::text(ItemKind::User, "input")])
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("captured-batch"))
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(10), driver.next())
            .await
            .is_err()
    );
    assert_eq!(gate.attempts.load(Ordering::SeqCst), 1);
    gate.open.store(true, Ordering::SeqCst);
    finished(driver.next().await.unwrap());
    assert_eq!(gate.completed.load(Ordering::SeqCst), 1);
    assert_eq!(
        count(&events, "batch"),
        2,
        "each completed first/last callback runs once"
    );
    assert!(
        !driver
            .snapshot()
            .transcript
            .iter()
            .flat_map(|item| &item.parts)
            .any(|part| matches!(part, Part::ToolCall(_) | Part::ToolResult(_)))
    );
    assert_eq!(*stored.items.lock().unwrap(), driver.snapshot().transcript);
    driver.close().await.unwrap();
    assert_eq!(count(&events, "batch"), 2);
}

struct CleanupManager {
    inner: SimpleTaskManager,
    gate: Gate,
}
#[async_trait]
impl TaskManager for CleanupManager {
    async fn start_task(
        &self,
        request: TaskLaunchRequest,
        ctx: TaskStartContext,
    ) -> Result<TaskStartOutcome, agentkit_task_manager::TaskManagerError> {
        self.inner.start_task(request, ctx).await
    }
    async fn wait_for_turn(
        &self,
        id: &agentkit_core::TurnId,
        cancel: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, agentkit_task_manager::TaskManagerError> {
        self.inner.wait_for_turn(id, cancel).await
    }
    async fn take_pending_loop_updates(
        &self,
    ) -> Result<PendingLoopUpdates, agentkit_task_manager::TaskManagerError> {
        self.inner.take_pending_loop_updates().await
    }
    async fn on_turn_interrupted(
        &self,
        id: &agentkit_core::TurnId,
    ) -> Result<(), agentkit_task_manager::TaskManagerError> {
        self.gate.wait().await;
        self.inner.on_turn_interrupted(id).await
    }
    fn handle(&self) -> agentkit_task_manager::TaskManagerHandle {
        self.inner.handle()
    }
}

#[tokio::test]
async fn dropped_close_resumes_cleanup_and_terminal_callbacks_without_duplicates() {
    for phase in ["cleanup", "finish", "end", "session"] {
        let events = Arc::default();
        let observed: Arc<Mutex<Vec<AgentEvent>>> = Arc::default();
        let gate = Gate::new();
        let cleanup_gate = Gate::new();
        if phase != "cleanup" {
            cleanup_gate.open.store(true, Ordering::SeqCst);
        }
        let agent = Agent::builder()
            .model(ProbeModel::new([Script::Tool]))
            .tool_executor(ProbeExecutor)
            .task_manager(CleanupManager {
                inner: SimpleTaskManager::new(),
                gate: cleanup_gate.clone(),
            })
            .mutator(ProbeHook::new("first", &events))
            .mutator(BlockingHook {
                phase,
                gate: gate.clone(),
            })
            .mutator(ProbeHook::new("last", &events))
            .observer(Capture(observed.clone()))
            .input(vec![Item::text(ItemKind::User, "input")])
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("resumable-close"))
            .await
            .unwrap();
        assert!(matches!(
            driver.next().await.unwrap(),
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_))
        ));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), driver.close())
                .await
                .is_err(),
            "{phase}"
        );
        assert!(matches!(
            driver.next().await,
            Err(LoopError::InvalidState(_))
        ));
        assert!(
            driver
                .submit_input(vec![Item::text(ItemKind::User, "closed")])
                .is_err()
        );
        gate.open.store(true, Ordering::SeqCst);
        cleanup_gate.open.store(true, Ordering::SeqCst);
        driver.close().await.unwrap();
        driver.close().await.unwrap();
        assert_eq!(cleanup_gate.completed.load(Ordering::SeqCst), 1, "{phase}");
        if phase != "cleanup" {
            assert_eq!(gate.completed.load(Ordering::SeqCst), 1, "{phase}");
        }
        assert_eq!(count(&events, "turn_finish"), 2, "{phase}");
        assert_eq!(count(&events, "turn_end"), 2, "{phase}");
        assert_eq!(count(&events, "session_end"), 2, "{phase}");
        assert_eq!(
            observed
                .lock()
                .unwrap()
                .iter()
                .filter(|event| matches!(event, AgentEvent::TurnFinished(_)))
                .count(),
            1,
            "{phase}"
        );
    }
}

struct InterruptProgress {
    controller: agentkit_core::CancellationController,
    terminals: Arc<std::sync::atomic::AtomicUsize>,
}
#[async_trait]
impl LoopMutator for InterruptProgress {
    async fn on_model_progress(
        &self,
        _: &ModelTurnEvent,
        _: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        self.controller.interrupt();
        Ok(())
    }
    async fn on_turn_finish(
        &self,
        payload: TurnFinishPayload<'_>,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        assert_eq!(payload.result.finish_reason, FinishReason::Cancelled);
        assert!(payload.output.is_none());
        assert!(ctx.cancellation.as_ref().unwrap().is_cancelled());
        self.terminals.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn on_turn_end(
        &self,
        result: &TurnResult,
        ctx: LifecycleCtx<'_>,
    ) -> Result<(), LoopError> {
        assert_eq!(result.finish_reason, FinishReason::Cancelled);
        assert!(ctx.cancellation.as_ref().unwrap().is_cancelled());
        self.terminals.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[tokio::test]
async fn terminal_context_keeps_the_cancelled_active_checkpoint() {
    let controller = agentkit_core::CancellationController::new();
    let terminals = Arc::default();
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Reply]))
        .cancellation(controller.handle())
        .mutator(InterruptProgress {
            controller,
            terminals: Arc::clone(&terminals),
        })
        .input(vec![Item::text(ItemKind::User, "cancel during progress")])
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("checkpoint")).await.unwrap();
    let result = finished(driver.next().await.unwrap());
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    assert_eq!(terminals.load(Ordering::SeqCst), 2);
    assert_eq!(driver.snapshot().transcript.last(), result.items.last());
}

#[tokio::test]
async fn readonly_end_notification_failure_preserves_committed_result() {
    let events = Arc::default();
    let observed: Arc<Mutex<Vec<AgentEvent>>> = Arc::default();
    let mut hook = ProbeHook::new("1", &events);
    hook.fail = Some("turn_end");
    let agent = Agent::builder()
        .model(ProbeModel::new([Script::Reply]))
        .mutator(hook)
        .observer(Capture(observed.clone()))
        .input(vec![Item::text(ItemKind::User, "input")])
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("readonly-end"))
        .await
        .unwrap();
    let result = finished(driver.next().await.unwrap());
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert_eq!(driver.snapshot().transcript.last(), result.items.last());
    assert!(
        observed
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, AgentEvent::TurnFinished(turn) if turn == &result))
    );
    assert_eq!(count(&events, "turn_end"), 1);
    driver.close().await.unwrap();
    assert_eq!(count(&events, "turn_end"), 1);
}

struct SendOnlyModel(ProbeModel);
struct SendOnlySession {
    inner: ProbeSession,
    calls: std::cell::Cell<usize>,
}
struct SendOnlyTurn {
    inner: ProbeTurn,
    polls: std::cell::Cell<usize>,
}
#[async_trait]
impl ModelAdapter for SendOnlyModel {
    type Session = SendOnlySession;
    async fn start_session(&self, config: SessionConfig) -> Result<SendOnlySession, LoopError> {
        Ok(SendOnlySession {
            inner: self.0.start_session(config).await?,
            calls: std::cell::Cell::new(0),
        })
    }
}
#[async_trait]
impl ModelSession for SendOnlySession {
    type Turn = SendOnlyTurn;
    async fn begin_turn(
        &mut self,
        request: TurnRequest,
        cancel: Option<TurnCancellation>,
    ) -> Result<SendOnlyTurn, LoopError> {
        self.calls.set(self.calls.get() + 1);
        Ok(SendOnlyTurn {
            inner: self.inner.begin_turn(request, cancel).await?,
            polls: std::cell::Cell::new(0),
        })
    }
    fn provider_name(&self) -> Option<&str> {
        Some("send-only-provider")
    }
    fn model_name(&self) -> Option<&str> {
        Some("send-only-model")
    }
}
#[async_trait]
impl ModelTurn for SendOnlyTurn {
    async fn next_event(
        &mut self,
        cancel: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        self.polls.set(self.polls.get() + 1);
        self.inner.next_event(cancel).await
    }
}
fn require_send<T: Send>(value: T) -> T {
    value
}

#[tokio::test]
async fn public_lifecycle_futures_are_send_with_non_sync_session_and_turn() {
    let events = Arc::default();
    let agent = Agent::builder()
        .model(SendOnlyModel(ProbeModel::new([
            Script::Tool,
            Script::Reply,
            Script::BeginError,
        ])))
        .tool_executor(ProbeExecutor)
        .mutator(ProbeHook::new("send", &events))
        .build()
        .unwrap();
    let mut driver = require_send(agent.start(SessionConfig::new("send-only")))
        .await
        .unwrap();
    require_send(driver.submit_input_async(vec![Item::text(ItemKind::User, "tools")]))
        .await
        .unwrap();
    assert!(matches!(
        require_send(driver.next()).await.unwrap(),
        LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_))
    ));
    finished(require_send(driver.next()).await.unwrap());
    require_send(driver.submit_input_async(vec![Item::text(ItemKind::User, "error")]))
        .await
        .unwrap();
    assert!(matches!(
        require_send(driver.next()).await,
        Err(LoopError::Provider(_))
    ));
    require_send(driver.close()).await.unwrap();
}

#[tokio::test]
async fn earlier_queued_rejection_retains_new_async_input() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = Agent::builder()
        .model(model.clone())
        .mutator(RejectInput)
        .build()
        .unwrap();
    let mut driver = agent
        .start(SessionConfig::new("inverse-mixed-input"))
        .await
        .unwrap();
    driver
        .submit_input(vec![Item::text(ItemKind::User, "reject")])
        .unwrap();
    assert!(
        driver
            .submit_input_async(vec![Item::text(ItemKind::User, "B")])
            .await
            .is_err()
    );
    assert_eq!(
        driver
            .snapshot()
            .pending_input
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["B"]
    );
    finished(driver.next().await.unwrap());
    assert_eq!(
        model.requests.lock().unwrap()[0]
            .transcript
            .iter()
            .map(text)
            .collect::<Vec<_>>(),
        ["B"]
    );
}

#[tokio::test]
async fn dropped_next_terminal_delivery_is_resumed_by_close() {
    for phase in ["finish", "end"] {
        let events = Arc::default();
        let gate = Gate::new();
        let observed: Arc<Mutex<Vec<AgentEvent>>> = Arc::default();
        let agent = Agent::builder()
            .model(ProbeModel::new([Script::Reply]))
            .mutator(ProbeHook::new("first", &events))
            .mutator(BlockingHook {
                phase,
                gate: gate.clone(),
            })
            .mutator(ProbeHook::new("last", &events))
            .observer(Capture(observed.clone()))
            .input(vec![Item::text(ItemKind::User, "input")])
            .build()
            .unwrap();
        let mut driver = agent
            .start(SessionConfig::new("resume-next-terminal"))
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), driver.next())
                .await
                .is_err(),
            "{phase}"
        );
        gate.open.store(true, Ordering::SeqCst);
        driver.close().await.unwrap();
        driver.close().await.unwrap();
        assert_eq!(count(&events, "turn_finish"), 2, "{phase}");
        assert_eq!(count(&events, "turn_end"), 2, "{phase}");
        assert_eq!(gate.completed.load(Ordering::SeqCst), 1, "{phase}");
        let observed = observed.lock().unwrap();
        let terminals: Vec<&TurnResult> = observed
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TurnFinished(result) => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(terminals.len(), 1, "{phase}");
        let result = terminals[0];
        assert_eq!(
            result.finish_reason,
            FinishReason::Completed,
            "existing final outcome was resumed, not replaced"
        );
        assert_eq!(
            text(&result.items[0]),
            "reply:mfirst:mlast:ffirst:flast",
            "{phase}"
        );
        assert_eq!(
            driver.snapshot().transcript.last(),
            result.items.last(),
            "{phase}"
        );
    }
}
