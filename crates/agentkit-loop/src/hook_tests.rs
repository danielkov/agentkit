use super::*;
use agentkit_core::{Delta, PartId};
use agentkit_task_manager::{AsyncTaskManager, RoutingDecision};
use agentkit_tools_core::{ToolContext, ToolExecutionOutcome, ToolResult};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Notify;
use tokio::time::{Duration, timeout};

#[derive(Clone, Copy)]
enum Script {
    Reply,
    Tool,
    /// `n` text deltas, then a plain reply.
    Stream(usize),
    /// A tool-call attempt the provider supersedes, then a plain reply.
    Superseded,
}

#[derive(Clone, Default)]
struct ProbeModel {
    scripts: Arc<Mutex<VecDeque<Script>>>,
    requests: Arc<Mutex<Vec<TurnRequest>>>,
    configs: Arc<Mutex<Vec<SessionConfig>>>,
    /// Awaited before every event after the first, so the model cannot run
    /// ahead of an awaited delivery target.
    gate: Option<Arc<Notify>>,
}

impl ProbeModel {
    fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Arc::new(Mutex::new(scripts.into_iter().collect())),
            ..Self::default()
        }
    }

    fn gated(mut self, gate: Arc<Notify>) -> Self {
        self.gate = Some(gate);
        self
    }
}

struct ProbeSession(ProbeModel);
struct ProbeTurn {
    events: VecDeque<ModelTurnEvent>,
    gate: Option<Arc<Notify>>,
    produced: usize,
}

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
        let script = self.0.scripts.lock().unwrap().pop_front();
        let (item, finish_reason) = match script {
            Some(Script::Reply | Script::Stream(_) | Script::Superseded) => (
                Item::text(ItemKind::Assistant, "reply").with_id("message-reply"),
                FinishReason::Completed,
            ),
            Some(Script::Tool) => (
                Item::new(
                    ItemKind::Assistant,
                    vec![Part::ToolCall(ToolCallPart::new(
                        "call-probe",
                        "probe",
                        serde_json::json!({"value": 1}),
                    ))],
                )
                .with_id("message-tool"),
                FinishReason::ToolCall,
            ),
            None => return Err(LoopError::Provider("unexpected inference".into())),
        };
        let mut events = VecDeque::new();
        if let Some(Script::Stream(chunks)) = script {
            let part_id = PartId::new("part-stream");
            for chunk in 0..chunks {
                events.push_back(ModelTurnEvent::Delta(Delta::AppendText {
                    part_id: part_id.clone(),
                    chunk: format!("chunk-{chunk}"),
                }));
            }
        }
        if let Some(Script::Superseded) = script {
            events.push_back(ModelTurnEvent::ToolCall(ToolCallPart::new(
                "call-superseded",
                "probe",
                serde_json::json!({}),
            )));
            events.push_back(ModelTurnEvent::ResponseAttemptSuperseded);
        }
        if let Some(Part::ToolCall(call)) = item.parts.first() {
            events.push_back(ModelTurnEvent::ToolCall(call.clone()));
        }
        events.push_back(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason,
            output_items: vec![item],
            usage: None,
            metadata: MetadataMap::new(),
            model: None,
            response_id: None,
        }));
        Ok(ProbeTurn {
            events,
            gate: self.0.gate.clone(),
            produced: 0,
        })
    }
}

#[async_trait]
impl ModelTurn for ProbeTurn {
    async fn next_event(
        &mut self,
        _: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        if let Some(gate) = &self.gate
            && self.produced > 0
        {
            gate.notified().await;
        }
        self.produced += 1;
        Ok(self.events.pop_front())
    }
}

#[derive(Clone, Default)]
struct ProbeExecutor(Arc<Mutex<Vec<serde_json::Value>>>);

#[async_trait]
impl ToolExecutor for ProbeExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        Vec::new()
    }
    async fn execute(&self, request: ToolRequest, _: &mut ToolContext<'_>) -> ToolExecutionOutcome {
        self.0.lock().unwrap().push(request.input);
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

/// Appends its label to every editable value so ordering is observable.
struct Tag(&'static str);

#[async_trait]
impl LoopMutator for Tag {
    async fn on_session_start(
        &self,
        payload: SessionStartPayload<'_>,
        ctx: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        assert!(ctx.turn_id.is_none());
        let seen = payload
            .metadata
            .get("order")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_owned();
        payload
            .metadata
            .insert("order".into(), format!("{seen}{}", self.0).into());
        Ok(())
    }

    async fn on_model_request(
        &self,
        payload: ModelRequestPayload<'_>,
        ctx: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        assert!(ctx.turn_id.is_some());
        payload
            .transcript
            .push(Item::text(ItemKind::User, format!("ephemeral-{}", self.0)));
        Ok(())
    }

    async fn on_model_response(
        &self,
        payload: ModelResponsePayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        for item in payload.output {
            for part in item.parts.iter_mut() {
                match part {
                    Part::Text(text) => text.text.push_str(self.0),
                    Part::ToolCall(call) => call.input["tag"] = self.0.into(),
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

struct ForgeToolCall;

#[async_trait]
impl LoopMutator for ForgeToolCall {
    async fn on_model_response(
        &self,
        payload: ModelResponsePayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        for item in payload.output {
            for part in item.parts.iter_mut() {
                if let Part::ToolCall(call) = part {
                    call.id = ToolCallId::new("forged");
                }
            }
        }
        Ok(())
    }
}

struct OrphanToolResult;

#[async_trait]
impl LoopMutator for OrphanToolResult {
    async fn on_model_request(
        &self,
        payload: ModelRequestPayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        payload.transcript.push(Item::new(
            ItemKind::Tool,
            vec![Part::ToolResult(ToolResultPart {
                call_id: ToolCallId::new("orphan"),
                output: ToolOutput::Text("orphan".into()),
                is_error: false,
                metadata: MetadataMap::new(),
            })],
        ));
        Ok(())
    }
}

fn agent(
    model: &ProbeModel,
    executor: &ProbeExecutor,
    mutators: impl FnOnce(AgentBuilder<ProbeModel>) -> AgentBuilder<ProbeModel>,
) -> Agent<ProbeModel> {
    let builder = Agent::builder()
        .model(model.clone())
        .tool_executor(executor.clone())
        .input(vec![Item::text(ItemKind::User, "hi")]);
    mutators(builder).build().unwrap()
}

fn text(item: &Item) -> &str {
    match &item.parts[0] {
        Part::Text(text) => &text.text,
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn session_start_edits_reach_the_adapter_in_registration_order() {
    let model = ProbeModel::new([]);
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.mutator(Tag("a")).mutator(Tag("b"))
    });
    agent.start(SessionConfig::new("session")).await.unwrap();
    let configs = model.configs.lock().unwrap();
    assert_eq!(configs[0].metadata["order"], "ab");
}

#[tokio::test]
async fn request_edits_are_inference_local_and_response_edits_are_committed() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.mutator(Tag("a")).mutator(Tag("b"))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected finished turn");
    };

    let request = model.requests.lock().unwrap()[0].clone();
    let sent: Vec<&str> = request.transcript.iter().map(text).collect();
    assert_eq!(sent, ["hi", "ephemeral-a", "ephemeral-b"]);

    assert_eq!(text(&result.items[0]), "replyab");
    let transcript = driver.snapshot().transcript;
    let committed: Vec<&str> = transcript.iter().map(text).collect();
    assert_eq!(committed, ["hi", "replyab"]);
}

#[tokio::test]
async fn response_edits_to_tool_arguments_reach_the_executor() {
    let model = ProbeModel::new([Script::Tool, Script::Reply]);
    let executor = ProbeExecutor::default();
    let agent = agent(&model, &executor, |builder| builder.mutator(Tag("a")));
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    loop {
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Finished(_) => break,
            other => panic!("unexpected step {other:?}"),
        }
    }
    assert_eq!(
        executor.0.lock().unwrap().as_slice(),
        [serde_json::json!({"value": 1, "tag": "a"})]
    );
}

#[tokio::test]
async fn forged_tool_linkage_is_rejected_before_commit_or_execution() {
    let model = ProbeModel::new([Script::Tool]);
    let executor = ProbeExecutor::default();
    let agent = agent(&model, &executor, |builder| builder.mutator(ForgeToolCall));
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert!(executor.0.lock().unwrap().is_empty());
    assert!(
        driver
            .snapshot()
            .transcript
            .iter()
            .all(|item| item.kind != ItemKind::Assistant)
    );
}

#[tokio::test]
async fn protocol_invalid_request_edits_are_rejected_before_inference() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.mutator(OrphanToolResult)
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert!(model.requests.lock().unwrap().is_empty());
}

#[derive(Clone, Copy, PartialEq)]
enum Hook {
    SessionStart,
    Request,
    Response,
}

/// Fails `hook` once with `error`, then behaves as a no-op.
struct FailOnce {
    hook: Hook,
    error: fn() -> LoopError,
    armed: AtomicBool,
}

impl FailOnce {
    fn new(hook: Hook, error: fn() -> LoopError) -> Self {
        Self {
            hook,
            error,
            armed: true.into(),
        }
    }

    fn check(&self, hook: Hook) -> Result<(), LoopError> {
        if self.hook == hook && self.armed.swap(false, Ordering::SeqCst) {
            Err((self.error)())
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl LoopMutator for FailOnce {
    async fn on_session_start(
        &self,
        _: SessionStartPayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        self.check(Hook::SessionStart)
    }

    async fn on_model_request(
        &self,
        _: ModelRequestPayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        self.check(Hook::Request)
    }

    async fn on_model_response(
        &self,
        _: ModelResponsePayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        self.check(Hook::Response)
    }
}

fn hook_error() -> LoopError {
    LoopError::Mutator("hook failed".into())
}

#[tokio::test]
async fn session_start_failure_never_starts_the_adapter_session() {
    let model = ProbeModel::new([]);
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.mutator(FailOnce::new(Hook::SessionStart, hook_error))
    });
    let Err(error) = agent.start(SessionConfig::new("session")).await else {
        panic!("expected start to fail");
    };
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert!(model.configs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn response_failure_commits_nothing_and_the_session_recovers() {
    let model = ProbeModel::new([Script::Tool, Script::Reply]);
    let executor = ProbeExecutor::default();
    let agent = agent(&model, &executor, |builder| {
        builder.mutator(FailOnce::new(Hook::Response, hook_error))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert!(executor.0.lock().unwrap().is_empty());
    let kinds: Vec<ItemKind> = driver
        .snapshot()
        .transcript
        .iter()
        .map(|item| item.kind)
        .collect();
    assert_eq!(kinds, [ItemKind::User]);

    let request = match driver.next().await.unwrap() {
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(request)) => request,
        other => panic!("unexpected step {other:?}"),
    };
    request
        .submit(&mut driver, vec![Item::text(ItemKind::User, "again")])
        .unwrap();
    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
}

#[tokio::test]
async fn cancelled_request_hook_finishes_cancelled_without_inference() {
    let model = ProbeModel::new([Script::Reply]);
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.mutator(FailOnce::new(Hook::Request, || LoopError::Cancelled))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    assert!(model.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancelled_response_hook_finishes_cancelled_without_committing_output() {
    let model = ProbeModel::new([Script::Tool]);
    let executor = ProbeExecutor::default();
    let agent = agent(&model, &executor, |builder| {
        builder.mutator(FailOnce::new(Hook::Response, || LoopError::Cancelled))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    assert!(executor.0.lock().unwrap().is_empty());
    assert!(
        driver
            .snapshot()
            .transcript
            .iter()
            .flat_map(|item| &item.parts)
            .all(|part| !matches!(part, Part::ToolCall(_)))
    );
}

// ---------------------------------------------------------------------------
// Transactional mutation, canonical rewrite, terminal funnel, delivery.
//
// Everything below drives the real `Agent` builder and `LoopDriver`, and reads
// results only through the public surface: `snapshot`, `next`, `submit`,
// `retire_interrupted_turn`, `take_delivery_errors` and the registered
// observer/delivery traits.
// ---------------------------------------------------------------------------

/// One ordered notification, from whichever channel produced it. Transcript
/// notifications and delivered facts share a log so their relative order is
/// what the assertions read.
#[derive(Clone, Debug, PartialEq)]
enum Note {
    Append(String),
    Rewrite(Vec<String>),
    Progress(String),
    BeforeFinish(FinishReason),
    TurnFinished(FinishReason),
}

#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<Note>>>);

impl Log {
    fn push(&self, note: Note) {
        self.0.lock().unwrap().push(note);
    }

    fn notes(&self) -> Vec<Note> {
        self.0.lock().unwrap().clone()
    }

    fn rewrites(&self) -> Vec<Vec<String>> {
        self.notes()
            .into_iter()
            .filter_map(|note| match note {
                Note::Rewrite(items) => Some(items),
                _ => None,
            })
            .collect()
    }

    fn count(&self, matching: fn(&Note) -> bool) -> usize {
        self.notes().iter().filter(|note| matching(note)).count()
    }
}

fn label(item: &Item) -> String {
    match item.parts.first() {
        Some(Part::Text(text)) => text.text.clone(),
        Some(Part::ToolCall(call)) => format!("call:{}", call.id.0),
        Some(Part::ToolResult(result)) => format!("result:{}", result.call_id.0),
        _ => format!("{:?}", item.kind),
    }
}

fn labels(items: &[Item]) -> Vec<String> {
    items.iter().map(label).collect()
}

struct Persistor(Log);

impl TranscriptObserver for Persistor {
    fn on_transcript_event(&self, event: TranscriptEvent<'_>) {
        self.0.push(Note::Append(label(event.item)));
    }

    fn on_transcript_rewrite(&self, event: TranscriptRewriteEvent<'_>) {
        self.0.push(Note::Rewrite(labels(event.items)));
    }
}

/// What a [`Mutation`] does to the candidate transcript.
#[derive(Clone, Copy)]
enum Edit {
    /// Take `&mut` access and write the same value back.
    Touch,
    /// Append one item.
    Append,
    /// Replace the whole history with a single summary item.
    Replace,
    /// Append a `tool_result` that answers no call.
    Orphan,
    /// Remove everything, leaving nothing for the model to answer.
    Erase,
}

/// What a [`Mutation`] does after editing.
#[derive(Clone, Copy)]
enum After {
    Return,
    Fail,
    Cancel,
    /// Never returns, so the caller's future can be dropped mid-mutation.
    Hang,
}

/// Edits the candidate transcript at one mutation point, then returns, fails,
/// cancels or hangs — the four ways a mutation can end.
struct Mutation {
    point: MutationPoint,
    edit: Edit,
    after: After,
}

impl Mutation {
    fn new(point: MutationPoint, edit: Edit, after: After) -> Self {
        Self { point, edit, after }
    }
}

/// Records every mutation point it runs at, without editing anything.
struct RecordPoints(Arc<Mutex<Vec<MutationPoint>>>);

#[async_trait]
impl LoopMutator for RecordPoints {
    async fn mutate(
        &self,
        _: &mut TranscriptCursor<'_>,
        ctx: LoopCtx<'_>,
    ) -> Result<(), LoopError> {
        self.0.lock().unwrap().push(ctx.point);
        Ok(())
    }
}

/// Cancels once at [`MutationPoint::TurnStarted`], when the test arms it.
struct CancelWhenArmed(Arc<AtomicBool>);

#[async_trait]
impl LoopMutator for CancelWhenArmed {
    async fn mutate(
        &self,
        _: &mut TranscriptCursor<'_>,
        ctx: LoopCtx<'_>,
    ) -> Result<(), LoopError> {
        if ctx.point == MutationPoint::TurnStarted && self.0.swap(false, Ordering::SeqCst) {
            return Err(LoopError::Cancelled);
        }
        Ok(())
    }
}

#[async_trait]
impl LoopMutator for Mutation {
    async fn mutate(
        &self,
        cursor: &mut TranscriptCursor<'_>,
        ctx: LoopCtx<'_>,
    ) -> Result<(), LoopError> {
        if ctx.point != self.point {
            return Ok(());
        }
        match self.edit {
            Edit::Touch => {
                if let Some(first) = cursor.first().cloned() {
                    cursor[0] = first;
                }
            }
            Edit::Append => cursor.push(Item::text(ItemKind::Context, "injected")),
            Edit::Replace => **cursor = vec![Item::text(ItemKind::User, "summary")],
            Edit::Orphan => cursor.push(Item::new(
                ItemKind::Tool,
                vec![Part::ToolResult(ToolResultPart {
                    call_id: ToolCallId::new("orphan"),
                    output: ToolOutput::Text("orphan".into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                })],
            )),
            Edit::Erase => cursor.clear(),
        }
        match self.after {
            After::Return => Ok(()),
            After::Fail => Err(LoopError::Mutator("mutation failed".into())),
            After::Cancel => Err(LoopError::Cancelled),
            After::Hang => std::future::pending().await,
        }
    }
}

/// Records every delivered fact, optionally failing one kind of fact and
/// optionally releasing a model gate so progress timing is observable.
struct Target {
    log: Log,
    fail: Option<NativeFactKind>,
    gate: Option<Arc<Notify>>,
}

impl Target {
    fn new(log: Log) -> Self {
        Self {
            log,
            fail: None,
            gate: None,
        }
    }

    fn failing(mut self, fact: NativeFactKind) -> Self {
        self.fail = Some(fact);
        self
    }

    fn releasing(mut self, gate: Arc<Notify>) -> Self {
        self.gate = Some(gate);
        self
    }
}

#[async_trait]
impl NativeDelivery for Target {
    async fn deliver(&self, fact: NativeFact<'_>, ctx: HookCtx<'_>) -> Result<(), DeliveryError> {
        assert!(ctx.turn_id.is_some(), "every fact belongs to a turn");
        match fact {
            NativeFact::Progress(progress) => {
                // A consumer sees `Progress` with `cancellation == None` when
                // the agent was built without a cancellation handle, so the
                // variant is the only thing that identifies a terminal fact.
                let rendered = match progress {
                    NativeProgress::Model(ModelTurnEvent::Delta(Delta::AppendText {
                        chunk,
                        ..
                    })) => format!("delta:{chunk}"),
                    NativeProgress::Model(ModelTurnEvent::ToolCall(call)) => {
                        format!("call:{}", call.id.0)
                    }
                    NativeProgress::Model(ModelTurnEvent::ResponseAttemptSuperseded) => {
                        "superseded".into()
                    }
                    NativeProgress::Model(_) => "model".into(),
                    NativeProgress::ToolDetached(result) => {
                        format!("detached:{}", result.call_id.0)
                    }
                };
                self.log.push(Note::Progress(rendered));
                if let Some(gate) = &self.gate {
                    gate.notify_one();
                }
            }
            NativeFact::BeforeFinish(result) => {
                assert!(
                    ctx.cancellation.is_none(),
                    "terminal delivery runs uncancelled"
                );
                self.log
                    .push(Note::BeforeFinish(result.finish_reason.clone()));
            }
            NativeFact::TurnFinished(result) => {
                assert!(
                    ctx.cancellation.is_none(),
                    "terminal delivery runs uncancelled"
                );
                self.log
                    .push(Note::TurnFinished(result.finish_reason.clone()));
            }
        }
        let kind = match fact {
            NativeFact::Progress(_) => NativeFactKind::Progress,
            NativeFact::BeforeFinish(_) => NativeFactKind::BeforeFinish,
            NativeFact::TurnFinished(_) => NativeFactKind::TurnFinished,
        };
        match self.fail {
            Some(failing) if failing == kind => Err(DeliveryError::new("target offline")),
            _ => Ok(()),
        }
    }
}

/// Records the disposition every response hook saw.
struct RecordDisposition(Arc<Mutex<Vec<ResponseDisposition>>>);

#[async_trait]
impl LoopMutator for RecordDisposition {
    async fn on_model_response(
        &self,
        payload: ModelResponsePayload<'_>,
        _: HookCtx<'_>,
    ) -> Result<(), LoopError> {
        self.0.lock().unwrap().push(payload.disposition);
        Ok(())
    }
}

#[tokio::test]
async fn failed_mutation_rolls_back_and_notifies_nothing() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .mutator(Mutation::new(
                MutationPoint::TurnStarted,
                Edit::Append,
                After::Fail,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert_eq!(labels(&driver.snapshot().transcript), ["hi"]);
    assert!(log.rewrites().is_empty(), "{:?}", log.notes());
    assert!(
        model.requests.lock().unwrap().is_empty(),
        "a rolled-back mutation never reaches inference"
    );
}

#[tokio::test]
async fn invalid_mutation_rolls_back_and_notifies_nothing() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .mutator(Mutation::new(
                MutationPoint::TurnStarted,
                Edit::Orphan,
                After::Return,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert_eq!(labels(&driver.snapshot().transcript), ["hi"]);
    assert!(log.rewrites().is_empty(), "{:?}", log.notes());
}

#[tokio::test]
async fn dropped_mutation_rolls_back_and_notifies_nothing() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .mutator(Mutation::new(
                MutationPoint::TurnStarted,
                Edit::Replace,
                After::Hang,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    // Dropping the `next()` future abandons the candidate mid-chain.
    assert!(
        timeout(Duration::from_millis(50), driver.next())
            .await
            .is_err(),
        "the hanging mutation should not let next() resolve"
    );
    assert_eq!(labels(&driver.snapshot().transcript), ["hi"]);
    assert!(log.rewrites().is_empty(), "{:?}", log.notes());
}

#[tokio::test]
async fn cancelled_mutation_rolls_back_and_finishes_the_turn_cancelled() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()))
            .mutator(Mutation::new(
                MutationPoint::TurnStarted,
                Edit::Replace,
                After::Cancel,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected a finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    assert_eq!(labels(&driver.snapshot().transcript), ["hi"]);
    assert!(log.rewrites().is_empty(), "{:?}", log.notes());
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "a cancelled turn start still ends exactly once: {:?}",
        log.notes()
    );
}

#[tokio::test]
async fn mutation_writing_the_same_value_notifies_nothing() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .mutator(Mutation::new(
                MutationPoint::TurnStarted,
                Edit::Touch,
                After::Return,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected a finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert!(
        log.rewrites().is_empty(),
        "taking &mut access is not a change: {:?}",
        log.notes()
    );
}

#[tokio::test]
async fn full_rewrite_notifies_one_canonical_snapshot() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .mutator(Mutation::new(
                MutationPoint::TurnStarted,
                Edit::Replace,
                After::Return,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(_) = driver.next().await.unwrap() else {
        panic!("expected a finished turn");
    };
    assert_eq!(
        log.rewrites(),
        [vec!["summary".to_owned()]],
        "exactly one rewrite, carrying the whole canonical transcript"
    );
    assert_eq!(
        labels(&driver.snapshot().transcript),
        ["summary", "reply"],
        "the rewrite is what inference and later appends build on"
    );
    let request = model.requests.lock().unwrap()[0].clone();
    assert_eq!(labels(&request.transcript), ["summary"]);
}

#[tokio::test]
async fn turn_start_runs_once_for_a_turn_that_never_dispatches() {
    let model = ProbeModel::new([]);
    let log = Log::default();
    let points = Arc::new(Mutex::new(Vec::new()));
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()))
            .mutator(RecordPoints(points.clone()))
            .mutator(Mutation::new(
                MutationPoint::AfterTurnEnded,
                Edit::Erase,
                After::Return,
            ))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected a finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert!(result.items.is_empty());
    assert!(
        model.requests.lock().unwrap().is_empty(),
        "an emptied transcript must not dispatch an assistant prefill"
    );
    assert_eq!(
        points.lock().unwrap().as_slice(),
        [MutationPoint::TurnStarted, MutationPoint::AfterTurnEnded],
        "the turn-start point runs once, even though no inference followed"
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{:?}",
        log.notes()
    );
}

#[tokio::test]
async fn turn_start_sees_admitted_input() {
    let model = ProbeModel::new([Script::Reply]);
    let seen = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));

    struct SeeInput(Arc<Mutex<Vec<Vec<String>>>>);

    #[async_trait]
    impl LoopMutator for SeeInput {
        async fn mutate(
            &self,
            cursor: &mut TranscriptCursor<'_>,
            ctx: LoopCtx<'_>,
        ) -> Result<(), LoopError> {
            if ctx.point == MutationPoint::TurnStarted {
                self.0.lock().unwrap().push(labels(cursor));
            }
            Ok(())
        }
    }

    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.mutator(SeeInput(seen.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let _ = driver.next().await.unwrap();

    assert_eq!(seen.lock().unwrap().as_slice(), [vec!["hi".to_owned()]]);
}

#[tokio::test]
async fn prefinish_precedes_the_terminal_commit_and_the_finish() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let _ = driver.next().await.unwrap();

    assert_eq!(
        log.notes(),
        [
            Note::Append("hi".into()),
            Note::BeforeFinish(FinishReason::Completed),
            Note::Append("reply".into()),
            Note::TurnFinished(FinishReason::Completed),
        ],
        "prefinish observation lands before the terminal output commits"
    );
}

#[tokio::test]
async fn cancelled_partial_is_delivered_before_it_commits() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()))
            .mutator(FailOnce::new(Hook::Request, || LoopError::Cancelled))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected a finished turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    let notes = log.notes();
    let before = notes
        .iter()
        .position(|note| matches!(note, Note::BeforeFinish(_)))
        .expect("prefinish delivered");
    let partial = notes
        .iter()
        .position(|note| matches!(note, Note::Append(text) if text.starts_with("Previous")))
        .expect("cancellation partial committed");
    assert!(
        before < partial,
        "the partial is observable before it commits: {notes:?}"
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{notes:?}"
    );
}

#[tokio::test]
async fn disposition_follows_the_loop_branch_through_tools_and_supersession() {
    let model = ProbeModel::new([Script::Tool, Script::Superseded]);
    let dispositions = Arc::new(Mutex::new(Vec::new()));
    let executor = ProbeExecutor::default();
    let agent = agent(&model, &executor, |builder| {
        builder.mutator(RecordDisposition(dispositions.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    loop {
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Finished(_) => break,
            other => panic!("unexpected step {other:?}"),
        }
    }

    assert_eq!(
        dispositions.lock().unwrap().as_slice(),
        [
            ResponseDisposition::ContinueWithTools,
            ResponseDisposition::FinishTurnCandidate,
        ],
        "the superseded attempt's tool call must not make the replacement \
         response look like a tool round"
    );
}

#[tokio::test]
async fn disposition_cannot_be_moved_by_rewriting_linkage() {
    let model = ProbeModel::new([Script::Tool]);
    let executor = ProbeExecutor::default();
    let agent = agent(&model, &executor, |builder| builder.mutator(ForgeToolCall));
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Mutator(_)), "{error:?}");
    assert!(executor.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn progress_is_delivered_while_the_model_call_is_still_running() {
    let gate = Arc::new(Notify::new());
    let model = ProbeModel::new([Script::Stream(3)]).gated(gate.clone());
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.delivery(Target::new(log.clone()).releasing(gate.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    // The model refuses to produce its next event until the previous one has
    // been delivered, so buffering progress until `next()` returns deadlocks.
    let step = timeout(Duration::from_secs(5), driver.next())
        .await
        .expect("progress must be delivered inside next(), not buffered")
        .unwrap();
    assert!(matches!(step, LoopStep::Finished(_)));
    assert_eq!(
        log.notes()
            .into_iter()
            .take_while(|note| matches!(note, Note::Progress(_)))
            .collect::<Vec<_>>(),
        [
            Note::Progress("delta:chunk-0".into()),
            Note::Progress("delta:chunk-1".into()),
            Note::Progress("delta:chunk-2".into()),
        ]
    );
}

#[tokio::test]
async fn a_tool_round_turn_ends_exactly_once_without_recommitting_output() {
    let model = ProbeModel::new([Script::Tool, Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    loop {
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Finished(_) => break,
            other => panic!("unexpected step {other:?}"),
        }
    }

    assert_eq!(
        labels(&driver.snapshot().transcript),
        ["hi", "call:call-probe", "result:call-probe", "reply"],
        "the tool round's output commits once, for dispatch, not again at the finish"
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{:?}",
        log.notes()
    );
    assert_eq!(log.count(|note| matches!(note, Note::BeforeFinish(_))), 1);
}

#[tokio::test]
async fn an_errored_turn_ends_exactly_once() {
    // No script left: the second inference fails inside the loop.
    let model = ProbeModel::new([]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.delivery(Target::new(log.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let error = driver.next().await.unwrap_err();
    assert!(matches!(error, LoopError::Provider(_)), "{error:?}");
    assert_eq!(
        log.notes(),
        [
            Note::BeforeFinish(FinishReason::Error),
            Note::TurnFinished(FinishReason::Error),
        ],
        "an errored turn still ends once, through the same funnel"
    );
}

/// Fails every foreground-task cleanup so retirement reports an error.
struct FailCleanup(SimpleTaskManager);

#[async_trait]
impl TaskManager for FailCleanup {
    async fn start_task(
        &self,
        request: agentkit_task_manager::TaskLaunchRequest,
        ctx: agentkit_task_manager::TaskStartContext,
    ) -> Result<TaskStartOutcome, agentkit_task_manager::TaskManagerError> {
        self.0.start_task(request, ctx).await
    }

    async fn wait_for_turn(
        &self,
        turn_id: &agentkit_core::TurnId,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, agentkit_task_manager::TaskManagerError> {
        self.0.wait_for_turn(turn_id, cancellation).await
    }

    async fn take_pending_loop_updates(
        &self,
    ) -> Result<PendingLoopUpdates, agentkit_task_manager::TaskManagerError> {
        self.0.take_pending_loop_updates().await
    }

    async fn on_turn_interrupted(
        &self,
        _: &agentkit_core::TurnId,
    ) -> Result<(), agentkit_task_manager::TaskManagerError> {
        Err(agentkit_task_manager::TaskManagerError::Internal(
            "cleanup unavailable".into(),
        ))
    }

    fn handle(&self) -> agentkit_task_manager::TaskManagerHandle {
        self.0.handle()
    }
}

/// Requires approval for every call, so a turn can be retired with tool work
/// still outstanding.
struct ApprovalExecutor;

#[async_trait]
impl ToolExecutor for ApprovalExecutor {
    fn specs(&self) -> Vec<ToolSpec> {
        Vec::new()
    }
    async fn execute(&self, request: ToolRequest, _: &mut ToolContext<'_>) -> ToolExecutionOutcome {
        ToolExecutionOutcome::Interrupted(agentkit_tools_core::ToolInterruption::ApprovalRequired(
            ApprovalRequest {
                task_id: None,
                call_id: Some(request.call_id),
                id: "approval:probe".into(),
                request_kind: "probe".into(),
                reason: agentkit_tools_core::ApprovalReason::SensitivePath,
                summary: "probe needs approval".into(),
                metadata: MetadataMap::new(),
            },
        ))
    }
}

#[tokio::test]
async fn retirement_ends_the_turn_once_even_when_cleanup_fails() {
    let model = ProbeModel::new([Script::Tool]);
    let log = Log::default();
    let agent = Agent::builder()
        .model(model.clone())
        .tool_executor(ApprovalExecutor)
        .input(vec![Item::text(ItemKind::User, "hi")])
        .task_manager(FailCleanup(SimpleTaskManager::new()))
        .transcript_observer(Persistor(log.clone()))
        .delivery(Target::new(log.clone()))
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_)) = driver.next().await.unwrap()
    else {
        panic!("expected an approval interrupt");
    };
    let error = driver.retire_interrupted_turn().await.unwrap_err();
    assert!(matches!(error, LoopError::Tool(_)), "{error:?}");
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "the ending is delivered before the cleanup error surfaces: {:?}",
        log.notes()
    );
    assert!(
        driver.retire_interrupted_turn().await.unwrap().is_none(),
        "a retired turn cannot be retired again"
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{:?}",
        log.notes()
    );
}

#[tokio::test]
async fn delivery_failure_is_separate_from_the_operation_result() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()).failing(NativeFactKind::TurnFinished))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("a failed delivery must not fail the turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert_eq!(labels(&result.items), ["reply"]);
    assert_eq!(labels(&driver.snapshot().transcript), ["hi", "reply"]);
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "no replay and no second ending: {:?}",
        log.notes()
    );

    let failures = driver.take_delivery_errors();
    assert_eq!(failures.failures.len(), 1, "{failures:?}");
    assert_eq!(failures.failures[0].fact, NativeFactKind::TurnFinished);
    assert_eq!(failures.dropped, 0);
    assert!(
        driver.take_delivery_errors().is_empty(),
        "draining clears the diagnostics"
    );
}

#[tokio::test]
async fn failing_prefinish_delivery_leaves_the_turn_and_later_facts_intact() {
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()).failing(NativeFactKind::BeforeFinish))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("a failed prefinish delivery must not fail the turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert_eq!(labels(&result.items), ["reply"]);
    assert_eq!(
        log.notes(),
        [
            Note::Append("hi".into()),
            Note::BeforeFinish(FinishReason::Completed),
            Note::Append("reply".into()),
            Note::TurnFinished(FinishReason::Completed),
        ],
        "the terminal output still commits and the post-commit fact still \
         arrives after the prefinish target failed"
    );
    assert_eq!(labels(&driver.snapshot().transcript), ["hi", "reply"]);

    let failures = driver.take_delivery_errors();
    assert_eq!(failures.failures.len(), 1, "{failures:?}");
    assert_eq!(failures.failures[0].fact, NativeFactKind::BeforeFinish);
    assert_eq!(failures.dropped, 0);
}

#[tokio::test]
async fn failing_progress_delivery_leaves_the_turn_and_later_facts_intact() {
    let model = ProbeModel::new([Script::Stream(2)]);
    let log = Log::default();
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(Target::new(log.clone()).failing(NativeFactKind::Progress))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("a failed progress delivery must not fail the turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert_eq!(labels(&result.items), ["reply"]);
    assert_eq!(
        log.notes(),
        [
            Note::Append("hi".into()),
            Note::Progress("delta:chunk-0".into()),
            Note::Progress("delta:chunk-1".into()),
            Note::BeforeFinish(FinishReason::Completed),
            Note::Append("reply".into()),
            Note::TurnFinished(FinishReason::Completed),
        ],
        "a failed progress delivery neither stops the stream nor the turn"
    );

    let failures = driver.take_delivery_errors();
    assert_eq!(failures.failures.len(), 2, "{failures:?}");
    assert!(
        failures
            .failures
            .iter()
            .all(|failure| failure.fact == NativeFactKind::Progress)
    );
    assert_eq!(failures.dropped, 0);
}

#[tokio::test]
async fn delivery_diagnostics_are_bounded_and_reset_on_drain() {
    // More failures than the retained bound, so the buffer has to discard.
    const FACTS: usize = 70;
    let model = ProbeModel::new([Script::Stream(FACTS)]);
    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder.delivery(Target::new(Log::default()).failing(NativeFactKind::Progress))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();
    let LoopStep::Finished(_) = driver.next().await.unwrap() else {
        panic!("expected a finished turn");
    };

    let failures = driver.take_delivery_errors();
    assert!(
        failures.failures.len() < FACTS,
        "the buffer is bounded: {}",
        failures.failures.len()
    );
    assert_eq!(
        failures.failures.len() + failures.dropped as usize,
        FACTS,
        "nothing vanishes from the accounting: {failures:?}"
    );
    assert!(
        failures
            .failures
            .iter()
            .all(|failure| failure.fact == NativeFactKind::Progress)
    );

    assert!(
        driver.take_delivery_errors().is_empty(),
        "draining resets both the retained failures and the dropped count"
    );
}

/// Routes every tool call to the background, so a turn ends by returning
/// `AwaitingInput` and the result arrives later as an out-of-band loop update.
fn background_task_manager() -> AsyncTaskManager {
    AsyncTaskManager::new().routing(|_: &ToolRequest| RoutingDecision::Background)
}

#[tokio::test]
async fn background_round_and_its_synthetic_turn_each_end_exactly_once() {
    let model = ProbeModel::new([Script::Tool, Script::Reply]);
    let log = Log::default();
    let points = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder()
        .model(model.clone())
        .tool_executor(ProbeExecutor::default())
        .input(vec![Item::text(ItemKind::User, "hi")])
        .task_manager(background_task_manager())
        .transcript_observer(Persistor(log.clone()))
        .delivery(Target::new(log.clone()))
        .mutator(RecordPoints(points.clone()))
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    // The only call detached, so the turn ends through the `AwaitingInput`
    // branch rather than a finished step.
    let LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) = driver.next().await.unwrap() else {
        panic!("expected the turn to end awaiting input");
    };
    assert_eq!(
        log.count(|note| matches!(note, Note::BeforeFinish(_))),
        1,
        "{:?}",
        log.notes()
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{:?}",
        log.notes()
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::Progress(text) if text.starts_with("detached:"))),
        1,
        "the detach placeholder is delivered as progress: {:?}",
        log.notes()
    );

    // The background result opens a synthetic turn that no host input asked
    // for; it gets its own turn-start point and its own single ending.
    timeout(Duration::from_secs(5), driver.wait_for_loop_update())
        .await
        .expect("the background result should arrive")
        .unwrap();
    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected the synthetic turn to finish");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert_eq!(
        points.lock().unwrap().as_slice(),
        [
            MutationPoint::TurnStarted,
            MutationPoint::AfterTurnEnded,
            MutationPoint::TurnStarted,
            MutationPoint::AfterTurnEnded,
        ],
        "the synthetic background-resolution turn runs the turn-start point too"
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::BeforeFinish(_))),
        2,
        "{:?}",
        log.notes()
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        2,
        "{:?}",
        log.notes()
    );
}

#[tokio::test]
async fn cancelling_a_synthetic_turn_start_requeues_its_background_resolutions() {
    let model = ProbeModel::new([Script::Tool, Script::Reply]);
    let armed = Arc::new(AtomicBool::new(false));
    let agent = Agent::builder()
        .model(model.clone())
        .tool_executor(ProbeExecutor::default())
        .input(vec![Item::text(ItemKind::User, "hi")])
        .task_manager(background_task_manager())
        .mutator(CancelWhenArmed(armed.clone()))
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) = driver.next().await.unwrap() else {
        panic!("expected the turn to end awaiting input");
    };
    timeout(Duration::from_secs(5), driver.wait_for_loop_update())
        .await
        .expect("the background result should arrive")
        .unwrap();

    // Cancel the synthetic turn at its start: it carries nothing yet, so its
    // resolutions must survive for the next step rather than be dropped.
    armed.store(true, Ordering::SeqCst);
    let LoopStep::Finished(cancelled) = driver.next().await.unwrap() else {
        panic!("expected a cancelled turn");
    };
    assert_eq!(cancelled.finish_reason, FinishReason::Cancelled);
    let after_cancel = labels(&driver.snapshot().transcript);
    assert_eq!(
        after_cancel,
        ["hi", "call:call-probe", "result:call-probe"],
        "the cancelled turn start committed no resolution"
    );

    let LoopStep::Finished(result) = driver.next().await.unwrap() else {
        panic!("expected the requeued resolution to drive a turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Completed);
    assert_eq!(labels(&result.items), ["reply"]);
    assert_eq!(
        driver.snapshot().transcript.len(),
        after_cancel.len() + 2,
        "the requeued resolution and the reply both landed: {:?}",
        labels(&driver.snapshot().transcript)
    );
}

#[tokio::test]
async fn cancelling_pending_approvals_ends_the_turn_exactly_once() {
    let model = ProbeModel::new([Script::Tool]);
    let log = Log::default();
    let agent = Agent::builder()
        .model(model.clone())
        .tool_executor(ApprovalExecutor)
        .input(vec![Item::text(ItemKind::User, "hi")])
        .transcript_observer(Persistor(log.clone()))
        .delivery(Target::new(log.clone()))
        .build()
        .unwrap();
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    let LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_)) = driver.next().await.unwrap()
    else {
        panic!("expected an approval interrupt");
    };
    let Some(LoopStep::Finished(result)) = driver.cancel_pending_approvals().await.unwrap() else {
        panic!("expected cancelling the approvals to finish the turn");
    };
    assert_eq!(result.finish_reason, FinishReason::Cancelled);
    assert_eq!(
        log.count(|note| matches!(note, Note::BeforeFinish(_))),
        1,
        "{:?}",
        log.notes()
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{:?}",
        log.notes()
    );

    assert!(
        driver.cancel_pending_approvals().await.unwrap().is_none(),
        "there is nothing left to cancel"
    );
    assert_eq!(
        log.count(|note| matches!(note, Note::TurnFinished(_))),
        1,
        "{:?}",
        log.notes()
    );
}

#[tokio::test]
async fn dropping_next_inside_prefinish_loses_the_output_and_retirement_redelivers() {
    // The documented hard-abort hole: prefinish is awaited before anything
    // commits, so a dropped future there strands the turn.
    let model = ProbeModel::new([Script::Reply]);
    let log = Log::default();

    /// Hangs on prefinish so the test can drop the driver future inside it.
    struct HangOnPrefinish(Log);

    #[async_trait]
    impl NativeDelivery for HangOnPrefinish {
        async fn deliver(&self, fact: NativeFact<'_>, _: HookCtx<'_>) -> Result<(), DeliveryError> {
            if let NativeFact::BeforeFinish(result) = fact {
                self.0
                    .push(Note::BeforeFinish(result.finish_reason.clone()));
                std::future::pending::<()>().await;
            }
            if let NativeFact::TurnFinished(result) = fact {
                self.0
                    .push(Note::TurnFinished(result.finish_reason.clone()));
            }
            Ok(())
        }
    }

    let agent = agent(&model, &ProbeExecutor::default(), |builder| {
        builder
            .transcript_observer(Persistor(log.clone()))
            .delivery(HangOnPrefinish(log.clone()))
    });
    let mut driver = agent.start(SessionConfig::new("session")).await.unwrap();

    assert!(
        timeout(Duration::from_millis(50), driver.next())
            .await
            .is_err(),
        "the hanging prefinish delivery should not let next() resolve"
    );
    assert_eq!(
        labels(&driver.snapshot().transcript),
        ["hi"],
        "the terminal output candidate is lost, not committed"
    );

    // The turn is still active, so retirement re-runs the transition — and
    // reaches the same hanging prefinish, which is why this is a timeout.
    assert!(
        timeout(Duration::from_millis(50), driver.retire_interrupted_turn())
            .await
            .is_err(),
        "retirement re-enters the hanging prefinish delivery"
    );
    assert_eq!(
        log.notes(),
        [
            Note::Append("hi".into()),
            Note::BeforeFinish(FinishReason::Completed),
            Note::BeforeFinish(FinishReason::Cancelled),
        ],
        "prefinish is delivered twice for one logical turn: once for the lost \
         output candidate, once for the cancelled retirement"
    );
}
