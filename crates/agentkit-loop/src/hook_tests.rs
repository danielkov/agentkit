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
}

#[derive(Clone, Default)]
struct ProbeModel {
    scripts: Arc<Mutex<VecDeque<Script>>>,
    requests: Arc<Mutex<Vec<TurnRequest>>>,
    configs: Arc<Mutex<Vec<SessionConfig>>>,
}

impl ProbeModel {
    fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Arc::new(Mutex::new(scripts.into_iter().collect())),
            ..Self::default()
        }
    }
}

struct ProbeSession(ProbeModel);
struct ProbeTurn(VecDeque<ModelTurnEvent>);

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
            Some(Script::Reply) => (
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
        Ok(ProbeTurn(events))
    }
}

#[async_trait]
impl ModelTurn for ProbeTurn {
    async fn next_event(
        &mut self,
        _: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        Ok(self.0.pop_front())
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
