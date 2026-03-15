use std::sync::Arc;

use agentkit_capabilities::CapabilityContext;
use agentkit_core::{
    Delta, FinishReason, Item, ItemKind, MetadataMap, Part, SessionId, ToolCallPart, ToolOutput,
    ToolResultPart, Usage,
};
use agentkit_tools_core::{
    ApprovalDecision, ApprovalRequest, AuthRequest, AuthResolution, BasicToolExecutor,
    PermissionChecker, ToolContext, ToolError, ToolExecutionOutcome, ToolExecutor, ToolRegistry,
    ToolRequest, ToolSpec,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub session_id: SessionId,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnRequest {
    pub session_id: SessionId,
    pub turn_id: agentkit_core::TurnId,
    pub transcript: Vec<Item>,
    pub available_tools: Vec<ToolSpec>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelTurnResult {
    pub finish_reason: FinishReason,
    pub output_items: Vec<Item>,
    pub usage: Option<Usage>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelTurnEvent {
    Delta(Delta),
    ToolCall(ToolCallPart),
    Usage(Usage),
    Finished(ModelTurnResult),
}

#[async_trait]
pub trait ModelAdapter: Send + Sync {
    type Session: ModelSession;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError>;
}

#[async_trait]
pub trait ModelSession: Send {
    type Turn: ModelTurn;

    async fn begin_turn(&mut self, request: TurnRequest) -> Result<Self::Turn, LoopError>;
}

#[async_trait]
pub trait ModelTurn: Send {
    async fn next_event(&mut self) -> Result<Option<ModelTurnEvent>, LoopError>;
}

pub trait LoopObserver: Send {
    fn handle_event(&mut self, event: AgentEvent);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AgentEvent {
    RunStarted {
        session_id: SessionId,
    },
    TurnStarted {
        session_id: SessionId,
        turn_id: agentkit_core::TurnId,
    },
    InputAccepted {
        session_id: SessionId,
        items: Vec<Item>,
    },
    ContentDelta(Delta),
    ToolCallRequested(ToolCallPart),
    ApprovalRequired(ApprovalRequest),
    AuthRequired(AuthRequest),
    ApprovalResolved {
        approved: bool,
    },
    AuthResolved {
        provided: bool,
    },
    UsageUpdated(Usage),
    Warning {
        message: String,
    },
    RunFailed {
        message: String,
    },
    TurnFinished(TurnResult),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRequest {
    pub session_id: SessionId,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnResult {
    pub turn_id: agentkit_core::TurnId,
    pub finish_reason: FinishReason,
    pub items: Vec<Item>,
    pub usage: Option<Usage>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopInterrupt {
    ApprovalRequest(ApprovalRequest),
    AuthRequest(AuthRequest),
    AwaitingInput(InputRequest),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LoopStep {
    Interrupt(LoopInterrupt),
    Finished(TurnResult),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoopSnapshot {
    pub session_id: SessionId,
    pub transcript: Vec<Item>,
    pub pending_input: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DriverState {
    Idle,
    AwaitingApproval,
    AwaitingAuth,
}

pub struct Agent<M>
where
    M: ModelAdapter,
{
    model: M,
    tools: ToolRegistry,
    permissions: Arc<dyn PermissionChecker>,
    observers: Vec<Box<dyn LoopObserver>>,
}

impl<M> Agent<M>
where
    M: ModelAdapter,
{
    pub fn builder() -> AgentBuilder<M> {
        AgentBuilder::default()
    }

    pub async fn start(self, config: SessionConfig) -> Result<LoopDriver<M::Session>, LoopError> {
        let session_id = config.session_id.clone();
        let session = self.model.start_session(config).await?;
        let tool_executor = Arc::new(BasicToolExecutor::new(self.tools.clone()));
        let tool_specs = tool_executor.specs();
        let mut driver = LoopDriver {
            session_id: session_id.clone(),
            session: Some(session),
            tool_executor,
            tool_specs,
            permissions: self.permissions,
            observers: self.observers,
            transcript: Vec::new(),
            pending_input: Vec::new(),
            pending_approval: None,
            pending_auth: None,
            next_turn_index: 1,
            state: DriverState::Idle,
        };
        driver.emit(AgentEvent::RunStarted { session_id });
        Ok(driver)
    }
}

pub struct AgentBuilder<M>
where
    M: ModelAdapter,
{
    model: Option<M>,
    tools: ToolRegistry,
    permissions: Arc<dyn PermissionChecker>,
    observers: Vec<Box<dyn LoopObserver>>,
}

impl<M> Default for AgentBuilder<M>
where
    M: ModelAdapter,
{
    fn default() -> Self {
        Self {
            model: None,
            tools: ToolRegistry::new(),
            permissions: Arc::new(AllowAllPermissions),
            observers: Vec::new(),
        }
    }
}

impl<M> AgentBuilder<M>
where
    M: ModelAdapter,
{
    pub fn model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    pub fn tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn permissions(mut self, permissions: impl PermissionChecker + 'static) -> Self {
        self.permissions = Arc::new(permissions);
        self
    }

    pub fn observer(mut self, observer: impl LoopObserver + 'static) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    pub fn build(self) -> Result<Agent<M>, LoopError> {
        let model = self
            .model
            .ok_or_else(|| LoopError::InvalidState("model adapter is required".into()))?;
        Ok(Agent {
            model,
            tools: self.tools,
            permissions: self.permissions,
            observers: self.observers,
        })
    }
}

pub struct LoopDriver<S>
where
    S: ModelSession,
{
    session_id: SessionId,
    session: Option<S>,
    tool_executor: Arc<dyn ToolExecutor>,
    tool_specs: Vec<ToolSpec>,
    permissions: Arc<dyn PermissionChecker>,
    observers: Vec<Box<dyn LoopObserver>>,
    transcript: Vec<Item>,
    pending_input: Vec<Item>,
    pending_approval: Option<ApprovalRequest>,
    pending_auth: Option<AuthRequest>,
    next_turn_index: u64,
    state: DriverState,
}

impl<S> LoopDriver<S>
where
    S: ModelSession,
{
    pub fn submit_input(&mut self, input: Vec<Item>) -> Result<(), LoopError> {
        if self.state != DriverState::Idle {
            return Err(LoopError::InvalidState(
                "cannot submit input while an interrupt is pending".into(),
            ));
        }
        self.emit(AgentEvent::InputAccepted {
            session_id: self.session_id.clone(),
            items: input.clone(),
        });
        self.pending_input.extend(input);
        Ok(())
    }

    pub fn resolve_approval(&mut self, decision: ApprovalDecision) -> Result<(), LoopError> {
        let Some(_request) = self.pending_approval.take() else {
            return Err(LoopError::InvalidState(
                "no approval request is pending".into(),
            ));
        };
        self.state = DriverState::Idle;
        self.emit(AgentEvent::ApprovalResolved {
            approved: matches!(decision, ApprovalDecision::Approve),
        });
        Ok(())
    }

    pub fn resolve_auth(&mut self, resolution: AuthResolution) -> Result<(), LoopError> {
        let Some(_request) = self.pending_auth.take() else {
            return Err(LoopError::InvalidState("no auth request is pending".into()));
        };
        self.state = DriverState::Idle;
        self.emit(AgentEvent::AuthResolved {
            provided: matches!(resolution, AuthResolution::Provided(_)),
        });
        Ok(())
    }

    pub fn snapshot(&self) -> LoopSnapshot {
        LoopSnapshot {
            session_id: self.session_id.clone(),
            transcript: self.transcript.clone(),
            pending_input: self.pending_input.clone(),
        }
    }

    pub async fn next(&mut self) -> Result<LoopStep, LoopError> {
        if self.state != DriverState::Idle {
            return Err(LoopError::InvalidState(
                "cannot advance while an interrupt is pending".into(),
            ));
        }

        if self.pending_input.is_empty() {
            return Ok(LoopStep::Interrupt(LoopInterrupt::AwaitingInput(
                InputRequest {
                    session_id: self.session_id.clone(),
                    reason: "driver is waiting for input".into(),
                },
            )));
        }

        let turn_id = agentkit_core::TurnId::new(format!("turn-{}", self.next_turn_index));
        self.next_turn_index += 1;
        self.emit(AgentEvent::TurnStarted {
            session_id: self.session_id.clone(),
            turn_id: turn_id.clone(),
        });

        self.transcript.append(&mut self.pending_input);
        loop {
            let request = TurnRequest {
                session_id: self.session_id.clone(),
                turn_id: turn_id.clone(),
                transcript: self.transcript.clone(),
                available_tools: self.tool_specs.clone(),
                metadata: MetadataMap::new(),
            };

            let session = self
                .session
                .as_mut()
                .ok_or_else(|| LoopError::InvalidState("model session is not available".into()))?;
            let mut turn = session.begin_turn(request).await?;
            let mut saw_tool_call = false;
            let mut tool_results = Vec::new();

            while let Some(event) = turn.next_event().await? {
                match event {
                    ModelTurnEvent::Delta(delta) => self.emit(AgentEvent::ContentDelta(delta)),
                    ModelTurnEvent::Usage(usage) => self.emit(AgentEvent::UsageUpdated(usage)),
                    ModelTurnEvent::ToolCall(call) => {
                        saw_tool_call = true;
                        self.emit(AgentEvent::ToolCallRequested(call.clone()));

                        let tool_request = ToolRequest {
                            call_id: call.id.clone(),
                            tool_name: agentkit_tools_core::ToolName::new(call.name.clone()),
                            input: call.input.clone(),
                            session_id: self.session_id.clone(),
                            turn_id: turn_id.clone(),
                            metadata: call.metadata.clone(),
                        };
                        let tool_metadata = tool_request.metadata.clone();
                        let mut tool_ctx = ToolContext {
                            capability: CapabilityContext {
                                session_id: Some(&self.session_id),
                                turn_id: Some(&turn_id),
                                metadata: &tool_metadata,
                            },
                            permissions: self.permissions.as_ref(),
                            resources: &(),
                        };

                        match self
                            .tool_executor
                            .execute(tool_request, &mut tool_ctx)
                            .await
                        {
                            ToolExecutionOutcome::Completed(result) => {
                                tool_results.push(Item {
                                    id: None,
                                    kind: ItemKind::Tool,
                                    parts: vec![Part::ToolResult(result.result)],
                                    metadata: result.metadata,
                                });
                            }
                            ToolExecutionOutcome::Interrupted(
                                agentkit_tools_core::ToolInterruption::ApprovalRequired(request),
                            ) => {
                                self.pending_approval = Some(request.clone());
                                self.state = DriverState::AwaitingApproval;
                                self.emit(AgentEvent::ApprovalRequired(request.clone()));
                                return Ok(LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(
                                    request,
                                )));
                            }
                            ToolExecutionOutcome::Interrupted(
                                agentkit_tools_core::ToolInterruption::AuthRequired(request),
                            ) => {
                                self.pending_auth = Some(request.clone());
                                self.state = DriverState::AwaitingAuth;
                                self.emit(AgentEvent::AuthRequired(request.clone()));
                                return Ok(LoopStep::Interrupt(LoopInterrupt::AuthRequest(
                                    request,
                                )));
                            }
                            ToolExecutionOutcome::Failed(error) => {
                                self.emit(AgentEvent::Warning {
                                    message: format!("tool {} failed: {}", call.name, error),
                                });
                                tool_results.push(Item {
                                    id: None,
                                    kind: ItemKind::Tool,
                                    parts: vec![Part::ToolResult(ToolResultPart {
                                        call_id: call.id.clone(),
                                        output: ToolOutput::Text(error.to_string()),
                                        is_error: true,
                                        metadata: call.metadata.clone(),
                                    })],
                                    metadata: MetadataMap::new(),
                                });
                            }
                        }
                    }
                    ModelTurnEvent::Finished(result) => {
                        self.transcript.extend(result.output_items.clone());

                        if saw_tool_call {
                            self.transcript.append(&mut tool_results);
                            break;
                        }

                        let turn_result = TurnResult {
                            turn_id,
                            finish_reason: result.finish_reason,
                            items: result.output_items,
                            usage: result.usage,
                            metadata: result.metadata,
                        };
                        self.emit(AgentEvent::TurnFinished(turn_result.clone()));
                        return Ok(LoopStep::Finished(turn_result));
                    }
                }
            }

            if saw_tool_call {
                continue;
            }

            return Err(LoopError::Provider(
                "model turn ended without a Finished event".into(),
            ));
        }
    }

    fn emit(&mut self, event: AgentEvent) {
        for observer in &mut self.observers {
            observer.handle_event(event.clone());
        }
    }
}

struct AllowAllPermissions;

impl PermissionChecker for AllowAllPermissions {
    fn evaluate(
        &self,
        _request: &dyn agentkit_tools_core::PermissionRequest,
    ) -> agentkit_tools_core::PermissionDecision {
        agentkit_tools_core::PermissionDecision::Allow
    }
}

#[derive(Debug, Error)]
pub enum LoopError {
    #[error("invalid driver state: {0}")]
    InvalidState(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("tool error: {0}")]
    Tool(#[from] ToolError),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use agentkit_core::{ItemKind, Part, TextPart, ToolCallId, ToolOutput, ToolResultPart};
    use agentkit_tools_core::{
        FileSystemPermissionRequest, PermissionCode, PermissionDecision, PermissionDenial, Tool,
        ToolAnnotations, ToolName, ToolResult, ToolSpec,
    };
    use serde_json::{Value, json};

    use super::*;

    struct FakeAdapter;

    struct FakeSession;

    struct FakeTurn {
        events: VecDeque<ModelTurnEvent>,
    }

    #[async_trait]
    impl ModelAdapter for FakeAdapter {
        type Session = FakeSession;

        async fn start_session(&self, _config: SessionConfig) -> Result<Self::Session, LoopError> {
            Ok(FakeSession)
        }
    }

    #[async_trait]
    impl ModelSession for FakeSession {
        type Turn = FakeTurn;

        async fn begin_turn(&mut self, request: TurnRequest) -> Result<Self::Turn, LoopError> {
            let has_tool_result = request.transcript.iter().any(|item| {
                item.kind == ItemKind::Tool
                    && item
                        .parts
                        .iter()
                        .any(|part| matches!(part, Part::ToolResult(_)))
            });
            let tool_name = request
                .available_tools
                .first()
                .map(|tool| tool.name.0.clone())
                .unwrap_or_else(|| "echo".into());

            let events = if has_tool_result {
                let result_text = request
                    .transcript
                    .iter()
                    .rev()
                    .find_map(|item| {
                        item.parts.iter().find_map(|part| match part {
                            Part::ToolResult(ToolResultPart {
                                output: ToolOutput::Text(text),
                                ..
                            }) => Some(text.clone()),
                            _ => None,
                        })
                    })
                    .unwrap_or_else(|| "missing".into());

                VecDeque::from([ModelTurnEvent::Finished(ModelTurnResult {
                    finish_reason: FinishReason::Completed,
                    output_items: vec![Item {
                        id: None,
                        kind: ItemKind::Assistant,
                        parts: vec![Part::Text(TextPart {
                            text: format!("tool said: {result_text}"),
                            metadata: MetadataMap::new(),
                        })],
                        metadata: MetadataMap::new(),
                    }],
                    usage: None,
                    metadata: MetadataMap::new(),
                })])
            } else {
                VecDeque::from([
                    ModelTurnEvent::ToolCall(agentkit_core::ToolCallPart {
                        id: ToolCallId::new("call-1"),
                        name: tool_name.clone(),
                        input: json!({ "value": "pong" }),
                        metadata: MetadataMap::new(),
                    }),
                    ModelTurnEvent::Finished(ModelTurnResult {
                        finish_reason: FinishReason::ToolCall,
                        output_items: vec![Item {
                            id: None,
                            kind: ItemKind::Assistant,
                            parts: vec![Part::ToolCall(agentkit_core::ToolCallPart {
                                id: ToolCallId::new("call-1"),
                                name: tool_name,
                                input: json!({ "value": "pong" }),
                                metadata: MetadataMap::new(),
                            })],
                            metadata: MetadataMap::new(),
                        }],
                        usage: None,
                        metadata: MetadataMap::new(),
                    }),
                ])
            };

            Ok(FakeTurn { events })
        }
    }

    #[async_trait]
    impl ModelTurn for FakeTurn {
        async fn next_event(&mut self) -> Result<Option<ModelTurnEvent>, LoopError> {
            Ok(self.events.pop_front())
        }
    }

    #[derive(Clone)]
    struct EchoTool {
        spec: ToolSpec,
    }

    impl Default for EchoTool {
        fn default() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("echo"),
                    description: "Echo back a value".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "value": { "type": "string" }
                        },
                        "required": ["value"],
                        "additionalProperties": false
                    }),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn proposed_requests(
            &self,
            request: &agentkit_tools_core::ToolRequest,
        ) -> Result<
            Vec<Box<dyn agentkit_tools_core::PermissionRequest>>,
            agentkit_tools_core::ToolError,
        > {
            Ok(vec![Box::new(FileSystemPermissionRequest::Read {
                path: "/tmp/echo".into(),
                metadata: request.metadata.clone(),
            })])
        }

        async fn invoke(
            &self,
            request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            let value = request
                .input
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    agentkit_tools_core::ToolError::InvalidInput("missing value".into())
                })?;

            Ok(ToolResult {
                result: ToolResultPart {
                    call_id: request.call_id,
                    output: ToolOutput::Text(value.into()),
                    is_error: false,
                    metadata: MetadataMap::new(),
                },
                duration: None,
                metadata: MetadataMap::new(),
            })
        }
    }

    struct DenyFsReads;

    impl PermissionChecker for DenyFsReads {
        fn evaluate(
            &self,
            request: &dyn agentkit_tools_core::PermissionRequest,
        ) -> PermissionDecision {
            if request.kind() == "filesystem.read" {
                return PermissionDecision::Deny(PermissionDenial {
                    code: PermissionCode::PathNotAllowed,
                    message: "reads denied in test".into(),
                    metadata: MetadataMap::new(),
                });
            }

            PermissionDecision::Allow
        }
    }

    #[derive(Clone)]
    struct AuthTool {
        spec: ToolSpec,
    }

    impl Default for AuthTool {
        fn default() -> Self {
            Self {
                spec: ToolSpec {
                    name: ToolName::new("auth-tool"),
                    description: "Always requires auth".into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                },
            }
        }
    }

    #[async_trait]
    impl Tool for AuthTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            _request: agentkit_tools_core::ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, agentkit_tools_core::ToolError> {
            let mut details = MetadataMap::new();
            details.insert("server_id".into(), json!("mock"));
            details.insert("scope".into(), json!("secret.read"));

            Err(agentkit_tools_core::ToolError::AuthRequired(AuthRequest {
                provider: "mcp.mock".into(),
                details,
            }))
        }
    }

    #[tokio::test]
    async fn loop_continues_after_completed_tool_call() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-1"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();

        match result {
            LoopStep::Finished(turn) => {
                assert_eq!(turn.finish_reason, FinishReason::Completed);
                assert_eq!(turn.items.len(), 1);
                match &turn.items[0].parts[0] {
                    Part::Text(text) => assert_eq!(text.text, "tool said: pong"),
                    other => panic!("unexpected part: {other:?}"),
                }
            }
            other => panic!("unexpected loop step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_uses_injected_permission_checker() {
        let tools = ToolRegistry::new().with(EchoTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(DenyFsReads)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-2"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();

        match result {
            LoopStep::Finished(turn) => match &turn.items[0].parts[0] {
                Part::Text(text) => assert!(text.text.contains("tool permission denied")),
                other => panic!("unexpected part: {other:?}"),
            },
            other => panic!("unexpected loop step: {other:?}"),
        }
    }

    #[tokio::test]
    async fn loop_surfaces_auth_interruptions_from_tools() {
        let tools = ToolRegistry::new().with(AuthTool::default());
        let agent = Agent::builder()
            .model(FakeAdapter)
            .tools(tools)
            .permissions(AllowAllPermissions)
            .build()
            .unwrap();

        let mut driver = agent
            .start(SessionConfig {
                session_id: SessionId::new("session-3"),
                metadata: MetadataMap::new(),
            })
            .await
            .unwrap();

        driver
            .submit_input(vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "ping".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }])
            .unwrap();

        let result = driver.next().await.unwrap();

        match result {
            LoopStep::Interrupt(LoopInterrupt::AuthRequest(request)) => {
                assert_eq!(request.provider, "mcp.mock");
                assert_eq!(request.details.get("scope"), Some(&json!("secret.read")));
            }
            other => panic!("unexpected loop step: {other:?}"),
        }
    }
}
