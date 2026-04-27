//! End-to-end approval interrupt: a tool declares a permission request
//! that the host's [`PermissionChecker`] gates behind approval. The loop
//! emits [`LoopInterrupt::ApprovalRequest`]; the test resolves it via
//! `pending.approve(&mut driver)`, and the tool subsequently executes.

use std::any::Any;

use agentkit_core::{
    Item, ItemKind, MetadataMap, MetadataMap as Meta, Part, ToolCallId, ToolCallPart, ToolOutput,
    ToolResultPart,
};
use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_tools_core::{
    ApprovalRequest, ApprovalReason, PermissionChecker, PermissionDecision, PermissionRequest,
    Tool, ToolContext, ToolError, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::json;

const APPROVAL_KIND: &str = "test.dangerous_op";

struct DangerRequest {
    summary: String,
    metadata: MetadataMap,
}

impl PermissionRequest for DangerRequest {
    fn kind(&self) -> &'static str {
        APPROVAL_KIND
    }
    fn summary(&self) -> String {
        self.summary.clone()
    }
    fn metadata(&self) -> &MetadataMap {
        &self.metadata
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct GatedTool {
    spec: ToolSpec,
}

impl GatedTool {
    fn new() -> Self {
        Self {
            spec: ToolSpec::new(
                "danger",
                "Performs a dangerous op gated by approval.",
                json!({ "type": "object", "additionalProperties": true }),
            ),
        }
    }
}

#[async_trait]
impl Tool for GatedTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn proposed_requests(
        &self,
        _request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        Ok(vec![Box::new(DangerRequest {
            summary: "execute danger".into(),
            metadata: Meta::new(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::new(ToolResultPart::success(
            request.call_id,
            ToolOutput::text("danger:executed"),
        )))
    }
}

struct ApproveDangerOnly;

impl PermissionChecker for ApproveDangerOnly {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision {
        if request.kind() == APPROVAL_KIND {
            PermissionDecision::RequireApproval(ApprovalRequest::new(
                "approval:test",
                APPROVAL_KIND,
                ApprovalReason::EscalatedRisk,
                request.summary(),
            ))
        } else {
            PermissionDecision::Allow
        }
    }
}

#[tokio::test]
async fn approval_interrupt_pauses_then_resumes_after_approve() {
    let mock = MockAdapter::new();
    mock.enqueue(TurnScript::tool_call(ToolCallPart::new(
        ToolCallId::new("call-danger"),
        "danger",
        json!({}),
    )));
    mock.enqueue(TurnScript::text("done"));

    let agent = Agent::builder()
        .model(mock.clone())
        .add_tool_source(ToolRegistry::new().with(GatedTool::new()))
        .permissions(ApproveDangerOnly)
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new("approval-flow"),
            vec![Item::text(ItemKind::User, "go")],
        )
        .await
        .unwrap();

    // Drive until the approval interrupt fires.
    let pending = loop {
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => break pending,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                panic!("expected approval interrupt before AwaitingInput")
            }
            LoopStep::Finished(_) => panic!("expected approval interrupt before Finished"),
        }
    };
    assert_eq!(pending.request.request_kind, APPROVAL_KIND);
    assert_eq!(pending.request.summary, "execute danger");

    // Approve and resume.
    pending.approve(&mut driver).unwrap();

    let final_turn = loop {
        match driver.next().await.unwrap() {
            LoopStep::Finished(result) => break result,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            other => panic!("unexpected step after approval: {other:?}"),
        }
    };

    // Final assistant text comes from the second scripted turn.
    let final_text = final_turn
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::Assistant)
        .flat_map(|item| item.parts.iter())
        .find_map(|part| match part {
            Part::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .expect("final assistant text");
    assert_eq!(final_text, "done");

    // Continuation turn saw the approved tool's result in its transcript.
    let observed = mock.observed();
    assert_eq!(observed.len(), 2);
    let approved_result = observed[1]
        .transcript
        .iter()
        .flat_map(|item| item.parts.iter())
        .find_map(|part| match part {
            Part::ToolResult(result) => match &result.output {
                ToolOutput::Text(text) => Some(text.clone()),
                _ => None,
            },
            _ => None,
        })
        .expect("approved tool result in continuation transcript");
    assert_eq!(approved_result, "danger:executed");
}

#[tokio::test]
async fn approval_interrupt_denies_to_synthetic_error_result() {
    let mock = MockAdapter::new();
    mock.enqueue(TurnScript::tool_call(ToolCallPart::new(
        ToolCallId::new("call-danger"),
        "danger",
        json!({}),
    )));
    mock.enqueue(TurnScript::text("understood"));

    let agent = Agent::builder()
        .model(mock.clone())
        .add_tool_source(ToolRegistry::new().with(GatedTool::new()))
        .permissions(ApproveDangerOnly)
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new("approval-deny"),
            vec![Item::text(ItemKind::User, "go")],
        )
        .await
        .unwrap();

    let pending = loop {
        match driver.next().await.unwrap() {
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => break pending,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            other => panic!("expected approval interrupt, got {other:?}"),
        }
    };

    pending
        .deny_with_reason(&mut driver, "policy says no")
        .unwrap();

    loop {
        match driver.next().await.unwrap() {
            LoopStep::Finished(_) => break,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            other => panic!("unexpected step after deny: {other:?}"),
        }
    }

    let observed = mock.observed();
    let denial = observed[1]
        .transcript
        .iter()
        .flat_map(|item| item.parts.iter())
        .find_map(|part| match part {
            Part::ToolResult(result) if result.is_error => match &result.output {
                ToolOutput::Text(text) => Some(text.clone()),
                _ => None,
            },
            _ => None,
        })
        .expect("denied tool result with is_error=true");
    assert_eq!(denial, "policy says no");
}
