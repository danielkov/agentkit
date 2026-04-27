//! [`BasicToolExecutor`] over multiple [`ToolSource`]s with the same tool
//! name. Verifies that [`CollisionPolicy::FirstWins`] and
//! [`CollisionPolicy::LastWins`] route invocations to the right source.

use std::sync::Arc;

use agentkit_core::{Item, ItemKind, ToolCallId, ToolCallPart, ToolOutput};
use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
use agentkit_integration_tests::mock_tool::StaticTool;
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_tools_core::{
    BasicToolExecutor, CollisionPolicy, ToolExecutor, ToolRegistry, ToolSource,
};
use serde_json::json;

#[tokio::test]
async fn first_wins_routes_to_earlier_source() {
    let (final_text, primary_calls, secondary_calls) =
        drive_collision(CollisionPolicy::FirstWins).await;
    assert_eq!(final_text, "primary");
    assert_eq!(primary_calls, 1, "primary source should answer the call");
    assert_eq!(secondary_calls, 0, "secondary should never be invoked");
}

#[tokio::test]
async fn last_wins_routes_to_later_source() {
    let (final_text, primary_calls, secondary_calls) =
        drive_collision(CollisionPolicy::LastWins).await;
    assert_eq!(final_text, "secondary");
    assert_eq!(primary_calls, 0, "primary should be shadowed");
    assert_eq!(secondary_calls, 1, "secondary source should answer the call");
}

async fn drive_collision(policy: CollisionPolicy) -> (String, usize, usize) {
    let primary = StaticTool::new("shared", "primary impl", ToolOutput::text("primary"));
    let secondary = StaticTool::new(
        "shared",
        "secondary impl",
        ToolOutput::text("secondary"),
    );

    let primary_source: Arc<dyn ToolSource> =
        Arc::new(ToolRegistry::new().with(primary.clone()));
    let secondary_source: Arc<dyn ToolSource> =
        Arc::new(ToolRegistry::new().with(secondary.clone()));

    let executor: Arc<dyn ToolExecutor> = Arc::new(
        BasicToolExecutor::new([primary_source, secondary_source]).with_collision_policy(policy),
    );

    // Two-turn script: call shared, then echo back the result body.
    let mock = MockAdapter::new();
    mock.enqueue(TurnScript::tool_call(ToolCallPart::new(
        ToolCallId::new("call-shared"),
        "shared",
        json!({}),
    )));
    // We don't know up front which source will answer, so emit a fixed
    // "OK" on the continuation turn and inspect the *previous* turn's
    // transcript (which carries the executed tool's result body).
    mock.enqueue(TurnScript::text("OK"));

    let agent = Agent::builder()
        .model(mock.clone())
        .tool_executor(executor)
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new("collision"),
            vec![Item::text(ItemKind::User, "go")],
        )
        .await
        .unwrap();

    loop {
        match driver.next().await.unwrap() {
            LoopStep::Finished(_) => break,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            other => panic!("unexpected step: {other:?}"),
        }
    }

    // The transcript handed to turn 2 contains the tool result body —
    // dig it out.
    let observed = mock.observed();
    let body = observed[1]
        .transcript
        .iter()
        .flat_map(|item| item.parts.iter())
        .find_map(|part| match part {
            agentkit_core::Part::ToolResult(result) => match &result.output {
                ToolOutput::Text(text) => Some(text.clone()),
                _ => None,
            },
            _ => None,
        })
        .expect("tool result text in continuation transcript");

    (body, primary.call_count(), secondary.call_count())
}
