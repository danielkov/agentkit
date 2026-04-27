//! Mid-session tool catalog mutations propagate to the model.
//!
//! The host owns the writer side of a [`dynamic_catalog`] pair; tools are
//! added/removed between turns. Each mutation should:
//!
//! - Surface as an [`AgentEvent::ToolCatalogChanged`] on the next turn
//!   boundary (drained by the loop and forwarded to observers).
//! - Cause the *next* [`TurnRequest`] handed to the model to reflect the
//!   updated catalog.

use std::sync::Arc;

use agentkit_core::{Item, ItemKind, ToolOutput};
use agentkit_integration_tests::mock_model::{MockAdapter, TurnScript};
use agentkit_integration_tests::mock_tool::StaticTool;
use agentkit_loop::{Agent, AgentEvent, LoopInterrupt, LoopObserver, LoopStep, SessionConfig};
use agentkit_tools_core::{ToolCatalogEvent, ToolName, ToolSource, dynamic_catalog};
use std::sync::Mutex;

#[derive(Clone, Default)]
struct EventCapture {
    events: Arc<Mutex<Vec<AgentEvent>>>,
}

impl EventCapture {
    fn catalog_events(&self) -> Vec<ToolCatalogEvent> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolCatalogChanged(event) => Some(event.clone()),
                _ => None,
            })
            .collect()
    }
}

impl LoopObserver for EventCapture {
    fn handle_event(&mut self, event: AgentEvent) {
        self.events.lock().unwrap().push(event);
    }
}

#[tokio::test]
async fn dynamic_registry_mutations_flow_to_next_turn() {
    let (writer, reader) = dynamic_catalog("dynamic");
    writer.upsert(Arc::new(StaticTool::new(
        "alpha",
        "Returns alpha-body.",
        ToolOutput::text("alpha-body"),
    )));
    let _ = reader.drain_catalog_events(); // ignore pre-session bootstrap

    // Three text-only turns. Catalog is mutated *between* turns; the
    // model's view (observed turns) and the observer's catalog events
    // must reflect the mutation on the very next turn.
    let mock = MockAdapter::new();
    mock.enqueue(TurnScript::text("t1"));
    mock.enqueue(TurnScript::text("t2"));
    mock.enqueue(TurnScript::text("t3"));

    let observer = EventCapture::default();
    let agent = Agent::builder()
        .model(mock.clone())
        .add_tool_source(reader)
        .observer(observer.clone())
        .build()
        .unwrap();

    let mut driver = agent
        .start(
            SessionConfig::new("dynamic-catalog"),
            vec![Item::text(ItemKind::User, "go")],
        )
        .await
        .unwrap();

    // --- Turn 1 -------------------------------------------------------
    drive_until_finished(&mut driver).await;
    assert!(observer.catalog_events().is_empty(), "no catalog churn yet");
    assert_eq!(mock.observed()[0].tool_names, vec!["alpha".to_string()]);

    // --- Mutate before turn 2: add `beta` -----------------------------
    writer.upsert(Arc::new(StaticTool::new(
        "beta",
        "Returns beta-body.",
        ToolOutput::text("beta-body"),
    )));

    // --- Turn 2 -------------------------------------------------------
    let pending = await_input_request(&mut driver).await;
    pending
        .submit(&mut driver, vec![Item::text(ItemKind::User, "next")])
        .unwrap();
    drive_until_finished(&mut driver).await;

    let cat_events = observer.catalog_events();
    assert!(
        cat_events.iter().any(|e| e.source == "dynamic"
            && e.added == vec!["beta".to_string()]
            && e.removed.is_empty()
            && e.changed.is_empty()),
        "expected an added=beta event, saw {cat_events:?}",
    );
    let mut tools_t2 = mock.observed()[1].tool_names.clone();
    tools_t2.sort();
    assert_eq!(tools_t2, vec!["alpha".to_string(), "beta".to_string()]);

    // --- Mutate before turn 3: remove alpha ---------------------------
    assert!(writer.remove(&ToolName::new("alpha")));

    // --- Turn 3 -------------------------------------------------------
    let pending = await_input_request(&mut driver).await;
    pending
        .submit(&mut driver, vec![Item::text(ItemKind::User, "again")])
        .unwrap();
    drive_until_finished(&mut driver).await;

    let cat_events = observer.catalog_events();
    assert!(
        cat_events.iter().any(|e| e.source == "dynamic"
            && e.removed == vec!["alpha".to_string()]
            && e.added.is_empty()
            && e.changed.is_empty()),
        "expected a removed=alpha event, saw {cat_events:?}",
    );
    assert_eq!(mock.observed()[2].tool_names, vec!["beta".to_string()]);
}

async fn drive_until_finished<S>(driver: &mut agentkit_loop::LoopDriver<S>)
where
    S: agentkit_loop::ModelSession,
{
    loop {
        match driver.next().await.unwrap() {
            LoopStep::Finished(_) => return,
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                panic!("script ran out unexpectedly")
            }
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                panic!("unexpected approval: {}", pending.request.summary)
            }
        }
    }
}

async fn await_input_request<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> agentkit_loop::InputRequest
where
    S: agentkit_loop::ModelSession,
{
    match driver.next().await.unwrap() {
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(req)) => req,
        other => panic!("expected AwaitingInput, got {other:?}"),
    }
}
