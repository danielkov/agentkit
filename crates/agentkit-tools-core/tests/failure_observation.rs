use agentkit_core::{failure::*, retry::*};
use agentkit_tools_core::{
    FailureObservationSlot, ObservationPublishError as Error, ObservationUpdate as Update,
};

fn receipt(id: &str) -> HostFatalReceipt {
    HostFatalReceipt {
        session_id: HostReceiptId::new("session").unwrap(),
        event_id: HostReceiptId::new(id).unwrap(),
        storage: FatalStorage::Unavailable,
    }
}
fn retry() -> ProviderFailure {
    ProviderFailure {
        route: ProviderRoute::Unknown,
        reason: ProviderFailureReason::Cancelled,
        last_attempt_reason: None,
        upstream: ProviderClassification::default(),
        accounting: RetryAccounting::default(),
    }
}
#[test]
fn independent_leaves_survive_updates_and_seal() {
    let slot = FailureObservationSlot::new();
    let publisher = slot.publisher();
    assert!(slot.snapshot().is_none());
    let mut effects = PossibleEffects::default();
    effects.source = ObservationSource::LocalSession;
    effects.tool_execution_start_reported = true;
    assert_eq!(publisher.publish_effects(effects), Ok(Update::Published));
    assert_eq!(
        publisher.publish_receipt(receipt("e-1")),
        Ok(Update::Published)
    );
    assert_eq!(publisher.publish_retry(retry()), Ok(Update::Published));
    effects.tool_execution_completion_reported = true;
    assert_eq!(publisher.publish_effects(effects), Ok(Update::Published));
    let frozen = slot.seal().unwrap();
    assert_eq!(frozen.effects(), Some(&effects));
    assert_eq!(frozen.receipt(), Some(&receipt("e-1")));
    assert_eq!(frozen.retry(), Some(&retry()));
    assert_eq!(slot.seal(), Some(frozen.clone()));
    assert_eq!(publisher.publish_effects(effects), Err(Error::Sealed));
    assert_eq!(publisher.publish_retry(retry()), Err(Error::Sealed));
    assert_eq!(
        publisher.publish_receipt(receipt("e-1")),
        Err(Error::Sealed)
    );
    let bytes = serde_json::to_vec(&frozen).unwrap();
    assert_eq!(FailureObservations::from_slice(&bytes).unwrap(), frozen);
}
#[test]
fn conflicts_regressions_and_invalid_values_do_not_mutate() {
    let slot = FailureObservationSlot::new();
    let publisher = slot.publisher();
    let mut effects = PossibleEffects::default();
    effects.source = ObservationSource::LocalSession;
    effects.tool_execution_start_reported = true;
    publisher.publish_effects(effects).unwrap();
    assert_eq!(publisher.publish_effects(effects), Ok(Update::Unchanged));
    let mut regression = effects;
    regression.tool_execution_start_reported = false;
    assert_eq!(
        publisher.publish_effects(regression),
        Err(Error::Regression)
    );
    regression = effects;
    regression.source = ObservationSource::AcpNotifications;
    assert_eq!(publisher.publish_effects(regression), Err(Error::Conflict));
    publisher.publish_receipt(receipt("e-1")).unwrap();
    assert_eq!(
        publisher.publish_receipt(receipt("e-1")),
        Ok(Update::Unchanged)
    );
    assert_eq!(
        publisher.publish_receipt(receipt("e-2")),
        Err(Error::Conflict)
    );
    publisher.publish_retry(retry()).unwrap();
    let mut other = retry();
    other.accounting.attempts = 1;
    assert_eq!(publisher.publish_retry(other), Err(Error::Conflict));
    other.upstream.http_status = Some(900);
    assert_eq!(publisher.publish_retry(other), Err(Error::Invalid));
    let frozen = slot.snapshot().unwrap();
    assert_eq!(frozen.effects(), Some(&effects));
    assert_eq!(frozen.receipt(), Some(&receipt("e-1")));
    assert_eq!(frozen.retry(), Some(&retry()));
}
#[test]
fn publication_and_sealing_are_linearized_without_cross_task_state() {
    let first = FailureObservationSlot::new();
    let second = FailureObservationSlot::new();
    let publisher = first.publisher();
    let thread = std::thread::spawn(move || publisher.publish_receipt(receipt("e-1")));
    thread.join().unwrap().unwrap();
    assert!(first.seal().unwrap().receipt().is_some());
    assert!(second.snapshot().is_none());
    second.seal();
    let publisher = second.publisher();
    assert_eq!(
        std::thread::spawn(move || publisher.publish_receipt(receipt("e-2")))
            .join()
            .unwrap(),
        Err(Error::Sealed)
    );
}
#[test]
fn observation_wire_is_closed_at_nested_boundaries() {
    for value in [
        serde_json::json!({"effects":{"replay_safe":true}}),
        serde_json::json!({"receipt":{"session_id":"s","event_id":"e","storage":"unavailable","path":"PRIVATE"}}),
        serde_json::json!({"metadata":{"version":1}}),
    ] {
        assert!(FailureObservations::from_slice(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
