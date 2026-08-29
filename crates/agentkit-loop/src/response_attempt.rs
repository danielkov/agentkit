//! Opt-in support for replacing an already visible upstream response attempt.

use agentkit_core::{Delta, PartId, PartKind};

use crate::{ModelTurnEvent, SessionConfig};

/// Session metadata key used to opt in to response-attempt replacement.
pub const SESSION_METADATA_KEY: &str = "kit.internal.acp_v2_response_attempt_replacement";
const MARKER_PART_ID: &str = "\0kit.response-attempt-replacement";

/// Enables response-attempt replacement for an upstream consumer that knows
/// how to discard all output from the preceding attempt.
pub fn enable(config: &mut SessionConfig) {
    config
        .metadata
        .insert(SESSION_METADATA_KEY.into(), serde_json::Value::Bool(true));
}

/// Returns whether the upstream consumer explicitly opted in.
pub fn enabled(config: &SessionConfig) -> bool {
    config
        .metadata
        .get(SESSION_METADATA_KEY)
        .and_then(serde_json::Value::as_bool)
        == Some(true)
}

/// Builds the reserved event that invalidates the preceding response attempt.
pub fn marker_event() -> ModelTurnEvent {
    ModelTurnEvent::Delta(Delta::BeginPart {
        part_id: PartId::new(MARKER_PART_ID),
        kind: PartKind::Custom,
    })
}

/// Returns whether a content delta is the reserved replacement marker.
pub fn is_marker(delta: &Delta) -> bool {
    matches!(
        delta,
        Delta::BeginPart { part_id, kind: PartKind::Custom }
            if part_id.0 == MARKER_PART_ID
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_is_explicit_and_marker_is_reserved() {
        let mut config = SessionConfig::new("test");
        assert!(!enabled(&config));
        enable(&mut config);
        assert!(enabled(&config));

        let ModelTurnEvent::Delta(delta) = marker_event() else {
            panic!("marker must be a delta");
        };
        assert!(is_marker(&delta));
    }
}
