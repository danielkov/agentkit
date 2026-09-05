use agentkit_core::failure::{FailureCode, FailureMetadataV1};
use agentkit_tools_core::{
    DiagnosticFailureKind, DiagnosticToolFailure, ToolError, ToolExecutionOutcome, ToolFailureKind,
};
use serde_json::json;

#[test]
fn historical_error_shapes_remain_readable_and_unchanged() {
    for value in [
        json!("Cancelled"),
        json!({"NotFound":"missing"}),
        json!({"InvalidInput":"old"}),
        json!({"ExecutionFailed":"old"}),
        json!({"Unavailable":"old"}),
        json!({"Internal":"old"}),
    ] {
        let error: ToolError = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(error).unwrap(), value);
    }
}

#[test]
fn diagnostic_failure_and_cancellation_are_native_and_payload_free() {
    for kind in [
        DiagnosticFailureKind::ExecutionFailed,
        DiagnosticFailureKind::Internal,
        DiagnosticFailureKind::Cancelled,
    ] {
        let error = ToolError::diagnostic(DiagnosticToolFailure {
            kind,
            metadata: FailureMetadataV1::new(FailureCode::ChildFailed),
        });
        assert_eq!(
            error.is_cancelled(),
            kind == DiagnosticFailureKind::Cancelled
        );
        assert_eq!(
            error.failure_metadata().unwrap().code(),
            FailureCode::ChildFailed
        );
        assert!(!error.is_permission_denied());
        let outcome = ToolExecutionOutcome::Failed(error.clone());
        let encoded = serde_json::to_value(&outcome).unwrap();
        assert_eq!(
            serde_json::from_value::<ToolExecutionOutcome>(encoded).unwrap(),
            outcome
        );
        assert_eq!(
            error.failure_info().metadata,
            error.failure_metadata().cloned()
        );
        assert!(!error.to_string().contains("session"));
    }
    assert!(ToolError::Cancelled.is_cancelled());
    assert_eq!(
        ToolError::ExecutionFailed("legacy".into()).failure_kind(),
        ToolFailureKind::ExecutionFailed
    );
    assert_eq!(
        ToolError::Unavailable("legacy".into())
            .failure_info()
            .metadata,
        None
    );
}

#[test]
fn diagnostic_schema_rejects_recursive_and_raw_errors() {
    for value in [
        json!({"Diagnostic":{"kind":"execution_failed","metadata":{"version":1},"source":"PRIVATE"}}),
        json!({"Diagnostic":{"kind":"unavailable","metadata":{"version":1}}}),
        json!({"Diagnostic":{"kind":"cancelled","metadata":{"version":99}}}),
        json!({"Diagnostic":{"kind":"internal","metadata":{"version":1,"cause":{"Internal":"PRIVATE"}}}}),
    ] {
        assert!(serde_json::from_value::<ToolError>(value).is_err());
    }
}

#[test]
fn new_diagnostic_decode_errors_never_echo_unknown_fields_or_kinds() {
    for value in [
        json!({"kind":"PRIVATE_KIND","metadata":{"version":1}}),
        json!({"kind":"cancelled","metadata":{"version":1},"PRIVATE_FIELD":"PRIVATE_BODY"}),
    ] {
        let error = serde_json::from_value::<DiagnosticToolFailure>(value.clone()).unwrap_err();
        assert!(!error.to_string().contains("PRIVATE"));
        let error =
            serde_json::from_value::<agentkit_tools_core::ToolFailureInfo>(value).unwrap_err();
        assert!(!error.to_string().contains("PRIVATE"));
    }
}
