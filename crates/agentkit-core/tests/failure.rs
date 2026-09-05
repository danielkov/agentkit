use agentkit_core::failure::*;
use agentkit_core::retry::*;
use serde_json::json;
use std::time::Duration;

fn sample() -> FailureMetadataV1 {
    let mut effects = PossibleEffects::default();
    effects.source = ObservationSource::AcpNotifications;
    effects.tool_execution_start_reported = true;
    FailureMetadataV1::new(FailureCode::ChildFailed)
        .with_receipt(HostFatalReceipt {
            session_id: HostReceiptId::new("session-1").unwrap(),
            event_id: HostReceiptId::new("e-123_456").unwrap(),
            storage: FatalStorage::Unavailable,
        })
        .with_effects(effects)
        .with_retry(ProviderFailure {
            route: ProviderRoute::OpenAiResponses,
            reason: ProviderFailureReason::RetryExhausted,
            last_attempt_reason: Some(ProviderFailureReason::ResponseFailed),
            upstream: ProviderClassification::default(),
            accounting: RetryAccounting {
                attempts: u64::MAX,
                completed_backoff: Duration::MAX,
                elapsed: Duration::MAX,
            },
        })
        .unwrap()
}

#[test]
fn current_writer_roundtrips_exact_bounded_values() {
    let sample = sample();
    let bytes = serde_json::to_vec(&sample).unwrap();
    assert!(bytes.len() < MAX_FAILURE_METADATA_BYTES);
    assert_eq!(FailureMetadataV1::from_slice(&bytes).unwrap(), sample);
    let value = serde_json::to_value(sample).unwrap();
    assert_eq!(value["version"], 1);
    assert_eq!(
        value["retry"]["accounting"]["completed_backoff"]["nanos"],
        999_999_999
    );
    assert_eq!(value["receipt"]["storage"], "unavailable");
}

#[test]
fn absent_facts_are_unknown_and_effects_remain_incomplete() {
    let sample = FailureMetadataV1::from_slice(br#"{"version":1}"#).unwrap();
    assert_eq!(sample, FailureMetadataV1::default());
    assert_eq!(
        serde_json::to_value(sample).unwrap(),
        json!({"version":1,"code":"unknown"})
    );
    let effects: PossibleEffects = serde_json::from_value(json!({})).unwrap();
    assert!(effects.observation_incomplete());
    assert!(!effects.tool_execution_start_reported);
}

#[test]
fn identifiers_validate_without_echoing_rejected_input() {
    for value in [
        "",
        "../private",
        "customer@example.com",
        "line\nbreak",
        "é",
        &"x".repeat(129),
    ] {
        assert_eq!(
            HostReceiptId::new(value).unwrap_err().to_string(),
            "invalid failure metadata"
        );
        assert!(serde_json::from_value::<HostReceiptId>(json!(value)).is_err());
    }
    assert!(HostReceiptId::new("x".repeat(128)).is_ok());
}

#[test]
fn every_new_nested_object_is_closed_and_errors_are_static() {
    let original = serde_json::to_value(sample()).unwrap();
    for path in [
        "",
        "/retry",
        "/retry/upstream",
        "/retry/accounting",
        "/retry/accounting/elapsed",
        "/receipt",
        "/effects",
    ] {
        let mut malformed = original.clone();
        malformed
            .pointer_mut(path)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("PRIVATE_TOKEN".into(), json!("PRIVATE_BODY"));
        let error =
            FailureMetadataV1::from_slice(&serde_json::to_vec(&malformed).unwrap()).unwrap_err();
        assert_eq!(
            format!("{error:?}: {error}"),
            "FailureMetadataDecodeError: invalid failure metadata"
        );
    }
}

#[test]
fn malformed_versions_enums_counts_and_complete_claims_are_rejected() {
    for (path, bad) in [
        ("/version", json!(2)),
        ("/code", json!("private-code")),
        ("/retry/route", json!("https://private")),
        ("/retry/upstream/http_status", json!(99)),
        ("/retry/upstream/http_status", json!(600)),
        ("/retry/accounting/attempts", json!(-1)),
        ("/retry/accounting/attempts", json!(1.5)),
        ("/retry/accounting/elapsed/nanos", json!(1_000_000_000)),
        ("/effects/observation_incomplete", json!(false)),
        ("/receipt/storage", json!("pending")),
    ] {
        let mut value = serde_json::to_value(sample()).unwrap();
        *value
            .pointer_mut(path)
            .unwrap_or_else(|| panic!("missing path {path}")) = bad;
        assert!(
            FailureMetadataV1::from_slice(&serde_json::to_vec(&value).unwrap()).is_err(),
            "{path}"
        );
    }
}

#[test]
fn unknown_provider_spellings_normalize_without_retaining_private_strings() {
    let mut value = serde_json::to_value(sample()).unwrap();
    value["retry"]["upstream"]["code"] = json!("PRIVATE_PROVIDER_TEXT");
    let decoded = FailureMetadataV1::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(
        decoded.retry().unwrap().upstream.code,
        UpstreamErrorKind::Unknown
    );
    assert!(!format!("{decoded:?}").contains("PRIVATE"));
    assert!(!serde_json::to_string(&decoded).unwrap().contains("PRIVATE"));
}

#[test]
fn oversized_deep_duplicate_and_mixed_unrecognized_inputs_fail_closed() {
    for bytes in [
        vec![b' '; MAX_FAILURE_METADATA_BYTES + 1],
        br#"{"version":1,"version":2}"#.to_vec(),
        br#"{"version":1,"effects":{"source":"unknown","replay_safe":true}}"#.to_vec(),
        format!("{}0{}", "[".repeat(200), "]".repeat(200)).into_bytes(),
    ] {
        assert!(FailureMetadataV1::from_slice(&bytes).is_err());
    }
}

#[test]
fn standalone_legacy_retry_serde_stays_permissive() {
    let mut retry = serde_json::to_value(sample().retry().unwrap()).unwrap();
    retry["future_field"] = json!(true);
    assert!(serde_json::from_value::<ProviderFailure>(retry).is_ok());
}

#[test]
fn direct_metadata_deserialize_errors_do_not_quote_rejected_fields_or_variants() {
    for value in [
        json!({"version":1,"PRIVATE_FIELD":"PRIVATE_BODY"}),
        json!({"version":1,"code":"PRIVATE_CODE"}),
        json!({"version":1,"effects":{"source":"PRIVATE_SOURCE"}}),
    ] {
        let error = serde_json::from_value::<FailureMetadataV1>(value).unwrap_err();
        assert!(!error.to_string().contains("PRIVATE"));
    }
}
