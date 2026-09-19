//! Exercise validated content and real reservation transitions, not synthetic byte charges.
use super::*;

// Avoid multiplying the largest payload allocations under the parallel test runner.
static LARGE_MEDIA_TEST: Mutex<()> = Mutex::new(());

fn image(payload_bytes: usize) -> wire::ContentBlock {
    wire::ContentBlock::Image(wire::ImageContent::new(
        "A".repeat(payload_bytes),
        "image/png",
    ))
}

fn text_at_size(serialized_bytes: usize) -> Vec<wire::ContentBlock> {
    let empty = vec![wire::ContentBlock::Text(wire::TextContent::new(""))];
    let overhead = serde_json::to_vec(&empty).unwrap().len();
    vec![wire::ContentBlock::Text(wire::TextContent::new(
        "x".repeat(serialized_bytes - overhead),
    ))]
}

fn fixture() -> (AcpIntegration, wire::SessionId, Arc<IntegrationSession>) {
    let integration = AcpIntegration::default();
    let id = wire::SessionId::new("budget-session");
    let (client, _messages) = ClientHandle::channel();
    integration
        .bind_session(AcpSessionBinding::new(
            id.clone(),
            AgentkitSessionId::new("budget-agentkit"),
            client,
        ))
        .unwrap();
    let session = integration.session(&id).unwrap();
    session.injection.start_turn();
    (integration, id, session)
}

fn reserve(
    integration: &AcpIntegration,
    id: &wire::SessionId,
    content: Vec<wire::ContentBlock>,
) -> Result<ReservedInject, agent_client_protocol::Error> {
    integration.reserve_inject(wire::InjectSessionRequest::new(
        id.clone(),
        wire::SessionInjectMode::Steer,
        content,
    ))
}

fn ready(mut reservation: ReservedInject) -> wire::MessageId {
    let id = reservation.message_id().clone();
    reservation.commit();
    reservation.activate();
    id
}

fn charged(session: &IntegrationSession) -> InjectionBytes {
    session.injection.state.lock().unwrap().pending_bytes
}

fn deliver(controller: &InjectionController) -> PendingInject {
    match controller.boundary_action(false, false) {
        BoundaryAction::Deliver(pending) => pending,
        _ => panic!("ready injection should be delivered"),
    }
}

#[test]
fn only_inline_payloads_are_charged_to_media() {
    let content = vec![
        wire::ContentBlock::Image(
            wire::ImageContent::new("AAAA", "image/png").uri("https://example.test/image"),
        ),
        wire::ContentBlock::Audio(wire::AudioContent::new("AAAA", "audio/wav")),
        wire::ContentBlock::Resource(wire::EmbeddedResource::new(
            wire::EmbeddedResourceResource::BlobResourceContents(
                wire::BlobResourceContents::new("AAAA", "file:///blob")
                    .mime_type("application/octet-stream"),
            ),
        )),
        wire::ContentBlock::Resource(wire::EmbeddedResource::new(
            wire::EmbeddedResourceResource::TextResourceContents(wire::TextResourceContents::new(
                "embedded text",
                "file:///text",
            )),
        )),
        wire::ContentBlock::Text(wire::TextContent::new("ordinary text")),
    ];
    let (_, bytes) = validate_inject_content(&content).unwrap();
    assert_eq!(bytes.media, 12);
    assert_eq!(
        bytes.content,
        serde_json::to_vec(&content).unwrap().len() - 12
    );

    // Malformed payload escapes still consume serialized media bytes, not raw length.
    let escaped = vec![wire::ContentBlock::Image(wire::ImageContent::new(
        "\"\\\n",
        "image/png",
    ))];
    let (_, bytes) = validate_inject_content(&escaped).unwrap();
    let serialized_payload = serde_json::to_vec("\"\\\n").unwrap().len() - 2;
    assert_eq!(bytes.media, serialized_payload);
    assert_eq!(
        bytes.content,
        serde_json::to_vec(&escaped).unwrap().len() - serialized_payload
    );

    let metadata_heavy = vec![wire::ContentBlock::Image(
        wire::ImageContent::new("AAAA", "image/png").uri("u".repeat(MAX_PENDING_INJECTION_BYTES)),
    )];
    assert!(
        validate_inject_content(&metadata_heavy).is_err(),
        "URI bytes must not borrow the media allowance"
    );
}

#[test]
fn content_limit_is_exact_and_independent_of_media() {
    let (integration, id, session) = fixture();
    let exact = text_at_size(MAX_PENDING_INJECTION_BYTES);
    let (_, bytes) = validate_inject_content(&exact).unwrap();
    assert_eq!(
        bytes,
        InjectionBytes {
            content: MAX_PENDING_INJECTION_BYTES,
            media: 0
        }
    );
    let reservation = reserve(&integration, &id, exact).unwrap();
    assert_eq!(charged(&session), bytes);
    assert!(reserve(&integration, &id, vec![image(4)]).is_err());
    drop(reservation);
    assert_eq!(charged(&session), InjectionBytes::default());
    assert!(validate_inject_content(&text_at_size(MAX_PENDING_INJECTION_BYTES + 1)).is_err());

    // Splitting a legal per-request size across requests must not evade the cap.
    let first = reserve(
        &integration,
        &id,
        text_at_size(MAX_PENDING_INJECTION_BYTES / 2),
    )
    .unwrap();
    let second = reserve(
        &integration,
        &id,
        text_at_size(MAX_PENDING_INJECTION_BYTES / 2),
    )
    .unwrap();
    assert_eq!(charged(&session).content, MAX_PENDING_INJECTION_BYTES);
    assert!(reserve(&integration, &id, text_at_size(128)).is_err());
    drop((first, second));
    assert_eq!(charged(&session), InjectionBytes::default());
}

#[test]
fn twenty_mib_source_image_remains_media_through_reservation() {
    let _large_media = LARGE_MEDIA_TEST.lock().unwrap();
    let (integration, id, session) = fixture();
    let raw_bytes: usize = 20 * 1024 * 1024;
    let encoded_bytes = raw_bytes.div_ceil(3) * 4;
    // Valid base64 for 20 MiB of zero bytes, without allocating the raw buffer.
    let mut payload = "A".repeat(encoded_bytes - 1);
    payload.push('=');
    let reservation = reserve(
        &integration,
        &id,
        vec![wire::ContentBlock::Image(wire::ImageContent::new(
            payload,
            "image/png",
        ))],
    )
    .unwrap();
    assert_eq!(charged(&session).media, encoded_bytes);
    assert!(charged(&session).content < MAX_PENDING_INJECTION_BYTES);
    let message_id = ready(reservation);
    let pending = deliver(&session.injection);
    assert_eq!(pending.message_id, message_id);
    let wire::ContentBlock::Image(original) = &pending.content[0] else {
        panic!("image lost")
    };
    let Part::Media(media) = &pending.items[0].parts[0] else {
        panic!("image converted to text")
    };
    assert_eq!(media.modality, Modality::Image);
    let DataRef::InlineText(data) = &media.data else {
        panic!("inline payload lost")
    };
    let original_payload: &str = original.data.as_ref();
    assert_eq!(
        data.strip_prefix("data:image/png;base64,").unwrap(),
        original_payload
    );
    session.injection.finish_delivery(&pending, true);
    assert_eq!(charged(&session), InjectionBytes::default());
}

#[test]
fn media_exact_limit_aggregate_inflight_and_failed_replacement() {
    let _large_media = LARGE_MEDIA_TEST.lock().unwrap();
    let (integration, id, session) = fixture();
    let half = MAX_PENDING_INJECTION_MEDIA_BYTES / 2;
    let first = ready(reserve(&integration, &id, vec![image(half)]).unwrap());
    let second = ready(reserve(&integration, &id, vec![image(half)]).unwrap());
    let before = charged(&session);
    assert_eq!(before.media, MAX_PENDING_INJECTION_MEDIA_BYTES);
    assert!(reserve(&integration, &id, vec![image(4)]).is_err());

    // An individually valid replacement fails atomically against the aggregate cap.
    let replacement = vec![image(half + 4)];
    let (items, bytes) = validate_inject_content(&replacement).unwrap();
    assert!(
        session
            .injection
            .replace_transition(&first, &replacement, &items, bytes)
            .is_err()
    );
    drop((replacement, items));
    assert_eq!(charged(&session), before);
    let pending = deliver(&session.injection);
    assert_eq!(pending.message_id, first);
    assert_eq!(pending.commitment, InjectCommitment::Ready);
    let wire::ContentBlock::Image(original) = &pending.content[0] else {
        panic!("image lost")
    };
    assert_eq!(original.data.len(), half);
    assert_eq!(pending.bytes.media, half);
    assert_eq!(
        pending.items,
        content_blocks_to_items(&pending.content).unwrap()
    );
    assert_eq!(charged(&session), before, "delivery still retains payloads");
    assert!(reserve(&integration, &id, vec![image(4)]).is_err());
    session.injection.finish_delivery(&pending, true);
    drop(pending);
    let extra = reserve(&integration, &id, vec![image(4)]).unwrap();
    extra.discard();
    let pending = deliver(&session.injection);
    assert_eq!(
        pending.message_id, second,
        "failed replacement preserves FIFO"
    );
    session.injection.finish_delivery(&pending, false);
    drop(pending);
    assert_eq!(charged(&session), InjectionBytes::default());

    // Exercise validation's per-request boundary independently of admission.
    let exact = vec![image(MAX_PENDING_INJECTION_MEDIA_BYTES)];
    let (items, bytes) = validate_inject_content(&exact).unwrap();
    assert_eq!(bytes.media, MAX_PENDING_INJECTION_MEDIA_BYTES);
    drop((exact, items));
    assert!(validate_inject_content(&[image(MAX_PENDING_INJECTION_MEDIA_BYTES + 1)]).is_err());
}

#[test]
fn failed_content_replacement_preserves_identity_content_and_fifo() {
    let (integration, id, session) = fixture();
    let first = ready(
        reserve(
            &integration,
            &id,
            text_at_size(MAX_PENDING_INJECTION_BYTES / 2),
        )
        .unwrap(),
    );
    let second = ready(
        reserve(
            &integration,
            &id,
            text_at_size(MAX_PENDING_INJECTION_BYTES / 2),
        )
        .unwrap(),
    );
    let before = charged(&session);
    let replacement = text_at_size(MAX_PENDING_INJECTION_BYTES / 2 + 1);
    let (items, bytes) = validate_inject_content(&replacement).unwrap();
    assert!(
        session
            .injection
            .replace_transition(&first, &replacement, &items, bytes)
            .is_err()
    );
    assert_eq!(charged(&session), before);
    for id in [first, second] {
        let pending = deliver(&session.injection);
        assert_eq!(pending.message_id, id);
        assert_eq!(
            pending.content,
            text_at_size(MAX_PENDING_INJECTION_BYTES / 2)
        );
        assert_eq!(
            pending.items,
            content_blocks_to_items(&pending.content).unwrap()
        );
        session.injection.finish_delivery(&pending, true);
    }
    assert_eq!(charged(&session), InjectionBytes::default());
}

#[test]
fn rollback_revoke_discard_and_close_release_both_budgets() {
    let (integration, id, session) = fixture();
    let content = || {
        vec![
            wire::ContentBlock::Text(wire::TextContent::new("retained text")),
            image(512 * 1024),
        ]
    };
    let reservation = reserve(&integration, &id, content()).unwrap();
    let bytes = charged(&session);
    assert!(bytes.content > 0 && bytes.media > 0);
    drop(reservation); // Uncommitted request rollback.
    assert_eq!(charged(&session), InjectionBytes::default());
    assert_eq!(session.injection.state.lock().unwrap().accepted_count, 0);
    let mut reservation = reserve(&integration, &id, content()).unwrap();
    reservation.commit();
    drop(reservation); // Response failure after commitment must also release retention.
    assert_eq!(charged(&session), InjectionBytes::default());
    let revoked = ready(reserve(&integration, &id, content()).unwrap());
    assert!(matches!(
        session.injection.revoke_transition(&revoked),
        PendingTransition::Applied
    ));
    assert_eq!(charged(&session), InjectionBytes::default());
    reserve(&integration, &id, content()).unwrap().discard();
    assert_eq!(charged(&session), InjectionBytes::default());

    ready(reserve(&integration, &id, content()).unwrap());
    let pending = deliver(&session.injection);
    ready(reserve(&integration, &id, content()).unwrap());
    let reserved = reserve(&integration, &id, content()).unwrap();
    session.injection.close_session();
    assert_eq!(
        charged(&session),
        bytes.saturating_add(bytes),
        "close releases Ready entries, not owned reservations or delivery"
    );
    drop(reserved);
    assert_eq!(charged(&session), bytes);
    session.injection.finish_delivery(&pending, false);
    assert_eq!(charged(&session), InjectionBytes::default());
}

#[test]
fn cancellation_retains_budget_and_content_until_next_turn() {
    let (integration, id, session) = fixture();
    let message_id = ready(reserve(&integration, &id, vec![image(512 * 1024)]).unwrap());
    let before = charged(&session);
    session.injection.cancel_turn();
    assert_eq!(charged(&session), before);
    assert!(matches!(
        session.injection.boundary_action(false, false),
        BoundaryAction::Complete(AcpInjectionBoundary::Stopped)
    ));
    session.injection.reset_cancellation();
    session.injection.start_turn();
    let pending = deliver(&session.injection);
    assert_eq!(pending.message_id, message_id);
    assert_eq!(pending.bytes, before);
    session.injection.finish_delivery(&pending, true);
    assert_eq!(charged(&session), InjectionBytes::default());
}

#[tokio::test]
async fn media_replacement_grows_shrinks_and_changes_budget_class() {
    let (integration, id, session) = fixture();
    let message_id = ready(reserve(&integration, &id, vec![image(512 * 1024)]).unwrap());
    for content in [
        vec![image(1024 * 1024)],
        vec![image(4)],
        text_at_size(128),
        vec![image(512 * 1024)],
    ] {
        let (_, expected) = validate_inject_content(&content).unwrap();
        integration
            .replace_inject(wire::ReplaceInjectSessionRequest::new(
                id.clone(),
                message_id.clone(),
                content.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(charged(&session), expected);
        let state = session.injection.state.lock().unwrap();
        assert_eq!(state.pending.len(), 1);
        assert_eq!(state.pending[0].content, content);
        assert_eq!(
            state.pending[0].items,
            content_blocks_to_items(&content).unwrap()
        );
    }
    let pending = deliver(&session.injection);
    assert_eq!(pending.message_id, message_id);
    session.injection.finish_delivery(&pending, true);
    assert_eq!(charged(&session), InjectionBytes::default());
}

#[test]
fn poison_recovery_preserves_both_budget_counters() {
    let (integration, id, session) = fixture();
    let message_id = ready(
        reserve(
            &integration,
            &id,
            vec![
                wire::ContentBlock::Text(wire::TextContent::new("mixed content")),
                image(512 * 1024),
            ],
        )
        .unwrap(),
    );
    // Poison an otherwise valid state; no production panic hooks are needed.
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = session.injection.state.lock().unwrap();
        panic!("test unwind while owning valid state");
    }));
    assert!(poisoned.is_err());
    assert!(session.injection.state.is_poisoned());
    let content = vec![image(4)];
    let (items, bytes) = validate_inject_content(&content).unwrap();
    assert!(matches!(
        session
            .injection
            .replace_transition(&message_id, &content, &items, bytes)
            .unwrap(),
        PendingTransition::Applied
    ));
    {
        let state = session
            .injection
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert_eq!(state.pending_bytes, bytes);
        assert_eq!(state.pending[0].content, content);
        assert_eq!(state.pending[0].items, items);
    }
    session.injection.discard(&message_id);
    let state = session
        .injection
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    assert_eq!(state.pending_bytes, InjectionBytes::default());
    assert!(state.pending.is_empty());
}
