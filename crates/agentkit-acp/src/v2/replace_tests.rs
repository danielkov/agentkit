use super::*;
use std::time::Duration;

fn text_content(text: &str) -> Vec<wire::ContentBlock> {
    vec![wire::ContentBlock::Text(wire::TextContent::new(text))]
}

fn fixture() -> (AcpIntegration, AcpSessionHandle) {
    let integration = AcpIntegration::default();
    let (sink, _receiver) = ClientHandle::channel();
    let session = integration
        .bind_session(AcpSessionBinding::new(
            wire::SessionId::new("replace-session"),
            AgentkitSessionId::new("replace-loop"),
            sink,
        ))
        .unwrap();
    session.prepare_injection_turn();
    session.start_injection_turn();
    (integration, session)
}

fn reserve(integration: &AcpIntegration, session: &AcpSessionHandle, text: &str) -> ReservedInject {
    integration
        .reserve_inject(wire::InjectSessionRequest::new(
            session.acp_session_id().clone(),
            wire::SessionInjectMode::Steer,
            text_content(text),
        ))
        .unwrap()
}

fn accept(integration: &AcpIntegration, session: &AcpSessionHandle, text: &str) -> wire::MessageId {
    let mut reserved = reserve(integration, session, text);
    let id = reserved.message_id().clone();
    reserved.commit();
    reserved.activate();
    id
}

fn request(
    session: &AcpSessionHandle,
    id: &wire::MessageId,
    text: &str,
) -> wire::ReplaceInjectSessionRequest {
    wire::ReplaceInjectSessionRequest::new(
        session.acp_session_id().clone(),
        id.clone(),
        text_content(text),
    )
}

fn take_delivery(injection: &InjectionController) -> PendingInject {
    match injection.boundary_action(false, false) {
        BoundaryAction::Deliver(pending) => pending,
        _ => panic!("expected a ready injection at the boundary"),
    }
}

fn assert_text(pending: &PendingInject, expected: &str) {
    assert_eq!(pending.content, text_content(expected));
    assert_eq!(pending.items.len(), 1);
    assert_eq!(pending.items[0].kind, ItemKind::User);
    assert_eq!(pending.items[0].parts.len(), 1);
    assert!(matches!(&pending.items[0].parts[0], Part::Text(text) if text.text == expected));
    assert_eq!(
        pending.bytes,
        serde_json::to_vec(&pending.content).unwrap().len()
    );
    assert_eq!(pending.commitment, InjectCommitment::Ready);
}

fn assert_inject_error(error: agent_client_protocol::Error, reason: &str, id: &wire::MessageId) {
    assert_eq!(
        i32::from(error.code),
        if reason == "already_delivered" {
            -32010
        } else {
            -32002
        }
    );
    assert_eq!(
        error.data,
        Some(json!({ "reason": reason, "messageId": id }))
    );
}

#[tokio::test]
async fn replacement_preserves_fifo_identity_and_acceptance_count() {
    let (integration, session) = fixture();
    let first = accept(&integration, &session, "first");
    let second = accept(&integration, &session, "old second");
    let third = accept(&integration, &session, "third");
    let response = integration
        .replace_inject(request(&session, &second, "new second"))
        .await
        .unwrap();
    assert_eq!(response.message_id, second);
    integration
        .replace_inject(request(&session, &second, "final second"))
        .await
        .unwrap();

    let injection = &session.session.injection;
    {
        let state = injection.state.lock().unwrap();
        assert_eq!(state.accepted_count, 3);
        assert_eq!(state.pending.len(), 3);
        assert_eq!(
            state.pending_bytes,
            state.pending.iter().map(|p| p.bytes).sum::<usize>()
        );
    }
    for (id, text) in [(first, "first"), (second, "final second"), (third, "third")] {
        let pending = take_delivery(injection);
        assert_eq!(pending.message_id, id);
        assert_text(&pending, text);
        injection.finish_delivery(&pending, true);
    }
    let state = injection.state.lock().unwrap();
    assert!(state.pending.is_empty());
    assert_eq!(state.pending_bytes, 0);
    assert_eq!(state.accepted_count, 3);
}

#[tokio::test]
async fn replacement_grows_shrinks_and_rejects_over_budget_atomically() {
    let (integration, session) = fixture();
    let first = accept(&integration, &session, "first");
    let second = accept(&integration, &session, "second");
    let injection = &session.session.injection;
    let in_flight = take_delivery(injection);
    assert_eq!(in_flight.message_id, first);
    let overhead = serde_json::to_vec(&text_content("")).unwrap().len();
    let full = "x".repeat(MAX_PENDING_INJECTION_BYTES - in_flight.bytes - overhead);
    integration
        .replace_inject(request(&session, &second, &full))
        .await
        .unwrap();
    assert_eq!(
        injection.state.lock().unwrap().pending_bytes,
        MAX_PENDING_INJECTION_BYTES
    );

    let error = integration
        .replace_inject(request(&session, &second, &(full.clone() + "x")))
        .await
        .unwrap_err();
    assert_eq!(i32::from(error.code), -32602);
    {
        let state = injection.state.lock().unwrap();
        assert_eq!(state.pending_bytes, MAX_PENDING_INJECTION_BYTES);
        assert_eq!(state.accepted_count, 2);
        assert_eq!(state.delivering.as_ref(), Some(&first));
        assert_eq!(state.pending.len(), 1);
        assert_eq!(state.pending[0].message_id, second);
        assert_text(&state.pending[0], &full);
    }

    integration
        .replace_inject(request(&session, &second, "small"))
        .await
        .unwrap();
    let small_bytes = serde_json::to_vec(&text_content("small")).unwrap().len();
    assert_eq!(
        injection.state.lock().unwrap().pending_bytes,
        in_flight.bytes + small_bytes
    );
    injection.finish_delivery(&in_flight, true);
    assert_eq!(injection.state.lock().unwrap().pending_bytes, small_bytes);
    let pending = take_delivery(injection);
    assert_text(&pending, "small");
    injection.finish_delivery(&pending, true);
    assert_eq!(injection.state.lock().unwrap().pending_bytes, 0);
}

#[tokio::test]
async fn replacement_waits_until_response_is_committed_and_activated() {
    let (integration, session) = fixture();
    let mut reserved = reserve(&integration, &session, "original");
    let id = reserved.message_id().clone();
    let replacement = integration.replace_inject(request(&session, &id, "replacement"));
    tokio::pin!(replacement);
    assert!(futures_util::poll!(replacement.as_mut()).is_pending());
    assert!(matches!(
        session.session.injection.boundary_action(false, false),
        BoundaryAction::Wait
    ));
    assert_eq!(
        session.session.injection.state.lock().unwrap().pending[0].content,
        text_content("original")
    );
    reserved.commit();
    assert!(futures_util::poll!(replacement.as_mut()).is_pending());
    assert_eq!(
        session.session.injection.state.lock().unwrap().pending[0].commitment,
        InjectCommitment::Committed
    );
    reserved.activate();
    tokio::time::timeout(Duration::from_secs(1), replacement)
        .await
        .unwrap()
        .unwrap();
    let pending = take_delivery(&session.session.injection);
    assert_eq!(pending.message_id, id);
    assert_text(&pending, "replacement");
    session.session.injection.finish_delivery(&pending, true);
}

#[tokio::test]
async fn replacement_of_abandoned_reservation_wakes_with_unknown_id() {
    let (integration, session) = fixture();
    let reserved = reserve(&integration, &session, "original");
    let id = reserved.message_id().clone();
    let replacement = integration.replace_inject(request(&session, &id, "replacement"));
    tokio::pin!(replacement);
    assert!(futures_util::poll!(replacement.as_mut()).is_pending());
    drop(reserved);
    let error = tokio::time::timeout(Duration::from_secs(1), replacement)
        .await
        .unwrap()
        .unwrap_err();
    assert_inject_error(error, "unknown_message_id", &id);
    let state = session.session.injection.state.lock().unwrap();
    assert!(state.pending.is_empty());
    assert_eq!(state.pending_bytes, 0);
    assert_eq!(state.accepted_count, 0);
}

#[tokio::test]
async fn replacement_reports_unknown_revoked_and_delivered_ids() {
    let (integration, session) = fixture();
    let unknown = wire::MessageId::new("never-accepted");
    let error = integration
        .replace_inject(request(&session, &unknown, "new"))
        .await
        .unwrap_err();
    assert_inject_error(error, "unknown_message_id", &unknown);

    let revoked = accept(&integration, &session, "revoked");
    integration
        .revoke_inject(wire::RevokeInjectSessionRequest::new(
            session.acp_session_id().clone(),
            revoked.clone(),
        ))
        .await
        .unwrap();
    let error = integration
        .replace_inject(request(&session, &revoked, "new"))
        .await
        .unwrap_err();
    assert_inject_error(error, "unknown_message_id", &revoked);

    let delivered = accept(&integration, &session, "delivered");
    let pending = take_delivery(&session.session.injection);
    session.session.injection.finish_delivery(&pending, true);
    let error = integration
        .replace_inject(request(&session, &delivered, "new"))
        .await
        .unwrap_err();
    assert_inject_error(error, "already_delivered", &delivered);
}

#[tokio::test]
async fn delivery_winner_blocks_replacement_until_delivery_outcome() {
    for delivered in [true, false] {
        let (integration, session) = fixture();
        let id = accept(&integration, &session, "original");
        let pending = take_delivery(&session.session.injection);
        let replacement = integration.replace_inject(request(&session, &id, "replacement"));
        tokio::pin!(replacement);
        assert!(futures_util::poll!(replacement.as_mut()).is_pending());
        assert_text(&pending, "original");
        session
            .session
            .injection
            .finish_delivery(&pending, delivered);
        let error = tokio::time::timeout(Duration::from_secs(1), replacement)
            .await
            .unwrap()
            .unwrap_err();
        assert_inject_error(
            error,
            if delivered {
                "already_delivered"
            } else {
                "unknown_message_id"
            },
            &id,
        );
        assert_eq!(
            session
                .session
                .injection
                .state
                .lock()
                .unwrap()
                .pending_bytes,
            0
        );
    }
}

#[test]
fn simultaneous_delivery_and_replacement_have_one_linearized_outcome() {
    for _ in 0..32 {
        let (integration, session) = fixture();
        let id = accept(&integration, &session, "original");
        let injection = &session.session.injection;
        let content = text_content("replacement");
        let items = content_blocks_to_items(&content).unwrap();
        let bytes = serde_json::to_vec(&content).unwrap().len();
        let barrier = std::sync::Barrier::new(2);
        let (transition, pending) = std::thread::scope(|scope| {
            let replacing = scope.spawn(|| {
                barrier.wait();
                injection
                    .replace_transition(&id, &content, &items, bytes)
                    .unwrap()
            });
            barrier.wait();
            let pending = take_delivery(injection);
            (replacing.join().unwrap(), pending)
        });
        assert_eq!(pending.message_id, id);
        match transition {
            PendingTransition::Applied => assert_text(&pending, "replacement"),
            PendingTransition::WaitForDelivery => assert_text(&pending, "original"),
            _ => panic!("replacement either wins or waits for the in-flight delivery"),
        }
        injection.finish_delivery(&pending, true);
        let state = injection.state.lock().unwrap();
        assert!(state.pending.is_empty());
        assert_eq!(state.pending_bytes, 0);
        assert_eq!(state.accepted_count, 1);
        assert_eq!(state.delivered.len(), 1);
    }
}

#[tokio::test]
async fn replacement_survives_cancellation_and_delivers_in_next_turn() {
    let (integration, session) = fixture();
    let id = accept(&integration, &session, "original");
    session.interrupt();
    integration
        .replace_inject(request(&session, &id, "after cancellation"))
        .await
        .unwrap();
    assert!(matches!(
        session.session.injection.boundary_action(false, false),
        BoundaryAction::Complete(AcpInjectionBoundary::Stopped)
    ));
    session.prepare_injection_turn();
    session.start_injection_turn();
    let pending = take_delivery(&session.session.injection);
    assert_eq!(pending.message_id, id);
    assert_text(&pending, "after cancellation");
    session.session.injection.finish_delivery(&pending, true);
}

#[tokio::test]
async fn close_wakes_waiting_replacements_and_discards_pending_bytes() {
    for committed in [false, true] {
        let (integration, session) = fixture();
        let mut reserved = reserve(&integration, &session, "original");
        let id = reserved.message_id().clone();
        if committed {
            reserved.commit();
        }
        let replacement = integration.replace_inject(request(&session, &id, "replacement"));
        tokio::pin!(replacement);
        assert!(futures_util::poll!(replacement.as_mut()).is_pending());
        session.close();
        let error = tokio::time::timeout(Duration::from_secs(1), replacement)
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(i32::from(error.code), -32002);
        assert_eq!(
            error.data,
            Some(json!({ "sessionId": session.acp_session_id() }))
        );
        if committed {
            reserved.activate();
        } else {
            drop(reserved);
        }
        let state = session.session.injection.state.lock().unwrap();
        assert!(state.pending.is_empty());
        assert_eq!(state.pending_bytes, 0);
    }

    let (integration, session) = fixture();
    let id = accept(&integration, &session, "original");
    integration
        .replace_inject(request(&session, &id, "replacement"))
        .await
        .unwrap();
    session.close();
    let error = integration
        .replace_inject(request(&session, &id, "too late"))
        .await
        .unwrap_err();
    assert_eq!(i32::from(error.code), -32002);
    assert_eq!(
        session
            .session
            .injection
            .state
            .lock()
            .unwrap()
            .pending_bytes,
        0
    );
}

#[tokio::test]
async fn replacement_reuses_inject_content_validation_without_mutating_queue() {
    let (integration, session) = fixture();
    let id = accept(&integration, &session, "original");
    let invalid: wire::ContentBlock = serde_json::from_value(json!({
        "type": "future-unsupported-block", "value": "unsupported"
    }))
    .unwrap();
    let inject_error = match integration.reserve_inject(wire::InjectSessionRequest::new(
        session.acp_session_id().clone(),
        wire::SessionInjectMode::Steer,
        vec![invalid.clone()],
    )) {
        Ok(_) => panic!("unsupported inject content must fail"),
        Err(error) => error,
    };
    let replace_error = integration
        .replace_inject(wire::ReplaceInjectSessionRequest::new(
            session.acp_session_id().clone(),
            id.clone(),
            vec![invalid],
        ))
        .await
        .unwrap_err();
    assert_eq!(i32::from(replace_error.code), i32::from(inject_error.code));
    assert_eq!(replace_error.message, inject_error.message);
    assert_eq!(replace_error.data, inject_error.data);
    let pending = take_delivery(&session.session.injection);
    assert_eq!(pending.message_id, id);
    assert_text(&pending, "original");
    session.session.injection.finish_delivery(&pending, true);
}
