use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use agentkit_core::{
    CustomPart, DataRef, Delta, FinishReason, Item, ItemKind, MediaPart, MetadataMap, Modality,
    Part, ReasoningPart, TextPart, ToolCallPart,
};
use agentkit_loop::{
    LoopError, ModelTurnEvent, ModelTurnResult, ProviderFailureReason, set_provider_finish_reasons,
};
use base64::Engine;
use serde_json::Value;
use thiserror::Error;

use crate::retry::{classify_value, context_length_error, local_error, provider_error};
use crate::usage::UsageSnapshot;

static NEXT_SYNTHETIC_TURN_ID: AtomicU64 = AtomicU64::new(1);
static PROCESS_NAMESPACE: OnceLock<u64> = OnceLock::new();

pub(crate) fn next_synthetic_turn_namespace() -> String {
    let epoch = *PROCESS_NAMESPACE.get_or_init(|| {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        nanos ^ (std::process::id() as u64).rotate_left(32)
    });
    let seq = NEXT_SYNTHETIC_TURN_ID.fetch_add(1, Ordering::Relaxed);
    format!("{epoch:x}_{seq}")
}

#[derive(Debug, Error)]
pub(crate) enum ResponseError {
    #[error("invalid Gemini response: {0}")]
    Protocol(String),
}

/// Parses a buffered `generateContent` JSON response into `ModelTurnEvent`s.
pub(crate) fn build_turn_from_response(body: &str) -> Result<VecDeque<ModelTurnEvent>, LoopError> {
    let raw: Value =
        serde_json::from_str(body).map_err(|_| local_error(ProviderFailureReason::Protocol))?;

    if raw.get("error").is_some() {
        if let Some(err) = context_length_error(body, None) {
            return Err(err);
        }
        return Err(provider_error(
            ProviderFailureReason::ResponseFailed,
            classify_value(&raw, None),
        ));
    }

    let usage = UsageSnapshot::parse(raw.get("usageMetadata")).map(|u| u.to_usage());
    let response_id = raw
        .get("responseId")
        .and_then(Value::as_str)
        .map(str::to_string);
    let model = raw
        .get("modelVersion")
        .and_then(Value::as_str)
        .map(str::to_string);

    let mut metadata = MetadataMap::new();
    if let Some(m) = &model {
        metadata.insert("gemini.model_version".into(), Value::String(m.clone()));
    }
    if let Some(feedback) = raw.get("promptFeedback") {
        metadata.insert("gemini.prompt_feedback".into(), feedback.clone());
    }

    let candidate = raw
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first());
    let prompt_block_reason = raw
        .pointer("/promptFeedback/blockReason")
        .and_then(Value::as_str);

    if candidate.is_none() && prompt_block_reason.is_none() {
        return Err(local_error(ProviderFailureReason::Protocol));
    }

    let mut parts: Vec<Part> = Vec::new();
    let synthetic_turn_ns = next_synthetic_turn_namespace();
    let mut call_counter: usize = 0;
    let mut provider_finish_reason: Option<String> = None;

    if let Some(candidate) = candidate {
        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
            provider_finish_reason = Some(reason.to_string());
            metadata.insert("gemini.finish_reason".into(), Value::String(reason.into()));
        }
        for (raw_key, meta_key) in [
            ("groundingMetadata", "gemini.grounding_metadata"),
            ("urlContextMetadata", "gemini.url_context_metadata"),
            ("safetyRatings", "gemini.safety_ratings"),
            ("citationMetadata", "gemini.citation_metadata"),
        ] {
            if let Some(val) = candidate.get(raw_key) {
                metadata.insert(meta_key.into(), val.clone());
            }
        }
        if let Some(raw_parts) = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
        {
            for raw_part in raw_parts {
                if let Some(part) = parse_part(raw_part, &synthetic_turn_ns, &mut call_counter)
                    .map_err(|_| local_error(ProviderFailureReason::Protocol))?
                {
                    parts.push(part);
                }
            }
        }
    } else if let Some(block_reason) = prompt_block_reason {
        provider_finish_reason = Some(block_reason.to_string());
        metadata.insert(
            "gemini.block_reason".into(),
            Value::String(block_reason.into()),
        );
    }

    let has_tool_calls = parts.iter().any(|p| matches!(p, Part::ToolCall(_)));
    let finish_reason = if candidate.is_none()
        && raw
            .pointer("/promptFeedback/blockReason")
            .and_then(Value::as_str)
            .is_some()
    {
        FinishReason::Blocked
    } else {
        map_finish_reason(provider_finish_reason.as_deref(), has_tool_calls)
    };

    let mut events = VecDeque::new();
    if let Some(usage) = usage.clone() {
        events.push_back(ModelTurnEvent::Usage(usage));
    }
    for part in &parts {
        if let Part::ToolCall(call) = part {
            events.push_back(ModelTurnEvent::ToolCall(call.clone()));
        }
    }

    let mut turn_metadata = metadata.clone();
    if let Some(reason) = &provider_finish_reason {
        set_provider_finish_reasons(&mut turn_metadata, [reason.as_str()]);
    }

    if parts.is_empty() {
        events.push_back(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason,
            output_items: Vec::new(),
            usage,
            metadata: turn_metadata,
            model,
            response_id,
        }));
        return Ok(events);
    }

    for part in &parts {
        events.push_back(ModelTurnEvent::Delta(Delta::CommitPart {
            part: part.clone(),
        }));
    }

    let item = Item {
        id: response_id.clone().map(Into::into),
        kind: ItemKind::Assistant,
        parts,
        metadata,
        usage: None,
        finish_reason: None,
        created_at: None,
    };

    events.push_back(ModelTurnEvent::Finished(ModelTurnResult {
        finish_reason,
        output_items: vec![item],
        usage,
        metadata: turn_metadata,
        model,
        response_id,
    }));

    Ok(events)
}

pub(crate) fn parse_part(
    part: &Value,
    synthetic_turn_ns: &str,
    call_counter: &mut usize,
) -> Result<Option<Part>, ResponseError> {
    let thought = part
        .get("thought")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let signature = part
        .get("thoughtSignature")
        .and_then(Value::as_str)
        .map(str::to_string);

    if let Some(fc) = part.get("functionCall") {
        let name = fc
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| ResponseError::Protocol("functionCall missing name".into()))?
            .to_string();
        let (id, has_provider_id) = match fc.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => (id.to_string(), true),
            _ => {
                let synth = format!("gemini_call_{synthetic_turn_ns}_{}", *call_counter);
                *call_counter += 1;
                (synth, false)
            }
        };
        let args = fc
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let mut call = ToolCallPart::new(id, name, args);
        if has_provider_id {
            call.metadata
                .insert("gemini.has_call_id".into(), Value::Bool(true));
        }
        if let Some(sig) = signature {
            call.metadata
                .insert("gemini.thought_signature".into(), Value::String(sig));
        }
        return Ok(Some(Part::ToolCall(call)));
    }

    if thought {
        let text = part.get("text").and_then(Value::as_str).map(str::to_string);
        let mut meta = MetadataMap::new();
        if let Some(sig) = signature {
            meta.insert("gemini.thought_signature".into(), Value::String(sig));
        }
        let reasoning = match text {
            Some(summary) => ReasoningPart::summary(summary).with_metadata(meta),
            None => ReasoningPart {
                summary: None,
                data: None,
                redacted: true,
                metadata: meta,
            },
        };
        return Ok(Some(Part::Reasoning(reasoning)));
    }

    if let Some(text) = part.get("text").and_then(Value::as_str) {
        let mut text_part = TextPart::new(text);
        if let Some(sig) = signature {
            text_part
                .metadata
                .insert("gemini.thought_signature".into(), Value::String(sig));
        }
        return Ok(Some(Part::Text(text_part)));
    }

    if let Some(inline) = part.get("inlineData") {
        let mime = inline
            .get("mimeType")
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream");
        let b64 = inline
            .get("data")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| ResponseError::Protocol(format!("invalid inlineData base64: {e}")))?;
        let modality = modality_from_mime(mime);
        let mut media = MediaPart::new(modality, mime, DataRef::InlineBytes(bytes));
        if let Some(sig) = signature {
            media
                .metadata
                .insert("gemini.thought_signature".into(), Value::String(sig));
        }
        return Ok(Some(Part::Media(media)));
    }

    // Server-side tool invocations, executable code, and code execution results
    for key in [
        "toolCall",
        "toolResponse",
        "executableCode",
        "codeExecutionResult",
    ] {
        if part.get(key).is_some() {
            return Ok(Some(Part::Custom(
                CustomPart::new(format!("gemini.{key}")).with_value(part.clone()),
            )));
        }
    }

    if signature.is_some() {
        let mut meta = MetadataMap::new();
        if let Some(sig) = signature {
            meta.insert("gemini.thought_signature".into(), Value::String(sig));
        }
        return Ok(Some(Part::Reasoning(ReasoningPart {
            summary: None,
            data: None,
            redacted: true,
            metadata: meta,
        })));
    }

    Ok(None)
}

pub(crate) fn modality_from_mime(mime: &str) -> Modality {
    if mime.starts_with("image/") {
        Modality::Image
    } else if mime.starts_with("audio/") {
        Modality::Audio
    } else if mime.starts_with("video/") {
        Modality::Video
    } else {
        Modality::Binary
    }
}

pub(crate) fn map_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
    match reason {
        Some("STOP") => {
            if has_tool_calls {
                FinishReason::ToolCall
            } else {
                FinishReason::Completed
            }
        }
        Some("MAX_TOKENS") => FinishReason::MaxTokens,
        Some(
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION",
        ) => FinishReason::Blocked,
        Some(other) => FinishReason::Other(other.into()),
        None => {
            if has_tool_calls {
                FinishReason::ToolCall
            } else {
                FinishReason::Completed
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_text_thinking_and_usage() {
        let body = json!({
            "responseId": "resp_1",
            "modelVersion": "gemini-2.5-flash",
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        { "thought": true, "text": "Let me think.", "thoughtSignature": "sig-t" },
                        { "text": "The answer is 42." }
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "cachedContentTokenCount": 60,
                "candidatesTokenCount": 12,
                "thoughtsTokenCount": 25
            }
        })
        .to_string();

        let events: Vec<_> = build_turn_from_response(&body)
            .unwrap()
            .into_iter()
            .collect();
        let ModelTurnEvent::Usage(usage) = &events[0] else {
            panic!("expected usage");
        };
        let tokens = usage.tokens.as_ref().unwrap();
        assert_eq!(tokens.input_tokens, 100);
        assert_eq!(tokens.cached_input_tokens, Some(60));
        assert_eq!(tokens.output_tokens, 37);
        assert_eq!(tokens.reasoning_tokens, Some(25));

        let ModelTurnEvent::Finished(result) = events.last().unwrap() else {
            panic!("expected Finished");
        };
        assert_eq!(result.finish_reason, FinishReason::Completed);
        assert_eq!(result.model.as_deref(), Some("gemini-2.5-flash"));
        assert_eq!(result.response_id.as_deref(), Some("resp_1"));
        assert_eq!(
            result.metadata["agentkit.provider_finish_reasons"],
            json!(["STOP"])
        );
        let parts = &result.output_items[0].parts;
        assert!(
            matches!(&parts[0], Part::Reasoning(r) if r.metadata["gemini.thought_signature"] == "sig-t")
        );
        assert!(matches!(&parts[1], Part::Text(t) if t.text == "The answer is 42."));
    }

    #[test]
    fn maps_stop_with_function_call_to_tool_call_finish_reason() {
        let body = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": { "name": "lookup", "args": { "id": 7 } },
                        "thoughtSignature": "fc-sig"
                    }]
                },
                "finishReason": "STOP"
            }]
        })
        .to_string();

        let events: Vec<_> = build_turn_from_response(&body)
            .unwrap()
            .into_iter()
            .collect();
        assert!(events.iter().any(|e| matches!(e, ModelTurnEvent::ToolCall(c) if c.name == "lookup" && c.metadata["gemini.thought_signature"] == "fc-sig")));
        let ModelTurnEvent::Finished(result) = events.last().unwrap() else {
            panic!("expected Finished");
        };
        assert_eq!(result.finish_reason, FinishReason::ToolCall);
    }

    #[test]
    fn maps_prompt_feedback_block_to_blocked() {
        let body = json!({
            "promptFeedback": { "blockReason": "SAFETY" }
        })
        .to_string();
        let events: Vec<_> = build_turn_from_response(&body)
            .unwrap()
            .into_iter()
            .collect();
        let ModelTurnEvent::Finished(result) = events.last().unwrap() else {
            panic!("expected Finished");
        };
        assert_eq!(result.finish_reason, FinishReason::Blocked);
        assert!(result.output_items.is_empty());
    }

    #[test]
    fn synthetic_call_ids_are_unique_across_model_turns() {
        let body = json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {
                    "parts": [{ "functionCall": { "name": "lookup", "args": {} } }]
                }
            }]
        })
        .to_string();
        let next_id = || {
            build_turn_from_response(&body)
                .unwrap()
                .into_iter()
                .find_map(|ev| match ev {
                    ModelTurnEvent::ToolCall(call) => Some(call.id),
                    _ => None,
                })
                .unwrap()
        };
        assert_ne!(next_id(), next_id());
    }

    #[test]
    fn preserves_thought_signature_on_generated_media_and_server_tool_parts() {
        let body = json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        { "toolCall": { "id": "st_1", "toolType": "GOOGLE_SEARCH", "args": { "query": "rust" } } },
                        { "toolResponse": { "id": "st_1", "toolType": "GOOGLE_SEARCH", "response": {} } },
                        { "inlineData": { "mimeType": "image/png", "data": "AQID" }, "thoughtSignature": "img-sig" }
                    ]
                }
            }]
        })
        .to_string();
        let events: Vec<_> = build_turn_from_response(&body)
            .unwrap()
            .into_iter()
            .collect();
        let ModelTurnEvent::Finished(result) = events.last().unwrap() else {
            panic!("expected Finished");
        };
        let parts = &result.output_items[0].parts;
        assert!(matches!(&parts[0], Part::Custom(c) if c.kind == "gemini.toolCall"));
        assert!(matches!(&parts[1], Part::Custom(c) if c.kind == "gemini.toolResponse"));
        assert!(
            matches!(&parts[2], Part::Media(m) if m.metadata["gemini.thought_signature"] == "img-sig")
        );
    }

    #[test]
    fn rejects_empty_buffered_response_and_classifies_http_200_error_envelopes() {
        let empty_err = build_turn_from_response("{}").unwrap_err();
        assert_eq!(
            empty_err.provider_failure().unwrap().reason,
            ProviderFailureReason::Protocol
        );

        let err_envelope = json!({
            "error": {
                "code": 503,
                "status": "UNAVAILABLE",
                "message": "backend unavailable"
            }
        })
        .to_string();
        let failure = build_turn_from_response(&err_envelope)
            .unwrap_err()
            .provider_failure()
            .copied()
            .unwrap();
        assert_eq!(failure.reason, ProviderFailureReason::ResponseFailed);
        assert_eq!(
            failure.upstream.error_type,
            agentkit_loop::UpstreamErrorKind::ServiceUnavailableError
        );
    }
}
