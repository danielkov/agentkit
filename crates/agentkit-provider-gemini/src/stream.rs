//! Stateful Gemini SSE-event-to-`ModelTurnEvent` translator.
//!
//! Gemini's `:streamGenerateContent?alt=sse` endpoint sends a sequence of
//! `GenerateContentResponse` JSON chunks inside `data:` frames. Text and thought
//! chunks arrive incrementally; `functionCall` parts arrive complete; the final
//! chunk carries `finishReason` and cumulative `usageMetadata`.

use agentkit_core::{
    Delta, FinishReason, Item, ItemKind, MetadataMap, Part, PartId, PartKind, ReasoningPart,
    TextPart,
};
use agentkit_loop::{LoopError, ModelTurnEvent, ModelTurnResult, set_provider_finish_reasons};
use serde_json::Value;

use crate::response::{map_finish_reason, next_synthetic_turn_id, parse_part};
pub(crate) use crate::sse::{SseDecoder, SseEvent};
use crate::usage::UsageSnapshot;

enum ActivePart {
    Text { part_id: PartId, buffer: String },
    Reasoning { part_id: PartId, buffer: String },
}

pub(crate) struct EventTranslator {
    active: Option<ActivePart>,
    committed_parts: Vec<Part>,
    next_part_index: usize,
    synthetic_turn_id: u64,
    call_counter: usize,
    usage: Option<UsageSnapshot>,
    metadata: MetadataMap,
    response_id: Option<String>,
    model: Option<String>,
    provider_finish_reason: Option<String>,
    prompt_blocked: bool,
    saw_terminal_signal: bool,
    finished: bool,
}

impl EventTranslator {
    pub(crate) fn new() -> Self {
        Self {
            active: None,
            committed_parts: Vec::new(),
            next_part_index: 0,
            synthetic_turn_id: next_synthetic_turn_id(),
            call_counter: 0,
            usage: None,
            metadata: MetadataMap::new(),
            response_id: None,
            model: None,
            provider_finish_reason: None,
            prompt_blocked: false,
            saw_terminal_signal: false,
            finished: false,
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.finished
    }

    pub(crate) fn saw_terminal_signal(&self) -> bool {
        self.saw_terminal_signal
    }

    /// Consumes a single decoded SSE event and produces zero or more `ModelTurnEvent`s.
    pub(crate) fn handle(&mut self, event: &SseEvent) -> Result<Vec<ModelTurnEvent>, LoopError> {
        if self.finished {
            return Ok(Vec::new());
        }
        let trimmed = event.data.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        let raw: Value = serde_json::from_str(trimmed)
            .map_err(|e| LoopError::Provider(format!("invalid Gemini SSE JSON: {e}")))?;

        // Check for inline error payload inside the SSE stream.
        if raw.get("error").is_some() {
            if let Some(err) = crate::retry::context_length_error(trimmed, None) {
                return Err(err);
            }
            let classification = crate::retry::classify_value(&raw, None);
            return Err(crate::retry::provider_error(
                agentkit_loop::ProviderFailureReason::ResponseFailed,
                classification,
            ));
        }

        let mut events = Vec::new();

        if let Some(id) = raw.get("responseId").and_then(Value::as_str) {
            self.response_id = Some(id.to_string());
        }
        if let Some(model) = raw.get("modelVersion").and_then(Value::as_str) {
            self.model = Some(model.to_string());
            self.metadata
                .insert("gemini.model_version".into(), Value::String(model.into()));
        }
        if let Some(feedback) = raw.get("promptFeedback") {
            self.metadata
                .insert("gemini.prompt_feedback".into(), feedback.clone());
            if let Some(block_reason) = feedback.get("blockReason").and_then(Value::as_str) {
                self.prompt_blocked = true;
                self.saw_terminal_signal = true;
                self.provider_finish_reason = Some(block_reason.to_string());
                self.metadata.insert(
                    "gemini.block_reason".into(),
                    Value::String(block_reason.into()),
                );
            }
        }

        if let Some(snapshot) = UsageSnapshot::parse(raw.get("usageMetadata")) {
            match &mut self.usage {
                Some(existing) => existing.merge(snapshot),
                None => self.usage = Some(snapshot),
            }
        }

        if let Some(candidate) = raw
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
        {
            for (raw_key, meta_key) in [
                ("groundingMetadata", "gemini.grounding_metadata"),
                ("urlContextMetadata", "gemini.url_context_metadata"),
                ("safetyRatings", "gemini.safety_ratings"),
                ("citationMetadata", "gemini.citation_metadata"),
            ] {
                if let Some(val) = candidate.get(raw_key) {
                    self.metadata.insert(meta_key.into(), val.clone());
                }
            }

            if let Some(parts) = candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
            {
                for raw_part in parts {
                    self.handle_part(raw_part, &mut events)?;
                }
            }

            if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str)
                && !reason.is_empty()
                && reason != "FINISH_REASON_UNSPECIFIED"
            {
                self.provider_finish_reason = Some(reason.to_string());
                self.metadata
                    .insert("gemini.finish_reason".into(), Value::String(reason.into()));
                self.saw_terminal_signal = true;
            }
        }

        Ok(events)
    }

    /// Finalizes the stream at EOF once a terminal signal (`finishReason` or
    /// `promptFeedback.blockReason`) has been observed.
    pub(crate) fn finalize(&mut self) -> Vec<ModelTurnEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        self.flush_active(&mut events);

        let usage = self.usage.as_ref().map(UsageSnapshot::to_usage);
        if let Some(u) = usage.clone() {
            events.push(ModelTurnEvent::Usage(u));
        }

        let has_tool_calls = self
            .committed_parts
            .iter()
            .any(|p| matches!(p, Part::ToolCall(_)));
        let finish_reason = if self.prompt_blocked && self.committed_parts.is_empty() {
            FinishReason::Blocked
        } else {
            map_finish_reason(self.provider_finish_reason.as_deref(), has_tool_calls)
        };

        let mut turn_metadata = self.metadata.clone();
        if let Some(reason) = &self.provider_finish_reason {
            set_provider_finish_reasons(&mut turn_metadata, [reason.as_str()]);
        }

        let output_items = if self.committed_parts.is_empty() {
            Vec::new()
        } else {
            vec![Item {
                id: self.response_id.clone().map(Into::into),
                kind: ItemKind::Assistant,
                parts: std::mem::take(&mut self.committed_parts),
                metadata: self.metadata.clone(),
                usage: None,
                finish_reason: None,
                created_at: None,
            }]
        };

        events.push(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason,
            output_items,
            usage,
            metadata: turn_metadata,
            model: self.model.clone(),
            response_id: self.response_id.clone(),
        }));

        events
    }

    fn handle_part(
        &mut self,
        raw_part: &Value,
        events: &mut Vec<ModelTurnEvent>,
    ) -> Result<(), LoopError> {
        let thought = raw_part
            .get("thought")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let signature = raw_part
            .get("thoughtSignature")
            .and_then(Value::as_str)
            .map(str::to_string);

        // Incremental unsigned reasoning text (coalesced into active part).
        // Signed reasoning parts are kept separate to preserve exact part boundaries.
        if thought
            && signature.is_none()
            && let Some(chunk) = raw_part.get("text").and_then(Value::as_str)
        {
            if !matches!(self.active, Some(ActivePart::Reasoning { .. })) {
                self.flush_active(events);
                let part_id = self.alloc_part_id();
                events.push(ModelTurnEvent::Delta(Delta::BeginPart {
                    part_id: part_id.clone(),
                    kind: PartKind::Reasoning,
                }));
                self.active = Some(ActivePart::Reasoning {
                    part_id,
                    buffer: String::new(),
                });
            }
            if let Some(ActivePart::Reasoning { part_id, buffer }) = &mut self.active
                && !chunk.is_empty()
            {
                buffer.push_str(chunk);
                events.push(ModelTurnEvent::Delta(Delta::AppendText {
                    part_id: part_id.clone(),
                    chunk: chunk.to_string(),
                }));
            }
            return Ok(());
        }

        // Incremental unsigned visible text (coalesced into active part).
        // Signed text parts (including empty signed text) are committed separately.
        if !thought
            && signature.is_none()
            && raw_part.get("functionCall").is_none()
            && let Some(chunk) = raw_part.get("text").and_then(Value::as_str)
        {
            if !matches!(self.active, Some(ActivePart::Text { .. })) {
                self.flush_active(events);
                let part_id = self.alloc_part_id();
                events.push(ModelTurnEvent::Delta(Delta::BeginPart {
                    part_id: part_id.clone(),
                    kind: PartKind::Text,
                }));
                self.active = Some(ActivePart::Text {
                    part_id,
                    buffer: String::new(),
                });
            }
            if let Some(ActivePart::Text { part_id, buffer }) = &mut self.active
                && !chunk.is_empty()
            {
                buffer.push_str(chunk);
                events.push(ModelTurnEvent::Delta(Delta::AppendText {
                    part_id: part_id.clone(),
                    chunk: chunk.to_string(),
                }));
            }
            return Ok(());
        }

        // Signed text/reasoning or non-streaming part (functionCall, inlineData,
        // toolCall/toolResponse, executableCode, redacted thought, etc.)
        self.flush_active(events);
        if let Some(part) = parse_part(raw_part, self.synthetic_turn_id, &mut self.call_counter)
            .map_err(|e| LoopError::Provider(e.to_string()))?
        {
            match &part {
                Part::Text(t) => {
                    let part_id = self.alloc_part_id();
                    events.push(ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: part_id.clone(),
                        kind: PartKind::Text,
                    }));
                    if !t.text.is_empty() {
                        events.push(ModelTurnEvent::Delta(Delta::AppendText {
                            part_id,
                            chunk: t.text.clone(),
                        }));
                    }
                }
                Part::Reasoning(r) => {
                    let part_id = self.alloc_part_id();
                    events.push(ModelTurnEvent::Delta(Delta::BeginPart {
                        part_id: part_id.clone(),
                        kind: PartKind::Reasoning,
                    }));
                    if let Some(summary) = &r.summary
                        && !summary.is_empty()
                    {
                        events.push(ModelTurnEvent::Delta(Delta::AppendText {
                            part_id,
                            chunk: summary.clone(),
                        }));
                    }
                }
                Part::ToolCall(call) => {
                    events.push(ModelTurnEvent::ToolCall(call.clone()));
                }
                _ => {}
            }
            events.push(ModelTurnEvent::Delta(Delta::CommitPart {
                part: part.clone(),
            }));
            self.committed_parts.push(part);
        }
        Ok(())
    }

    fn flush_active(&mut self, events: &mut Vec<ModelTurnEvent>) {
        let Some(active) = self.active.take() else {
            return;
        };
        let part = match active {
            ActivePart::Text { buffer, .. } => Part::Text(TextPart::new(buffer)),
            ActivePart::Reasoning { buffer, .. } => Part::Reasoning(ReasoningPart::summary(buffer)),
        };
        events.push(ModelTurnEvent::Delta(Delta::CommitPart {
            part: part.clone(),
        }));
        self.committed_parts.push(part);
    }

    fn alloc_part_id(&mut self) -> PartId {
        let id = PartId::new(format!("part-{}", self.next_part_index));
        self.next_part_index += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn streams_thoughts_then_text_then_finish() {
        let mut tr = EventTranslator::new();
        let e1 = tr
            .handle(&SseEvent {
                data: json!({
                    "responseId": "r1",
                    "modelVersion": "gemini-2.5-pro",
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "thought": true, "text": "Step 1. " }] }
                    }]
                })
                .to_string(),
            })
            .unwrap();
        assert!(matches!(
            e1[0],
            ModelTurnEvent::Delta(Delta::BeginPart {
                kind: PartKind::Reasoning,
                ..
            })
        ));
        assert!(matches!(
            e1[1],
            ModelTurnEvent::Delta(Delta::AppendText { ref chunk, .. }) if chunk == "Step 1. "
        ));

        let e2 = tr
            .handle(&SseEvent {
                data: json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "text": "Hello " }] }
                    }]
                })
                .to_string(),
            })
            .unwrap();
        // Reasoning part committed, Text part begun & appended
        assert!(matches!(
            e2[0],
            ModelTurnEvent::Delta(Delta::CommitPart {
                part: Part::Reasoning(_)
            })
        ));
        assert!(matches!(
            e2[1],
            ModelTurnEvent::Delta(Delta::BeginPart {
                kind: PartKind::Text,
                ..
            })
        ));

        let e3 = tr
            .handle(&SseEvent {
                data: json!({
                    "candidates": [{
                        "content": { "role": "model", "parts": [{ "text": "world!" }] },
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 5,
                        "thoughtsTokenCount": 8
                    }
                })
                .to_string(),
            })
            .unwrap();
        assert!(matches!(
            e3[0],
            ModelTurnEvent::Delta(Delta::AppendText { ref chunk, .. }) if chunk == "world!"
        ));
        assert!(tr.saw_terminal_signal());

        let final_events = tr.finalize();
        let ModelTurnEvent::Finished(result) = final_events.last().unwrap() else {
            panic!("expected Finished");
        };
        assert_eq!(result.finish_reason, FinishReason::Completed);
        assert_eq!(result.output_items[0].parts.len(), 2);
        assert_eq!(
            result
                .usage
                .as_ref()
                .unwrap()
                .tokens
                .as_ref()
                .unwrap()
                .reasoning_tokens,
            Some(8)
        );
    }

    #[test]
    fn preserves_part_boundaries_for_streamed_thought_signatures() {
        let mut translator = EventTranslator::new();
        let mut events = Vec::new();
        translator
            .handle_part(&json!({ "text": "Answer" }), &mut events)
            .unwrap();
        translator
            .handle_part(
                &json!({ "text": "", "thoughtSignature": "sig-1" }),
                &mut events,
            )
            .unwrap();
        translator
            .handle_part(
                &json!({ "text": "", "thoughtSignature": "sig-2" }),
                &mut events,
            )
            .unwrap();
        translator.flush_active(&mut events);
        let [Part::Text(answer), Part::Text(s1), Part::Text(s2)] =
            translator.committed_parts.as_slice()
        else {
            panic!(
                "expected three separate text parts, got {:?}",
                translator.committed_parts
            );
        };
        assert_eq!(answer.text, "Answer");
        assert!(!answer.metadata.contains_key("gemini.thought_signature"));
        assert!(s1.text.is_empty());
        assert_eq!(s1.metadata["gemini.thought_signature"], "sig-1");
        assert!(s2.text.is_empty());
        assert_eq!(s2.metadata["gemini.thought_signature"], "sig-2");
    }
}
