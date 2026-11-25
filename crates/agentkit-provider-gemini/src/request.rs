use std::collections::HashMap;

use agentkit_core::{Item, ItemKind, Part, PartKind, ToolOutput};
use agentkit_loop::{LoopError, PromptCacheMode, PromptCacheStrategy, TurnRequest};
use agentkit_tools_core::ToolSpec;
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::config::{GeminiConfig, OutputFormat};
use crate::media::{file_to_part, media_to_part};

/// Internal error produced while building a Gemini request body.
#[derive(Debug, Error)]
pub(crate) enum BuildError {
    #[error("unsupported content part {part_kind:?} on role {role:?}")]
    UnsupportedPart { role: ItemKind, part_kind: PartKind },
    #[error("unsupported data reference: {0}")]
    UnsupportedDataRef(String),
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("tool result for call_id {0:?} has no matching function name in transcript")]
    MissingToolName(String),
    #[error("{0}")]
    CacheViolation(String),
}

impl From<BuildError> for LoopError {
    fn from(error: BuildError) -> Self {
        LoopError::Provider(error.to_string())
    }
}

/// Builds the Gemini `generateContent` / `streamGenerateContent` JSON request body.
pub(crate) fn build_request_body(
    config: &GeminiConfig,
    request: &TurnRequest,
) -> Result<Value, BuildError> {
    validate_prompt_cache(request)?;

    // Track call_id -> (tool_name, had_provider_id) from assistant tool calls so
    // subsequent Tool items can populate Gemini's required `functionResponse.name`.
    let mut call_lookup: HashMap<String, (String, bool)> = HashMap::new();
    let mut system_parts: Vec<Value> = Vec::new();
    let mut contents: Vec<ContentMessage> = Vec::new();

    for item in &request.transcript {
        match item.kind {
            ItemKind::System | ItemKind::Developer | ItemKind::Context => {
                extend_system_parts(&mut system_parts, item)?;
            }
            ItemKind::User => {
                let parts = build_user_parts(&item.parts)?;
                append_content(&mut contents, "user", parts);
            }
            ItemKind::Assistant => {
                for part in &item.parts {
                    if let Part::ToolCall(call) = part {
                        let has_id = call
                            .metadata
                            .get("gemini.has_call_id")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        call_lookup.insert(call.id.0.clone(), (call.name.clone(), has_id));
                    }
                }
                let parts = build_model_parts(&item.parts)?;
                if !parts.is_empty() {
                    append_content(&mut contents, "model", parts);
                }
            }
            ItemKind::Tool => {
                let parts = build_tool_response_parts(&item.parts, &call_lookup)?;
                append_content(&mut contents, "user", parts);
            }
            ItemKind::Notification => {
                let part = build_notification_part(&item.parts)?;
                append_content(&mut contents, "user", vec![part]);
            }
        }
    }

    let mut body = Map::new();

    if !system_parts.is_empty() {
        body.insert("systemInstruction".into(), json!({ "parts": system_parts }));
    }

    body.insert(
        "contents".into(),
        Value::Array(
            contents
                .into_iter()
                .map(ContentMessage::into_value)
                .collect(),
        ),
    );

    // Tools: function declarations + built-in Gemini server tools
    let mut tools: Vec<Value> = Vec::new();
    if !request.available_tools.is_empty() {
        let declarations: Vec<Value> = request
            .available_tools
            .iter()
            .map(build_function_declaration)
            .collect();
        tools.push(json!({ "functionDeclarations": declarations }));
    }
    for server_tool in &config.server_tools {
        tools.push(server_tool.to_json());
    }
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }

    if let Some(choice) = &config.tool_choice {
        body.insert("toolConfig".into(), choice.to_json());
    }

    if !config.safety_settings.is_empty() {
        body.insert(
            "safetySettings".into(),
            Value::Array(config.safety_settings.iter().map(|s| s.to_json()).collect()),
        );
    }

    if let Some(cached) = &config.cached_content {
        body.insert("cachedContent".into(), Value::String(cached.clone()));
    }

    let generation_config = build_generation_config(config);
    if !generation_config.is_empty() {
        body.insert("generationConfig".into(), Value::Object(generation_config));
    }

    Ok(Value::Object(body))
}

fn validate_prompt_cache(request: &TurnRequest) -> Result<(), BuildError> {
    let Some(cache) = request.cache.as_ref() else {
        return Ok(());
    };
    if cache.mode == PromptCacheMode::Required
        && matches!(cache.strategy, PromptCacheStrategy::Explicit { .. })
    {
        return Err(BuildError::CacheViolation(
            "Gemini does not support inline cache breakpoints; use GeminiConfig::with_cached_content for explicit caching or PromptCacheStrategy::Automatic for implicit caching".into(),
        ));
    }
    Ok(())
}

fn build_generation_config(config: &GeminiConfig) -> Map<String, Value> {
    let mut gc = Map::new();
    if let Some(max_tokens) = config.max_output_tokens {
        gc.insert("maxOutputTokens".into(), Value::from(max_tokens));
    }
    if let Some(temp) = config.temperature {
        gc.insert("temperature".into(), json_number(temp as f64));
    }
    if let Some(top_p) = config.top_p {
        gc.insert("topP".into(), json_number(top_p as f64));
    }
    if let Some(top_k) = config.top_k {
        gc.insert("topK".into(), Value::from(top_k));
    }
    if let Some(stops) = &config.stop_sequences {
        gc.insert(
            "stopSequences".into(),
            Value::Array(stops.iter().cloned().map(Value::String).collect()),
        );
    }
    if let Some(seed) = config.seed {
        gc.insert("seed".into(), Value::from(seed));
    }
    if let Some(penalty) = config.presence_penalty {
        gc.insert("presencePenalty".into(), json_number(penalty as f64));
    }
    if let Some(penalty) = config.frequency_penalty {
        gc.insert("frequencyPenalty".into(), json_number(penalty as f64));
    }
    if let Some(thinking) = &config.thinking {
        gc.insert("thinkingConfig".into(), thinking.to_json());
    }
    if let Some(format) = &config.output_format {
        match format {
            OutputFormat::Json => {
                gc.insert(
                    "responseMimeType".into(),
                    Value::String("application/json".into()),
                );
            }
            OutputFormat::JsonSchema { schema } => {
                gc.insert(
                    "responseMimeType".into(),
                    Value::String("application/json".into()),
                );
                gc.insert("responseJsonSchema".into(), schema.clone());
            }
        }
    }
    gc
}

fn json_number(n: f64) -> Value {
    serde_json::Number::from_f64(n).map_or(Value::Null, Value::Number)
}

struct ContentMessage {
    role: &'static str,
    parts: Vec<Value>,
}

impl ContentMessage {
    fn into_value(self) -> Value {
        json!({
            "role": self.role,
            "parts": self.parts,
        })
    }
}

fn append_content(contents: &mut Vec<ContentMessage>, role: &'static str, parts: Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    if let Some(last) = contents.last_mut()
        && last.role == role
    {
        last.parts.extend(parts);
        return;
    }
    contents.push(ContentMessage { role, parts });
}

fn extend_system_parts(parts: &mut Vec<Value>, item: &Item) -> Result<(), BuildError> {
    for part in &item.parts {
        match part {
            Part::Text(text) => parts.push(json!({ "text": text.text })),
            Part::Structured(structured) => parts.push(json!({
                "text": serde_json::to_string_pretty(&structured.value)?,
            })),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    parts.push(json!({ "text": summary }));
                }
            }
            _ => {
                return Err(BuildError::UnsupportedPart {
                    role: item.kind,
                    part_kind: part_kind(part),
                });
            }
        }
    }
    Ok(())
}

fn build_notification_part(parts: &[Part]) -> Result<Value, BuildError> {
    let mut buf = String::new();
    for part in parts {
        match part {
            Part::Text(text) => buf.push_str(&text.text),
            Part::Structured(structured) => {
                buf.push_str(&serde_json::to_string_pretty(&structured.value)?);
            }
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    buf.push_str(summary);
                }
            }
            _ => {
                return Err(BuildError::UnsupportedPart {
                    role: ItemKind::Notification,
                    part_kind: part_kind(part),
                });
            }
        }
    }
    Ok(json!({
        "text": format!("<system-reminder>\n{buf}\n</system-reminder>"),
    }))
}

fn build_user_parts(parts: &[Part]) -> Result<Vec<Value>, BuildError> {
    let mut out = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => out.push(json!({ "text": text.text })),
            Part::Structured(structured) => out.push(json!({
                "text": serde_json::to_string_pretty(&structured.value)?,
            })),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    out.push(json!({ "text": summary }));
                }
            }
            Part::Media(media) => out.push(media_to_part(media)?),
            Part::File(file) => out.push(file_to_part(file)?),
            Part::ToolCall(_) | Part::ToolResult(_) | Part::Custom(_) => {
                return Err(BuildError::UnsupportedPart {
                    role: ItemKind::User,
                    part_kind: part_kind(part),
                });
            }
        }
    }
    Ok(out)
}

fn build_model_parts(parts: &[Part]) -> Result<Vec<Value>, BuildError> {
    let mut out = Vec::new();
    for part in parts {
        match part {
            Part::Reasoning(reasoning) => {
                let signature = reasoning
                    .metadata
                    .get("gemini.thought_signature")
                    .and_then(Value::as_str);
                let mut obj = Map::new();
                if let Some(summary) = &reasoning.summary {
                    obj.insert("thought".into(), Value::Bool(true));
                    obj.insert("text".into(), Value::String(summary.clone()));
                } else if signature.is_some() {
                    obj.insert("thought".into(), Value::Bool(true));
                }
                if let Some(sig) = signature {
                    obj.insert("thoughtSignature".into(), Value::String(sig.to_string()));
                }
                if !obj.is_empty() {
                    out.push(Value::Object(obj));
                }
            }
            Part::Text(text) => {
                let signature = text
                    .metadata
                    .get("gemini.thought_signature")
                    .and_then(Value::as_str);
                if text.text.is_empty() && signature.is_none() {
                    continue;
                }
                let mut obj = Map::new();
                obj.insert("text".into(), Value::String(text.text.clone()));
                if let Some(sig) = signature {
                    obj.insert("thoughtSignature".into(), Value::String(sig.to_string()));
                }
                out.push(Value::Object(obj));
            }
            Part::Structured(structured) => {
                out.push(json!({
                    "text": serde_json::to_string(&structured.value)?,
                }));
            }
            Part::ToolCall(call) => {
                let mut fc = Map::new();
                fc.insert("name".into(), Value::String(call.name.clone()));
                fc.insert("args".into(), normalize_call_args(&call.input));
                if call
                    .metadata
                    .get("gemini.has_call_id")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    fc.insert("id".into(), Value::String(call.id.0.clone()));
                }
                let mut obj = Map::new();
                obj.insert("functionCall".into(), Value::Object(fc));
                if let Some(sig) = call
                    .metadata
                    .get("gemini.thought_signature")
                    .and_then(Value::as_str)
                {
                    obj.insert("thoughtSignature".into(), Value::String(sig.to_string()));
                }
                out.push(Value::Object(obj));
            }
            Part::Media(media) => out.push(media_to_part(media)?),
            Part::File(file) => out.push(file_to_part(file)?),
            Part::Custom(custom) => {
                if let Some(raw_key) = custom.kind.strip_prefix("gemini.")
                    && let Some(value) = &custom.value
                {
                    // If the stashed value is already a full part object containing the key,
                    // round-trip it verbatim; otherwise wrap it under its key.
                    if value
                        .as_object()
                        .is_some_and(|map| map.contains_key(raw_key))
                    {
                        out.push(value.clone());
                    } else {
                        let mut obj = Map::new();
                        obj.insert(raw_key.to_string(), value.clone());
                        out.push(Value::Object(obj));
                    }
                }
            }
            Part::ToolResult(_) => {
                return Err(BuildError::UnsupportedPart {
                    role: ItemKind::Assistant,
                    part_kind: PartKind::ToolResult,
                });
            }
        }
    }
    Ok(out)
}

fn normalize_call_args(input: &Value) -> Value {
    match input {
        Value::Object(_) => input.clone(),
        Value::Null => json!({}),
        other => json!({ "input": other }),
    }
}

fn build_tool_response_parts(
    parts: &[Part],
    call_lookup: &HashMap<String, (String, bool)>,
) -> Result<Vec<Value>, BuildError> {
    let mut out = Vec::new();
    for part in parts {
        let Part::ToolResult(result) = part else {
            return Err(BuildError::UnsupportedPart {
                role: ItemKind::Tool,
                part_kind: part_kind(part),
            });
        };
        let (name, has_id) = call_lookup
            .get(&result.call_id.0)
            .cloned()
            .or_else(|| {
                result
                    .metadata
                    .get("gemini.tool_name")
                    .or_else(|| result.metadata.get("tool_name"))
                    .and_then(Value::as_str)
                    .map(|name| (name.to_string(), false))
            })
            .ok_or_else(|| BuildError::MissingToolName(result.call_id.0.clone()))?;

        let (response_obj, multimodal_parts) =
            tool_output_to_gemini(&result.output, result.is_error)?;

        let mut fr = Map::new();
        fr.insert("name".into(), Value::String(name));
        fr.insert("response".into(), response_obj);
        if has_id {
            fr.insert("id".into(), Value::String(result.call_id.0.clone()));
        }
        if !multimodal_parts.is_empty() {
            fr.insert("parts".into(), Value::Array(multimodal_parts));
        }

        out.push(json!({ "functionResponse": fr }));
    }
    Ok(out)
}

fn tool_output_to_gemini(
    output: &ToolOutput,
    is_error: bool,
) -> Result<(Value, Vec<Value>), BuildError> {
    let key = if is_error { "error" } else { "output" };
    match output {
        ToolOutput::Text(text) => Ok((json!({ key: text }), Vec::new())),
        ToolOutput::Structured(value) => match value {
            Value::Object(_) if !is_error => Ok((value.clone(), Vec::new())),
            other => Ok((json!({ key: other }), Vec::new())),
        },
        ToolOutput::Parts(parts) => {
            let mut text_chunks = Vec::new();
            let mut multimodal = Vec::new();
            for part in parts {
                match part {
                    Part::Text(t) => text_chunks.push(t.text.clone()),
                    Part::Structured(s) => text_chunks.push(serde_json::to_string(&s.value)?),
                    Part::Media(m) => multimodal.push(media_to_part(m)?),
                    Part::File(f) => multimodal.push(file_to_part(f)?),
                    _ => {
                        return Err(BuildError::UnsupportedPart {
                            role: ItemKind::Tool,
                            part_kind: part_kind(part),
                        });
                    }
                }
            }
            let response = if text_chunks.is_empty() {
                json!({ key: "ok" })
            } else {
                json!({ key: text_chunks.join("\n") })
            };
            Ok((response, multimodal))
        }
        ToolOutput::Files(files) => {
            let mut multimodal = Vec::new();
            for file in files {
                multimodal.push(file_to_part(file)?);
            }
            Ok((json!({ key: "files_attached" }), multimodal))
        }
    }
}

fn build_function_declaration(spec: &ToolSpec) -> Value {
    let mut decl = Map::new();
    decl.insert("name".into(), Value::String(spec.name.to_string()));
    if !spec.description.is_empty() {
        decl.insert(
            "description".into(),
            Value::String(spec.description.clone()),
        );
    }
    if !spec.input_schema.is_null() {
        decl.insert("parametersJsonSchema".into(), spec.input_schema.clone());
    }
    Value::Object(decl)
}

fn part_kind(part: &Part) -> PartKind {
    match part {
        Part::Text(_) => PartKind::Text,
        Part::Structured(_) => PartKind::Structured,
        Part::Media(_) => PartKind::Media,
        Part::File(_) => PartKind::File,
        Part::ToolCall(_) => PartKind::ToolCall,
        Part::ToolResult(_) => PartKind::ToolResult,
        Part::Reasoning(_) => PartKind::Reasoning,
        Part::Custom(_) => PartKind::Custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GeminiServerTool, ThinkingConfig, ThinkingLevel, ToolChoice};
    use agentkit_core::{
        DataRef, MediaPart, MetadataMap, Modality, ReasoningPart, SessionId, TextPart,
        ToolCallPart, ToolResultPart, TurnId,
    };

    fn empty_turn(transcript: Vec<Item>, tools: Vec<ToolSpec>) -> TurnRequest {
        TurnRequest {
            session_id: SessionId::new("s1"),
            turn_id: TurnId::new("t1"),
            transcript,
            available_tools: tools,
            cache: None,
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn hoists_system_and_coalesces_user_and_notification() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let req = empty_turn(
            vec![
                Item::text(ItemKind::System, "Be concise."),
                Item::text(ItemKind::Developer, "Use JSON."),
                Item::text(ItemKind::User, "Hello"),
                Item::notification("Background task done"),
            ],
            vec![],
        );
        let body = build_request_body(&cfg, &req).unwrap();
        assert_eq!(
            body["systemInstruction"]["parts"],
            json!([{ "text": "Be concise." }, { "text": "Use JSON." }])
        );
        assert_eq!(body["contents"].as_array().unwrap().len(), 1);
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "Hello");
        assert!(
            body["contents"][0]["parts"][1]["text"]
                .as_str()
                .unwrap()
                .contains("<system-reminder>")
        );
    }

    #[test]
    fn round_trips_thought_signature_and_correlates_tool_responses() {
        let cfg = GeminiConfig::new("k", "gemini-3-pro")
            .unwrap()
            .with_thinking(ThinkingConfig::level(ThinkingLevel::High))
            .with_tool_choice(ToolChoice::Auto)
            .with_server_tool(GeminiServerTool::GoogleSearch);

        let mut call_meta = MetadataMap::new();
        call_meta.insert("gemini.thought_signature".into(), json!("sig-xyz"));
        call_meta.insert("gemini.has_call_id".into(), json!(true));

        let mut call = ToolCallPart::new("fc_1", "get_weather", json!({ "city": "Tokyo" }));
        call.metadata = call_meta;

        let mut reasoning_meta = MetadataMap::new();
        reasoning_meta.insert("gemini.thought_signature".into(), json!("rsig-1"));
        let reasoning = ReasoningPart::summary("Thinking...").with_metadata(reasoning_meta);

        let result = ToolResultPart::success("fc_1", ToolOutput::Text("22C sunny".into()));

        let req = empty_turn(
            vec![
                Item::text(ItemKind::User, "Weather in Tokyo?"),
                Item::new(
                    ItemKind::Assistant,
                    vec![Part::Reasoning(reasoning), Part::ToolCall(call)],
                ),
                Item::new(ItemKind::Tool, vec![Part::ToolResult(result)]),
            ],
            vec![ToolSpec::new(
                "get_weather",
                "Gets weather",
                json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            )],
        );

        let body = build_request_body(&cfg, &req).unwrap();
        let model_parts = body["contents"][1]["parts"].as_array().unwrap();
        assert_eq!(model_parts[0]["thought"], true);
        assert_eq!(model_parts[0]["thoughtSignature"], "rsig-1");
        assert_eq!(model_parts[1]["functionCall"]["name"], "get_weather");
        assert_eq!(model_parts[1]["functionCall"]["id"], "fc_1");
        assert_eq!(model_parts[1]["thoughtSignature"], "sig-xyz");

        let tool_part = &body["contents"][2]["parts"][0]["functionResponse"];
        assert_eq!(tool_part["name"], "get_weather");
        assert_eq!(tool_part["id"], "fc_1");
        assert_eq!(tool_part["response"], json!({ "output": "22C sunny" }));

        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["type"],
            "object"
        );
        assert_eq!(body["tools"][1], json!({ "googleSearch": {} }));
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({ "thinkingLevel": "HIGH", "includeThoughts": true })
        );
    }

    #[test]
    fn encodes_multimodal_inline_and_uri_media() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let req = empty_turn(
            vec![Item::new(
                ItemKind::User,
                vec![
                    Part::Text(TextPart::new("Describe these")),
                    Part::Media(MediaPart::new(
                        Modality::Image,
                        "image/png",
                        DataRef::InlineBytes(vec![1, 2, 3]),
                    )),
                    Part::Media(MediaPart::new(
                        Modality::Audio,
                        "audio/mp3",
                        DataRef::Uri("gs://bucket/clip.mp3".into()),
                    )),
                ],
            )],
            vec![],
        );
        let body = build_request_body(&cfg, &req).unwrap();
        let parts = body["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "AQID");
        assert_eq!(parts[2]["fileData"]["mimeType"], "audio/mp3");
        assert_eq!(parts[2]["fileData"]["fileUri"], "gs://bucket/clip.mp3");
    }
}
