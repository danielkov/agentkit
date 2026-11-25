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

#[derive(Clone)]
struct CallOrigin {
    name: String,
    has_provider_id: bool,
    batch_id: usize,
    call_index: usize,
}

/// Builds the Gemini `generateContent` / `streamGenerateContent` JSON request body.
pub(crate) fn build_request_body(
    config: &GeminiConfig,
    request: &TurnRequest,
) -> Result<Value, BuildError> {
    validate_prompt_cache(request)?;

    let cache_disabled = matches!(
        request.cache.as_ref().map(|c| c.mode),
        Some(PromptCacheMode::Disabled)
    );
    let active_cached_content = if cache_disabled {
        None
    } else {
        config.cached_content.as_deref()
    };

    if active_cached_content.is_some()
        && (!config.server_tools.is_empty() || config.tool_choice.is_some())
    {
        return Err(BuildError::CacheViolation(
            "cannot combine cachedContent with live server_tools or tool_choice; define tool declarations and toolConfig in the cached content resource".into(),
        ));
    }

    // Track call_id -> CallOrigin from assistant tool calls so subsequent Tool
    // items can populate Gemini's required `functionResponse.name` and restore
    // parallel invocation order.
    let mut call_lookup: HashMap<String, CallOrigin> = HashMap::new();
    let mut next_batch_id: usize = 0;
    let mut system_parts: Vec<Value> = Vec::new();
    let mut contents: Vec<ContentMessage> = Vec::new();

    for item in &request.transcript {
        match item.kind {
            ItemKind::System | ItemKind::Developer | ItemKind::Context => {
                extend_system_parts(&mut system_parts, item)?;
            }
            ItemKind::User => {
                let parts = build_user_parts(&item.parts)?;
                append_content(
                    &mut contents,
                    "user",
                    parts.into_iter().map(ContentPart::plain).collect(),
                );
            }
            ItemKind::Assistant => {
                let mut call_index: usize = 0;
                let batch_id = next_batch_id;
                for part in &item.parts {
                    if let Part::ToolCall(call) = part {
                        let has_provider_id = call
                            .metadata
                            .get("gemini.has_call_id")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        call_lookup.insert(
                            call.id.0.clone(),
                            CallOrigin {
                                name: call.name.clone(),
                                has_provider_id,
                                batch_id,
                                call_index,
                            },
                        );
                        call_index += 1;
                    }
                }
                if call_index > 0 {
                    next_batch_id += 1;
                }
                let parts = build_model_parts(&item.parts)?;
                if !parts.is_empty() {
                    append_content(
                        &mut contents,
                        "model",
                        parts.into_iter().map(ContentPart::plain).collect(),
                    );
                }
            }
            ItemKind::Tool => {
                let parts = build_tool_response_parts(&item.parts, &call_lookup)?;
                append_content(&mut contents, "user", parts);
            }
            ItemKind::Notification => {
                let part = build_notification_part(&item.parts)?;
                append_content(&mut contents, "user", vec![ContentPart::plain(part)]);
            }
        }
    }

    let mut body = Map::new();

    // When explicit `cachedContent` is active, omit cache-owned fields
    // (`systemInstruction`, `tools`, `toolConfig`) so Vertex/Gemini do not
    // reject redeclared context while keeping registered tools locally executable.
    if let Some(cached) = active_cached_content {
        body.insert("cachedContent".into(), Value::String(cached.to_string()));
    } else {
        if !system_parts.is_empty() {
            body.insert("systemInstruction".into(), json!({ "parts": system_parts }));
        }

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

        let mixed_tools = !request.available_tools.is_empty() && !config.server_tools.is_empty();
        let mut tool_config = match &config.tool_choice {
            Some(choice) => choice.to_json().as_object().cloned().unwrap_or_default(),
            None => Map::new(),
        };
        if mixed_tools {
            tool_config.insert("includeServerSideToolInvocations".into(), Value::Bool(true));
        }
        if !tool_config.is_empty() {
            body.insert("toolConfig".into(), Value::Object(tool_config));
        }
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

    if !config.safety_settings.is_empty() {
        body.insert(
            "safetySettings".into(),
            Value::Array(config.safety_settings.iter().map(|s| s.to_json()).collect()),
        );
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

struct ContentPart {
    value: Value,
    /// Optional `(batch_id, call_index)` for `functionResponse` parts so
    /// parallel responses are reordered to match their model invocation order.
    tool_order: Option<(usize, usize)>,
}

impl ContentPart {
    fn plain(value: Value) -> Self {
        Self {
            value,
            tool_order: None,
        }
    }

    fn tool_response(value: Value, order: Option<(usize, usize)>) -> Self {
        Self {
            value,
            tool_order: order,
        }
    }
}

struct ContentMessage {
    role: &'static str,
    parts: Vec<ContentPart>,
}

impl ContentMessage {
    fn into_value(mut self) -> Value {
        // Reorder contiguous runs of keyed `functionResponse` parts so parallel
        // tool completions always arrive in original function-call order.
        let mut i = 0;
        while i < self.parts.len() {
            if self.parts[i].tool_order.is_some() {
                let start = i;
                while i < self.parts.len() && self.parts[i].tool_order.is_some() {
                    i += 1;
                }
                self.parts[start..i].sort_by_key(|p| p.tool_order.unwrap());
            } else {
                i += 1;
            }
        }
        json!({
            "role": self.role,
            "parts": self.parts.into_iter().map(|p| p.value).collect::<Vec<_>>(),
        })
    }
}

fn append_content(contents: &mut Vec<ContentMessage>, role: &'static str, parts: Vec<ContentPart>) {
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
            Part::Media(media) => {
                let mut part = media_to_part(media)?;
                if let Some(sig) = media
                    .metadata
                    .get("gemini.thought_signature")
                    .and_then(Value::as_str)
                    && let Some(obj) = part.as_object_mut()
                {
                    obj.insert("thoughtSignature".into(), Value::String(sig.to_string()));
                }
                out.push(part);
            }
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
            Part::Media(media) => {
                let mut part = media_to_part(media)?;
                if let Some(sig) = media
                    .metadata
                    .get("gemini.thought_signature")
                    .and_then(Value::as_str)
                    && let Some(obj) = part.as_object_mut()
                {
                    obj.insert("thoughtSignature".into(), Value::String(sig.to_string()));
                }
                out.push(part);
            }
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
    call_lookup: &HashMap<String, CallOrigin>,
) -> Result<Vec<ContentPart>, BuildError> {
    let mut out = Vec::new();
    for part in parts {
        let Part::ToolResult(result) = part else {
            return Err(BuildError::UnsupportedPart {
                role: ItemKind::Tool,
                part_kind: part_kind(part),
            });
        };
        let (name, has_id, order) = match call_lookup.get(&result.call_id.0) {
            Some(origin) => (
                origin.name.clone(),
                origin.has_provider_id,
                Some((origin.batch_id, origin.call_index)),
            ),
            None => {
                let name = result
                    .metadata
                    .get("gemini.tool_name")
                    .or_else(|| result.metadata.get("tool_name"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| BuildError::MissingToolName(result.call_id.0.clone()))?;
                (name, false, None)
            }
        };

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

        out.push(ContentPart::tool_response(
            json!({ "functionResponse": fr }),
            order,
        ));
    }
    Ok(out)
}

fn push_tool_file_part(
    file: &agentkit_core::FilePart,
    text_chunks: &mut Vec<String>,
    multimodal: &mut Vec<Value>,
) -> Result<(), BuildError> {
    let encoded = file_to_part(file)?;
    if let Some(text) = encoded.get("text").and_then(Value::as_str)
        && encoded.get("inlineData").is_none()
        && encoded.get("fileData").is_none()
    {
        text_chunks.push(text.to_string());
    } else {
        multimodal.push(encoded);
    }
    Ok(())
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
                    Part::File(f) => push_tool_file_part(f, &mut text_chunks, &mut multimodal)?,
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
            let mut text_chunks = Vec::new();
            let mut multimodal = Vec::new();
            for file in files {
                push_tool_file_part(file, &mut text_chunks, &mut multimodal)?;
            }
            let response = if text_chunks.is_empty() {
                json!({ key: "files_attached" })
            } else {
                json!({ key: text_chunks.join("\n") })
            };
            Ok((response, multimodal))
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

    #[test]
    fn orders_parallel_tool_responses_by_function_call_order() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let call = |id, n| Part::ToolCall(ToolCallPart::new(id, "lookup", json!({ "n": n })));
        let result = |id, text: &str| {
            Item::new(
                ItemKind::Tool,
                vec![Part::ToolResult(ToolResultPart::success(
                    id,
                    ToolOutput::Text(text.into()),
                ))],
            )
        };
        let model = Item::new(ItemKind::Assistant, vec![call("a", 1), call("b", 2)]);
        let req = empty_turn(
            vec![model, result("b", "second"), result("a", "first")],
            vec![],
        );
        let body = build_request_body(&cfg, &req).unwrap();
        assert_eq!(
            body["contents"][1]["parts"][0]["functionResponse"]["response"]["output"],
            "first"
        );
        assert_eq!(
            body["contents"][1]["parts"][1]["functionResponse"]["response"]["output"],
            "second"
        );
    }

    #[test]
    fn enables_server_side_tool_invocations_for_mixed_tools() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_server_tool(GeminiServerTool::GoogleSearch);
        let req = empty_turn(
            vec![Item::text(ItemKind::User, "Hello")],
            vec![ToolSpec::new(
                "local_fn",
                "desc",
                json!({ "type": "object" }),
            )],
        );
        let body = build_request_body(&cfg, &req).unwrap();
        assert_eq!(body["toolConfig"]["includeServerSideToolInvocations"], true);
    }

    #[test]
    fn honors_cache_disablement_and_omits_cache_owned_fields() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash")
            .unwrap()
            .with_cached_content("cachedContents/example");
        let mut req = empty_turn(
            vec![
                Item::text(ItemKind::System, "sys"),
                Item::text(ItemKind::User, "Hello"),
            ],
            vec![ToolSpec::new(
                "local_fn",
                "desc",
                json!({ "type": "object" }),
            )],
        );
        let cached_body = build_request_body(&cfg, &req).unwrap();
        assert_eq!(cached_body["cachedContent"], "cachedContents/example");
        assert!(cached_body.get("systemInstruction").is_none());
        assert!(cached_body.get("tools").is_none());
        assert!(cached_body.get("toolConfig").is_none());

        req.cache = Some(agentkit_loop::PromptCacheRequest::disabled());
        let uncached_body = build_request_body(&cfg, &req).unwrap();
        assert!(uncached_body.get("cachedContent").is_none());
        assert!(uncached_body.get("systemInstruction").is_some());
        assert!(uncached_body.get("tools").is_some());
    }

    #[test]
    fn keeps_plain_text_files_out_of_function_response_parts() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let file = agentkit_core::FilePart {
            name: Some("note.txt".into()),
            mime_type: Some("text/plain".into()),
            data: DataRef::InlineText("hello from file".into()),
            metadata: MetadataMap::new(),
        };
        let req = empty_turn(
            vec![
                Item::new(
                    ItemKind::Assistant,
                    vec![Part::ToolCall(ToolCallPart::new("c1", "read", json!({})))],
                ),
                Item::new(
                    ItemKind::Tool,
                    vec![Part::ToolResult(ToolResultPart::success(
                        "c1",
                        ToolOutput::Files(vec![file]),
                    ))],
                ),
            ],
            vec![],
        );
        let body = build_request_body(&cfg, &req).unwrap();
        let fr = &body["contents"][1]["parts"][0]["functionResponse"];
        assert_eq!(fr["response"]["output"], "hello from file");
        assert!(fr.get("parts").is_none());
    }

    #[test]
    fn round_trips_thought_signature_on_assistant_media() {
        let cfg = GeminiConfig::new("k", "gemini-2.5-flash").unwrap();
        let mut media = MediaPart::new(
            Modality::Image,
            "image/png",
            DataRef::InlineBytes(vec![1, 2, 3]),
        );
        media
            .metadata
            .insert("gemini.thought_signature".into(), json!("img-sig-1"));
        let req = empty_turn(
            vec![Item::new(ItemKind::Assistant, vec![Part::Media(media)])],
            vec![],
        );
        let body = build_request_body(&cfg, &req).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0]["thoughtSignature"],
            "img-sig-1"
        );
    }
}
