use agentkit_core::{Item, ItemKind, Part, PartKind};
use agentkit_loop::TurnRequest;
use serde_json::{Value, json};

use crate::CompletionsProvider;
use crate::error::CompletionsError;
use crate::media::{file_to_content, media_to_content};

pub(crate) fn build_request_body<P: CompletionsProvider>(
    provider: &P,
    request: &TurnRequest,
) -> Result<Value, CompletionsError> {
    let mut body = serde_json::Map::new();

    // Merge provider config (model, temperature, etc.) into body
    let config_value =
        serde_json::to_value(provider.config()).map_err(CompletionsError::Serialize)?;
    if let Value::Object(fields) = config_value {
        for (key, value) in fields {
            body.insert(key, value);
        }
    }

    body.insert(
        "messages".into(),
        Value::Array(build_messages(&request.transcript)?),
    );
    body.insert("stream".into(), Value::Bool(false));
    body.insert(
        "tools".into(),
        Value::Array(build_tools(&request.available_tools)),
    );
    body.insert(
        "parallel_tool_calls".into(),
        Value::Bool(!request.available_tools.is_empty()),
    );
    body.insert("user".into(), Value::String(request.session_id.0.clone()));

    Ok(Value::Object(body))
}

fn build_tools(tool_specs: &[agentkit_tools_core::ToolSpec]) -> Vec<Value> {
    tool_specs
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name.0,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect()
}

fn build_messages(transcript: &[Item]) -> Result<Vec<Value>, CompletionsError> {
    let mut messages = Vec::new();

    for item in transcript {
        match item.kind {
            ItemKind::Tool => {
                for part in &item.parts {
                    let Part::ToolResult(result) = part else {
                        return Err(CompletionsError::UnsupportedPart {
                            role: item.kind,
                            part_kind: part_kind(part),
                        });
                    };
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": result.call_id.0,
                        "content": tool_output_to_string(&result.output),
                    }));
                }
            }
            _ => messages.push(build_message(item)?),
        }
    }

    Ok(messages)
}

fn build_message(item: &Item) -> Result<Value, CompletionsError> {
    match item.kind {
        ItemKind::System | ItemKind::Developer | ItemKind::Context => Ok(json!({
            "role": role_for_item_kind(item.kind),
            "content": stringify_parts(&item.parts, item.kind)?,
        })),
        ItemKind::User => Ok(json!({
            "role": "user",
            "content": build_user_content(&item.parts)?,
        })),
        ItemKind::Assistant => build_assistant_message(item),
        ItemKind::Tool => Err(CompletionsError::InvalidTranscript(
            "tool items must be expanded at the transcript level".into(),
        )),
    }
}

fn build_assistant_message(item: &Item) -> Result<Value, CompletionsError> {
    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();

    for part in &item.parts {
        match part {
            Part::Text(text) => content_parts.push(json!({
                "type": "text",
                "text": text.text,
            })),
            Part::Structured(structured) => content_parts.push(json!({
                "type": "text",
                "text": serde_json::to_string(&structured.value).map_err(CompletionsError::Serialize)?,
            })),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    content_parts.push(json!({
                        "type": "text",
                        "text": summary,
                    }));
                }
            }
            Part::ToolCall(call) => tool_calls.push(json!({
                "id": call.id.0,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": serde_json::to_string(&call.input).map_err(CompletionsError::Serialize)?,
                }
            })),
            Part::ToolResult(_) => {
                return Err(CompletionsError::UnsupportedPart {
                    role: item.kind,
                    part_kind: PartKind::ToolResult,
                });
            }
            Part::Media(_) | Part::File(_) | Part::Custom(_) => {
                return Err(CompletionsError::UnsupportedPart {
                    role: item.kind,
                    part_kind: part_kind(part),
                });
            }
        }
    }

    let content = if content_parts.is_empty() {
        Value::Null
    } else if content_parts.len() == 1 && content_parts[0]["type"] == "text" {
        Value::String(
            content_parts[0]["text"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        )
    } else {
        Value::Array(content_parts)
    };

    Ok(json!({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }))
}

fn build_user_content(parts: &[Part]) -> Result<Value, CompletionsError> {
    let mut content = Vec::new();

    for part in parts {
        match part {
            Part::Text(text) => content.push(json!({
                "type": "text",
                "text": text.text,
            })),
            Part::Structured(structured) => content.push(json!({
                "type": "text",
                "text": serde_json::to_string_pretty(&structured.value)
                    .map_err(CompletionsError::Serialize)?,
            })),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    content.push(json!({
                        "type": "text",
                        "text": summary,
                    }));
                }
            }
            Part::Media(media) => content.push(media_to_content(media)?),
            Part::File(file) => content.push(file_to_content(file)?),
            Part::ToolCall(_) | Part::ToolResult(_) | Part::Custom(_) => {
                return Err(CompletionsError::UnsupportedPart {
                    role: ItemKind::User,
                    part_kind: part_kind(part),
                });
            }
        }
    }

    if content.len() == 1 && content[0]["type"] == "text" {
        Ok(Value::String(
            content[0]["text"].as_str().unwrap_or_default().to_string(),
        ))
    } else {
        Ok(Value::Array(content))
    }
}

fn stringify_parts(parts: &[Part], role: ItemKind) -> Result<String, CompletionsError> {
    let mut segments = Vec::new();

    for part in parts {
        match part {
            Part::Text(text) => segments.push(text.text.clone()),
            Part::Structured(structured) => segments.push(
                serde_json::to_string_pretty(&structured.value)
                    .map_err(CompletionsError::Serialize)?,
            ),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    segments.push(summary.clone());
                }
            }
            _ => {
                return Err(CompletionsError::UnsupportedPart {
                    role,
                    part_kind: part_kind(part),
                });
            }
        }
    }

    Ok(segments.join("\n\n"))
}

fn role_for_item_kind(kind: ItemKind) -> &'static str {
    match kind {
        ItemKind::System | ItemKind::Context => "system",
        ItemKind::Developer => "developer",
        ItemKind::User => "user",
        ItemKind::Assistant => "assistant",
        ItemKind::Tool => "tool",
    }
}

fn tool_output_to_string(output: &agentkit_core::ToolOutput) -> String {
    match output {
        agentkit_core::ToolOutput::Text(text) => text.clone(),
        agentkit_core::ToolOutput::Structured(value) => value.to_string(),
        agentkit_core::ToolOutput::Parts(parts) => {
            serde_json::to_string(parts).unwrap_or_else(|_| "[]".into())
        }
        agentkit_core::ToolOutput::Files(files) => {
            serde_json::to_string(files).unwrap_or_else(|_| "[]".into())
        }
    }
}

fn part_kind(part: &Part) -> PartKind {
    match part {
        Part::Text(_) => PartKind::Text,
        Part::Media(_) => PartKind::Media,
        Part::File(_) => PartKind::File,
        Part::Structured(_) => PartKind::Structured,
        Part::Reasoning(_) => PartKind::Reasoning,
        Part::ToolCall(_) => PartKind::ToolCall,
        Part::ToolResult(_) => PartKind::ToolResult,
        Part::Custom(_) => PartKind::Custom,
    }
}
