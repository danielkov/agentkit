//! Integration test for the `#[tool]` attribute macro.
//!
//! Exercises name/description defaults, name override, doc-comment-as-
//! description, schema derivation, and the round-trip through
//! `Tool::invoke`.

use agentkit_core::{MetadataMap, SessionId, ToolCallId, ToolOutput, ToolResultPart, TurnId};
use agentkit_tools_core::{
    AllowAllPermissions, OwnedToolContext, Tool, ToolError, ToolName, ToolRequest, ToolResult,
};
use agentkit_tools_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(JsonSchema, Deserialize)]
struct GreetInput {
    /// Person to greet.
    name: String,
}

/// Greet a person by name.
#[tool]
async fn greet(input: GreetInput) -> Result<ToolResult, ToolError> {
    Ok(ToolResult::new(ToolResultPart::success(
        ToolCallId::new("greet-call"),
        ToolOutput::text(format!("hello, {}", input.name)),
    )))
}

#[derive(JsonSchema, Deserialize)]
struct EchoInput {
    value: String,
}

#[tool(name = "speak", description = "Echo the input value back")]
async fn echo(input: EchoInput) -> Result<ToolResult, ToolError> {
    Ok(ToolResult::new(ToolResultPart::success(
        ToolCallId::new("echo-call"),
        ToolOutput::text(input.value),
    )))
}

fn build_request(input: serde_json::Value) -> ToolRequest {
    ToolRequest {
        call_id: ToolCallId::new("c"),
        tool_name: ToolName::new("greet"),
        input,
        session_id: SessionId::new("s"),
        turn_id: TurnId::new("t"),
        metadata: MetadataMap::new(),
    }
}

fn ctx() -> OwnedToolContext {
    OwnedToolContext {
        session_id: SessionId::new("s"),
        turn_id: TurnId::new("t"),
        metadata: MetadataMap::new(),
        permissions: Arc::new(AllowAllPermissions),
        resources: Arc::new(()),
        cancellation: None,
    }
}

#[tokio::test]
async fn tool_attribute_defaults_name_to_function_ident() {
    let spec = greet.spec();
    assert_eq!(spec.name.0, "greet");
    assert_eq!(spec.description, "Greet a person by name.");
    assert!(spec.input_schema.is_object());
    assert!(
        spec.input_schema
            .get("properties")
            .and_then(|p| p.get("name"))
            .is_some()
    );
}

#[tokio::test]
async fn tool_attribute_invoke_decodes_input_and_returns_result() {
    let owned = ctx();
    let mut ctx = owned.borrowed();
    let request = build_request(json!({ "name": "world" }));
    let result = greet.invoke(request, &mut ctx).await.unwrap();
    match result.result.output {
        ToolOutput::Text(text) => assert_eq!(text, "hello, world"),
        other => panic!("unexpected output: {other:?}"),
    }
}

#[tokio::test]
async fn tool_attribute_explicit_name_and_description_override() {
    let spec = echo.spec();
    assert_eq!(spec.name.0, "speak");
    assert_eq!(spec.description, "Echo the input value back");
}

#[tokio::test]
async fn tool_attribute_invoke_propagates_invalid_input_error() {
    let owned = ctx();
    let mut ctx = owned.borrowed();
    let request = build_request(json!({ "wrong_field": "world" }));
    let err = greet.invoke(request, &mut ctx).await.unwrap_err();
    assert!(matches!(err, ToolError::InvalidInput(_)));
}
