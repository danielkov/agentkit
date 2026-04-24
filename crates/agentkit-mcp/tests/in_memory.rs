//! Integration tests against an in-memory rmcp server connected over tokio
//! duplex pipes. Replaces the legacy fake-transport unit tests; covers the
//! agentkit-side adapters end-to-end through a real rmcp client+server pair.

use std::sync::Arc;

use agentkit_capabilities::{CapabilityContext, Invocable, InvocableOutput, InvocableRequest};
use agentkit_core::{DataRef, MetadataMap, ToolOutput};
use agentkit_mcp::{
    McpClientHandler, McpConnection, McpInvocable, McpServerCapabilities, McpServerId,
    McpToolAdapter,
};
use agentkit_tools_core::{
    PermissionChecker, PermissionDecision, PermissionRequest, Tool, ToolContext, ToolName,
    ToolRequest,
};
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    Annotated, CallToolResult, Content, ErrorData as McpError, GetPromptRequestParams,
    GetPromptResult, ListPromptsResult, ListResourcesResult, PaginatedRequestParams, Prompt,
    PromptArgument, PromptMessage, PromptMessageRole, RawResource, ReadResourceRequestParams,
    ReadResourceResult, ResourceContents, ServerCapabilities, ServerInfo,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::Deserialize;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct EchoArgs {
    text: String,
}

#[derive(Clone)]
struct InMemoryServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl Default for InMemoryServer {
    fn default() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl InMemoryServer {
    #[tool(description = "Echoes the supplied text back as the tool result.")]
    async fn echo(
        &self,
        Parameters(EchoArgs { text }): Parameters<EchoArgs>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for InMemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_prompts()
                .build(),
        )
    }

    async fn list_resources(
        &self,
        _params: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(vec![Annotated::new(
            RawResource::new("memo:welcome", "welcome"),
            None,
        )]))
    }

    async fn read_resource(
        &self,
        params: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        if params.uri == "memo:welcome" {
            Ok(ReadResourceResult::new(vec![ResourceContents::text(
                "hello from the in-memory MCP server",
                params.uri,
            )
            .with_mime_type("text/plain")]))
        } else {
            Err(McpError::invalid_params("unknown resource", None))
        }
    }

    async fn list_prompts(
        &self,
        _params: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult::with_all_items(vec![Prompt::new(
            "summarize",
            Some("Summarize the supplied text."),
            Some(vec![
                PromptArgument::new("text")
                    .with_description("Text to summarize.")
                    .with_required(true),
            ]),
        )]))
    }

    async fn get_prompt(
        &self,
        params: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        if params.name != "summarize" {
            return Err(McpError::invalid_params("unknown prompt", None));
        }
        let argument = params
            .arguments
            .as_ref()
            .and_then(|args| args.get("text"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_owned();
        Ok(GetPromptResult::new(vec![PromptMessage::new_text(
            PromptMessageRole::User,
            format!("Please summarize: {argument}"),
        )])
        .with_description("Summarize the supplied text."))
    }
}

async fn connect_in_memory() -> McpConnection {
    let (server_io, client_io) = tokio::io::duplex(8 * 1024);
    tokio::spawn(async move {
        match InMemoryServer::default().serve(server_io).await {
            Ok(running) => {
                let _ = running.waiting().await;
            }
            Err(error) => {
                eprintln!("server init failed: {error:?}");
            }
        }
    });

    let (handler, notifications) = McpClientHandler::with_channel();
    let service = handler
        .serve(client_io)
        .await
        .expect("client init succeeds");

    McpConnection::from_running_service(McpServerId::new("in-memory"), service, notifications)
}

struct AllowAll;

impl PermissionChecker for AllowAll {
    fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

#[tokio::test]
async fn capabilities_reflect_server_advertisement() {
    let connection = connect_in_memory().await;
    let caps = connection.capabilities();
    assert!(caps.tools.is_some());
    assert!(caps.resources.is_some());
    assert!(caps.prompts.is_some());
}

#[tokio::test]
async fn capabilities_all_helper_returns_full_set() {
    let caps = McpServerCapabilities::all();
    assert!(caps.tools.is_some());
    assert!(caps.resources.is_some());
    assert!(caps.prompts.is_some());
    assert!(caps.logging.is_some());
}

#[tokio::test]
async fn discovery_returns_advertised_tools_resources_prompts() {
    let connection = connect_in_memory().await;
    let snapshot = connection.discover().await.expect("discover succeeds");

    assert_eq!(snapshot.tools.len(), 1);
    assert_eq!(snapshot.tools[0].name, "echo");

    assert_eq!(snapshot.resources.len(), 1);
    assert_eq!(snapshot.resources[0].id, "memo:welcome");
    assert_eq!(snapshot.resources[0].name, "welcome");

    assert_eq!(snapshot.prompts.len(), 1);
    assert_eq!(snapshot.prompts[0].id, "summarize");
}

#[tokio::test]
async fn call_tool_returns_text_payload() {
    let connection = connect_in_memory().await;
    let value = connection
        .call_tool("echo", serde_json::json!({ "text": "hi" }))
        .await
        .expect("tool call succeeds");
    let content_text = value
        .get("content")
        .and_then(|content| content.as_array())
        .and_then(|content| content.first())
        .and_then(|first| first.get("text"))
        .and_then(|text| text.as_str())
        .unwrap_or_default();
    assert_eq!(content_text, "hi");
}

#[tokio::test]
async fn read_resource_returns_inline_text() {
    let connection = connect_in_memory().await;
    let resource = connection
        .read_resource("memo:welcome")
        .await
        .expect("resource read succeeds");
    match resource.data {
        DataRef::InlineText(text) => {
            assert_eq!(text, "hello from the in-memory MCP server");
        }
        other => panic!("unexpected data ref: {other:?}"),
    }
}

#[tokio::test]
async fn get_prompt_renders_user_message() {
    let connection = connect_in_memory().await;
    let prompt = connection
        .get_prompt("summarize", serde_json::json!({ "text": "essay body" }))
        .await
        .expect("prompt fetch succeeds");
    assert_eq!(prompt.items.len(), 1);
    let Some(agentkit_core::Part::Text(text_part)) = prompt.items[0].parts.first() else {
        panic!("expected a text prompt message");
    };
    assert!(text_part.text.contains("essay body"));
}

#[tokio::test]
async fn tool_adapter_propagates_call_through_running_service() {
    let connection = Arc::new(connect_in_memory().await);
    let server_id = connection.server_id().clone();
    let snapshot = connection.discover().await.unwrap();
    let descriptor = snapshot.tools[0].clone();

    let adapter = McpToolAdapter::new(&server_id, connection.clone(), descriptor);
    assert_eq!(adapter.spec().name.0.as_str(), "mcp_in-memory_echo");

    let metadata = MetadataMap::new();
    let mut ctx = ToolContext {
        capability: CapabilityContext {
            session_id: None,
            turn_id: None,
            metadata: &metadata,
        },
        permissions: &AllowAll,
        resources: &(),
        cancellation: None,
    };

    let result = adapter
        .invoke(
            ToolRequest {
                call_id: "call-1".into(),
                tool_name: ToolName::new("mcp_in-memory_echo"),
                input: serde_json::json!({ "text": "via-adapter" }),
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                metadata: MetadataMap::new(),
            },
            &mut ctx,
        )
        .await
        .expect("adapter invoke succeeds");

    match result.result.output {
        ToolOutput::Text(text) => assert_eq!(text, "via-adapter"),
        other => panic!("unexpected output: {other:?}"),
    }
}

#[tokio::test]
async fn invocable_adapter_returns_text_output() {
    let connection = Arc::new(connect_in_memory().await);
    let snapshot = connection.discover().await.unwrap();
    let invocable = McpInvocable::new(connection.clone(), snapshot.tools[0].clone());

    let metadata = MetadataMap::new();
    let mut ctx = CapabilityContext {
        session_id: None,
        turn_id: None,
        metadata: &metadata,
    };

    let result = invocable
        .invoke(
            InvocableRequest::new(serde_json::json!({ "text": "hello-invocable" })),
            &mut ctx,
        )
        .await
        .expect("invocable invoke succeeds");

    match result.output {
        InvocableOutput::Text(text) => assert_eq!(text, "hello-invocable"),
        other => panic!("unexpected output: {other:?}"),
    }
}
