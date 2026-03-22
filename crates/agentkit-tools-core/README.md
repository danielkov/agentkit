# agentkit-tools-core

Core abstractions for defining, registering, executing, and governing tools in agentkit.

This crate provides:

- **`Tool` trait** — the interface every tool implements (spec + async invoke)
- **`ToolRegistry`** — a name-keyed collection of tools
- **`BasicToolExecutor`** — looks up tools, checks permissions, invokes them
- **Permission system** — composable policies (`PathPolicy`, `CommandPolicy`, `McpServerPolicy`, `CustomKindPolicy`) combined via `CompositePermissionChecker`
- **Interruption types** — `ApprovalRequest` and `AuthRequest` for human-in-the-loop flows
- **Capability bridge** — `ToolCapabilityProvider` adapts a registry into the agentkit capability layer

## Defining a tool

Implement the `Tool` trait with a `ToolSpec` and an async `invoke` method.
If the tool performs operations that need permission checks (filesystem
access, shell commands), override `proposed_requests` to declare them.

```rust,no_run
use agentkit_core::{MetadataMap, ToolOutput, ToolResultPart};
use agentkit_tools_core::{
    FileSystemPermissionRequest, PermissionRequest, Tool, ToolContext, ToolError,
    ToolName, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;

/// A tool that reads a file and returns its contents.
struct ReadFileTool {
    spec: ToolSpec,
}

impl ReadFileTool {
    fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("read_file"),
                description: "Read a file from the local filesystem".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute file path" }
                    },
                    "required": ["path"]
                }),
                annotations: Default::default(),
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    // Declare that invoking this tool requires filesystem-read permission.
    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let path = request.input["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        Ok(vec![Box::new(FileSystemPermissionRequest::Read {
            path: PathBuf::from(path),
            metadata: MetadataMap::new(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let path = request.input["path"].as_str().unwrap();
        let content = std::fs::read_to_string(path)
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Text(content),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: None,
            metadata: MetadataMap::new(),
        })
    }
}
```

## Registering tools and executing them

Build a `ToolRegistry`, wrap it in a `BasicToolExecutor`, and call
`execute` with a `ToolRequest` and `ToolContext`.

```rust,no_run
use agentkit_capabilities::CapabilityContext;
use agentkit_core::{MetadataMap, SessionId, ToolOutput, ToolResultPart, TurnId};
use agentkit_tools_core::{
    BasicToolExecutor, PermissionChecker, PermissionDecision, Tool, ToolContext, ToolError,
    ToolExecutionOutcome, ToolExecutor, ToolName, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::json;

// A minimal tool for demonstration.
struct EchoTool { spec: ToolSpec }

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> &ToolSpec { &self.spec }
    async fn invoke(
        &self, request: ToolRequest, _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Structured(json!({ "ok": true })),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: None,
            metadata: MetadataMap::new(),
        })
    }
}

// Permit everything — real agents should use CompositePermissionChecker.
struct AllowAll;
impl PermissionChecker for AllowAll {
    fn evaluate(
        &self, _request: &dyn agentkit_tools_core::PermissionRequest,
    ) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

# #[tokio::main]
# async fn main() {
let registry = ToolRegistry::new().with(EchoTool {
    spec: ToolSpec {
        name: ToolName::new("echo"),
        description: "Return a fixed payload".into(),
        input_schema: json!({ "type": "object" }),
        annotations: Default::default(),
        metadata: MetadataMap::new(),
    },
});

let executor = BasicToolExecutor::new(registry);
let metadata = MetadataMap::new();
let mut ctx = ToolContext {
    capability: CapabilityContext {
        session_id: Some(&SessionId::new("s1")),
        turn_id: Some(&TurnId::new("t1")),
        metadata: &metadata,
    },
    permissions: &AllowAll,
    resources: &(),
    cancellation: None,
};

let outcome = executor
    .execute(
        ToolRequest {
            call_id: "call-1".into(),
            tool_name: ToolName::new("echo"),
            input: json!({}),
            session_id: SessionId::new("s1"),
            turn_id: TurnId::new("t1"),
            metadata: MetadataMap::new(),
        },
        &mut ctx,
    )
    .await;

assert!(matches!(outcome, ToolExecutionOutcome::Completed(_)));
# }
```

## Setting up permissions

Use `CompositePermissionChecker` to layer multiple policies. Policies are
evaluated in order; the first `Deny` short-circuits. Any `RequireApproval`
is returned unless a later policy denies.

```rust
use agentkit_tools_core::{
    CommandPolicy, CompositePermissionChecker, PathPolicy,
    McpServerPolicy, PermissionDecision,
};

let permissions = CompositePermissionChecker::new(PermissionDecision::Allow)
    // Allow filesystem access under the workspace; protect .env files.
    .with_policy(
        PathPolicy::new()
            .allow_root("/workspace/project")
            .protect_root("/workspace/project/.env"),
    )
    // Allow git and cargo; require approval for anything else.
    .with_policy(
        CommandPolicy::new()
            .allow_executable("git")
            .allow_executable("cargo")
            .deny_executable("rm")
            .require_approval_for_unknown(true),
    )
    // Trust a specific MCP server.
    .with_policy(
        McpServerPolicy::new()
            .trust_server("github-mcp"),
    );
```
