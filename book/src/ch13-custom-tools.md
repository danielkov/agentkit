# Writing custom tools

Custom tools are the primary extension mechanism in agentkit. This chapter shows how to implement tools from simple to sophisticated, including preflight actions, custom permission types, and shared resources.

## A minimal tool

```rust
use agentkit_tools_core::*;
use agentkit_core::*;
use async_trait::async_trait;
use serde_json::json;

pub struct EchoTool {
    spec: ToolSpec,
}

impl EchoTool {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("echo"),
                description: "Return the input unchanged".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    },
                    "required": ["message"]
                }),
                annotations: ToolAnnotations {
                    read_only_hint: true,
                    ..Default::default()
                },
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let message = request.input["message"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("missing message".into()))?;

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Text(message.to_string()),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: None,
            metadata: MetadataMap::new(),
        })
    }
}
```

Register it:

```rust
let registry = ToolRegistry::new().with(EchoTool::new());
```

## Using ToolContext

Tools receive a `ToolContext` that provides access to the current session, permissions, cancellation state, and shared resources:

```rust
async fn invoke(&self, request: ToolRequest, ctx: &mut ToolContext<'_>) -> Result<ToolResult, ToolError> {
    // Check cancellation
    if let Some(ref cancel) = ctx.cancellation {
        if cancel.is_cancelled() {
            return Err(ToolError::Cancelled);
        }
    }

    // Access shared resources
    let resources = ctx.resources;

    // Access session identity
    let session_id = ctx.session_id;

    // ...
}
```

## Adding preflight actions

For tools with side effects, implement `PreflightTool` to expose proposed actions before execution:

```rust
impl PreflightTool for DeployTool {
    fn proposed_actions(&self, request: &ToolRequest)
        -> Result<Vec<Box<dyn PermissionRequest>>, ToolError>
    {
        let env = request.input["environment"].as_str().unwrap_or("unknown");
        Ok(vec![Box::new(DeployPermissionRequest {
            environment: env.to_string(),
            service: "my-service".into(),
            metadata: MetadataMap::new(),
        })])
    }
}
```

The executor evaluates these before calling `invoke()`. If any are denied or require approval, execution stops before any side effects occur.

## Custom permission requests

Define your own permission request types:

```rust
pub struct DeployPermissionRequest {
    pub environment: String,
    pub service: String,
    pub metadata: MetadataMap,
}

impl PermissionRequest for DeployPermissionRequest {
    fn kind(&self) -> &'static str { "myapp.deploy" }
    fn summary(&self) -> String {
        format!("Deploy {} to {}", self.service, self.environment)
    }
    fn metadata(&self) -> &MetadataMap { &self.metadata }
    fn as_any(&self) -> &dyn Any { self }
}
```

Host policies can match on `kind()` generically, or downcast through `as_any()` for type-safe field access.

## Shared resources via ToolResources

If your tool needs session-scoped state (like the filesystem tools' read-before-write tracker), implement `ToolResources`:

```rust
pub trait ToolResources: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}
```

Register resources when building the agent, and downcast in your tool's `invoke()` method.

## Tool composition patterns

### Nesting agents as tools

A powerful pattern: implement a tool that runs a nested agent loop. The outer agent calls the tool with a task description, the tool starts an inner agent, runs it to completion, and returns the result.

```text
Outer agent (orchestrator):
  Model: "I need to research this codebase and write a report"
  Model: ToolCall(subagent, { task: "Find all uses of unsafe code", tools: ["fs", "shell"] })
         │
         ▼
  Inner agent (researcher):
    Model: ToolCall(fs.read_file, { path: "src/lib.rs" })
    Model: ToolCall(shell.exec, { executable: "grep", args: ["-r", "unsafe", "src/"] })
    Model: "Found 3 uses of unsafe in parser.rs, codec.rs, and ffi.rs..."
         │
         ▼
  Outer agent receives: "Found 3 uses of unsafe..."
  Model: "Based on my research, here's the report..."
```

The inner agent has its own transcript, tools, and session. It doesn't share state with the outer agent — this isolation prevents context pollution and makes the sub-agent's scope explicit.

The [`openrouter-subagent-tool`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-subagent-tool) example shows a complete implementation of this pattern.

### Tool registries from crates

Organize related tools into crate-level `registry()` functions:

```rust
pub fn registry() -> ToolRegistry {
    ToolRegistry::new()
        .with(ToolA::default())
        .with(ToolB::default())
}
```

Host applications merge registries from multiple crates:

```rust
let registry = my_tools::registry()
    .merge(agentkit_tool_fs::registry())
    .merge(agentkit_tool_shell::registry());
```

### Stateful tools

Tools that need to maintain state across invocations (counters, caches, connection pools) should use `ToolResources`:

```rust
struct MyToolResources {
    cache: Mutex<HashMap<String, String>>,
    http_client: reqwest::Client,
}

impl ToolResources for MyToolResources {
    fn as_any(&self) -> &dyn Any { self }
}

// In your tool's invoke():
let resources = ctx.resources
    .as_any()
    .downcast_ref::<MyToolResources>()
    .expect("MyToolResources not registered");

let mut cache = resources.cache.lock().unwrap();
```

Register resources when building the agent:

```rust
let agent = Agent::builder()
    .model(adapter)
    .resources(MyToolResources::new())
    .build()?;
```

All tools in the session share the same `ToolResources` instance. This is how the filesystem tools share their read-before-write tracker — `FileSystemToolResources` implements `ToolResources` and is downcast in each tool's `invoke()`.

> **Example:** [`openrouter-subagent-tool`](https://github.com/danielkov/agentkit/tree/main/examples/openrouter-subagent-tool) implements a custom tool that runs a nested agent as a tool call.
>
> **Crate:** [`agentkit-tools-core`](https://github.com/danielkov/agentkit/tree/main/crates/agentkit-tools-core) — `Tool`, `ToolRegistry`, `ToolResources`, `ToolContext`.
