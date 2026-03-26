# Tools

`agentkit-tools-core` is the tool execution contract crate.

## Built-in tools

### Filesystem (`agentkit-tool-fs`)

- `fs.read_file` — supports optional `from` / `to` line ranges
- `fs.write_file`
- `fs.replace_in_file`
- `fs.move`
- `fs.delete`
- `fs.list_directory`
- `fs.create_directory`

The filesystem crate supports session-scoped read-before-write enforcement through `FileSystemToolResources` and `FileSystemToolPolicy`.

### Shell (`agentkit-tool-shell`)

- `shell.exec`

## Tool trait

The base tool trait:

```rust
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;

    async fn invoke(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext,
    ) -> Result<ToolResult, ToolError>;
}
```

## Custom tools

A simple custom tool:

```rust
pub struct EchoTool;

impl Tool for EchoTool {
    fn spec(&self) -> &ToolSpec { /* ... */ }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext,
    ) -> Result<ToolResult, ToolError> {
        /* ... */
    }
}
```

## Tool registry

```rust
pub struct ToolRegistry {
    tools: BTreeMap<ToolName, Arc<dyn Tool>>,
}
```

Register tools and expose them to the model:

- Register one or many tools
- Look up by name
- Iterate specs
- Freeze for execution

## Tool execution path

1. `ToolRegistry` — lookup
2. `ToolExecutor` — orchestration
3. Permission preflight
4. Tool invocation
5. Normalization into `ToolResult`

## Preflight actions

Tools can optionally expose what they plan to do before execution:

```rust
pub trait PreflightTool: Tool {
    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError>;
}
```

This lets the executor inspect intended actions, run permission checks, and raise approval interrupts before side effects occur.
