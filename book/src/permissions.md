# Permissions

Permissions are a shared policy layer for anything the agent may try to do that has safety implications.

## Decision model

Permission checks produce one of three outcomes:

- **Allow** — proceed
- **Deny** — structured denial with a reason
- **Require approval** — pause and ask the host

## Permission checker

```rust
pub trait PermissionChecker: Send + Sync {
    fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision;
}
```

## Built-in request types

- `ShellPermissionRequest` — executable, argv, cwd, env keys, timeout hint
- `FileSystemPermissionRequest` — Read, Write, Edit, Delete, Move, List, CreateDir variants
- `McpPermissionRequest` — Connect, InvokeTool, ReadResource, FetchPrompt, UseAuthScope variants

## Policy composition

Hosts can layer multiple policies:

```rust
pub struct CompositePermissionChecker {
    policies: Vec<Box<dyn PermissionPolicy>>,
    fallback: PermissionDecision,
}
```

Precedence (v1):

1. Explicit deny wins
2. Require-approval wins over allow
3. Allow wins over no-opinion
4. Fallback applies if no policy matches

## Built-in policy helpers

- `PathPolicy` — allow, deny, or require approval for filesystem paths
- `CommandPolicy` — allow, deny, or require approval for shell actions
- `McpServerPolicy` — govern MCP server connection and usage
- `CustomKindPolicy` — handle custom tool action kinds

## Custom permission requests

Custom tools can define their own request types:

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
