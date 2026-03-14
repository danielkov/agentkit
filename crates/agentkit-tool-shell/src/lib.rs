use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use agentkit_core::{MetadataMap, ToolOutput, ToolResultPart};
use agentkit_tools_core::{
    PermissionRequest, ShellPermissionRequest, Tool, ToolAnnotations, ToolContext, ToolError,
    ToolName, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;
use tokio::time::timeout;

pub fn registry() -> ToolRegistry {
    ToolRegistry::new().with(ShellExecTool::default())
}

#[derive(Clone, Debug)]
pub struct ShellExecTool {
    spec: ToolSpec,
}

impl Default for ShellExecTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("shell.exec"),
                description: "Execute a shell command and capture stdout, stderr, and exit status."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "executable": { "type": "string" },
                        "argv": {
                            "type": "array",
                            "items": { "type": "string" },
                            "default": []
                        },
                        "cwd": { "type": "string" },
                        "env": {
                            "type": "object",
                            "additionalProperties": { "type": "string" }
                        },
                        "timeout_ms": { "type": "integer", "minimum": 1 }
                    },
                    "required": ["executable"],
                    "additionalProperties": false
                }),
                annotations: ToolAnnotations {
                    destructive_hint: true,
                    needs_approval_hint: true,
                    ..ToolAnnotations::default()
                },
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct ShellExecInput {
    executable: String,
    #[serde(default)]
    argv: Vec<String>,
    cwd: Option<PathBuf>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    timeout_ms: Option<u64>,
}

#[async_trait]
impl Tool for ShellExecTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let input: ShellExecInput = parse_input(request)?;
        Ok(vec![Box::new(ShellPermissionRequest {
            executable: input.executable,
            argv: input.argv,
            cwd: input.cwd,
            env_keys: input.env.keys().cloned().collect(),
            metadata: request.metadata.clone(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let input: ShellExecInput = parse_input(&request)?;
        let mut command = Command::new(&input.executable);
        command.args(&input.argv);
        if let Some(cwd) = &input.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &input.env {
            command.env(key, value);
        }

        let duration_start = std::time::Instant::now();
        let output = if let Some(timeout_ms) = input.timeout_ms {
            timeout(Duration::from_millis(timeout_ms), command.output())
                .await
                .map_err(|_| {
                    ToolError::ExecutionFailed(format!("command timed out after {timeout_ms}ms"))
                })?
                .map_err(|error| {
                    ToolError::ExecutionFailed(format!("failed to spawn command: {error}"))
                })?
        } else {
            command.output().await.map_err(|error| {
                ToolError::ExecutionFailed(format!("failed to spawn command: {error}"))
            })?
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let status = output.status.code();
        let success = output.status.success();

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Structured(json!({
                    "stdout": stdout,
                    "stderr": stderr,
                    "success": success,
                    "exit_code": status,
                })),
                is_error: !success,
                metadata: MetadataMap::new(),
            },
            duration: Some(duration_start.elapsed()),
            metadata: MetadataMap::new(),
        })
    }
}

fn parse_input(request: &ToolRequest) -> Result<ShellExecInput, ToolError> {
    serde_json::from_value(request.input.clone())
        .map_err(|error| ToolError::InvalidInput(format!("invalid tool input: {error}")))
}

#[cfg(test)]
mod tests {
    use agentkit_capabilities::CapabilityContext;
    use agentkit_core::{SessionId, TurnId};
    use agentkit_tools_core::{
        BasicToolExecutor, PermissionChecker, PermissionCode, PermissionDecision, PermissionDenial,
        ToolExecutionOutcome, ToolExecutor,
    };

    use super::*;

    struct AllowAll;

    impl PermissionChecker for AllowAll {
        fn evaluate(
            &self,
            _request: &dyn agentkit_tools_core::PermissionRequest,
        ) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    struct DenyCommands;

    impl PermissionChecker for DenyCommands {
        fn evaluate(
            &self,
            _request: &dyn agentkit_tools_core::PermissionRequest,
        ) -> PermissionDecision {
            PermissionDecision::Deny(PermissionDenial {
                code: PermissionCode::CommandNotAllowed,
                message: "commands denied in test".into(),
                metadata: MetadataMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn shell_tool_executes_and_captures_output() {
        let executor = BasicToolExecutor::new(registry());
        let metadata = MetadataMap::new();
        let mut ctx = ToolContext {
            capability: CapabilityContext {
                session_id: Some(&SessionId::new("session-1")),
                turn_id: Some(&TurnId::new("turn-1")),
                metadata: &metadata,
            },
            permissions: &AllowAll,
            resources: &(),
        };

        let result = executor
            .execute(
                ToolRequest {
                    call_id: "call-1".into(),
                    tool_name: ToolName::new("shell.exec"),
                    input: json!({
                        "executable": "sh",
                        "argv": ["-c", "printf hello"]
                    }),
                    session_id: "session-1".into(),
                    turn_id: "turn-1".into(),
                    metadata: MetadataMap::new(),
                },
                &mut ctx,
            )
            .await;

        match result {
            ToolExecutionOutcome::Completed(result) => {
                let value = match result.result.output {
                    ToolOutput::Structured(value) => value,
                    other => panic!("unexpected output: {other:?}"),
                };
                assert_eq!(value["stdout"], "hello");
                assert_eq!(value["success"], true);
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[tokio::test]
    async fn shell_tool_respects_permission_denial() {
        let executor = BasicToolExecutor::new(registry());
        let metadata = MetadataMap::new();
        let mut ctx = ToolContext {
            capability: CapabilityContext {
                session_id: Some(&SessionId::new("session-1")),
                turn_id: Some(&TurnId::new("turn-1")),
                metadata: &metadata,
            },
            permissions: &DenyCommands,
            resources: &(),
        };

        let result = executor
            .execute(
                ToolRequest {
                    call_id: "call-2".into(),
                    tool_name: ToolName::new("shell.exec"),
                    input: json!({
                        "executable": "sh",
                        "argv": ["-c", "printf nope"]
                    }),
                    session_id: "session-1".into(),
                    turn_id: "turn-1".into(),
                    metadata: MetadataMap::new(),
                },
                &mut ctx,
            )
            .await;

        assert!(matches!(
            result,
            ToolExecutionOutcome::Failed(ToolError::PermissionDenied(_))
        ));
    }
}
