use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use agentkit_core::{MetadataMap, ToolOutput, ToolResultPart};
use agentkit_tools_core::{
    FileSystemPermissionRequest, PermissionRequest, Tool, ToolAnnotations, ToolContext, ToolError,
    ToolName, ToolRegistry, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

pub fn registry() -> ToolRegistry {
    ToolRegistry::new()
        .with(ReadFileTool::default())
        .with(WriteFileTool::default())
        .with(ListDirectoryTool::default())
        .with(CreateDirectoryTool::default())
}

#[derive(Debug, Error)]
pub enum FileSystemToolError {
    #[error("path {0} is not valid UTF-8")]
    InvalidUtf8Path(PathBuf),
}

#[derive(Clone, Debug)]
pub struct ReadFileTool {
    spec: ToolSpec,
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("fs.read_file"),
                description: "Read a UTF-8 text file from disk.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                annotations: ToolAnnotations {
                    read_only_hint: true,
                    idempotent_hint: true,
                    ..ToolAnnotations::default()
                },
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[derive(Deserialize)]
struct ReadFileInput {
    path: PathBuf,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let input: ReadFileInput = parse_input(&request.input)?;
        Ok(vec![Box::new(FileSystemPermissionRequest::Read {
            path: input.path,
            metadata: request.metadata.clone(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let started = Instant::now();
        let input: ReadFileInput = parse_input(&request.input)?;
        let contents = fs::read_to_string(&input.path)
            .map_err(|error| ToolError::ExecutionFailed(format!("failed to read file: {error}")))?;

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Text(contents),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: Some(started.elapsed()),
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct WriteFileTool {
    spec: ToolSpec,
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("fs.write_file"),
                description: "Write UTF-8 text to a file, creating parent directories if needed."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "contents": { "type": "string" },
                        "create_parents": { "type": "boolean", "default": true }
                    },
                    "required": ["path", "contents"],
                    "additionalProperties": false
                }),
                annotations: ToolAnnotations {
                    destructive_hint: true,
                    idempotent_hint: false,
                    ..ToolAnnotations::default()
                },
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[derive(Deserialize)]
struct WriteFileInput {
    path: PathBuf,
    contents: String,
    #[serde(default = "default_true")]
    create_parents: bool,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let input: WriteFileInput = parse_input(&request.input)?;
        Ok(vec![Box::new(FileSystemPermissionRequest::Write {
            path: input.path,
            metadata: request.metadata.clone(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let started = Instant::now();
        let input: WriteFileInput = parse_input(&request.input)?;

        if input.create_parents
            && let Some(parent) = input.path.parent()
        {
            fs::create_dir_all(parent).map_err(|error| {
                ToolError::ExecutionFailed(format!(
                    "failed to create parent directories for {}: {error}",
                    input.path.display()
                ))
            })?;
        }

        fs::write(&input.path, input.contents.as_bytes()).map_err(|error| {
            ToolError::ExecutionFailed(format!("failed to write file: {error}"))
        })?;

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Structured(json!({
                    "path": input.path.display().to_string(),
                    "bytes_written": input.contents.len(),
                })),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: Some(started.elapsed()),
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ListDirectoryTool {
    spec: ToolSpec,
}

impl Default for ListDirectoryTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("fs.list_directory"),
                description: "List the entries in a directory.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                annotations: ToolAnnotations {
                    read_only_hint: true,
                    idempotent_hint: true,
                    ..ToolAnnotations::default()
                },
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[derive(Deserialize)]
struct ListDirectoryInput {
    path: PathBuf,
}

#[async_trait]
impl Tool for ListDirectoryTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let input: ListDirectoryInput = parse_input(&request.input)?;
        Ok(vec![Box::new(FileSystemPermissionRequest::List {
            path: input.path,
            metadata: request.metadata.clone(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let started = Instant::now();
        let input: ListDirectoryInput = parse_input(&request.input)?;
        let mut entries = Vec::new();

        for entry in fs::read_dir(&input.path).map_err(|error| {
            ToolError::ExecutionFailed(format!(
                "failed to list directory {}: {error}",
                input.path.display()
            ))
        })? {
            let entry = entry.map_err(|error| {
                ToolError::ExecutionFailed(format!(
                    "failed to read directory entry in {}: {error}",
                    input.path.display()
                ))
            })?;
            let file_type = entry.file_type().map_err(|error| {
                ToolError::ExecutionFailed(format!(
                    "failed to inspect directory entry in {}: {error}",
                    input.path.display()
                ))
            })?;
            entries.push(json!({
                "name": path_name(entry.path())?,
                "path": entry.path().display().to_string(),
                "kind": file_kind_label(&file_type),
            }));
        }

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Structured(Value::Array(entries)),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: Some(started.elapsed()),
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct CreateDirectoryTool {
    spec: ToolSpec,
}

impl Default for CreateDirectoryTool {
    fn default() -> Self {
        Self {
            spec: ToolSpec {
                name: ToolName::new("fs.create_directory"),
                description: "Create a directory and any missing parent directories.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
                annotations: ToolAnnotations {
                    destructive_hint: true,
                    idempotent_hint: true,
                    ..ToolAnnotations::default()
                },
                metadata: MetadataMap::new(),
            },
        }
    }
}

#[derive(Deserialize)]
struct CreateDirectoryInput {
    path: PathBuf,
}

#[async_trait]
impl Tool for CreateDirectoryTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn proposed_requests(
        &self,
        request: &ToolRequest,
    ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
        let input: CreateDirectoryInput = parse_input(&request.input)?;
        Ok(vec![Box::new(FileSystemPermissionRequest::CreateDir {
            path: input.path,
            metadata: request.metadata.clone(),
        })])
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let started = Instant::now();
        let input: CreateDirectoryInput = parse_input(&request.input)?;
        fs::create_dir_all(&input.path).map_err(|error| {
            ToolError::ExecutionFailed(format!(
                "failed to create directory {}: {error}",
                input.path.display()
            ))
        })?;

        Ok(ToolResult {
            result: ToolResultPart {
                call_id: request.call_id,
                output: ToolOutput::Structured(json!({
                    "path": input.path.display().to_string(),
                    "created": true,
                })),
                is_error: false,
                metadata: MetadataMap::new(),
            },
            duration: Some(started.elapsed()),
            metadata: MetadataMap::new(),
        })
    }
}

fn parse_input<T>(value: &Value) -> Result<T, ToolError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(value.clone())
        .map_err(|error| ToolError::InvalidInput(format!("invalid tool input: {error}")))
}

fn default_true() -> bool {
    true
}

fn path_name(path: impl AsRef<Path>) -> Result<String, ToolError> {
    let path = path.as_ref();
    let name = path.file_name().ok_or_else(|| {
        ToolError::ExecutionFailed(format!("path {} has no file name", path.display()))
    })?;

    name.to_str().map(|value| value.to_string()).ok_or_else(|| {
        ToolError::ExecutionFailed(
            FileSystemToolError::InvalidUtf8Path(path.to_path_buf()).to_string(),
        )
    })
}

fn file_kind_label(file_type: &fs::FileType) -> &'static str {
    if file_type.is_dir() {
        "directory"
    } else if file_type.is_symlink() {
        "symlink"
    } else {
        "file"
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use agentkit_capabilities::CapabilityContext;
    use agentkit_core::{SessionId, ToolCallId, TurnId};
    use agentkit_tools_core::{PermissionChecker, PermissionDecision, ToolExecutor};

    use super::*;

    struct AllowAll;

    impl PermissionChecker for AllowAll {
        fn evaluate(&self, _request: &dyn PermissionRequest) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("agentkit-{name}-{nanos}"))
    }

    #[test]
    fn registry_exposes_expected_tools() {
        let specs = registry().specs();
        let names: Vec<_> = specs.into_iter().map(|spec| spec.name.0).collect();
        assert!(names.contains(&"fs.read_file".into()));
        assert!(names.contains(&"fs.write_file".into()));
        assert!(names.contains(&"fs.list_directory".into()));
        assert!(names.contains(&"fs.create_directory".into()));
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let root = temp_dir("fs");
        fs::create_dir_all(&root).unwrap();
        let target = root.join("note.txt");
        let session_id = SessionId::new("session-1");
        let turn_id = TurnId::new("turn-1");

        let executor = agentkit_tools_core::BasicToolExecutor::new(registry());
        let metadata = MetadataMap::new();
        let mut ctx = ToolContext {
            capability: CapabilityContext {
                session_id: Some(&session_id),
                turn_id: Some(&turn_id),
                metadata: &metadata,
            },
            permissions: &AllowAll,
            resources: &(),
        };

        let write = executor
            .execute(
                ToolRequest {
                    call_id: ToolCallId::new("call-write"),
                    tool_name: ToolName::new("fs.write_file"),
                    input: json!({
                        "path": target.display().to_string(),
                        "contents": "hello"
                    }),
                    session_id: session_id.clone(),
                    turn_id: turn_id.clone(),
                    metadata: MetadataMap::new(),
                },
                &mut ctx,
            )
            .await;
        assert!(matches!(
            write,
            agentkit_tools_core::ToolExecutionOutcome::Completed(_)
        ));

        let read = executor
            .execute(
                ToolRequest {
                    call_id: ToolCallId::new("call-read"),
                    tool_name: ToolName::new("fs.read_file"),
                    input: json!({
                        "path": target.display().to_string()
                    }),
                    session_id: session_id.clone(),
                    turn_id: turn_id.clone(),
                    metadata: MetadataMap::new(),
                },
                &mut ctx,
            )
            .await;

        match read {
            agentkit_tools_core::ToolExecutionOutcome::Completed(result) => {
                assert_eq!(result.result.output, ToolOutput::Text("hello".into()));
            }
            other => panic!("unexpected outcome: {other:?}"),
        }

        let _ = fs::remove_dir_all(root);
    }
}
