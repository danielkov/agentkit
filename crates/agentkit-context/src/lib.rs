use std::path::{Path, PathBuf};

use agentkit_core::{Item, ItemKind, MetadataMap, Part, TextPart};
use async_trait::async_trait;
use futures_lite::StreamExt;
use serde_json::Value;
use thiserror::Error;

const DEFAULT_AGENTS_FILE: &str = "AGENTS.md";
const DEFAULT_SKILL_FILE: &str = "SKILL.md";

#[async_trait]
pub trait ContextSource: Send + Sync {
    async fn load(&self) -> Result<Vec<Item>, ContextError>;
}

#[derive(Default)]
pub struct ContextLoader {
    sources: Vec<Box<dyn ContextSource>>,
}

impl ContextLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_source(mut self, source: impl ContextSource + 'static) -> Self {
        self.sources.push(Box::new(source));
        self
    }

    pub async fn load(&self) -> Result<Vec<Item>, ContextError> {
        let mut items = Vec::new();

        for source in &self.sources {
            items.extend(source.load().await?);
        }

        Ok(items)
    }
}

#[derive(Clone, Debug)]
pub struct AgentsMd {
    start_dir: PathBuf,
    file_name: String,
}

impl AgentsMd {
    pub fn discover(start_dir: impl Into<PathBuf>) -> Self {
        Self {
            start_dir: start_dir.into(),
            file_name: DEFAULT_AGENTS_FILE.into(),
        }
    }

    pub fn with_file_name(mut self, file_name: impl Into<String>) -> Self {
        self.file_name = file_name.into();
        self
    }

    pub async fn resolve(&self) -> Result<Option<PathBuf>, ContextError> {
        find_in_ancestors(&self.start_dir, &self.file_name).await
    }
}

#[async_trait]
impl ContextSource for AgentsMd {
    async fn load(&self) -> Result<Vec<Item>, ContextError> {
        let Some(path) = self.resolve().await? else {
            return Ok(Vec::new());
        };
        let body =
            async_fs::read_to_string(&path)
                .await
                .map_err(|error| ContextError::ReadFailed {
                    path: path.clone(),
                    error,
                })?;

        Ok(vec![context_item(
            format!(
                "[Loaded AGENTS]\nPath: {}\n\n{}",
                path.display(),
                body.trim_end()
            ),
            metadata_for("agents_md", &path, None),
        )])
    }
}

#[derive(Clone, Debug)]
pub struct SkillsDirectory {
    root: PathBuf,
    skill_file_name: String,
}

impl SkillsDirectory {
    pub fn from_dir(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            skill_file_name: DEFAULT_SKILL_FILE.into(),
        }
    }

    pub fn with_skill_file_name(mut self, skill_file_name: impl Into<String>) -> Self {
        self.skill_file_name = skill_file_name.into();
        self
    }
}

#[async_trait]
impl ContextSource for SkillsDirectory {
    async fn load(&self) -> Result<Vec<Item>, ContextError> {
        if !path_exists(&self.root).await? {
            return Ok(Vec::new());
        }

        let skill_paths = collect_skill_files(&self.root, &self.skill_file_name).await?;
        let mut items = Vec::with_capacity(skill_paths.len());

        for path in skill_paths {
            let body = async_fs::read_to_string(&path).await.map_err(|error| {
                ContextError::ReadFailed {
                    path: path.clone(),
                    error,
                }
            })?;
            let skill_name = path
                .parent()
                .and_then(Path::file_name)
                .map(|value| value.to_string_lossy().into_owned());

            items.push(context_item(
                format!(
                    "[Loaded Skill]\nName: {}\nPath: {}\n\n{}",
                    skill_name.clone().unwrap_or_else(|| "unknown".into()),
                    path.display(),
                    body.trim_end()
                ),
                metadata_for("skill", &path, skill_name),
            ));
        }

        Ok(items)
    }
}

fn context_item(text: String, metadata: MetadataMap) -> Item {
    Item {
        id: None,
        kind: ItemKind::Context,
        parts: vec![Part::Text(TextPart {
            text,
            metadata: MetadataMap::new(),
        })],
        metadata,
    }
}

fn metadata_for(source_kind: &str, path: &Path, name: Option<String>) -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert(
        "agentkit.context.source".into(),
        Value::String(source_kind.into()),
    );
    metadata.insert(
        "agentkit.context.path".into(),
        Value::String(path.display().to_string()),
    );
    if let Some(name) = name {
        metadata.insert("agentkit.context.name".into(), Value::String(name));
    }
    metadata
}

async fn path_exists(path: &Path) -> Result<bool, ContextError> {
    match async_fs::metadata(path).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(ContextError::InspectFailed {
            path: path.to_path_buf(),
            error,
        }),
    }
}

async fn find_in_ancestors(
    start_dir: &Path,
    file_name: &str,
) -> Result<Option<PathBuf>, ContextError> {
    let mut current = start_dir.to_path_buf();

    loop {
        let candidate = current.join(file_name);
        if path_exists(&candidate).await? {
            return Ok(Some(candidate));
        }
        let Some(parent) = current.parent() else {
            return Ok(None);
        };
        current = parent.to_path_buf();
    }
}

async fn collect_skill_files(
    root: &Path,
    skill_file_name: &str,
) -> Result<Vec<PathBuf>, ContextError> {
    let mut pending = vec![root.to_path_buf()];
    let mut skill_paths = Vec::new();

    while let Some(dir_path) = pending.pop() {
        let mut read_dir =
            async_fs::read_dir(&dir_path)
                .await
                .map_err(|error| ContextError::InspectFailed {
                    path: dir_path.clone(),
                    error,
                })?;

        while let Some(entry) = read_dir.next().await {
            let entry = entry.map_err(|error| ContextError::InspectFailed {
                path: dir_path.clone(),
                error,
            })?;
            let path = entry.path();
            let file_type =
                entry
                    .file_type()
                    .await
                    .map_err(|error| ContextError::InspectFailed {
                        path: path.clone(),
                        error,
                    })?;

            if file_type.is_dir() {
                pending.push(path);
                continue;
            }

            if file_type.is_file() && path.file_name().is_some_and(|name| name == skill_file_name) {
                skill_paths.push(path);
            }
        }
    }

    skill_paths.sort();
    Ok(skill_paths)
}

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("failed to inspect {path}: {error}")]
    InspectFailed {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("failed to read {path}: {error}")]
    ReadFailed {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[tokio::test]
    async fn discovers_agents_file_in_ancestors() {
        let root = temp_path("agentkit-context-agents");
        let nested = root.join("nested/project");
        async_fs::create_dir_all(&nested).await.unwrap();
        let agents_path = root.join("AGENTS.md");
        async_fs::write(&agents_path, "project = lantern")
            .await
            .unwrap();

        let items = AgentsMd::discover(&nested).load().await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, ItemKind::Context);
        assert_eq!(
            items[0].metadata.get("agentkit.context.source"),
            Some(&Value::String("agents_md".into()))
        );

        async_fs::remove_dir_all(&root).await.unwrap();
    }

    #[tokio::test]
    async fn loads_skills_recursively() {
        let root = temp_path("agentkit-context-skills");
        let skill_dir = root.join("skills/release-notes");
        async_fs::create_dir_all(&skill_dir).await.unwrap();
        async_fs::write(skill_dir.join("SKILL.md"), "# Release Notes")
            .await
            .unwrap();

        let items = SkillsDirectory::from_dir(root.join("skills"))
            .load()
            .await
            .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(
            items[0].metadata.get("agentkit.context.name"),
            Some(&Value::String("release-notes".into()))
        );

        async_fs::remove_dir_all(&root).await.unwrap();
    }

    fn temp_path(prefix: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{suffix}"))
    }
}
