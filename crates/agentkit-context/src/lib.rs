use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use agentkit_core::{Item, ItemKind, MetadataMap, Part, TextPart};
use async_trait::async_trait;
use futures_lite::StreamExt;
use serde_json::Value;
use thiserror::Error;

const DEFAULT_AGENTS_FILE: &str = "AGENTS.md";
const DEFAULT_SKILL_FILE: &str = "SKILL.md";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentsMdMode {
    Nearest,
    All,
}

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
    mode: AgentsMdMode,
    file_name: String,
    explicit_paths: Vec<PathBuf>,
    search_dirs: Vec<PathBuf>,
}

impl AgentsMd {
    pub fn discover(start_dir: impl Into<PathBuf>) -> Self {
        Self {
            start_dir: start_dir.into(),
            mode: AgentsMdMode::Nearest,
            file_name: DEFAULT_AGENTS_FILE.into(),
            explicit_paths: Vec::new(),
            search_dirs: Vec::new(),
        }
    }

    pub fn discover_all(start_dir: impl Into<PathBuf>) -> Self {
        Self::discover(start_dir).with_mode(AgentsMdMode::All)
    }

    pub fn with_mode(mut self, mode: AgentsMdMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_file_name(mut self, file_name: impl Into<String>) -> Self {
        self.file_name = file_name.into();
        self
    }

    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.explicit_paths.push(path.into());
        self
    }

    pub fn with_search_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.search_dirs.push(dir.into());
        self
    }

    pub async fn resolve(&self) -> Result<Option<PathBuf>, ContextError> {
        Ok(self.resolve_all().await?.into_iter().next())
    }

    pub async fn resolve_all(&self) -> Result<Vec<PathBuf>, ContextError> {
        let mut paths = Vec::new();

        for path in &self.explicit_paths {
            if path_exists(path).await? {
                paths.push(path.clone());
            }
        }

        for dir in &self.search_dirs {
            let candidate = dir.join(&self.file_name);
            if path_exists(&candidate).await? {
                paths.push(candidate);
            }
        }

        paths.extend(
            find_in_ancestors_with_mode(
                &self.start_dir,
                &self.file_name,
                self.mode == AgentsMdMode::All,
            )
            .await?,
        );

        let mut seen = BTreeSet::new();
        paths.retain(|path| seen.insert(path.clone()));
        if self.mode == AgentsMdMode::Nearest {
            Ok(paths.into_iter().rev().take(1).collect())
        } else {
            Ok(paths)
        }
    }
}

#[async_trait]
impl ContextSource for AgentsMd {
    async fn load(&self) -> Result<Vec<Item>, ContextError> {
        let paths = self.resolve_all().await?;
        let mut items = Vec::with_capacity(paths.len());

        for path in paths {
            let body = async_fs::read_to_string(&path).await.map_err(|error| {
                ContextError::ReadFailed {
                    path: path.clone(),
                    error,
                }
            })?;

            items.push(context_item(
                format!(
                    "[Loaded AGENTS]\nPath: {}\n\n{}",
                    path.display(),
                    body.trim_end()
                ),
                metadata_for("agents_md", &path, None),
            ));
        }

        Ok(items)
    }
}

#[derive(Clone, Debug)]
pub struct SkillsDirectory {
    roots: Vec<PathBuf>,
    skill_file_name: String,
}

impl SkillsDirectory {
    pub fn from_dir(root: impl Into<PathBuf>) -> Self {
        Self {
            roots: vec![root.into()],
            skill_file_name: DEFAULT_SKILL_FILE.into(),
        }
    }

    pub fn with_dir(mut self, root: impl Into<PathBuf>) -> Self {
        self.roots.push(root.into());
        self
    }

    pub fn with_skill_file_name(mut self, skill_file_name: impl Into<String>) -> Self {
        self.skill_file_name = skill_file_name.into();
        self
    }
}

#[async_trait]
impl ContextSource for SkillsDirectory {
    async fn load(&self) -> Result<Vec<Item>, ContextError> {
        let mut skill_paths = Vec::new();
        for root in &self.roots {
            if !path_exists(root).await? {
                continue;
            }
            skill_paths.extend(collect_skill_files(root, &self.skill_file_name).await?);
        }
        skill_paths.sort();
        skill_paths.dedup();

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

async fn find_in_ancestors_with_mode(
    start_dir: &Path,
    file_name: &str,
    include_all: bool,
) -> Result<Vec<PathBuf>, ContextError> {
    let mut current = start_dir.to_path_buf();
    let mut matches = Vec::new();

    loop {
        let candidate = current.join(file_name);
        if path_exists(&candidate).await? {
            matches.push(candidate);
            if !include_all {
                break;
            }
        }
        let Some(parent) = current.parent() else {
            break;
        };
        current = parent.to_path_buf();
    }

    matches.reverse();
    Ok(matches)
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
    async fn discovers_all_agents_files_when_requested() {
        let root = temp_path("agentkit-context-agents-all");
        let nested = root.join("nested/project");
        async_fs::create_dir_all(&nested).await.unwrap();
        async_fs::write(root.join("AGENTS.md"), "project = lantern")
            .await
            .unwrap();
        async_fs::write(root.join("nested/AGENTS.md"), "team = orbit")
            .await
            .unwrap();

        let items = AgentsMd::discover_all(&nested).load().await.unwrap();
        assert_eq!(items.len(), 2);

        async_fs::remove_dir_all(&root).await.unwrap();
    }

    #[tokio::test]
    async fn loads_agents_from_explicit_search_paths() {
        let root = temp_path("agentkit-context-agents-explicit");
        let nested = root.join("nested/project");
        let shared = root.join("shared");
        async_fs::create_dir_all(&nested).await.unwrap();
        async_fs::create_dir_all(&shared).await.unwrap();
        async_fs::write(shared.join("AGENTS.md"), "policy = explicit")
            .await
            .unwrap();

        let items = AgentsMd::discover(&nested)
            .with_search_dir(&shared)
            .load()
            .await
            .unwrap();
        assert_eq!(items.len(), 1);
        assert!(
            items[0]
                .metadata
                .get("agentkit.context.path")
                .and_then(Value::as_str)
                .is_some_and(|path| path.ends_with("/shared/AGENTS.md"))
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

    #[tokio::test]
    async fn loads_skills_from_multiple_roots() {
        let root = temp_path("agentkit-context-skills-multi");
        let root_a = root.join("skills-a/release-notes");
        let root_b = root.join("skills-b/deploy");
        async_fs::create_dir_all(&root_a).await.unwrap();
        async_fs::create_dir_all(&root_b).await.unwrap();
        async_fs::write(root_a.join("SKILL.md"), "# Release Notes")
            .await
            .unwrap();
        async_fs::write(root_b.join("SKILL.md"), "# Deploy")
            .await
            .unwrap();

        let items = SkillsDirectory::from_dir(root.join("skills-a"))
            .with_dir(root.join("skills-b"))
            .load()
            .await
            .unwrap();
        assert_eq!(items.len(), 2);

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
