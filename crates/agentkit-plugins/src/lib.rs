//! Parse and validate portable [Agent Plugins](https://agent-plugins.org/) packages.
//!
//! This crate is deliberately information-first: it discovers standard assets
//! and exposes their metadata, paths, and portable declarations. Runtime
//! concerns such as skill activation, MCP process launch, `PLUGIN_DATA`, and
//! installation remain with the consuming application and satellite crates.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use http::{HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;
use url::{Host, Url};

/// Canonical Agent Plugins 1.0 manifest schema identifier.
pub const PLUGIN_SCHEMA_1_0_0: &str = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";
/// Canonical Agent Plugins 1.0 MCP schema identifier.
pub const MCP_SCHEMA_1_0_0: &str = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json";

const MANIFEST_FIELDS: &[&str] = &[
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "extensions",
];

// ---------------------------------------------------------------------------
// Public package model
// ---------------------------------------------------------------------------

/// A validated Agent Plugins package and its discovered portable assets.
#[derive(Clone, Debug)]
pub struct AgentPlugin {
    root: PathBuf,
    manifest: PluginManifest,
    skills: Vec<PluginSkill>,
    mcp_servers: Vec<PluginMcpServer>,
    diagnostics: Vec<PluginDiagnostic>,
}

impl AgentPlugin {
    /// Load an Agent Plugins package from a directory.
    ///
    /// Manifest failures reject the package. Invalid component locations,
    /// skills, MCP documents, and individual MCP entries are isolated and
    /// reported through [`Self::diagnostics`] as required by the standard.
    pub fn load(root: impl Into<PathBuf>) -> Result<Self, PluginError> {
        let requested_root = root.into();
        let root = fs::canonicalize(&requested_root).map_err(|error| PluginError::RootResolve {
            path: requested_root,
            error,
        })?;
        if !root.is_dir() {
            return Err(PluginError::InvalidRoot { path: root });
        }

        let manifest_path = root.join("plugin.json");
        let resolved_manifest =
            canonical_file(&manifest_path).map_err(|error| PluginError::ManifestRead {
                path: manifest_path.clone(),
                error,
            })?;
        if !is_contained(&root, &resolved_manifest) {
            return Err(PluginError::ManifestOutsideRoot {
                path: resolved_manifest,
            });
        }

        let manifest_content =
            fs::read_to_string(&resolved_manifest).map_err(|error| PluginError::ManifestRead {
                path: manifest_path.clone(),
                error,
            })?;
        let manifest_value: Value = serde_json::from_str(&manifest_content).map_err(|error| {
            PluginError::ManifestParse {
                path: manifest_path,
                error,
            }
        })?;

        let mut diagnostics = Vec::new();
        let manifest = parse_manifest(manifest_value, &resolved_manifest, &mut diagnostics)?;
        let skills = discover_skills(&root, &mut diagnostics);
        let mcp_servers = discover_mcp(&root, &mut diagnostics);

        Ok(Self {
            root,
            manifest,
            skills,
            mcp_servers,
            diagnostics,
        })
    }

    /// Filesystem-resolved plugin root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Validated portable manifest.
    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    /// Valid Agent Skills discovered as immediate children of `skills/`.
    pub fn skills(&self) -> &[PluginSkill] {
        &self.skills
    }

    /// Exact directories containing the discovered skills.
    ///
    /// Pass these to `agentkit_tool_skills::SkillRegistry::from_skill_dirs`
    /// to compose the package with agentkit's existing skill runtime.
    pub fn skill_directories(&self) -> Vec<PathBuf> {
        self.skills
            .iter()
            .map(|skill| skill.directory.clone())
            .collect()
    }

    /// Valid portable MCP server declarations.
    pub fn mcp_servers(&self) -> &[PluginMcpServer] {
        &self.mcp_servers
    }

    /// Non-fatal validation and discovery diagnostics.
    pub fn diagnostics(&self) -> &[PluginDiagnostic] {
        &self.diagnostics
    }

    /// Opaque manifest data for a client extension namespace.
    ///
    /// The core loader intentionally does not validate namespace-owned data.
    pub fn extension_manifest(&self, namespace: &str) -> Option<&Value> {
        self.manifest.extensions.get(namespace)
    }

    /// Resolve an existing extension directory while enforcing containment.
    pub fn extension_dir(&self, namespace: &str) -> Option<PathBuf> {
        if namespace.is_empty()
            || namespace == "."
            || namespace.contains(['/', '\\'])
            || Path::new(namespace).components().count() != 1
            || matches!(
                Path::new(namespace).components().next(),
                Some(Component::ParentDir | Component::RootDir | Component::Prefix(_))
            )
        {
            return None;
        }
        let path = fs::canonicalize(self.root.join(namespace)).ok()?;
        (path.is_dir() && is_contained(&self.root, &path)).then_some(path)
    }
}

/// Portable fields from `plugin.json`.
#[derive(Clone, Debug)]
pub struct PluginManifest {
    pub schema: String,
    pub name: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<PluginAuthor>,
    pub homepage: Option<String>,
    pub repository: Option<String>,
    pub license: Option<String>,
    pub keywords: Vec<String>,
    /// Opaque client-extension values keyed by namespace.
    pub extensions: BTreeMap<String, Value>,
}

/// Author metadata from a plugin manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginAuthor {
    pub name: Option<String>,
    pub email: Option<String>,
    pub url: Option<String>,
}

/// A validated skill location inside a plugin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginSkill {
    pub name: String,
    pub description: String,
    /// Absolute logical skill directory beneath the plugin's `skills/` path.
    pub directory: PathBuf,
    /// Filesystem-resolved absolute `SKILL.md` path.
    pub skill_file: PathBuf,
}

/// One valid server declaration from `mcp.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginMcpServer {
    pub name: String,
    pub transport: PluginMcpTransport,
}

/// Portable MCP transport data. Placeholder strings remain unexpanded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PluginMcpTransport {
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<String>,
    },
    StreamableHttp {
        url: String,
        headers: BTreeMap<String, String>,
    },
    /// Deprecated MCP HTTP+SSE transport. Runtime support is optional.
    Sse {
        url: String,
        headers: BTreeMap<String, String>,
    },
}

/// Fatal package-level load errors.
#[derive(Debug, Error)]
pub enum PluginError {
    #[error("failed to resolve plugin root {path}: {error}")]
    RootResolve {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("plugin root is not a directory: {path}")]
    InvalidRoot { path: PathBuf },
    #[error("failed to read plugin manifest {path}: {error}")]
    ManifestRead {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("plugin manifest resolves outside the plugin root: {path}")]
    ManifestOutsideRoot { path: PathBuf },
    #[error("invalid JSON in plugin manifest {path}: {error}")]
    ManifestParse {
        path: PathBuf,
        #[source]
        error: serde_json::Error,
    },
    #[error("unsupported plugin schema {schema}")]
    UnsupportedSchema { schema: String },
    #[error("invalid manifest field {field}: {reason}")]
    ManifestInvalid { field: String, reason: String },
}

/// A non-fatal issue scoped to the narrowest affected package component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginDiagnostic {
    pub kind: PluginDiagnosticKind,
    pub path: Option<PathBuf>,
    pub message: String,
}

/// Stable diagnostic categories for programmatic handling.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginDiagnosticKind {
    UnknownManifestField,
    InvalidExtensionsField,
    SkillsLocationInvalid,
    SkillSkipped,
    McpDisabled,
    McpServerSkipped,
}

// ---------------------------------------------------------------------------
// Manifest validation
// ---------------------------------------------------------------------------

fn parse_manifest(
    value: Value,
    path: &Path,
    diagnostics: &mut Vec<PluginDiagnostic>,
) -> Result<PluginManifest, PluginError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_manifest("<root>", "expected a JSON object"))?;

    for field in object.keys() {
        if !MANIFEST_FIELDS.contains(&field.as_str()) {
            diagnostics.push(diagnostic(
                PluginDiagnosticKind::UnknownManifestField,
                Some(path),
                format!("unknown manifest field `{field}` was ignored"),
            ));
        }
    }

    let schema = required_string(object, "$schema")?;
    if schema != PLUGIN_SCHEMA_1_0_0 {
        return Err(PluginError::UnsupportedSchema { schema });
    }

    let name = required_string(object, "name")?;
    if !valid_plugin_name(&name) {
        return Err(invalid_manifest(
            "name",
            "must be 1-64 lowercase alphanumeric, hyphen, or period characters; start and end alphanumeric; and contain neither `--` nor `..`",
        ));
    }

    let version = optional_string(object, "version")?;
    let description = optional_string(object, "description")?;
    let homepage = optional_string(object, "homepage")?;
    let repository = optional_string(object, "repository")?;
    let license = optional_string(object, "license")?;
    let author = parse_author(object.get("author"))?;
    let keywords = parse_keywords(object.get("keywords"))?;
    let extensions = match object.get("extensions") {
        None => BTreeMap::new(),
        Some(Value::Object(values)) => values
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        Some(_) => {
            diagnostics.push(diagnostic(
                PluginDiagnosticKind::InvalidExtensionsField,
                Some(path),
                "non-object `extensions` field was ignored",
            ));
            BTreeMap::new()
        }
    };

    Ok(PluginManifest {
        schema,
        name,
        version,
        description,
        author,
        homepage,
        repository,
        license,
        keywords,
        extensions,
    })
}

fn parse_author(value: Option<&Value>) -> Result<Option<PluginAuthor>, PluginError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid_manifest("author", "expected an object"))?;
    for key in object.keys() {
        if !["name", "email", "url"].contains(&key.as_str()) {
            return Err(invalid_manifest("author", format!("unknown field `{key}`")));
        }
    }
    Ok(Some(PluginAuthor {
        name: optional_string(object, "name")?,
        email: optional_string(object, "email")?,
        url: optional_string(object, "url")?,
    }))
}

fn parse_keywords(value: Option<&Value>) -> Result<Vec<String>, PluginError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| invalid_manifest("keywords", "expected an array of strings"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid_manifest("keywords", "expected an array of strings"))
        })
        .collect()
}

fn required_string(object: &Map<String, Value>, field: &str) -> Result<String, PluginError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| invalid_manifest(field, "missing or not a string"))
}

fn optional_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, PluginError> {
    match object.get(field) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid_manifest(field, "expected a string")),
    }
}

fn invalid_manifest(field: impl Into<String>, reason: impl Into<String>) -> PluginError {
    PluginError::ManifestInvalid {
        field: field.into(),
        reason: reason.into(),
    }
}

fn valid_plugin_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 || name.contains("--") || name.contains("..") {
        return false;
    }
    let bytes = name.as_bytes();
    bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.last().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-' || *byte == b'.'
        })
}

// ---------------------------------------------------------------------------
// Skill discovery
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    license: Option<String>,
    compatibility: Option<String>,
    metadata: Option<BTreeMap<String, String>>,
    #[serde(rename = "allowed-tools")]
    allowed_tools: Option<String>,
}

fn discover_skills(root: &Path, diagnostics: &mut Vec<PluginDiagnostic>) -> Vec<PluginSkill> {
    let skills_path = root.join("skills");
    if fs::symlink_metadata(&skills_path).is_err() {
        return Vec::new();
    }

    let resolved_skills = match fs::canonicalize(&skills_path) {
        Ok(path) if path.is_dir() && is_contained(root, &path) => path,
        _ => {
            diagnostics.push(diagnostic(
                PluginDiagnosticKind::SkillsLocationInvalid,
                Some(&skills_path),
                "`skills` must resolve to a directory inside the plugin root",
            ));
            return Vec::new();
        }
    };

    let entries = match fs::read_dir(&resolved_skills) {
        Ok(entries) => entries,
        Err(error) => {
            diagnostics.push(diagnostic(
                PluginDiagnosticKind::SkillsLocationInvalid,
                Some(&skills_path),
                format!("failed to inspect `skills`: {error}"),
            ));
            return Vec::new();
        }
    };

    let mut children = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    children.sort();

    let mut skills = Vec::new();
    for directory in children {
        let _resolved_directory = match fs::canonicalize(&directory) {
            Ok(path) if path.is_dir() && is_contained(root, &path) => path,
            _ => {
                diagnostics.push(diagnostic(
                    PluginDiagnosticKind::SkillSkipped,
                    Some(&directory),
                    "skill directory does not resolve inside the plugin root",
                ));
                continue;
            }
        };
        let logical_name = match directory.file_name().and_then(|name| name.to_str()) {
            Some(name) => name.to_owned(),
            None => continue,
        };
        let skill_path = directory.join("SKILL.md");
        if fs::symlink_metadata(&skill_path).is_err() {
            continue;
        }
        let resolved_skill = match canonical_file(&skill_path) {
            Ok(path) if is_contained(root, &path) => path,
            _ => {
                diagnostics.push(diagnostic(
                    PluginDiagnosticKind::SkillSkipped,
                    Some(&skill_path),
                    format!("skill `{logical_name}` does not resolve to a regular file inside the plugin root"),
                ));
                continue;
            }
        };

        match parse_skill(&resolved_skill, &logical_name) {
            Some((name, description)) => skills.push(PluginSkill {
                name,
                description,
                directory,
                skill_file: resolved_skill,
            }),
            None => diagnostics.push(diagnostic(
                PluginDiagnosticKind::SkillSkipped,
                Some(&skill_path),
                format!(
                    "skill `{logical_name}` does not conform to the Agent Skills specification"
                ),
            )),
        }
    }
    skills
}

fn parse_skill(path: &Path, parent_name: &str) -> Option<(String, String)> {
    let content = fs::read_to_string(path).ok()?;
    let yaml = split_frontmatter(&content)?;
    let frontmatter = parse_yaml_lenient(yaml)?;
    let name = frontmatter.name?.trim().to_owned();
    let description = frontmatter.description?.trim().to_owned();
    if !valid_skill_name(&name, parent_name)
        || description.is_empty()
        || description.len() > 1024
        || frontmatter
            .compatibility
            .as_deref()
            .is_some_and(|value| value.is_empty() || value.len() > 500)
    {
        return None;
    }
    // Deserializing the complete standard frontmatter above validates the
    // optional field types even though package discovery only exposes catalog
    // metadata. Keep the bindings used so future compiler lints do not obscure
    // that validation boundary.
    let _ = (
        frontmatter.license,
        frontmatter.metadata,
        frontmatter.allowed_tools,
    );
    Some((name, description))
}

fn split_frontmatter(content: &str) -> Option<&str> {
    let stripped = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))?;
    if stripped.starts_with("---") {
        return None;
    }
    stripped
        .split_once("\n---\n")
        .map(|(yaml, _)| yaml)
        .or_else(|| stripped.split_once("\r\n---\r\n").map(|(yaml, _)| yaml))
}

fn parse_yaml_lenient(yaml: &str) -> Option<SkillFrontmatter> {
    if let Ok(frontmatter) = serde_saphyr::from_str(yaml) {
        return Some(frontmatter);
    }
    let fixed = yaml
        .lines()
        .map(|line| {
            if let Some((key, value)) = line.split_once(':') {
                let value = value.trim();
                if !value.is_empty()
                    && !value.starts_with('"')
                    && !value.starts_with('\'')
                    && value.contains(':')
                {
                    return format!("{key}: \"{}\"", value.replace('"', "\\\""));
                }
            }
            line.to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n");
    serde_saphyr::from_str(&fixed).ok()
}

fn valid_skill_name(name: &str, parent_name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name == parent_name
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

// ---------------------------------------------------------------------------
// MCP discovery and validation
// ---------------------------------------------------------------------------

fn discover_mcp(root: &Path, diagnostics: &mut Vec<PluginDiagnostic>) -> Vec<PluginMcpServer> {
    let mcp_path = root.join("mcp.json");
    if fs::symlink_metadata(&mcp_path).is_err() {
        return Vec::new();
    }
    let resolved = match canonical_file(&mcp_path) {
        Ok(path) if is_contained(root, &path) => path,
        _ => {
            disable_mcp(
                diagnostics,
                &mcp_path,
                "`mcp.json` must resolve to a regular file inside the plugin root",
            );
            return Vec::new();
        }
    };
    let content = match fs::read_to_string(&resolved) {
        Ok(content) => content,
        Err(error) => {
            disable_mcp(
                diagnostics,
                &mcp_path,
                format!("failed to read `mcp.json`: {error}"),
            );
            return Vec::new();
        }
    };
    let value: Value = match serde_json::from_str(&content) {
        Ok(value) => value,
        Err(error) => {
            disable_mcp(diagnostics, &mcp_path, format!("invalid JSON: {error}"));
            return Vec::new();
        }
    };
    let Some(object) = value.as_object() else {
        disable_mcp(diagnostics, &mcp_path, "expected a JSON object");
        return Vec::new();
    };
    if object
        .keys()
        .any(|key| key != "$schema" && key != "mcpServers")
    {
        disable_mcp(
            diagnostics,
            &mcp_path,
            "unknown top-level field in `mcp.json`",
        );
        return Vec::new();
    }
    if object.get("$schema").and_then(Value::as_str) != Some(MCP_SCHEMA_1_0_0) {
        disable_mcp(
            diagnostics,
            &mcp_path,
            "unsupported or mismatched Agent Plugins MCP schema",
        );
        return Vec::new();
    }
    let Some(servers) = object.get("mcpServers").and_then(Value::as_object) else {
        disable_mcp(diagnostics, &mcp_path, "`mcpServers` must be an object");
        return Vec::new();
    };

    let mut names = servers.keys().collect::<Vec<_>>();
    names.sort();
    names
        .into_iter()
        .filter_map(|name| match parse_mcp_server(root, &servers[name]) {
            Ok(transport) => Some(PluginMcpServer {
                name: name.clone(),
                transport,
            }),
            Err(reason) => {
                diagnostics.push(diagnostic(
                    PluginDiagnosticKind::McpServerSkipped,
                    Some(&mcp_path),
                    format!("MCP server `{name}` was skipped: {reason}"),
                ));
                None
            }
        })
        .collect()
}

fn parse_mcp_server(root: &Path, value: &Value) -> Result<PluginMcpTransport, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "expected an object".to_owned())?;
    let transport = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing string field `type`".to_owned())?;
    match transport {
        "stdio" => parse_stdio(root, object),
        "streamable-http" => parse_remote(object, false),
        "sse" => parse_remote(object, true),
        other => Err(format!("unsupported transport type `{other}`")),
    }
}

fn parse_stdio(root: &Path, object: &Map<String, Value>) -> Result<PluginMcpTransport, String> {
    reject_unknown_fields(object, &["type", "command", "args", "env", "cwd"])?;
    let command = object
        .get("command")
        .and_then(Value::as_str)
        .filter(|command| !command.is_empty())
        .ok_or_else(|| "missing non-empty string field `command`".to_owned())?
        .to_owned();
    validate_command(root, &command)?;
    let args = string_array(object.get("args"), "args")?;
    let env = string_map(object.get("env"), "env")?;
    if env.contains_key("PLUGIN_ROOT") || env.contains_key("PLUGIN_DATA") {
        return Err("`env` may not define PLUGIN_ROOT or PLUGIN_DATA".to_owned());
    }
    let cwd = match object.get("cwd") {
        None => None,
        Some(Value::String(value)) => {
            validate_cwd(root, value)?;
            Some(value.clone())
        }
        Some(_) => return Err("`cwd` must be a string".to_owned()),
    };
    Ok(PluginMcpTransport::Stdio {
        command,
        args,
        env,
        cwd,
    })
}

fn parse_remote(object: &Map<String, Value>, sse: bool) -> Result<PluginMcpTransport, String> {
    reject_unknown_fields(object, &["type", "url", "headers"])?;
    let url = object
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .ok_or_else(|| "missing non-empty string field `url`".to_owned())?
        .to_owned();
    validate_remote_url(&url)?;
    let headers = string_map(object.get("headers"), "headers")?;
    validate_headers(&headers)?;
    if sse {
        Ok(PluginMcpTransport::Sse { url, headers })
    } else {
        Ok(PluginMcpTransport::StreamableHttp { url, headers })
    }
}

fn reject_unknown_fields(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(format!("unknown field `{field}`"));
    }
    Ok(())
}

fn string_array(value: Option<&Value>, field: &str) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| format!("`{field}` must be an array of strings"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("`{field}` must be an array of strings"))
        })
        .collect()
}

fn string_map(value: Option<&Value>, field: &str) -> Result<BTreeMap<String, String>, String> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    value
        .as_object()
        .ok_or_else(|| format!("`{field}` must be an object of strings"))?
        .iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_owned()))
                .ok_or_else(|| format!("`{field}` must be an object of strings"))
        })
        .collect()
}

fn validate_command(root: &Path, command: &str) -> Result<(), String> {
    if command.contains('\0') || command.contains('\n') || command.contains('\r') {
        return Err("`command` must be one executable token".to_owned());
    }
    if let Some(relative) = command.strip_prefix("./") {
        validate_relative_suffix(relative, "command")?;
        validate_existing_ancestor(root, relative, "command")?;
        return Ok(());
    }
    if command.contains('/') || command.contains('\\') {
        return Err("`command` must be a bare executable name or begin with `./`".to_owned());
    }
    Ok(())
}

fn validate_cwd(root: &Path, cwd: &str) -> Result<(), String> {
    let (suffix, plugin_rooted) = if let Some(suffix) = cwd.strip_prefix("./") {
        (suffix, true)
    } else if matches!(cwd, "${PLUGIN_ROOT}" | "${PLUGIN_DATA}") {
        return Ok(());
    } else if let Some(suffix) = cwd.strip_prefix("${PLUGIN_ROOT}/") {
        (suffix, true)
    } else if let Some(suffix) = cwd.strip_prefix("${PLUGIN_DATA}/") {
        (suffix, false)
    } else {
        return Err(
            "`cwd` must be plugin-relative or rooted at ${PLUGIN_ROOT} or ${PLUGIN_DATA}"
                .to_owned(),
        );
    };
    validate_relative_suffix(suffix, "cwd")?;
    if plugin_rooted {
        validate_existing_ancestor(root, suffix, "cwd")?;
    }
    Ok(())
}

fn validate_existing_ancestor(root: &Path, relative: &str, field: &str) -> Result<(), String> {
    let mut candidate = root.join(relative);
    while fs::symlink_metadata(&candidate).is_err() {
        if !candidate.pop() || candidate == root {
            return Ok(());
        }
    }
    let resolved = fs::canonicalize(&candidate)
        .map_err(|error| format!("failed to resolve `{field}` path: {error}"))?;
    if !is_contained(root, &resolved) {
        return Err(format!("`{field}` resolves outside its permitted root"));
    }
    Ok(())
}

fn validate_relative_suffix(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty() || Path::new(value).is_absolute() {
        return Err(format!("`{field}` must name a contained path"));
    }
    let mut depth = 0usize;
    for component in Path::new(value).components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir if depth > 0 => depth -= 1,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("`{field}` escapes its permitted root"));
            }
        }
    }
    Ok(())
}

fn validate_remote_url(value: &str) -> Result<(), String> {
    let url = Url::parse(value).map_err(|error| format!("invalid URL: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("URL scheme must be HTTP or HTTPS".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() || url.fragment().is_some() {
        return Err("URL must not contain user information or a fragment".to_owned());
    }
    let host = url
        .host()
        .ok_or_else(|| "URL must include a host".to_owned())?;
    let loopback = match host {
        Host::Domain(domain) => domain == "localhost",
        Host::Ipv4(address) => address.is_loopback(),
        Host::Ipv6(address) => address.is_loopback(),
    };
    if url.scheme() == "http" && !loopback {
        return Err("non-loopback MCP URLs must use HTTPS".to_owned());
    }
    Ok(())
}

fn validate_headers(headers: &BTreeMap<String, String>) -> Result<(), String> {
    let mut names = BTreeSet::new();
    for (name, value) in headers {
        HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| format!("invalid HTTP header name `{name}`: {error}"))?;
        HeaderValue::from_str(value)
            .map_err(|error| format!("invalid value for HTTP header `{name}`: {error}"))?;
        if !names.insert(name.to_ascii_lowercase()) {
            return Err(format!("duplicate case-insensitive HTTP header `{name}`"));
        }
    }
    Ok(())
}

fn disable_mcp(diagnostics: &mut Vec<PluginDiagnostic>, path: &Path, message: impl Into<String>) {
    diagnostics.push(diagnostic(
        PluginDiagnosticKind::McpDisabled,
        Some(path),
        message,
    ));
}

// ---------------------------------------------------------------------------
// Filesystem and diagnostics helpers
// ---------------------------------------------------------------------------

fn canonical_file(path: &Path) -> std::io::Result<PathBuf> {
    let resolved = fs::canonicalize(path)?;
    if !resolved.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path is not a regular file",
        ));
    }
    Ok(resolved)
}

fn is_contained(root: &Path, candidate: &Path) -> bool {
    candidate == root || candidate.starts_with(root)
}

fn diagnostic(
    kind: PluginDiagnosticKind,
    path: Option<&Path>,
    message: impl Into<String>,
) -> PluginDiagnostic {
    PluginDiagnostic {
        kind,
        path: path.map(Path::to_path_buf),
        message: message.into(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_plugin(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agentkit-plugin-{label}-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_manifest(root: &Path, extra: &str) {
        fs::write(
            root.join("plugin.json"),
            format!("{{\"$schema\":\"{PLUGIN_SCHEMA_1_0_0}\",\"name\":\"test-plugin\"{extra}}}"),
        )
        .unwrap();
    }

    fn write_skill(root: &Path, directory: &str, name: &str) {
        let dir = root.join("skills").join(directory);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Test skill.\n---\nInstructions."),
        )
        .unwrap();
    }

    #[test]
    fn loads_manifest_and_preserves_opaque_extensions() {
        let root = temp_plugin("manifest");
        write_manifest(
            &root,
            ",\"version\":\"not-semver\",\"extensions\":{\"com.example\":42},\"future\":true",
        );

        let plugin = AgentPlugin::load(&root).unwrap();
        assert_eq!(plugin.manifest().version.as_deref(), Some("not-semver"));
        assert_eq!(
            plugin.extension_manifest("com.example"),
            Some(&Value::from(42))
        );
        assert!(
            plugin
                .diagnostics()
                .iter()
                .any(|d| d.kind == PluginDiagnosticKind::UnknownManifestField)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_invalid_manifest_name() {
        let root = temp_plugin("bad-name");
        fs::write(
            root.join("plugin.json"),
            format!("{{\"$schema\":\"{PLUGIN_SCHEMA_1_0_0}\",\"name\":\"Bad--Name\"}}"),
        )
        .unwrap();
        assert!(matches!(
            AgentPlugin::load(&root),
            Err(PluginError::ManifestInvalid { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discovers_only_immediate_valid_skills() {
        let root = temp_plugin("skills");
        write_manifest(&root, "");
        write_skill(&root, "valid", "valid");
        write_skill(&root.join("skills/valid"), "nested", "nested");
        write_skill(&root, "wrong-dir", "other-name");

        let plugin = AgentPlugin::load(&root).unwrap();
        assert_eq!(plugin.skills().len(), 1);
        assert_eq!(plugin.skills()[0].name, "valid");
        assert!(
            plugin
                .diagnostics()
                .iter()
                .any(|d| d.kind == PluginDiagnosticKind::SkillSkipped)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn isolates_invalid_mcp_entries() {
        let root = temp_plugin("mcp");
        write_manifest(&root, "");
        fs::write(
            root.join("mcp.json"),
            format!(
                r#"{{"$schema":"{MCP_SCHEMA_1_0_0}","mcpServers":{{
                    "local":{{"type":"stdio","command":"./bin/server","args":["${{PLUGIN_DATA}}/db"]}},
                    "remote":{{"type":"streamable-http","url":"https://example.com/mcp","headers":{{"X-Tenant":"public"}}}},
                    "bad":{{"type":"stdio","command":"node","extra":true}}
                }}}}"#
            ),
        )
        .unwrap();

        let plugin = AgentPlugin::load(&root).unwrap();
        assert_eq!(plugin.mcp_servers().len(), 2);
        assert!(
            plugin
                .diagnostics()
                .iter()
                .any(|d| d.kind == PluginDiagnosticKind::McpServerSkipped)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disables_only_mcp_for_invalid_top_level_document() {
        let root = temp_plugin("mcp-disabled");
        write_manifest(&root, "");
        write_skill(&root, "valid", "valid");
        fs::write(
            root.join("mcp.json"),
            format!("{{\"$schema\":\"{MCP_SCHEMA_1_0_0}\",\"mcpServers\":{{}},\"extra\":true}}"),
        )
        .unwrap();

        let plugin = AgentPlugin::load(&root).unwrap();
        assert_eq!(plugin.skills().len(), 1);
        assert!(plugin.mcp_servers().is_empty());
        assert!(
            plugin
                .diagnostics()
                .iter()
                .any(|d| d.kind == PluginDiagnosticKind::McpDisabled)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_remote_urls_headers_and_stdio_paths() {
        assert!(validate_remote_url("https://example.com/mcp").is_ok());
        assert!(validate_remote_url("http://localhost:3000/mcp").is_ok());
        assert!(validate_remote_url("http://127.0.0.1/mcp").is_ok());
        assert!(validate_remote_url("http://example.com/mcp").is_err());
        assert!(validate_remote_url("https://user@example.com/mcp").is_err());
        let root = temp_plugin("cwd-validation");
        assert!(validate_cwd(&root, "${PLUGIN_ROOT}/work").is_ok());
        assert!(validate_cwd(&root, "${PLUGIN_DATA}/work").is_ok());
        assert!(validate_cwd(&root, "./work").is_ok());
        assert!(validate_cwd(&root, "../work").is_err());

        let headers = BTreeMap::from([
            ("X-Test".to_owned(), "one".to_owned()),
            ("x-test".to_owned(), "two".to_owned()),
        ]);
        assert!(validate_headers(&headers).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_plugin_rooted_mcp_paths_through_escaping_symlinks() {
        use std::os::unix::fs::symlink;

        let root = temp_plugin("mcp-path-escape");
        let outside = temp_plugin("mcp-path-outside");
        symlink(&outside, root.join("escape")).unwrap();

        assert!(validate_cwd(&root, "./escape/work").is_err());
        assert!(validate_command(&root, "./escape/server").is_err());

        fs::remove_file(root.join("escape")).unwrap();
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn skips_skill_symlink_that_escapes_root() {
        use std::os::unix::fs::symlink;

        let root = temp_plugin("escape");
        let outside = temp_plugin("outside");
        write_manifest(&root, "");
        fs::create_dir_all(root.join("skills/escaped")).unwrap();
        fs::write(
            outside.join("SKILL.md"),
            "---\nname: escaped\ndescription: Escape.\n---\nBody.",
        )
        .unwrap();
        symlink(
            outside.join("SKILL.md"),
            root.join("skills/escaped/SKILL.md"),
        )
        .unwrap();

        let plugin = AgentPlugin::load(&root).unwrap();
        assert!(plugin.skills().is_empty());
        assert!(
            plugin
                .diagnostics()
                .iter()
                .any(|d| d.kind == PluginDiagnosticKind::SkillSkipped)
        );

        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(outside).unwrap();
    }
}
