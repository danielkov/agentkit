//! Skills over MCP (SEP-2640): metadata discovery, separate from skill activation.

use std::collections::BTreeSet;

use rmcp::model::{CacheScope, ClientRequest, CustomRequest, ServerResult};
use rmcp::service::{Peer, RoleClient};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};

use super::{AuthOperation, McpConnection, McpError, McpServerId, operation_error};

/// The extension identifier advertised in `server/discover`.
pub const MCP_SKILLS_EXTENSION_ID: &str = "io.modelcontextprotocol/skills";

/// A skill entry as advertised by its originating MCP server.
/// Pair `uri` with the connection's server ID; the frontmatter name is not unique.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSkill {
    /// URI of `SKILL.md` on the originating server.
    pub uri: String,
    /// All advertised frontmatter fields, including name and description.
    pub frontmatter: Map<String, Value>,
    /// Supporting files, or an indication that the skill generates content.
    pub resources: McpSkillResources,
}

/// An advertised file and its integrity metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSkillFile {
    /// Resource URI on the originating server.
    pub uri: String,
    /// SHA-256 digest prefixed with `sha256:`.
    pub digest: String,
    /// Raw byte size.
    pub size: u64,
}

/// The files advertised by a skill.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpSkillResources {
    /// An explicit manifest, including `SKILL.md` itself.
    Manifest(Vec<McpSkillFile>),
    /// The literal `"dynamic"`, denoting generated content.
    Dynamic(McpSkillDynamic),
}

/// Wire marker for a dynamically generated skill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpSkillDynamic {
    /// Generated content.
    Dynamic,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Complete<T> {
    result_type: CompleteResultType,
    #[serde(flatten)]
    payload: T,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum CompleteResultType {
    Complete,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListPage {
    // Require and decode the cache contract even though this adapter does not cache.
    #[serde(rename = "ttlMs")]
    _ttl_ms: i64,
    #[serde(rename = "cacheScope")]
    _cache_scope: CacheScope,
    skills: Vec<McpSkill>,
    next_cursor: Option<String>,
}

#[derive(Deserialize)]
struct GetResult {
    skill: McpSkill,
}

impl McpConnection {
    /// Lists all pages of skill metadata without reading any skill files.
    /// Returns an empty list without sending a request if skills are not advertised.
    pub async fn list_skills(&self) -> Result<Vec<McpSkill>, McpError> {
        let peer = self.peer();
        if !supports_skills(&peer) {
            return Ok(Vec::new());
        }
        let mut skills = Vec::new();
        let mut seen = BTreeSet::new();
        let mut params = json!({});
        loop {
            let page: ListPage = request(&peer, self.server_id(), "skills/list", params).await?;
            skills.extend(page.skills);
            let Some(cursor) = page.next_cursor else {
                return Ok(skills);
            };
            if !seen.insert(cursor.clone()) {
                return Err(McpError::Protocol("skills/list repeated a cursor".into()));
            }
            params = json!({"cursor": cursor});
        }
    }

    /// Retrieves metadata for a known skill URI, including unlisted skills.
    /// The caller is responsible for approval and integrity checks before using files.
    pub async fn get_skill(&self, uri: &str) -> Result<McpSkill, McpError> {
        let peer = self.peer();
        if !supports_skills(&peer) {
            return Err(McpError::Protocol("server did not advertise skills".into()));
        }
        let result: GetResult =
            request(&peer, self.server_id(), "skills/get", json!({"uri": uri})).await?;
        if result.skill.uri != uri {
            return Err(McpError::Protocol(
                "skills/get returned a different URI".into(),
            ));
        }
        Ok(result.skill)
    }
}

fn supports_skills(peer: &Peer<RoleClient>) -> bool {
    peer.peer_info().is_some_and(|info| {
        !info.protocol_version.has_initialize()
            && info.capabilities.resources.is_some()
            && info
                .capabilities
                .extensions
                .as_ref()
                .is_some_and(|extensions| extensions.contains_key(MCP_SKILLS_EXTENSION_ID))
    })
}

async fn request<T: DeserializeOwned>(
    peer: &Peer<RoleClient>,
    server_id: &McpServerId,
    method: &str,
    params: Value,
) -> Result<T, McpError> {
    let result = peer
        .send_request(ClientRequest::CustomRequest(CustomRequest::new(
            method,
            Some(params.clone()),
        )))
        .await
        .map_err(|error| {
            operation_error(server_id, method, error, || AuthOperation::McpOther {
                server_id: server_id.to_string(),
                method: method.to_string(),
                params,
                metadata: Default::default(),
            })
        })?;
    let ServerResult::CustomResult(result) = result else {
        return Err(McpError::Protocol(format!("unexpected {method} response")));
    };
    let Complete {
        result_type: CompleteResultType::Complete,
        payload,
        ..
    } = serde_json::from_value(result.0)?;
    Ok(payload)
}
