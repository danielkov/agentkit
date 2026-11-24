//! Skills over MCP (SEP-2640): metadata discovery, separate from skill activation.

use std::collections::BTreeSet;

use rmcp::model::{GetSkillRequestParams, PaginatedRequestParams};
use rmcp::service::{Peer, RoleClient, ServiceError};
use serde_json::{Value, json};

use super::{AuthOperation, McpConnection, McpError, McpServerId, operation_error};

pub use rmcp::model::{
    SKILLS_EXTENSION_ID as MCP_SKILLS_EXTENSION_ID, Skill as McpSkill,
    SkillFrontmatter as McpSkillFrontmatter, SkillResource as McpSkillFile,
    SkillResources as McpSkillResources,
};

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
        let mut cursor = None;
        loop {
            let page = peer
                .list_skills(Some(
                    PaginatedRequestParams::default().with_cursor(cursor.clone()),
                ))
                .await
                .map_err(|error| {
                    skill_error(self.server_id(), "skills/list", error, || {
                        cursor
                            .as_ref()
                            .map_or_else(|| json!({}), |cursor| json!({"cursor":cursor}))
                    })
                })?;
            skills.extend(page.skills);
            let Some(next) = page.next_cursor else {
                return Ok(skills);
            };
            if !seen.insert(next.clone()) {
                return Err(McpError::Protocol("skills/list repeated a cursor".into()));
            }
            cursor = Some(next);
        }
    }

    /// Retrieves metadata for a known skill URI, including unlisted skills.
    /// The caller is responsible for approval and integrity checks before using files.
    pub async fn get_skill(&self, uri: &str) -> Result<McpSkill, McpError> {
        let peer = self.peer();
        if !supports_skills(&peer) {
            return Err(McpError::Protocol("server did not advertise skills".into()));
        }
        let result = peer
            .get_skill(GetSkillRequestParams::new(uri))
            .await
            .map_err(|error| {
                skill_error(self.server_id(), "skills/get", error, || json!({"uri":uri}))
            })?;
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
        info.capabilities.resources.is_some() && info.capabilities.supports_skills()
    })
}

fn skill_error(
    server_id: &McpServerId,
    method: &str,
    error: ServiceError,
    params: impl FnOnce() -> Value,
) -> McpError {
    operation_error(server_id, method, error, || AuthOperation::McpOther {
        server_id: server_id.to_string(),
        method: method.to_string(),
        params: params(),
        metadata: Default::default(),
    })
}
