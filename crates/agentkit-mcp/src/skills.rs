//! Skills over MCP (SEP-2640): metadata discovery, separate from skill activation.

use std::{collections::BTreeSet, mem::size_of, time::Duration};

use rmcp::model::{GetSkillRequestParams, ListSkillsResult, PaginatedRequestParams, ResultType};
use rmcp::service::{Peer, RoleClient, ServiceError};
use serde_json::{Map, Value, json};

use super::{AuthOperation, ConnectedPeer, McpConnection, McpError, McpServerId, operation_error};

pub use rmcp::model::{
    SKILLS_EXTENSION_ID as MCP_SKILLS_EXTENSION_ID, Skill as McpSkill,
    SkillFrontmatter as McpSkillFrontmatter, SkillResource as McpSkillFile,
    SkillResources as McpSkillResources,
};

/// Limits for a single skills inventory scan.
///
/// Byte limits are checked after rmcp decodes a response. They charge decoded
/// strings, containers, and item storage, not wire bytes or exact allocator
/// usage. rmcp's transport limits still bound the initial response read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpSkillScanLimits {
    /// Maximum number of skills passed to the visitor.
    pub max_entries: usize,
    /// Maximum number of pages requested, including empty pages.
    pub max_pages: usize,
    /// Maximum decoded bytes charged for one skill.
    pub max_entry_decoded_bytes: usize,
    /// Maximum decoded bytes charged for one page.
    pub max_page_decoded_bytes: usize,
    /// Maximum decoded bytes charged across all accepted pages, including
    /// duplicate detection keys.
    pub max_total_decoded_bytes: usize,
    /// Timeout for each page request, not including time spent in the visitor.
    pub page_timeout: Duration,
}

impl Default for McpSkillScanLimits {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_pages: 64,
            max_entry_decoded_bytes: 256 * 1024,
            max_page_decoded_bytes: 2 * 1024 * 1024,
            max_total_decoded_bytes: 8 * 1024 * 1024,
            page_timeout: Duration::from_secs(15),
        }
    }
}

/// Why a skills inventory scan did not reach the end of the inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum McpSkillScanLimit {
    EntryCount,
    PageCount,
    EntryDecodedBytes,
    PageDecodedBytes,
    TotalDecodedBytes,
}

/// Completion status of a skills inventory scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpSkillScanOutcome {
    Complete,
    Truncated(McpSkillScanLimit),
}

impl McpConnection {
    /// Lists all skill metadata without reading any skill files.
    /// Returns an empty list without sending a request if skills are not advertised.
    /// Use [`Self::visit_skills`] to opt in to scan limits.
    pub async fn list_skills(&self) -> Result<Vec<McpSkill>, McpError> {
        let mut skills = Vec::new();
        self.scan_skills(None, |skill| skills.push(skill)).await?;
        Ok(skills)
    }

    /// Visits decoded skill metadata one page at a time, in server order, on
    /// the peer captured at the start of the scan. Does not read skill files.
    ///
    /// A page exceeding a byte limit is not visited. The entry count limit can
    /// stop within a page. Earlier visits are not undone on truncation or error.
    /// If skills are not advertised, the scan completes without any requests.
    pub async fn visit_skills(
        &self,
        limits: McpSkillScanLimits,
        visit: impl FnMut(McpSkill),
    ) -> Result<McpSkillScanOutcome, McpError> {
        self.scan_skills(Some(limits), visit).await
    }

    async fn scan_skills(
        &self,
        mut budget: Option<McpSkillScanLimits>,
        mut visit: impl FnMut(McpSkill),
    ) -> Result<McpSkillScanOutcome, McpError> {
        use McpSkillScanLimit as Limit;
        use McpSkillScanOutcome::{Complete, Truncated};

        let peer = self.peer();
        if !supports_skills(&peer.peer) {
            return Ok(Complete);
        }
        let mut seen_cursors = BTreeSet::new();
        let mut seen_uris = BTreeSet::new();
        let mut cursor = None;
        loop {
            if let Some(budget) = &mut budget {
                if budget.max_entries == 0 {
                    return Ok(Truncated(Limit::EntryCount));
                }
                if budget.max_pages == 0 {
                    return Ok(Truncated(Limit::PageCount));
                }
                if budget.max_total_decoded_bytes < size_of::<ListSkillsResult>() {
                    return Ok(Truncated(Limit::TotalDecodedBytes));
                }
                budget.max_pages -= 1;
            }
            let params = PaginatedRequestParams::default().with_cursor(cursor.clone());
            let request = peer.peer.list_skills(Some(params));
            let result = if let Some(budget) = &budget {
                tokio::time::timeout(budget.page_timeout, request)
                    .await
                    .map_err(|_| McpError::Timeout {
                        operation: "skills/list",
                        duration: budget.page_timeout,
                    })?
            } else {
                request.await
            };
            let page = result.map_err(|error| {
                skill_error(&peer, self.server_id(), "skills/list", error, || {
                    cursor
                        .as_ref()
                        .map_or_else(|| json!({}), |cursor| json!({"cursor":cursor}))
                })
            })?;
            require_complete("skills/list", page.result_type.as_ref())?;
            let count = if let Some(budget) = &mut budget {
                let count = page.skills.len().min(budget.max_entries);
                let charge = match page_charge(&page, count, budget) {
                    Ok(charge) => charge,
                    Err(limit) => return Ok(Truncated(limit)),
                };
                let Some(bytes_left) = budget.max_total_decoded_bytes.checked_sub(charge) else {
                    return Ok(Truncated(Limit::TotalDecodedBytes));
                };
                budget.max_total_decoded_bytes = bytes_left;
                budget.max_entries -= count;
                count
            } else {
                page.skills.len()
            };

            for skill in page.skills.iter().take(count) {
                if !seen_uris.insert(skill.uri.clone()) {
                    return Err(McpError::Protocol(
                        "skills/list repeated a skill URI".into(),
                    ));
                }
            }
            if let Some(next) = &page.next_cursor
                && !seen_cursors.insert(next.clone())
            {
                return Err(McpError::Protocol("skills/list repeated a cursor".into()));
            }
            let partial_page = count < page.skills.len();
            for skill in page.skills.into_iter().take(count) {
                visit(skill);
            }
            if partial_page {
                return Ok(Truncated(Limit::EntryCount));
            }
            let Some(next) = page.next_cursor else {
                return Ok(Complete);
            };
            cursor = Some(next);
        }
    }

    /// Retrieves metadata for a known skill URI, including unlisted skills.
    /// The caller is responsible for approval and integrity checks before using files.
    pub async fn get_skill(&self, uri: &str) -> Result<McpSkill, McpError> {
        let peer = self.peer();
        if !supports_skills(&peer.peer) {
            return Err(McpError::Protocol("server did not advertise skills".into()));
        }
        let result = peer
            .peer
            .get_skill(GetSkillRequestParams::new(uri))
            .await
            .map_err(|error| {
                skill_error(
                    &peer,
                    self.server_id(),
                    "skills/get",
                    error,
                    || json!({"uri":uri}),
                )
            })?;
        require_complete("skills/get", result.result_type.as_ref())?;
        if result.skill.uri != uri {
            return Err(McpError::Protocol(
                "skills/get returned a different URI".into(),
            ));
        }
        Ok(result.skill)
    }
}

// Account for the decoded tree directly: a serialized length would miss the
// amplification of dense arrays/objects and would traverse or copy data again.
fn page_charge(
    page: &ListSkillsResult,
    eligible_entries: usize,
    limits: &McpSkillScanLimits,
) -> Result<usize, McpSkillScanLimit> {
    let mut bytes = size_of::<ListSkillsResult>()
        .saturating_add(page.skills.capacity().saturating_mul(size_of::<McpSkill>()));
    if let Some(meta) = &page.meta {
        bytes = bytes.saturating_add(object_bytes(&meta.0));
    }
    if let Some(cursor) = &page.next_cursor {
        bytes = bytes.saturating_add(cursor.capacity());
    }
    let mut key_bytes = page
        .next_cursor
        .as_ref()
        .map_or(0, |cursor| key_charge(cursor));
    for (index, skill) in page.skills.iter().enumerate() {
        let heap = skill_heap_bytes(skill);
        if size_of::<McpSkill>().saturating_add(heap) > limits.max_entry_decoded_bytes {
            return Err(McpSkillScanLimit::EntryDecodedBytes);
        }
        bytes = bytes.saturating_add(heap);
        if index < eligible_entries {
            key_bytes = key_bytes.saturating_add(key_charge(&skill.uri));
        }
    }
    if bytes > limits.max_page_decoded_bytes {
        return Err(McpSkillScanLimit::PageDecodedBytes);
    }
    Ok(bytes.saturating_add(key_bytes))
}

fn skill_heap_bytes(skill: &McpSkill) -> usize {
    let mut bytes = skill
        .uri
        .capacity()
        .saturating_add(skill.frontmatter.name.capacity())
        .saturating_add(skill.frontmatter.description.capacity())
        .saturating_add(object_bytes(&skill.frontmatter.extra));
    if let McpSkillResources::Files(files) = &skill.resources {
        bytes = bytes.saturating_add(files.capacity().saturating_mul(size_of::<McpSkillFile>()));
        for file in files {
            bytes = bytes
                .saturating_add(file.uri.capacity())
                .saturating_add(file.digest.capacity());
        }
    }
    if let Some(meta) = &skill.meta {
        bytes = bytes.saturating_add(object_bytes(&meta.0));
    }
    bytes
}

// Include decoded key/slot storage even when a key or its value is empty.
fn object_bytes(object: &Map<String, Value>) -> usize {
    object.iter().fold(0_usize, |bytes, (key, value)| {
        bytes
            .saturating_add(2 * size_of::<(String, Value)>())
            .saturating_add(key.capacity())
            .saturating_add(value_heap_bytes(value))
    })
}

fn value_heap_bytes(value: &Value) -> usize {
    match value {
        Value::String(string) => string.capacity(),
        Value::Array(items) => items.iter().fold(
            items.capacity().saturating_mul(size_of::<Value>()),
            |bytes, item| bytes.saturating_add(value_heap_bytes(item)),
        ),
        Value::Object(object) => object_bytes(object),
        _ => 0,
    }
}

fn key_charge(key: &str) -> usize {
    (2 * size_of::<String>()).saturating_add(key.len())
}

fn require_complete(method: &str, result_type: Option<&ResultType>) -> Result<(), McpError> {
    if let Some(result_type) = result_type.filter(|kind| !kind.is_complete()) {
        return Err(McpError::Protocol(format!(
            "{method} returned unsupported resultType {result_type}"
        )));
    }
    Ok(())
}

fn supports_skills(peer: &Peer<RoleClient>) -> bool {
    peer.peer_info().is_some_and(|info| {
        info.capabilities.resources.is_some() && info.capabilities.supports_skills()
    })
}

fn skill_error(
    peer: &ConnectedPeer,
    server_id: &McpServerId,
    method: &str,
    error: ServiceError,
    params: impl FnOnce() -> Value,
) -> McpError {
    if let Some(error) = peer.stdio_error(&error) {
        return error;
    }
    operation_error(server_id, method, error, || AuthOperation::McpOther {
        server_id: server_id.to_string(),
        method: method.to_string(),
        params: params(),
        metadata: Default::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_charge_counts_only_eligible_uri_keys() {
        let page: ListSkillsResult = serde_json::from_value(json!({
            "skills": [
                {"uri":"skill://first/SKILL.md", "frontmatter":{"name":"first", "description":"first"}, "resources":"dynamic"},
                {"uri":"skill://second/SKILL.md", "frontmatter":{"name":"second", "description":"second"}, "resources":"dynamic"}
            ]
        }))
        .unwrap();
        let limits = McpSkillScanLimits::default();
        let decoded = size_of::<ListSkillsResult>()
            + page.skills.capacity() * size_of::<McpSkill>()
            + page.skills.iter().map(skill_heap_bytes).sum::<usize>();
        let prefix = decoded + key_charge(&page.skills[0].uri);
        assert_eq!(page_charge(&page, 1, &limits), Ok(prefix));
        assert_eq!(
            page_charge(&page, 2, &limits),
            Ok(prefix + key_charge(&page.skills[1].uri))
        );
    }
}
