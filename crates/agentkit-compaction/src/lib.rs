use std::sync::Arc;

use agentkit_core::{Item, MetadataMap, SessionId, TurnId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionReason {
    TranscriptTooLong,
    Manual,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionRequest {
    pub session_id: SessionId,
    pub turn_id: Option<TurnId>,
    pub transcript: Vec<Item>,
    pub reason: CompactionReason,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionResult {
    pub transcript: Vec<Item>,
    pub replaced_items: usize,
    pub metadata: MetadataMap,
}

pub trait CompactionTrigger: Send + Sync {
    fn should_compact(
        &self,
        session_id: &SessionId,
        turn_id: Option<&TurnId>,
        transcript: &[Item],
    ) -> Option<CompactionReason>;
}

#[async_trait]
pub trait Compactor: Send + Sync {
    async fn compact(
        &self,
        request: CompactionRequest,
    ) -> Result<CompactionResult, CompactionError>;
}

#[derive(Clone)]
pub struct CompactionConfig {
    pub trigger: Arc<dyn CompactionTrigger>,
    pub compactor: Arc<dyn Compactor>,
}

impl CompactionConfig {
    pub fn new(
        trigger: impl CompactionTrigger + 'static,
        compactor: impl Compactor + 'static,
    ) -> Self {
        Self {
            trigger: Arc::new(trigger),
            compactor: Arc::new(compactor),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemCountTrigger {
    pub max_items: usize,
}

impl ItemCountTrigger {
    pub fn new(max_items: usize) -> Self {
        Self { max_items }
    }
}

impl CompactionTrigger for ItemCountTrigger {
    fn should_compact(
        &self,
        _session_id: &SessionId,
        _turn_id: Option<&TurnId>,
        transcript: &[Item],
    ) -> Option<CompactionReason> {
        (transcript.len() > self.max_items).then_some(CompactionReason::TranscriptTooLong)
    }
}

#[derive(Debug, Error)]
pub enum CompactionError {
    #[error("compaction failed: {0}")]
    Failed(String),
}

#[cfg(test)]
mod tests {
    use agentkit_core::{ItemKind, Part, TextPart};

    use super::*;

    fn item(text: &str) -> Item {
        Item {
            id: None,
            kind: ItemKind::User,
            parts: vec![Part::Text(TextPart {
                text: text.into(),
                metadata: MetadataMap::new(),
            })],
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn item_count_trigger_fires_after_limit() {
        let trigger = ItemCountTrigger::new(2);
        let transcript = vec![item("a"), item("b"), item("c")];
        assert_eq!(
            trigger.should_compact(&SessionId::new("s"), None, &transcript),
            Some(CompactionReason::TranscriptTooLong)
        );
    }
}
