use std::collections::BTreeSet;
use std::sync::Arc;

use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TurnId};
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryRequest {
    pub session_id: SessionId,
    pub turn_id: Option<TurnId>,
    pub items: Vec<Item>,
    pub reason: CompactionReason,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryResult {
    pub items: Vec<Item>,
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
pub trait CompactionBackend: Send + Sync {
    async fn summarize(&self, request: SummaryRequest) -> Result<SummaryResult, CompactionError>;
}

pub struct CompactionContext<'a> {
    pub backend: Option<&'a dyn CompactionBackend>,
    pub metadata: &'a MetadataMap,
}

#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    async fn apply(
        &self,
        request: CompactionRequest,
        ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError>;
}

#[derive(Clone)]
pub struct CompactionConfig {
    pub trigger: Arc<dyn CompactionTrigger>,
    pub strategy: Arc<dyn CompactionStrategy>,
    pub backend: Option<Arc<dyn CompactionBackend>>,
    pub metadata: MetadataMap,
}

impl CompactionConfig {
    pub fn new(
        trigger: impl CompactionTrigger + 'static,
        strategy: impl CompactionStrategy + 'static,
    ) -> Self {
        Self {
            trigger: Arc::new(trigger),
            strategy: Arc::new(strategy),
            backend: None,
            metadata: MetadataMap::new(),
        }
    }

    pub fn with_backend(mut self, backend: impl CompactionBackend + 'static) -> Self {
        self.backend = Some(Arc::new(backend));
        self
    }

    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
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

#[derive(Clone, Default)]
pub struct CompactionPipeline {
    strategies: Vec<Arc<dyn CompactionStrategy>>,
}

impl CompactionPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_strategy(mut self, strategy: impl CompactionStrategy + 'static) -> Self {
        self.strategies.push(Arc::new(strategy));
        self
    }
}

#[async_trait]
impl CompactionStrategy for CompactionPipeline {
    async fn apply(
        &self,
        mut request: CompactionRequest,
        ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError> {
        let mut replaced_items = 0;
        let mut metadata = MetadataMap::new();

        for strategy in &self.strategies {
            let result = strategy.apply(request.clone(), ctx).await?;
            request.transcript = result.transcript;
            replaced_items += result.replaced_items;
            metadata.extend(result.metadata);
        }

        Ok(CompactionResult {
            transcript: request.transcript,
            replaced_items,
            metadata,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropReasoningStrategy {
    drop_empty_items: bool,
}

impl DropReasoningStrategy {
    pub fn new() -> Self {
        Self {
            drop_empty_items: true,
        }
    }

    pub fn drop_empty_items(mut self, value: bool) -> Self {
        self.drop_empty_items = value;
        self
    }
}

#[async_trait]
impl CompactionStrategy for DropReasoningStrategy {
    async fn apply(
        &self,
        request: CompactionRequest,
        _ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError> {
        let mut transcript = Vec::with_capacity(request.transcript.len());
        let mut replaced_items = 0;

        for mut item in request.transcript {
            let original_len = item.parts.len();
            item.parts
                .retain(|part| !matches!(part, Part::Reasoning(_)));
            let changed = item.parts.len() != original_len;
            if item.parts.is_empty() && self.drop_empty_items {
                if changed {
                    replaced_items += 1;
                }
                continue;
            }
            if changed {
                replaced_items += 1;
            }
            transcript.push(item);
        }

        Ok(CompactionResult {
            transcript,
            replaced_items,
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropFailedToolResultsStrategy {
    drop_empty_items: bool,
}

impl DropFailedToolResultsStrategy {
    pub fn new() -> Self {
        Self {
            drop_empty_items: true,
        }
    }

    pub fn drop_empty_items(mut self, value: bool) -> Self {
        self.drop_empty_items = value;
        self
    }
}

#[async_trait]
impl CompactionStrategy for DropFailedToolResultsStrategy {
    async fn apply(
        &self,
        request: CompactionRequest,
        _ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError> {
        let mut transcript = Vec::with_capacity(request.transcript.len());
        let mut replaced_items = 0;

        for mut item in request.transcript {
            let original_len = item.parts.len();
            item.parts.retain(|part| {
                !matches!(
                    part,
                    Part::ToolResult(result) if result.is_error
                )
            });
            let changed = item.parts.len() != original_len;
            if item.parts.is_empty() && self.drop_empty_items {
                if changed {
                    replaced_items += 1;
                }
                continue;
            }
            if changed {
                replaced_items += 1;
            }
            transcript.push(item);
        }

        Ok(CompactionResult {
            transcript,
            replaced_items,
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeepRecentStrategy {
    keep_last: usize,
    preserve_kinds: BTreeSet<ItemKind>,
}

impl KeepRecentStrategy {
    pub fn new(keep_last: usize) -> Self {
        Self {
            keep_last,
            preserve_kinds: BTreeSet::new(),
        }
    }

    pub fn preserve_kind(mut self, kind: ItemKind) -> Self {
        self.preserve_kinds.insert(kind);
        self
    }
}

#[async_trait]
impl CompactionStrategy for KeepRecentStrategy {
    async fn apply(
        &self,
        request: CompactionRequest,
        _ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError> {
        let removable = removable_indices(&request.transcript, &self.preserve_kinds);
        if removable.len() <= self.keep_last {
            return Ok(CompactionResult {
                transcript: request.transcript,
                replaced_items: 0,
                metadata: MetadataMap::new(),
            });
        }

        let keep_indices = removable
            .iter()
            .skip(removable.len() - self.keep_last)
            .copied()
            .collect::<BTreeSet<_>>();
        let transcript = request
            .transcript
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| {
                (self.preserve_kinds.contains(&item.kind) || keep_indices.contains(&index))
                    .then_some(item)
            })
            .collect::<Vec<_>>();

        Ok(CompactionResult {
            transcript,
            replaced_items: removable.len() - self.keep_last,
            metadata: MetadataMap::new(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummarizeOlderStrategy {
    keep_last: usize,
    preserve_kinds: BTreeSet<ItemKind>,
}

impl SummarizeOlderStrategy {
    pub fn new(keep_last: usize) -> Self {
        Self {
            keep_last,
            preserve_kinds: BTreeSet::new(),
        }
    }

    pub fn preserve_kind(mut self, kind: ItemKind) -> Self {
        self.preserve_kinds.insert(kind);
        self
    }
}

#[async_trait]
impl CompactionStrategy for SummarizeOlderStrategy {
    async fn apply(
        &self,
        request: CompactionRequest,
        ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError> {
        let Some(backend) = ctx.backend else {
            return Err(CompactionError::MissingBackend(
                "summarize strategy requires a compaction backend".into(),
            ));
        };

        let removable = removable_indices(&request.transcript, &self.preserve_kinds);
        if removable.len() <= self.keep_last {
            return Ok(CompactionResult {
                transcript: request.transcript,
                replaced_items: 0,
                metadata: MetadataMap::new(),
            });
        }

        let summary_indices = removable[..removable.len() - self.keep_last].to_vec();
        let first_summary_index = summary_indices[0];
        let summary_index_set = summary_indices.iter().copied().collect::<BTreeSet<_>>();
        let summary_items = summary_indices
            .iter()
            .map(|index| request.transcript[*index].clone())
            .collect::<Vec<_>>();
        let summary = backend
            .summarize(SummaryRequest {
                session_id: request.session_id.clone(),
                turn_id: request.turn_id.clone(),
                items: summary_items,
                reason: request.reason.clone(),
                metadata: request.metadata.clone(),
            })
            .await?;

        let mut transcript = Vec::new();
        let mut inserted_summary = false;
        let mut summary_output = Some(summary.items);
        for (index, item) in request.transcript.into_iter().enumerate() {
            if summary_index_set.contains(&index) {
                if !inserted_summary && index == first_summary_index {
                    transcript.extend(summary_output.take().unwrap_or_default());
                    inserted_summary = true;
                }
                continue;
            }
            transcript.push(item);
        }

        Ok(CompactionResult {
            transcript,
            replaced_items: summary_indices.len(),
            metadata: summary.metadata,
        })
    }
}

fn removable_indices(transcript: &[Item], preserve_kinds: &BTreeSet<ItemKind>) -> Vec<usize> {
    transcript
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (!preserve_kinds.contains(&item.kind)).then_some(index))
        .collect()
}

#[derive(Debug, Error)]
pub enum CompactionError {
    #[error("missing compaction backend: {0}")]
    MissingBackend(String),
    #[error("compaction failed: {0}")]
    Failed(String),
}

#[cfg(test)]
mod tests {
    use agentkit_core::{Part, TextPart, ToolOutput, ToolResultPart};

    use super::*;

    fn user_item(text: &str) -> Item {
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

    fn assistant_with_reasoning() -> Item {
        Item {
            id: None,
            kind: ItemKind::Assistant,
            parts: vec![
                Part::Reasoning(agentkit_core::ReasoningPart {
                    summary: Some("think".into()),
                    data: None,
                    redacted: false,
                    metadata: MetadataMap::new(),
                }),
                Part::Text(TextPart {
                    text: "answer".into(),
                    metadata: MetadataMap::new(),
                }),
            ],
            metadata: MetadataMap::new(),
        }
    }

    fn failed_tool_item() -> Item {
        Item {
            id: None,
            kind: ItemKind::Tool,
            parts: vec![Part::ToolResult(ToolResultPart {
                call_id: "call-1".into(),
                output: ToolOutput::Text("failed".into()),
                is_error: true,
                metadata: MetadataMap::new(),
            })],
            metadata: MetadataMap::new(),
        }
    }

    #[test]
    fn item_count_trigger_fires_after_limit() {
        let trigger = ItemCountTrigger::new(2);
        let transcript = vec![user_item("a"), user_item("b"), user_item("c")];
        assert_eq!(
            trigger.should_compact(&SessionId::new("s"), None, &transcript),
            Some(CompactionReason::TranscriptTooLong)
        );
    }

    #[tokio::test]
    async fn pipeline_applies_local_strategies_in_order() {
        let request = CompactionRequest {
            session_id: "s".into(),
            turn_id: None,
            transcript: vec![
                user_item("a"),
                assistant_with_reasoning(),
                failed_tool_item(),
                user_item("b"),
                user_item("c"),
            ],
            reason: CompactionReason::TranscriptTooLong,
            metadata: MetadataMap::new(),
        };
        let pipeline = CompactionPipeline::new()
            .with_strategy(DropReasoningStrategy::new())
            .with_strategy(DropFailedToolResultsStrategy::new())
            .with_strategy(
                KeepRecentStrategy::new(2)
                    .preserve_kind(ItemKind::System)
                    .preserve_kind(ItemKind::Context),
            );
        let mut ctx = CompactionContext {
            backend: None,
            metadata: &MetadataMap::new(),
        };

        let result = pipeline.apply(request, &mut ctx).await.unwrap();
        assert_eq!(result.transcript.len(), 2);
        assert!(result.replaced_items >= 2);
        assert!(result.transcript.iter().all(|item| {
            item.parts
                .iter()
                .all(|part| !matches!(part, Part::Reasoning(_)))
        }));
    }

    struct FakeBackend;

    #[async_trait]
    impl CompactionBackend for FakeBackend {
        async fn summarize(
            &self,
            request: SummaryRequest,
        ) -> Result<SummaryResult, CompactionError> {
            Ok(SummaryResult {
                items: vec![Item {
                    id: None,
                    kind: ItemKind::Context,
                    parts: vec![Part::Text(TextPart {
                        text: format!("summary of {} items", request.items.len()),
                        metadata: MetadataMap::new(),
                    })],
                    metadata: MetadataMap::new(),
                }],
                metadata: MetadataMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn summarize_strategy_uses_backend() {
        let request = CompactionRequest {
            session_id: "s".into(),
            turn_id: None,
            transcript: vec![user_item("a"), user_item("b"), user_item("c")],
            reason: CompactionReason::TranscriptTooLong,
            metadata: MetadataMap::new(),
        };
        let strategy = SummarizeOlderStrategy::new(1);
        let mut ctx = CompactionContext {
            backend: Some(&FakeBackend),
            metadata: &MetadataMap::new(),
        };

        let result = strategy.apply(request, &mut ctx).await.unwrap();
        assert_eq!(result.replaced_items, 2);
        assert_eq!(result.transcript.len(), 2);
        match &result.transcript[0].parts[0] {
            Part::Text(text) => assert_eq!(text.text, "summary of 2 items"),
            other => panic!("unexpected part: {other:?}"),
        }
    }
}
