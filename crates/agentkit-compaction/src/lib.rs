//! Transcript compaction primitives for reducing context size while preserving
//! useful state.
//!
//! This crate provides the building blocks for compacting an agent transcript
//! when it grows too large. The main concepts are:
//!
//! - **Triggers** ([`CompactionTrigger`]) decide *when* compaction should run
//!   (e.g. after a certain item count is exceeded).
//! - **Strategies** ([`CompactionStrategy`]) decide *how* the transcript is
//!   transformed: dropping reasoning, removing failed tool results, keeping
//!   only recent items, or summarising older items via a backend.
//! - **Pipelines** ([`CompactionPipeline`]) chain multiple strategies into a
//!   single pass.
//! - **Backends** ([`CompactionBackend`]) provide provider-backed
//!   summarisation for strategies that need it (e.g.
//!   [`SummarizeOlderStrategy`]).
//!
//! Combine these pieces through [`CompactionConfig`] and hand the config to
//! `agentkit-loop` (or your own runtime) to keep transcripts under control.

use std::collections::BTreeSet;
use std::sync::Arc;

use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TurnCancellation, TurnId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The reason a compaction was triggered.
///
/// Returned by [`CompactionTrigger::should_compact`] and forwarded to
/// strategies so they can adapt their behaviour.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionReason {
    /// The transcript exceeded a configured item count.
    TranscriptTooLong,
    /// A caller explicitly requested compaction.
    Manual,
    /// An application-specific reason described by the inner string.
    Custom(String),
}

/// Input to a [`CompactionStrategy`].
///
/// Carries the full transcript together with session context so that
/// strategies can decide which items to keep, drop, or summarise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionRequest {
    /// Identifier for the current session.
    pub session_id: SessionId,
    /// Identifier for the turn that triggered compaction, if any.
    pub turn_id: Option<TurnId>,
    /// The transcript to compact.
    pub transcript: Vec<Item>,
    /// Why compaction was triggered.
    pub reason: CompactionReason,
    /// Arbitrary key-value metadata forwarded through the pipeline.
    pub metadata: MetadataMap,
}

/// Output of a [`CompactionStrategy`].
///
/// Contains the compacted transcript along with metadata about what changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionResult {
    /// The compacted transcript.
    pub transcript: Vec<Item>,
    /// How many items were removed or replaced during compaction.
    pub replaced_items: usize,
    /// Metadata produced by the strategy (e.g. summarisation statistics).
    pub metadata: MetadataMap,
}

/// Request sent to a [`CompactionBackend`] asking it to summarise a set of
/// transcript items.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryRequest {
    /// Identifier for the current session.
    pub session_id: SessionId,
    /// Identifier for the turn that triggered compaction, if any.
    pub turn_id: Option<TurnId>,
    /// The transcript items to summarise.
    pub items: Vec<Item>,
    /// Why compaction was triggered.
    pub reason: CompactionReason,
    /// Arbitrary key-value metadata forwarded from the pipeline.
    pub metadata: MetadataMap,
}

/// Response from a [`CompactionBackend`] containing the summarised items.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryResult {
    /// The summary items that replace the originals in the transcript.
    pub items: Vec<Item>,
    /// Metadata produced during summarisation (e.g. token counts).
    pub metadata: MetadataMap,
}

/// Decides whether compaction should run for a given transcript.
///
/// Implement this trait to create custom triggers. The built-in
/// [`ItemCountTrigger`] fires when the transcript exceeds a fixed item count.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::{CompactionReason, CompactionTrigger, ItemCountTrigger};
/// use agentkit_core::SessionId;
///
/// let trigger = ItemCountTrigger::new(32);
/// let items = Vec::new();
/// assert!(trigger.should_compact(&SessionId::new("s"), None, &items).is_none());
/// ```
pub trait CompactionTrigger: Send + Sync {
    /// Returns `Some(reason)` if compaction should run, `None` otherwise.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The current session identifier.
    /// * `turn_id` - The turn that is being evaluated, if any.
    /// * `transcript` - The full transcript to inspect.
    fn should_compact(
        &self,
        session_id: &SessionId,
        turn_id: Option<&TurnId>,
        transcript: &[Item],
    ) -> Option<CompactionReason>;
}

/// Provider-backed summarisation service.
///
/// Implement this trait to connect a language model (or any other
/// summarisation service) so that strategies like [`SummarizeOlderStrategy`]
/// can condense older transcript items into a shorter summary.
///
/// # Errors
///
/// Implementations should return [`CompactionError::Failed`] when
/// summarisation cannot be completed, or [`CompactionError::Cancelled`] when
/// the cancellation token is signalled.
#[async_trait]
pub trait CompactionBackend: Send + Sync {
    /// Summarise the given items into a shorter set of replacement items.
    ///
    /// # Arguments
    ///
    /// * `request` - The items to summarise together with session context.
    /// * `cancellation` - An optional cancellation token; implementations
    ///   should check this periodically and bail early when cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError`] on failure or cancellation.
    async fn summarize(
        &self,
        request: SummaryRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<SummaryResult, CompactionError>;
}

/// Runtime context passed to each [`CompactionStrategy`] during execution.
///
/// Provides access to an optional [`CompactionBackend`] (needed by
/// [`SummarizeOlderStrategy`]), shared metadata, and a cancellation token
/// that strategies should respect.
pub struct CompactionContext<'a> {
    /// An optional backend for strategies that need to call an external
    /// summarisation service.
    pub backend: Option<&'a dyn CompactionBackend>,
    /// Shared metadata available to all strategies in the pipeline.
    pub metadata: &'a MetadataMap,
    /// Cancellation token; strategies should check this and return
    /// [`CompactionError::Cancelled`] when signalled.
    pub cancellation: Option<TurnCancellation>,
}

/// A single compaction step that transforms a transcript.
///
/// Strategies are the core abstraction in this crate. Each strategy receives
/// the transcript inside a [`CompactionRequest`] and returns a
/// [`CompactionResult`] with the (possibly shorter) transcript.
///
/// Built-in strategies:
///
/// | Strategy | What it does |
/// |---|---|
/// | [`DropReasoningStrategy`] | Strips reasoning parts from items |
/// | [`DropFailedToolResultsStrategy`] | Removes errored tool results |
/// | [`KeepRecentStrategy`] | Keeps only the N most recent removable items |
/// | [`SummarizeOlderStrategy`] | Replaces older items with a backend-generated summary |
///
/// Use [`CompactionPipeline`] to chain multiple strategies together.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::DropReasoningStrategy;
///
/// // Strategies are composable via CompactionPipeline
/// let strategy = DropReasoningStrategy::new();
/// ```
#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// Apply this strategy to the transcript in `request`.
    ///
    /// # Arguments
    ///
    /// * `request` - The transcript and session context to compact.
    /// * `ctx` - Runtime context providing the backend, metadata, and
    ///   cancellation token.
    ///
    /// # Errors
    ///
    /// Returns [`CompactionError`] on failure or cancellation.
    async fn apply(
        &self,
        request: CompactionRequest,
        ctx: &mut CompactionContext<'_>,
    ) -> Result<CompactionResult, CompactionError>;
}

/// Top-level configuration that bundles a trigger, strategy, and optional
/// backend into a single value you can hand to `agentkit-loop`.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::{
///     CompactionConfig, CompactionPipeline, DropReasoningStrategy,
///     ItemCountTrigger, KeepRecentStrategy,
/// };
/// use agentkit_core::ItemKind;
///
/// let config = CompactionConfig::new(
///     ItemCountTrigger::new(32),
///     CompactionPipeline::new()
///         .with_strategy(DropReasoningStrategy::new())
///         .with_strategy(
///             KeepRecentStrategy::new(24)
///                 .preserve_kind(ItemKind::System)
///                 .preserve_kind(ItemKind::Context),
///         ),
/// );
/// ```
#[derive(Clone)]
pub struct CompactionConfig {
    /// The trigger that decides when compaction should run.
    pub trigger: Arc<dyn CompactionTrigger>,
    /// The strategy (or pipeline of strategies) to execute.
    pub strategy: Arc<dyn CompactionStrategy>,
    /// An optional backend for strategies that need summarisation.
    pub backend: Option<Arc<dyn CompactionBackend>>,
    /// Metadata forwarded to every strategy invocation.
    pub metadata: MetadataMap,
}

impl CompactionConfig {
    /// Create a new configuration with the given trigger and strategy.
    ///
    /// The backend defaults to `None`. Use [`with_backend`](Self::with_backend)
    /// to attach one when your pipeline includes [`SummarizeOlderStrategy`].
    ///
    /// # Arguments
    ///
    /// * `trigger` - Decides when compaction should fire.
    /// * `strategy` - The strategy (or [`CompactionPipeline`]) to execute.
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

    /// Attach a [`CompactionBackend`] for strategies that require
    /// summarisation (e.g. [`SummarizeOlderStrategy`]).
    pub fn with_backend(mut self, backend: impl CompactionBackend + 'static) -> Self {
        self.backend = Some(Arc::new(backend));
        self
    }

    /// Set metadata that will be forwarded to every strategy invocation.
    pub fn with_metadata(mut self, metadata: MetadataMap) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A [`CompactionTrigger`] that fires when the transcript exceeds a fixed
/// number of items.
///
/// This is the simplest built-in trigger: once `transcript.len()` is greater
/// than `max_items`, it returns
/// [`CompactionReason::TranscriptTooLong`].
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::{CompactionTrigger, ItemCountTrigger};
/// use agentkit_core::SessionId;
///
/// let trigger = ItemCountTrigger::new(100);
/// // An empty transcript does not trigger compaction.
/// assert!(trigger.should_compact(&SessionId::new("s"), None, &[]).is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemCountTrigger {
    /// Maximum number of items allowed before compaction fires.
    pub max_items: usize,
}

impl ItemCountTrigger {
    /// Create a trigger that fires when the transcript has more than
    /// `max_items` items.
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

/// An ordered sequence of [`CompactionStrategy`] steps executed one after
/// another.
///
/// Each strategy receives the transcript produced by the previous one,
/// creating a pipeline effect. The pipeline itself implements
/// [`CompactionStrategy`], so it can be nested or used anywhere a single
/// strategy is expected.
///
/// The pipeline checks the [`CompactionContext::cancellation`] token between
/// steps and returns [`CompactionError::Cancelled`] early if cancellation is
/// signalled.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::{
///     CompactionPipeline, DropFailedToolResultsStrategy,
///     DropReasoningStrategy, KeepRecentStrategy,
/// };
/// use agentkit_core::ItemKind;
///
/// let pipeline = CompactionPipeline::new()
///     .with_strategy(DropReasoningStrategy::new())
///     .with_strategy(DropFailedToolResultsStrategy::new())
///     .with_strategy(
///         KeepRecentStrategy::new(24)
///             .preserve_kind(ItemKind::System)
///             .preserve_kind(ItemKind::Context),
///     );
/// ```
#[derive(Clone, Default)]
pub struct CompactionPipeline {
    strategies: Vec<Arc<dyn CompactionStrategy>>,
}

impl CompactionPipeline {
    /// Create an empty pipeline with no strategies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a strategy to the end of the pipeline.
    ///
    /// Strategies run in the order they are added.
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
            if ctx
                .cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                return Err(CompactionError::Cancelled);
            }
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

/// Strategy that removes [`Part::Reasoning`] parts from every item.
///
/// Reasoning parts contain chain-of-thought content that is useful during
/// generation but rarely needed once the answer has been produced. Stripping
/// them reduces transcript size without losing user-visible content.
///
/// When `drop_empty_items` is `true` (the default), items that become empty
/// after reasoning removal are dropped entirely.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::DropReasoningStrategy;
///
/// let strategy = DropReasoningStrategy::new();
///
/// // Keep items that become empty after stripping reasoning:
/// let keep_empties = DropReasoningStrategy::new().drop_empty_items(false);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropReasoningStrategy {
    drop_empty_items: bool,
}

impl DropReasoningStrategy {
    /// Create a new strategy that drops reasoning parts and removes items
    /// that become empty as a result.
    pub fn new() -> Self {
        Self {
            drop_empty_items: true,
        }
    }

    /// Control whether items that become empty after reasoning removal are
    /// dropped from the transcript.
    ///
    /// Defaults to `true`.
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

/// Strategy that removes [`Part::ToolResult`] parts where `is_error` is
/// `true`.
///
/// Failed tool invocations clutter the transcript and can confuse the model
/// on subsequent turns. This strategy strips those results while leaving
/// successful tool output intact.
///
/// When `drop_empty_items` is `true` (the default), items that become empty
/// after removal are dropped entirely.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::DropFailedToolResultsStrategy;
///
/// let strategy = DropFailedToolResultsStrategy::new();
///
/// // Keep items that become empty after stripping failed results:
/// let keep_empties = DropFailedToolResultsStrategy::new().drop_empty_items(false);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropFailedToolResultsStrategy {
    drop_empty_items: bool,
}

impl DropFailedToolResultsStrategy {
    /// Create a new strategy that drops failed tool results and removes
    /// items that become empty as a result.
    pub fn new() -> Self {
        Self {
            drop_empty_items: true,
        }
    }

    /// Control whether items that become empty after failed-result removal
    /// are dropped from the transcript.
    ///
    /// Defaults to `true`.
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

/// Strategy that keeps only the `N` most recent removable items and drops
/// the rest.
///
/// Items whose [`ItemKind`] is in the `preserve_kinds` set are always
/// retained regardless of their position. This lets you protect system
/// prompts and context items while trimming older conversation turns.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::KeepRecentStrategy;
/// use agentkit_core::ItemKind;
///
/// let strategy = KeepRecentStrategy::new(16)
///     .preserve_kind(ItemKind::System)
///     .preserve_kind(ItemKind::Context);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeepRecentStrategy {
    keep_last: usize,
    preserve_kinds: BTreeSet<ItemKind>,
}

impl KeepRecentStrategy {
    /// Create a strategy that keeps the last `keep_last` removable items.
    pub fn new(keep_last: usize) -> Self {
        Self {
            keep_last,
            preserve_kinds: BTreeSet::new(),
        }
    }

    /// Mark an [`ItemKind`] as preserved so that items of this kind are never
    /// dropped, regardless of their position in the transcript.
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

/// Strategy that replaces older transcript items with a backend-generated
/// summary.
///
/// The most recent `keep_last` removable items are kept verbatim. Everything
/// older (excluding items with a preserved [`ItemKind`]) is sent to the
/// configured [`CompactionBackend`] for summarisation. The summary items
/// replace the originals at their position in the transcript.
///
/// This strategy requires a backend. If [`CompactionContext::backend`] is
/// `None`, [`CompactionError::MissingBackend`] is returned.
///
/// # Example
///
/// ```rust
/// use agentkit_compaction::SummarizeOlderStrategy;
/// use agentkit_core::ItemKind;
///
/// let strategy = SummarizeOlderStrategy::new(8)
///     .preserve_kind(ItemKind::System);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummarizeOlderStrategy {
    keep_last: usize,
    preserve_kinds: BTreeSet<ItemKind>,
}

impl SummarizeOlderStrategy {
    /// Create a strategy that keeps the last `keep_last` removable items and
    /// summarises everything older.
    pub fn new(keep_last: usize) -> Self {
        Self {
            keep_last,
            preserve_kinds: BTreeSet::new(),
        }
    }

    /// Mark an [`ItemKind`] as preserved so that items of this kind are never
    /// summarised, regardless of their position in the transcript.
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
            .summarize(
                SummaryRequest {
                    session_id: request.session_id.clone(),
                    turn_id: request.turn_id.clone(),
                    items: summary_items,
                    reason: request.reason.clone(),
                    metadata: request.metadata.clone(),
                },
                ctx.cancellation.clone(),
            )
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

/// Errors that can occur during compaction.
#[derive(Debug, Error)]
pub enum CompactionError {
    /// The operation was cancelled via the [`TurnCancellation`] token.
    #[error("compaction cancelled")]
    Cancelled,
    /// A strategy that requires a [`CompactionBackend`] was invoked without
    /// one.
    #[error("missing compaction backend: {0}")]
    MissingBackend(String),
    /// A catch-all for other failures (e.g. backend errors).
    #[error("compaction failed: {0}")]
    Failed(String),
}

#[cfg(test)]
mod tests {
    use agentkit_core::{CancellationController, Part, TextPart, ToolOutput, ToolResultPart};

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
            cancellation: None,
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
            _cancellation: Option<TurnCancellation>,
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
            cancellation: None,
        };

        let result = strategy.apply(request, &mut ctx).await.unwrap();
        assert_eq!(result.replaced_items, 2);
        assert_eq!(result.transcript.len(), 2);
        match &result.transcript[0].parts[0] {
            Part::Text(text) => assert_eq!(text.text, "summary of 2 items"),
            other => panic!("unexpected part: {other:?}"),
        }
    }

    #[tokio::test]
    async fn pipeline_stops_when_cancelled() {
        let controller = CancellationController::new();
        let checkpoint = controller.handle().checkpoint();
        controller.interrupt();
        let request = CompactionRequest {
            session_id: "s".into(),
            turn_id: None,
            transcript: vec![user_item("a"), user_item("b"), user_item("c")],
            reason: CompactionReason::TranscriptTooLong,
            metadata: MetadataMap::new(),
        };
        let pipeline = CompactionPipeline::new().with_strategy(DropReasoningStrategy::new());
        let mut ctx = CompactionContext {
            backend: None,
            metadata: &MetadataMap::new(),
            cancellation: Some(checkpoint),
        };

        let error = pipeline.apply(request, &mut ctx).await.unwrap_err();
        assert!(matches!(error, CompactionError::Cancelled));
    }
}
