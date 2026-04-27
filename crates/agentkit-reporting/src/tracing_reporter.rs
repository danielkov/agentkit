//! Tracing-based reporter.
//!
//! [`TracingReporter`] converts [`AgentEvent`]s into [`tracing`] events,
//! bridging the observer system with the Rust ecosystem's structured logging /
//! distributed tracing infrastructure.
//!
//! This module is only available when the `tracing` feature is enabled.

use agentkit_loop::{AgentEvent, LoopObserver};

/// Reporter that emits [`tracing`] events for every [`AgentEvent`].
///
/// Event levels are chosen to match typical subscriber filter defaults:
///
/// | Agent event | Tracing level |
/// |---|---|
/// | `RunStarted`, `TurnStarted`, `TurnFinished` | `INFO` |
/// | `ToolCallRequested`, `ApprovalRequired/Resolved`, `AuthRequired/Resolved` | `INFO` |
/// | `InputAccepted`, `UsageUpdated`, `CompactionStarted/Finished` | `DEBUG` |
/// | `ContentDelta` | `TRACE` |
/// | `Warning` | `WARN` |
/// | `RunFailed` | `ERROR` |
///
/// All events are emitted under the target `"agentkit"`.
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::TracingReporter;
///
/// let reporter = TracingReporter::new();
/// ```
#[derive(Default)]
pub struct TracingReporter {
    _private: (),
}

impl TracingReporter {
    /// Creates a new `TracingReporter`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl LoopObserver for TracingReporter {
    fn handle_event(&mut self, event: AgentEvent) {
        match &event {
            AgentEvent::RunStarted { session_id } => {
                tracing::info!(target: "agentkit", session_id = %session_id, "agent run started");
            }
            AgentEvent::TurnStarted {
                session_id,
                turn_id,
            } => {
                tracing::info!(target: "agentkit", session_id = %session_id, turn_id = %turn_id, "turn started");
            }
            AgentEvent::InputAccepted { session_id, items } => {
                tracing::debug!(target: "agentkit", session_id = %session_id, item_count = items.len(), "input accepted");
            }
            AgentEvent::ContentDelta(delta) => {
                tracing::trace!(target: "agentkit", ?delta, "content delta");
            }
            AgentEvent::ToolCallRequested(call) => {
                tracing::info!(target: "agentkit", tool = %call.name, "tool call requested");
            }
            AgentEvent::ToolResultReceived(result) => {
                tracing::info!(
                    target: "agentkit",
                    call_id = %result.call_id,
                    is_error = result.is_error,
                    "tool result received"
                );
            }
            AgentEvent::ApprovalRequired(request) => {
                tracing::info!(target: "agentkit", summary = %request.summary, reason = ?request.reason, "approval required");
            }
            AgentEvent::ApprovalResolved { approved } => {
                tracing::info!(target: "agentkit", approved, "approval resolved");
            }
            AgentEvent::ToolCatalogChanged(event) => {
                tracing::info!(
                    target: "agentkit",
                    source = %event.source,
                    added = event.added.len(),
                    removed = event.removed.len(),
                    changed = event.changed.len(),
                    "tool catalog changed"
                );
            }
            AgentEvent::CompactionStarted {
                session_id,
                turn_id,
                reason,
            } => {
                let turn = turn_id.as_ref().map(|t| t.to_string()).unwrap_or_default();
                tracing::debug!(
                    target: "agentkit",
                    session_id = %session_id,
                    turn_id = %turn,
                    reason = ?reason,
                    "compaction started"
                );
            }
            AgentEvent::CompactionFinished {
                session_id,
                turn_id,
                replaced_items,
                transcript_len,
                ..
            } => {
                let turn = turn_id.as_ref().map(|t| t.to_string()).unwrap_or_default();
                tracing::debug!(
                    target: "agentkit",
                    session_id = %session_id,
                    turn_id = %turn,
                    replaced_items,
                    transcript_len,
                    "compaction finished"
                );
            }
            AgentEvent::UsageUpdated(usage) => {
                if let Some(tokens) = &usage.tokens {
                    tracing::debug!(
                        target: "agentkit",
                        input_tokens = tokens.input_tokens,
                        output_tokens = tokens.output_tokens,
                        "usage updated"
                    );
                }
            }
            AgentEvent::Warning { message } => {
                tracing::warn!(target: "agentkit", message, "agent warning");
            }
            AgentEvent::RunFailed { message } => {
                tracing::error!(target: "agentkit", message, "agent run failed");
            }
            AgentEvent::TurnFinished(result) => {
                tracing::info!(
                    target: "agentkit",
                    finish_reason = ?result.finish_reason,
                    item_count = result.items.len(),
                    "turn finished"
                );
            }
        }
    }
}
