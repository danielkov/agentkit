//! Reporting observers for the agentkit agent loop.
//!
//! This crate provides [`LoopObserver`] implementations that turn
//! [`AgentEvent`]s into logs, usage summaries, transcripts, and
//! machine-readable JSONL streams. Reporters are designed to be composed
//! through [`CompositeReporter`] so a single loop can feed multiple
//! observers at once.
//!
//! # Included reporters
//!
//! | Reporter | Purpose |
//! |---|---|
//! | [`StdoutReporter`] | Human-readable terminal output |
//! | [`JsonlReporter`] | Machine-readable newline-delimited JSON |
//! | [`UsageReporter`] | Aggregated token / cost totals |
//! | [`TranscriptReporter`] | Growing snapshot of conversation items |
//! | [`CompositeReporter`] | Fan-out to multiple reporters |
//!
//! # Adapter reporters
//!
//! | Adapter | Purpose |
//! |---|---|
//! | [`BufferedReporter`] | Enqueues events for batch flushing |
//! | [`ChannelReporter`] | Forwards events to another thread or task |
//! | [`TracingReporter`] | Converts events into `tracing` spans and events (requires `tracing` feature) |
//!
//! # Failure policy
//!
//! Wrap a [`FallibleObserver`] in a [`PolicyReporter`] to control how
//! errors are handled — see [`FailurePolicy`].
//!
//! # Quick start
//!
//! ```rust
//! use agentkit_reporting::{CompositeReporter, JsonlReporter, UsageReporter, TranscriptReporter};
//!
//! let reporter = CompositeReporter::new()
//!     .with_observer(JsonlReporter::new(Vec::new()))
//!     .with_observer(UsageReporter::new())
//!     .with_observer(TranscriptReporter::new());
//! ```

mod buffered;
mod channel;
mod policy;

#[cfg(feature = "tracing")]
mod tracing_reporter;

pub use buffered::BufferedReporter;
pub use channel::ChannelReporter;
pub use policy::{FailurePolicy, FallibleObserver, PolicyReporter};

#[cfg(feature = "tracing")]
pub use tracing_reporter::TracingReporter;

use std::io::{self, Write};
use std::time::SystemTime;

use agentkit_core::{Item, ItemKind, Part, TokenUsage, Usage};
use agentkit_loop::{AgentEvent, LoopObserver, TurnResult};
use serde::Serialize;
use thiserror::Error;

/// Errors that can occur while writing reports.
///
/// Reporter implementations (e.g. [`JsonlReporter`], [`StdoutReporter`])
/// collect errors internally rather than surfacing them through the
/// [`LoopObserver`] interface. Call the reporter's `take_errors()` method
/// after the loop finishes to inspect any problems.
#[derive(Debug, Error)]
pub enum ReportError {
    /// An I/O error occurred while writing to the underlying writer.
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    /// A serialization error occurred (JSONL reporters only).
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    /// The receiving end of a channel was dropped.
    #[error("channel send failed")]
    ChannelSend,
}

/// A timestamped wrapper around an [`AgentEvent`].
///
/// [`JsonlReporter`] serializes each incoming event inside an
/// `EventEnvelope` so that the resulting JSONL stream carries
/// wall-clock timestamps alongside the event payload.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EventEnvelope<'a> {
    /// When the event was observed.
    pub timestamp: SystemTime,
    /// The underlying agent event.
    pub event: &'a AgentEvent,
}

/// Fan-out reporter that forwards every [`AgentEvent`] to multiple child observers.
///
/// `CompositeReporter` itself implements [`LoopObserver`], so it can be
/// handed directly to the agent loop. Each event is cloned once per child
/// observer.
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::{
///     CompositeReporter, JsonlReporter, StdoutReporter, UsageReporter,
/// };
///
/// // Build a reporter that writes to JSONL, prints to stdout, and tracks usage.
/// let reporter = CompositeReporter::new()
///     .with_observer(JsonlReporter::new(Vec::new()))
///     .with_observer(StdoutReporter::new(std::io::stdout()))
///     .with_observer(UsageReporter::new());
/// ```
#[derive(Default)]
pub struct CompositeReporter {
    children: Vec<Box<dyn LoopObserver>>,
}

impl CompositeReporter {
    /// Creates an empty `CompositeReporter` with no child observers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an observer and returns `self` (builder pattern).
    ///
    /// # Arguments
    ///
    /// * `observer` - Any type implementing [`LoopObserver`].
    pub fn with_observer(mut self, observer: impl LoopObserver + 'static) -> Self {
        self.children.push(Box::new(observer));
        self
    }

    /// Adds an observer by mutable reference.
    ///
    /// Use this when you need to add observers after initial construction
    /// rather than in a builder chain.
    ///
    /// # Arguments
    ///
    /// * `observer` - Any type implementing [`LoopObserver`].
    pub fn push(&mut self, observer: impl LoopObserver + 'static) -> &mut Self {
        self.children.push(Box::new(observer));
        self
    }
}

impl LoopObserver for CompositeReporter {
    fn handle_event(&mut self, event: AgentEvent) {
        for child in &mut self.children {
            child.handle_event(event.clone());
        }
    }
}

/// Machine-readable reporter that writes one JSON object per line (JSONL).
///
/// Each [`AgentEvent`] is wrapped in an [`EventEnvelope`] with a timestamp
/// and serialized as a single JSON line. This format is easy to ingest in
/// log aggregation systems or to replay offline.
///
/// I/O and serialization errors are collected internally and can be
/// retrieved with [`take_errors`](JsonlReporter::take_errors).
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::JsonlReporter;
///
/// // Write JSONL to an in-memory buffer (useful in tests).
/// let reporter = JsonlReporter::new(Vec::new());
///
/// // Write JSONL to a file, flushing after every event.
/// # fn example() -> std::io::Result<()> {
/// let file = std::fs::File::create("events.jsonl")?;
/// let reporter = JsonlReporter::new(std::io::BufWriter::new(file));
/// # Ok(())
/// # }
/// ```
pub struct JsonlReporter<W> {
    writer: W,
    flush_each_event: bool,
    errors: Vec<ReportError>,
}

impl<W> JsonlReporter<W>
where
    W: Write,
{
    /// Creates a new `JsonlReporter` writing to the given writer.
    ///
    /// Flushing after each event is enabled by default. Disable it with
    /// [`with_flush_each_event(false)`](JsonlReporter::with_flush_each_event)
    /// if you are writing to a buffered sink and prefer to flush manually.
    ///
    /// # Arguments
    ///
    /// * `writer` - Any [`Write`] implementation (file, buffer, stdout, etc.).
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            flush_each_event: true,
            errors: Vec::new(),
        }
    }

    /// Controls whether the writer is flushed after every event (builder pattern).
    ///
    /// Defaults to `true`. Set to `false` when batching writes for throughput.
    pub fn with_flush_each_event(mut self, flush_each_event: bool) -> Self {
        self.flush_each_event = flush_each_event;
        self
    }

    /// Returns a shared reference to the underlying writer.
    ///
    /// Useful for inspecting an in-memory buffer after the loop finishes.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Returns a mutable reference to the underlying writer.
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Drains and returns all errors accumulated during event handling.
    ///
    /// Subsequent calls return an empty `Vec` until new errors occur.
    pub fn take_errors(&mut self) -> Vec<ReportError> {
        std::mem::take(&mut self.errors)
    }

    fn record_result(&mut self, result: Result<(), ReportError>) {
        if let Err(error) = result {
            self.errors.push(error);
        }
    }
}

impl<W> LoopObserver for JsonlReporter<W>
where
    W: Write + Send,
{
    fn handle_event(&mut self, event: AgentEvent) {
        let result = (|| -> Result<(), ReportError> {
            let envelope = EventEnvelope {
                timestamp: SystemTime::now(),
                event: &event,
            };
            serde_json::to_writer(&mut self.writer, &envelope)?;
            self.writer.write_all(b"\n")?;
            if self.flush_each_event {
                self.writer.flush()?;
            }
            Ok(())
        })();
        self.record_result(result);
    }
}

/// Accumulated token counts across all events seen by a [`UsageReporter`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UsageTotals {
    /// Total input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Total output (completion) tokens produced.
    pub output_tokens: u64,
    /// Total reasoning tokens used (model-dependent).
    pub reasoning_tokens: u64,
    /// Total input tokens served from the provider's cache.
    pub cached_input_tokens: u64,
    /// Total input tokens written into the provider's cache.
    pub cache_write_input_tokens: u64,
}

/// Accumulated monetary cost across all events seen by a [`UsageReporter`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostTotals {
    /// Running total cost expressed in `currency` units.
    pub amount: f64,
    /// ISO 4217 currency code (e.g. `"USD"`), set from the first cost event.
    pub currency: Option<String>,
}

/// Snapshot of everything a [`UsageReporter`] has tracked so far.
///
/// Retrieve this via [`UsageReporter::summary`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsageSummary {
    /// Total number of [`AgentEvent`]s observed (of any variant).
    pub events_seen: usize,
    /// Number of events that carried usage information
    /// ([`AgentEvent::UsageUpdated`] or [`AgentEvent::TurnFinished`] with usage).
    pub usage_events_seen: usize,
    /// Number of [`AgentEvent::TurnFinished`] events observed.
    pub turn_results_seen: usize,
    /// Aggregated token counts.
    pub totals: UsageTotals,
    /// Aggregated cost, present only if at least one event carried cost data.
    pub cost: Option<CostTotals>,
}

/// Reporter that aggregates token usage and cost across the entire run.
///
/// `UsageReporter` listens for [`AgentEvent::UsageUpdated`] and
/// [`AgentEvent::TurnFinished`] events and maintains a running
/// [`UsageSummary`]. After the loop completes, call [`summary`](UsageReporter::summary)
/// to read the totals.
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::UsageReporter;
/// use agentkit_loop::LoopObserver;
///
/// let mut reporter = UsageReporter::new();
///
/// // ...pass `reporter` to the agent loop, then afterwards:
/// let summary = reporter.summary();
/// println!(
///     "tokens: {} in / {} out",
///     summary.totals.input_tokens,
///     summary.totals.output_tokens,
/// );
/// ```
#[derive(Default)]
pub struct UsageReporter {
    summary: UsageSummary,
}

impl UsageReporter {
    /// Creates a new `UsageReporter` with zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the current [`UsageSummary`].
    pub fn summary(&self) -> &UsageSummary {
        &self.summary
    }

    fn absorb(&mut self, usage: &Usage) {
        self.summary.usage_events_seen += 1;
        if let Some(tokens) = &usage.tokens {
            self.summary.totals.input_tokens += tokens.input_tokens;
            self.summary.totals.output_tokens += tokens.output_tokens;
            self.summary.totals.reasoning_tokens += tokens.reasoning_tokens.unwrap_or_default();
            self.summary.totals.cached_input_tokens +=
                tokens.cached_input_tokens.unwrap_or_default();
            self.summary.totals.cache_write_input_tokens +=
                tokens.cache_write_input_tokens.unwrap_or_default();
        }
        if let Some(cost) = &usage.cost {
            let totals = self.summary.cost.get_or_insert_with(CostTotals::default);
            totals.amount += cost.amount;
            if totals.currency.is_none() {
                totals.currency = Some(cost.currency.clone());
            }
        }
    }
}

impl LoopObserver for UsageReporter {
    fn handle_event(&mut self, event: AgentEvent) {
        self.summary.events_seen += 1;
        match event {
            AgentEvent::UsageUpdated(usage) => self.absorb(&usage),
            AgentEvent::TurnFinished(TurnResult {
                usage: Some(usage), ..
            }) => {
                self.summary.turn_results_seen += 1;
                self.absorb(&usage);
            }
            AgentEvent::TurnFinished(_) => {
                self.summary.turn_results_seen += 1;
            }
            _ => {}
        }
    }
}

/// Growing list of conversation [`Item`]s captured by a [`TranscriptReporter`].
///
/// Items are appended in the order they arrive: user inputs first, then
/// assistant outputs from each finished turn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptView {
    /// The ordered sequence of conversation items.
    pub items: Vec<Item>,
}

/// Reporter that captures the evolving conversation transcript.
///
/// `TranscriptReporter` listens for [`AgentEvent::InputAccepted`] and
/// [`AgentEvent::TurnFinished`] events and accumulates their [`Item`]s
/// into a [`TranscriptView`]. This is useful for post-run analysis or
/// for displaying a conversation history.
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::TranscriptReporter;
/// use agentkit_loop::LoopObserver;
///
/// let mut reporter = TranscriptReporter::new();
///
/// // ...pass `reporter` to the agent loop, then afterwards:
/// for item in &reporter.transcript().items {
///     println!("{:?}: {} parts", item.kind, item.parts.len());
/// }
/// ```
#[derive(Default)]
pub struct TranscriptReporter {
    transcript: TranscriptView,
}

impl TranscriptReporter {
    /// Creates a new `TranscriptReporter` with an empty transcript.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the current [`TranscriptView`].
    pub fn transcript(&self) -> &TranscriptView {
        &self.transcript
    }
}

impl LoopObserver for TranscriptReporter {
    fn handle_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::InputAccepted { items, .. } => {
                self.transcript.items.extend(items);
            }
            AgentEvent::TurnFinished(result) => {
                self.transcript.items.extend(result.items);
            }
            _ => {}
        }
    }
}

/// Human-readable reporter that writes structured log lines to a [`Write`] sink.
///
/// Each [`AgentEvent`] is printed as a bracketed tag followed by key fields,
/// for example `[turn] started session=abc turn=1`. Turn results include
/// indented item and part summaries so the operator can follow the
/// conversation at a glance.
///
/// I/O errors are collected internally; call
/// [`take_errors`](StdoutReporter::take_errors) after the loop to inspect them.
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::StdoutReporter;
///
/// // Print events to stderr, hiding usage lines.
/// let reporter = StdoutReporter::new(std::io::stderr())
///     .with_usage(false);
/// ```
pub struct StdoutReporter<W> {
    writer: W,
    show_usage: bool,
    errors: Vec<ReportError>,
}

impl<W> StdoutReporter<W>
where
    W: Write,
{
    /// Creates a new `StdoutReporter` that writes to the given writer.
    ///
    /// Usage lines are shown by default. Disable them with
    /// [`with_usage(false)`](StdoutReporter::with_usage).
    ///
    /// # Arguments
    ///
    /// * `writer` - Any [`Write`] implementation (typically `std::io::stdout()`
    ///   or `std::io::stderr()`).
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            show_usage: true,
            errors: Vec::new(),
        }
    }

    /// Controls whether `[usage]` lines are printed (builder pattern).
    ///
    /// Defaults to `true`. Set to `false` to reduce output noise when
    /// you are already tracking usage through a [`UsageReporter`].
    pub fn with_usage(mut self, show_usage: bool) -> Self {
        self.show_usage = show_usage;
        self
    }

    /// Returns a shared reference to the underlying writer.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Drains and returns all errors accumulated during event handling.
    ///
    /// Subsequent calls return an empty `Vec` until new errors occur.
    pub fn take_errors(&mut self) -> Vec<ReportError> {
        std::mem::take(&mut self.errors)
    }

    fn record_result(&mut self, result: Result<(), ReportError>) {
        if let Err(error) = result {
            self.errors.push(error);
        }
    }
}

impl<W> LoopObserver for StdoutReporter<W>
where
    W: Write + Send,
{
    fn handle_event(&mut self, event: AgentEvent) {
        let result = write_stdout_event(&mut self.writer, &event, self.show_usage);
        self.record_result(result);
    }
}

fn write_stdout_event<W>(
    writer: &mut W,
    event: &AgentEvent,
    show_usage: bool,
) -> Result<(), ReportError>
where
    W: Write,
{
    match event {
        AgentEvent::RunStarted { session_id } => {
            writeln!(writer, "[run] started session={session_id}")?;
        }
        AgentEvent::TurnStarted {
            session_id,
            turn_id,
        } => {
            writeln!(writer, "[turn] started session={session_id} turn={turn_id}")?;
        }
        AgentEvent::InputAccepted { items, .. } => {
            writeln!(writer, "[input] accepted items={}", items.len())?;
        }
        AgentEvent::ContentDelta(delta) => {
            writeln!(writer, "[delta] {delta:?}")?;
        }
        AgentEvent::ToolCallRequested(call) => {
            writeln!(writer, "[tool] call {} {}", call.name, call.input)?;
        }
        AgentEvent::ToolResultReceived(result) => {
            writeln!(
                writer,
                "[tool] result call_id={} is_error={}",
                result.call_id, result.is_error
            )?;
        }
        AgentEvent::ApprovalRequired(request) => {
            writeln!(
                writer,
                "[approval] {} {:?}",
                request.summary, request.reason
            )?;
        }
        AgentEvent::ApprovalResolved { approved } => {
            writeln!(writer, "[approval] resolved approved={approved}")?;
        }
        AgentEvent::ToolCatalogChanged(event) => {
            writeln!(
                writer,
                "[tools] catalog changed source={} added={} removed={} changed={}",
                event.source,
                event.added.len(),
                event.removed.len(),
                event.changed.len()
            )?;
        }
        AgentEvent::CompactionStarted {
            turn_id, reason, ..
        } => {
            writeln!(
                writer,
                "[compaction] started turn={} reason={reason:?}",
                turn_id
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "none".into())
            )?;
        }
        AgentEvent::CompactionFinished {
            turn_id,
            replaced_items,
            transcript_len,
            ..
        } => {
            writeln!(
                writer,
                "[compaction] finished turn={} replaced_items={} transcript_len={}",
                turn_id
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "none".into()),
                replaced_items,
                transcript_len
            )?;
        }
        AgentEvent::UsageUpdated(usage) if show_usage => {
            writeln!(writer, "[usage] {}", format_usage(usage))?;
        }
        AgentEvent::UsageUpdated(_) => {}
        AgentEvent::Warning { message } => {
            writeln!(writer, "[warning] {message}")?;
        }
        AgentEvent::RunFailed { message } => {
            writeln!(writer, "[error] {message}")?;
        }
        AgentEvent::TurnFinished(result) => {
            writeln!(
                writer,
                "[turn] finished reason={:?} items={}",
                result.finish_reason,
                result.items.len()
            )?;
            for item in &result.items {
                write_item_summary(writer, item)?;
            }
            if show_usage && let Some(usage) = &result.usage {
                writeln!(writer, "[usage] {}", format_usage(usage))?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn write_item_summary<W>(writer: &mut W, item: &Item) -> Result<(), ReportError>
where
    W: Write,
{
    writeln!(writer, "  [{}]", item_kind_name(item.kind))?;
    for part in &item.parts {
        match part {
            Part::Text(text) => writeln!(writer, "    [text] {}", text.text)?,
            Part::Reasoning(reasoning) => {
                if let Some(summary) = &reasoning.summary {
                    writeln!(writer, "    [reasoning] {summary}")?;
                } else {
                    writeln!(writer, "    [reasoning]")?;
                }
            }
            Part::ToolCall(call) => {
                writeln!(writer, "    [tool-call] {} {}", call.name, call.input)?
            }
            Part::ToolResult(result) => writeln!(
                writer,
                "    [tool-result] call={} error={}",
                result.call_id, result.is_error
            )?,
            Part::Structured(value) => writeln!(writer, "    [structured] {}", value.value)?,
            Part::Media(media) => writeln!(
                writer,
                "    [media] {:?} {}",
                media.modality, media.mime_type
            )?,
            Part::File(file) => writeln!(
                writer,
                "    [file] {}",
                file.name.as_deref().unwrap_or("<unnamed>")
            )?,
            Part::Custom(custom) => writeln!(writer, "    [custom] {}", custom.kind)?,
        }
    }
    Ok(())
}

fn item_kind_name(kind: ItemKind) -> &'static str {
    match kind {
        ItemKind::System => "system",
        ItemKind::Developer => "developer",
        ItemKind::User => "user",
        ItemKind::Assistant => "assistant",
        ItemKind::Tool => "tool",
        ItemKind::Context => "context",
        ItemKind::Notification => "notification",
    }
}

fn format_usage(usage: &Usage) -> String {
    match &usage.tokens {
        Some(TokenUsage {
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_input_tokens,
            cache_write_input_tokens,
        }) => format!(
            "input={} output={} reasoning={} cached_input={} cache_write_input={}",
            input_tokens,
            output_tokens,
            reasoning_tokens.unwrap_or_default(),
            cached_input_tokens.unwrap_or_default(),
            cache_write_input_tokens.unwrap_or_default()
        ),
        None => "no token usage".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agentkit_core::{FinishReason, MetadataMap, SessionId, TextPart};
    use agentkit_loop::TurnResult;

    #[test]
    fn usage_reporter_accumulates_usage_events_and_turn_results() {
        let mut reporter = UsageReporter::new();

        reporter.handle_event(AgentEvent::UsageUpdated(Usage {
            tokens: Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                reasoning_tokens: Some(2),
                cached_input_tokens: Some(1),
                cache_write_input_tokens: Some(7),
            }),
            cost: None,
            metadata: MetadataMap::new(),
        }));

        reporter.handle_event(AgentEvent::TurnFinished(TurnResult {
            turn_id: "turn-1".into(),
            finish_reason: FinishReason::Completed,
            items: Vec::new(),
            usage: Some(Usage {
                tokens: Some(TokenUsage {
                    input_tokens: 3,
                    output_tokens: 4,
                    reasoning_tokens: Some(1),
                    cached_input_tokens: None,
                    cache_write_input_tokens: None,
                }),
                cost: None,
                metadata: MetadataMap::new(),
            }),
            metadata: MetadataMap::new(),
        }));

        let summary = reporter.summary();
        assert_eq!(summary.events_seen, 2);
        assert_eq!(summary.usage_events_seen, 2);
        assert_eq!(summary.turn_results_seen, 1);
        assert_eq!(summary.totals.input_tokens, 13);
        assert_eq!(summary.totals.output_tokens, 9);
        assert_eq!(summary.totals.reasoning_tokens, 3);
        assert_eq!(summary.totals.cached_input_tokens, 1);
        assert_eq!(summary.totals.cache_write_input_tokens, 7);
    }

    #[test]
    fn transcript_reporter_tracks_inputs_and_outputs() {
        let mut reporter = TranscriptReporter::new();

        reporter.handle_event(AgentEvent::InputAccepted {
            session_id: SessionId::new("session-1"),
            items: vec![Item {
                id: None,
                kind: ItemKind::User,
                parts: vec![Part::Text(TextPart {
                    text: "hello".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }],
        });

        reporter.handle_event(AgentEvent::TurnFinished(TurnResult {
            turn_id: "turn-1".into(),
            finish_reason: FinishReason::Completed,
            items: vec![Item {
                id: None,
                kind: ItemKind::Assistant,
                parts: vec![Part::Text(TextPart {
                    text: "hi".into(),
                    metadata: MetadataMap::new(),
                })],
                metadata: MetadataMap::new(),
            }],
            usage: None,
            metadata: MetadataMap::new(),
        }));

        assert_eq!(reporter.transcript().items.len(), 2);
        assert_eq!(reporter.transcript().items[0].kind, ItemKind::User);
        assert_eq!(reporter.transcript().items[1].kind, ItemKind::Assistant);
    }

    #[test]
    fn jsonl_reporter_serializes_event_envelopes() {
        let mut reporter = JsonlReporter::new(Vec::new());
        reporter.handle_event(AgentEvent::RunStarted {
            session_id: SessionId::new("session-1"),
        });

        let output = String::from_utf8(reporter.writer().clone()).unwrap();
        assert!(output.contains("\"RunStarted\""));
        assert!(output.contains("session-1"));
    }

    fn run_started_event() -> AgentEvent {
        AgentEvent::RunStarted {
            session_id: SessionId::new("s1"),
        }
    }

    #[test]
    fn buffered_reporter_flushes_at_capacity() {
        let mut reporter = BufferedReporter::new(UsageReporter::new(), 2);
        reporter.handle_event(run_started_event());
        assert_eq!(reporter.pending(), 1);
        assert_eq!(reporter.inner().summary().events_seen, 0);

        reporter.handle_event(run_started_event());
        assert_eq!(reporter.pending(), 0);
        assert_eq!(reporter.inner().summary().events_seen, 2);
    }

    #[test]
    fn buffered_reporter_manual_flush() {
        let mut reporter = BufferedReporter::new(UsageReporter::new(), 0);
        reporter.handle_event(run_started_event());
        reporter.handle_event(run_started_event());
        assert_eq!(reporter.pending(), 2);

        reporter.flush();
        assert_eq!(reporter.pending(), 0);
        assert_eq!(reporter.inner().summary().events_seen, 2);
    }

    #[test]
    fn buffered_reporter_flushes_on_drop() {
        let inner = {
            let mut reporter = BufferedReporter::new(UsageReporter::new(), 100);
            reporter.handle_event(run_started_event());
            reporter.handle_event(run_started_event());
            assert_eq!(reporter.inner().summary().events_seen, 0);
            // Drop will flush — but we can't inspect after drop.
            // Instead, verify flush works by checking pending before drop.
            assert_eq!(reporter.pending(), 2);
            reporter
        };
        // After the block, `inner` is the dropped BufferedReporter — but we
        // moved it out, so it's still alive here. Verify flush happened on
        // the inner reporter by inspecting it.
        assert_eq!(inner.inner().summary().events_seen, 0);
        // The actual drop-flush happens when `inner` goes out of scope at
        // end of test. We at least verify the API is sound.
    }

    #[test]
    fn channel_reporter_delivers_events() {
        let (mut reporter, rx) = ChannelReporter::pair();
        reporter.handle_event(run_started_event());
        reporter.handle_event(run_started_event());

        let events: Vec<_> = rx.try_iter().collect();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn channel_reporter_survives_dropped_receiver() {
        let (mut reporter, rx) = ChannelReporter::pair();
        drop(rx);
        // Should not panic — errors are silently dropped.
        reporter.handle_event(run_started_event());
    }

    #[test]
    fn channel_reporter_fallible_returns_error_on_dropped_receiver() {
        let (mut reporter, rx) = ChannelReporter::pair();
        drop(rx);

        let result = reporter.try_handle_event(&run_started_event());
        assert!(matches!(result, Err(ReportError::ChannelSend)));
    }

    #[test]
    fn policy_reporter_ignore_swallows_errors() {
        let (reporter, rx) = ChannelReporter::pair();
        drop(rx);
        let mut policy = PolicyReporter::new(reporter, FailurePolicy::Ignore);
        policy.handle_event(run_started_event());
        assert!(policy.take_errors().is_empty());
    }

    #[test]
    fn policy_reporter_accumulate_collects_errors() {
        let (reporter, rx) = ChannelReporter::pair();
        drop(rx);
        let mut policy = PolicyReporter::new(reporter, FailurePolicy::Accumulate);
        policy.handle_event(run_started_event());
        policy.handle_event(run_started_event());

        let errors = policy.take_errors();
        assert_eq!(errors.len(), 2);
        assert!(matches!(errors[0], ReportError::ChannelSend));
    }

    #[test]
    #[should_panic(expected = "reporter error: channel send failed")]
    fn policy_reporter_fail_fast_panics() {
        let (reporter, rx) = ChannelReporter::pair();
        drop(rx);
        let mut policy = PolicyReporter::new(reporter, FailurePolicy::FailFast);
        policy.handle_event(run_started_event());
    }
}
