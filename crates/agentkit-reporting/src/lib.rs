use std::io::{self, Write};
use std::time::SystemTime;

use agentkit_core::{Item, ItemKind, Part, TokenUsage, Usage};
use agentkit_loop::{AgentEvent, LoopObserver, TurnResult};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReportError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EventEnvelope<'a> {
    pub timestamp: SystemTime,
    pub event: &'a AgentEvent,
}

#[derive(Default)]
pub struct CompositeReporter {
    children: Vec<Box<dyn LoopObserver>>,
}

impl CompositeReporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_observer(mut self, observer: impl LoopObserver + 'static) -> Self {
        self.children.push(Box::new(observer));
        self
    }

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

pub struct JsonlReporter<W> {
    writer: W,
    flush_each_event: bool,
    errors: Vec<ReportError>,
}

impl<W> JsonlReporter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            flush_each_event: true,
            errors: Vec::new(),
        }
    }

    pub fn with_flush_each_event(mut self, flush_each_event: bool) -> Self {
        self.flush_each_event = flush_each_event;
        self
    }

    pub fn writer(&self) -> &W {
        &self.writer
    }

    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.writer
    }

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UsageTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cached_input_tokens: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostTotals {
    pub amount: f64,
    pub currency: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UsageSummary {
    pub events_seen: usize,
    pub usage_events_seen: usize,
    pub turn_results_seen: usize,
    pub totals: UsageTotals,
    pub cost: Option<CostTotals>,
}

#[derive(Default)]
pub struct UsageReporter {
    summary: UsageSummary,
}

impl UsageReporter {
    pub fn new() -> Self {
        Self::default()
    }

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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TranscriptView {
    pub items: Vec<Item>,
}

#[derive(Default)]
pub struct TranscriptReporter {
    transcript: TranscriptView,
}

impl TranscriptReporter {
    pub fn new() -> Self {
        Self::default()
    }

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

pub struct StdoutReporter<W> {
    writer: W,
    show_usage: bool,
    errors: Vec<ReportError>,
}

impl<W> StdoutReporter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            show_usage: true,
            errors: Vec::new(),
        }
    }

    pub fn with_usage(mut self, show_usage: bool) -> Self {
        self.show_usage = show_usage;
        self
    }

    pub fn writer(&self) -> &W {
        &self.writer
    }

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
    }
}

fn format_usage(usage: &Usage) -> String {
    match &usage.tokens {
        Some(TokenUsage {
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_input_tokens,
        }) => format!(
            "input={} output={} reasoning={} cached_input={}",
            input_tokens,
            output_tokens,
            reasoning_tokens.unwrap_or_default(),
            cached_input_tokens.unwrap_or_default()
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
}
