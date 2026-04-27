use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

use agentkit_compaction::{
    CompactionBackend, CompactionConfig, CompactionPipeline, CompactionReason,
    DropFailedToolResultsStrategy, DropReasoningStrategy, ItemCountTrigger, KeepRecentStrategy,
    SummarizeOlderStrategy, SummaryRequest, SummaryResult,
};
use agentkit_core::{
    Item, ItemKind, MetadataMap, Part, ReasoningPart, ToolCallId, ToolOutput, ToolResultPart,
    TurnCancellation,
};
use agentkit_loop::{
    Agent, AgentEvent, LoopInterrupt, LoopObserver, LoopStep, PromptCacheRequest,
    PromptCacheRetention, SessionConfig,
};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit_tools_core::{PermissionChecker, PermissionDecision};
use async_trait::async_trait;

const ROOT_SYSTEM_PROMPT: &str = "\
You are a root agent operating on a long transcript.
If context items marked as compaction summaries are present, treat them as authoritative compressed memory.
Preserve exact identifiers, codenames, paths, and decisions from the transcript.
When the user asks for a codename, return only the exact codename and no extra text.
";

const COMPACTION_SYSTEM_PROMPT: &str = "\
You are a compaction agent.
Your job is to compress transcript history into a durable context note for another agent.
Preserve exact secrets, codenames, identifiers, paths, commands, decisions, and unresolved work.
Omit chatter, chain-of-thought, and failed or redundant tool calls unless they matter to future turns.
Return only the compacted note as plain text.
";

const DEFAULT_SECRET: &str = "GOLDFINCH-17";
const DEFAULT_PROMPT: &str = "What is the sealed release codename? Return only the codename.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShowcaseMode {
    Structural,
    Semantic,
    Hybrid,
}

impl ShowcaseMode {
    pub fn all() -> [Self; 3] {
        [Self::Structural, Self::Semantic, Self::Hybrid]
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Semantic => "semantic",
            Self::Hybrid => "hybrid",
        }
    }
}

impl fmt::Display for ShowcaseMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ShowcaseMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "structural" => Ok(Self::Structural),
            "semantic" => Ok(Self::Semantic),
            "hybrid" => Ok(Self::Hybrid),
            other => Err(format!("unknown compaction mode: {other}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompactionEventRecord {
    pub reason: Option<CompactionReason>,
    pub replaced_items: usize,
    pub transcript_len: usize,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug)]
pub struct ShowcaseRun {
    pub mode: ShowcaseMode,
    pub prompt: String,
    pub output: String,
    pub seed_transcript: Vec<Item>,
    pub final_transcript: Vec<Item>,
    pub compaction_events: Vec<CompactionEventRecord>,
}

pub async fn run_mode(
    mode: ShowcaseMode,
    prompt: Option<&str>,
) -> Result<ShowcaseRun, Box<dyn Error>> {
    dotenvy::dotenv().ok();

    let config = OpenRouterConfig::from_env()?
        .with_temperature(0.0)
        .with_max_completion_tokens(384);
    let adapter = OpenRouterAdapter::new(config)?;
    let observer = RecordingObserver::default();

    let agent = Agent::builder()
        .model(adapter.clone())
        .permissions(AllowAll)
        .compaction(compaction_config(mode, adapter))
        .observer(observer.clone())
        .build()?;

    let prompt = prompt.unwrap_or(DEFAULT_PROMPT).to_owned();
    let seed_transcript = build_seed_transcript(mode, &prompt);

    let mut driver = agent
        .start(
            SessionConfig::new(format!("openrouter-compaction-agent-{mode}")).with_cache(
                PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
            ),
            seed_transcript.clone(),
        )
        .await?;

    let output = run_to_completion(&mut driver).await?;
    let snapshot = driver.snapshot();

    Ok(ShowcaseRun {
        mode,
        prompt,
        output,
        seed_transcript,
        final_transcript: snapshot.transcript,
        compaction_events: observer.compaction_events(),
    })
}

pub fn default_prompt() -> &'static str {
    DEFAULT_PROMPT
}

pub fn format_item(item: &Item) -> String {
    let mut sections = Vec::new();

    for part in &item.parts {
        match part {
            Part::Text(text) => sections.push(text.text.clone()),
            Part::Reasoning(reasoning) => sections.push(format!(
                "[reasoning] {}",
                reasoning
                    .summary
                    .clone()
                    .unwrap_or_else(|| "unspecified".into())
            )),
            Part::ToolResult(result) => sections.push(format!(
                "[tool result{}] {}",
                if result.is_error { " error" } else { "" },
                tool_output_to_string(&result.output)
            )),
            Part::ToolCall(call) => {
                sections.push(format!("[tool call] {} {}", call.name, call.input))
            }
            Part::Structured(value) => sections.push(value.value.to_string()),
            Part::Media(_) | Part::File(_) | Part::Custom(_) => {}
        }
    }

    format!("{}: {}", item.kind_label(), sections.join(" | "))
}

trait ItemKindLabel {
    fn kind_label(&self) -> &'static str;
}

impl ItemKindLabel for Item {
    fn kind_label(&self) -> &'static str {
        match self.kind {
            ItemKind::System => "system",
            ItemKind::Developer => "developer",
            ItemKind::User => "user",
            ItemKind::Assistant => "assistant",
            ItemKind::Tool => "tool",
            ItemKind::Context => "context",
        }
    }
}

fn tool_output_to_string(output: &ToolOutput) -> String {
    match output {
        ToolOutput::Text(text) => text.clone(),
        ToolOutput::Structured(value) => value.to_string(),
        ToolOutput::Parts(parts) => format!("{} parts", parts.len()),
        ToolOutput::Files(files) => format!("{} files", files.len()),
    }
}

fn compaction_config(mode: ShowcaseMode, adapter: OpenRouterAdapter) -> CompactionConfig {
    match mode {
        ShowcaseMode::Structural => CompactionConfig::new(
            ItemCountTrigger::new(10),
            CompactionPipeline::new()
                .with_strategy(DropReasoningStrategy::new())
                .with_strategy(DropFailedToolResultsStrategy::new())
                .with_strategy(
                    KeepRecentStrategy::new(8)
                        .preserve_kind(ItemKind::System)
                        .preserve_kind(ItemKind::Context),
                ),
        )
        .with_metadata(mode_metadata(mode)),
        ShowcaseMode::Semantic => CompactionConfig::new(
            ItemCountTrigger::new(8),
            SummarizeOlderStrategy::new(4)
                .preserve_kind(ItemKind::System)
                .preserve_kind(ItemKind::Context),
        )
        .with_backend(NestedLoopCompactionBackend::new(adapter))
        .with_metadata(mode_metadata(mode)),
        ShowcaseMode::Hybrid => CompactionConfig::new(
            ItemCountTrigger::new(10),
            CompactionPipeline::new()
                .with_strategy(DropReasoningStrategy::new())
                .with_strategy(DropFailedToolResultsStrategy::new())
                .with_strategy(
                    SummarizeOlderStrategy::new(4)
                        .preserve_kind(ItemKind::System)
                        .preserve_kind(ItemKind::Context),
                )
                .with_strategy(
                    KeepRecentStrategy::new(6)
                        .preserve_kind(ItemKind::System)
                        .preserve_kind(ItemKind::Context),
                ),
        )
        .with_backend(NestedLoopCompactionBackend::new(adapter))
        .with_metadata(mode_metadata(mode)),
    }
}

fn mode_metadata(mode: ShowcaseMode) -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert("mode".into(), mode.as_str().into());
    metadata
}

fn build_seed_transcript(mode: ShowcaseMode, prompt: &str) -> Vec<Item> {
    let secret = match mode {
        ShowcaseMode::Structural => DEFAULT_SECRET,
        ShowcaseMode::Semantic => DEFAULT_SECRET,
        ShowcaseMode::Hybrid => DEFAULT_SECRET,
    };

    match mode {
        ShowcaseMode::Structural => vec![
            Item::text(ItemKind::System, ROOT_SYSTEM_PROMPT),
            Item::text(
                ItemKind::User,
                "Start a release prep notebook for the spring launch.",
            ),
            assistant_with_reasoning(
                "I should summarize the plan before answering.",
                "Notebook started. We are targeting the spring launch.",
            ),
            failed_tool_item("git status timed out while checking the release branch."),
            Item::text(
                ItemKind::User,
                "Record the sealed release codename for the final checklist.",
            ),
            Item::text(
                ItemKind::Assistant,
                &format!("Recorded. The sealed release codename is {secret}."),
            ),
            assistant_with_reasoning(
                "I can expand on the checklist if needed.",
                "The checklist currently covers packaging, QA sign-off, and staged rollout.",
            ),
            Item::text(
                ItemKind::User,
                "Note that deploys happen from /srv/releases/spring.",
            ),
            Item::text(
                ItemKind::Assistant,
                "Noted. Deploys happen from /srv/releases/spring.",
            ),
            failed_tool_item(
                "dry-run packaging failed because the artifact bucket was unavailable.",
            ),
            Item::text(ItemKind::User, prompt),
        ],
        ShowcaseMode::Semantic => vec![
            Item::text(ItemKind::System, ROOT_SYSTEM_PROMPT),
            Item::text(
                ItemKind::User,
                "For the spring launch: the sealed release codename is GOLDFINCH-17 and deploys happen from /srv/releases/spring.",
            ),
            Item::text(
                ItemKind::Assistant,
                "Stored. I will preserve the codename and deployment path for future turns.",
            ),
            Item::text(ItemKind::User, "Reminder: QA sign-off is owned by Marta."),
            Item::text(ItemKind::Assistant, "Stored. Marta owns QA sign-off."),
            Item::text(
                ItemKind::User,
                "Reminder: staged rollout begins on Tuesday.",
            ),
            Item::text(
                ItemKind::Assistant,
                "Stored. Staged rollout begins on Tuesday.",
            ),
            Item::text(
                ItemKind::User,
                "Reminder: customer comms go out after QA sign-off.",
            ),
            Item::text(
                ItemKind::Assistant,
                "Stored. Customer comms go out after QA sign-off.",
            ),
            Item::text(ItemKind::User, prompt),
        ],
        ShowcaseMode::Hybrid => vec![
            Item::text(ItemKind::System, ROOT_SYSTEM_PROMPT),
            Item::text(
                ItemKind::User,
                "Archive this release memory: the sealed release codename is GOLDFINCH-17 and the rollback script lives at scripts/rollback-spring.sh.",
            ),
            assistant_with_reasoning(
                "I should restate the important identifiers.",
                "Stored. The release codename is GOLDFINCH-17 and the rollback script lives at scripts/rollback-spring.sh.",
            ),
            failed_tool_item("grep failed because the release notes file was missing."),
            Item::text(ItemKind::User, "Reminder: QA sign-off is owned by Marta."),
            Item::text(ItemKind::Assistant, "Stored. Marta owns QA sign-off."),
            assistant_with_reasoning(
                "I should note rollout sequencing.",
                "The staged rollout begins on Tuesday and customer comms follow QA sign-off.",
            ),
            failed_tool_item("ls failed because the artifact directory was not mounted."),
            Item::text(
                ItemKind::User,
                "Reminder: deploys happen from /srv/releases/spring.",
            ),
            Item::text(
                ItemKind::Assistant,
                "Stored. Deploys happen from /srv/releases/spring.",
            ),
            Item::text(ItemKind::User, prompt),
        ],
    }
}

#[derive(Clone)]
struct NestedLoopCompactionBackend {
    adapter: OpenRouterAdapter,
}

impl NestedLoopCompactionBackend {
    fn new(adapter: OpenRouterAdapter) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl CompactionBackend for NestedLoopCompactionBackend {
    async fn summarize(
        &self,
        request: SummaryRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<SummaryResult, agentkit_compaction::CompactionError> {
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return Err(agentkit_compaction::CompactionError::Cancelled);
        }

        let mut builder = Agent::builder()
            .model(self.adapter.clone())
            .permissions(AllowAll);
        if let Some(cancellation) = cancellation.as_ref() {
            builder = builder.cancellation(cancellation.handle().clone());
        }
        let agent = builder
            .build()
            .map_err(|error| agentkit_compaction::CompactionError::Failed(error.to_string()))?;

        let mut metadata = MetadataMap::new();
        metadata.insert("compaction_agent".into(), "nested-loop".into());

        let mut driver = agent
            .start(
                SessionConfig::new(format!(
                    "{}-compactor-{}",
                    request.session_id,
                    request.turn_id.clone().unwrap_or_else(|| "manual".into())
                ))
                .with_cache(
                    PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
                ),
                vec![
                    Item::text(ItemKind::System, COMPACTION_SYSTEM_PROMPT),
                    context_item("compaction transcript", &render_items(&request.items)),
                    Item::text(
                        ItemKind::User,
                        "Compress the supplied transcript into a context note for future turns. Preserve exact identifiers and actionable facts. Return only the compacted note.",
                    ),
                ],
            )
            .await
            .map_err(|error| agentkit_compaction::CompactionError::Failed(error.to_string()))?;

        let summary = run_to_completion(&mut driver)
            .await
            .map_err(|error| agentkit_compaction::CompactionError::Failed(error.to_string()))?;

        Ok(SummaryResult {
            items: vec![context_item("compaction summary", &summary)],
            metadata,
        })
    }
}

#[derive(Clone)]
struct RecordingObserver {
    compaction_events: Arc<Mutex<Vec<CompactionEventRecord>>>,
    pending_reason: Arc<Mutex<Option<CompactionReason>>>,
}

impl Default for RecordingObserver {
    fn default() -> Self {
        Self {
            compaction_events: Arc::new(Mutex::new(Vec::new())),
            pending_reason: Arc::new(Mutex::new(None)),
        }
    }
}

impl RecordingObserver {
    fn compaction_events(&self) -> Vec<CompactionEventRecord> {
        self.compaction_events
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }
}

impl LoopObserver for RecordingObserver {
    fn handle_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::CompactionStarted { reason, .. } => {
                *self
                    .pending_reason
                    .lock()
                    .unwrap_or_else(|err| err.into_inner()) = Some(reason);
            }
            AgentEvent::CompactionFinished {
                replaced_items,
                transcript_len,
                metadata,
                ..
            } => {
                let reason = self
                    .pending_reason
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .take();
                self.compaction_events
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .push(CompactionEventRecord {
                        reason,
                        replaced_items,
                        transcript_len,
                        metadata,
                    });
            }
            _ => {}
        }
    }
}

struct AllowAll;

impl PermissionChecker for AllowAll {
    fn evaluate(
        &self,
        _request: &dyn agentkit_tools_core::PermissionRequest,
    ) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

fn context_item(title: &str, text: &str) -> Item {
    let mut metadata = MetadataMap::new();
    metadata.insert("title".into(), title.into());
    Item::text(ItemKind::Context, text).with_metadata(metadata)
}

fn assistant_with_reasoning(reasoning: &str, text: &str) -> Item {
    Item::new(
        ItemKind::Assistant,
        vec![
            Part::Reasoning(ReasoningPart::summary(reasoning)),
            Part::text(text),
        ],
    )
}

fn failed_tool_item(text: &str) -> Item {
    Item::new(
        ItemKind::Tool,
        vec![Part::ToolResult(ToolResultPart::error(
            ToolCallId::new("failed-tool"),
            ToolOutput::text(text),
        ))],
    )
}

fn render_items(items: &[Item]) -> String {
    items.iter().map(format_item).collect::<Vec<_>>().join("\n")
}

async fn run_to_completion<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> Result<String, Box<dyn Error>>
where
    S: agentkit_loop::ModelSession,
{
    loop {
        match driver.next().await? {
            LoopStep::Finished(result) => return Ok(collect_assistant_output(&result.items)),
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                return Err(
                    format!("unexpected approval request: {}", pending.request.summary).into(),
                );
            }
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                return Err("loop requested more input before finishing".into());
            }
        }
    }
}

fn collect_assistant_output(items: &[Item]) -> String {
    let mut sections = Vec::new();

    for item in items {
        if item.kind != ItemKind::Assistant {
            continue;
        }

        for part in &item.parts {
            match part {
                Part::Text(text) => sections.push(text.text.clone()),
                Part::Structured(value) => sections.push(value.value.to_string()),
                Part::Reasoning(_)
                | Part::ToolCall(_)
                | Part::ToolResult(_)
                | Part::Media(_)
                | Part::File(_)
                | Part::Custom(_) => {}
            }
        }
    }

    sections.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY and a live OpenRouter model"]
    async fn semantic_compaction_preserves_the_codename() {
        let run = run_mode(ShowcaseMode::Semantic, None).await.unwrap();
        assert!(
            run.output.contains(DEFAULT_SECRET),
            "expected semantic compaction to preserve the secret, got {:?}",
            run.output
        );
        assert!(
            !run.compaction_events.is_empty(),
            "expected semantic compaction to fire"
        );
    }
}
