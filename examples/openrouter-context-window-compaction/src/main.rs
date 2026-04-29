//! Demonstrates a context-window-aware compaction trigger driven by
//! `Usage` reports from the OpenRouter adapter.
//!
//! - `ContextWindowTrigger` reads a shared `AtomicU64` of last-known input
//!   tokens and fires when it crosses `context_length * percentage / 100`.
//! - `UsageObserver` watches `AgentEvent::UsageUpdated` and writes the
//!   provider-reported `input_tokens` into the same atomic.
//! - The model's `context_length` is fetched from OpenRouter's
//!   `/api/v1/models` catalog at startup, so the threshold matches the
//!   pinned model rather than a hardcoded value.

use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agentkit_compaction::{
    CompactionBackend, CompactionConfig, CompactionError, CompactionPipeline, CompactionReason,
    CompactionTrigger, DropFailedToolResultsStrategy, DropReasoningStrategy,
    SummarizeOlderStrategy, SummaryRequest, SummaryResult,
};
use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TurnCancellation, TurnId};
use agentkit_http::Http;
use agentkit_loop::{
    Agent, AgentEvent, InputRequest, LoopInterrupt, LoopObserver, LoopStep, PromptCacheRequest,
    PromptCacheRetention, SessionConfig,
};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit_tools_core::{PermissionChecker, PermissionDecision};
use async_trait::async_trait;
use serde::Deserialize;

const SYSTEM_PROMPT: &str = "\
You are a concise assistant. Answer in a single sentence under 40 words. \
If the transcript contains a compaction summary item, treat it as authoritative memory.";

const COMPACTION_SYSTEM_PROMPT: &str = "\
You are a compaction agent. Compress transcript history into a durable \
context note for another agent that has lost the original messages. \
Preserve every named person, every year and date, every place, every \
plot point, and every actionable fact. Drop chatter, narration, and \
chain-of-thought. Return only the compacted note as plain text.";

/// Embedded prose sized so that, after first-turn processing, the
/// provider-reported `input_tokens` crosses 60% of the pinned model's real
/// context window.
const STORY: &str = include_str!("story.txt");

/// Pinned to an OpenAI-routed model so `usage.prompt_tokens` is the real
/// (tiktoken) count, not a normalised billing approximation. Some
/// OpenRouter routes for free or community-hosted models cap reported
/// tokens regardless of input size, which makes a context-window trigger
/// impossible to fire reliably on those routes.
const DEFAULT_MODEL: &str = "openai/gpt-3.5-turbo";
/// Fire compaction once input tokens reach this share of the window.
const DEFAULT_PERCENTAGE: u32 = 60;

#[derive(Clone, Debug)]
struct ContextWindowTrigger {
    max_context_tokens: u64,
    percentage: u32,
    last_input_tokens: Arc<AtomicU64>,
}

impl ContextWindowTrigger {
    fn new(max_context_tokens: u64) -> Self {
        Self {
            max_context_tokens,
            percentage: 80,
            last_input_tokens: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Fire compaction once the last reported `input_tokens` reaches this
    /// percentage of `max_context_tokens`. Values are clamped to `0..=100`.
    fn with_percentage(mut self, percentage: u32) -> Self {
        self.percentage = percentage.min(100);
        self
    }

    fn threshold(&self) -> u64 {
        self.max_context_tokens
            .saturating_mul(self.percentage as u64)
            / 100
    }
}

impl CompactionTrigger for ContextWindowTrigger {
    fn should_compact(
        &self,
        _session_id: &SessionId,
        turn_id: Option<&TurnId>,
        transcript: &[Item],
    ) -> Option<CompactionReason> {
        let last = self.last_input_tokens.load(Ordering::Acquire);
        let threshold = self.threshold();
        let fire = last >= threshold;
        println!(
            "[trigger] turn={} transcript_len={} last_input_tokens={last} threshold={threshold} -> {}",
            turn_id.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
            transcript.len(),
            if fire { "FIRE" } else { "skip" },
        );
        fire.then(|| {
            CompactionReason::Custom(format!(
                "input_tokens={last} >= threshold={threshold} (window={}, {}%)",
                self.max_context_tokens, self.percentage
            ))
        })
    }
}

impl LoopObserver for ContextWindowTrigger {
    fn handle_event(&mut self, event: AgentEvent) {
        if let AgentEvent::UsageUpdated(usage) = event
            && let Some(tokens) = usage.tokens
        {
            self.last_input_tokens
                .store(tokens.input_tokens, Ordering::Release);
        }
    }
}

/// Prints turn, usage, compaction, and failure events to stdout/stderr.
/// Kept separate from `UsageObserver` so display logic doesn't muddy the
/// trigger's data source.
#[derive(Clone, Default)]
struct DisplayObserver;

impl LoopObserver for DisplayObserver {
    fn handle_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TurnStarted { turn_id, .. } => {
                println!("[turn] {turn_id} started");
            }
            AgentEvent::UsageUpdated(usage) => {
                if let Some(tokens) = usage.tokens {
                    println!(
                        "[usage] input_tokens={} output_tokens={} cached={:?}",
                        tokens.input_tokens, tokens.output_tokens, tokens.cached_input_tokens
                    );
                } else {
                    println!("[usage] provider returned no token counts");
                }
            }
            AgentEvent::TurnFinished(result) => {
                println!(
                    "[turn] {} finished reason={:?} items={}",
                    result.turn_id,
                    result.finish_reason,
                    result.items.len(),
                );
            }
            AgentEvent::CompactionStarted { reason, .. } => {
                println!("[compaction] start reason={reason:?}");
            }
            AgentEvent::CompactionFinished {
                replaced_items,
                transcript_len,
                ..
            } => {
                println!(
                    "[compaction] finished replaced={replaced_items} transcript_len={transcript_len}"
                );
            }
            AgentEvent::Warning { message } => {
                eprintln!("[warning] {message}");
            }
            AgentEvent::RunFailed { message } => {
                eprintln!("[run-failed] {message}");
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

/// `CompactionBackend` that runs a nested `Agent` loop over the same
/// OpenRouter adapter to summarise older transcript items into a single
/// `Context` item. `SummarizeOlderStrategy` calls this when the trigger
/// fires.
#[derive(Clone)]
struct NestedLoopCompactionBackend {
    adapter: OpenRouterAdapter,
}

#[async_trait]
impl CompactionBackend for NestedLoopCompactionBackend {
    async fn summarize(
        &self,
        request: SummaryRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<SummaryResult, CompactionError> {
        if cancellation
            .as_ref()
            .is_some_and(TurnCancellation::is_cancelled)
        {
            return Err(CompactionError::Cancelled);
        }

        let mut builder = Agent::builder()
            .model(self.adapter.clone())
            .permissions(AllowAll);
        if let Some(c) = cancellation.as_ref() {
            builder = builder.cancellation(c.handle().clone());
        }
        let agent = builder
            .build()
            .map_err(|e| CompactionError::Failed(e.to_string()))?;

        let rendered = render_items(&request.items);
        let mut driver = agent
            .start(
                SessionConfig::new(format!("{}-compactor", request.session_id)),
                vec![
                    Item::text(ItemKind::System, COMPACTION_SYSTEM_PROMPT),
                    Item::text(
                        ItemKind::User,
                        format!(
                            "Compress the transcript below into a comprehensive context note. \
                             Preserve every name, year, place, and event.\n\n{rendered}"
                        ),
                    ),
                ],
            )
            .await
            .map_err(|e| CompactionError::Failed(e.to_string()))?;

        let summary = run_to_completion(&mut driver)
            .await
            .map_err(|e| CompactionError::Failed(e.to_string()))?;

        println!("[compaction] backend produced {} chars", summary.len());

        Ok(SummaryResult {
            items: vec![Item::text(ItemKind::Context, summary)],
            metadata: MetadataMap::new(),
        })
    }
}

fn render_items(items: &[Item]) -> String {
    items
        .iter()
        .map(|item| {
            let kind = match item.kind {
                ItemKind::User => "USER",
                ItemKind::Assistant => "ASSISTANT",
                ItemKind::System => "SYSTEM",
                ItemKind::Developer => "DEVELOPER",
                ItemKind::Tool => "TOOL",
                ItemKind::Context => "CONTEXT",
                ItemKind::Notification => "NOTIFICATION",
            };
            let body = item
                .parts
                .iter()
                .filter_map(|p| match p {
                    Part::Text(t) => Some(t.text.clone()),
                    Part::Structured(v) => Some(v.value.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("[{kind}]\n{body}")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

async fn run_to_completion<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> Result<String, Box<dyn Error>>
where
    S: agentkit_loop::ModelSession,
{
    loop {
        match driver.next().await? {
            LoopStep::Finished(result) => {
                let mut sections = Vec::new();
                for item in result.items {
                    if item.kind != ItemKind::Assistant {
                        continue;
                    }
                    for part in item.parts {
                        if let Part::Text(text) = part {
                            sections.push(text.text);
                        }
                    }
                }
                return Ok(sections.join("\n"));
            }
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                return Err("compaction sub-agent unexpectedly awaiting input".into());
            }
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(_)) => {
                return Err("compaction sub-agent unexpectedly required approval".into());
            }
        }
    }
}

#[derive(Deserialize)]
struct EndpointsResponse {
    data: EndpointsData,
}

#[derive(Deserialize)]
struct EndpointsData {
    endpoints: Vec<EndpointInfo>,
}

#[derive(Deserialize)]
struct EndpointInfo {
    context_length: Option<u64>,
}

/// Looks up `context_length` for `model` from OpenRouter's
/// `/api/v1/models/{model}/endpoints` route. Chat completion responses
/// don't carry the window size, and the full `/v1/models` catalog is
/// hundreds of entries, so the per-model endpoints route is the cheapest
/// way to fetch it.
async fn fetch_context_length(api_key: &str, model: &str) -> Result<u64, Box<dyn Error>> {
    let http = Http::new(reqwest::Client::new());
    let url = format!("https://openrouter.ai/api/v1/models/{model}/endpoints");
    let response = http
        .get(url)
        .bearer_auth(api_key)
        .send()
        .await?
        .error_for_status()?;
    let body: EndpointsResponse = response.json().await?;
    // OpenRouter may route the same model to multiple upstream providers
    // with different advertised windows. We don't know which one a given
    // request will land on, so take the smallest as a conservative budget.
    // (For deterministic routing, pin a provider via the `provider` field
    // in the request body and look up that endpoint specifically.)
    body.data
        .endpoints
        .iter()
        .filter_map(|e| e.context_length)
        .min()
        .ok_or_else(|| format!("no endpoint for {model} reports context_length").into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();

    let percentage = parse_env_u32("CONTEXT_PERCENTAGE").unwrap_or(DEFAULT_PERCENTAGE);
    let api_key = env::var("OPENROUTER_API_KEY")
        .map_err(|_| "OPENROUTER_API_KEY must be set in the environment or .env")?;

    let mut config = OpenRouterConfig::from_env()?
        .with_temperature(0.0)
        .with_max_completion_tokens(512);
    let env_model = env::var("OPENROUTER_MODEL").ok();
    if let Some(envm) = env_model.as_deref()
        && envm != DEFAULT_MODEL
    {
        eprintln!(
            "[note] ignoring OPENROUTER_MODEL={envm}; this example pins {DEFAULT_MODEL} \
             (OpenAI-routed, reports native token counts)"
        );
    }
    config.model = DEFAULT_MODEL.into();
    let model = config.model.clone();
    let adapter = OpenRouterAdapter::new(config)?;

    println!("[startup] fetching context_length for {model} from OpenRouter...");
    let context_length = fetch_context_length(&api_key, &model).await?;

    let trigger = ContextWindowTrigger::new(context_length).with_percentage(percentage);

    let strategy = CompactionPipeline::new()
        .with_strategy(DropReasoningStrategy::new())
        .with_strategy(DropFailedToolResultsStrategy::new())
        .with_strategy(
            SummarizeOlderStrategy::new(1)
                .preserve_kind(ItemKind::System)
                .preserve_kind(ItemKind::Context),
        );

    let backend = NestedLoopCompactionBackend {
        adapter: adapter.clone(),
    };

    let agent = Agent::builder()
        .model(adapter)
        .permissions(AllowAll)
        .compaction(CompactionConfig::new(trigger.clone(), strategy).with_backend(backend))
        .observer(trigger.clone())
        .observer(DisplayObserver)
        .build()?;

    println!(
        "[config] model={model} context_length={context_length} percentage={percentage}% threshold={}",
        trigger.threshold()
    );

    let prompts = [
        format!(
            "Below is a short story. Read it, then summarize it in 3 to 4 sentences. \
             Preserve names, dates, and places.\n\n---\n{STORY}---"
        ),
        "Based on your summary, what year did Marcus's wife die?".to_owned(),
        "Based on your summary, who is Holger Kvist?".to_owned(),
    ];

    let mut driver = agent
        .start(
            SessionConfig::new("openrouter-context-window-compaction").with_cache(
                PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
            ),
            vec![
                Item::text(ItemKind::System, SYSTEM_PROMPT),
                Item::text(ItemKind::User, &prompts[0]),
            ],
        )
        .await?;

    println!("\nyou> {}", short(&prompts[0]));
    let mut pending = drive_to_idle(&mut driver).await?;

    for prompt in prompts.iter().skip(1) {
        println!("\nyou> {prompt}");
        pending.submit(&mut driver, vec![Item::text(ItemKind::User, prompt)])?;
        pending = drive_to_idle(&mut driver).await?;
    }

    let snapshot = driver.snapshot();
    println!(
        "\n[done] transcript_len={} pending_input={}",
        snapshot.transcript.len(),
        snapshot.pending_input.len()
    );

    Ok(())
}

async fn drive_to_idle<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> Result<InputRequest, Box<dyn Error>>
where
    S: agentkit_loop::ModelSession,
{
    loop {
        match driver.next().await? {
            LoopStep::Finished(result) => {
                if result.items.is_empty() {
                    println!("(no items returned this turn)");
                }
                for item in result.items {
                    let label = match item.kind {
                        ItemKind::Assistant => "assistant",
                        ItemKind::Tool => "tool",
                        ItemKind::System => "system",
                        ItemKind::User => "user",
                        ItemKind::Developer => "developer",
                        ItemKind::Context => "context",
                        ItemKind::Notification => "notification",
                    };
                    if item.parts.is_empty() {
                        println!("{label}> (empty)");
                        continue;
                    }
                    for part in item.parts {
                        match part {
                            Part::Text(text) => println!("{label}> {}", text.text),
                            Part::Reasoning(r) => println!(
                                "{label}> [reasoning] {}",
                                r.summary.unwrap_or_else(|| "<no summary>".into())
                            ),
                            Part::ToolCall(c) => {
                                println!("{label}> [tool call] {} {}", c.name, c.input)
                            }
                            Part::ToolResult(r) => {
                                println!("{label}> [tool result error={}]", r.is_error)
                            }
                            Part::Structured(v) => println!("{label}> [structured] {}", v.value),
                            Part::Media(_) | Part::File(_) | Part::Custom(_) => {
                                println!("{label}> [non-text part]")
                            }
                        }
                    }
                }
            }
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(req)) => return Ok(req),
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                return Err(
                    format!("unexpected approval interrupt: {}", pending.request.summary).into(),
                );
            }
        }
    }
}

fn parse_env_u32(key: &str) -> Option<u32> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

fn short(text: &str) -> String {
    let head: String = text.chars().take(80).collect();
    if text.chars().count() > 80 {
        format!("{head}...")
    } else {
        head
    }
}
