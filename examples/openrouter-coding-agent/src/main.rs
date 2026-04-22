//! Claude-Code-style REPL on top of the agentkit loop.
//!
//! Wires together:
//!
//! - `agentkit-loop` — multi-turn driver with tools, observers, compaction.
//! - `agentkit-provider-openrouter` — the model provider.
//! - `agentkit-tool-fs` + `agentkit-tool-shell` — the editing/execution toolset.
//! - A custom [`LoopObserver`] that streams assistant deltas, renders tool
//!   calls, surfaces compaction events, and feeds token usage into a
//!   context-percentage compaction trigger.
//! - A [`QueuedAdapter`] wrapper that lets the user type ahead during an
//!   in-flight turn; queued messages are injected at the next tool-round
//!   boundary without cancelling the turn.
//!
//! Slash commands: `/exit` or `/quit`. Ctrl-C cancels the in-flight turn.

use std::collections::HashMap;
use std::env;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use agentkit_compaction::{
    CompactionConfig, CompactionPipeline, CompactionReason, CompactionTrigger,
    DropFailedToolResultsStrategy, DropReasoningStrategy, KeepRecentStrategy,
};
use agentkit_core::{
    CancellationController, Delta, FinishReason, Item, ItemKind, MetadataMap, PartId, PartKind,
    SessionId, TurnCancellation, TurnId,
};
use agentkit_loop::{
    Agent, AgentEvent, LoopDriver, LoopError, LoopInterrupt, LoopObserver, LoopStep, ModelAdapter,
    ModelSession, PromptCacheRequest, PromptCacheRetention, SessionConfig, TurnRequest,
};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit_tools_core::{
    ApprovalDecision, ApprovalReason, ApprovalRequest, CommandPolicy, CompositePermissionChecker,
    PathPolicy, PermissionCode, PermissionDecision, PermissionDenial, ToolRegistry,
};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

const SYSTEM_PROMPT: &str = "\
You are a careful repository assistant operating inside a Claude-Code-style REPL.
Inspect the repository with the available fs.* and shell.exec tools instead of guessing.
Prefer concise answers. When using tools, use paths relative to the working directory.
If the user sends a new message while you're mid-task, integrate it into your plan for
the next step rather than restarting.
";

const DEFAULT_MAX_CONTEXT_TOKENS: u64 = 200_000;

// =============================================================================
// Inbox + QueuedAdapter — mid-turn user-message injection.
// =============================================================================

/// Two queues. `pending` holds user messages typed since the last round and not
/// yet seen by the model. `injected_history` records items the wrapper has
/// folded into a model request but that are not yet reflected in the driver's
/// own transcript — the main loop drains this on turn end and replays via
/// `submit_input` to keep both views consistent.
#[derive(Default)]
struct InboxInner {
    pending: Vec<Item>,
    injected_history: Vec<Item>,
}

#[derive(Clone, Default)]
struct Inbox(Arc<Mutex<InboxInner>>);

impl Inbox {
    fn push_user(&self, text: impl Into<String>) {
        self.0
            .lock()
            .unwrap()
            .pending
            .push(Item::text(ItemKind::User, text));
    }

    /// Wrapper hook. Drains pending, records into history, returns items for
    /// this round.
    fn drain_for_injection(&self) -> Vec<Item> {
        let mut inner = self.0.lock().unwrap();
        let items = std::mem::take(&mut inner.pending);
        inner.injected_history.extend(items.iter().cloned());
        items
    }

    /// Turn-end hook. Returns history so the caller can sync the driver's
    /// transcript via `submit_input`.
    fn drain_history(&self) -> Vec<Item> {
        std::mem::take(&mut self.0.lock().unwrap().injected_history)
    }

    /// Turn-end hook. Returns messages typed after the last round started
    /// (never seen by the model). Caller should `submit_input` and auto-advance.
    fn drain_late_pending(&self) -> Vec<Item> {
        std::mem::take(&mut self.0.lock().unwrap().pending)
    }
}

struct QueuedAdapter<A> {
    inner: A,
    inbox: Inbox,
}

#[async_trait]
impl<A> ModelAdapter for QueuedAdapter<A>
where
    A: ModelAdapter + Send + Sync,
    A::Session: Send,
{
    type Session = QueuedSession<A::Session>;

    async fn start_session(&self, config: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(QueuedSession {
            inner: self.inner.start_session(config).await?,
            inbox: self.inbox.clone(),
        })
    }
}

struct QueuedSession<S> {
    inner: S,
    inbox: Inbox,
}

#[async_trait]
impl<S> ModelSession for QueuedSession<S>
where
    S: ModelSession + Send,
{
    type Turn = S::Turn;

    async fn begin_turn(
        &mut self,
        mut request: TurnRequest,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Self::Turn, LoopError> {
        let injected = self.inbox.drain_for_injection();
        if !injected.is_empty() {
            request.transcript.extend(injected);
        }
        self.inner.begin_turn(request, cancellation).await
    }
}

// =============================================================================
// Token meter + compaction trigger (fires at 80% of context window).
// =============================================================================

#[derive(Clone)]
struct TokenMeter {
    current: Arc<AtomicU64>,
    threshold: u64,
    max: u64,
}

impl TokenMeter {
    fn new(max: u64) -> Self {
        Self {
            current: Arc::new(AtomicU64::new(0)),
            threshold: max * 4 / 5,
            max,
        }
    }

    fn record(&self, total: u64) {
        self.current.store(total, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.current.store(0, Ordering::Relaxed);
    }

    fn read(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }
}

struct TokenBudgetTrigger {
    meter: TokenMeter,
}

impl CompactionTrigger for TokenBudgetTrigger {
    fn should_compact(
        &self,
        _session: &SessionId,
        _turn: Option<&TurnId>,
        _transcript: &[Item],
    ) -> Option<CompactionReason> {
        if self.meter.read() >= self.meter.threshold {
            // Reset so the next trigger requires fresh usage data.
            self.meter.reset();
            Some(CompactionReason::Custom("context-window-80pct".into()))
        } else {
            None
        }
    }
}

// =============================================================================
// Renderer — claude-code-style output via LoopObserver.
// =============================================================================

struct Renderer {
    part_kinds: HashMap<PartId, PartKind>,
    streaming_text: bool,
    meter: TokenMeter,
}

impl Renderer {
    fn new(meter: TokenMeter) -> Self {
        Self {
            part_kinds: HashMap::new(),
            streaming_text: false,
            meter,
        }
    }

    fn end_text_stream(&mut self) {
        if self.streaming_text {
            println!();
            self.streaming_text = false;
        }
    }
}

impl LoopObserver for Renderer {
    fn handle_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::ContentDelta(Delta::BeginPart { part_id, kind }) => {
                self.part_kinds.insert(part_id, kind);
            }
            AgentEvent::ContentDelta(Delta::AppendText { part_id, chunk }) => {
                if matches!(self.part_kinds.get(&part_id), Some(PartKind::Text)) {
                    self.streaming_text = true;
                    print!("{chunk}");
                    let _ = std::io::stdout().flush();
                }
            }
            // Non-streaming providers (e.g. OpenRouter today) deliver the
            // finished part in one shot via CommitPart, without prior
            // AppendText deltas. Render here so the assistant reply is
            // visible. For streaming providers, AppendText has already
            // printed the text, so we only close the line.
            AgentEvent::ContentDelta(Delta::CommitPart { part }) => match part {
                agentkit_core::Part::Text(text) => {
                    if self.streaming_text {
                        self.end_text_stream();
                    } else {
                        println!("{}", text.text);
                    }
                }
                agentkit_core::Part::Reasoning(r) => {
                    self.end_text_stream();
                    if let Some(summary) = r.summary {
                        for line in summary.lines() {
                            println!("· {line}");
                        }
                    }
                }
                agentkit_core::Part::Structured(s) => {
                    self.end_text_stream();
                    println!("{}", s.value);
                }
                _ => {}
            },
            AgentEvent::ToolCallRequested(call) => {
                self.end_text_stream();
                let args =
                    serde_json::to_string(&call.input).unwrap_or_else(|_| call.input.to_string());
                println!("⏺ {}({})", call.name, truncate(&args, 160));
            }
            AgentEvent::UsageUpdated(usage) => {
                if let Some(tokens) = usage.tokens.as_ref() {
                    let total = tokens.input_tokens + tokens.output_tokens;
                    self.meter.record(total);
                }
            }
            AgentEvent::CompactionStarted { reason, .. } => {
                self.end_text_stream();
                println!("✻ compacting transcript ({reason:?})…");
            }
            AgentEvent::CompactionFinished {
                replaced_items,
                transcript_len,
                ..
            } => {
                println!("✻ compacted: replaced {replaced_items}, now {transcript_len} items");
            }
            AgentEvent::Warning { message } => {
                self.end_text_stream();
                eprintln!("⚠ {message}");
            }
            AgentEvent::RunFailed { message } => {
                self.end_text_stream();
                eprintln!("✗ {message}");
            }
            AgentEvent::TurnFinished(result) => {
                self.end_text_stream();
                if matches!(result.finish_reason, FinishReason::Cancelled) {
                    println!("— turn cancelled");
                }
                if let Some(tokens) = result.usage.as_ref().and_then(|u| u.tokens.as_ref()) {
                    let pct = self.meter.read() as f64 / self.meter.max.max(1) as f64 * 100.0;
                    println!(
                        "⟡ {} in · {} out · context {:.0}% of {}",
                        tokens.input_tokens, tokens.output_tokens, pct, self.meter.max
                    );
                }
            }
            _ => {}
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

// =============================================================================
// REPL.
// =============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let config = OpenRouterConfig::from_env()?;
    let model_name = config.model.clone();
    let inbox = Inbox::default();
    let adapter = QueuedAdapter {
        inner: OpenRouterAdapter::new(config)?,
        inbox: inbox.clone(),
    };

    let mut tools = ToolRegistry::new();
    merge_registry(&mut tools, agentkit_tool_fs::registry());
    merge_registry(&mut tools, agentkit_tool_shell::registry());

    let workspace_root = env::current_dir()?;
    let permissions = CompositePermissionChecker::new(PermissionDecision::Deny(PermissionDenial {
        code: PermissionCode::UnknownRequest,
        message: "tool request is not covered by any policy".into(),
        metadata: MetadataMap::new(),
    }))
    .with_policy(
        PathPolicy::new()
            .allow_root(workspace_root.clone())
            .require_approval_outside_allowed(true),
    )
    .with_policy(
        CommandPolicy::new()
            .allow_cwd(workspace_root.clone())
            .require_approval_for_unknown(true),
    );

    let max_ctx = env::var("AGENTKIT_MAX_CONTEXT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS);
    let meter = TokenMeter::new(max_ctx);

    let cancellation = CancellationController::new();

    let agent = Agent::builder()
        .model(adapter)
        .tools(tools)
        .permissions(permissions)
        .cancellation(cancellation.handle())
        .observer(Renderer::new(meter.clone()))
        .compaction(CompactionConfig::new(
            TokenBudgetTrigger {
                meter: meter.clone(),
            },
            CompactionPipeline::new()
                .with_strategy(DropReasoningStrategy::new())
                .with_strategy(DropFailedToolResultsStrategy::new())
                .with_strategy(
                    KeepRecentStrategy::new(16)
                        .preserve_kind(ItemKind::System)
                        .preserve_kind(ItemKind::Context),
                ),
        ))
        .build()?;

    let mut driver = agent
        .start(SessionConfig::new("openrouter-coding-agent").with_cache(
            PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
        ))
        .await?;

    // Seed the system prompt so the first user turn sees it.
    driver.submit_input(vec![Item::text(ItemKind::System, SYSTEM_PROMPT)])?;

    print_banner(&model_name, &workspace_root, max_ctx);
    repl(&mut driver, &inbox, &cancellation).await
}

fn merge_registry(target: &mut ToolRegistry, source: ToolRegistry) {
    for tool in source.tools() {
        target.register_arc(tool);
    }
}

fn print_banner(model: &str, root: &Path, max_ctx: u64) {
    println!("openrouter-coding-agent  ({model})");
    println!("cwd: {}", root.display());
    println!("context: {max_ctx} tokens · compaction at 80%");
    println!("Ctrl-C cancels the current turn · /exit quits · type ahead to queue input\n");
}

async fn repl<S>(
    driver: &mut LoopDriver<S>,
    inbox: &Inbox,
    cancellation: &CancellationController,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: ModelSession + Send,
{
    let (line_tx, mut line_rx) = mpsc::unbounded_channel::<String>();

    // Single stdin line reader. Every line goes through this channel.
    tokio::spawn(async move {
        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });

    // Ctrl-C -> cancel the current turn (the driver's cancellation controller
    // bumps its generation, any in-flight model call / tool round bails out).
    {
        let cc = cancellation.clone();
        tokio::spawn(async move {
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    break;
                }
                cc.interrupt();
            }
        });
    }

    let mut quit_after_turn = false;

    'outer: loop {
        prompt();
        let Some(first_line) = line_rx.recv().await else {
            break 'outer;
        };
        let trimmed = first_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, "/exit" | "/quit") {
            break 'outer;
        }

        driver.submit_input(vec![Item::text(ItemKind::User, first_line)])?;

        'turn: loop {
            let step = run_next_with_queue(
                driver,
                &mut line_rx,
                inbox,
                cancellation,
                &mut quit_after_turn,
            )
            .await?;

            match step {
                LoopStep::Finished(_) | LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
                    // Sync: the wrapper injected these into model requests; now
                    // feed them into the driver's own transcript so subsequent
                    // turns don't "forget" them.
                    let history = inbox.drain_history();
                    if !history.is_empty() {
                        driver.submit_input(history)?;
                    }
                    // Anything typed after the final round started never reached
                    // the model — deliver it now by auto-advancing into a new turn.
                    let late = inbox.drain_late_pending();
                    if !late.is_empty() {
                        driver.submit_input(late)?;
                        continue 'turn;
                    }
                    break 'turn;
                }
                LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                    // Drain any lines that arrived before the approval prompt
                    // appears — they were typed as queued messages or slash
                    // commands during the turn, not as answers. Only NEW input
                    // (post-prompt) is consumed as the approval response.
                    let mut pre_quit = false;
                    let mut pre_cancel = false;
                    while let Ok(line) = line_rx.try_recv() {
                        match handle_queue_line(&line, inbox) {
                            LineAction::Skip | LineAction::Enqueued => {}
                            LineAction::Quit => pre_quit = true,
                            LineAction::Cancel => pre_cancel = true,
                        }
                    }
                    if pre_quit {
                        pending.deny_with_reason(driver, "user requested quit")?;
                        quit_after_turn = true;
                        break 'turn;
                    }
                    if pre_cancel {
                        pending.deny_with_reason(driver, "user cancelled the turn")?;
                        cancellation.interrupt();
                        continue 'turn;
                    }

                    let decision = prompt_approval(&pending.request, &mut line_rx).await?;
                    match decision {
                        ApprovalDecision::Approve => pending.approve(driver)?,
                        ApprovalDecision::Deny { reason: None } => pending.deny(driver)?,
                        ApprovalDecision::Deny {
                            reason: Some(reason),
                        } => pending.deny_with_reason(driver, reason)?,
                    }
                    continue 'turn;
                }
                LoopStep::Interrupt(LoopInterrupt::AuthRequest(req)) => {
                    eprintln!(
                        "✗ auth required for {} — interactive auth not wired in this example",
                        req.provider
                    );
                    break 'turn;
                }
            }
        }

        if quit_after_turn {
            break 'outer;
        }
    }

    Ok(())
}

/// Drive a single `driver.next()` while concurrently routing stdin lines into
/// the inbox (typed-ahead user messages for mid-turn injection).
async fn run_next_with_queue<S>(
    driver: &mut LoopDriver<S>,
    line_rx: &mut mpsc::UnboundedReceiver<String>,
    inbox: &Inbox,
    cancellation: &CancellationController,
    quit_after_turn: &mut bool,
) -> Result<LoopStep, LoopError>
where
    S: ModelSession + Send,
{
    let mut fut = Box::pin(driver.next());
    let mut line_closed = false;
    loop {
        tokio::select! {
            biased;
            step = &mut fut => return step,
            maybe_line = line_rx.recv(), if !line_closed => {
                match maybe_line {
                    Some(line) => match handle_queue_line(&line, inbox) {
                        LineAction::Skip | LineAction::Enqueued => {}
                        LineAction::Quit => {
                            *quit_after_turn = true;
                            cancellation.interrupt();
                        }
                        LineAction::Cancel => {
                            cancellation.interrupt();
                        }
                    },
                    None => {
                        line_closed = true;
                        *quit_after_turn = true;
                        cancellation.interrupt();
                    }
                }
            }
        }
    }
}

async fn prompt_approval(
    req: &ApprovalRequest,
    line_rx: &mut mpsc::UnboundedReceiver<String>,
) -> Result<ApprovalDecision, Box<dyn std::error::Error>> {
    println!();
    println!("✻ approval required: {}", approval_label(&req.reason));
    println!("  {}", req.summary);
    print!("  approve? [y/N] (or type a deny reason): ");
    std::io::stdout().flush()?;
    let Some(line) = line_rx.recv().await else {
        return Ok(ApprovalDecision::Deny {
            reason: Some("stdin closed".into()),
        });
    };
    let trimmed = line.trim();
    Ok(match trimmed.to_lowercase().as_str() {
        "y" | "yes" => ApprovalDecision::Approve,
        "" | "n" | "no" => ApprovalDecision::Deny { reason: None },
        _ => ApprovalDecision::Deny {
            reason: Some(trimmed.to_string()),
        },
    })
}

enum LineAction {
    Skip,
    Enqueued,
    Quit,
    Cancel,
}

/// Classify a stdin line received during (or just before) an active turn.
///
/// User messages are pushed to the inbox here; slash commands are surfaced via
/// [`LineAction`] so callers can apply the appropriate side-effect
/// (cancel/quit) in their own control-flow context.
fn handle_queue_line(line: &str, inbox: &Inbox) -> LineAction {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return LineAction::Skip;
    }
    if matches!(trimmed, "/exit" | "/quit") {
        return LineAction::Quit;
    }
    if trimmed == "/cancel" {
        return LineAction::Cancel;
    }
    println!(
        "  ⎿ queued (will reach model at next round): {}",
        truncate(trimmed, 140)
    );
    inbox.push_user(line.to_string());
    LineAction::Enqueued
}

fn approval_label(reason: &ApprovalReason) -> &'static str {
    match reason {
        ApprovalReason::PolicyRequiresConfirmation => "policy requires confirmation",
        ApprovalReason::EscalatedRisk => "escalated risk",
        ApprovalReason::UnknownTarget => "unknown target",
        ApprovalReason::SensitivePath => "sensitive path",
        ApprovalReason::SensitiveCommand => "sensitive command",
        ApprovalReason::SensitiveServer => "sensitive MCP server",
        ApprovalReason::SensitiveAuthScope => "sensitive auth scope",
    }
}

fn prompt() {
    print!("\n› ");
    let _ = std::io::stdout().flush();
}
