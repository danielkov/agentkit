//! Interactive REPL against Google's Gemini API demonstrating the
//! `agentkit-provider-gemini` feature surface.
//!
//! Credentials and model are read from the environment via
//! `GeminiConfig::from_env` (`GEMINI_API_KEY` / `GOOGLE_API_KEY` /
//! `GEMINI_AUTH_TOKEN`, `GEMINI_MODEL`, `GEMINI_BASE_URL`, …).

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use agentkit_core::{CancellationController, Delta, Item, ItemKind, Part, Usage};
use agentkit_http::ResilienceConfig;
use agentkit_loop::{
    Agent, AgentEvent, InputRequest, LoopInterrupt, LoopObserver, LoopStep, ObservedEvent,
    SessionConfig,
};
use agentkit_provider_gemini::{
    GeminiAdapter, GeminiConfig, GeminiServerTool, ThinkingConfig, ThinkingLevel,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let cli = match CliArgs::parse(std::env::args().skip(1))? {
        CliOutcome::Run(cli) => cli,
        CliOutcome::Help => {
            print_help();
            return Ok(());
        }
    };

    let config = build_config(&cli)?;
    println!(
        "gemini-chat (model={} streaming={})",
        config.model, config.streaming
    );

    let adapter = GeminiAdapter::new(config)?;
    let cancellation = CancellationController::new();
    let turn_active = Arc::new(AtomicBool::new(false));
    spawn_signal_router(cancellation.clone(), Arc::clone(&turn_active));

    let usage_slot: Arc<Mutex<Option<Usage>>> = Arc::new(Mutex::new(None));
    println!(
        "Type a prompt and press enter. Use /exit or /quit to leave, Ctrl-C to cancel the current turn."
    );

    let Some(first_prompt) = read_prompt()? else {
        return Ok(());
    };

    let mut transcript = Vec::new();
    if let Some(system) = &cli.system {
        transcript.push(Item::text(ItemKind::System, system.clone()));
    }

    let agent = Agent::builder()
        .model(adapter)
        .cancellation(cancellation.handle())
        .observer(StreamPrinter::new(Arc::clone(&usage_slot)))
        .transcript(transcript)
        .input(vec![Item::text(ItemKind::User, first_prompt)])
        .build()?;

    let mut driver = agent.start(SessionConfig::new("gemini-chat")).await?;

    loop {
        usage_slot.lock().expect("usage slot poisoned").take();
        turn_active.store(true, Ordering::SeqCst);
        let result = run_turn(&mut driver, &usage_slot).await;
        turn_active.store(false, Ordering::SeqCst);
        let pending_input = result?;

        let Some(prompt) = read_prompt()? else {
            break;
        };
        pending_input.submit(&mut driver, vec![Item::text(ItemKind::User, prompt)])?;
    }

    Ok(())
}

fn read_prompt() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut line = String::new();
    loop {
        print!("you> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            println!();
            return Ok(None);
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "/exit" || prompt == "/quit" {
            return Ok(None);
        }
        return Ok(Some(prompt.to_string()));
    }
}

fn spawn_signal_router(cancellation: CancellationController, turn_active: Arc<AtomicBool>) {
    tokio::spawn(async move {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                return;
            }
            if turn_active.load(Ordering::SeqCst) {
                cancellation.interrupt();
            } else {
                eprintln!("\n[exit]");
                std::process::exit(0);
            }
        }
    });
}

async fn run_turn<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    usage_slot: &Arc<Mutex<Option<Usage>>>,
) -> Result<InputRequest, Box<dyn std::error::Error>>
where
    S: agentkit_loop::ModelSession,
{
    loop {
        match driver.next().await? {
            LoopStep::Finished(result) => {
                println!();
                if let Some(usage) = usage_slot
                    .lock()
                    .expect("usage slot poisoned")
                    .take()
                    .or(result.usage)
                    && let Some(tokens) = usage.tokens
                {
                    println!(
                        "[usage: input={} output={} reasoning={} cached={}]",
                        tokens.input_tokens,
                        tokens.output_tokens,
                        tokens
                            .reasoning_tokens
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "-".into()),
                        tokens
                            .cached_input_tokens
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "-".into()),
                    );
                }
            }
            LoopStep::Interrupt(LoopInterrupt::AfterToolResult(_)) => continue,
            LoopStep::Interrupt(LoopInterrupt::AwaitingInput(req)) => return Ok(req),
            LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(pending)) => {
                return Err(
                    format!("unhandled approval request: {}", pending.request.summary).into(),
                );
            }
        }
    }
}

struct CliArgs {
    streaming: bool,
    google_search: bool,
    code_exec: bool,
    url_context: bool,
    thinking_budget: Option<i32>,
    thinking_level: Option<ThinkingLevel>,
    system: Option<String>,
    temperature: Option<f32>,
}

enum CliOutcome {
    Run(CliArgs),
    Help,
}

impl CliArgs {
    fn parse<I: IntoIterator<Item = String>>(
        args: I,
    ) -> Result<CliOutcome, Box<dyn std::error::Error>> {
        let mut cli = CliArgs {
            streaming: true,
            google_search: false,
            code_exec: false,
            url_context: false,
            thinking_budget: None,
            thinking_level: None,
            system: None,
            temperature: None,
        };
        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--help" | "-h" => return Ok(CliOutcome::Help),
                "--streaming" => cli.streaming = true,
                "--no-streaming" => cli.streaming = false,
                "--google-search" => cli.google_search = true,
                "--code-exec" => cli.code_exec = true,
                "--url-context" => cli.url_context = true,
                "--thinking-budget" => {
                    let v = iter
                        .next()
                        .ok_or("--thinking-budget requires an integer")?
                        .parse::<i32>()?;
                    cli.thinking_budget = Some(v);
                }
                "--thinking-level" => {
                    let raw = iter
                        .next()
                        .ok_or("--thinking-level requires minimal|low|medium|high")?;
                    cli.thinking_level = Some(match raw.to_ascii_lowercase().as_str() {
                        "minimal" => ThinkingLevel::Minimal,
                        "low" => ThinkingLevel::Low,
                        "medium" => ThinkingLevel::Medium,
                        "high" => ThinkingLevel::High,
                        other => return Err(format!("unknown thinking level: {other}").into()),
                    });
                }
                "--system" => {
                    cli.system = Some(iter.next().ok_or("--system requires text")?);
                }
                "--temperature" => {
                    let t = iter
                        .next()
                        .ok_or("--temperature requires a float")?
                        .parse::<f32>()?;
                    cli.temperature = Some(t);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }
        Ok(CliOutcome::Run(cli))
    }
}

fn print_help() {
    println!(
        "gemini-chat — demonstrate agentkit-provider-gemini\n\n\
         USAGE:\n    gemini-chat [OPTIONS]\n\n\
         OPTIONS:\n    \
             --streaming                 Use SSE streaming (default)\n    \
             --no-streaming              Use buffered generateContent\n    \
             --google-search             Enable Google Search grounding\n    \
             --code-exec                 Enable server-side code execution\n    \
             --url-context               Enable URL context retrieval\n    \
             --thinking-budget N         Gemini 2.5 thinking token budget\n    \
             --thinking-level LEVEL      Gemini 3+ thinking level (minimal|low|medium|high)\n    \
             --system TEXT               Seed the transcript with a system prompt\n    \
             --temperature F             Sampling temperature\n    \
             --help, -h                  Print this help\n"
    );
}

fn build_config(cli: &CliArgs) -> Result<GeminiConfig, Box<dyn std::error::Error>> {
    let mut config = GeminiConfig::from_env()?
        .with_streaming(cli.streaming)
        .with_resilience(ResilienceConfig::default());
    if let Some(t) = cli.temperature {
        config = config.with_temperature(t);
    }
    if let Some(level) = cli.thinking_level {
        config = config.with_thinking(ThinkingConfig::level(level));
    } else if let Some(budget) = cli.thinking_budget {
        config = config.with_thinking(ThinkingConfig::budget(budget));
    }
    if cli.google_search {
        config = config.with_server_tool(GeminiServerTool::GoogleSearch);
    }
    if cli.code_exec {
        config = config.with_server_tool(GeminiServerTool::CodeExecution);
    }
    if cli.url_context {
        config = config.with_server_tool(GeminiServerTool::UrlContext);
    }
    Ok(config)
}

struct StreamPrinter {
    usage_slot: Arc<Mutex<Option<Usage>>>,
    state: Mutex<StreamPrinterState>,
}

struct StreamPrinterState {
    in_assistant_turn: bool,
    stdout: io::Stdout,
}

impl StreamPrinter {
    fn new(usage_slot: Arc<Mutex<Option<Usage>>>) -> Self {
        Self {
            usage_slot,
            state: Mutex::new(StreamPrinterState {
                in_assistant_turn: false,
                stdout: io::stdout(),
            }),
        }
    }
}

impl StreamPrinterState {
    fn ensure_prefix(&mut self) {
        if !self.in_assistant_turn {
            let _ = self.stdout.write_all(b"assistant> ");
            let _ = self.stdout.flush();
            self.in_assistant_turn = true;
        }
    }
}

impl LoopObserver for StreamPrinter {
    fn handle_event(&self, event: ObservedEvent) {
        let mut state = self.state.lock().unwrap();
        match event.event {
            AgentEvent::TurnStarted { .. } => state.in_assistant_turn = false,
            AgentEvent::ContentDelta(Delta::AppendText { chunk, .. }) => {
                state.ensure_prefix();
                let _ = state.stdout.write_all(chunk.as_bytes());
                let _ = state.stdout.flush();
            }
            AgentEvent::ContentDelta(Delta::CommitPart {
                part: Part::Text(t),
            }) if !state.in_assistant_turn => {
                state.ensure_prefix();
                let _ = state.stdout.write_all(t.text.as_bytes());
                let _ = state.stdout.flush();
            }
            AgentEvent::UsageUpdated(usage) => {
                drop(state);
                if let Ok(mut slot) = self.usage_slot.lock() {
                    *slot = Some(usage);
                }
            }
            AgentEvent::TurnFinished(_) => state.in_assistant_turn = false,
            _ => {}
        }
    }
}
