use std::env;
use std::io;
use std::path::PathBuf;

use agentkit_context::{AgentsMd, ContextLoader, SkillsDirectory};
use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TextPart};
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit_reporting::{CompositeReporter, StdoutReporter};

const SYSTEM_PROMPT: &str = "\
You are a careful repository assistant.
Use the loaded context items as authoritative project guidance.
If the user asks about project instructions or skills, answer from the provided context before making assumptions.
Prefer concise answers.
";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let args = Args::parse()?;
    let context_items = ContextLoader::new()
        .with_source(AgentsMd::discover(&args.context_root))
        .with_source(SkillsDirectory::from_dir(args.context_root.join("skills")))
        .load()
        .await?;

    let config = OpenRouterConfig::from_env()?;
    let adapter = OpenRouterAdapter::new(config)?;
    let reporter =
        CompositeReporter::new().with_observer(StdoutReporter::new(io::stderr()).with_usage(false));

    let agent = Agent::builder().model(adapter).observer(reporter).build()?;

    let mut driver = agent
        .start(SessionConfig {
            session_id: SessionId::new("openrouter-context-agent"),
            metadata: MetadataMap::new(),
        })
        .await?;

    let mut input = vec![system_item()];
    input.extend(context_items);
    input.push(user_item(&args.prompt));
    driver.submit_input(input)?;

    run_to_completion(&mut driver).await
}

struct Args {
    context_root: PathBuf,
    prompt: String,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut args = env::args().skip(1);
        let mut context_root: Option<PathBuf> = None;
        let mut prompt_parts = Vec::new();

        while let Some(arg) = args.next() {
            if arg == "--context-root" {
                let value = args.next().ok_or("missing value for --context-root")?;
                context_root = Some(PathBuf::from(value));
                continue;
            }
            prompt_parts.push(arg);
            prompt_parts.extend(args);
            break;
        }

        let prompt = prompt_parts.join(" ");
        if prompt.trim().is_empty() {
            return Err(
                "usage: cargo run -p openrouter-context-agent -- [--context-root <path>] '<prompt>'"
                    .into(),
            );
        }

        Ok(Self {
            context_root: context_root.unwrap_or(env::current_dir()?),
            prompt,
        })
    }
}

fn system_item() -> Item {
    Item {
        id: None,
        kind: ItemKind::System,
        parts: vec![Part::Text(TextPart {
            text: SYSTEM_PROMPT.into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    }
}

fn user_item(prompt: &str) -> Item {
    Item {
        id: None,
        kind: ItemKind::User,
        parts: vec![Part::Text(TextPart {
            text: prompt.into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    }
}

async fn run_to_completion<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: agentkit_loop::ModelSession,
{
    match driver.next().await? {
        LoopStep::Finished(result) => {
            for item in result.items {
                if item.kind == ItemKind::Assistant {
                    print_assistant_item(item);
                }
            }
            Ok(())
        }
        LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(request)) => {
            Err(format!("unexpected approval request: {}", request.summary).into())
        }
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => {
            Err("loop requested more input before finishing".into())
        }
    }
}

fn print_assistant_item(item: Item) {
    let mut saw_output = false;

    for part in item.parts {
        match part {
            Part::Text(text) => {
                if !saw_output {
                    println!("[output]");
                    saw_output = true;
                }
                println!("{}", text.text);
            }
            Part::Reasoning(reasoning) => {
                if let Some(summary) = reasoning.summary {
                    println!("[reasoning]");
                    println!("{summary}");
                }
            }
            Part::Structured(value) => {
                if !saw_output {
                    println!("[output]");
                    saw_output = true;
                }
                println!("{}", value.value);
            }
            Part::ToolCall(call) => {
                println!("[tool call] {} {}", call.name, call.input);
            }
            Part::Media(_) | Part::File(_) | Part::ToolResult(_) | Part::Custom(_) => {}
        }
    }
}
