use std::env;
use std::io;

use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TextPart};
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit_reporting::{CompositeReporter, StdoutReporter};

const SYSTEM_PROMPT: &str = "\
You are a careful repository assistant.
When the user asks about files or project contents, inspect the repository with the available fs.* tools instead of guessing.
Prefer concise answers.
If you use tools, use relative paths from the current working directory when possible.
";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let prompt = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.trim().is_empty() {
        return Err("usage: cargo run -p openrouter-coding-agent -- '<prompt>'".into());
    }

    let config = OpenRouterConfig::from_env()?;
    let adapter = OpenRouterAdapter::new(config)?;
    let tools = agentkit_tool_fs::registry();
    let reporter =
        CompositeReporter::new().with_observer(StdoutReporter::new(io::stderr()).with_usage(false));

    let agent = Agent::builder()
        .model(adapter)
        .tools(tools)
        .observer(reporter)
        .build()?;

    let mut driver = agent
        .start(SessionConfig {
            session_id: SessionId::new("openrouter-coding-agent"),
            metadata: MetadataMap::new(),
        })
        .await?;

    driver.submit_input(vec![system_item(), user_item(&prompt)])?;
    run_to_completion(&mut driver).await
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
            Part::ToolCall(call) => {
                println!("[tool call] {} {}", call.name, call.input);
            }
            Part::Structured(value) => {
                if !saw_output {
                    println!("[output]");
                    saw_output = true;
                }
                println!("{}", value.value);
            }
            Part::Media(_) | Part::File(_) | Part::ToolResult(_) | Part::Custom(_) => {}
        }
    }
}
