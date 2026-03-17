use std::io::{self, Write};

use agentkit_core::{
    CancellationController, Item, ItemKind, MetadataMap, Part, SessionId, TextPart,
};
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let config = OpenRouterConfig::from_env()?;
    let adapter = OpenRouterAdapter::new(config)?;
    let cancellation = CancellationController::new();
    let agent = Agent::builder()
        .model(adapter)
        .cancellation(cancellation.handle())
        .build()?;
    let mut driver = agent
        .start(SessionConfig {
            session_id: SessionId::new("openrouter-chat"),
            metadata: MetadataMap::new(),
        })
        .await?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("openrouter-chat");
        println!("Type a prompt and press enter. Use /exit to quit. Press Ctrl-C to cancel the current turn.");
        repl(&mut driver, cancellation).await?;
    } else {
        let prompt = args.join(" ");
        submit_user_prompt(&mut driver, &prompt)?;
        run_turn(&mut driver, &cancellation).await?;
    }

    Ok(())
}

async fn repl<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    cancellation: CancellationController,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: agentkit_loop::ModelSession,
{
    let mut line = String::new();

    loop {
        print!("you> ");
        io::stdout().flush()?;
        line.clear();

        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }

        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "/exit" || prompt == "/quit" {
            break;
        }

        submit_user_prompt(driver, prompt)?;
        run_turn(driver, &cancellation).await?;
    }

    Ok(())
}

fn submit_user_prompt<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: agentkit_loop::ModelSession,
{
    driver.submit_input(vec![Item {
        id: None,
        kind: ItemKind::User,
        parts: vec![Part::Text(TextPart {
            text: prompt.into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    }])?;

    Ok(())
}

async fn run_turn<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
    cancellation: &CancellationController,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: agentkit_loop::ModelSession,
{
    let interrupt = cancellation.clone();
    let ctrl_c = tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        interrupt.interrupt();
    });
    let step = driver.next().await;
    ctrl_c.abort();

    match step? {
        LoopStep::Finished(result) => {
            if result.finish_reason == agentkit_core::FinishReason::Cancelled {
                eprintln!("turn cancelled");
            }
            for item in result.items {
                if item.kind != ItemKind::Assistant {
                    continue;
                }

                render_assistant_item(item.parts);
            }
            Ok(())
        }
        LoopStep::Interrupt(LoopInterrupt::AwaitingInput(_)) => Ok(()),
        LoopStep::Interrupt(LoopInterrupt::ApprovalRequest(request)) => {
            eprintln!("approval required: {}", request.summary);
            Ok(())
        }
        LoopStep::Interrupt(LoopInterrupt::AuthRequest(request)) => {
            eprintln!("auth required: {}", request.provider);
            Ok(())
        }
    }
}

fn render_assistant_item(parts: Vec<Part>) {
    let mut reasoning_sections = Vec::new();
    let mut output_sections = Vec::new();
    let mut side_channels = Vec::new();

    for part in parts {
        match part {
            Part::Text(text) => output_sections.push(text.text),
            Part::Reasoning(reasoning) => {
                if let Some(summary) = reasoning.summary {
                    reasoning_sections.push(summary);
                }
            }
            Part::ToolCall(call) => {
                side_channels.push(format!("[tool call] {} {}", call.name, call.input));
            }
            other => {
                side_channels.push(format!("[part {}]", part_kind_name(&other)));
            }
        }
    }

    println!("assistant>");

    if !reasoning_sections.is_empty() {
        println!("[reasoning]");
        for section in &reasoning_sections {
            println!("{section}");
        }
        println!();
    }

    if !output_sections.is_empty() {
        println!("[output]");
        for section in &output_sections {
            println!("{section}");
        }
    }

    if !side_channels.is_empty() {
        if !output_sections.is_empty() || !reasoning_sections.is_empty() {
            println!();
        }
        for line in side_channels {
            println!("{line}");
        }
    }
}

fn part_kind_name(part: &Part) -> &'static str {
    match part {
        Part::Text(_) => "text",
        Part::Media(_) => "media",
        Part::File(_) => "file",
        Part::Structured(_) => "structured",
        Part::Reasoning(_) => "reasoning",
        Part::ToolCall(_) => "tool_call",
        Part::ToolResult(_) => "tool_result",
        Part::Custom(_) => "custom",
    }
}
