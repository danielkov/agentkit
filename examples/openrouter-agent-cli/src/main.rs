use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use agentkit_compaction::{
    CompactionConfig, CompactionError, CompactionRequest, CompactionResult, Compactor,
    ItemCountTrigger,
};
use agentkit_context::{AgentsMd, ContextLoader, SkillsDirectory};
use agentkit_core::{Item, ItemKind, MetadataMap, Part, SessionId, TextPart};
use agentkit_loop::{Agent, LoopInterrupt, LoopStep, SessionConfig};
use agentkit_mcp::{
    McpServerConfig, McpServerId, McpServerManager, McpTransportBinding, StdioTransportConfig,
};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
use agentkit_reporting::{CompositeReporter, StdoutReporter};
use agentkit_tools_core::{
    CommandPolicy, CompositePermissionChecker, PathPolicy, PermissionCode, PermissionDecision,
    PermissionDenial, ToolRegistry,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const SYSTEM_PROMPT: &str = "\
You are a careful coding agent.
Use loaded context as authoritative project guidance.
Use fs.* tools for repository inspection and edits.
Use shell.exec only for simple read-only inspection commands when that is more appropriate than fs tools.
If an MCP tool is available and relevant, use it instead of guessing.
Prefer concise answers and avoid making claims you did not verify.
";

const DEFAULT_PROMPT: &str = "Use fs.read_file on ./Cargo.toml and report the workspace member count. Return only the integer.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();

    let args = Args::parse()?;
    if args.serve_mock_mcp {
        return run_mock_server().await;
    }

    let workspace_root = env::current_dir()?;
    let context_root = args.context_root.unwrap_or_else(|| workspace_root.clone());
    let context_items = ContextLoader::new()
        .with_source(
            AgentsMd::discover_all(&context_root).with_search_dir(context_root.join(".agent")),
        )
        .with_source(
            SkillsDirectory::from_dir(context_root.join("skills"))
                .with_dir(context_root.join(".agent/skills")),
        )
        .load()
        .await?;

    let config = OpenRouterConfig::from_env()?
        .with_temperature(0.0)
        .with_max_completion_tokens(512);
    let adapter = OpenRouterAdapter::new(config)?;
    let reporter =
        CompositeReporter::new().with_observer(StdoutReporter::new(io::stderr()).with_usage(false));

    let mut tools = ToolRegistry::new();
    merge_registry(&mut tools, agentkit_tool_fs::registry());
    merge_registry(&mut tools, agentkit_tool_shell::registry());

    let mut manager = if args.mcp_mock {
        let mut manager = McpServerManager::new().with_server(McpServerConfig::new(
            "mock",
            McpTransportBinding::Stdio(
                StdioTransportConfig::new(env::current_exe()?.display().to_string())
                    .with_arg("--serve-mock-mcp")
                    .with_env("MCP_SECRET", "LANTERN-SECRET-93B7"),
            ),
        ));
        manager.connect_server(&McpServerId::new("mock")).await?;
        merge_registry(&mut tools, manager.tool_registry());
        Some(manager)
    } else {
        None
    };

    let permissions = CompositePermissionChecker::new(PermissionDecision::Deny(PermissionDenial {
        code: PermissionCode::UnknownRequest,
        message: "tool action is not allowed by the mini CLI policy".into(),
        metadata: MetadataMap::new(),
    }))
    .with_policy(
        PathPolicy::new()
            .allow_root(workspace_root.clone())
            .require_approval_outside_allowed(false),
    )
    .with_policy(
        CommandPolicy::new()
            .allow_cwd(workspace_root.clone())
            .allow_executable("pwd")
            .allow_executable("ls")
            .allow_executable("cat")
            .require_approval_for_unknown(false),
    );

    let agent = Agent::builder()
        .model(adapter)
        .tools(tools)
        .permissions(permissions)
        .compaction(CompactionConfig::new(
            ItemCountTrigger::new(12),
            TailCompactor::new(8),
        ))
        .observer(reporter)
        .build()?;

    let mut driver = agent
        .start(SessionConfig {
            session_id: SessionId::new("openrouter-agent-cli"),
            metadata: MetadataMap::new(),
        })
        .await?;

    let mut input = vec![text_item(ItemKind::System, SYSTEM_PROMPT)];
    input.extend(context_items);
    input.push(text_item(ItemKind::User, &args.prompt));
    driver.submit_input(input)?;

    let result = run_to_completion(&mut driver).await;

    if let Some(manager) = manager.as_mut() {
        let _ = manager.disconnect_server(&McpServerId::new("mock")).await;
    }

    result
}

struct Args {
    context_root: Option<PathBuf>,
    mcp_mock: bool,
    serve_mock_mcp: bool,
    prompt: String,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = env::args().skip(1);
        let mut context_root = None;
        let mut mcp_mock = false;
        let mut serve_mock_mcp = false;
        let mut prompt_parts = Vec::new();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--context-root" => {
                    let value = args.next().ok_or("missing value for --context-root")?;
                    context_root = Some(PathBuf::from(value));
                }
                "--mcp-mock" => mcp_mock = true,
                "--serve-mock-mcp" => serve_mock_mcp = true,
                _ => {
                    prompt_parts.push(arg);
                    prompt_parts.extend(args);
                    break;
                }
            }
        }

        Ok(Self {
            context_root,
            mcp_mock,
            serve_mock_mcp,
            prompt: if prompt_parts.is_empty() {
                DEFAULT_PROMPT.into()
            } else {
                prompt_parts.join(" ")
            },
        })
    }
}

#[derive(Clone)]
struct TailCompactor {
    keep_recent_items: usize,
}

impl TailCompactor {
    fn new(keep_recent_items: usize) -> Self {
        Self { keep_recent_items }
    }
}

#[async_trait]
impl Compactor for TailCompactor {
    async fn compact(
        &self,
        request: CompactionRequest,
    ) -> Result<CompactionResult, CompactionError> {
        let original_len = request.transcript.len();
        let mut preserved = Vec::new();
        let mut recent = Vec::new();

        for item in request.transcript {
            match item.kind {
                ItemKind::System | ItemKind::Context => preserved.push(item),
                _ => recent.push(item),
            }
        }

        let keep_from = recent.len().saturating_sub(self.keep_recent_items);
        preserved.extend(recent.into_iter().skip(keep_from));
        let replaced_items = original_len.saturating_sub(preserved.len());

        Ok(CompactionResult {
            replaced_items,
            transcript: preserved,
            metadata: MetadataMap::new(),
        })
    }
}

fn merge_registry(target: &mut ToolRegistry, source: ToolRegistry) {
    for tool in source.tools() {
        target.register_arc(tool);
    }
}

fn text_item(kind: ItemKind, text: &str) -> Item {
    Item {
        id: None,
        kind,
        parts: vec![Part::Text(TextPart {
            text: text.into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    }
}

async fn run_to_completion<S>(
    driver: &mut agentkit_loop::LoopDriver<S>,
) -> Result<(), Box<dyn Error>>
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
        LoopStep::Interrupt(LoopInterrupt::AuthRequest(request)) => {
            Err(format!("unexpected auth request from {}", request.provider).into())
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

async fn run_mock_server() -> Result<(), Box<dyn Error>> {
    let secret = env::var("MCP_SECRET").unwrap_or_else(|_| "LANTERN-SECRET-93B7".into());
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut writer = stdout;
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            break;
        }
        let message: Value = serde_json::from_str(line.trim())?;
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            continue;
        };

        match method {
            "initialize" => {
                write_response(
                    &mut writer,
                    message["id"].clone(),
                    json!({ "capabilities": {} }),
                )
                .await?;
            }
            "notifications/initialized" => {}
            "tools/list" => {
                write_response(
                    &mut writer,
                    message["id"].clone(),
                    json!({
                        "tools": [{
                            "name": "reveal_secret",
                            "description": "Reveal the sealed launch code.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": false
                            }
                        }]
                    }),
                )
                .await?;
            }
            "resources/list" => {
                write_response(
                    &mut writer,
                    message["id"].clone(),
                    json!({ "resources": [] }),
                )
                .await?;
            }
            "prompts/list" => {
                write_response(&mut writer, message["id"].clone(), json!({ "prompts": [] }))
                    .await?;
            }
            "tools/call" => {
                let name = message["params"]["name"].as_str().unwrap_or_default();
                let result = if name == "reveal_secret" {
                    json!({
                        "content": [{
                            "type": "text",
                            "text": secret,
                        }]
                    })
                } else {
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("unknown tool: {name}"),
                        }]
                    })
                };
                write_response(&mut writer, message["id"].clone(), result).await?;
            }
            _ => {
                if let Some(id) = message.get("id").cloned() {
                    write_error(&mut writer, id, -32601, format!("unknown method: {method}"))
                        .await?;
                }
            }
        }
    }

    Ok(())
}

async fn write_response(
    writer: &mut tokio::io::Stdout,
    id: Value,
    result: Value,
) -> Result<(), Box<dyn Error>> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    writer.write_all(payload.to_string().as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

async fn write_error(
    writer: &mut tokio::io::Stdout,
    id: Value,
    code: i64,
    message: String,
) -> Result<(), Box<dyn Error>> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        }
    });
    writer.write_all(payload.to_string().as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}
