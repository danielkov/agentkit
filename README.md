# agentkit

`agentkit` is a Rust toolkit for building LLM agent applications such as coding agents, assistant CLIs, and multi-agent tools.

The project is intentionally split into small crates behind feature flags so hosts can pull in only the pieces they need.

## Current status

`agentkit` is past the design-only stage. The repo currently includes working implementations for:

- normalized transcript, content-part, and delta types with fluent builders
- a runtime-agnostic loop driver with blocking interrupts for approval, auth, and input
- trait-based tools, permissions, approvals, and auth handoff
- built-in filesystem, shell, and skills tools
- context loading for `AGENTS.md` and skills directories
- MCP transports, discovery, tool/resource/prompt adapters, auth replay, and lifecycle management
- reporting observers
- compaction triggers, strategy pipelines, and backend-driven semantic compaction
- async task management with foreground/background scheduling, routing policies, and detach-after-timeout
- optional turn cancellation with resumable sessions
- prompt caching with automatic and explicit strategies, retention hints, and cache keys
- a generic completions adapter base for building provider crates
- provider adapters for OpenRouter, OpenAI, Anthropic, Cerebras, Ollama, vLLM, Groq, and Mistral

The repo also ships multiple examples that exercise these pieces end to end.

## Crates

- `agentkit-core`
  - transcript, parts, deltas, IDs, usage, and cancellation primitives
- `agentkit-capabilities`
  - lower-level invocable/resource/prompt abstraction
- `agentkit-tools-core`
  - tools, registry, executor, permissions, approvals, auth requests
- `agentkit-loop`
  - model session abstraction, driver, interrupts, tool roundtrips
- `agentkit-context`
  - `AGENTS.md` and skills loading
- `agentkit-mcp`
  - MCP transports, discovery, lifecycle, auth, replay, adapters
- `agentkit-reporting`
  - loop observers and reporting adapters
- `agentkit-compaction`
  - compaction triggers, strategies, pipelines, backend hooks
- `agentkit-task-manager`
  - task scheduling for tool execution: foreground, background, and detach-after-timeout routing
- `agentkit-tool-fs`
  - filesystem tools
- `agentkit-tool-shell`
  - shell execution tool
- `agentkit-tool-skills`
  - progressive skill discovery and activation
- `agentkit-http`
  - HTTP transport abstraction (`HttpClient`, `Http`, `HttpRequestBuilder`) with a default reqwest-backed implementation and an optional `reqwest-middleware` adapter
- `agentkit-adapter-completions`
  - generic chat completions adapter base for building provider crates
- `agentkit-provider-openrouter`
  - OpenRouter adapter
- `agentkit-provider-openai`
  - OpenAI adapter
- `agentkit-provider-anthropic`
  - Anthropic Messages API adapter with streaming, prompt caching, extended thinking, and server-side tools (web search, web fetch, code execution)
- `agentkit-provider-cerebras`
  - Cerebras Inference API adapter with streaming, reasoning, strict JSON schema, compression (msgpack/gzip), predicted outputs, service tiers, and Files + Batch API
- `agentkit-provider-ollama`
  - Ollama adapter
- `agentkit-provider-vllm`
  - vLLM adapter
- `agentkit-provider-groq`
  - Groq adapter
- `agentkit-provider-mistral`
  - Mistral adapter
- `agentkit`
  - umbrella crate with feature-gated re-exports

## Built-in tools today

Filesystem:

- `fs.read_file`
  - supports optional `from` / `to` line ranges
- `fs.write_file`
- `fs.replace_in_file`
- `fs.move`
- `fs.delete`
- `fs.list_directory`
- `fs.create_directory`

Shell:

- `shell.exec`

The filesystem crate also supports session-scoped read-before-write enforcement through `FileSystemToolResources` and `FileSystemToolPolicy`.

## Quick start

1. Set your OpenRouter API key and model — either through environment variables or directly in code via `OpenRouterConfig::new(api_key, model)`.
2. Run one of the examples.

Example commands:

```bash
cargo run -p openrouter-chat -- "hello"
```

```bash
cargo run -p openrouter-coding-agent -- \
  "Use fs.read_file on ./Cargo.toml and return only the workspace member count as an integer."
```

```bash
cargo run -p openrouter-agent-cli -- --mcp-mock \
  "Return only the secret from the MCP tool."
```

## Example progression

- `openrouter-chat`
  - minimal chat loop
  - now supports `Ctrl-C` turn cancellation
- `openrouter-coding-agent`
  - one-shot coding-oriented prompt runner with filesystem tools
- `openrouter-context-agent`
  - context loading from `AGENTS.md` and skills
- `openrouter-mcp-tool`
  - MCP tool discovery and invocation
- `openrouter-subagent-tool`
  - custom tool that runs a nested agent
- `openrouter-compaction-agent`
  - structural, semantic, and hybrid compaction
  - semantic compaction uses a nested agent as the backend
- `openrouter-parallel-agent`
  - async task manager with foreground fs tools and detach-after-timeout shell tools
  - `TaskManagerHandle` event stream printed to stderr
- `openrouter-agent-cli`
  - combined example using context, tools, shell, MCP, compaction, and reporting
- `anthropic-chat`
  - streaming REPL against Anthropic's Messages API, with server tools
    (`--web-search`, `--web-fetch`, `--code-exec`), extended thinking
    (`--thinking`), and a streaming / buffered toggle (`--streaming` /
    `--no-streaming`)
- `cerebras-chat`
  - interactive REPL against Cerebras `/v1/chat/completions`; CLI flags
    cover every `CerebrasConfig` knob (sampling, reasoning, response
    format, compression, service tier, predicted outputs, local tools)
    and slash commands (`/show`, `/usage`, `/ratelimit`, `/headers`,
    `/models`, `/reset`) surface runtime state
- `cerebras-batch`
  - one-shot CLI over the Cerebras Files + Batch APIs: `files upload|list|get|content|delete`,
    `batches create|submit|list|get|cancel|wait`, and `run` to submit → wait → dump outputs

## Examples

### Minimal chat

Build an agent with a provider adapter, start a session with prompt caching, and drive the loop:

```rust
use agentkit_core::{Item, ItemKind};
use agentkit_loop::{Agent, LoopStep, PromptCacheRequest, PromptCacheRetention, SessionConfig};
use agentkit_provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};

let adapter = OpenRouterAdapter::new(
    OpenRouterConfig::new("sk-or-v1-...", "openrouter/auto")
        .with_temperature(0.0),
)?;

let agent = Agent::builder().model(adapter).build()?;

let mut driver = agent
    .start(SessionConfig::new("chat").with_cache(
        PromptCacheRequest::automatic().with_retention(PromptCacheRetention::Short),
    ))
    .await?;

driver.submit_input(vec![Item::text(ItemKind::User, "Hello!")])?;

match driver.next().await? {
    LoopStep::Finished(result) => { /* render result.items */ }
    LoopStep::Interrupt(interrupt) => { /* resolve approval, auth, or input */ }
}
```

### Tools and permissions

Register filesystem tools with a path-scoped permission policy:

```rust
use agentkit_core::MetadataMap;
use agentkit_loop::Agent;
use agentkit_tools_core::{
    CompositePermissionChecker, PathPolicy, PermissionCode, PermissionDecision, PermissionDenial,
};

let permissions = CompositePermissionChecker::new(PermissionDecision::Deny(PermissionDenial {
    code: PermissionCode::UnknownRequest,
    message: "not allowed by policy".into(),
    metadata: MetadataMap::new(),
}))
.with_policy(
    PathPolicy::new()
        .allow_root(std::env::current_dir()?)
        .require_approval_outside_allowed(false),
);

let agent = Agent::builder()
    .model(adapter)
    .tools(agentkit_tool_fs::registry())
    .permissions(permissions)
    .build()?;
```

### Reporting

Compose multiple observers to log output, track usage, and record transcripts:

```rust
use agentkit_reporting::{CompositeReporter, JsonlReporter, StdoutReporter, UsageReporter};

let reporter = CompositeReporter::new()
    .with_observer(StdoutReporter::new(std::io::stderr()).with_usage(false))
    .with_observer(JsonlReporter::new(Vec::new()))
    .with_observer(UsageReporter::new());

let agent = Agent::builder()
    .model(adapter)
    .observer(reporter)
    .build()?;
```

### Compaction

Configure structural compaction that drops reasoning and failed tool results, then keeps the most recent items:

```rust
use agentkit_compaction::{
    CompactionConfig, CompactionPipeline, DropFailedToolResultsStrategy,
    DropReasoningStrategy, ItemCountTrigger, KeepRecentStrategy,
};
use agentkit_core::ItemKind;

let compaction = CompactionConfig::new(
    ItemCountTrigger::new(10),
    CompactionPipeline::new()
        .with_strategy(DropReasoningStrategy::new())
        .with_strategy(DropFailedToolResultsStrategy::new())
        .with_strategy(
            KeepRecentStrategy::new(8)
                .preserve_kind(ItemKind::System)
                .preserve_kind(ItemKind::Context),
        ),
);

let agent = Agent::builder()
    .model(adapter)
    .compaction(compaction)
    .build()?;
```

### Async task management

Route shell commands to background execution with automatic detach-after-timeout:

```rust
use agentkit_task_manager::{AsyncTaskManager, RoutingDecision};
use std::time::Duration;

let task_manager = AsyncTaskManager::new().routing(|req: &agentkit_tools_core::ToolRequest| {
    if req.tool_name.0 == "shell.exec" {
        RoutingDecision::ForegroundThenDetachAfter(Duration::from_secs(5))
    } else {
        RoutingDecision::Foreground
    }
});

let agent = Agent::builder()
    .model(adapter)
    .tools(tools)
    .task_manager(task_manager)
    .build()?;
```

## Feature flags

The umbrella crate re-exports subcrates behind feature flags.

Default flags:

- `core`
- `capabilities`
- `tools`
- `task-manager`
- `loop`
- `reporting`

Optional flags:

- `compaction`
- `context`
- `mcp`
- `adapter-completions`
- `provider-openrouter`
- `provider-openai`
- `provider-anthropic`
- `provider-cerebras`
- `provider-ollama`
- `provider-vllm`
- `provider-groq`
- `provider-mistral`
- `tool-fs`
- `tool-shell`
- `tool-skills`

More detail is in [docs/feature-flags.md](./docs/feature-flags.md).

## Docs

- [docs/getting-started.md](./docs/getting-started.md)
- [docs/architecture.md](./docs/architecture.md)
- [docs/core.md](./docs/core.md)
- [docs/tools.md](./docs/tools.md)
- [docs/loop.md](./docs/loop.md)
- [docs/permissions.md](./docs/permissions.md)
- [docs/capabilities.md](./docs/capabilities.md)
- [docs/context.md](./docs/context.md)
- [docs/mcp.md](./docs/mcp.md)
- [docs/compaction.md](./docs/compaction.md)
- [docs/reporting.md](./docs/reporting.md)
- [docs/feature-flags.md](./docs/feature-flags.md)
- [docs/README.md](./docs/README.md)
