//! # agentkit
//!
//! Composable building blocks for agentic loops.
//!
//! `agentkit` is a feature-gated umbrella crate that re-exports every crate in
//! the agentkit workspace. Enable only the features you need and access them
//! through a single dependency.
//!
//! ## Default features
//!
//! The following modules are available with default features enabled:
//!
//! | Feature | Module | Contents |
//! |---|---|---|
//! | `core` | [`core`] | Shared types: [`core::Item`], [`core::Part`], [`core::SessionId`], [`core::Usage`], cancellation primitives |
//! | `capabilities` | [`capabilities`] | Capability traits: [`capabilities::Invocable`], [`capabilities::CapabilityProvider`] |
//! | `tools` | [`tools`] | Tool abstractions: [`tools::Tool`], [`tools::ToolRegistry`], [`tools::ToolSpec`], permission types |
//! | `loop` | [`loop_`] | Agent loop: [`loop_::Agent`], [`loop_::AgentBuilder`], [`loop_::LoopDriver`], [`loop_::LoopStep`] |
//! | `reporting` | [`reporting`] | Loop observers: [`reporting::StdoutReporter`], [`reporting::JsonlReporter`], [`reporting::UsageReporter`] |
//!
//! ## Optional features
//!
//! | Feature | Module | Contents |
//! |---|---|---|
//! | `compaction` | [`compaction`] | Transcript compaction triggers, strategies, and pipelines |
//! | `context` | [`context`] | `AGENTS.md` discovery and context loading |
//! | `mcp` | [`mcp`] | Model Context Protocol (MCP) server connections |
//! | `provider-openrouter` | [`provider_openrouter`] | OpenRouter [`loop_::ModelAdapter`] implementation |
//! | `tool-fs` | [`tool_fs`] | Filesystem tools (read, write, edit, move, delete, list, mkdir) |
//! | `tool-shell` | [`tool_shell`] | Shell execution tool (`shell.exec`) |
//! | `tool-skills` | [`tool_skills`] | Progressive Agent Skills discovery and activation |
//!
//! ## Example: building and running an agent
//!
//! This example uses the `provider-openrouter` and `reporting` features to
//! build a minimal agent, submit a user message, and drive the loop until the
//! model finishes its turn.
//!
//! ```rust,ignore
//! use agentkit::core::{Item, ItemKind, MetadataMap, Part, SessionId, TextPart};
//! use agentkit::loop_::{Agent, LoopStep, SessionConfig};
//! use agentkit::provider_openrouter::{OpenRouterAdapter, OpenRouterConfig};
//! use agentkit::reporting::StdoutReporter;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let adapter = OpenRouterAdapter::new(OpenRouterConfig::from_env()?)?;
//!
//!     let agent = Agent::builder()
//!         .model(adapter)
//!         .observer(StdoutReporter::new(std::io::stdout()))
//!         .build()?;
//!
//!     let mut driver = agent
//!         .start(SessionConfig {
//!             session_id: SessionId::new("demo"),
//!             metadata: MetadataMap::new(),
//!         })
//!         .await?;
//!
//!     driver.submit_input(vec![Item {
//!         id: None,
//!         kind: ItemKind::User,
//!         parts: vec![Part::Text(TextPart {
//!             text: "What is the capital of France?".into(),
//!             metadata: MetadataMap::new(),
//!         })],
//!         metadata: MetadataMap::new(),
//!     }])?;
//!
//!     loop {
//!         match driver.next().await? {
//!             LoopStep::Finished(result) => {
//!                 println!("Finished: {:?}", result.finish_reason);
//!                 break;
//!             }
//!             LoopStep::Interrupt(interrupt) => {
//!                 // Resolve the interrupt (approval, auth, or input) then continue.
//!                 println!("Interrupt: {interrupt:?}");
//!                 break; // a real app would resolve and loop
//!             }
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Example: composing reporters
//!
//! Multiple [`loop_::LoopObserver`] implementations can be combined through
//! [`reporting::CompositeReporter`] so that a single loop feeds several
//! observers at once.
//!
//! ```rust
//! use agentkit::reporting::{
//!     CompositeReporter, JsonlReporter, UsageReporter, TranscriptReporter,
//! };
//!
//! let reporter = CompositeReporter::new()
//!     .with_observer(JsonlReporter::new(Vec::new()))
//!     .with_observer(UsageReporter::new())
//!     .with_observer(TranscriptReporter::new());
//! ```

/// Core types shared by every agentkit crate.
///
/// Provides the fundamental data model: [`core::Item`] and [`core::Part`] for
/// representing conversation content, ID newtypes such as [`core::SessionId`]
/// and [`core::TurnId`], [`core::Usage`] for token/cost tracking, and
/// cancellation primitives ([`core::CancellationController`],
/// [`core::TurnCancellation`]).
#[cfg(feature = "core")]
pub use agentkit_core as core;

/// Capability abstractions for tools, resources, and prompts.
///
/// Defines the [`capabilities::Invocable`] trait for callable capabilities,
/// [`capabilities::ResourceProvider`] and [`capabilities::PromptProvider`] for
/// serving resources and prompts, and [`capabilities::CapabilityProvider`] for
/// bundling all three into a single object. These traits form the foundation
/// that the [`tools`] module builds on.
#[cfg(feature = "capabilities")]
pub use agentkit_capabilities as capabilities;

/// Tool definitions, registry, permission checking, and execution.
///
/// Contains the [`tools::Tool`] trait, [`tools::ToolRegistry`] for collecting
/// tools, [`tools::ToolSpec`] for declaring tool schemas, and the permission
/// system ([`tools::PermissionChecker`], [`tools::PermissionPolicy`],
/// [`tools::CompositePermissionChecker`]). The [`tools::BasicToolExecutor`]
/// bridges the registry with the agent loop.
#[cfg(feature = "tools")]
pub use agentkit_tools_core as tools;

/// Agent loop orchestration: sessions, turns, tool dispatch, and interrupts.
///
/// The main entry point is [`loop_::Agent`], built via [`loop_::AgentBuilder`].
/// Calling [`loop_::Agent::start`] yields a [`loop_::LoopDriver`] that
/// produces [`loop_::LoopStep`]s -- either a finished turn or an interrupt
/// (approval, auth, or input request) that the host must resolve before the
/// loop continues. Also defines [`loop_::ModelAdapter`], the trait that
/// provider crates implement.
///
/// Re-exported as `loop_` (with trailing underscore) because `loop` is a
/// reserved keyword in Rust.
#[cfg(feature = "loop")]
pub use agentkit_loop as loop_;

/// Loop observers for logging, usage tracking, and transcript recording.
///
/// Provides [`reporting::StdoutReporter`] for human-readable terminal output,
/// [`reporting::JsonlReporter`] for machine-readable JSONL streams,
/// [`reporting::UsageReporter`] for aggregated token/cost totals,
/// [`reporting::TranscriptReporter`] for a growing snapshot of conversation
/// items, and [`reporting::CompositeReporter`] for fanning out events to
/// multiple observers.
#[cfg(feature = "reporting")]
pub use agentkit_reporting as reporting;

/// Transcript compaction triggers, strategies, and pipelines.
///
/// Use this module to keep transcripts from growing without bound. Combine
/// [`compaction::CompactionTrigger`]s (which decide *when* to compact) with
/// [`compaction::CompactionStrategy`]s (which decide *how*) through a
/// [`compaction::CompactionPipeline`], and hand the resulting
/// [`compaction::CompactionConfig`] to the agent builder.
///
/// Requires the `compaction` feature.
#[cfg(feature = "compaction")]
pub use agentkit_compaction as compaction;

/// Context loaders for `AGENTS.md` files.
///
/// Discovers and loads project-level agent instructions into
/// [`core::Item`]s with [`core::ItemKind::Context`]. See
/// [`context::ContextLoader`] and [`context::AgentsMd`].
///
/// Requires the `context` feature.
#[cfg(feature = "context")]
pub use agentkit_context as context;

/// Model Context Protocol (MCP) server connections.
///
/// Connects to MCP servers over stdio or SSE transports, discovers their
/// tools, resources, and prompts, and exposes them as agentkit
/// [`capabilities::CapabilityProvider`]s and [`tools::Tool`] implementations
/// that plug directly into the agent loop.
///
/// Requires the `mcp` feature.
#[cfg(feature = "mcp")]
pub use agentkit_mcp as mcp;

/// OpenRouter [`loop_::ModelAdapter`] implementation.
///
/// Provides [`provider_openrouter::OpenRouterAdapter`] and
/// [`provider_openrouter::OpenRouterConfig`] for connecting the agent loop to
/// any model available through the [OpenRouter](https://openrouter.ai) API.
/// Supports streaming, tool calls, usage reporting, and cancellation.
///
/// Requires the `provider-openrouter` feature.
#[cfg(feature = "provider-openrouter")]
pub use agentkit_provider_openrouter as provider_openrouter;

/// Filesystem tools: read, write, edit, move, delete, list, and mkdir.
///
/// Call [`tool_fs::registry()`] to get a [`tools::ToolRegistry`] pre-loaded
/// with all filesystem tools. Each tool integrates with the permission system
/// via [`tools::FileSystemPermissionRequest`].
///
/// Requires the `tool-fs` feature.
#[cfg(feature = "tool-fs")]
pub use agentkit_tool_fs as tool_fs;

/// Shell execution tool (`shell.exec`).
///
/// Call [`tool_shell::registry()`] to get a [`tools::ToolRegistry`] containing
/// the shell execution tool. Supports custom working directories, environment
/// variables, timeouts, and cooperative turn cancellation.
///
/// Requires the `tool-shell` feature.
#[cfg(feature = "tool-shell")]
pub use agentkit_tool_shell as tool_shell;

/// Agent Skills tool for progressive skill discovery and activation.
///
/// Provides [`tool_skills::SkillRegistry`] which discovers `SKILL.md` files
/// and exposes an `activate_skill` tool for on-demand loading. Skills are
/// listed in the tool description (catalog tier) and their full content is
/// returned only when the model activates them.
///
/// Requires the `tool-skills` feature.
#[cfg(feature = "tool-skills")]
pub use agentkit_tool_skills as tool_skills;

/// Convenience re-exports from all enabled feature modules.
///
/// Pulls in every public item from every enabled module via glob imports.
/// Useful for quick prototyping but may cause name collisions in larger
/// projects -- prefer qualified imports (e.g. `agentkit::core::Item`) for
/// production code.
pub mod prelude {
    #[cfg(feature = "capabilities")]
    pub use crate::capabilities::*;
    #[cfg(feature = "compaction")]
    pub use crate::compaction::*;
    #[cfg(feature = "context")]
    pub use crate::context::*;
    #[cfg(feature = "core")]
    pub use crate::core::*;
    #[cfg(feature = "loop")]
    pub use crate::loop_::*;
    #[cfg(feature = "mcp")]
    pub use crate::mcp::*;
    #[cfg(feature = "provider-openrouter")]
    pub use crate::provider_openrouter::*;
    #[cfg(feature = "reporting")]
    pub use crate::reporting::*;
    #[cfg(feature = "tool-fs")]
    pub use crate::tool_fs::{
        registry as fs_registry, CreateDirectoryTool, DeleteTool, FileSystemToolError,
        FileSystemToolPolicy, FileSystemToolResources, ListDirectoryTool, MoveTool, ReadFileTool,
        ReplaceInFileTool, WriteFileTool,
    };
    #[cfg(feature = "tool-shell")]
    pub use crate::tool_shell::{registry as shell_registry, ShellExecTool};
    #[cfg(feature = "tools")]
    pub use crate::tools::*;
}
