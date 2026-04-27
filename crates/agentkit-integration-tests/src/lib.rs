//! Test support for agentkit end-to-end integration tests.
//!
//! Two pieces:
//!
//! - [`mock_model`] — a scriptable, introspectable [`ModelAdapter`] used as a
//!   stand-in for a real provider. Tests enqueue a sequence of "turn scripts"
//!   describing what the model should emit (text, tool calls, finish), and
//!   inspect every [`TurnRequest`] the loop hands the model to assert that
//!   the transcript and advertised tool catalog look right.
//!
//! - [`mcp_server`] — an in-memory rmcp server bound to a tokio duplex,
//!   plus a ready-made [`McpHandlerConfig`]/[`McpServerManager`] wiring so
//!   tests can exercise the full agentkit ↔ rmcp ↔ MCP server stack
//!   without spawning a child process.

pub mod http_mcp_server;
pub mod mcp_server;
pub mod mock_model;
pub mod mock_tool;
pub mod snapshot;
