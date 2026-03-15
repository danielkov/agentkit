#[cfg(feature = "capabilities")]
pub use agentkit_capabilities as capabilities;
#[cfg(feature = "compaction")]
pub use agentkit_compaction as compaction;
#[cfg(feature = "context")]
pub use agentkit_context as context;
#[cfg(feature = "core")]
pub use agentkit_core as core;
#[cfg(feature = "loop")]
pub use agentkit_loop as loop_;
#[cfg(feature = "mcp")]
pub use agentkit_mcp as mcp;
#[cfg(feature = "provider-openrouter")]
pub use agentkit_provider_openrouter as provider_openrouter;
#[cfg(feature = "reporting")]
pub use agentkit_reporting as reporting;
#[cfg(feature = "tool-fs")]
pub use agentkit_tool_fs as tool_fs;
#[cfg(feature = "tool-shell")]
pub use agentkit_tool_shell as tool_shell;
#[cfg(feature = "tools")]
pub use agentkit_tools_core as tools;

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
    pub use crate::tool_fs::*;
    #[cfg(feature = "tool-shell")]
    pub use crate::tool_shell::*;
    #[cfg(feature = "tools")]
    pub use crate::tools::*;
}
