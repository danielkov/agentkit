# agentkit-plugins

Information-first support for the [Agent Plugins 1.0](https://agent-plugins.org/) package format.

The crate parses and validates `plugin.json`, discovers immediate Agent Skills, and exposes portable MCP declarations from `mcp.json`. It does not install plugins, create runtime data directories, launch MCP servers, or proxy plugin assets as capabilities.

```rust,no_run
use agentkit_plugins::AgentPlugin;
use agentkit_tool_skills::SkillRegistry;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let plugin = AgentPlugin::load("./plugins/acme-tools")?;
let skills = SkillRegistry::from_skill_dirs(plugin.skill_directories())
    .discover_skills()
    .await;

for server in plugin.mcp_servers() {
    println!("portable MCP declaration: {}", server.name);
}
# Ok(())
# }
```

MCP placeholder expansion and `PLUGIN_DATA` are intentionally deferred until a host chooses to materialize and launch an MCP configuration.
