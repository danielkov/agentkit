# Agent Plugins

`agentkit-plugins` parses and validates [Agent Plugins 1.0](https://agent-plugins.org/) packages without turning a plugin into a parallel capability system.

```text
plugin/
├── plugin.json
├── skills/*/SKILL.md
└── mcp.json
```

## Information-first loading

```rust,ignore
use agentkit_plugins::AgentPlugin;
use agentkit_tool_skills::SkillRegistry;

let plugin = AgentPlugin::load("./plugins/acme")?;

let skills = SkillRegistry::from_skill_dirs(plugin.skill_directories())
    .discover_skills()
    .await;

for diagnostic in plugin.diagnostics() {
    eprintln!("{:?}: {}", diagnostic.kind, diagnostic.message);
}

for server in plugin.mcp_servers() {
    // The host chooses how and whether to map this portable declaration into
    // an agentkit_mcp::McpServerConfig.
    println!("{}: {:?}", server.name, server.transport);
}
```

The loader returns validated manifest metadata, exact skill directories, raw portable MCP declarations, opaque client-extension data, and structured diagnostics. Invalid skills and MCP entries are isolated according to the standard.

## Deliberate boundaries

The crate does not install or update plugins, choose permissions, create a `SkillRegistry`, launch MCP servers, expand MCP placeholders, or choose a `PLUGIN_DATA` directory. Those are host/runtime decisions. `PLUGIN_DATA` is only needed if a host later launches a plugin's stdio MCP server; it is a writable persistent directory selected for that installed plugin instance.

Use the umbrella crate's `plugins` feature to access the parser as `agentkit::plugins`. The feature remains independent of `tool-skills` and `mcp`, allowing consumers to inspect packages without pulling in either runtime.
