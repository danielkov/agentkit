# agentkit-tool-skills

Progressive Agent Skills discovery and activation for agentkit.

This crate discovers `SKILL.md` files, builds a lightweight catalog for the
model, and exposes an `activate_skill` tool that loads full skill instructions
on demand.

## What it provides

- recursive skill discovery from one or more roots
- frontmatter parsing for skill metadata
- per-session activation tracking to avoid duplicate loads
- progressive disclosure so the model sees descriptions first and bodies later

## Example

```rust,no_run
use agentkit_tool_skills::SkillRegistry;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let registry = SkillRegistry::from_paths(vec![
    "./skills".into(),
    "./.agents/skills".into(),
])
.discover_skills()
.await;

let mut tools = agentkit_tools_core::ToolRegistry::new();
registry.register_tool(&mut tools);
# Ok(())
# }
```
