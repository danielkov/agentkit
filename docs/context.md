# agentkit-context

`agentkit-context` loads prompt/context material into normalized `Item`s.

## Built-in sources

### `AgentsMd`

Use `AgentsMd` for repository or workspace instructions.

Supported modes:

- `AgentsMd::discover(path)`
  - nearest ancestor `AGENTS.md`
- `AgentsMd::discover_all(path)`
  - stacked ancestor `AGENTS.md` files from root to nearest match

Supported configuration:

- `with_file_name(...)`
- `with_search_dir(...)`
- `with_path(...)`
- `with_mode(...)`

This makes it possible to combine:

- ancestor discovery
- explicit extra search directories
- explicit file paths

without hardcoding lookup logic in the host application.

### `SkillsDirectory`

Use `SkillsDirectory` for recursive skills loading.

Supported configuration:

- `SkillsDirectory::from_dir(path)`
- `with_dir(...)`
- `with_skill_file_name(...)`

This supports one or more skill roots with recursive `SKILL.md` discovery.

## Composition

Use `ContextLoader` to combine multiple context sources:

```rust
let items = ContextLoader::new()
    .with_source(AgentsMd::discover_all(workspace_root))
    .with_source(
        SkillsDirectory::from_dir(workspace_root.join("skills"))
            .with_dir(workspace_root.join(".agent/skills")),
    )
    .load()
    .await?;
```

The resulting items are ordinary `ItemKind::Context` transcript entries, so the loop and providers do not need a separate context path.
