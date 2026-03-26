# Context loading

`agentkit-context` loads prompt and context material into normalized `Item`s.

## Built-in sources

### `AgentsMd`

Load repository or workspace instructions from `AGENTS.md` files.

Supported modes:

- `AgentsMd::discover(path)` — nearest ancestor `AGENTS.md`
- `AgentsMd::discover_all(path)` — stacked ancestor files from root to nearest match

Configuration methods:

- `with_file_name(...)` — custom file name
- `with_search_dir(...)` — additional search directories
- `with_path(...)` — explicit file paths
- `with_mode(...)` — discovery mode

## Composition

Use `ContextLoader` to combine multiple context sources:

```rust
let items = ContextLoader::new()
    .with_source(AgentsMd::discover_all(workspace_root))
    .load()
    .await?;
```

The resulting items are ordinary `ItemKind::Context` transcript entries, so the loop and providers do not need a separate context path.
