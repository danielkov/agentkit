# agentkit-context

Context loaders for workspace-local agent instructions.

This crate provides:

- upward discovery for `AGENTS.md`
- a composable `ContextLoader` for combining multiple sources
- context items that are already shaped for `agentkit-core` transcripts

Use it to inject repository instructions or other local guidance into agent
sessions.

## Example

```rust,no_run
use agentkit_context::{AgentsMd, ContextLoader};
use agentkit_core::{Item, ItemKind, MetadataMap, Part, TextPart};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = std::env::current_dir()?;

    // Load AGENTS.md files from all ancestors and one extra search directory.
    let context_items = ContextLoader::new()
        .with_source(
            AgentsMd::discover_all(&workspace_root)
                .with_search_dir(workspace_root.join(".agent")),
        )
        .load()
        .await?;

    // Context items are ordinary transcript entries (ItemKind::Context).
    // Prepend them to the input alongside a system prompt and user message,
    // then submit everything to the agent loop.
    let mut input = vec![Item {
        id: None,
        kind: ItemKind::System,
        parts: vec![Part::Text(TextPart {
            text: "You are a helpful assistant.".into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    }];
    input.extend(context_items);
    input.push(Item {
        id: None,
        kind: ItemKind::User,
        parts: vec![Part::Text(TextPart {
            text: "What are the project guidelines?".into(),
            metadata: MetadataMap::new(),
        })],
        metadata: MetadataMap::new(),
    });

    println!("transcript has {} items", input.len());
    Ok(())
}
```
