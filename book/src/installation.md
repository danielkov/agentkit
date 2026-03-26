# Installation

## Requirements

- Rust 1.88 or later

## Adding agentkit to your project

```sh
cargo add agentkit
```

Or add it to your `Cargo.toml`:

```toml
[dependencies]
agentkit = "0.1"
```

## Feature flags

By default, agentkit enables: `core`, `capabilities`, `tools`, `task-manager`, `loop`, and `reporting`.

To keep your build lean, disable defaults and pick only what you need:

```toml
[dependencies]
agentkit = { version = "0.1", default-features = false, features = ["core", "loop"] }
```

Optional features: `compaction`, `context`, `mcp`, `provider-openrouter`, `tool-fs`, `tool-shell`.

See [Feature flags](./feature-flags.md) for details.

## Building from source

```sh
git clone https://github.com/danielkov/agentkit.git
cd agentkit
cargo build
```

## Running the examples

1. Create a `.env` file in the repo root:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/hunter-alpha
```

2. Run an example:

```sh
cargo run -p openrouter-chat -- "hello"
```
