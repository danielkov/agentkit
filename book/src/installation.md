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
agentkit = "0.2"
```

## Minimal dependency set

By default, agentkit enables: `core`, `capabilities`, `tools`, `task-manager`, `loop`, and `reporting`.

To keep your build lean, disable defaults and pick only what you need:

```toml
[dependencies]
agentkit = { version = "0.2", default-features = false, features = ["core", "loop"] }
```

See the [Feature flags reference](./feature-flags.md) for the full list.

## Building from source

```sh
git clone https://github.com/danielkov/agentkit.git
cd agentkit
cargo build
```

## Running the examples

The examples use OpenRouter as the model provider. Create a `.env` file in the repo root:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/hunter-alpha
```

Then run any example:

```sh
cargo run -p openrouter-chat -- "hello"
```

The examples are referenced throughout this book. Each chapter points to the relevant example that exercises the concepts being discussed.
