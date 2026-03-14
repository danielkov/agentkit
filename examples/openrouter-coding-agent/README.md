# `openrouter-coding-agent`

One-shot OpenRouter example that wires together:

- `agentkit-loop`
- `agentkit-provider-openrouter`
- `agentkit-tool-fs`
- `agentkit-reporting`

It accepts a single prompt argument, runs one agent turn to completion, and exits.

## Run

```bash
cargo run -p openrouter-coding-agent -- \
  "Use fs.read_file on ./Cargo.toml and return the workspace member paths exactly as a JSON array. Do not include prose."
```

That prompt is a good first smoke test because it forces a repository read and has a narrow expected output shape.

Another predictable prompt:

```bash
cargo run -p openrouter-coding-agent -- \
  "Use fs.read_file on ./Cargo.toml and tell me how many workspace members are defined. Return only the integer."
```

## Notes

- The example loads environment variables from the workspace `.env`.
- Filesystem tools operate relative to the current working directory, so run this from the repository root if you use `./Cargo.toml`.
- The built-in loop currently uses permissive tool execution internally, so this example is for integration testing, not sandboxed execution.
