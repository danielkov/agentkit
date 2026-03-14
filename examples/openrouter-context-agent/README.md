# `openrouter-context-agent`

One-shot OpenRouter example that wires together:

- `agentkit-loop`
- `agentkit-provider-openrouter`
- `agentkit-context`
- `agentkit-reporting`

It accepts a single prompt argument, loads `AGENTS.md` plus any `SKILL.md` files under `<context-root>/skills`, runs one turn to completion, and exits.

## Run

```bash
cargo run -p openrouter-context-agent -- \
  --context-root examples/openrouter-context-agent/fixtures/project \
  "From the loaded context only, return a JSON object with keys codename and skills. The skills value must be a JSON array of skill names. Do not include prose."
```

That prompt is a good smoke test because the fixture context is static and the expected answer shape is narrow.

Another predictable prompt:

```bash
cargo run -p openrouter-context-agent -- \
  --context-root examples/openrouter-context-agent/fixtures/project \
  "What deployment environment does the project instructions prefer? Return only the environment name."
```

## Notes

- The example loads environment variables from the workspace `.env`.
- If `--context-root` is omitted, the current working directory is used.
- `AGENTS.md` is discovered by searching the context root and its ancestors.
- Skills are loaded recursively from `<context-root>/skills`.
