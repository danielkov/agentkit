#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_SECONDS="${WAIT_SECONDS:-10}"

crate_version() {
  local package_id
  package_id="$(cargo pkgid -p "$1")"
  package_id="${package_id##*#}"
  printf '%s\n' "${package_id##*@}"
}

CRATES=(
  agentkit-core
  agentkit-http
  agentkit-capabilities
  agentkit-context
  agentkit-tools-core
  agentkit-tools-derive
  agentkit-task-manager
  agentkit-loop
  agentkit-acp
  agentkit-compaction
  agentkit-adapter-completions
  agentkit-reporting
  agentkit-mcp
  agentkit-plugins
  agentkit-tool-fs
  agentkit-tool-shell
  agentkit-tool-compose
  agentkit-tool-skills
  agentkit-provider-openrouter
  agentkit-provider-openai
  agentkit-provider-ollama
  agentkit-provider-vllm
  agentkit-provider-groq
  agentkit-provider-mistral
  agentkit-provider-anthropic
  agentkit-provider-baseten
  agentkit-provider-cerebras
  agentkit
  how-cli
)

crate_exists() {
  local crate="$1"
  local version="$2"
  python3 - "$crate" "$version" <<'PY' >/dev/null 2>&1
import sys
import urllib.error
import urllib.request

crate, version = sys.argv[1], sys.argv[2]
url = f"https://crates.io/api/v1/crates/{crate}/{version}"

try:
    with urllib.request.urlopen(url):
        pass
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        raise SystemExit(1)
    raise
PY
}

publish_and_wait() {
  local crate="$1"
  local version
  version="$(crate_version "$crate")"

  if crate_exists "$crate" "$version"; then
    echo "Skipping ${crate}@${version}; already present on crates.io."
    return 0
  fi

  echo "Publishing ${crate}@${version}..."
  cargo publish -p "$crate" --locked --no-verify

  echo "Waiting for ${crate}@${version} to appear on crates.io..."
  until crate_exists "$crate" "$version"; do
    sleep "$WAIT_SECONDS"
  done
}

main() {
  cargo check --workspace

  for crate in "${CRATES[@]}"; do
    publish_and_wait "$crate"
  done
}

main "$@"
