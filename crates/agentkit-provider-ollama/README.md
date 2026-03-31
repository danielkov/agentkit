# agentkit-provider-ollama

Ollama provider integration for AgentKit.

This crate exposes an AgentKit model adapter backed by Ollama's
completions-compatible API surface. It uses the shared completions adapter
plumbing and translates Ollama request and response shapes into AgentKit types.
