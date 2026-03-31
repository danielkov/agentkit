# agentkit-provider-mistral

Mistral provider integration for AgentKit.

This crate exposes an AgentKit model adapter backed by Mistral's
completions-compatible API surface. It reuses the shared completions adapter
layer and normalizes Mistral responses for the agent loop.
