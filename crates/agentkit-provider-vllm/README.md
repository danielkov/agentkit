# agentkit-provider-vllm

vLLM provider integration for AgentKit.

This crate exposes an AgentKit model adapter backed by vLLM's
completions-compatible API surface. It reuses the shared completions adapter
layer and normalizes vLLM responses for AgentKit sessions.
