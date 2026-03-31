# agentkit-provider-groq

Groq provider integration for AgentKit.

This crate exposes an AgentKit model adapter backed by Groq's
completions-compatible API surface. It reuses the shared completions adapter
plumbing and normalizes Groq responses into AgentKit's core types.
