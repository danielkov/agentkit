# agentkit-adapter-completions

Shared completion-style adapter plumbing for AgentKit model providers.

This crate provides the common request and response translation layer used by
providers that expose a completions-compatible API surface. It is primarily an
internal integration crate for provider implementations such as:

- `agentkit-provider-openai`
- `agentkit-provider-openrouter`
- `agentkit-provider-ollama`
- `agentkit-provider-vllm`
- `agentkit-provider-groq`
- `agentkit-provider-mistral`

Applications will usually depend on a concrete provider crate instead of using
this crate directly.
