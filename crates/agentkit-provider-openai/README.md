# agentkit-provider-openai

OpenAI provider integration for AgentKit.

This crate exposes an AgentKit model adapter backed by OpenAI. It handles
request translation, response normalization, usage reporting, and prompt cache
integration for OpenAI-backed sessions.

Applications that want an OpenAI-powered agent will usually use this crate
through the umbrella `agentkit` crate's `provider-openai` feature, or depend on
it directly when assembling a smaller runtime.
