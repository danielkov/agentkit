# agentkit docs

This directory holds the design and implementation-facing documents for `agentkit`.

Start here:

- [`v1-boundary.md`](/Users/danielkov/projects/agentkit/docs/v1-boundary.md): the proposed product boundary, crate split, feature flags, and core abstractions for v1.
- [`v1-plan.md`](/Users/danielkov/projects/agentkit/docs/v1-plan.md): the recommended implementation order and the docs backlog to grow alongside the codebase.
- [`architecture.md`](/Users/danielkov/projects/agentkit/docs/architecture.md): the current crate layout and runtime control flow.
- [`getting-started.md`](/Users/danielkov/projects/agentkit/docs/getting-started.md): entry points, example progression, and minimal assembly shape.
- [`feature-flags.md`](/Users/danielkov/projects/agentkit/docs/feature-flags.md): umbrella crate feature flags and typical combinations.
- [`core.md`](/Users/danielkov/projects/agentkit/docs/core.md): the proposed scope and API contract for the `agentkit-core` crate.
- [`compaction.md`](/Users/danielkov/projects/agentkit/docs/compaction.md): compaction trigger/compactor traits and loop integration.
- [`capabilities.md`](/Users/danielkov/projects/agentkit/docs/capabilities.md): the lower-level capability abstraction shared by tools and MCP.
- [`context.md`](/Users/danielkov/projects/agentkit/docs/context.md): built-in `AGENTS.md` and skills loading behavior.
- [`loop.md`](/Users/danielkov/projects/agentkit/docs/loop.md): the proposed operational model and public API for the `agentkit-loop` crate.
- [`reporting.md`](/Users/danielkov/projects/agentkit/docs/reporting.md): the proposed design for the `agentkit-reporting` crate and event-consumer adapters.
- [`tools.md`](/Users/danielkov/projects/agentkit/docs/tools.md): the proposed design for `agentkit-tools-core` and the built-in tool boundaries.
- [`mcp.md`](/Users/danielkov/projects/agentkit/docs/mcp.md): the proposed design for the `agentkit-mcp` integration crate.
- [`permissions.md`](/Users/danielkov/projects/agentkit/docs/permissions.md): the proposed shared policy and approval model across tools and MCP.

The intent is to keep `agentkit` narrow: a reusable Rust toolkit for building agent applications, not a full hosted platform or a single opinionated agent product.
