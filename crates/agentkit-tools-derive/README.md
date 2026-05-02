# agentkit-tools-derive

Procedural macros for declaring agentkit tools.

The `#[tool]` attribute turns an async function into a unit-struct that
implements `agentkit_tools_core::Tool`. The struct is named the same as
the function, so registering it reads naturally:

```rust,ignore
use agentkit_tools_core::{ToolError, ToolResult, ToolResultPart, ToolOutput};
use agentkit_tools_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(JsonSchema, Deserialize)]
struct WeatherInput {
    /// City name to look up.
    location: String,
}

#[tool(description = "Fetch the current weather for a location")]
async fn get_weather(input: WeatherInput) -> Result<ToolResult, ToolError> {
    Ok(ToolResult {
        result: ToolResultPart {
            call_id: Default::default(),
            output: ToolOutput::Text(format!("sunny in {}", input.location)),
            is_error: false,
            metadata: Default::default(),
        },
        duration: None,
        metadata: Default::default(),
    })
}

// Registers as `get_weather` in the catalog.
let registry = agentkit_tools_core::ToolRegistry::new().with(get_weather);
```

Schema is derived from the input type via `schemars::JsonSchema`. The
input struct must implement `JsonSchema` and `serde::Deserialize`. Tool
name defaults to the function's identifier; override with
`#[tool(name = "explicit_name")]`.
