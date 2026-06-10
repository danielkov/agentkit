//! Lua tool composition for agentkit.
//!
//! This crate provides [`ComposeTool`], a tool that runs a sandboxed Lua script
//! and lets that script call the current AgentKit tool catalog through a
//! synchronous-looking `tool(name, input)` helper.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use agentkit_core::{MetadataMap, ToolCallId, ToolOutput, ToolResultPart, TurnCancellation};
use agentkit_tools_core::{
    ApprovalRequest, PermissionCode, PermissionDenial, Tool, ToolAnnotations, ToolContext,
    ToolError, ToolExecutionOutcome, ToolExecutionScope, ToolInterruption, ToolName, ToolRegistry,
    ToolRequest, ToolResult, ToolSource, ToolSpec,
};
use async_trait::async_trait;
use mlua::{HookTriggers, Lua, LuaSerdeExt, Value as LuaValue, VmState};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

pub const COMPOSE_TOOL_NAME: &str = "compose";

/// Metadata key set on an approval interrupt to identify the nested tool call
/// inside the parent compose run that produced it.
pub const COMPOSE_CHILD_CALL_ID_METADATA_KEY: &str = "agentkit.compose.child_call_id";

/// Creates a [`ToolRegistry`] pre-populated with [`ComposeTool`].
pub fn registry() -> ToolRegistry {
    registry_with_config(ComposeConfig::default())
}

/// Creates a [`ToolRegistry`] pre-populated with [`ComposeTool`] using `config`.
pub fn registry_with_config(config: ComposeConfig) -> ToolRegistry {
    ToolRegistry::new().with(ComposeTool::new(config))
}

/// Configuration for [`ComposeTool`].
#[derive(Clone, Debug)]
pub struct ComposeConfig {
    pub max_script_bytes: usize,
    pub max_nested_tool_calls: usize,
    pub max_result_bytes: usize,
    pub max_instruction_count: u64,
    pub allow_recursive_compose: bool,
    pub allowed_tools: Option<BTreeSet<ToolName>>,
}

impl Default for ComposeConfig {
    fn default() -> Self {
        Self {
            max_script_bytes: 64 * 1024,
            max_nested_tool_calls: 64,
            max_result_bytes: 1024 * 1024,
            max_instruction_count: 1_000_000,
            allow_recursive_compose: false,
            allowed_tools: None,
        }
    }
}

impl ComposeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_script_bytes(mut self, value: usize) -> Self {
        self.max_script_bytes = value;
        self
    }

    pub fn with_max_nested_tool_calls(mut self, value: usize) -> Self {
        self.max_nested_tool_calls = value;
        self
    }

    pub fn with_max_result_bytes(mut self, value: usize) -> Self {
        self.max_result_bytes = value;
        self
    }

    pub fn with_max_instruction_count(mut self, value: u64) -> Self {
        self.max_instruction_count = value;
        self
    }

    pub fn allow_recursive_compose(mut self, value: bool) -> Self {
        self.allow_recursive_compose = value;
        self
    }

    pub fn with_allowed_tools<I>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = ToolName>,
    {
        self.allowed_tools = Some(names.into_iter().collect());
        self
    }

    fn allows(&self, name: &ToolName) -> bool {
        if !self.allow_recursive_compose && name.0 == COMPOSE_TOOL_NAME {
            return false;
        }
        self.allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.contains(name))
    }
}

/// Tool that executes sandboxed Lua scripts over the current tool catalog.
#[derive(Clone)]
pub struct ComposeTool {
    spec: ToolSpec,
    config: ComposeConfig,
    states: Arc<Mutex<BTreeMap<ToolCallId, ComposeRunState>>>,
    wrapped: Option<ToolRegistry>,
    catalog_snapshot: Option<Vec<ToolSpec>>,
}

impl ComposeTool {
    /// Builds a compose tool with no catalog snapshot. The tool description
    /// stays generic; the model has to use `tools()` at runtime to discover
    /// what's available. Prefer [`wrap`](Self::wrap) when possible — the
    /// model writes correct scripts on the first try when it sees concrete
    /// input/output schemas at planning time.
    pub fn new(config: ComposeConfig) -> Self {
        Self::build(config, None, None)
    }

    /// Wraps a registry of child tools. The resulting [`ToolSource`] still
    /// advertises every child tool individually to the model AND adds the
    /// `compose` entry whose description enumerates each child's
    /// input/output schemas.
    ///
    /// ```rust
    /// use agentkit_core::{ToolOutput, ToolResultPart};
    /// use agentkit_tool_compose::{ComposeConfig, ComposeTool};
    /// use agentkit_tools_core::{ToolError, ToolRegistry, ToolResult, ToolSource};
    /// use agentkit_tools_derive::tool;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(JsonSchema, Deserialize)]
    /// struct EchoInput { message: String }
    ///
    /// /// Echo the input back as the tool result.
    /// #[tool(read_only)]
    /// async fn echo(input: EchoInput) -> Result<ToolResult, ToolError> {
    ///     Ok(ToolResult::new(ToolResultPart::success(
    ///         "call",
    ///         ToolOutput::text(input.message),
    ///     )))
    /// }
    ///
    /// let tool_source = ComposeTool::wrap(ToolRegistry::new().with(echo))
    ///     .with_config(ComposeConfig::new().with_max_instruction_count(12_000));
    ///
    /// // Model sees both `echo` and `compose`; compose's description
    /// // enumerates echo's input/output schemas.
    /// let names: Vec<String> = tool_source.specs().into_iter().map(|s| s.name.0).collect();
    /// assert!(names.iter().any(|n| n == "compose"));
    /// assert!(names.iter().any(|n| n == "echo"));
    /// ```
    pub fn wrap(registry: impl Into<ToolRegistry>) -> Self {
        let registry = registry.into();
        let snapshot = registry.specs();
        Self::build(ComposeConfig::default(), Some(snapshot), Some(registry))
    }

    /// Replaces the configuration and rebuilds the compose tool description so
    /// it reflects the new permission filter.
    pub fn with_config(self, config: ComposeConfig) -> Self {
        Self::build(config, self.catalog_snapshot, self.wrapped)
    }

    fn build(
        config: ComposeConfig,
        catalog_snapshot: Option<Vec<ToolSpec>>,
        wrapped: Option<ToolRegistry>,
    ) -> Self {
        let filtered: Option<Vec<ToolSpec>> = catalog_snapshot.as_ref().map(|snap| {
            snap.iter()
                .filter(|spec| config.allows(&spec.name))
                .cloned()
                .collect()
        });
        Self {
            spec: ToolSpec::new(
                COMPOSE_TOOL_NAME,
                Self::compose_description(filtered.as_deref()),
                json!({
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Lua script to execute. Return a value to make it the compose result."
                        },
                        "input": {
                            "description": "Optional JSON value exposed to Lua as global input."
                        }
                    },
                    "required": ["script"],
                    "additionalProperties": false
                }),
            )
            .with_annotations(ToolAnnotations::new()),
            config,
            states: Arc::new(Mutex::new(BTreeMap::new())),
            wrapped,
            catalog_snapshot,
        }
    }

    fn compose_description(catalog: Option<&[ToolSpec]>) -> String {
        let mut description = String::from(
            "Run a sandboxed Lua script that composes available tools through tool(name, input). \
             Prefer this tool whenever a task takes more than two tool calls: iterating over \
             list results, paginating, fetching details per item, filtering or aggregating tool \
             output, or chaining reads into writes. The whole script executes in a single \
             round-trip — one compose call replaces N individual calls — and only the script's \
             return value enters the conversation, so intermediate results never consume \
             context. The script sees a global `input` (the JSON value passed alongside the \
             script) and may call `tools()` to enumerate the visible tool catalog at runtime. \
             Return any Lua value to make it the compose result.\n\n\
             Example — scan every page, drill into matches, return only the summary:\n\
             local page, hits = 1, {}\n\
             repeat\n\
             \x20 local r = tool('list_items', { page = page })\n\
             \x20 for _, it in ipairs(r.items) do\n\
             \x20   if it.status == 'open' then hits[#hits + 1] = tool('get_item', { id = it.id }) end\n\
             \x20 end\n\
             \x20 page = page + 1\n\
             until page > r.total_pages\n\
             return { count = #hits, items = hits }",
        );
        if let Some(catalog) = catalog {
            if catalog.is_empty() {
                return description;
            }
            description.push_str(
                "\n\nReturn shapes of tools accessible via tool(name, input) (input schemas are \
                 already provided by the top-level tool catalog):\n",
            );
            for spec in catalog {
                description.push_str("\n- ");
                description.push_str(spec.name.0.as_str());
                description.push_str(": ");
                match spec.output_schema.as_ref() {
                    Some(schema) => description.push_str(&Self::compact_schema(schema)),
                    None => description.push_str("<undocumented>"),
                }
            }
        }
        description
    }

    fn compact_schema(value: &Value) -> String {
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    }

    fn visible_specs(&self, scope: &ToolExecutionScope) -> Vec<ToolSpec> {
        scope
            .executor
            .specs()
            .into_iter()
            .filter(|spec| self.config.allows(&spec.name))
            .collect()
    }
}

impl ToolSource for ComposeTool {
    fn specs(&self) -> Vec<ToolSpec> {
        let mut specs = Vec::new();
        specs.push(self.spec.clone());
        if let Some(reg) = &self.wrapped {
            specs.extend(reg.specs());
        }
        specs
    }

    fn get(&self, name: &ToolName) -> Option<Arc<dyn Tool>> {
        if name.0.as_str() == COMPOSE_TOOL_NAME {
            return Some(Arc::new(self.clone()));
        }
        self.wrapped.as_ref().and_then(|reg| reg.get(name))
    }
}

#[derive(Debug, Deserialize)]
struct ComposeInput {
    script: String,
    #[serde(default)]
    input: Value,
}

#[derive(Clone, Debug, Default)]
struct ComposeRunState {
    records: Vec<ChildRecord>,
    pending: Option<PendingChild>,
}

#[derive(Clone, Debug)]
struct ChildRecord {
    name: ToolName,
    input: Value,
    output: Value,
}

#[derive(Clone, Debug)]
struct PendingChild {
    index: usize,
    name: ToolName,
    input: Value,
    approval_id: String,
}

#[derive(Debug)]
struct ComposeInterrupt(ToolInterruption);

impl fmt::Display for ComposeInterrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compose interrupted")
    }
}

impl Error for ComposeInterrupt {}

#[derive(Debug)]
struct ComposeFailure(ToolError);

impl fmt::Display for ComposeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for ComposeFailure {}

#[async_trait]
impl Tool for ComposeTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        match self.invoke_outcome(request, ctx).await {
            ToolExecutionOutcome::Completed(result) => Ok(result),
            ToolExecutionOutcome::Interrupted(_) => Err(ToolError::Internal(
                "compose produced an approval interrupt through invoke".into(),
            )),
            ToolExecutionOutcome::Failed(error) => Err(error),
        }
    }

    async fn invoke_outcome(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> ToolExecutionOutcome {
        match self.invoke_outcome_inner(request, ctx).await {
            Ok(result) => ToolExecutionOutcome::Completed(result),
            Err(ComposeOutcome::Interrupted(interruption)) => {
                ToolExecutionOutcome::Interrupted(interruption)
            }
            Err(ComposeOutcome::Failed(error)) => ToolExecutionOutcome::Failed(error),
        }
    }
}

enum ComposeOutcome {
    Interrupted(ToolInterruption),
    Failed(ToolError),
}

impl ComposeTool {
    async fn invoke_outcome_inner(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ComposeOutcome> {
        let input: ComposeInput = serde_json::from_value(request.input.clone())
            .map_err(|error| ComposeOutcome::Failed(ToolError::InvalidInput(error.to_string())))?;
        if input.script.len() > self.config.max_script_bytes {
            return Err(ComposeOutcome::Failed(ToolError::InvalidInput(format!(
                "compose script exceeds {} bytes",
                self.config.max_script_bytes
            ))));
        }
        let Some(scope) = ctx.execution_scope.clone() else {
            return Err(ComposeOutcome::Failed(ToolError::Unavailable(
                "compose requires a tool execution scope".into(),
            )));
        };
        if ctx
            .cancellation
            .as_ref()
            .is_some_and(|cancellation| cancellation.is_cancelled())
        {
            return Err(ComposeOutcome::Failed(ToolError::Cancelled));
        }

        {
            let mut states = self.states.lock().await;
            if ctx.approved_request.is_none() {
                states.remove(&request.call_id);
            }
            states.entry(request.call_id.clone()).or_default();
        }

        let cleanup_call_id = request.call_id.clone();
        let visible_specs = self.visible_specs(&scope);
        let cancellation = ctx.cancellation.clone();
        let approved_request = ctx.approved_request.clone();
        let outcome = self
            .run_script(
                request,
                input,
                scope,
                cancellation,
                approved_request,
                visible_specs,
            )
            .await;
        if !matches!(outcome, Err(ComposeOutcome::Interrupted(_))) {
            self.states.lock().await.remove(&cleanup_call_id);
        }
        outcome
    }

    async fn run_script(
        &self,
        request: ToolRequest,
        input: ComposeInput,
        scope: ToolExecutionScope,
        cancellation: Option<TurnCancellation>,
        approved_request: Option<ApprovalRequest>,
        visible_specs: Vec<ToolSpec>,
    ) -> Result<ToolResult, ComposeOutcome> {
        let lua = Lua::new();
        install_instruction_limit(
            &lua,
            self.config.max_instruction_count,
            cancellation.clone(),
        )
        .map_err(lua_error_to_outcome)?;
        install_sandbox(&lua).map_err(lua_error_to_outcome)?;
        let globals = lua.globals();
        globals
            .set(
                "input",
                lua.to_value(&input.input).map_err(lua_error_to_outcome)?,
            )
            .map_err(lua_error_to_outcome)?;

        let specs_value = serde_json::to_value(&visible_specs)
            .map_err(|error| ComposeOutcome::Failed(ToolError::Internal(error.to_string())))?;
        globals
            .set(
                "tools",
                lua.create_function(move |lua, ()| lua.to_value(&specs_value))
                    .map_err(lua_error_to_outcome)?,
            )
            .map_err(lua_error_to_outcome)?;

        let config = self.config.clone();
        let states = self.states.clone();
        let parent_call_id = request.call_id.clone();
        let session_id = request.session_id.clone();
        let turn_id = request.turn_id.clone();
        let call_counter = Arc::new(AtomicUsize::new(0));

        let tool_fn = lua
            .create_async_function(move |lua, (name, lua_input): (String, LuaValue)| {
                let config = config.clone();
                let states = states.clone();
                let scope = scope.clone();
                let parent_call_id = parent_call_id.clone();
                let session_id = session_id.clone();
                let turn_id = turn_id.clone();
                let approved_request = approved_request.clone();
                let call_counter = call_counter.clone();
                let cancellation = cancellation.clone();
                async move {
                    if cancellation
                        .as_ref()
                        .is_some_and(|cancellation| cancellation.is_cancelled())
                    {
                        return Err(mlua::Error::external(ComposeFailure(ToolError::Cancelled)));
                    }
                    let index = call_counter.fetch_add(1, Ordering::SeqCst);
                    if index >= config.max_nested_tool_calls {
                        return Err(mlua::Error::external(ComposeFailure(
                            ToolError::ExecutionFailed(format!(
                                "compose exceeded {} nested tool calls",
                                config.max_nested_tool_calls
                            )),
                        )));
                    }

                    let tool_name = ToolName::new(name);
                    if !config.allows(&tool_name) {
                        return Err(mlua::Error::external(ComposeFailure(
                            ToolError::PermissionDenied(PermissionDenial {
                                code: PermissionCode::CustomPolicyDenied,
                                message: format!("compose cannot call tool {}", tool_name.0),
                                metadata: MetadataMap::new(),
                            }),
                        )));
                    }
                    let child_input: Value = lua.from_value(lua_input)?;

                    let replayed = {
                        let state = states.lock().await;
                        state
                            .get(&parent_call_id)
                            .and_then(|state| state.records.get(index))
                            .map(|record| {
                                (
                                    record.name.clone(),
                                    record.input.clone(),
                                    record.output.clone(),
                                )
                            })
                    };
                    if let Some((recorded_name, recorded_input, recorded_output)) = replayed {
                        if recorded_name == tool_name && recorded_input == child_input {
                            return lua.to_value(&recorded_output);
                        }
                        return Err(mlua::Error::external(ComposeFailure(
                            ToolError::ExecutionFailed(format!(
                                "compose replay diverged at nested tool call {index}"
                            )),
                        )));
                    }

                    let child_call_id =
                        ToolCallId::new(format!("{}:compose:{}", parent_call_id.0.as_str(), index));
                    let child_request = ToolRequest {
                        call_id: child_call_id.clone(),
                        tool_name: tool_name.clone(),
                        input: child_input.clone(),
                        session_id: session_id.clone(),
                        turn_id: turn_id.clone(),
                        metadata: MetadataMap::new(),
                    };

                    let is_approved_pending = {
                        let state = states.lock().await;
                        let pending = state
                            .get(&parent_call_id)
                            .and_then(|state| state.pending.as_ref());
                        pending.is_some_and(|pending| {
                            pending.index == index
                                && pending.name == tool_name
                                && pending.input == child_input
                                && approved_request
                                    .as_ref()
                                    .is_some_and(|approval| approval.id.0 == pending.approval_id)
                        })
                    };

                    let outcome = if is_approved_pending {
                        let approval = approved_request.as_ref().ok_or_else(|| {
                            mlua::Error::external(ComposeFailure(ToolError::Internal(
                                "missing compose approval request".into(),
                            )))
                        })?;
                        scope.execute_approved_child(child_request, approval).await
                    } else {
                        scope.execute_child(child_request).await
                    };

                    match outcome {
                        ToolExecutionOutcome::Completed(result) => {
                            let output = tool_output_to_json(result.result.output)
                                .map_err(|error| mlua::Error::external(ComposeFailure(error)))?;
                            {
                                let mut state = states.lock().await;
                                let run_state = state.entry(parent_call_id.clone()).or_default();
                                if run_state.records.len() != index {
                                    return Err(mlua::Error::external(ComposeFailure(
                                        ToolError::ExecutionFailed(format!(
                                            "compose replay cannot append nested tool call {index}"
                                        )),
                                    )));
                                }
                                run_state.records.push(ChildRecord {
                                    name: tool_name,
                                    input: child_input,
                                    output: output.clone(),
                                });
                                run_state.pending = None;
                            }
                            lua.to_value(&output)
                        }
                        ToolExecutionOutcome::Interrupted(ToolInterruption::ApprovalRequired(
                            mut approval,
                        )) => {
                            approval.metadata.insert(
                                COMPOSE_CHILD_CALL_ID_METADATA_KEY.into(),
                                Value::String(child_call_id.0.clone()),
                            );
                            {
                                let mut state = states.lock().await;
                                let run_state = state.entry(parent_call_id.clone()).or_default();
                                run_state.pending = Some(PendingChild {
                                    index,
                                    name: tool_name,
                                    input: child_input,
                                    approval_id: approval.id.0.clone(),
                                });
                            }
                            Err(mlua::Error::external(ComposeInterrupt(
                                ToolInterruption::ApprovalRequired(approval),
                            )))
                        }
                        ToolExecutionOutcome::Failed(error) => {
                            Err(mlua::Error::external(ComposeFailure(error)))
                        }
                    }
                }
            })
            .map_err(lua_error_to_outcome)?;
        globals.set("tool", tool_fn).map_err(lua_error_to_outcome)?;

        let result = match lua
            .load(input.script.as_str())
            .set_name("compose")
            .eval_async::<LuaValue>()
            .await
        {
            Ok(value) => value,
            Err(error) => return Err(lua_error_to_outcome(error)),
        };
        let json_result: Value = lua.from_value(result).map_err(lua_error_to_outcome)?;
        let result_bytes = serde_json::to_vec(&json_result)
            .map_err(|error| ComposeOutcome::Failed(ToolError::Internal(error.to_string())))?
            .len();
        if result_bytes > self.config.max_result_bytes {
            return Err(ComposeOutcome::Failed(ToolError::ExecutionFailed(format!(
                "compose result exceeds {} bytes",
                self.config.max_result_bytes
            ))));
        }
        Ok(ToolResult::new(ToolResultPart::success(
            request.call_id,
            ToolOutput::structured(json_result),
        )))
    }
}

fn install_sandbox(lua: &Lua) -> Result<(), mlua::Error> {
    let globals = lua.globals();
    for name in [
        "collectgarbage",
        "dofile",
        "load",
        "loadfile",
        "require",
        "io",
        "os",
        "package",
        "debug",
    ] {
        globals.set(name, LuaValue::Nil)?;
    }
    Ok(())
}

fn install_instruction_limit(
    lua: &Lua,
    max_instruction_count: u64,
    cancellation: Option<agentkit_core::TurnCancellation>,
) -> Result<(), mlua::Error> {
    if max_instruction_count == 0 {
        return Ok(());
    }
    let step = max_instruction_count.min(1_000) as u32;
    let seen = Arc::new(AtomicU64::new(0));
    lua.set_global_hook(
        HookTriggers::new().every_nth_instruction(step),
        move |_lua, _debug| {
            if cancellation
                .as_ref()
                .is_some_and(|cancellation| cancellation.is_cancelled())
            {
                return Err(mlua::Error::external(ComposeFailure(ToolError::Cancelled)));
            }
            let previous = seen.fetch_add(u64::from(step), Ordering::Relaxed);
            if previous.saturating_add(u64::from(step)) > max_instruction_count {
                return Err(mlua::Error::external(ComposeFailure(
                    ToolError::ExecutionFailed(format!(
                        "compose exceeded {max_instruction_count} Lua instructions"
                    )),
                )));
            }
            Ok(VmState::Continue)
        },
    )
}

fn tool_output_to_json(output: ToolOutput) -> Result<Value, ToolError> {
    match output {
        ToolOutput::Text(text) => Ok(Value::String(text)),
        ToolOutput::Structured(value) => Ok(value),
        other => {
            serde_json::to_value(other).map_err(|error| ToolError::Internal(error.to_string()))
        }
    }
}

fn lua_error_to_outcome(error: mlua::Error) -> ComposeOutcome {
    match &error {
        mlua::Error::CallbackError { cause, .. }
        | mlua::Error::BadArgument { cause, .. }
        | mlua::Error::WithContext { cause, .. } => {
            return lua_error_to_outcome((**cause).clone());
        }
        mlua::Error::ExternalError(inner) => {
            if let Some(interrupt) = inner.downcast_ref::<ComposeInterrupt>() {
                return ComposeOutcome::Interrupted(interrupt.0.clone());
            }
            if let Some(failure) = inner.downcast_ref::<ComposeFailure>() {
                return ComposeOutcome::Failed(failure.0.clone());
            }
        }
        _ => {}
    }
    ComposeOutcome::Failed(ToolError::ExecutionFailed(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use agentkit_core::{ApprovalId, SessionId, TurnId};
    use agentkit_tools_core::{
        AllowAllPermissions, ApprovalReason, ApprovalRequest, BasicToolExecutor, PermissionChecker,
        PermissionDecision, PermissionRequest, ToolExecutionScope, ToolExecutor,
    };
    use serde_json::json;

    use super::*;

    #[derive(Clone)]
    struct EchoTool {
        spec: ToolSpec,
        calls: Arc<AtomicUsize>,
    }

    impl EchoTool {
        fn new() -> Self {
            Self {
                spec: ToolSpec::new("echo", "echo input", json!({"type": "object"})),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::structured(request.input),
            )))
        }
    }

    struct ApprovalPermissionRequest {
        metadata: MetadataMap,
    }

    impl PermissionRequest for ApprovalPermissionRequest {
        fn kind(&self) -> &'static str {
            "compose.test.approval"
        }

        fn summary(&self) -> String {
            "approval required".into()
        }

        fn metadata(&self) -> &MetadataMap {
            &self.metadata
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Clone)]
    struct ApprovalEchoTool {
        spec: ToolSpec,
        calls: Arc<AtomicUsize>,
    }

    impl ApprovalEchoTool {
        fn new() -> Self {
            Self {
                spec: ToolSpec::new("approval_echo", "approval echo", json!({"type": "object"})),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Tool for ApprovalEchoTool {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        fn proposed_requests(
            &self,
            _request: &ToolRequest,
        ) -> Result<Vec<Box<dyn PermissionRequest>>, ToolError> {
            Ok(vec![Box::new(ApprovalPermissionRequest {
                metadata: MetadataMap::new(),
            })])
        }

        async fn invoke(
            &self,
            request: ToolRequest,
            _ctx: &mut ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::structured(request.input),
            )))
        }
    }

    struct RequireApproval;

    impl PermissionChecker for RequireApproval {
        fn evaluate(&self, request: &dyn PermissionRequest) -> PermissionDecision {
            PermissionDecision::RequireApproval(ApprovalRequest {
                task_id: None,
                call_id: None,
                id: ApprovalId::new("approval:test"),
                request_kind: request.kind().into(),
                reason: ApprovalReason::PolicyRequiresConfirmation,
                summary: request.summary(),
                metadata: request.metadata().clone(),
            })
        }
    }

    fn request(script: &str, input: Value) -> ToolRequest {
        ToolRequest {
            call_id: ToolCallId::new("compose-call"),
            tool_name: ToolName::new(COMPOSE_TOOL_NAME),
            input: json!({ "script": script, "input": input }),
            session_id: SessionId::new("session"),
            turn_id: TurnId::new("turn"),
            metadata: MetadataMap::new(),
        }
    }

    fn owned_context(
        executor: Arc<dyn ToolExecutor>,
        permissions: Arc<dyn PermissionChecker>,
    ) -> agentkit_tools_core::OwnedToolContext {
        let session_id = SessionId::new("session");
        let turn_id = TurnId::new("turn");
        let metadata = MetadataMap::new();
        let resources: Arc<dyn agentkit_tools_core::ToolResources> = Arc::new(());
        let scope = ToolExecutionScope {
            executor,
            session_id: session_id.clone(),
            turn_id: turn_id.clone(),
            permissions: permissions.clone(),
            resources: resources.clone(),
            cancellation: None,
        };
        agentkit_tools_core::OwnedToolContext {
            session_id,
            turn_id,
            metadata,
            permissions,
            resources,
            cancellation: None,
            execution_scope: Some(scope),
            approved_request: None,
        }
    }

    async fn execute_compose(
        config: ComposeConfig,
        child: impl Tool + 'static,
        req: ToolRequest,
    ) -> ToolExecutionOutcome {
        let compose = ComposeTool::new(config);
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose).with(child),
        ));
        let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
        let mut ctx = owned.borrowed();
        executor.execute(req, &mut ctx).await
    }

    #[tokio::test]
    async fn converts_lua_result_to_structured_json() {
        let outcome = execute_compose(
            ComposeConfig::default(),
            EchoTool::new(),
            request(
                "return { count = input.count + 1, label = 'ok' }",
                json!({ "count": 2 }),
            ),
        )
        .await;

        match outcome {
            ToolExecutionOutcome::Completed(result) => {
                assert_eq!(
                    result.result.output,
                    ToolOutput::structured(json!({ "count": 3, "label": "ok" }))
                );
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_function_calls_child_tool() {
        let child = EchoTool::new();
        let calls = child.calls.clone();
        let outcome = execute_compose(
            ComposeConfig::default(),
            child,
            request(
                "local out = tool('echo', { value = input.value }); return out",
                json!({ "value": 7 }),
            ),
        )
        .await;

        match outcome {
            ToolExecutionOutcome::Completed(result) => {
                assert_eq!(
                    result.result.output,
                    ToolOutput::structured(json!({ "value": 7 }))
                );
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[tokio::test]
    async fn tools_excludes_compose_by_default() {
        let outcome = execute_compose(
            ComposeConfig::default(),
            EchoTool::new(),
            request(
                "for _, spec in ipairs(tools()) do if spec.name == 'compose' then return 'bad' end end; return 'ok'",
                Value::Null,
            ),
        )
        .await;

        match outcome {
            ToolExecutionOutcome::Completed(result) => {
                assert_eq!(result.result.output, ToolOutput::structured(json!("ok")));
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[tokio::test]
    async fn sandbox_removes_os_io_and_require() {
        for script in [
            "return os.getenv('HOME')",
            "return io.open('Cargo.toml')",
            "return require('x')",
        ] {
            let outcome = execute_compose(
                ComposeConfig::default(),
                EchoTool::new(),
                request(script, Value::Null),
            )
            .await;
            assert!(matches!(outcome, ToolExecutionOutcome::Failed(_)));
        }
    }

    #[tokio::test]
    async fn nested_tool_call_limit_fails() {
        let outcome = execute_compose(
            ComposeConfig::default().with_max_nested_tool_calls(0),
            EchoTool::new(),
            request("return tool('echo', {})", Value::Null),
        )
        .await;

        assert!(matches!(outcome, ToolExecutionOutcome::Failed(_)));
    }

    #[tokio::test]
    async fn instruction_limit_fails() {
        let outcome = execute_compose(
            ComposeConfig::default().with_max_instruction_count(25),
            EchoTool::new(),
            request(
                "local x = 0; for i = 1, 100000 do x = x + 1 end; return x",
                Value::Null,
            ),
        )
        .await;

        assert!(matches!(outcome, ToolExecutionOutcome::Failed(_)));
    }

    #[tokio::test]
    async fn nested_approval_replays_completed_children_once() {
        let compose = ComposeTool::new(ComposeConfig::default());
        let states = compose.states.clone();
        let first = EchoTool::new();
        let gated = ApprovalEchoTool::new();
        let first_calls = first.calls.clone();
        let gated_calls = gated.calls.clone();
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose).with(first).with(gated),
        ));
        let permissions: Arc<dyn PermissionChecker> = Arc::new(RequireApproval);
        let req = request(
            "local a = tool('echo', { value = 1 }); local b = tool('approval_echo', { value = a.value + 1 }); return b",
            Value::Null,
        );

        let owned = owned_context(executor.clone(), permissions.clone());
        let mut ctx = owned.borrowed();
        let first_outcome = executor.execute(req.clone(), &mut ctx).await;
        let approval = match first_outcome {
            ToolExecutionOutcome::Interrupted(ToolInterruption::ApprovalRequired(approval)) => {
                approval
            }
            other => panic!("unexpected first outcome: {other:?}"),
        };
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(gated_calls.load(Ordering::SeqCst), 0);
        // After an approval interrupt, the per-call replay state must persist so
        // the resumed run can replay completed children and re-issue the
        // pending one.
        assert!(
            !states.lock().await.is_empty(),
            "compose run state must be retained across approval interrupts"
        );

        let owned = owned_context(executor.clone(), permissions);
        let outcome = executor.execute_approved_owned(req, &approval, owned).await;
        match outcome {
            ToolExecutionOutcome::Completed(result) => {
                assert_eq!(
                    result.result.output,
                    ToolOutput::structured(json!({ "value": 2 }))
                );
            }
            other => panic!("unexpected approved outcome: {other:?}"),
        }
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(gated_calls.load(Ordering::SeqCst), 1);
        // Once the compose run completes, the state-map entry must be cleared.
        assert!(
            states.lock().await.is_empty(),
            "compose run state must be cleared after a successful resume"
        );
    }

    #[tokio::test]
    async fn state_map_cleared_after_successful_run() {
        let compose = ComposeTool::new(ComposeConfig::default());
        let states = compose.states.clone();
        let child = EchoTool::new();
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose).with(child),
        ));
        let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
        let mut ctx = owned.borrowed();
        let outcome = executor
            .execute(
                request("return tool('echo', { value = 1 })", Value::Null),
                &mut ctx,
            )
            .await;
        assert!(matches!(outcome, ToolExecutionOutcome::Completed(_)));
        assert!(
            states.lock().await.is_empty(),
            "compose run state must be cleared after a successful run"
        );
    }

    #[tokio::test]
    async fn state_map_cleared_after_script_eval_failure() {
        // Regression: an oversized result returned from the Lua script used
        // to leak the state-map entry created during the run's setup.
        let compose = ComposeTool::new(ComposeConfig::default().with_max_result_bytes(1));
        let states = compose.states.clone();
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose).with(EchoTool::new()),
        ));
        let owned = owned_context(executor.clone(), Arc::new(AllowAllPermissions));
        let mut ctx = owned.borrowed();
        let outcome = executor
            .execute(
                request(
                    "return 'this string is far longer than one byte'",
                    Value::Null,
                ),
                &mut ctx,
            )
            .await;
        assert!(
            matches!(outcome, ToolExecutionOutcome::Failed(_)),
            "expected oversized compose result to fail",
        );
        assert!(
            states.lock().await.is_empty(),
            "compose run state must be cleared after a script-eval failure"
        );
    }

    #[tokio::test]
    async fn concurrent_runs_over_disjoint_call_ids() {
        let compose = ComposeTool::new(ComposeConfig::default());
        let child = EchoTool::new();
        let child_calls = child.calls.clone();
        let executor: Arc<dyn ToolExecutor> = Arc::new(BasicToolExecutor::from_registry(
            ToolRegistry::new().with(compose).with(child),
        ));

        let make_request = |call: &str, base: i64| ToolRequest {
            call_id: ToolCallId::new(call),
            tool_name: ToolName::new(COMPOSE_TOOL_NAME),
            input: json!({
                "script": "local a = tool('echo', { value = input.base }); local b = tool('echo', { value = a.value + 1 }); return { a = a.value, b = b.value }",
                "input": { "base": base },
            }),
            session_id: SessionId::new("session"),
            turn_id: TurnId::new("turn"),
            metadata: MetadataMap::new(),
        };

        let permissions: Arc<dyn PermissionChecker> = Arc::new(AllowAllPermissions);

        let executor_a = executor.clone();
        let permissions_a = permissions.clone();
        let req_a = make_request("compose-call-a", 10);
        let handle_a = tokio::spawn(async move {
            let owned = owned_context(executor_a.clone(), permissions_a);
            let mut ctx = owned.borrowed();
            executor_a.execute(req_a, &mut ctx).await
        });

        let executor_b = executor.clone();
        let permissions_b = permissions.clone();
        let req_b = make_request("compose-call-b", 100);
        let handle_b = tokio::spawn(async move {
            let owned = owned_context(executor_b.clone(), permissions_b);
            let mut ctx = owned.borrowed();
            executor_b.execute(req_b, &mut ctx).await
        });

        let outcome_a = handle_a.await.expect("compose A join");
        let outcome_b = handle_b.await.expect("compose B join");

        match outcome_a {
            ToolExecutionOutcome::Completed(result) => assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "a": 10, "b": 11 }))
            ),
            other => panic!("unexpected outcome A: {other:?}"),
        }
        match outcome_b {
            ToolExecutionOutcome::Completed(result) => assert_eq!(
                result.result.output,
                ToolOutput::structured(json!({ "a": 100, "b": 101 }))
            ),
            other => panic!("unexpected outcome B: {other:?}"),
        }

        // Two compose runs, each making two nested calls.
        assert_eq!(child_calls.load(Ordering::SeqCst), 4);
    }
}
