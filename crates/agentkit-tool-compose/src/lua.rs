//! Sandboxed Lua execution backend (the default).

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::Poll;

use agentkit_core::TurnCancellation;
use agentkit_tools_core::{ToolError, ToolInterruption, ToolName, ToolSpec};
use async_trait::async_trait;
use mlua::{HookTriggers, Lua, LuaSerdeExt, Value as LuaValue, VmState};
use serde_json::Value;

use crate::{
    BackendRun, CallKey, ComposeBackend, ComposeOutcome, DispatchError, render_catalog_shapes,
};

/// Executes compose scripts as sandboxed Lua with a synchronous-looking
/// `tool(name, input)` helper.
#[derive(Clone, Copy, Debug, Default)]
pub struct LuaBackend;

#[async_trait]
impl ComposeBackend for LuaBackend {
    fn name(&self) -> &'static str {
        "lua"
    }

    fn script_description(&self) -> &'static str {
        "Lua script to execute. Return a value to make it the compose result."
    }

    fn description(&self, catalog: Option<&[ToolSpec]>) -> String {
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
            if !catalog.is_empty() {
                description.push_str(&render_catalog_shapes(catalog));
            }
        }
        description
    }

    async fn execute(&self, run: BackendRun) -> Result<Value, ComposeOutcome> {
        let lua = Lua::new();
        install_instruction_limit(
            &lua,
            run.config.max_instruction_count,
            run.cancellation.clone(),
        )
        .map_err(lua_error_to_outcome)?;
        install_sandbox(&lua).map_err(lua_error_to_outcome)?;
        let globals = lua.globals();
        globals
            .set(
                "input",
                lua.to_value(&run.input).map_err(lua_error_to_outcome)?,
            )
            .map_err(lua_error_to_outcome)?;

        let specs_value = serde_json::to_value(&run.visible_specs)
            .map_err(|error| ComposeOutcome::Failed(ToolError::Internal(error.to_string())))?;
        globals
            .set(
                "tools",
                lua.create_function(move |lua, ()| lua.to_value(&specs_value))
                    .map_err(lua_error_to_outcome)?,
            )
            .map_err(lua_error_to_outcome)?;

        // An approval interrupt ends the Lua attempt at the `tool()` call that
        // raised it: the call records it here and never resumes the script, so
        // `pcall` cannot catch it and no later script code or child call runs
        // before the host answers it.
        let interrupted: Arc<OnceLock<ToolInterruption>> = Arc::new(OnceLock::new());
        let dispatcher = run.dispatcher.clone();
        let interrupted_slot = interrupted.clone();
        let tool_fn = lua
            .create_async_function(move |lua, (name, lua_input): (String, LuaValue)| {
                let dispatcher = dispatcher.clone();
                let interrupted_slot = interrupted_slot.clone();
                async move {
                    let child_input: Value = lua.from_value(lua_input)?;
                    match dispatcher
                        .call(CallKey::Sequential, ToolName::new(name), child_input)
                        .await
                    {
                        Ok(output) => lua.to_value(&output),
                        Err(DispatchError::Interrupted(interruption)) => {
                            let _ = interrupted_slot.set(interruption);
                            std::future::pending::<()>().await;
                            unreachable!("a pending future never resolves")
                        }
                        Err(DispatchError::Failed(error)) => {
                            Err(mlua::Error::external(ComposeFailure(error)))
                        }
                    }
                }
            })
            .map_err(lua_error_to_outcome)?;
        globals.set("tool", tool_fn).map_err(lua_error_to_outcome)?;

        let mut evaluation = std::pin::pin!(
            lua.load(run.script.as_str())
                .set_name("compose")
                .eval_async::<LuaValue>()
        );
        // Stop polling the script as soon as a child call records an approval.
        let evaluated = std::future::poll_fn(|cx| match evaluation.as_mut().poll(cx) {
            Poll::Ready(result) => Poll::Ready(Some(result)),
            Poll::Pending if interrupted.get().is_some() => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        })
        .await;
        let Some(evaluated) = evaluated else {
            // A cancelled turn must not surface an approval prompt.
            if run.dispatcher.is_cancelled() {
                return Err(ComposeOutcome::Failed(ToolError::Cancelled));
            }
            let interruption = interrupted.get().cloned().ok_or_else(|| {
                ComposeOutcome::Failed(ToolError::Internal(
                    "compose approval interrupt was not recorded".into(),
                ))
            })?;
            return Err(ComposeOutcome::Interrupted(interruption));
        };
        let result = evaluated.map_err(lua_error_to_outcome)?;
        match lua.from_value(result.clone()) {
            Ok(value) => Ok(value),
            Err(original) => {
                let (normalized, changed) =
                    normalize_caught_failures(&lua, result).map_err(|outcome| *outcome)?;
                if !changed {
                    return Err(lua_error_to_outcome(original));
                }
                lua.from_value(normalized).map_err(lua_error_to_outcome)
            }
        }
    }
}

/// Replaces child tool failures a script caught with `pcall` and returned
/// (at the root or inside tables) with their message, so the result
/// serializes. Other error values are left alone. Returns whether anything
/// changed. A caught backend stop (cancellation or the instruction budget)
/// fails the run instead of becoming data.
fn normalize_caught_failures(
    lua: &Lua,
    value: LuaValue,
) -> Result<(LuaValue, bool), Box<ComposeOutcome>> {
    let to_outcome = |error: mlua::Error| Box::new(lua_error_to_outcome(error));
    let mut changed = false;
    let root = match value {
        LuaValue::Error(error) => match caught_failure_message(lua, &error)? {
            Some(message) => {
                changed = true;
                message
            }
            None => LuaValue::Error(error),
        },
        other => other,
    };
    let mut pending = Vec::new();
    if let LuaValue::Table(table) = &root {
        pending.push(table.clone());
    }
    // Tables can be shared or cyclic: walk iteratively and visit each once.
    let mut seen = HashSet::new();
    while let Some(table) = pending.pop() {
        if !seen.insert(table.to_pointer() as usize) {
            continue;
        }
        let mut replacements = Vec::new();
        for pair in table.pairs::<LuaValue, LuaValue>() {
            let (key, entry) = pair.map_err(to_outcome)?;
            match entry {
                LuaValue::Error(error) => {
                    if let Some(message) = caught_failure_message(lua, &error)? {
                        replacements.push((key, message));
                    }
                }
                LuaValue::Table(child) => pending.push(child),
                _ => {}
            }
        }
        changed |= !replacements.is_empty();
        for (key, message) in replacements {
            table.raw_set(key, message).map_err(to_outcome)?;
        }
    }
    Ok((root, changed))
}

/// The message for a caught child tool failure, or `None` for any other
/// error value.
fn caught_failure_message(
    lua: &Lua,
    error: &mlua::Error,
) -> Result<Option<LuaValue>, Box<ComposeOutcome>> {
    match compose_error(error) {
        Some(ComposeError::Stop(stop)) => Err(Box::new(ComposeOutcome::Failed(stop.clone()))),
        Some(ComposeError::Child(ToolError::Cancelled)) => {
            Err(Box::new(ComposeOutcome::Failed(ToolError::Cancelled)))
        }
        Some(ComposeError::Child(failure)) => lua
            .create_string(failure.to_string())
            .map(|message| Some(LuaValue::String(message)))
            .map_err(|error| Box::new(lua_error_to_outcome(error))),
        None => Ok(None),
    }
}

enum ComposeError<'a> {
    Child(&'a ToolError),
    Stop(&'a ToolError),
}

fn compose_error(error: &mlua::Error) -> Option<ComposeError<'_>> {
    match error {
        mlua::Error::CallbackError { cause, .. }
        | mlua::Error::BadArgument { cause, .. }
        | mlua::Error::WithContext { cause, .. } => compose_error(cause),
        mlua::Error::ExternalError(inner) => {
            if let Some(failure) = inner.downcast_ref::<ComposeFailure>() {
                return Some(ComposeError::Child(&failure.0));
            }
            inner
                .downcast_ref::<ComposeStop>()
                .map(|stop| ComposeError::Stop(&stop.0))
        }
        _ => None,
    }
}

/// A child tool call failed. A script may catch and return it as data.

#[derive(Debug)]
struct ComposeFailure(ToolError);

impl fmt::Display for ComposeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for ComposeFailure {}

/// The backend stopped the script (cancellation or the instruction budget).
/// Unlike a child failure, it fails the run even if the script catches it.
#[derive(Debug)]
struct ComposeStop(ToolError);

impl fmt::Display for ComposeStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for ComposeStop {}

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
    cancellation: Option<TurnCancellation>,
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
                return Err(mlua::Error::external(ComposeStop(ToolError::Cancelled)));
            }
            let previous = seen.fetch_add(u64::from(step), Ordering::Relaxed);
            if previous.saturating_add(u64::from(step)) > max_instruction_count {
                return Err(mlua::Error::external(ComposeStop(
                    ToolError::ExecutionFailed(format!(
                        "compose exceeded {max_instruction_count} Lua instructions"
                    )),
                )));
            }
            Ok(VmState::Continue)
        },
    )
}

fn lua_error_to_outcome(error: mlua::Error) -> ComposeOutcome {
    match &error {
        mlua::Error::CallbackError { cause, .. }
        | mlua::Error::BadArgument { cause, .. }
        | mlua::Error::WithContext { cause, .. } => {
            return lua_error_to_outcome((**cause).clone());
        }
        mlua::Error::ExternalError(inner) => {
            if let Some(failure) = inner.downcast_ref::<ComposeFailure>() {
                return ComposeOutcome::Failed(failure.0.clone());
            }
            if let Some(stop) = inner.downcast_ref::<ComposeStop>() {
                return ComposeOutcome::Failed(stop.0.clone());
            }
        }
        _ => {}
    }
    ComposeOutcome::Failed(ToolError::ExecutionFailed(error.to_string()))
}
