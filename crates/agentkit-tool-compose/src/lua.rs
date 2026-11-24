//! Sandboxed Lua execution backend (the default).

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
        // Why the attempt ended, recorded by a `tool()` call (approval or a
        // child's own cancellation) or by the VM hook (turn cancellation or the
        // instruction budget). It wins over whatever the script does next.
        let stopped: Arc<OnceLock<AttemptStop>> = Arc::new(OnceLock::new());
        let hook_main_thread = Arc::new(AtomicUsize::new(0));
        install_attempt_hook(
            &lua,
            run.config.max_instruction_count,
            run.cancellation.clone(),
            stopped.clone(),
            hook_main_thread.clone(),
        )
        .map_err(lua_error_to_outcome)?;
        // The script runs on one host-owned thread, created after the hook so
        // it inherits it. Only that thread may call `tool()`, and the hook
        // hands control back to the host from it.
        let main_thread = lua
            .load(run.script.as_str())
            .set_name("compose")
            .into_function()
            .and_then(|script| lua.create_thread(script))
            .map_err(lua_error_to_outcome)?;
        let main_thread_ptr = main_thread.to_pointer() as usize;
        hook_main_thread.store(main_thread_ptr, Ordering::Relaxed);
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

        // An approval interrupt or a child's own cancellation ends the attempt
        // at the `tool()` call that hit it: the call records the stop and never
        // resumes, so `pcall` cannot catch it. Because only the main thread may
        // dispatch, parking that call suspends the whole script and the host
        // sees the stop on the same poll. Any later dispatch (for example
        // during teardown) is refused before it reaches the dispatcher.
        let dispatcher = run.dispatcher.clone();
        let stopped_slot = stopped.clone();
        let tool_fn = lua
            .create_async_function(move |lua, (name, lua_input): (String, LuaValue)| {
                let dispatcher = dispatcher.clone();
                let stopped_slot = stopped_slot.clone();
                async move {
                    if stopped_slot.get().is_some() {
                        return Err(mlua::Error::external(ComposeStop(ToolError::Internal(
                            "compose attempt already stopped".into(),
                        ))));
                    }
                    // A coroutine could hold or abandon a call after it used a
                    // sequential ordinal, so calls are refused there before
                    // they use one.
                    if lua.current_thread().to_pointer() as usize != main_thread_ptr {
                        return Err(mlua::Error::external(ComposeFailure(
                            ToolError::ExecutionFailed(
                                "tool() must be called from the main compose script, not from a coroutine"
                                    .into(),
                            ),
                        )));
                    }
                    let child_input: Value = lua.from_value(lua_input)?;
                    let stop = match dispatcher
                        .call(CallKey::Sequential, ToolName::new(name), child_input)
                        .await
                    {
                        Ok(output) => return lua.to_value(&output),
                        Err(DispatchError::Interrupted(interruption)) => {
                            AttemptStop::Interrupted(interruption)
                        }
                        Err(DispatchError::Failed(ToolError::Cancelled)) => {
                            AttemptStop::Failed(ToolError::Cancelled)
                        }
                        Err(DispatchError::Failed(error)) => {
                            return Err(mlua::Error::external(ComposeFailure(error)));
                        }
                    };
                    let _ = stopped_slot.set(stop);
                    std::future::pending::<()>().await;
                    unreachable!("a pending future never resolves")
                }
            })
            .map_err(lua_error_to_outcome)?;
        globals.set("tool", tool_fn).map_err(lua_error_to_outcome)?;

        let mut evaluation = std::pin::pin!(
            main_thread
                .into_async::<LuaValue>(())
                .map_err(lua_error_to_outcome)?
        );
        // A recorded stop wins over whatever the poll returned, including a
        // finished script.
        let evaluated = std::future::poll_fn(|cx| {
            let polled = evaluation.as_mut().poll(cx);
            if stopped.get().is_some() {
                return Poll::Ready(None);
            }
            polled.map(Some)
        })
        .await;
        let Some(evaluated) = evaluated else {
            // A cancelled turn must not surface an approval prompt.
            if run.dispatcher.is_cancelled() {
                return Err(ComposeOutcome::Failed(ToolError::Cancelled));
            }
            return Err(match stopped.get().cloned() {
                Some(AttemptStop::Interrupted(interruption)) => {
                    ComposeOutcome::Interrupted(interruption)
                }
                Some(AttemptStop::Failed(error)) => ComposeOutcome::Failed(error),
                None => ComposeOutcome::Failed(ToolError::Internal(
                    "compose attempt stop was not recorded".into(),
                )),
            });
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

/// Why a Lua attempt ended at a `tool()` call.
#[derive(Clone, Debug)]
enum AttemptStop {
    Interrupted(ToolInterruption),
    Failed(ToolError),
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

/// How often the hook runs when no instruction budget is set, so a recorded
/// stop or a turn cancellation still reaches the host.
const STOP_CHECK_INSTRUCTIONS: u32 = 1_000;

/// Installs the VM hook that ends an attempt. Turn cancellation and the
/// instruction budget are recorded in `stopped` before control leaves the
/// hook, so a script that catches the resulting error cannot turn them into a
/// result. Once a stop is recorded, the hook hands control back to the host:
/// it yields the main thread where Lua allows it, and raises an error to
/// unwind frames that cannot yield (a coroutine, a metamethod, a C call).
fn install_attempt_hook(
    lua: &Lua,
    max_instruction_count: u64,
    cancellation: Option<TurnCancellation>,
    stopped: Arc<OnceLock<AttemptStop>>,
    main_thread_ptr: Arc<AtomicUsize>,
) -> Result<(), mlua::Error> {
    let step = if max_instruction_count == 0 {
        STOP_CHECK_INSTRUCTIONS
    } else {
        max_instruction_count.min(u64::from(STOP_CHECK_INSTRUCTIONS)) as u32
    };
    let seen = Arc::new(AtomicU64::new(0));
    // A yield the VM ignored (non-yieldable frame) leaves the hook running;
    // the next check raises instead, then yields again, and so on.
    let raise_next = Arc::new(AtomicBool::new(false));
    lua.set_global_hook(
        HookTriggers::new().every_nth_instruction(step),
        move |lua, _debug| {
            if stopped.get().is_none() {
                if cancellation
                    .as_ref()
                    .is_some_and(|cancellation| cancellation.is_cancelled())
                {
                    let _ = stopped.set(AttemptStop::Failed(ToolError::Cancelled));
                } else if max_instruction_count > 0 {
                    let previous = seen.fetch_add(u64::from(step), Ordering::Relaxed);
                    if previous.saturating_add(u64::from(step)) > max_instruction_count {
                        let _ = stopped.set(AttemptStop::Failed(ToolError::ExecutionFailed(
                            format!("compose exceeded {max_instruction_count} Lua instructions"),
                        )));
                    }
                }
            }
            let Some(stop) = stopped.get() else {
                return Ok(VmState::Continue);
            };
            // Only a yield of the script's own thread reaches the host; a
            // script coroutine's yield would just return to the script.
            let on_main = lua.current_thread().to_pointer() as usize
                == main_thread_ptr.load(Ordering::Relaxed);
            if on_main && !raise_next.swap(true, Ordering::Relaxed) {
                return Ok(VmState::Yield);
            }
            raise_next.store(false, Ordering::Relaxed);
            let error = match stop {
                AttemptStop::Failed(error) => error.clone(),
                AttemptStop::Interrupted(_) => {
                    ToolError::Internal("compose attempt already stopped".into())
                }
            };
            Err(mlua::Error::external(ComposeStop(error)))
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
