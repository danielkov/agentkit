//! Sandboxed Lua execution backend (the default).

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::Poll;

use agentkit_core::TurnCancellation;
use agentkit_tools_core::{ToolError, ToolInterruption, ToolName, ToolSpec};
use async_trait::async_trait;
use mlua::{HookTriggers, Lua, LuaSerdeExt, Value as LuaValue, VmState};
use serde_json::Value;

use crate::{
    BackendRun, CallKey, ChildCallError, ComposeBackend, ComposeOutcome, render_catalog_shapes,
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
             Call tool() from the main script, including loop bodies and plain functions it \
             calls, not from coroutines or from callbacks Lua cannot suspend, such as a \
             table.sort comparator. \
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
        // Why the attempt ended, recorded by a `tool()` call (approval, a
        // child's own cancellation, or a replay divergence) or by the VM hook
        // (turn cancellation or the instruction budget). It wins over whatever
        // the script does next.
        let stopped: Arc<OnceLock<AttemptStop>> = Arc::new(OnceLock::new());
        // The host-owned thread the script runs on; set once it exists.
        let main_thread_ptr = Arc::new(AtomicUsize::new(0));
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
        let dispatch_thread = main_thread_ptr.clone();
        let raw_tool = lua
            .create_async_function(move |lua, (name, lua_input): (String, LuaValue)| {
                // Everything that needs the VM happens before the dispatch
                // future exists: the future keeps only a weak handle, so a
                // parked call does not keep the VM (and with it this
                // dispatcher) alive after the attempt is dropped.
                let prepared = prepare_dispatch(
                    &lua,
                    &stopped_slot,
                    dispatch_thread.load(Ordering::Relaxed),
                    lua_input,
                );
                let weak_lua = lua.weak();
                drop(lua);
                let dispatcher = dispatcher.clone();
                let stopped_slot = stopped_slot.clone();
                async move {
                    let child_input = prepared?;
                    let stop = match dispatcher
                        .call_tracked(CallKey::Sequential, ToolName::new(name), child_input)
                        .await
                    {
                        Ok(output) => {
                            let lua = weak_lua.try_upgrade().ok_or_else(|| {
                                mlua::Error::external(ComposeStop(ToolError::Internal(
                                    "compose attempt already dropped".into(),
                                )))
                            })?;
                            return lua.to_value(&output);
                        }
                        Err(ChildCallError::Interrupted(interruption)) => {
                            AttemptStop::Interrupted(interruption)
                        }
                        Err(ChildCallError::Failed(ToolError::Cancelled)) => {
                            AttemptStop::Failed(ToolError::Cancelled)
                        }
                        // The child history no longer matches; nothing the
                        // script does next can be trusted.
                        Err(ChildCallError::ReplayDiverged(error)) => AttemptStop::Failed(error),
                        Err(ChildCallError::Failed(error)) => {
                            return Err(mlua::Error::external(ComposeFailure(error)));
                        }
                    };
                    let _ = stopped_slot.set(stop);
                    std::future::pending::<()>().await;
                    unreachable!("a pending future never resolves")
                }
            })
            .map_err(lua_error_to_outcome)?;
        let stop_check = stopped.clone();
        let is_stopped = lua
            .create_function(move |_, ()| Ok(stop_check.get().is_some()))
            .map_err(lua_error_to_outcome)?;
        // Host setup runs before the instruction hook is armed, so it never
        // counts against the script's budget.
        install_attempt_guards(&lua, raw_tool, is_stopped).map_err(lua_error_to_outcome)?;
        install_attempt_hook(
            &lua,
            run.config.max_instruction_count,
            run.cancellation.clone(),
            stopped.clone(),
        )
        .map_err(lua_error_to_outcome)?;
        // The script runs on one host-owned thread, created after the hook so
        // it and every coroutine it starts inherit it. Only that thread may
        // call `tool()`.
        let main_thread = compile_script(&lua, &run.script)
            .and_then(|script| lua.create_thread(script))
            .map_err(lua_error_to_outcome)?;
        main_thread_ptr.store(main_thread.to_pointer() as usize, Ordering::Relaxed);

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

/// Compiles the script the way `Chunk::eval` does: first as an expression,
/// so a bare `20 + 22` or `f(x)` returns its value, then as statements,
/// keeping the statement form's diagnostic when both fail.
fn compile_script(lua: &Lua, script: &str) -> Result<mlua::Function, mlua::Error> {
    lua.load(format!("return {script}"))
        .set_name("compose")
        .into_function()
        .or_else(|_| lua.load(script).set_name("compose").into_function())
}

/// Checks a `tool()` call and decodes its input before any child dispatch.
fn prepare_dispatch(
    lua: &Lua,
    stopped: &OnceLock<AttemptStop>,
    main_thread_ptr: usize,
    lua_input: LuaValue,
) -> Result<Value, mlua::Error> {
    if stopped.get().is_some() {
        return Err(mlua::Error::external(ComposeStop(ToolError::Internal(
            "compose attempt already stopped".into(),
        ))));
    }
    // A coroutine could hold or abandon a call after it used a sequential
    // ordinal, so calls are refused there before they use one.
    if lua.current_thread().to_pointer() as usize != main_thread_ptr {
        return Err(mlua::Error::external(ComposeFailure(
            ToolError::ExecutionFailed(
                "tool() must be called from the main compose script, not from a coroutine".into(),
            ),
        )));
    }
    lua.from_value(lua_input).map_err(|error| {
        mlua::Error::external(ComposeFailure(ToolError::InvalidInput(format!(
            "tool() input cannot be converted to JSON: {error}"
        ))))
    })
}

/// Lua-side guards installed before the script runs:
///
/// - `tool` is a wrapper that checks its arguments and that the calling frame
///   can yield before it reaches the raw dispatcher, which the script cannot
///   reach. A child call may suspend the script, so starting one from a frame
///   that cannot yield could use a sequential ordinal and then fail to
///   suspend.
/// - `pcall`, `xpcall`, `coroutine.resume` and `coroutine.close` re-raise once
///   the attempt has stopped, so no frame can catch a stop and keep running.
fn install_attempt_guards(
    lua: &Lua,
    raw_tool: mlua::Function,
    is_stopped: mlua::Function,
) -> Result<(), mlua::Error> {
    lua.load(ATTEMPT_GUARDS)
        .set_name("compose_guards")
        .call::<()>((raw_tool, is_stopped))
}

const ATTEMPT_GUARDS: &str = r#"
local raw_tool, is_stopped = ...
local is_yieldable = coroutine.isyieldable
local raw_pcall, raw_xpcall = pcall, xpcall
local raw_resume, raw_close = coroutine.resume, coroutine.close
local error, type = error, type

local function unless_stopped(...)
  if is_stopped() then
    error("compose attempt stopped", 0)
  end
  return ...
end

function tool(name, input)
  if type(name) ~= "string" then
    error("tool() expects a tool name string as its first argument, got " .. type(name), 2)
  end
  if not is_yieldable() then
    error("tool() cannot be called here: call it from a frame that can yield on the main " ..
          "compose script, not from a coroutine or a callback such as a sort comparator", 2)
  end
  return raw_tool(name, input)
end

function pcall(...) return unless_stopped(raw_pcall(...)) end
-- Lua runs the message handler before the error unwinds, with the hook
-- unable to interrupt it, so the handler is skipped once the attempt stops.
function xpcall(f, handler, ...)
  if type(handler) ~= "function" then
    return unless_stopped(raw_xpcall(f, handler, ...))
  end
  return unless_stopped(raw_xpcall(f, function(err)
    if is_stopped() then
      return err
    end
    return handler(err)
  end, ...))
end
coroutine.resume = function(...) return unless_stopped(raw_resume(...)) end
coroutine.close = function(...) return unless_stopped(raw_close(...)) end
"#;

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
/// stop or a turn cancellation still ends the script.
const STOP_CHECK_INSTRUCTIONS: u32 = 1_000;

/// Installs the VM hook that ends an attempt. Turn cancellation and the
/// instruction budget are recorded in `stopped` before control leaves the
/// hook, so a script cannot turn them into a result. Once any stop is
/// recorded, the hook raises on every check; the attempt guards re-raise
/// through `pcall` and friends, so the error unwinds every frame, including
/// ones that cannot yield, and the host's poll returns.
fn install_attempt_hook(
    lua: &Lua,
    max_instruction_count: u64,
    cancellation: Option<TurnCancellation>,
    stopped: Arc<OnceLock<AttemptStop>>,
) -> Result<(), mlua::Error> {
    let step = if max_instruction_count == 0 {
        STOP_CHECK_INSTRUCTIONS
    } else {
        max_instruction_count.min(u64::from(STOP_CHECK_INSTRUCTIONS)) as u32
    };
    let seen = Arc::new(AtomicU64::new(0));
    lua.set_global_hook(
        HookTriggers::new().every_nth_instruction(step),
        move |_lua, _debug| {
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
            match stopped.get() {
                None => Ok(VmState::Continue),
                Some(AttemptStop::Failed(error)) => {
                    Err(mlua::Error::external(ComposeStop(error.clone())))
                }
                Some(AttemptStop::Interrupted(_)) => Err(mlua::Error::external(ComposeStop(
                    ToolError::Internal("compose attempt stopped".into()),
                ))),
            }
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
