//! Scenario abstractions shared by every benchmark case.
//!
//! A scenario owns a deterministic in-memory "world" (the mock SaaS backend),
//! exposes it through granular tools, and scores the run afterwards from the
//! world's final state plus whatever the agent submitted via `submit_result`.

use std::error::Error;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use agentkit_core::{ToolOutput, ToolResultPart};
use agentkit_tools_core::{
    CompositePermissionChecker, Tool, ToolContext, ToolError, ToolName, ToolRegistry, ToolRequest,
    ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{Value, json};

pub type BenchError = Box<dyn Error + Send + Sync>;

/// Which tool surface the agent gets for a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Scenario tools only, called one model round-trip at a time.
    Granular,
    /// Scenario tools plus the `compose` Lua tool wrapping them.
    Compose,
    /// `shell_exec` only (file-backed scenarios); the Bash-pipeline reference.
    Bash,
}

impl Arm {
    pub fn as_str(self) -> &'static str {
        match self {
            Arm::Granular => "granular",
            Arm::Compose => "compose",
            Arm::Bash => "bash",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "granular" => Some(Arm::Granular),
            "compose" => Some(Arm::Compose),
            "bash" => Some(Arm::Bash),
            _ => None,
        }
    }
}

/// Outcome of a scenario's verifier.
pub struct Score {
    /// 0.0..=1.0; partial credit per scenario-specific rubric.
    pub accuracy: f64,
    /// Human-readable rubric breakdown for the report.
    pub notes: Vec<String>,
}

/// Everything the harness needs to run one fresh attempt at a scenario.
pub struct ScenarioInstance {
    pub tools: ToolRegistry,
    pub user_prompt: String,
    pub permissions: Option<CompositePermissionChecker>,
    /// Slot `submit_result` writes into; the harness checks it for presence.
    pub submission: Submission,
    pub scorer: Box<dyn FnOnce() -> Score + Send>,
}

pub trait Scenario: Send + Sync {
    fn name(&self) -> &'static str;

    /// Arms this scenario supports. `Bash` only makes sense where the world is
    /// reachable from a shell (i.e. file-backed scenarios).
    fn arms(&self) -> Vec<Arm> {
        vec![Arm::Granular, Arm::Compose]
    }

    /// Builds a fresh world + tool registry. Must be deterministic.
    fn setup(&self, arm: Arm) -> Result<ScenarioInstance, BenchError>;
}

/// One system prompt for every scenario and arm, deliberately neutral about
/// composition so tool *descriptions* (not the prompt) drive tool preference.
pub const SYSTEM_PROMPT: &str = "\
You are an operations assistant completing a task with the available tools. \
Work autonomously and never ask the user questions. Be efficient: keep the \
number of steps as low as you can. When the task is complete, call the \
`submit_result` tool exactly once with the JSON shape requested in the task, \
then stop.";

static TOOL_LATENCY: OnceLock<Duration> = OnceLock::new();

/// Simulated per-call latency for mock tools, mimicking a remote MCP server.
/// Applied uniformly to every scenario tool in every arm.
pub fn set_tool_latency(latency: Duration) {
    let _ = TOOL_LATENCY.set(latency);
}

fn tool_latency() -> Duration {
    TOOL_LATENCY.get().copied().unwrap_or(Duration::ZERO)
}

/// A scenario tool backed by a synchronous closure over the shared world.
pub struct FnTool {
    spec: ToolSpec,
    #[allow(clippy::type_complexity)]
    handler: Box<dyn Fn(&Value) -> Result<Value, String> + Send + Sync>,
}

impl FnTool {
    pub fn new(
        name: &str,
        description: &str,
        input_schema: Value,
        output_schema: Value,
        handler: impl Fn(&Value) -> Result<Value, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            spec: ToolSpec::new(ToolName::new(name), description, input_schema)
                .with_output_schema(output_schema),
            handler: Box::new(handler),
        }
    }
}

#[async_trait]
impl Tool for FnTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let latency = tool_latency();
        if !latency.is_zero() {
            tokio::time::sleep(latency).await;
        }
        match (self.handler)(&request.input) {
            Ok(output) => Ok(ToolResult::new(ToolResultPart::success(
                request.call_id,
                ToolOutput::structured(output),
            ))),
            Err(message) => Ok(ToolResult::new(ToolResultPart::error(
                request.call_id,
                ToolOutput::text(message),
            ))),
        }
    }
}

/// Shared slot the `submit_result` tool writes the agent's final answer into.
pub type Submission = Arc<Mutex<Option<Value>>>;

/// Builds the `submit_result` tool every scenario registers. `answer_schema`
/// describes the scenario-specific payload so the model knows the exact shape.
pub fn submit_result_tool(answer_schema: Value) -> (FnTool, Submission) {
    let submission: Submission = Arc::new(Mutex::new(None));
    let slot = submission.clone();
    let tool = FnTool::new(
        "submit_result",
        "Submit the final answer for the task. Call exactly once, when the task is complete.",
        json!({
            "type": "object",
            "properties": { "answer": answer_schema },
            "required": ["answer"],
            "additionalProperties": false
        }),
        json!({
            "type": "object",
            "properties": { "recorded": { "type": "boolean" } },
            "required": ["recorded"]
        }),
        move |input| {
            let answer = input
                .get("answer")
                .cloned()
                .ok_or_else(|| "missing required field `answer`".to_string())?;
            *slot.lock().expect("submission lock") = Some(answer);
            Ok(json!({ "recorded": true }))
        },
    );
    (tool, submission)
}

/// 1-based pagination envelope used by the mock list endpoints.
pub fn paginate(items: Vec<Value>, page: u64, per_page: usize) -> Value {
    let total_items = items.len();
    let total_pages = total_items.div_ceil(per_page).max(1);
    let page = page.max(1) as usize;
    let start = (page - 1).saturating_mul(per_page);
    let slice: Vec<Value> = items.into_iter().skip(start).take(per_page).collect();
    json!({
        "items": slice,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_items": total_items,
    })
}

/// Output schema for the [`paginate`] envelope.
pub fn page_schema(item_schema: Value) -> Value {
    json!({
        "type": "object",
        "properties": {
            "items": { "type": "array", "items": item_schema },
            "page": { "type": "integer" },
            "per_page": { "type": "integer" },
            "total_pages": { "type": "integer" },
            "total_items": { "type": "integer" }
        },
        "required": ["items", "page", "total_pages", "total_items"]
    })
}

pub fn get_u64(input: &Value, key: &str, default: u64) -> u64 {
    input.get(key).and_then(Value::as_u64).unwrap_or(default)
}

pub fn get_str<'a>(input: &'a Value, key: &str) -> Result<&'a str, String> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required string field `{key}`"))
}

/// Symmetric set-overlap score (F1) between a submitted and expected id set.
pub fn f1(submitted: &[String], expected: &[String]) -> f64 {
    if expected.is_empty() {
        return if submitted.is_empty() { 1.0 } else { 0.0 };
    }
    if submitted.is_empty() {
        return 0.0;
    }
    let hits = submitted.iter().filter(|id| expected.contains(id)).count() as f64;
    let precision = hits / submitted.len() as f64;
    let recall = hits / expected.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}
