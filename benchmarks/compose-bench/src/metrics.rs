//! Run-level metric collection via the loop's [`LoopObserver`] event stream.
//!
//! The completions adapter emits exactly one `UsageUpdated` event per model
//! request, so counting those events gives the number of API round-trips and
//! summing them gives cumulative token usage. `ToolCallRequested` fires for
//! top-level tool calls only; tools invoked *inside* a compose script do not
//! re-enter the loop and are therefore not counted here.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use agentkit_loop::{AgentEvent, LoopObserver};
use serde::Serialize;

#[derive(Default, Clone, Serialize)]
pub struct MetricsState {
    pub model_requests: u64,
    pub tool_calls: u64,
    pub compose_calls: u64,
    pub tool_call_names: BTreeMap<String, u64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_write_tokens: u64,
    pub reasoning_tokens: u64,
    /// High-water mark of (input + cached input + output) tokens in a single
    /// request — i.e. how full the context window got.
    pub peak_context_tokens: u64,
    pub cost_usd: f64,
    pub cost_reported: bool,
}

pub struct MetricsObserver(pub Arc<Mutex<MetricsState>>);

impl LoopObserver for MetricsObserver {
    fn handle_event(&self, event: AgentEvent) {
        let mut state = self.0.lock().expect("metrics lock");
        match event {
            AgentEvent::UsageUpdated(usage) => {
                state.model_requests += 1;
                if let Some(tokens) = usage.tokens.as_ref() {
                    let cached = tokens.cached_input_tokens.unwrap_or(0);
                    state.input_tokens += tokens.input_tokens;
                    state.output_tokens += tokens.output_tokens;
                    state.cached_input_tokens += cached;
                    state.cache_write_tokens += tokens.cache_write_input_tokens.unwrap_or(0);
                    state.reasoning_tokens += tokens.reasoning_tokens.unwrap_or(0);
                    let context = tokens.input_tokens + cached + tokens.output_tokens;
                    state.peak_context_tokens = state.peak_context_tokens.max(context);
                }
                if let Some(cost) = usage.cost.as_ref() {
                    state.cost_usd += cost.amount;
                    state.cost_reported = true;
                }
            }
            AgentEvent::ToolCallRequested(call) => {
                state.tool_calls += 1;
                if call.name == agentkit_tool_compose::COMPOSE_TOOL_NAME {
                    state.compose_calls += 1;
                }
                *state
                    .tool_call_names
                    .entry(call.name.to_string())
                    .or_default() += 1;
            }
            _ => {}
        }
    }
}

/// One benchmark run, as persisted to `runs.jsonl`.
#[derive(Serialize)]
pub struct RunRecord {
    pub scenario: String,
    pub arm: String,
    pub rep: u32,
    pub model: String,
    pub wall_ms: u128,
    #[serde(flatten)]
    pub metrics: MetricsState,
    pub accuracy: f64,
    pub submitted: bool,
    pub completed: bool,
    pub failure: Option<String>,
    pub notes: Vec<String>,
}
