//! Mock [`Tool`] implementations that record their own invocations.
//!
//! Two flavours:
//!
//! - [`RecordingTool`] — wraps any closure that maps a [`ToolRequest`] to a
//!   [`ToolOutput`]. Records every invocation for assertion. Use this when
//!   tests want to script tool behaviour and inspect what got called.
//! - [`StaticTool`] — even thinner: returns the same [`ToolOutput`] for
//!   every call. Records invocations.

use std::sync::{Arc, Mutex};

use agentkit_core::{ToolOutput, ToolResultPart};
use agentkit_tools_core::{
    Tool, ToolContext, ToolError, ToolName, ToolRequest, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::json;

type Handler =
    dyn Fn(&ToolRequest) -> Result<ToolOutput, ToolError> + Send + Sync + 'static;

/// Mock tool with a closure-based handler.
///
/// Cheap to clone — shares the invocation log across clones.
#[derive(Clone)]
pub struct RecordingTool {
    spec: ToolSpec,
    invocations: Arc<Mutex<Vec<ToolRequest>>>,
    handler: Arc<Handler>,
}

impl RecordingTool {
    /// Build a tool with the given spec and handler. The handler is called
    /// for every invocation and its result is wrapped into a successful
    /// [`ToolResult`] (or an error if the closure returns one).
    pub fn new(
        spec: ToolSpec,
        handler: impl Fn(&ToolRequest) -> Result<ToolOutput, ToolError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            spec,
            invocations: Arc::new(Mutex::new(Vec::new())),
            handler: Arc::new(handler),
        }
    }

    /// Returns every recorded invocation, in call order.
    pub fn invocations(&self) -> Vec<ToolRequest> {
        self.invocations.lock().unwrap().clone()
    }

    /// Returns the number of recorded invocations.
    pub fn call_count(&self) -> usize {
        self.invocations.lock().unwrap().len()
    }
}

#[async_trait]
impl Tool for RecordingTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        _ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        self.invocations.lock().unwrap().push(request.clone());
        let output = (self.handler)(&request)?;
        Ok(ToolResult::new(ToolResultPart::success(
            request.call_id,
            output,
        )))
    }
}

/// Static-output mock tool. Returns the same [`ToolOutput`] for every call.
#[derive(Clone)]
pub struct StaticTool {
    inner: RecordingTool,
}

impl StaticTool {
    /// Build a tool returning `output` (cloned) for every invocation. The
    /// input schema accepts anything.
    pub fn new(name: impl Into<String>, description: impl Into<String>, output: ToolOutput) -> Self {
        let spec = ToolSpec::new(
            ToolName::new(name),
            description,
            json!({ "type": "object", "additionalProperties": true }),
        );
        let cloneable = output.clone();
        let inner = RecordingTool::new(spec, move |_| Ok(cloneable.clone()));
        Self { inner }
    }

    /// Returns every recorded invocation, in call order.
    pub fn invocations(&self) -> Vec<ToolRequest> {
        self.inner.invocations()
    }

    /// Returns the number of recorded invocations.
    pub fn call_count(&self) -> usize {
        self.inner.call_count()
    }
}

#[async_trait]
impl Tool for StaticTool {
    fn spec(&self) -> &ToolSpec {
        self.inner.spec()
    }

    async fn invoke(
        &self,
        request: ToolRequest,
        ctx: &mut ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        self.inner.invoke(request, ctx).await
    }
}
