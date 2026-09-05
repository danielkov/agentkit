use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use agentkit_core::{
    Item, MetadataMap, TaskId, ToolCallId, ToolResultPart, TurnCancellation, TurnId,
};
use agentkit_tools_core::{
    ApprovalRequest, OwnedToolContext, ToolError, ToolExecutionOutcome, ToolExecutor, ToolRequest,
};
use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{Mutex, Notify, mpsc, oneshot};
use tokio::task::JoinHandle;

/// Host-owned typed terminal facts; never accepted from request metadata.
pub const TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY: &str =
    "agentkit.tool.failure_observations";

pub const TOOL_RESULT_FAILURE_METADATA_KEY: &str = "agentkit.tool.failure";

fn strip_failure_metadata(metadata: &mut MetadataMap) {
    for key in [
        TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY,
        TOOL_RESULT_FAILURE_METADATA_KEY,
        TOOL_RESULT_FAILURE_KIND_METADATA_KEY,
        TOOL_RESULT_NOT_STARTED_METADATA_KEY,
    ] {
        metadata.remove(key);
    }
}

/// Read the closed typed projection. Stored JSON still requires host provenance
/// validation at external transport boundaries; this helper validates shape only.
pub fn tool_failure_info(
    result: &ToolResultPart,
) -> Result<
    Option<agentkit_tools_core::ToolFailureInfo>,
    agentkit_core::failure::FailureMetadataDecodeError,
> {
    result
        .metadata
        .get(TOOL_RESULT_FAILURE_METADATA_KEY)
        .map(agentkit_tools_core::ToolFailureInfo::from_value)
        .transpose()
}

pub fn task_failure_observations(
    result: &ToolResultPart,
) -> Result<
    Option<agentkit_core::failure::FailureObservations>,
    agentkit_core::failure::FailureMetadataDecodeError,
> {
    result
        .metadata
        .get(TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY)
        .map(agentkit_core::failure::FailureObservations::from_value)
        .transpose()
}

fn attach_observations(
    resolution: &mut TaskResolution,
    observations: Option<&agentkit_core::failure::FailureObservations>,
) {
    if let Some(observations) = observations
        && let TaskResolution::Item(item) = resolution
    {
        for part in &mut item.parts {
            if let agentkit_core::Part::ToolResult(result) = part {
                result.metadata.insert(
                    TOOL_RESULT_FAILURE_OBSERVATIONS_METADATA_KEY.into(),
                    serde_json::to_value(observations).expect("finite task observations"),
                );
            }
        }
    }
}

fn write_failure_metadata(metadata: &mut MetadataMap, error: &ToolError, not_started: bool) {
    strip_failure_metadata(metadata);
    metadata.insert(
        TOOL_RESULT_FAILURE_METADATA_KEY.into(),
        serde_json::to_value(error.failure_info()).expect("finite failure facts"),
    );
    if error.is_permission_denied() {
        metadata.insert(
            TOOL_RESULT_FAILURE_KIND_METADATA_KEY.into(),
            TOOL_RESULT_FAILURE_KIND_PERMISSION_DENIED.into(),
        );
    }
    if not_started {
        metadata.insert(TOOL_RESULT_NOT_STARTED_METADATA_KEY.into(), true.into());
    }
}

pub const TOOL_RESULT_FAILURE_KIND_METADATA_KEY: &str = "agentkit.tool.failure_kind";
pub const TOOL_RESULT_FAILURE_KIND_PERMISSION_DENIED: &str = "permission_denied";
/// Marks a synthetic error result whose tool never began executing (failed
/// lookup, proposed-request error, or permission-checker denial). Distinct
/// from [`TOOL_RESULT_FAILURE_KIND_METADATA_KEY`]: a tool can fail with a
/// permission denial mid-execution, in which case it *did* start.
pub const TOOL_RESULT_NOT_STARTED_METADATA_KEY: &str = "agentkit.tool.not_started";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskKind {
    Foreground,
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuePolicy {
    NotifyOnly,
    RequestContinue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryMode {
    ToLoop,
    Manual,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskSnapshot {
    pub id: TaskId,
    pub turn_id: TurnId,
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub kind: TaskKind,
    pub metadata: MetadataMap,
    /// Selected terminal facts; cancellation may have no observed diagnostics.
    pub failure: Option<agentkit_tools_core::ToolFailureInfo>,
    /// Frozen task-owned scope, deliberately separate from native child facts.
    pub failure_observations: Option<agentkit_core::failure::FailureObservations>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TaskEvent {
    Started(TaskSnapshot),
    Detached(TaskSnapshot),
    Completed(TaskSnapshot, ToolResultPart),
    Cancelled(TaskSnapshot),
    Failed(TaskSnapshot, ToolError),
    ContinueRequested,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskApproval {
    pub task_id: TaskId,
    pub tool_request: ToolRequest,
    pub approval: ApprovalRequest,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TaskResolution {
    Item(Item),
    Approval(TaskApproval),
}

#[derive(Clone, Debug, PartialEq)]
pub enum TaskStartOutcome {
    Ready(Box<TaskResolution>),
    Pending { task_id: TaskId, kind: TaskKind },
}

#[derive(Clone, Debug, PartialEq)]
pub enum TurnTaskUpdate {
    Resolution(Box<TaskResolution>),
    Detached(Box<TaskSnapshot>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PendingLoopUpdates {
    pub resolutions: VecDeque<TaskResolution>,
}

/// How a task should be invoked. Mutually exclusive between plain execution
/// and resuming after approval.
#[derive(Clone, Debug, Default)]
pub enum TaskLaunchKind {
    /// Execute the tool with the active permission policy.
    #[default]
    Plain,
    /// Re-execute a previously-interrupted call after the user approved it.
    Approved(ApprovalRequest),
}

#[derive(Clone, Debug)]
pub struct TaskLaunchRequest {
    pub task_id: Option<TaskId>,
    pub request: ToolRequest,
    pub kind: TaskLaunchKind,
}

impl TaskLaunchRequest {
    /// Plain launch (no prior approval / auth).
    pub fn plain(task_id: Option<TaskId>, request: ToolRequest) -> Self {
        Self {
            task_id,
            request,
            kind: TaskLaunchKind::Plain,
        }
    }

    /// Resume after the user approved the call.
    pub fn approved(
        task_id: Option<TaskId>,
        request: ToolRequest,
        approval: ApprovalRequest,
    ) -> Self {
        Self {
            task_id,
            request,
            kind: TaskLaunchKind::Approved(approval),
        }
    }
}

#[derive(Clone)]
pub struct TaskStartContext {
    pub executor: Arc<dyn ToolExecutor>,
    pub tool_context: OwnedToolContext,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TaskManagerError {
    #[error("task not found: {0}")]
    NotFound(TaskId),
    #[error("task is not running: {0}")]
    NotRunning(TaskId),
    #[error("task is already running in the background: {0}")]
    AlreadyBackground(TaskId),
    #[error("task is already running: {0}")]
    AlreadyRunning(TaskId),
    #[error("invalid task continuation: {0}")]
    InvalidContinuation(TaskId),
    #[error("task manager internal error: {0}")]
    Internal(String),
}

pub trait TaskRoutingPolicy: Send + Sync {
    fn route(&self, request: &ToolRequest) -> RoutingDecision;
}

impl<F> TaskRoutingPolicy for F
where
    F: Fn(&ToolRequest) -> RoutingDecision + Send + Sync,
{
    fn route(&self, request: &ToolRequest) -> RoutingDecision {
        self(request)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingDecision {
    Foreground,
    Background,
    ForegroundThenDetachAfter(Duration),
}

struct DefaultRoutingPolicy;

impl TaskRoutingPolicy for DefaultRoutingPolicy {
    fn route(&self, _request: &ToolRequest) -> RoutingDecision {
        RoutingDecision::Foreground
    }
}

#[async_trait]
pub trait TaskManager: Send + Sync {
    async fn start_task(
        &self,
        request: TaskLaunchRequest,
        ctx: TaskStartContext,
    ) -> Result<TaskStartOutcome, TaskManagerError>;

    async fn wait_for_turn(
        &self,
        turn_id: &TurnId,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, TaskManagerError>;

    async fn take_pending_loop_updates(&self) -> Result<PendingLoopUpdates, TaskManagerError>;

    /// Wait until an update is available for delivery back into the agent loop.
    ///
    /// The default never resolves because custom managers that do not produce
    /// out-of-band loop updates have nothing to wake a host for.
    async fn wait_for_loop_update(&self) -> Result<(), TaskManagerError> {
        std::future::pending().await
    }

    async fn on_turn_interrupted(&self, turn_id: &TurnId) -> Result<(), TaskManagerError>;

    /// Drain the real frozen foreground cancellation results before a loop
    /// synthesizes results for remaining unanswered calls. Custom managers that
    /// delegate interruption must also delegate this method.
    fn take_interrupted_task_updates(
        &self,
        _session_id: &agentkit_core::SessionId,
        _call_ids: &[ToolCallId],
    ) -> Vec<TurnTaskUpdate> {
        Vec::new()
    }

    /// Terminalize an approval already delivered to the caller and transfer its
    /// frozen result directly, without enqueuing a second delivery. Wrappers
    /// must delegate this method alongside interruption and its scoped drain.
    async fn close_suspended_task(
        &self,
        _task_id: &TaskId,
        _approval_id: &agentkit_core::ApprovalId,
        _error: ToolError,
    ) -> Result<Option<TaskResolution>, TaskManagerError> {
        Ok(None)
    }

    fn handle(&self) -> TaskManagerHandle;
}

#[async_trait]
trait TaskManagerControl: Send + Sync {
    async fn next_event(&self) -> Option<TaskEvent>;
    async fn cancel(&self, task_id: TaskId) -> Result<(), TaskManagerError>;
    async fn detach(&self, task_id: TaskId) -> Result<(), TaskManagerError>;
    async fn list_running(&self) -> Vec<TaskSnapshot>;
    async fn list_completed(&self) -> Vec<TaskSnapshot>;
    async fn list_suspended(&self) -> Vec<TaskSnapshot>;
    async fn drain_ready_items(&self) -> Vec<Item>;
    async fn set_continue_policy(
        &self,
        task_id: TaskId,
        policy: ContinuePolicy,
    ) -> Result<(), TaskManagerError>;
    async fn set_delivery_mode(
        &self,
        task_id: TaskId,
        mode: DeliveryMode,
    ) -> Result<(), TaskManagerError>;
    async fn wait_for_idle(&self);
}

#[derive(Clone)]
pub struct TaskManagerHandle {
    inner: Arc<dyn TaskManagerControl>,
}

impl TaskManagerHandle {
    pub async fn next_event(&self) -> Option<TaskEvent> {
        self.inner.next_event().await
    }

    pub async fn cancel(&self, task_id: TaskId) -> Result<(), TaskManagerError> {
        self.inner.cancel(task_id).await
    }

    /// Detach a running foreground task so it continues in the background.
    pub async fn detach(&self, task_id: TaskId) -> Result<(), TaskManagerError> {
        self.inner.detach(task_id).await
    }

    pub async fn list_running(&self) -> Vec<TaskSnapshot> {
        self.inner.list_running().await
    }

    pub async fn list_completed(&self) -> Vec<TaskSnapshot> {
        self.inner.list_completed().await
    }

    pub async fn drain_ready_items(&self) -> Vec<Item> {
        self.inner.drain_ready_items().await
    }

    pub async fn set_continue_policy(
        &self,
        task_id: TaskId,
        policy: ContinuePolicy,
    ) -> Result<(), TaskManagerError> {
        self.inner.set_continue_policy(task_id, policy).await
    }

    pub async fn set_delivery_mode(
        &self,
        task_id: TaskId,
        mode: DeliveryMode,
    ) -> Result<(), TaskManagerError> {
        self.inner.set_delivery_mode(task_id, mode).await
    }

    /// Invocations awaiting approval are suspended, not terminally completed.
    pub async fn list_suspended(&self) -> Vec<TaskSnapshot> {
        self.inner.list_suspended().await
    }

    /// Wait until all running tasks have completed.
    pub async fn wait_for_idle(&self) {
        self.inner.wait_for_idle().await
    }
}

pub struct SimpleTaskManager {
    state: Arc<HandleState>,
}

impl SimpleTaskManager {
    pub fn new() -> Self {
        Self {
            state: Arc::new(HandleState::default()),
        }
    }
}

impl Default for SimpleTaskManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TaskManager for SimpleTaskManager {
    async fn start_task(
        &self,
        request: TaskLaunchRequest,
        ctx: TaskStartContext,
    ) -> Result<TaskStartOutcome, TaskManagerError> {
        let mut request = request;
        strip_failure_metadata(&mut request.request.metadata);
        let mut ctx = ctx;
        strip_failure_metadata(&mut ctx.tool_context.metadata);
        let task_id = request
            .task_id
            .clone()
            .unwrap_or_else(|| self.state.next_task_id());
        let generation = Arc::new(());
        let cancel_signal = Arc::new(Notify::new());
        let observations;
        {
            let mut tasks = self
                .state
                .tasks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            observations = if let Some(record) = tasks.get(&task_id) {
                if record.running {
                    return Err(TaskManagerError::AlreadyRunning(task_id));
                }
                let Some(saved) = &record.suspended else {
                    return Err(TaskManagerError::InvalidContinuation(task_id));
                };
                if !continuation_matches(saved, &request) {
                    return Err(TaskManagerError::InvalidContinuation(task_id));
                }
                observation_slot(record.snapshot.failure_observations.as_ref())
            } else {
                if matches!(&request.kind, TaskLaunchKind::Approved(_)) {
                    return Err(TaskManagerError::InvalidContinuation(task_id));
                }
                observation_slot(None)
            };
            let snapshot = TaskSnapshot {
                id: task_id.clone(),
                turn_id: request.request.turn_id.clone(),
                call_id: request.request.call_id.clone(),
                tool_name: request.request.tool_name.to_string(),
                kind: TaskKind::Foreground,
                metadata: request.request.metadata.clone(),
                failure: None,
                failure_observations: None,
            };
            tasks.insert(
                task_id.clone(),
                TaskRecord {
                    cancel_signal: Some(cancel_signal.clone()),
                    generation: generation.clone(),
                    session_id: request.request.session_id.clone(),
                    suspended: None,
                    observations: observations.clone(),
                    snapshot: snapshot.clone(),
                    continue_policy: ContinuePolicy::NotifyOnly,
                    delivery_mode: DeliveryMode::ToLoop,
                    running: true,
                    invocation_admitted: true,
                    completed: false,
                    join: None,
                },
            );
            let _ = self.state.events_tx.send(TaskEvent::Started(snapshot));
        }
        // Registered synchronously before the first await. Dropping the inline
        // future seals all producer clones and retains a cancellation result.
        let mut owner = InlineOwner {
            state: self.state.clone(),
            task_id: task_id.clone(),
            generation: generation.clone(),
            armed: true,
        };
        ctx.tool_context.failure_observer = Some(observations.publisher());
        let invoke = async {
            match &request.kind {
                TaskLaunchKind::Approved(approval) => {
                    ctx.executor
                        .execute_approved_owned(request.request.clone(), approval, ctx.tool_context)
                        .await
                }
                TaskLaunchKind::Plain => {
                    ctx.executor
                        .execute_owned(request.request.clone(), ctx.tool_context)
                        .await
                }
            }
        };
        let outcome = tokio::select! {
            outcome = invoke => outcome,
            _ = cancel_signal.notified() => ToolExecutionOutcome::Failed(ToolError::Cancelled),
        };
        let mut tasks = self
            .state
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let record = tasks
            .get_mut(&task_id)
            .expect("inline owner retains registered record");
        if !Arc::ptr_eq(&record.generation, &generation) {
            return Err(TaskManagerError::InvalidContinuation(task_id));
        }
        if !record.running {
            let resolution = cancellation_resolution(&record.snapshot);
            let _ = drain_interrupted_updates(
                &self.state.interrupted_updates,
                &record.session_id,
                std::slice::from_ref(&record.snapshot.call_id),
            );
            owner.armed = false;
            return Ok(TaskStartOutcome::Ready(Box::new(resolution)));
        }
        let not_started = matches!(&outcome, ToolExecutionOutcome::FailedBeforeInvocation(_));
        let error = match &outcome {
            ToolExecutionOutcome::Failed(error)
            | ToolExecutionOutcome::FailedBeforeInvocation(error) => Some(error.clone()),
            _ => None,
        };
        let mut resolution = map_outcome_to_resolution(Some(task_id), request.request, outcome);
        record.running = false;
        record.completed = !matches!(&resolution, TaskResolution::Approval(_));
        record.suspended = match &resolution {
            TaskResolution::Approval(task) => {
                Some(SuspendedTask::new(&task.tool_request, &task.approval))
            }
            _ => None,
        };
        record.snapshot.failure = error.as_ref().map(ToolError::failure_info);
        if not_started {
            record
                .snapshot
                .metadata
                .insert(TOOL_RESULT_NOT_STARTED_METADATA_KEY.into(), true.into());
        }
        let frozen = observations.seal();
        record.snapshot.failure_observations = if error.is_some() || record.suspended.is_some() {
            frozen
        } else {
            None
        };
        if error.is_some() {
            attach_observations(
                &mut resolution,
                record.snapshot.failure_observations.as_ref(),
            );
        }
        if let Some(error) = error {
            let event = if error.is_cancelled() {
                TaskEvent::Cancelled(record.snapshot.clone())
            } else {
                TaskEvent::Failed(record.snapshot.clone(), error)
            };
            let _ = self.state.events_tx.send(event);
        } else if let TaskResolution::Item(item) = &resolution {
            for part in &item.parts {
                if let agentkit_core::Part::ToolResult(result) = part {
                    let _ = self.state.events_tx.send(TaskEvent::Completed(
                        record.snapshot.clone(),
                        result.clone(),
                    ));
                }
            }
        }
        owner.armed = false;
        self.state.notify.notify_waiters();
        Ok(TaskStartOutcome::Ready(Box::new(resolution)))
    }

    async fn wait_for_turn(
        &self,
        _turn_id: &TurnId,
        _cancellation: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, TaskManagerError> {
        Ok(None)
    }

    async fn take_pending_loop_updates(&self) -> Result<PendingLoopUpdates, TaskManagerError> {
        Ok(PendingLoopUpdates::default())
    }

    async fn on_turn_interrupted(&self, turn_id: &TurnId) -> Result<(), TaskManagerError> {
        let ids: Vec<_> = self
            .state
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .filter(|record| {
                record.snapshot.turn_id == *turn_id
                    && (record.running || record.suspended.is_some())
            })
            .map(|record| record.snapshot.id.clone())
            .collect();
        for id in ids {
            self.state.cancel_inline(&id, None)?;
        }
        Ok(())
    }

    fn take_interrupted_task_updates(
        &self,
        session_id: &agentkit_core::SessionId,
        call_ids: &[ToolCallId],
    ) -> Vec<TurnTaskUpdate> {
        drain_interrupted_updates(&self.state.interrupted_updates, session_id, call_ids)
    }

    async fn close_suspended_task(
        &self,
        task_id: &TaskId,
        approval_id: &agentkit_core::ApprovalId,
        error: ToolError,
    ) -> Result<Option<TaskResolution>, TaskManagerError> {
        let mut tasks = self
            .state
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let record = tasks
            .get_mut(task_id)
            .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
        if record.completed {
            return Ok(take_interrupted_resolution(
                &self.state.interrupted_updates,
                task_id,
            ));
        }
        let result = close_suspended_record(record, approval_id, &error)?;
        if result.is_some() {
            let _ = self
                .state
                .events_tx
                .send(failure_event(record.snapshot.clone(), error));
            self.state.notify.notify_waiters();
        }
        Ok(result)
    }

    fn handle(&self) -> TaskManagerHandle {
        TaskManagerHandle {
            inner: self.state.clone(),
        }
    }
}

struct HandleState {
    next_task_index: AtomicU64,
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<TaskEvent>>>,
    events_tx: mpsc::UnboundedSender<TaskEvent>,
    tasks: std::sync::Mutex<BTreeMap<TaskId, TaskRecord>>,
    interrupted_updates: std::sync::Mutex<Vec<InterruptedUpdate>>,
    notify: Notify,
}
impl Default for HandleState {
    fn default() -> Self {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        Self {
            next_task_index: AtomicU64::new(0),
            events_rx: Mutex::new(Some(events_rx)),
            events_tx,
            tasks: std::sync::Mutex::new(BTreeMap::new()),
            interrupted_updates: std::sync::Mutex::new(Vec::new()),
            notify: Notify::new(),
        }
    }
}
struct InlineOwner {
    state: Arc<HandleState>,
    task_id: TaskId,
    generation: Arc<()>,
    armed: bool,
}
impl Drop for InlineOwner {
    fn drop(&mut self) {
        if self.armed {
            let _ = self
                .state
                .cancel_inline(&self.task_id, Some(&self.generation));
        }
    }
}

impl HandleState {
    fn cancel_inline(
        &self,
        task_id: &TaskId,
        generation: Option<&Arc<()>>,
    ) -> Result<(), TaskManagerError> {
        let mut tasks = self
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let record = tasks
            .get_mut(task_id)
            .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
        if generation.is_some_and(|generation| !Arc::ptr_eq(generation, &record.generation))
            || (!record.running && record.suspended.is_none())
        {
            return Ok(());
        }
        record.running = false;
        record.completed = true;
        record.suspended = None;
        record.snapshot.failure = Some(ToolError::Cancelled.failure_info());
        record.snapshot.failure_observations = record.observations.seal();
        self.interrupted_updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(InterruptedUpdate {
                task_id: task_id.clone(),
                session_id: record.session_id.clone(),
                update: TurnTaskUpdate::Resolution(Box::new(cancellation_resolution(
                    &record.snapshot,
                ))),
            });
        let _ = self
            .events_tx
            .send(TaskEvent::Cancelled(record.snapshot.clone()));
        if let Some(signal) = &record.cancel_signal {
            signal.notify_one();
        }
        self.notify.notify_waiters();
        Ok(())
    }

    fn next_task_id(&self) -> TaskId {
        let next = self.next_task_index.fetch_add(1, Ordering::SeqCst) + 1;
        TaskId::new(format!("task-{}", next))
    }
}

#[async_trait]
impl TaskManagerControl for HandleState {
    async fn next_event(&self) -> Option<TaskEvent> {
        let mut rx = self.events_rx.lock().await;
        match rx.as_mut() {
            Some(inner) => inner.recv().await,
            None => None,
        }
    }

    async fn cancel(&self, task_id: TaskId) -> Result<(), TaskManagerError> {
        self.cancel_inline(&task_id, None)
    }

    async fn detach(&self, task_id: TaskId) -> Result<(), TaskManagerError> {
        Err(TaskManagerError::NotFound(task_id))
    }

    async fn list_running(&self) -> Vec<TaskSnapshot> {
        self.tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .filter(|record| record.running)
            .map(|record| record.snapshot.clone())
            .collect()
    }

    async fn list_completed(&self) -> Vec<TaskSnapshot> {
        self.tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .filter(|record| record.completed)
            .map(|record| record.snapshot.clone())
            .collect()
    }

    async fn list_suspended(&self) -> Vec<TaskSnapshot> {
        self.tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .filter(|record| record.suspended.is_some())
            .map(|record| record.snapshot.clone())
            .collect()
    }

    async fn drain_ready_items(&self) -> Vec<Item> {
        Vec::new()
    }

    async fn set_continue_policy(
        &self,
        task_id: TaskId,
        _policy: ContinuePolicy,
    ) -> Result<(), TaskManagerError> {
        Err(TaskManagerError::NotFound(task_id))
    }

    async fn set_delivery_mode(
        &self,
        task_id: TaskId,
        _mode: DeliveryMode,
    ) -> Result<(), TaskManagerError> {
        Err(TaskManagerError::NotFound(task_id))
    }

    async fn wait_for_idle(&self) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self
                .tasks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .values()
                .any(|record| record.running)
            {
                return;
            }
            notified.await;
        }
    }
}

pub struct AsyncTaskManager {
    inner: Arc<AsyncInner>,
    routing: Arc<dyn TaskRoutingPolicy>,
}

impl AsyncTaskManager {
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        Self {
            inner: Arc::new(AsyncInner {
                state: Mutex::new(AsyncState::default()),
                interrupted_updates: std::sync::Mutex::new(Vec::new()),
                host_event_tx: event_tx,
                host_event_rx: Mutex::new(event_rx),
                notify: Notify::new(),
            }),
            routing: Arc::new(DefaultRoutingPolicy),
        }
    }

    pub fn routing(mut self, policy: impl TaskRoutingPolicy + 'static) -> Self {
        self.routing = Arc::new(policy);
        self
    }
}

impl Default for AsyncTaskManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct AsyncState {
    next_task_index: u64,
    tasks: BTreeMap<TaskId, TaskRecord>,
    per_turn_running: BTreeMap<TurnId, usize>,
    per_turn_updates: BTreeMap<TurnId, VecDeque<InterruptedUpdate>>,
    pending_loop_updates: VecDeque<(TaskId, TaskResolution)>,
    manual_ready_items: Vec<Item>,
}

#[derive(Clone)]
struct SuspendedTask {
    session_id: agentkit_core::SessionId,
    turn_id: TurnId,
    call_id: ToolCallId,
    tool_name: agentkit_tools_core::ToolName,
    approval_id: agentkit_core::ApprovalId,
}
impl SuspendedTask {
    fn new(request: &ToolRequest, approval: &ApprovalRequest) -> Self {
        Self {
            session_id: request.session_id.clone(),
            turn_id: request.turn_id.clone(),
            call_id: request.call_id.clone(),
            tool_name: request.tool_name.clone(),
            approval_id: approval.id.clone(),
        }
    }
}
fn continuation_matches(saved: &SuspendedTask, launch: &TaskLaunchRequest) -> bool {
    // Approval explicitly supports host-patched input. Correlate immutable
    // logical-call identity, not arguments or caller metadata.
    matches!(&launch.kind, TaskLaunchKind::Approved(approval) if approval.id == saved.approval_id)
        && launch.request.session_id == saved.session_id
        && launch.request.turn_id == saved.turn_id
        && launch.request.call_id == saved.call_id
        && launch.request.tool_name == saved.tool_name
}

fn observation_slot(
    previous: Option<&agentkit_core::failure::FailureObservations>,
) -> agentkit_tools_core::FailureObservationSlot {
    let slot = agentkit_tools_core::FailureObservationSlot::new();
    if let Some(previous) = previous {
        let publisher = slot.publisher();
        if let Some(value) = previous.effects() {
            publisher
                .publish_effects(*value)
                .expect("validated continuation effects");
        }
        if let Some(value) = previous.receipt() {
            publisher
                .publish_receipt(value.clone())
                .expect("validated continuation receipt");
        }
        if let Some(value) = previous.retry() {
            publisher
                .publish_retry(*value)
                .expect("validated continuation retry");
        }
    }
    slot
}

struct TaskRecord {
    cancel_signal: Option<Arc<Notify>>,
    suspended: Option<SuspendedTask>,
    session_id: agentkit_core::SessionId,
    generation: Arc<()>,
    observations: agentkit_tools_core::FailureObservationSlot,
    snapshot: TaskSnapshot,
    continue_policy: ContinuePolicy,
    delivery_mode: DeliveryMode,
    running: bool,
    invocation_admitted: bool,
    completed: bool,
    join: Option<JoinHandle<()>>,
}

struct InterruptedUpdate {
    task_id: TaskId,
    session_id: agentkit_core::SessionId,
    update: TurnTaskUpdate,
}

fn update_call_id(update: &TurnTaskUpdate) -> Option<&ToolCallId> {
    match update {
        TurnTaskUpdate::Detached(snapshot) => Some(&snapshot.call_id),
        TurnTaskUpdate::Resolution(resolution) => match resolution.as_ref() {
            TaskResolution::Approval(task) => Some(&task.tool_request.call_id),
            TaskResolution::Item(item) => item.parts.iter().find_map(|part| match part {
                agentkit_core::Part::ToolResult(result) => Some(&result.call_id),
                _ => None,
            }),
        },
    }
}

fn take_interrupted_resolution(
    queue: &std::sync::Mutex<Vec<InterruptedUpdate>>,
    task_id: &TaskId,
) -> Option<TaskResolution> {
    let mut queue = queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let index = queue.iter().position(|entry| {
        &entry.task_id == task_id && matches!(&entry.update, TurnTaskUpdate::Resolution(_))
    })?;
    match queue.remove(index).update {
        TurnTaskUpdate::Resolution(resolution) => Some(*resolution),
        _ => unreachable!(),
    }
}

fn drain_interrupted_updates(
    queue: &std::sync::Mutex<Vec<InterruptedUpdate>>,
    session_id: &agentkit_core::SessionId,
    call_ids: &[ToolCallId],
) -> Vec<TurnTaskUpdate> {
    let mut queue = queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut selected = Vec::new();
    let mut retained = Vec::new();
    for entry in std::mem::take(&mut *queue) {
        if &entry.session_id == session_id
            && update_call_id(&entry.update).is_some_and(|id| call_ids.contains(id))
        {
            selected.push(entry.update);
        } else {
            retained.push(entry);
        }
    }
    *queue = retained;
    selected
}

struct AsyncInner {
    interrupted_updates: std::sync::Mutex<Vec<InterruptedUpdate>>,
    state: Mutex<AsyncState>,
    host_event_tx: mpsc::UnboundedSender<TaskEvent>,
    host_event_rx: Mutex<mpsc::UnboundedReceiver<TaskEvent>>,
    notify: Notify,
}

impl AsyncInner {
    async fn next_task_id(&self) -> TaskId {
        let mut state = self.state.lock().await;
        state.next_task_index += 1;
        TaskId::new(format!("task-{}", state.next_task_index))
    }

    async fn detach_running_foreground(
        &self,
        task_id: &TaskId,
        generation: Option<&Arc<()>>,
    ) -> Result<(), TaskManagerError> {
        let mut state = self.state.lock().await;
        let (session_id, snapshot) = {
            let record = state
                .tasks
                .get_mut(task_id)
                .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
            if !record.running
                || generation.is_some_and(|generation| !Arc::ptr_eq(generation, &record.generation))
            {
                return Err(TaskManagerError::NotRunning(task_id.clone()));
            }
            if record.snapshot.kind == TaskKind::Background {
                return Err(TaskManagerError::AlreadyBackground(task_id.clone()));
            }
            record.snapshot.kind = TaskKind::Background;
            (record.session_id.clone(), record.snapshot.clone())
        };

        if let Some(count) = state.per_turn_running.get_mut(&snapshot.turn_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.per_turn_running.remove(&snapshot.turn_id);
            }
        }
        state
            .per_turn_updates
            .entry(snapshot.turn_id.clone())
            .or_default()
            .push_back(InterruptedUpdate {
                task_id: task_id.clone(),
                session_id,
                update: TurnTaskUpdate::Detached(Box::new(snapshot.clone())),
            });
        let _ = self.host_event_tx.send(TaskEvent::Detached(snapshot));
        self.notify.notify_waiters();
        Ok(())
    }

    async fn interrupt_turn(&self, turn_id: &TurnId) {
        let mut state = self.state.lock().await;
        // Transfer already-selected foreground winners before cancelling the
        // remaining tasks. A turn cancellation must not replace a queued result.
        if let Some(queued) = state.per_turn_updates.remove(turn_id) {
            let mut closed = self
                .interrupted_updates
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            closed.extend(queued.into_iter().filter(|entry| !matches!(&entry.update, TurnTaskUpdate::Resolution(resolution) if matches!(resolution.as_ref(), TaskResolution::Approval(_)))));
        }
        let interrupted: Vec<TaskId> = state
            .tasks
            .iter()
            .filter_map(|(id, record)| {
                (record.snapshot.turn_id == *turn_id
                    && ((record.snapshot.kind == TaskKind::Foreground && record.running)
                        || record.suspended.is_some()))
                .then_some(id.clone())
            })
            .collect();
        // An approval still in this queue has not been handed to LoopDriver.
        // Its background terminal result must retain its normal destination.
        let unsurfaced: Vec<_> = state
            .pending_loop_updates
            .iter()
            .filter_map(|(_, resolution)| match resolution {
                TaskResolution::Approval(task) => Some(task.task_id.clone()),
                _ => None,
            })
            .collect();
        state.pending_loop_updates.retain(|(_, resolution)| !matches!(resolution, TaskResolution::Approval(task) if interrupted.contains(&task.task_id)));
        let mut aborts = Vec::new();
        for task_id in interrupted {
            if let Some(record) = state.tasks.get_mut(&task_id) {
                record.running = false;
                record.suspended = None;
                record.completed = true;
                if !record.invocation_admitted {
                    record
                        .snapshot
                        .metadata
                        .insert(TOOL_RESULT_NOT_STARTED_METADATA_KEY.into(), true.into());
                }
                record.snapshot.failure = Some(ToolError::Cancelled.failure_info());
                record.snapshot.failure_observations = record.observations.seal();
                if let Some(join) = record.join.take() {
                    aborts.push(join);
                }
                let snapshot = record.snapshot.clone();
                let session_id = record.session_id.clone();
                let delivery_mode = record.delivery_mode;
                let continue_policy = record.continue_policy;
                let resolution = cancellation_resolution(&snapshot);
                if snapshot.kind == TaskKind::Background && delivery_mode == DeliveryMode::Manual {
                    if let TaskResolution::Item(item) = resolution {
                        state.manual_ready_items.push(item);
                    }
                } else if snapshot.kind == TaskKind::Background && unsurfaced.contains(&task_id) {
                    state
                        .pending_loop_updates
                        .push_back((task_id.clone(), resolution));
                    if continue_policy == ContinuePolicy::RequestContinue {
                        let _ = self.host_event_tx.send(TaskEvent::ContinueRequested);
                    }
                } else {
                    self.interrupted_updates
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(InterruptedUpdate {
                            task_id: task_id.clone(),
                            session_id,
                            update: TurnTaskUpdate::Resolution(Box::new(resolution)),
                        });
                }
                let _ = self.host_event_tx.send(TaskEvent::Cancelled(snapshot));
            }
        }
        state.per_turn_running.remove(turn_id);
        drop(state);
        for join in aborts {
            join.abort();
        }
        self.notify.notify_waiters();
    }
}

#[async_trait]
impl TaskManager for AsyncTaskManager {
    async fn start_task(
        &self,
        request: TaskLaunchRequest,
        ctx: TaskStartContext,
    ) -> Result<TaskStartOutcome, TaskManagerError> {
        let mut request = request;
        strip_failure_metadata(&mut request.request.metadata);
        let route = self.routing.route(&request.request);
        let task_id = match request.task_id.clone() {
            Some(existing) => existing,
            None => self.inner.next_task_id().await,
        };
        let initial_kind = match route {
            RoutingDecision::Background => TaskKind::Background,
            _ => TaskKind::Foreground,
        };
        let snapshot = TaskSnapshot {
            id: task_id.clone(),
            turn_id: request.request.turn_id.clone(),
            call_id: request.request.call_id.clone(),
            tool_name: request.request.tool_name.to_string(),
            kind: initial_kind,
            metadata: request.request.metadata.clone(),
            failure: None,
            failure_observations: None,
        };
        let mut state = self.inner.state.lock().await;
        let (observations, continue_policy, delivery_mode) =
            if let Some(record) = state.tasks.get(&task_id) {
                if record.running {
                    return Err(TaskManagerError::AlreadyRunning(task_id));
                }
                let Some(saved) = &record.suspended else {
                    return Err(TaskManagerError::InvalidContinuation(task_id));
                };
                if !continuation_matches(saved, &request) {
                    return Err(TaskManagerError::InvalidContinuation(task_id));
                }
                (
                    observation_slot(record.snapshot.failure_observations.as_ref()),
                    record.continue_policy,
                    record.delivery_mode,
                )
            } else {
                if matches!(&request.kind, TaskLaunchKind::Approved(_)) {
                    return Err(TaskManagerError::InvalidContinuation(task_id));
                }
                (
                    observation_slot(None),
                    ContinuePolicy::NotifyOnly,
                    DeliveryMode::ToLoop,
                )
            };
        let initial_kind = snapshot.kind;
        let generation = Arc::new(());
        state.tasks.insert(
            task_id.clone(),
            TaskRecord {
                cancel_signal: None,
                generation: generation.clone(),
                session_id: request.request.session_id.clone(),
                suspended: None,
                observations: observations.clone(),
                snapshot: snapshot.clone(),
                continue_policy,
                delivery_mode,
                running: true,
                invocation_admitted: false,
                completed: false,
                join: None,
            },
        );
        if initial_kind == TaskKind::Foreground {
            *state
                .per_turn_running
                .entry(snapshot.turn_id.clone())
                .or_default() += 1;
        }
        // Registration, Started publication, and worker ownership have no
        // cancellation point between them.
        let _ = self
            .inner
            .host_event_tx
            .send(TaskEvent::Started(snapshot.clone()));

        let event_tx = self.inner.host_event_tx.clone();
        let inner = self.inner.clone();
        let task_id_for_future = task_id.clone();
        let turn_id = snapshot.turn_id.clone();
        let kind = request.kind.clone();
        let exec_request = request.request.clone();
        let mut owned_ctx = ctx.tool_context.clone();
        owned_ctx.failure_observer = Some(observations.publisher());
        strip_failure_metadata(&mut owned_ctx.metadata);
        let generation_for_future = generation.clone();
        let executor = ctx.executor.clone();
        let route_copy = if initial_kind == TaskKind::Background {
            RoutingDecision::Background
        } else {
            route
        };
        let (start_tx, start_rx) = oneshot::channel();
        let join = tokio::spawn(async move {
            if start_rx.await.is_err() {
                return;
            }
            {
                let mut state = inner.state.lock().await;
                let Some(record) = state.tasks.get_mut(&task_id_for_future) else {
                    return;
                };
                if !record.running || !Arc::ptr_eq(&record.generation, &generation_for_future) {
                    return;
                }
                // Admission is not an effects observation. False does prove the
                // executor has not been entered when cancellation wins early.
                record.invocation_admitted = true;
            }
            if let RoutingDecision::ForegroundThenDetachAfter(duration) = route_copy {
                let inner = inner.clone();
                let task_id = task_id_for_future.clone();
                let generation = generation_for_future.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(duration).await;
                    let _ = inner
                        .detach_running_foreground(&task_id, Some(&generation))
                        .await;
                });
            }

            let outcome = match &kind {
                TaskLaunchKind::Approved(approval) => {
                    executor
                        .execute_approved_owned(exec_request.clone(), approval, owned_ctx)
                        .await
                }
                TaskLaunchKind::Plain => {
                    executor
                        .execute_owned(exec_request.clone(), owned_ctx)
                        .await
                }
            };

            let not_started = matches!(&outcome, ToolExecutionOutcome::FailedBeforeInvocation(_));
            let terminal_error = match &outcome {
                ToolExecutionOutcome::Failed(error)
                | ToolExecutionOutcome::FailedBeforeInvocation(error) => Some(error.clone()),
                _ => None,
            };
            let mut resolution =
                map_outcome_to_resolution(Some(task_id_for_future.clone()), exec_request, outcome);
            let completed_result = match &resolution {
                TaskResolution::Item(item) => item.parts.iter().find_map(|part| match part {
                    agentkit_core::Part::ToolResult(result) => Some(result.clone()),
                    _ => None,
                }),
                TaskResolution::Approval(_) => None,
            };

            {
                let mut state = inner.state.lock().await;
                let Some(record) = state.tasks.get_mut(&task_id_for_future) else {
                    return;
                };
                // Cancellation and completion compete for this single transition.
                if !record.running || !Arc::ptr_eq(&record.generation, &generation_for_future) {
                    return;
                }
                record.running = false;
                record.completed = !matches!(&resolution, TaskResolution::Approval(_));
                record.suspended = match &resolution {
                    TaskResolution::Approval(task) => {
                        Some(SuspendedTask::new(&task.tool_request, &task.approval))
                    }
                    _ => None,
                };
                record.snapshot.failure = terminal_error.as_ref().map(ToolError::failure_info);
                if not_started {
                    record
                        .snapshot
                        .metadata
                        .insert(TOOL_RESULT_NOT_STARTED_METADATA_KEY.into(), true.into());
                }
                let frozen = record.observations.seal();
                record.snapshot.failure_observations =
                    if terminal_error.is_some() || record.suspended.is_some() {
                        frozen
                    } else {
                        None
                    };
                attach_observations(
                    &mut resolution,
                    record.snapshot.failure_observations.as_ref(),
                );
                let snapshot = record.snapshot.clone();
                let continue_policy = record.continue_policy;
                let delivery_mode = record.delivery_mode;
                let current_kind = snapshot.kind;
                let session_id = record.session_id.clone();

                if current_kind == TaskKind::Foreground {
                    if let Some(count) = state.per_turn_running.get_mut(&turn_id) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            state.per_turn_running.remove(&turn_id);
                        }
                    }
                    state
                        .per_turn_updates
                        .entry(turn_id.clone())
                        .or_default()
                        .push_back(InterruptedUpdate {
                            task_id: task_id_for_future.clone(),
                            session_id,
                            update: TurnTaskUpdate::Resolution(Box::new(resolution.clone())),
                        });
                } else {
                    match &resolution {
                        TaskResolution::Item(_) if delivery_mode == DeliveryMode::ToLoop => {
                            state
                                .pending_loop_updates
                                .push_back((task_id_for_future.clone(), resolution.clone()));
                        }
                        TaskResolution::Approval(_) if delivery_mode == DeliveryMode::ToLoop => {
                            state
                                .pending_loop_updates
                                .push_back((task_id_for_future.clone(), resolution.clone()));
                        }
                        TaskResolution::Item(item) => {
                            state.manual_ready_items.push(item.clone());
                        }
                        TaskResolution::Approval(_) => {}
                    }
                }

                // Enqueue immutable lifecycle events in the same transaction;
                // no observer callbacks run while the task lock is held.
                if let Some(error) = terminal_error {
                    let event = if error.is_cancelled() {
                        TaskEvent::Cancelled(snapshot.clone())
                    } else {
                        TaskEvent::Failed(snapshot.clone(), error)
                    };
                    let _ = event_tx.send(event);
                } else if let Some(result) = completed_result {
                    let _ = event_tx.send(TaskEvent::Completed(snapshot.clone(), result));
                }
                if current_kind == TaskKind::Background
                    && delivery_mode == DeliveryMode::ToLoop
                    && continue_policy == ContinuePolicy::RequestContinue
                {
                    let _ = event_tx.send(TaskEvent::ContinueRequested);
                }
            }
            inner.notify.notify_waiters();
        });

        // Still holding the registration lock: dropping start_task at an await
        // cannot strand a registered task without its worker handle.
        let mut join = Some(join);
        if let Some(record) = state.tasks.get_mut(&task_id)
            && record.running
            && Arc::ptr_eq(&record.generation, &generation)
        {
            record.join = join.take();
        }
        drop(state);
        if let Some(join) = join {
            join.abort();
        } else {
            let _ = start_tx.send(());
        }
        Ok(TaskStartOutcome::Pending {
            task_id,
            kind: initial_kind,
        })
    }

    async fn wait_for_turn(
        &self,
        turn_id: &TurnId,
        cancellation: Option<TurnCancellation>,
    ) -> Result<Option<TurnTaskUpdate>, TaskManagerError> {
        loop {
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut state = self.inner.state.lock().await;
                if let Some(queue) = state.per_turn_updates.get_mut(turn_id)
                    && let Some(update) = queue.pop_front()
                {
                    return Ok(Some(update.update));
                }
                if state
                    .per_turn_running
                    .get(turn_id)
                    .copied()
                    .unwrap_or_default()
                    == 0
                {
                    return Ok(None);
                }
            }
            if cancellation
                .as_ref()
                .is_some_and(TurnCancellation::is_cancelled)
            {
                self.inner.interrupt_turn(turn_id).await;
                continue;
            }
            if let Some(cancellation) = cancellation.as_ref() {
                // Prefer an already-queued detach. If cancellation wins, interrupting
                // under the task-state lock makes it race atomically with detachment.
                tokio::select! {
                    biased;
                    _ = &mut notified => {}
                    _ = cancellation.cancelled() => {
                        self.inner.interrupt_turn(turn_id).await;
                    },
                }
            } else {
                notified.await;
            }
        }
    }

    async fn take_pending_loop_updates(&self) -> Result<PendingLoopUpdates, TaskManagerError> {
        let mut state = self.inner.state.lock().await;
        Ok(PendingLoopUpdates {
            resolutions: state
                .pending_loop_updates
                .drain(..)
                .map(|(_, resolution)| resolution)
                .collect(),
        })
    }

    async fn wait_for_loop_update(&self) -> Result<(), TaskManagerError> {
        loop {
            // Register before checking state so an update cannot land between
            // the empty check and awaiting the notification.
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let state = self.inner.state.lock().await;
                if !state.pending_loop_updates.is_empty() {
                    return Ok(());
                }
            }
            notified.await;
        }
    }

    async fn on_turn_interrupted(&self, turn_id: &TurnId) -> Result<(), TaskManagerError> {
        self.inner.interrupt_turn(turn_id).await;
        Ok(())
    }

    fn take_interrupted_task_updates(
        &self,
        session_id: &agentkit_core::SessionId,
        call_ids: &[ToolCallId],
    ) -> Vec<TurnTaskUpdate> {
        drain_interrupted_updates(&self.inner.interrupted_updates, session_id, call_ids)
    }

    async fn close_suspended_task(
        &self,
        task_id: &TaskId,
        approval_id: &agentkit_core::ApprovalId,
        error: ToolError,
    ) -> Result<Option<TaskResolution>, TaskManagerError> {
        let mut state = self.inner.state.lock().await;
        let record = state
            .tasks
            .get_mut(task_id)
            .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
        if record.completed {
            let turn_id = record.snapshot.turn_id.clone();
            if let Some(queue) = state.per_turn_updates.get_mut(&turn_id)
                && let Some(index) = queue.iter().position(|entry| {
                    &entry.task_id == task_id
                        && matches!(&entry.update, TurnTaskUpdate::Resolution(_))
                })
                && let Some(entry) = queue.remove(index)
                && let TurnTaskUpdate::Resolution(resolution) = entry.update
            {
                return Ok(Some(*resolution));
            }
            if let Some(index) = state
                .pending_loop_updates
                .iter()
                .position(|(id, _)| id == task_id)
            {
                return Ok(state
                    .pending_loop_updates
                    .remove(index)
                    .map(|(_, resolution)| resolution));
            }
            return Ok(take_interrupted_resolution(
                &self.inner.interrupted_updates,
                task_id,
            ));
        }
        let result = close_suspended_record(record, approval_id, &error)?;
        if result.is_some() {
            let snapshot = record.snapshot.clone();
            if let Some(queue) = state.per_turn_updates.get_mut(&snapshot.turn_id) {
                queue.retain(|entry| !matches!(&entry.update, TurnTaskUpdate::Resolution(resolution) if matches!(resolution.as_ref(), TaskResolution::Approval(task) if &task.task_id == task_id)));
            }
            state.pending_loop_updates.retain(|(_, resolution)| !matches!(resolution, TaskResolution::Approval(task) if &task.task_id == task_id));
            let _ = self
                .inner
                .host_event_tx
                .send(failure_event(snapshot, error));
            self.inner.notify.notify_waiters();
        }
        Ok(result)
    }

    fn handle(&self) -> TaskManagerHandle {
        TaskManagerHandle {
            inner: self.inner.clone(),
        }
    }
}

#[async_trait]
impl TaskManagerControl for AsyncInner {
    async fn next_event(&self) -> Option<TaskEvent> {
        self.host_event_rx.lock().await.recv().await
    }

    async fn cancel(&self, task_id: TaskId) -> Result<(), TaskManagerError> {
        let mut state = self.state.lock().await;
        let record = state
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
        if !record.running && record.suspended.is_none() {
            return Ok(());
        }
        let session_id = record.session_id.clone();
        let was_running = record.running;
        record.suspended = None;
        let join = record.join.take();
        record.running = false;
        if !record.invocation_admitted {
            record
                .snapshot
                .metadata
                .insert(TOOL_RESULT_NOT_STARTED_METADATA_KEY.into(), true.into());
        }
        record.completed = true;
        record.snapshot.failure = Some(ToolError::Cancelled.failure_info());
        record.snapshot.failure_observations = record.observations.seal();
        let snapshot = record.snapshot.clone();
        let delivery_mode = record.delivery_mode;
        let continue_policy = record.continue_policy;
        if was_running
            && record.snapshot.kind == TaskKind::Foreground
            && let Some(count) = state.per_turn_running.get_mut(&snapshot.turn_id)
        {
            *count = count.saturating_sub(1);
            if *count == 0 {
                state.per_turn_running.remove(&snapshot.turn_id);
            }
        }
        if let Some(queue) = state.per_turn_updates.get_mut(&snapshot.turn_id) {
            queue.retain(|entry| !matches!(&entry.update, TurnTaskUpdate::Resolution(resolution) if matches!(resolution.as_ref(), TaskResolution::Approval(task) if task.task_id == task_id)));
        }
        state.pending_loop_updates.retain(|(_, resolution)| !matches!(resolution, TaskResolution::Approval(task) if task.task_id == task_id));
        let resolution = cancellation_resolution(&snapshot);
        if snapshot.kind == TaskKind::Foreground {
            state
                .per_turn_updates
                .entry(snapshot.turn_id.clone())
                .or_default()
                .push_back(InterruptedUpdate {
                    task_id: task_id.clone(),
                    session_id,
                    update: TurnTaskUpdate::Resolution(Box::new(resolution)),
                });
        } else if delivery_mode == DeliveryMode::ToLoop {
            state
                .pending_loop_updates
                .push_back((task_id.clone(), resolution));
        } else if let TaskResolution::Item(item) = resolution {
            state.manual_ready_items.push(item);
        }
        let request_continue = snapshot.kind == TaskKind::Background
            && delivery_mode == DeliveryMode::ToLoop
            && continue_policy == ContinuePolicy::RequestContinue;
        let _ = self.host_event_tx.send(TaskEvent::Cancelled(snapshot));
        if request_continue {
            let _ = self.host_event_tx.send(TaskEvent::ContinueRequested);
        }
        drop(state);
        if let Some(join) = join {
            join.abort();
        }
        self.notify.notify_waiters();
        Ok(())
    }

    async fn detach(&self, task_id: TaskId) -> Result<(), TaskManagerError> {
        self.detach_running_foreground(&task_id, None).await
    }

    async fn list_running(&self) -> Vec<TaskSnapshot> {
        let state = self.state.lock().await;
        state
            .tasks
            .values()
            .filter(|record| record.running)
            .map(|record| record.snapshot.clone())
            .collect()
    }

    async fn list_completed(&self) -> Vec<TaskSnapshot> {
        let state = self.state.lock().await;
        state
            .tasks
            .values()
            .filter(|record| record.completed)
            .map(|record| record.snapshot.clone())
            .collect()
    }

    async fn list_suspended(&self) -> Vec<TaskSnapshot> {
        self.state
            .lock()
            .await
            .tasks
            .values()
            .filter(|record| record.suspended.is_some())
            .map(|record| record.snapshot.clone())
            .collect()
    }

    async fn drain_ready_items(&self) -> Vec<Item> {
        let mut state = self.state.lock().await;
        std::mem::take(&mut state.manual_ready_items)
    }

    async fn set_continue_policy(
        &self,
        task_id: TaskId,
        policy: ContinuePolicy,
    ) -> Result<(), TaskManagerError> {
        let mut state = self.state.lock().await;
        let record = state
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
        record.continue_policy = policy;
        Ok(())
    }

    async fn set_delivery_mode(
        &self,
        task_id: TaskId,
        mode: DeliveryMode,
    ) -> Result<(), TaskManagerError> {
        let mut state = self.state.lock().await;
        let record = state
            .tasks
            .get_mut(&task_id)
            .ok_or_else(|| TaskManagerError::NotFound(task_id.clone()))?;
        record.delivery_mode = mode;
        Ok(())
    }

    async fn wait_for_idle(&self) {
        loop {
            {
                let state = self.state.lock().await;
                if !state.tasks.values().any(|r| r.running) {
                    return;
                }
            }
            self.notify.notified().await;
        }
    }
}

fn close_suspended_record(
    record: &mut TaskRecord,
    approval_id: &agentkit_core::ApprovalId,
    error: &ToolError,
) -> Result<Option<TaskResolution>, TaskManagerError> {
    let Some(saved) = &record.suspended else {
        return Ok(None);
    };
    if &saved.approval_id != approval_id {
        return Err(TaskManagerError::InvalidContinuation(
            record.snapshot.id.clone(),
        ));
    }
    record.suspended = None;
    record.completed = true;
    record.snapshot.failure = Some(error.failure_info());
    record.snapshot.failure_observations = record.observations.seal();
    Ok(Some(failure_resolution(&record.snapshot, error.clone())))
}

fn failure_event(snapshot: TaskSnapshot, error: ToolError) -> TaskEvent {
    if error.is_cancelled() {
        TaskEvent::Cancelled(snapshot)
    } else {
        TaskEvent::Failed(snapshot, error)
    }
}

fn cancellation_resolution(snapshot: &TaskSnapshot) -> TaskResolution {
    failure_resolution(snapshot, ToolError::Cancelled)
}

fn failure_resolution(snapshot: &TaskSnapshot, error: ToolError) -> TaskResolution {
    let request = ToolRequest {
        call_id: snapshot.call_id.clone(),
        tool_name: snapshot.tool_name.as_str().into(),
        input: serde_json::Value::Null,
        session_id: "".into(),
        turn_id: snapshot.turn_id.clone(),
        metadata: snapshot.metadata.clone(),
    };
    let outcome = if snapshot
        .metadata
        .get(TOOL_RESULT_NOT_STARTED_METADATA_KEY)
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        ToolExecutionOutcome::FailedBeforeInvocation(error)
    } else {
        ToolExecutionOutcome::Failed(error)
    };
    let mut resolution = map_outcome_to_resolution(Some(snapshot.id.clone()), request, outcome);
    attach_observations(&mut resolution, snapshot.failure_observations.as_ref());
    resolution
}

fn map_outcome_to_resolution(
    task_id: Option<TaskId>,
    mut request: ToolRequest,
    outcome: ToolExecutionOutcome,
) -> TaskResolution {
    strip_failure_metadata(&mut request.metadata);
    let not_started = matches!(&outcome, ToolExecutionOutcome::FailedBeforeInvocation(_));
    match outcome {
        ToolExecutionOutcome::Completed(mut result) => {
            strip_failure_metadata(&mut result.result.metadata);
            strip_failure_metadata(&mut result.metadata);
            TaskResolution::Item(Item {
                id: None,
                kind: agentkit_core::ItemKind::Tool,
                parts: vec![agentkit_core::Part::ToolResult(result.result)],
                metadata: result.metadata,
                usage: None,
                finish_reason: None,
                created_at: None,
            })
        }
        ToolExecutionOutcome::Interrupted(
            agentkit_tools_core::ToolInterruption::ApprovalRequired(mut approval),
        ) => {
            let task_id = task_id.unwrap_or_default();
            approval.task_id = Some(task_id.clone());
            TaskResolution::Approval(TaskApproval {
                task_id,
                tool_request: request,
                approval,
            })
        }
        ToolExecutionOutcome::FailedBeforeInvocation(error)
        | ToolExecutionOutcome::Failed(error) => {
            let mut metadata = request.metadata;
            write_failure_metadata(&mut metadata, &error, not_started);
            TaskResolution::Item(Item {
                id: None,
                kind: agentkit_core::ItemKind::Tool,
                parts: vec![agentkit_core::Part::ToolResult(ToolResultPart {
                    call_id: request.call_id,
                    output: agentkit_core::ToolOutput::Text(error.to_string()),
                    is_error: true,
                    metadata,
                })],
                metadata: MetadataMap::new(),
                usage: None,
                finish_reason: None,
                created_at: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    mod failure_tests;
    use std::collections::BTreeMap;
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    use agentkit_core::{
        CancellationController, ItemKind, Part, SessionId, ToolOutput, TurnCancellation,
    };
    use agentkit_tools_core::{
        ApprovalReason, PermissionChecker, PermissionDecision, ToolAnnotations, ToolInterruption,
        ToolName, ToolResult, ToolSpec,
    };
    use serde_json::json;
    use tokio::sync::Notify;
    use tokio::time::{Duration, timeout};

    use super::*;

    struct AllowAllPermissions;

    impl PermissionChecker for AllowAllPermissions {
        fn evaluate(
            &self,
            _request: &dyn agentkit_tools_core::PermissionRequest,
        ) -> PermissionDecision {
            PermissionDecision::Allow
        }
    }

    #[derive(Clone)]
    enum TestBehavior {
        ObserveBlock {
            entered: StdArc<AtomicBool>,
            release: StdArc<Notify>,
            publisher:
                StdArc<std::sync::Mutex<Option<agentkit_tools_core::FailureObservationPublisher>>>,
            outcome: ToolExecutionOutcome,
        },
        Block {
            entered: StdArc<AtomicBool>,
            release: StdArc<Notify>,
            output: &'static str,
        },
        Approval,
    }

    #[derive(Clone)]
    struct TestExecutor {
        behaviors: BTreeMap<String, TestBehavior>,
    }

    impl TestExecutor {
        fn new(behaviors: impl IntoIterator<Item = (impl Into<String>, TestBehavior)>) -> Self {
            Self {
                behaviors: behaviors
                    .into_iter()
                    .map(|(name, behavior)| (name.into(), behavior))
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl ToolExecutor for TestExecutor {
        fn specs(&self) -> Vec<ToolSpec> {
            self.behaviors
                .keys()
                .map(|name| ToolSpec {
                    name: ToolName::new(name),
                    description: format!("test tool {name}"),
                    input_schema: json!({
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }),
                    output_schema: None,
                    annotations: ToolAnnotations::default(),
                    metadata: MetadataMap::new(),
                })
                .collect()
        }

        async fn execute(
            &self,
            request: ToolRequest,
            _ctx: &mut agentkit_tools_core::ToolContext<'_>,
        ) -> ToolExecutionOutcome {
            match self.behaviors.get(request.tool_name.0.as_str()) {
                Some(TestBehavior::ObserveBlock {
                    entered,
                    release,
                    publisher,
                    outcome,
                }) => {
                    let observer = _ctx
                        .failure_observer()
                        .expect("manager installed publisher");
                    let mut effects = agentkit_core::failure::PossibleEffects::default();
                    effects.source = agentkit_core::failure::ObservationSource::LocalSession;
                    effects.tool_execution_start_reported = true;
                    observer.publish_effects(effects).unwrap();
                    *publisher.lock().unwrap() = Some(observer);
                    entered.store(true, AtomicOrdering::SeqCst);
                    release.notified().await;
                    outcome.clone()
                }
                Some(TestBehavior::Block {
                    entered,
                    release,
                    output,
                }) => {
                    entered.store(true, AtomicOrdering::SeqCst);
                    release.notified().await;
                    ToolExecutionOutcome::Completed(ToolResult {
                        result: ToolResultPart {
                            call_id: request.call_id,
                            output: ToolOutput::Text((*output).into()),
                            is_error: false,
                            metadata: request.metadata,
                        },
                        duration: None,
                        metadata: MetadataMap::new(),
                    })
                }
                Some(TestBehavior::Approval) => ToolExecutionOutcome::Interrupted(
                    ToolInterruption::ApprovalRequired(ApprovalRequest {
                        task_id: None,
                        call_id: Some(request.call_id.clone()),
                        id: "approval:test".into(),
                        request_kind: "tool.test".into(),
                        reason: ApprovalReason::SensitivePath,
                        summary: "requires approval".into(),
                        metadata: MetadataMap::new(),
                    }),
                ),
                None => ToolExecutionOutcome::Failed(ToolError::Unavailable(
                    request.tool_name.0.clone(),
                )),
            }
        }
    }

    struct NameRoutingPolicy {
        routes: BTreeMap<String, RoutingDecision>,
    }

    impl NameRoutingPolicy {
        fn new(routes: impl IntoIterator<Item = (impl Into<String>, RoutingDecision)>) -> Self {
            Self {
                routes: routes
                    .into_iter()
                    .map(|(name, decision)| (name.into(), decision))
                    .collect(),
            }
        }
    }

    impl TaskRoutingPolicy for NameRoutingPolicy {
        fn route(&self, request: &ToolRequest) -> RoutingDecision {
            self.routes
                .get(request.tool_name.0.as_str())
                .copied()
                .unwrap_or(RoutingDecision::Foreground)
        }
    }

    fn make_request(tool_name: &str, turn_id: &str, call_id: &str) -> ToolRequest {
        ToolRequest {
            call_id: ToolCallId::new(call_id),
            tool_name: ToolName::new(tool_name),
            input: json!({}),
            session_id: SessionId::new("session-1"),
            turn_id: TurnId::new(turn_id),
            metadata: MetadataMap::new(),
        }
    }

    fn make_context(
        executor: Arc<dyn ToolExecutor>,
        turn_id: &TurnId,
        cancellation: Option<TurnCancellation>,
    ) -> TaskStartContext {
        TaskStartContext {
            executor,
            tool_context: OwnedToolContext {
                failure_observer: None,
                session_id: SessionId::new("session-1"),
                turn_id: turn_id.clone(),
                metadata: MetadataMap::new(),
                permissions: Arc::new(AllowAllPermissions),
                resources: Arc::new(()),
                cancellation,
                execution_scope: None,
                approved_request: None,
            },
        }
    }

    async fn next_event(handle: &TaskManagerHandle) -> TaskEvent {
        timeout(Duration::from_secs(1), handle.next_event())
            .await
            .expect("timed out waiting for task event")
            .expect("task event stream ended unexpectedly")
    }

    async fn wait_until_entered(entered: &AtomicBool) {
        timeout(Duration::from_secs(1), async {
            while !entered.load(AtomicOrdering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("task never entered execution");
    }

    #[tokio::test]
    async fn simple_task_manager_executes_inline_and_assigns_task_ids() {
        let manager = SimpleTaskManager::new();
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "needs-approval",
            TestBehavior::Approval,
        )]));
        let request = make_request("needs-approval", "turn-1", "call-1");

        let outcome = manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap();

        match outcome {
            TaskStartOutcome::Ready(resolution) => match *resolution {
                TaskResolution::Approval(task) => {
                    assert!(!task.task_id.0.is_empty());
                    assert_eq!(task.approval.task_id.as_ref(), Some(&task.task_id));
                    assert_eq!(task.tool_request.call_id, request.call_id);
                }
                other => panic!("unexpected task resolution: {other:?}"),
            },
            other => panic!("unexpected start outcome: {other:?}"),
        }

        assert!(manager.handle().list_running().await.is_empty());
    }

    #[tokio::test]
    async fn async_manager_interrupt_cancels_foreground_only() {
        let fg_release = StdArc::new(Notify::new());
        let fg_entered = StdArc::new(AtomicBool::new(false));
        let bg_release = StdArc::new(Notify::new());
        let bg_entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([
            (
                "foreground",
                TestBehavior::Block {
                    entered: fg_entered.clone(),
                    release: fg_release.clone(),
                    output: "foreground-done",
                },
            ),
            (
                "background",
                TestBehavior::Block {
                    entered: bg_entered.clone(),
                    release: bg_release.clone(),
                    output: "background-done",
                },
            ),
        ]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([
            ("foreground", RoutingDecision::Foreground),
            ("background", RoutingDecision::Background),
        ]));
        let handle = manager.handle();
        let turn_id = TurnId::new("turn-1");

        let foreground = manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: make_request("foreground", "turn-1", "call-fg"),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor.clone(), &turn_id, None),
            )
            .await
            .unwrap();
        let background = manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: make_request("background", "turn-1", "call-bg"),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor.clone(), &turn_id, None),
            )
            .await
            .unwrap();

        assert!(matches!(
            foreground,
            TaskStartOutcome::Pending {
                kind: TaskKind::Foreground,
                ..
            }
        ));
        let background_id = match background {
            TaskStartOutcome::Pending {
                task_id,
                kind: TaskKind::Background,
            } => task_id,
            other => panic!("unexpected background outcome: {other:?}"),
        };

        let _ = next_event(&handle).await;
        let _ = next_event(&handle).await;
        wait_until_entered(fg_entered.as_ref()).await;
        wait_until_entered(bg_entered.as_ref()).await;

        manager.on_turn_interrupted(&turn_id).await.unwrap();

        match next_event(&handle).await {
            TaskEvent::Cancelled(snapshot) => assert_eq!(snapshot.tool_name, "foreground"),
            other => panic!("unexpected event after interrupt: {other:?}"),
        }

        let running = handle.list_running().await;
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].id, background_id);
        assert_eq!(running[0].tool_name, "background");

        bg_release.notify_waiters();
        match next_event(&handle).await {
            TaskEvent::Completed(snapshot, result) => {
                assert_eq!(snapshot.id, background_id);
                assert_eq!(result.output, ToolOutput::Text("background-done".into()));
            }
            other => panic!("unexpected completion event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn async_manager_can_manually_detach_a_foreground_task() {
        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "foreground",
            TestBehavior::Block {
                entered: entered.clone(),
                release: release.clone(),
                output: "done",
            },
        )]));
        let manager = AsyncTaskManager::new();
        let handle = manager.handle();
        let request = make_request("foreground", "turn-1", "call-1");

        let task_id = match manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap()
        {
            TaskStartOutcome::Pending { task_id, .. } => task_id,
            other => panic!("unexpected start outcome: {other:?}"),
        };

        let _ = next_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        handle.detach(task_id.clone()).await.unwrap();

        let running = handle.list_running().await;
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].kind, TaskKind::Background);
        match manager.wait_for_turn(&request.turn_id, None).await.unwrap() {
            Some(TurnTaskUpdate::Detached(snapshot)) => {
                assert_eq!(snapshot.id, task_id);
                assert_eq!(snapshot.kind, TaskKind::Background);
            }
            other => panic!("unexpected turn update: {other:?}"),
        }
        assert!(
            manager
                .wait_for_turn(&request.turn_id, None)
                .await
                .unwrap()
                .is_none()
        );
        match next_event(&handle).await {
            TaskEvent::Detached(snapshot) => assert_eq!(snapshot.id, task_id),
            other => panic!("unexpected event after detach: {other:?}"),
        }

        release.notify_waiters();
        timeout(Duration::from_secs(1), handle.wait_for_idle())
            .await
            .expect("wait_for_idle timed out");
    }

    #[tokio::test]
    async fn manual_detach_reports_invalid_task_states() {
        let missing = TaskId::new("missing");
        let manager = AsyncTaskManager::new();
        let handle = manager.handle();
        assert_eq!(
            handle.detach(missing.clone()).await,
            Err(TaskManagerError::NotFound(missing))
        );

        let approval_executor: Arc<dyn ToolExecutor> =
            Arc::new(TestExecutor::new([("approval", TestBehavior::Approval)]));
        let approval_request = make_request("approval", "turn-1", "call-approval");
        let completed_id = match manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: approval_request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(approval_executor, &approval_request.turn_id, None),
            )
            .await
            .unwrap()
        {
            TaskStartOutcome::Pending { task_id, .. } => task_id,
            other => panic!("unexpected start outcome: {other:?}"),
        };
        let _ = next_event(&handle).await;
        timeout(Duration::from_secs(1), handle.wait_for_idle())
            .await
            .expect("wait_for_idle timed out");
        assert_eq!(
            handle.detach(completed_id.clone()).await,
            Err(TaskManagerError::NotRunning(completed_id))
        );

        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let background_executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "background",
            TestBehavior::Block {
                entered: entered.clone(),
                release,
                output: "done",
            },
        )]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background",
            RoutingDecision::Background,
        )]));
        let handle = manager.handle();
        let request = make_request("background", "turn-2", "call-background");
        let background_id = match manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(background_executor, &request.turn_id, None),
            )
            .await
            .unwrap()
        {
            TaskStartOutcome::Pending { task_id, .. } => task_id,
            other => panic!("unexpected start outcome: {other:?}"),
        };
        let _ = next_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        assert_eq!(
            handle.detach(background_id.clone()).await,
            Err(TaskManagerError::AlreadyBackground(background_id.clone()))
        );
        handle.cancel(background_id).await.unwrap();
    }

    #[tokio::test]
    async fn async_manager_can_cancel_background_tasks_by_id() {
        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "background",
            TestBehavior::Block {
                entered: entered.clone(),
                release,
                output: "done",
            },
        )]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background",
            RoutingDecision::Background,
        )]));
        let handle = manager.handle();
        let request = make_request("background", "turn-1", "call-1");

        let task_id = match manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap()
        {
            TaskStartOutcome::Pending { task_id, .. } => task_id,
            other => panic!("unexpected start outcome: {other:?}"),
        };

        let _ = next_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        handle.cancel(task_id.clone()).await.unwrap();

        match next_event(&handle).await {
            TaskEvent::Cancelled(snapshot) => assert_eq!(snapshot.id, task_id),
            other => panic!("unexpected event after cancel: {other:?}"),
        }

        assert!(handle.list_running().await.is_empty());
    }

    #[tokio::test]
    async fn async_manager_manual_delivery_keeps_results_out_of_loop_updates() {
        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "background",
            TestBehavior::Block {
                entered: entered.clone(),
                release: release.clone(),
                output: "manual-done",
            },
        )]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background",
            RoutingDecision::Background,
        )]));
        let handle = manager.handle();
        let request = make_request("background", "turn-1", "call-1");

        let task_id = match manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap()
        {
            TaskStartOutcome::Pending { task_id, .. } => task_id,
            other => panic!("unexpected start outcome: {other:?}"),
        };

        let _ = next_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        handle
            .set_continue_policy(task_id.clone(), ContinuePolicy::RequestContinue)
            .await
            .unwrap();
        handle
            .set_delivery_mode(task_id, DeliveryMode::Manual)
            .await
            .unwrap();

        release.notify_waiters();
        match next_event(&handle).await {
            TaskEvent::Completed(_, result) => {
                assert_eq!(result.output, ToolOutput::Text("manual-done".into()))
            }
            other => panic!("unexpected event: {other:?}"),
        }

        assert!(
            timeout(Duration::from_millis(50), handle.next_event())
                .await
                .is_err()
        );
        assert!(
            manager
                .take_pending_loop_updates()
                .await
                .unwrap()
                .resolutions
                .is_empty()
        );

        let ready_items = handle.drain_ready_items().await;
        assert_eq!(ready_items.len(), 1);
        assert_eq!(ready_items[0].kind, ItemKind::Tool);
        match &ready_items[0].parts[0] {
            Part::ToolResult(result) => {
                assert_eq!(result.output, ToolOutput::Text("manual-done".into()))
            }
            other => panic!("unexpected ready item: {other:?}"),
        }
    }

    #[tokio::test]
    async fn async_manager_to_loop_delivery_can_request_continue() {
        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "background",
            TestBehavior::Block {
                entered: entered.clone(),
                release: release.clone(),
                output: "loop-done",
            },
        )]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background",
            RoutingDecision::Background,
        )]));
        let handle = manager.handle();
        let request = make_request("background", "turn-1", "call-1");

        let task_id = match manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(
                    executor,
                    &request.turn_id,
                    Some(TurnCancellation::new(
                        CancellationController::new().handle(),
                    )),
                ),
            )
            .await
            .unwrap()
        {
            TaskStartOutcome::Pending { task_id, .. } => task_id,
            other => panic!("unexpected start outcome: {other:?}"),
        };

        let _ = next_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        handle
            .set_continue_policy(task_id, ContinuePolicy::RequestContinue)
            .await
            .unwrap();

        release.notify_waiters();
        match next_event(&handle).await {
            TaskEvent::Completed(_, result) => {
                assert_eq!(result.output, ToolOutput::Text("loop-done".into()))
            }
            other => panic!("unexpected completion event: {other:?}"),
        }
        match next_event(&handle).await {
            TaskEvent::ContinueRequested => {}
            other => panic!("unexpected follow-up event: {other:?}"),
        }

        let updates = manager.take_pending_loop_updates().await.unwrap();
        assert_eq!(updates.resolutions.len(), 1);
        assert!(handle.drain_ready_items().await.is_empty());
    }

    #[tokio::test]
    async fn wait_for_loop_update_wakes_without_consuming_host_events() {
        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "background",
            TestBehavior::Block {
                entered: entered.clone(),
                release: release.clone(),
                output: "wake-done",
            },
        )]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background",
            RoutingDecision::Background,
        )]));
        let handle = manager.handle();
        let request = make_request("background", "turn-1", "call-1");

        manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap();
        wait_until_entered(entered.as_ref()).await;

        let waiting = manager.wait_for_loop_update();
        tokio::pin!(waiting);
        assert!(
            timeout(Duration::from_millis(20), &mut waiting)
                .await
                .is_err()
        );
        release.notify_waiters();
        timeout(Duration::from_secs(1), &mut waiting)
            .await
            .expect("loop update wake timed out")
            .unwrap();

        assert!(matches!(next_event(&handle).await, TaskEvent::Started(_)));
        assert!(matches!(
            next_event(&handle).await,
            TaskEvent::Completed(_, _)
        ));
        assert_eq!(
            manager
                .take_pending_loop_updates()
                .await
                .unwrap()
                .resolutions
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn wait_for_idle_returns_after_loop_updates_are_queued() {
        let release = StdArc::new(Notify::new());
        let entered = StdArc::new(AtomicBool::new(false));
        let executor: Arc<dyn ToolExecutor> = Arc::new(TestExecutor::new([(
            "background",
            TestBehavior::Block {
                entered: entered.clone(),
                release: release.clone(),
                output: "idle-done",
            },
        )]));
        let manager = AsyncTaskManager::new().routing(NameRoutingPolicy::new([(
            "background",
            RoutingDecision::Background,
        )]));
        let handle = manager.handle();
        let request = make_request("background", "turn-1", "call-1");

        let outcome = manager
            .start_task(
                TaskLaunchRequest {
                    task_id: None,
                    request: request.clone(),
                    kind: TaskLaunchKind::Plain,
                },
                make_context(executor, &request.turn_id, None),
            )
            .await
            .unwrap();
        assert!(matches!(outcome, TaskStartOutcome::Pending { .. }));

        let _ = next_event(&handle).await;
        wait_until_entered(entered.as_ref()).await;
        release.notify_waiters();

        timeout(Duration::from_secs(1), handle.wait_for_idle())
            .await
            .expect("wait_for_idle timed out");

        let updates = manager.take_pending_loop_updates().await.unwrap();
        assert_eq!(updates.resolutions.len(), 1);
        match &updates.resolutions[0] {
            TaskResolution::Item(item) => match &item.parts[0] {
                Part::ToolResult(result) => {
                    assert_eq!(result.call_id, request.call_id);
                    assert_eq!(result.output, ToolOutput::Text("idle-done".into()));
                }
                other => panic!("unexpected tool item: {other:?}"),
            },
            other => panic!("unexpected pending update: {other:?}"),
        }
    }
}
