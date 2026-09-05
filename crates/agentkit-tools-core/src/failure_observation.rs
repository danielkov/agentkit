//! Invocation-scoped latest-value observations. No queue, globals, or metadata trust.
use agentkit_core::failure::{FailureObservations, HostFatalReceipt, PossibleEffects};
use agentkit_core::retry::ProviderFailure;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationUpdate {
    Published,
    Unchanged,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ObservationPublishError {
    #[error("failure observations are sealed")]
    Sealed,
    #[error("invalid failure observation")]
    Invalid,
    #[error("conflicting failure observation")]
    Conflict,
    #[error("regressive failure observation")]
    Regression,
}
#[derive(Default)]
struct State {
    sealed: bool,
    value: FailureObservations,
}

/// Host controller. Create one per invocation; sealing freezes all producer clones.
#[derive(Clone, Default)]
pub struct FailureObservationSlot(Arc<Mutex<State>>);
/// Trusted host producer handle, not serializable and never accepted from metadata.
#[derive(Clone)]
pub struct FailureObservationPublisher(Arc<Mutex<State>>);

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    // No user callbacks execute under this lock. If a host unwinds, keep the last
    // accepted value rather than silently losing observations on cancellation.
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
fn snapshot(state: &State) -> Option<FailureObservations> {
    (!state.value.is_empty()).then(|| state.value.clone())
}
impl FailureObservationSlot {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn publisher(&self) -> FailureObservationPublisher {
        FailureObservationPublisher(self.0.clone())
    }
    pub fn snapshot(&self) -> Option<FailureObservations> {
        snapshot(&lock(&self.0))
    }
    pub fn seal(&self) -> Option<FailureObservations> {
        let mut state = lock(&self.0);
        state.sealed = true;
        snapshot(&state)
    }
}
impl FailureObservationPublisher {
    pub fn publish_effects(
        &self,
        value: PossibleEffects,
    ) -> Result<ObservationUpdate, ObservationPublishError> {
        if !value.observation_incomplete() {
            return Err(ObservationPublishError::Invalid);
        }
        let mut state = lock(&self.0);
        if state.sealed {
            return Err(ObservationPublishError::Sealed);
        }
        if let Some(old) = state.value.effects() {
            if old == &value {
                return Ok(ObservationUpdate::Unchanged);
            }
            if old.source != value.source {
                return Err(ObservationPublishError::Conflict);
            }
            if (old.assistant_output_observed && !value.assistant_output_observed)
                || (old.tool_emission_observed && !value.tool_emission_observed)
                || (old.tool_execution_start_reported && !value.tool_execution_start_reported)
                || (old.tool_execution_completion_reported
                    && !value.tool_execution_completion_reported)
            {
                return Err(ObservationPublishError::Regression);
            }
        }
        state.value = state.value.clone().with_effects(value);
        Ok(ObservationUpdate::Published)
    }
    pub fn publish_receipt(
        &self,
        value: HostFatalReceipt,
    ) -> Result<ObservationUpdate, ObservationPublishError> {
        let mut state = lock(&self.0);
        if state.sealed {
            return Err(ObservationPublishError::Sealed);
        }
        if let Some(old) = state.value.receipt() {
            return if old == &value {
                Ok(ObservationUpdate::Unchanged)
            } else {
                Err(ObservationPublishError::Conflict)
            };
        }
        state.value = state.value.clone().with_receipt(value);
        Ok(ObservationUpdate::Published)
    }
    pub fn publish_retry(
        &self,
        value: ProviderFailure,
    ) -> Result<ObservationUpdate, ObservationPublishError> {
        FailureObservations::default()
            .with_retry(value)
            .map_err(|_| ObservationPublishError::Invalid)?;
        let mut state = lock(&self.0);
        if state.sealed {
            return Err(ObservationPublishError::Sealed);
        }
        if let Some(old) = state.value.retry() {
            return if old == &value {
                Ok(ObservationUpdate::Unchanged)
            } else {
                Err(ObservationPublishError::Conflict)
            };
        }
        state.value = state
            .value
            .clone()
            .with_retry(value)
            .map_err(|_| ObservationPublishError::Invalid)?;
        Ok(ObservationUpdate::Published)
    }
}
