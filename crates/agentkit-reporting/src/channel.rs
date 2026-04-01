//! Channel-based reporter adapter.
//!
//! [`ChannelReporter`] forwards events to another thread or task via a
//! [`std::sync::mpsc::Sender`]. This keeps the observer contract synchronous
//! while allowing expensive event processing to happen off the driver's hot
//! path.

use std::sync::mpsc::{self, Receiver, Sender};

use agentkit_loop::{AgentEvent, LoopObserver};

use crate::ReportError;
use crate::policy::FallibleObserver;

/// Reporter adapter that forwards events over a channel.
///
/// Use this when you need expensive or async event processing without
/// blocking the agent loop. The receiving end can live on a dedicated
/// thread or async task.
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::ChannelReporter;
///
/// let (reporter, rx) = ChannelReporter::pair();
///
/// // Spawn a consumer on another thread.
/// std::thread::spawn(move || {
///     while let Ok(event) = rx.recv() {
///         println!("{event:?}");
///     }
/// });
///
/// // `reporter` implements `LoopObserver` — hand it to the agent loop.
/// ```
pub struct ChannelReporter {
    sender: Sender<AgentEvent>,
}

impl ChannelReporter {
    /// Creates a `ChannelReporter` from an existing sender.
    pub fn new(sender: Sender<AgentEvent>) -> Self {
        Self { sender }
    }

    /// Creates a `ChannelReporter` together with the receiving end of the
    /// channel.
    ///
    /// This is a convenience wrapper around [`std::sync::mpsc::channel`].
    pub fn pair() -> (Self, Receiver<AgentEvent>) {
        let (sender, receiver) = mpsc::channel();
        (Self { sender }, receiver)
    }
}

impl LoopObserver for ChannelReporter {
    fn handle_event(&mut self, event: AgentEvent) {
        // Silently drop if the receiver is gone — reporters are non-fatal.
        let _ = self.sender.send(event);
    }
}

impl FallibleObserver for ChannelReporter {
    fn try_handle_event(&mut self, event: &AgentEvent) -> Result<(), ReportError> {
        self.sender
            .send(event.clone())
            .map_err(|_| ReportError::ChannelSend)
    }
}
