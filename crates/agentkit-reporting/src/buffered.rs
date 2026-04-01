//! Batch-buffered reporter adapter.
//!
//! [`BufferedReporter`] enqueues events and forwards them to an inner
//! [`LoopObserver`] in batches — either when the buffer reaches a configured
//! capacity or on an explicit [`flush`](BufferedReporter::flush) call.
//! Remaining events are flushed automatically on drop.

use agentkit_loop::{AgentEvent, LoopObserver};

/// Reporter adapter that enqueues events for batch flushing.
///
/// Wraps any [`LoopObserver`] and delivers events in batches rather than
/// one-at-a-time. This is useful when the inner reporter benefits from
/// amortised work (e.g. a network sink that batches writes).
///
/// # Example
///
/// ```rust
/// use agentkit_reporting::{BufferedReporter, UsageReporter};
///
/// // Buffer up to 64 events before forwarding to the inner reporter.
/// let reporter = BufferedReporter::new(UsageReporter::new(), 64);
/// ```
pub struct BufferedReporter<T: LoopObserver> {
    inner: T,
    buffer: Vec<AgentEvent>,
    capacity: usize,
}

impl<T: LoopObserver> BufferedReporter<T> {
    /// Creates a new `BufferedReporter` that buffers up to `capacity` events
    /// before flushing them to `inner`.
    ///
    /// A capacity of `0` disables automatic flushing — events are only
    /// delivered when [`flush`](BufferedReporter::flush) is called explicitly
    /// (or on drop).
    pub fn new(inner: T, capacity: usize) -> Self {
        Self {
            inner,
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Delivers all buffered events to the inner observer and clears the
    /// buffer.
    pub fn flush(&mut self) {
        for event in self.buffer.drain(..) {
            self.inner.handle_event(event);
        }
    }

    /// Returns the number of events currently buffered.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// Returns a reference to the inner observer.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the inner observer.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T: LoopObserver> LoopObserver for BufferedReporter<T> {
    fn handle_event(&mut self, event: AgentEvent) {
        self.buffer.push(event);
        if self.capacity > 0 && self.buffer.len() >= self.capacity {
            self.flush();
        }
    }
}

impl<T: LoopObserver> Drop for BufferedReporter<T> {
    fn drop(&mut self) {
        self.flush();
    }
}
