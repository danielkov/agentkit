//! Bounded prompt admission for hosts that own their foreground turn actor.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use agent_client_protocol::{Error, RequestCancellation};
use agentkit_core::Item;
use tokio::sync::{Notify, oneshot};

use super::{content_blocks_to_items, wire};

/// Maximum number of pending unstable ACP v2 prompts per session.
pub const MAX_PENDING_PROMPTS: usize = 64;
/// Maximum serialized non-media content bytes retained by pending unstable prompts.
/// Includes text, metadata, URIs, and JSON structure, but excludes inline payloads.
pub const MAX_PENDING_PROMPT_BYTES: usize = 256 * 1024;
/// Maximum serialized inline image, audio, and embedded blob payload bytes per
/// session, including in-flight delivery. 28 MiB accommodates 20 MiB of source
/// media after base64 expansion and padding. Both original content and converted
/// Items retain payloads, so retained media can occupy twice this limit, plus
/// bounded metadata/data-URL overhead. This is not a transport or transcript cap.
pub const MAX_PENDING_PROMPT_MEDIA_BYTES: usize = 28 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PromptBytes {
    content: usize,
    media: usize,
}

impl PromptBytes {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            content: self.content.saturating_add(other.content),
            media: self.media.saturating_add(other.media),
        }
    }

    fn saturating_sub(self, other: Self) -> Self {
        Self {
            content: self.content.saturating_sub(other.content),
            media: self.media.saturating_sub(other.media),
        }
    }

    fn exceeds_budget(self) -> bool {
        self.content > MAX_PENDING_PROMPT_BYTES || self.media > MAX_PENDING_PROMPT_MEDIA_BYTES
    }
}

// Count serialized bytes without allocating a second serialized media buffer.
// Stop once even the combined budgets cannot accommodate the input.
#[derive(Default)]
struct PromptByteCounter(usize);

impl std::io::Write for PromptByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(bytes.len());
        if self.0 > MAX_PENDING_PROMPT_BYTES + MAX_PENDING_PROMPT_MEDIA_BYTES {
            return Err(std::io::Error::other(
                "pending session prompt budget exceeded",
            ));
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn prompt_budget_error() -> agent_client_protocol::Error {
    agent_client_protocol::Error::new(-32602, "pending session prompt budget exceeded")
}

fn measure_prompt(request: &wire::PromptRequest) -> Result<PromptBytes, Error> {
    let mut total = PromptByteCounter::default();
    serde_json::to_writer(&mut total, request).map_err(|_| prompt_budget_error())?;
    let mut media = 0usize;
    for block in &request.prompt {
        let payload: Option<&str> = match block {
            wire::ContentBlock::Image(image) => Some(image.data.as_ref()),
            wire::ContentBlock::Audio(audio) => Some(audio.data.as_ref()),
            wire::ContentBlock::Resource(resource) => match &resource.resource {
                wire::EmbeddedResourceResource::BlobResourceContents(blob) => {
                    Some(blob.blob.as_ref())
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(payload) = payload {
            let mut counter = PromptByteCounter::default();
            serde_json::to_writer(&mut counter, payload).map_err(|_| prompt_budget_error())?;
            // Keep the string quotes in the non-media budget. Counting serialized
            // strings also charges JSON escapes if the payload is malformed.
            media = media.saturating_add(counter.0 - 2);
        }
    }
    let bytes = PromptBytes {
        content: total.0 - media,
        media,
    };
    if bytes.exceeds_budget() {
        return Err(prompt_budget_error());
    }
    Ok(bytes)
}

/// Maximum live JSON-RPC attempts across pending and claimed prompts.
pub const MAX_PENDING_PROMPT_ATTEMPTS: usize = 256;

#[derive(Default)]
struct Queue {
    entries: VecDeque<Entry>,
    groups: HashMap<u64, Group>,
    next_group: u64,
    attempts: usize,
    bytes: PromptBytes,
    closed: bool,
}

struct Group {
    key: Option<wire::IdempotencyKey>,
    claimed: bool,
    attempts: Vec<Attempt>,
}

struct Attempt {
    token: Arc<()>,
    cancellation: Option<RequestCancellation>,
    reply: oneshot::Sender<PromptResult>,
}

impl Group {
    fn cancelled(&self) -> bool {
        self.attempts.iter().any(Attempt::cancelled)
    }

    fn settle(self, result: PromptResult) {
        for attempt in self.attempts {
            let _ = attempt.reply.send(result.clone());
        }
    }
}

impl Attempt {
    fn cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(RequestCancellation::is_cancelled)
    }
}

impl Queue {
    fn withdraw_unclaimed(&mut self, id: u64) -> Option<(Entry, Group)> {
        let index = self.entries.iter().position(|entry| entry.group == id)?;
        let entry = self.entries.remove(index).expect("entry index was found");
        let group = self.remove_group(&entry);
        Some((entry, group))
    }

    fn remove_group(&mut self, entry: &Entry) -> Group {
        let group = self
            .groups
            .remove(&entry.group)
            .expect("entry has a live group");
        self.attempts -= group.attempts.len();
        self.bytes = self.bytes.saturating_sub(entry.bytes);
        group
    }
}

#[derive(Default)]
struct Inner {
    queue: Mutex<Queue>,
    changed: Notify,
}

type PromptResult = Result<wire::PromptResponse, Error>;

struct Entry {
    group: u64,
    request: wire::PromptRequest,
    items: Vec<Item>,
    bytes: PromptBytes,
}

#[derive(Clone, Copy)]
enum Claim {
    One(bool),
    Batch(bool),
}

/// A per-session queue of not-yet-inserted ACP prompts.
///
/// Enqueue in receive order. Steers take precedence, and their relative order
/// and the order of other prompts are FIFO. Requests with the same idempotency
/// key share a live pending or claimed submission. The host manages replay of
/// already inserted keys with its conversation history separately.
#[derive(Clone)]
pub struct PendingPrompts {
    session_id: wire::SessionId,
    inner: Arc<Inner>,
}

impl PendingPrompts {
    /// Creates an empty prompt queue scoped to a session.
    pub fn new(session_id: wire::SessionId) -> Self {
        Self {
            session_id,
            inner: Arc::new(Inner::default()),
        }
    }

    // Each queued or claimed Entry has one Group, keyed by a private ID that
    // is never reused. Only unclaimed entries appear in the FIFO. `bytes`
    // covers all entries, and `attempts` covers all stored reply senders. All
    // writers commit these together under the lock; conversion and allocation
    // checks precede commitment. Cleanup moves owned entries/group senders out
    // before settling replies or constructing RAII claims (which reenter on drop).
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.inner
            .queue
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    }

    /// Retains a prompt until insertion or cancellation. A successful enqueue
    /// does not acknowledge insertion; use [`PendingPromptResponse::wait`].
    pub fn enqueue(
        &self,
        request: wire::PromptRequest,
        cancellation: RequestCancellation,
    ) -> Result<PendingPromptResponse, Error> {
        self.admit_inner(request, Some(cancellation), false, true)
    }

    // Ordinary single-turn headless admission preserves cancellation and
    // idempotency without applying the host delivery queue's byte budget.
    pub(super) fn admit_ordinary(
        &self,
        request: wire::PromptRequest,
        cancellation: RequestCancellation,
    ) -> Result<PendingPromptResponse, Error> {
        self.admit_inner(request, Some(cancellation), false, false)
    }

    // A single-turn host may accept retries while rejecting unrelated work.
    pub(super) fn retry(
        &self,
        request: wire::PromptRequest,
        cancellation: RequestCancellation,
    ) -> Result<PendingPromptResponse, Error> {
        self.admit_inner(request, Some(cancellation), true, false)
    }

    #[cfg(all(test, feature = "unstable-prompt-delivery"))]
    fn admit(
        &self,
        request: wire::PromptRequest,
        cancellation: Option<RequestCancellation>,
    ) -> Result<PendingPromptResponse, Error> {
        self.admit_inner(request, cancellation, false, true)
    }

    fn admit_inner(
        &self,
        request: wire::PromptRequest,
        cancellation: Option<RequestCancellation>,
        retry_only: bool,
        enforce_budget: bool,
    ) -> Result<PendingPromptResponse, Error> {
        if request.session_id != self.session_id || request.prompt.is_empty() {
            return Err(Error::new(
                -32602,
                "invalid pending prompt session or empty content",
            ));
        }
        #[cfg(feature = "unstable-prompt-delivery")]
        if !matches!(
            request.delivery,
            None | Some(wire::PromptDelivery::Steer | wire::PromptDelivery::Queue)
        ) {
            return Err(Error::new(-32602, "unsupported prompt delivery mode"));
        }
        let (reply, response) = oneshot::channel();
        let token = Arc::new(());
        let attempt = Attempt {
            token: token.clone(),
            cancellation: cancellation.clone(),
            reply,
        };
        let group = {
            let mut queue = self.lock();
            if queue.closed {
                return Err(Error::request_cancelled());
            }
            let live = request.idempotency_key.as_ref().and_then(|key| {
                queue
                    .groups
                    .iter()
                    .find_map(|(&id, group)| (group.key.as_ref() == Some(key)).then_some(id))
            });
            if attempt.cancelled() {
                match live {
                    Some(id) if !queue.groups[&id].claimed => {
                        let (_entry, withdrawn) = queue
                            .withdraw_unclaimed(id)
                            .expect("unclaimed group is queued");
                        drop(queue);
                        withdrawn.settle(Err(Error::request_cancelled()));
                        self.inner.changed.notify_waiters();
                        return Err(Error::request_cancelled());
                    }
                    None => return Err(Error::request_cancelled()),
                    // Already claimed, so this attempt still awaits the insertion result.
                    Some(_) => {}
                }
            }
            if retry_only && live.is_none() {
                return Err(crate::sdk_error(crate::AcpRuntimeError::Unsupported(
                    "session is already running a prompt".into(),
                )));
            }
            if queue.attempts >= MAX_PENDING_PROMPT_ATTEMPTS {
                return Err(Error::new(-32602, "pending prompt attempt limit reached"));
            }
            let id = if let Some(id) = live {
                queue
                    .groups
                    .get_mut(&id)
                    .expect("live group was found")
                    .attempts
                    .push(attempt);
                id
            } else {
                if queue.groups.len() >= MAX_PENDING_PROMPTS {
                    return Err(prompt_budget_error());
                }
                let (bytes, new_bytes) = if enforce_budget {
                    let bytes = measure_prompt(&request)?;
                    let new_bytes = queue.bytes.saturating_add(bytes);
                    if new_bytes.exceeds_budget() {
                        return Err(prompt_budget_error());
                    }
                    (bytes, new_bytes)
                } else {
                    (PromptBytes::default(), queue.bytes)
                };
                let items = content_blocks_to_items(&request.prompt)
                    .map_err(|err| Error::new(-32602, err.to_string()))?;
                let id = queue.next_group;
                let next = id.checked_add(1).ok_or_else(Error::internal_error)?;
                let group = Group {
                    key: request.idempotency_key.clone(),
                    claimed: false,
                    attempts: vec![attempt],
                };
                queue.groups.insert(id, group);
                queue.entries.push_back(Entry {
                    group: id,
                    request,
                    items,
                    bytes,
                });
                queue.next_group = next;
                queue.bytes = new_bytes;
                id
            };
            queue.attempts += 1;
            id
        };
        self.inner.changed.notify_waiters();
        Ok(PendingPromptResponse {
            queue: self.clone(),
            group,
            token,
            cancellation,
            response,
        })
    }

    fn withdraw(&self, id: u64) {
        let cancelled = self.lock().withdraw_unclaimed(id);
        if let Some((_entry, group)) = cancelled {
            group.settle(Err(Error::request_cancelled()));
            self.inner.changed.notify_waiters();
        }
    }

    fn detach(&self, id: u64, token: &Arc<()>) {
        let (attempt, orphan) = {
            let mut queue = self.lock();
            let Some(group) = queue.groups.get_mut(&id) else {
                return;
            };
            let Some(index) = group
                .attempts
                .iter()
                .position(|attempt| Arc::ptr_eq(&attempt.token, token))
            else {
                return;
            };
            let attempt = group.attempts.swap_remove(index);
            let withdraw = !group.claimed && (attempt.cancelled() || group.attempts.is_empty());
            queue.attempts -= 1;
            let orphan = if withdraw {
                Some(
                    queue
                        .withdraw_unclaimed(id)
                        .expect("unclaimed group is queued"),
                )
            } else {
                None
            };
            (attempt, orphan)
        };
        drop(attempt);
        if let Some((_entry, group)) = orphan {
            group.settle(Err(Error::request_cancelled()));
            self.inner.changed.notify_waiters();
        }
    }

    fn clear(&self, close: bool, only_ordinary: bool) {
        let mut cancelled = Vec::with_capacity(MAX_PENDING_PROMPTS);
        {
            let mut queue = self.lock();
            queue.closed |= close;
            let mut keep = VecDeque::with_capacity(queue.entries.len());
            for entry in std::mem::take(&mut queue.entries) {
                if only_ordinary && explicit_delivery(&entry.request) {
                    keep.push_back(entry);
                } else {
                    let group = queue.remove_group(&entry);
                    cancelled.push((entry, group));
                }
            }
            queue.entries = keep;
        }
        for (_entry, group) in cancelled {
            group.settle(Err(Error::request_cancelled()));
        }
        self.inner.changed.notify_waiters();
    }

    /// Completes every still-pending submission with `-32800`.
    pub fn cancel_all(&self) {
        self.clear(false, false);
    }

    /// Cancels only ordinary prompts. Explicit steer and queue survive session cancel.
    pub fn cancel_ordinary(&self) {
        self.clear(false, true);
    }

    /// Cancels pending requests and permanently seals the queue.
    pub fn close(&self) {
        self.clear(true, false);
    }

    /// Waits for pending work. The caller must also observe session shutdown.
    pub async fn ready(&self) {
        loop {
            let changed = self.inner.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.lock().entries.is_empty() {
                return;
            }
            changed.await;
        }
    }

    /// Claims the oldest steer first, otherwise the oldest remaining prompt.
    /// Claim only at the insertion boundary, after any pre-insertion awaits.
    pub fn claim_next(&self) -> Option<PendingPrompt> {
        self.claim(Claim::One(true)).pop()
    }

    /// Reports whether a steer is waiting. Cancellation may win before claim.
    pub fn has_steers(&self) -> bool {
        let queue = self.lock();
        queue
            .entries
            .iter()
            .any(|entry| is_steer(&entry.request) && !queue.groups[&entry.group].cancelled())
    }

    /// Claims the steers currently waiting, in receive order.
    pub fn claim_steers(&self) -> Vec<PendingPrompt> {
        self.claim(Claim::Batch(false))
    }

    /// Claims all waiting steers or, if none, one queue/ordinary prompt.
    pub fn claim_next_turn(&self) -> Vec<PendingPrompt> {
        self.claim(Claim::Batch(true))
    }

    /// Claims the oldest steer at a model safe point, skipping queued prompts.
    pub fn claim_steer(&self) -> Option<PendingPrompt> {
        self.claim(Claim::One(false)).pop()
    }

    fn claim(&self, mode: Claim) -> Vec<PendingPrompt> {
        let mut cancelled = Vec::with_capacity(MAX_PENDING_PROMPTS);
        let mut claims = Vec::with_capacity(MAX_PENDING_PROMPTS);
        {
            let mut queue = self.lock();
            let mut retained = VecDeque::with_capacity(queue.entries.len());
            for entry in std::mem::take(&mut queue.entries) {
                if queue.groups[&entry.group].cancelled() {
                    let group = queue.remove_group(&entry);
                    cancelled.push((entry, group));
                } else if is_steer(&entry.request)
                    && (matches!(mode, Claim::Batch(_)) || claims.is_empty())
                {
                    queue
                        .groups
                        .get_mut(&entry.group)
                        .expect("entry has a live group")
                        .claimed = true;
                    claims.push(entry);
                } else {
                    retained.push_back(entry);
                }
            }
            if claims.is_empty() && matches!(mode, Claim::One(true) | Claim::Batch(true)) {
                while let Some(entry) = retained.pop_front() {
                    // A retry could cancel this entry while we searched for
                    // a higher-priority steer. Recheck at fallback selection.
                    let group = queue
                        .groups
                        .get_mut(&entry.group)
                        .expect("entry has a live group");
                    if group.cancelled() {
                        let group = queue.remove_group(&entry);
                        cancelled.push((entry, group));
                    } else {
                        group.claimed = true;
                        claims.push(entry);
                        break;
                    }
                }
            }
            queue.entries = retained;
        }
        let claims = claims
            .into_iter()
            .map(|entry| PendingPrompt {
                queue: self.clone(),
                entry: Some(entry),
            })
            .collect();
        for (_entry, group) in cancelled {
            group.settle(Err(Error::request_cancelled()));
        }
        claims
    }
}

fn explicit_delivery(_request: &wire::PromptRequest) -> bool {
    #[cfg(feature = "unstable-prompt-delivery")]
    {
        return _request.delivery.is_some();
    }
    #[cfg(not(feature = "unstable-prompt-delivery"))]
    false
}

fn is_steer(_request: &wire::PromptRequest) -> bool {
    #[cfg(feature = "unstable-prompt-delivery")]
    {
        return _request.delivery == Some(wire::PromptDelivery::Steer);
    }
    #[cfg(not(feature = "unstable-prompt-delivery"))]
    false
}

/// A pending insertion result. Dropping the last waiter withdraws an unclaimed prompt.
#[must_use = "await the prompt response to keep its request pending"]
pub struct PendingPromptResponse {
    queue: PendingPrompts,
    group: u64,
    token: Arc<()>,
    cancellation: Option<RequestCancellation>,
    response: oneshot::Receiver<PromptResult>,
}

impl PendingPromptResponse {
    /// Reports insertion or rejection, handling cancellation of this request.
    pub async fn wait(mut self) -> PromptResult {
        if let Some(cancellation) = self.cancellation.take() {
            tokio::select! {
                biased;
                result = &mut self.response => return result.map_err(|_| Error::new(-32603, "prompt response lost"))?,
                () = cancellation.cancelled() => self.queue.withdraw(self.group),
            }
        }
        (&mut self.response)
            .await
            .map_err(|_| Error::new(-32603, "prompt response lost"))?
    }
}

impl Drop for PendingPromptResponse {
    fn drop(&mut self) {
        self.queue.detach(self.group, &self.token);
    }
}

/// Prompt claimed by the host when ready to insert it. Request cancellation
/// cannot withdraw a claim; the consumer must complete it with its outcome.
#[must_use = "complete a claimed prompt with its insertion result"]
pub struct PendingPrompt {
    queue: PendingPrompts,
    entry: Option<Entry>,
}

impl PendingPrompt {
    /// Returns the original prompt, including content and delivery mode.
    pub fn request(&self) -> &wire::PromptRequest {
        &self.entry.as_ref().expect("live claim").request
    }

    /// Moves the pre-converted model input out of the claim.
    pub fn take_items(&mut self) -> Vec<Item> {
        std::mem::take(&mut self.entry.as_mut().expect("live claim").items)
    }

    /// Completes the prompt. Only supply a success once the message is inserted.
    pub fn complete(mut self, result: PromptResult) {
        self.finish(result);
    }

    fn finish(&mut self, result: PromptResult) {
        if let Some(entry) = self.entry.take() {
            let group = self.queue.lock().remove_group(&entry);
            group.settle(result);
        }
    }
}

impl Drop for PendingPrompt {
    fn drop(&mut self) {
        if self.entry.is_some() {
            self.finish(Err(Error::new(-32603, "claimed prompt was not completed")));
        }
    }
}

#[cfg(all(test, feature = "unstable-prompt-delivery"))]
mod tests {
    use super::*;
    use agent_client_protocol::{Agent, Channel, Client};

    fn request(
        session_id: &wire::SessionId,
        text: &str,
        mode: Option<wire::PromptDelivery>,
    ) -> wire::PromptRequest {
        wire::PromptRequest::new(
            session_id.clone(),
            vec![wire::ContentBlock::Text(wire::TextContent::new(text))],
        )
        .delivery(mode)
    }

    fn text(prompt: &PendingPrompt) -> &str {
        match &prompt.request().prompt[0] {
            wire::ContentBlock::Text(text) => &text.text,
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn steer_precedence_preserves_mode_fifo_and_session_cancel_scope() {
        use wire::PromptDelivery::{Queue, Steer};
        let session = wire::SessionId::new("delivery");
        let pending = PendingPrompts::new(session.clone());
        let enqueue = |text, mode| pending.admit(request(&session, text, mode), None).unwrap();
        let responses = [
            enqueue("q1", Some(Queue)),
            enqueue("s1", Some(Steer)),
            enqueue("s2", Some(Steer)),
            enqueue("q2", Some(Queue)),
        ];
        let ordinary = enqueue("ordinary", None);
        pending.cancel_ordinary();
        assert_eq!(i32::from(ordinary.wait().await.unwrap_err().code), -32800);
        let steer_batch = pending.claim_next_turn();
        assert_eq!(
            steer_batch.iter().map(text).collect::<Vec<_>>(),
            ["s1", "s2"]
        );
        assert!(!pending.has_steers());
        for steer in steer_batch {
            let message_id = wire::MessageId::new(text(&steer).to_owned());
            steer.complete(Ok(wire::PromptResponse::new(message_id)));
        }
        for expected in ["q1", "q2"] {
            let batch = pending.claim_next_turn();
            assert_eq!(batch.len(), 1);
            assert_eq!(text(&batch[0]), expected);
            batch
                .into_iter()
                .next()
                .unwrap()
                .complete(Ok(wire::PromptResponse::new(expected)));
        }
        for (response, expected) in responses.into_iter().zip(["q1", "s1", "s2", "q2"]) {
            assert_eq!(
                response.wait().await.unwrap().message_id,
                wire::MessageId::new(expected)
            );
        }
    }

    #[tokio::test]
    async fn pending_and_claimed_inputs_share_limits_and_close_seals() {
        let session = wire::SessionId::new("limits");
        let pending = PendingPrompts::new(session.clone());
        let half = "a".repeat(MAX_PENDING_PROMPT_BYTES / 2);
        let enqueue = |text| pending.admit(request(&session, text, None), None);
        let response = enqueue(&half).unwrap();
        let claimed = pending.claim_next().unwrap();
        assert!(enqueue(&half).is_err());
        drop(claimed);
        assert_eq!(i32::from(response.wait().await.unwrap_err().code), -32603);
        let responses = (0..MAX_PENDING_PROMPTS)
            .map(|_| enqueue("x").unwrap())
            .collect::<Vec<_>>();
        assert!(enqueue("x").is_err());
        pending.close();
        for response in responses {
            assert_eq!(i32::from(response.wait().await.unwrap_err().code), -32800);
        }
        assert!(matches!(enqueue("x"), Err(err) if i32::from(err.code) == -32800));
        assert!(pending.claim_next_turn().is_empty());
    }

    #[tokio::test]
    async fn duplicate_waiters_are_bounded_and_detaching_one_does_not_drop_the_group() {
        let session = wire::SessionId::new("retry-limits");
        let pending = PendingPrompts::new(session.clone());
        let key = wire::IdempotencyKey::new("same").unwrap();
        let enqueue = || {
            pending.admit(
                request(&session, "same", None).idempotency_key(key.clone()),
                None,
            )
        };
        let mut responses = (0..MAX_PENDING_PROMPT_ATTEMPTS)
            .map(|_| enqueue().unwrap())
            .collect::<Vec<_>>();
        assert!(enqueue().is_err());
        drop(responses.pop());
        responses.push(enqueue().unwrap());
        let mut batch = pending.claim_next_turn();
        assert_eq!(batch.len(), 1);
        batch
            .pop()
            .unwrap()
            .complete(Ok(wire::PromptResponse::new("once")));
        for response in responses {
            assert_eq!(
                response.wait().await.unwrap().message_id,
                wire::MessageId::new("once")
            );
        }
        // AgentKit retains no completed keys. The host owns replay lookup.
        let later = enqueue().unwrap();
        assert!(pending.claim_next().is_some());
        assert_eq!(i32::from(later.wait().await.unwrap_err().code), -32603);
    }

    #[tokio::test]
    async fn helper_acknowledges_transcript_input_even_when_its_checkpoint_is_cancelled() {
        use crate::AcpRuntimeError;
        use crate::v2::{AcpIntegration, AcpSessionBinding, AcpSessionUpdateSink};
        use agentkit_core::{CancellationController, Part, SessionId};
        use agentkit_integration_tests::mock_model::MockAdapter;
        use agentkit_loop::{Agent, LoopError, SessionConfig};

        struct InterruptOnUpdate(CancellationController);
        #[async_trait::async_trait]
        impl AcpSessionUpdateSink for InterruptOnUpdate {
            fn update(&self, _: wire::UpdateSessionNotification) -> Result<(), AcpRuntimeError> {
                self.0.interrupt();
                Ok(())
            }
            async fn update_acknowledged(
                &self,
                update: wire::UpdateSessionNotification,
            ) -> Result<(), AcpRuntimeError> {
                self.update(update)
            }
            async fn flush(&self) -> Result<(), AcpRuntimeError> {
                Ok(())
            }
        }

        let acp_session = wire::SessionId::new("durable-steer");
        let model_session = SessionId::new("model-session");
        let cancellation = CancellationController::new();
        let integration = AcpIntegration::default();
        let handle = integration
            .bind_session(
                AcpSessionBinding::new(
                    acp_session.clone(),
                    model_session.clone(),
                    InterruptOnUpdate(cancellation.clone()),
                )
                .cancellation(cancellation.clone()),
            )
            .unwrap();
        let mut driver = Agent::builder()
            .model(MockAdapter::default())
            .cancellation(cancellation.handle())
            .build()
            .unwrap()
            .start(SessionConfig::new(model_session))
            .await
            .unwrap();
        let checkpoint = handle.cancellation_handle().checkpoint();
        let response = handle
            .pending_prompts()
            .admit(
                request(
                    &acp_session,
                    "durable steer",
                    Some(wire::PromptDelivery::Steer),
                ),
                None,
            )
            .unwrap();
        assert!(
            handle
                .deliver_pending_steers(&mut driver, checkpoint.clone())
                .await
                .unwrap()
        );
        assert_eq!(
            response.wait().await.unwrap().message_id,
            wire::MessageId::new("durable-steer-user-1")
        );
        let snapshot = driver.snapshot();
        assert!(snapshot.pending_input.is_empty());
        assert!(
            matches!(snapshot.transcript[0].parts.as_slice(), [Part::Text(text)] if text.text == "durable steer")
        );
        assert!(matches!(
            driver.next_with_cancellation(checkpoint).await,
            Err(LoopError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn wire_retries_share_cancellation_before_claim_and_insertion_after_claim() {
        let session = wire::SessionId::new("wire-cancel");
        let pending = PendingPrompts::new(session.clone());
        let (client_io, agent_io) = Channel::duplex();
        let (admitted, mut admission) = tokio::sync::mpsc::unbounded_channel();
        let server = tokio::spawn(
            Agent
                .v2()
                .on_receive_request(
                    async |_: wire::InitializeRequest, responder, _cx| {
                        responder.respond(
                            wire::InitializeResponse::new(
                                wire::ProtocolVersion::V2,
                                wire::Implementation::new("delivery-test-agent", "1"),
                            )
                            .capabilities(
                                wire::AgentCapabilities::new().session(
                                    wire::SessionCapabilities::new().prompt(
                                        wire::PromptCapabilities::new().delivery(
                                            wire::PromptDeliveryCapabilities::new(true, true),
                                        ),
                                    ),
                                ),
                            ),
                        )
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .on_receive_request(
                    {
                        let session = session.clone();
                        async move |_: wire::NewSessionRequest, responder, _cx| {
                            responder.respond(wire::NewSessionResponse::new(session.clone()))
                        }
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .on_receive_request(
                    {
                        let pending = pending.clone();
                        async move |request: wire::PromptRequest, responder, cx| {
                            let wait = pending.enqueue(request, responder.cancellation())?;
                            let _ = admitted.send(responder.cancellation());
                            cx.spawn(
                                async move { responder.respond_with_result(wait.wait().await) },
                            )?;
                            Ok(())
                        }
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .connect_to(agent_io),
        );
        let client = Client.v2().connect_with(client_io, async move |cx| {
            cx.send_request(wire::InitializeRequest::new(
                wire::ProtocolVersion::V2,
                wire::Implementation::new("delivery-test-client", "1"),
            ))
            .block_task()
            .await?;
            let cwd = std::env::current_dir().map_err(Error::into_internal_error)?;
            assert_eq!(
                cx.send_request(wire::NewSessionRequest::new(cwd))
                    .block_task()
                    .await?
                    .session_id,
                session
            );
            let key = wire::IdempotencyKey::new("same-submission").unwrap();
            let steer = |text| {
                request(&session, text, Some(wire::PromptDelivery::Steer))
                    .idempotency_key(key.clone())
            };
            let held = cx.send_request(steer("withdraw"));
            admission.recv().await.expect("original admission");
            let retry = cx.send_request(steer("withdraw"));
            admission.recv().await.expect("retry admission");
            assert!(pending.has_steers());
            retry.cancel()?;
            let (first_error, retry_error) = tokio::join!(held.block_task(), retry.block_task());
            for error in [first_error.unwrap_err(), retry_error.unwrap_err()] {
                assert_eq!(i32::from(error.code), -32800);
            }
            assert!(pending.claim_steers().is_empty());

            // An error frees the key. Two retries join before the next claim,
            // and a third attempt joins the already claimed submission.
            let original = cx.send_request(steer("first content wins"));
            admission.recv().await.expect("original admission");
            let retry = cx.send_request(steer("different retry content"));
            let marker = admission.recv().await.expect("retry admission");
            let mut batch = pending.claim_steers();
            assert_eq!(batch.len(), 1);
            let prompt = batch.pop().unwrap();
            assert_eq!(text(&prompt), "first content wins");
            retry.cancel()?;
            tokio::time::timeout(std::time::Duration::from_secs(2), marker.cancelled())
                .await
                .expect("claimed prompt cancellation did not reach the real SDK marker");
            let late_retry = cx.send_request(steer("late retry"));
            admission.recv().await.expect("late retry admission");
            assert!(pending.claim_next_turn().is_empty());
            prompt.complete(Ok(wire::PromptResponse::new("inserted")));
            let results = tokio::join!(
                original.block_task(),
                retry.block_task(),
                late_retry.block_task()
            );
            for result in [results.0?, results.1?, results.2?] {
                assert_eq!(result.message_id, wire::MessageId::new("inserted"));
            }
            Ok(())
        });
        tokio::time::timeout(std::time::Duration::from_secs(8), client)
            .await
            .unwrap()
            .unwrap();
        server.abort();
        let _ = server.await;
    }
}
