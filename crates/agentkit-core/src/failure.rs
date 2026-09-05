//! Closed, payload-free diagnostic facts. These observations never authorize replay.

use crate::retry::{
    ProviderClassification, ProviderFailure, ProviderFailureReason, ProviderRoute, RetryAccounting,
    UpstreamErrorKind,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::time::Duration;

/// Maximum encoded metadata accepted at an untrusted transport boundary.
pub const MAX_FAILURE_METADATA_BYTES: usize = 4096;

/// Static validation failure; rejected input is never included in its display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid failure metadata")]
pub struct FailureMetadataDecodeError;

/// Host-issued receipt identifier. Grammar validation is not host authentication.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct HostReceiptId(String);
impl HostReceiptId {
    pub fn new(value: impl AsRef<str>) -> Result<Self, FailureMetadataDecodeError> {
        let value = value.as_ref();
        if value.is_empty()
            || value.len() > 128
            || !value
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        {
            return Err(FailureMetadataDecodeError);
        }
        Ok(Self(value.to_owned()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl<'de> Deserialize<'de> for HostReceiptId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct IdVisitor;
        impl serde::de::Visitor<'_> for IdVisitor {
            type Value = HostReceiptId;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("bounded host receipt identifier")
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                HostReceiptId::new(value).map_err(serde::de::Error::custom)
            }
        }
        d.deserialize_str(IdVisitor)
            .map_err(|_| serde::de::Error::custom(FailureMetadataDecodeError))
    }
}

/// Retention at emission time, not a promise of eternal availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FatalStorage {
    Stored,
    /// Only when a retrievable in-memory record actually exists.
    MemoryOnly,
    Unavailable,
}

/// Allocate once at the emitting host boundary; never reconstruct from a path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostFatalReceipt {
    pub session_id: HostReceiptId,
    pub event_id: HostReceiptId,
    pub storage: FatalStorage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    ChildFailed,
    ChildTransportFailed,
    ChildCancelled,
    ProviderFailed,
    HostFailed,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSource {
    #[default]
    Unknown,
    /// Reports during a dispatched child prompt, not local invocation receipts.
    AcpNotifications,
    /// Cumulative within this live session owner, not earlier persisted history.
    LocalSession,
}

/// Recovery-owned positive facts. False means not observed; completion does not
/// imply success, commit, rollback, or completion of every invocation. Producers
/// retain true flags monotonically within one scope; do not merge foreign scopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PossibleEffects {
    pub source: ObservationSource,
    pub assistant_output_observed: bool,
    pub tool_emission_observed: bool,
    pub tool_execution_start_reported: bool,
    pub tool_execution_completion_reported: bool,
    #[serde(deserialize_with = "incomplete")]
    observation_incomplete: bool,
}
fn incomplete<'de, D: Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    if bool::deserialize(d)? {
        Ok(true)
    } else {
        Err(serde::de::Error::custom(FailureMetadataDecodeError))
    }
}
impl Default for PossibleEffects {
    fn default() -> Self {
        Self {
            source: ObservationSource::Unknown,
            assistant_output_observed: false,
            tool_emission_observed: false,
            tool_execution_start_reported: false,
            tool_execution_completion_reported: false,
            observation_incomplete: true,
        }
    }
}
impl PossibleEffects {
    pub fn observation_incomplete(&self) -> bool {
        self.observation_incomplete
    }
}

/// Versioned finite diagnostic leaf. No provider messages, paths, arbitrary JSON,
/// recursive errors, or host control capabilities belong in this durable value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FailureMetadataV1 {
    version: u8,
    code: FailureCode,
    #[serde(skip_serializing_if = "Option::is_none")]
    retry: Option<ProviderFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    receipt: Option<HostFatalReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effects: Option<PossibleEffects>,
}
impl Default for FailureMetadataV1 {
    fn default() -> Self {
        Self::new(FailureCode::Unknown)
    }
}
impl FailureMetadataV1 {
    pub fn new(code: FailureCode) -> Self {
        Self {
            version: 1,
            code,
            retry: None,
            receipt: None,
            effects: None,
        }
    }
    pub fn code(&self) -> FailureCode {
        self.code
    }
    pub fn retry(&self) -> Option<&ProviderFailure> {
        self.retry.as_ref()
    }
    pub fn receipt(&self) -> Option<&HostFatalReceipt> {
        self.receipt.as_ref()
    }
    pub fn effects(&self) -> Option<&PossibleEffects> {
        self.effects.as_ref()
    }
    pub fn with_retry(
        mut self,
        retry: ProviderFailure,
    ) -> Result<Self, FailureMetadataDecodeError> {
        if retry
            .upstream
            .http_status
            .is_some_and(|s| !(100..=599).contains(&s))
        {
            return Err(FailureMetadataDecodeError);
        }
        self.retry = Some(retry);
        Ok(self)
    }
    pub fn with_receipt(mut self, receipt: HostFatalReceipt) -> Self {
        self.receipt = Some(receipt);
        self
    }
    pub fn with_effects(mut self, effects: PossibleEffects) -> Self {
        self.effects = Some(effects);
        self
    }
    /// Use at untrusted boundaries BEFORE general JSON parsing. Fixed wire objects
    /// bound structural depth; serde's recursion limit also rejects nested input.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, FailureMetadataDecodeError> {
        if bytes.len() > MAX_FAILURE_METADATA_BYTES {
            return Err(FailureMetadataDecodeError);
        }
        serde_json::from_slice(bytes).map_err(|_| FailureMetadataDecodeError)
    }
}

// Strict wire layout only: canonical retry value enums and standalone serde stay
// unchanged. The new boundary closes nested objects instead of weakening legacy
// compatibility or maintaining a competing retry classification contract.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMetadata {
    version: u8,
    #[serde(default)]
    code: FailureCode,
    retry: Option<WireRetry>,
    receipt: Option<HostFatalReceipt>,
    effects: Option<PossibleEffects>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDuration {
    secs: u64,
    nanos: u32,
}
impl WireDuration {
    fn checked(self) -> Result<Duration, FailureMetadataDecodeError> {
        if self.nanos >= 1_000_000_000 {
            return Err(FailureMetadataDecodeError);
        }
        Ok(Duration::new(self.secs, self.nanos))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAccounting {
    attempts: u64,
    completed_backoff: WireDuration,
    elapsed: WireDuration,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireUpstream {
    error_type: UpstreamErrorKind,
    code: UpstreamErrorKind,
    http_status: Option<u16>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRetry {
    route: ProviderRoute,
    reason: ProviderFailureReason,
    last_attempt_reason: Option<ProviderFailureReason>,
    upstream: WireUpstream,
    accounting: WireAccounting,
}
impl WireRetry {
    fn checked(self) -> Result<ProviderFailure, FailureMetadataDecodeError> {
        Ok(ProviderFailure {
            route: self.route,
            reason: self.reason,
            last_attempt_reason: self.last_attempt_reason,
            upstream: ProviderClassification {
                error_type: self.upstream.error_type,
                code: self.upstream.code,
                http_status: self.upstream.http_status,
            },
            accounting: RetryAccounting {
                attempts: self.accounting.attempts,
                completed_backoff: self.accounting.completed_backoff.checked()?,
                elapsed: self.accounting.elapsed.checked()?,
            },
        })
    }
}
impl<'de> Deserialize<'de> for FailureMetadataV1 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = WireMetadata::deserialize(d)
            .map_err(|_| serde::de::Error::custom(FailureMetadataDecodeError))?;
        if wire.version != 1 {
            return Err(serde::de::Error::custom(FailureMetadataDecodeError));
        }
        let mut value = Self::new(wire.code);
        value.receipt = wire.receipt;
        value.effects = wire.effects;
        if let Some(retry) = wire.retry {
            value = value
                .with_retry(retry.checked().map_err(serde::de::Error::custom)?)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(value)
    }
}

/// Separately scoped, task-owned facts. Never merge these into a child's native
/// diagnostic identity. Optional final leaves are independent of terminal kind.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FailureObservations {
    #[serde(skip_serializing_if = "Option::is_none")]
    receipt: Option<HostFatalReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retry: Option<ProviderFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effects: Option<PossibleEffects>,
}
impl FailureObservations {
    /// Validate an already-parsed object without cloning arbitrary input trees.
    pub fn from_value(value: &serde_json::Value) -> Result<Self, FailureMetadataDecodeError> {
        Self::deserialize(value).map_err(|_| FailureMetadataDecodeError)
    }
    pub fn receipt(&self) -> Option<&HostFatalReceipt> {
        self.receipt.as_ref()
    }
    pub fn retry(&self) -> Option<&ProviderFailure> {
        self.retry.as_ref()
    }
    pub fn effects(&self) -> Option<&PossibleEffects> {
        self.effects.as_ref()
    }
    pub fn is_empty(&self) -> bool {
        self.receipt.is_none() && self.retry.is_none() && self.effects.is_none()
    }
    pub fn with_receipt(mut self, value: HostFatalReceipt) -> Self {
        self.receipt = Some(value);
        self
    }
    pub fn with_effects(mut self, value: PossibleEffects) -> Self {
        self.effects = Some(value);
        self
    }
    pub fn with_retry(
        mut self,
        value: ProviderFailure,
    ) -> Result<Self, FailureMetadataDecodeError> {
        FailureMetadataV1::default().with_retry(value)?;
        self.retry = Some(value);
        Ok(self)
    }
    pub fn from_slice(bytes: &[u8]) -> Result<Self, FailureMetadataDecodeError> {
        if bytes.len() > MAX_FAILURE_METADATA_BYTES {
            return Err(FailureMetadataDecodeError);
        }
        serde_json::from_slice(bytes).map_err(|_| FailureMetadataDecodeError)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireObservations {
    receipt: Option<HostFatalReceipt>,
    retry: Option<WireRetry>,
    effects: Option<PossibleEffects>,
}
impl<'de> Deserialize<'de> for FailureObservations {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = WireObservations::deserialize(d)
            .map_err(|_| serde::de::Error::custom(FailureMetadataDecodeError))?;
        let mut value = Self {
            receipt: wire.receipt,
            effects: wire.effects,
            retry: None,
        };
        if let Some(retry) = wire.retry {
            value = value
                .with_retry(retry.checked().map_err(serde::de::Error::custom)?)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(value)
    }
}
