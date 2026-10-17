use agentkit_core::{MetadataMap, TokenUsage, Usage};
use serde_json::Value;

/// Anthropic reports uncached input and cache buckets separately. Retain the
/// raw, optional counters until after merging streaming snapshots: normalizing
/// each delta first would lose omitted fields or count cached input twice.
#[derive(Clone, Default)]
pub(crate) struct UsageSnapshot {
    input: Option<u64>,
    output: Option<u64>,
    cache_read: Option<u64>,
    cache_write: Option<u64>,
    metadata: MetadataMap,
}

impl UsageSnapshot {
    pub(crate) fn parse(value: Option<&Value>) -> Option<Self> {
        let value = value?;
        let mut metadata = MetadataMap::new();
        for key in ["cache_creation", "service_tier", "server_tool_use"] {
            if let Some(value) = value.get(key) {
                metadata.insert(format!("anthropic.{key}"), value.clone());
            }
        }
        Some(Self {
            input: value.get("input_tokens").and_then(Value::as_u64),
            output: value.get("output_tokens").and_then(Value::as_u64),
            cache_read: value.get("cache_read_input_tokens").and_then(Value::as_u64),
            cache_write: value
                .get("cache_creation_input_tokens")
                .and_then(Value::as_u64),
            metadata,
        })
    }

    /// Counters in message_delta are cumulative, not increments. Only omitted
    /// fields retain earlier values; an explicitly reported zero replaces them.
    pub(crate) fn merge(&mut self, next: Self) {
        self.input = next.input.or(self.input);
        self.output = next.output.or(self.output);
        self.cache_read = next.cache_read.or(self.cache_read);
        self.cache_write = next.cache_write.or(self.cache_write);
        self.metadata.extend(next.metadata);
    }

    pub(crate) fn to_usage(&self) -> Usage {
        Usage {
            tokens: Some(TokenUsage {
                // Normalize to total prompt occupancy, as expected by AgentKit
                // consumers. Cache fields remain breakdowns of this total.
                input_tokens: self
                    .input
                    .unwrap_or(0)
                    .saturating_add(self.cache_read.unwrap_or(0))
                    .saturating_add(self.cache_write.unwrap_or(0)),
                output_tokens: self.output.unwrap_or(0),
                reasoning_tokens: None,
                cached_input_tokens: self.cache_read,
                cache_write_input_tokens: self.cache_write,
            }),
            cost: None,
            metadata: self.metadata.clone(),
        }
    }
}
