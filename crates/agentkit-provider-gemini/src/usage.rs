use agentkit_core::{MetadataMap, TokenUsage, Usage};
use serde_json::Value;

/// Cumulative snapshot of Gemini's `usageMetadata`.
///
/// Unlike Anthropic, Gemini's `promptTokenCount` already includes
/// `cachedContentTokenCount`, so we preserve it directly as `input_tokens`
/// without double-counting cached tokens.
#[derive(Clone, Default)]
pub(crate) struct UsageSnapshot {
    prompt_tokens: Option<u64>,
    tool_use_prompt_tokens: Option<u64>,
    candidates_tokens: Option<u64>,
    thoughts_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    metadata: MetadataMap,
}

impl UsageSnapshot {
    pub(crate) fn parse(value: Option<&Value>) -> Option<Self> {
        let value = value?;
        if !value.is_object() {
            return None;
        }
        let mut metadata = MetadataMap::new();
        for (raw_key, meta_key) in [
            ("totalTokenCount", "gemini.total_token_count"),
            (
                "toolUsePromptTokenCount",
                "gemini.tool_use_prompt_token_count",
            ),
            ("trafficType", "gemini.traffic_type"),
            ("promptTokensDetails", "gemini.prompt_tokens_details"),
            (
                "candidatesTokensDetails",
                "gemini.candidates_tokens_details",
            ),
            ("cacheTokensDetails", "gemini.cache_tokens_details"),
        ] {
            if let Some(v) = value.get(raw_key) {
                metadata.insert(meta_key.into(), v.clone());
            }
        }

        Some(Self {
            prompt_tokens: value.get("promptTokenCount").and_then(Value::as_u64),
            tool_use_prompt_tokens: value.get("toolUsePromptTokenCount").and_then(Value::as_u64),
            candidates_tokens: value.get("candidatesTokenCount").and_then(Value::as_u64),
            thoughts_tokens: value.get("thoughtsTokenCount").and_then(Value::as_u64),
            cached_tokens: value.get("cachedContentTokenCount").and_then(Value::as_u64),
            metadata,
        })
    }

    /// Merges a later streaming snapshot into this one. Gemini's streaming
    /// `usageMetadata` is cumulative; omitted fields retain earlier values.
    pub(crate) fn merge(&mut self, next: Self) {
        self.prompt_tokens = next.prompt_tokens.or(self.prompt_tokens);
        self.tool_use_prompt_tokens = next.tool_use_prompt_tokens.or(self.tool_use_prompt_tokens);
        self.candidates_tokens = next.candidates_tokens.or(self.candidates_tokens);
        self.thoughts_tokens = next.thoughts_tokens.or(self.thoughts_tokens);
        self.cached_tokens = next.cached_tokens.or(self.cached_tokens);
        self.metadata.extend(next.metadata);
    }

    pub(crate) fn to_usage(&self) -> Usage {
        let input_tokens = self
            .prompt_tokens
            .unwrap_or(0)
            .saturating_add(self.tool_use_prompt_tokens.unwrap_or(0));
        Usage {
            tokens: Some(TokenUsage {
                input_tokens,
                output_tokens: self.candidates_tokens.unwrap_or(0),
                reasoning_tokens: self.thoughts_tokens,
                cached_input_tokens: self.cached_tokens,
                cache_write_input_tokens: None,
            }),
            cost: None,
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn does_not_double_count_cached_tokens_and_maps_thoughts() {
        let raw = json!({
            "promptTokenCount": 1000,
            "cachedContentTokenCount": 800,
            "candidatesTokenCount": 120,
            "thoughtsTokenCount": 45,
            "totalTokenCount": 1165
        });
        let usage = UsageSnapshot::parse(Some(&raw)).unwrap().to_usage();
        assert_eq!(
            usage.tokens,
            Some(TokenUsage {
                input_tokens: 1000,
                output_tokens: 120,
                reasoning_tokens: Some(45),
                cached_input_tokens: Some(800),
                cache_write_input_tokens: None,
            })
        );
        assert_eq!(usage.metadata["gemini.total_token_count"], json!(1165));
    }

    #[test]
    fn merges_cumulative_snapshots() {
        let mut first = UsageSnapshot::parse(Some(&json!({
            "promptTokenCount": 500,
            "cachedContentTokenCount": 300
        })))
        .unwrap();
        let second = UsageSnapshot::parse(Some(&json!({
            "promptTokenCount": 500,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 12
        })))
        .unwrap();
        first.merge(second);
        let tokens = first.to_usage().tokens.unwrap();
        assert_eq!(tokens.input_tokens, 500);
        assert_eq!(tokens.output_tokens, 40);
        assert_eq!(tokens.reasoning_tokens, Some(12));
        assert_eq!(tokens.cached_input_tokens, Some(300));
    }
}
