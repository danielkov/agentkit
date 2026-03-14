use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

macro_rules! id_newtype {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

id_newtype!(SessionId);
id_newtype!(TurnId);
id_newtype!(MessageId);
id_newtype!(ToolCallId);
id_newtype!(ToolResultId);
id_newtype!(ProviderMessageId);
id_newtype!(ArtifactId);
id_newtype!(PartId);
id_newtype!(ApprovalId);

pub type MetadataMap = BTreeMap<String, Value>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: Option<MessageId>,
    pub kind: ItemKind,
    pub parts: Vec<Part>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItemKind {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Context,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Part {
    Text(TextPart),
    Media(MediaPart),
    File(FilePart),
    Structured(StructuredPart),
    Reasoning(ReasoningPart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
    Custom(CustomPart),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartKind {
    Text,
    Media,
    File,
    Structured,
    Reasoning,
    ToolCall,
    ToolResult,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaPart {
    pub modality: Modality,
    pub mime_type: String,
    pub data: DataRef,
    pub metadata: MetadataMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Audio,
    Image,
    Video,
    Binary,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataRef {
    InlineText(String),
    InlineBytes(Vec<u8>),
    Uri(String),
    Handle(ArtifactId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePart {
    pub name: Option<String>,
    pub mime_type: Option<String>,
    pub data: DataRef,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredPart {
    pub value: Value,
    pub schema: Option<Value>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningPart {
    pub summary: Option<String>,
    pub data: Option<DataRef>,
    pub redacted: bool,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallPart {
    pub id: ToolCallId,
    pub name: String,
    pub input: Value,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResultPart {
    pub call_id: ToolCallId,
    pub output: ToolOutput,
    pub is_error: bool,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolOutput {
    Text(String),
    Structured(Value),
    Parts(Vec<Part>),
    Files(Vec<FilePart>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomPart {
    pub kind: String,
    pub data: Option<DataRef>,
    pub value: Option<Value>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Delta {
    BeginPart {
        part_id: PartId,
        kind: PartKind,
    },
    AppendText {
        part_id: PartId,
        chunk: String,
    },
    AppendBytes {
        part_id: PartId,
        chunk: Vec<u8>,
    },
    ReplaceStructured {
        part_id: PartId,
        value: Value,
    },
    SetMetadata {
        part_id: PartId,
        metadata: MetadataMap,
    },
    CommitPart {
        part: Part,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub tokens: Option<TokenUsage>,
    pub cost: Option<CostUsage>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostUsage {
    pub amount: f64,
    pub currency: String,
    pub provider_amount: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Completed,
    ToolCall,
    MaxTokens,
    Cancelled,
    Blocked,
    Error,
    Other(String),
}

pub trait ItemView {
    fn kind(&self) -> ItemKind;
    fn parts(&self) -> &[Part];
    fn metadata(&self) -> &MetadataMap;
}

impl ItemView for Item {
    fn kind(&self) -> ItemKind {
        self.kind
    }

    fn parts(&self) -> &[Part] {
        &self.parts
    }

    fn metadata(&self) -> &MetadataMap {
        &self.metadata
    }
}

#[derive(Debug, Error)]
pub enum NormalizeError {
    #[error("unsupported content shape: {0}")]
    Unsupported(String),
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("invalid protocol state: {0}")]
    InvalidState(String),
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Normalize(#[from] NormalizeError),
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}
