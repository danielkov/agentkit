use std::fmt;
use std::sync::Arc;

use agentkit_core::{DataRef, Item, MetadataMap, SessionId, TurnId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

macro_rules! capability_id {
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
    };
}

capability_id!(CapabilityName);
capability_id!(ResourceId);
capability_id!(PromptId);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvocableSpec {
    pub name: CapabilityName,
    pub description: String,
    pub input_schema: Value,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvocableRequest {
    pub input: Value,
    pub session_id: Option<SessionId>,
    pub turn_id: Option<TurnId>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvocableResult {
    pub output: InvocableOutput,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InvocableOutput {
    Text(String),
    Structured(Value),
    Items(Vec<Item>),
    Data(DataRef),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDescriptor {
    pub id: ResourceId,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceContents {
    pub data: DataRef,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptDescriptor {
    pub id: PromptId,
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptContents {
    pub items: Vec<Item>,
    pub metadata: MetadataMap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityContext<'a> {
    pub session_id: Option<&'a SessionId>,
    pub turn_id: Option<&'a TurnId>,
    pub metadata: &'a MetadataMap,
}

#[async_trait]
pub trait Invocable: Send + Sync {
    fn spec(&self) -> &InvocableSpec;

    async fn invoke(
        &self,
        request: InvocableRequest,
        ctx: &mut CapabilityContext<'_>,
    ) -> Result<InvocableResult, CapabilityError>;
}

#[async_trait]
pub trait ResourceProvider: Send + Sync {
    async fn list_resources(&self) -> Result<Vec<ResourceDescriptor>, CapabilityError>;

    async fn read_resource(
        &self,
        id: &ResourceId,
        ctx: &mut CapabilityContext<'_>,
    ) -> Result<ResourceContents, CapabilityError>;
}

#[async_trait]
pub trait PromptProvider: Send + Sync {
    async fn list_prompts(&self) -> Result<Vec<PromptDescriptor>, CapabilityError>;

    async fn get_prompt(
        &self,
        id: &PromptId,
        args: Value,
        ctx: &mut CapabilityContext<'_>,
    ) -> Result<PromptContents, CapabilityError>;
}

pub trait CapabilityProvider: Send + Sync {
    fn invocables(&self) -> Vec<Arc<dyn Invocable>>;
    fn resources(&self) -> Vec<Arc<dyn ResourceProvider>>;
    fn prompts(&self) -> Vec<Arc<dyn PromptProvider>>;
}

#[derive(Debug, Error)]
pub enum CapabilityError {
    #[error("capability unavailable: {0}")]
    Unavailable(String),
    #[error("invalid capability input: {0}")]
    InvalidInput(String),
    #[error("capability execution failed: {0}")]
    ExecutionFailed(String),
}
