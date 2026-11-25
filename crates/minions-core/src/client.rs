//! Abstract LLM client trait and chat message types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{MinionError, Usage};

/// Conversation role for a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt instructions.
    System,
    /// User / caller message.
    User,
    /// Assistant / model response.
    Assistant,
    /// Tool output message.
    Tool,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Speaker role.
    pub role: Role,
    /// Text content of the message.
    pub content: String,
    /// Optional participant or tool name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Creates a system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
        }
    }

    /// Creates a user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
        }
    }

    /// Creates an assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
        }
    }
}

/// Optional per-request overrides for [`LlmClient::chat`].
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// Sampling temperature override.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate override.
    pub max_tokens: Option<u32>,
    /// Number of parallel completions to request.
    pub num_samples: Option<u32>,
}

/// Response returned from [`LlmClient::chat`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[must_use]
pub struct ChatResponse {
    /// One or more generated completion strings.
    pub responses: Vec<String>,
    /// Token usage statistics for this call.
    pub usage: Usage,
    /// Finish reasons reported by the provider.
    pub finish_reasons: Vec<String>,
}

impl ChatResponse {
    /// Returns the first completion text, if any.
    #[must_use]
    pub fn first_text(&self) -> Option<&str> {
        self.responses.first().map(String::as_str)
    }
}

/// Asynchronous LLM chat interface implemented by local and cloud clients.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Returns the configured model identifier.
    fn model_name(&self) -> &str;

    /// Executes a chat completion request.
    ///
    /// # Errors
    /// Returns [`MinionError`] if transport, authentication, or response parsing fails.
    async fn chat(
        &self,
        messages: &[Message],
        options: &ChatOptions,
    ) -> Result<ChatResponse, MinionError>;
}
