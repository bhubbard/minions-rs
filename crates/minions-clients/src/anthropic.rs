//! Anthropic Messages API client (`/v1/messages`).

use async_trait::async_trait;
use minions_core::{ChatOptions, ChatResponse, LlmClient, Message, MinionError, Role, Usage};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Client for Anthropic's Claude Messages API.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    model: String,
    api_key: String,
    base_url: String,
    max_tokens: u32,
    http: Client,
}

impl AnthropicClient {
    /// Creates an [`AnthropicClient`] targeting `https://api.anthropic.com`.
    #[must_use]
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            max_tokens: 4096,
            http: Client::new(),
        }
    }
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMsg<'a>>,
}

#[derive(Serialize)]
struct AnthropicMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicBlock {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

#[async_trait]
impl LlmClient for AnthropicClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat(
        &self,
        messages: &[Message],
        options: &ChatOptions,
    ) -> Result<ChatResponse, MinionError> {
        let mut system_parts = Vec::new();
        let mut chat_msgs = Vec::new();
        for msg in messages {
            match msg.role {
                Role::System => system_parts.push(msg.content.as_str()),
                Role::User | Role::Tool => chat_msgs.push(AnthropicMsg {
                    role: "user",
                    content: &msg.content,
                }),
                Role::Assistant => chat_msgs.push(AnthropicMsg {
                    role: "assistant",
                    content: &msg.content,
                }),
            }
        }

        let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: options.max_tokens.unwrap_or(self.max_tokens),
            system,
            messages: chat_msgs,
        };

        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| MinionError::Http(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MinionError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let parsed: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| MinionError::Http(e.to_string()))?;

        let text = parsed
            .content
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");
        let usage = parsed.usage.map_or_else(Usage::default, |u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            cached_prompt_tokens: u.cache_read_input_tokens,
            seen_prompt_tokens: 0,
        });

        Ok(ChatResponse {
            responses: vec![text],
            usage,
            finish_reasons: vec![parsed.stop_reason.unwrap_or_else(|| "end_turn".to_string())],
        })
    }
}
