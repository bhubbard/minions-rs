//! OpenAI and OpenAI-compatible HTTP client (`/v1/chat/completions`).

use async_trait::async_trait;
use minions_core::{ChatOptions, ChatResponse, LlmClient, Message, MinionError, Usage};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Client for OpenAI and compatible endpoints (Together, Groq, DeepSeek, vLLM, `anemll-server`).
#[derive(Debug, Clone)]
pub struct OpenAiClient {
    model: String,
    api_key: String,
    base_url: String,
    temperature: f32,
    http: Client,
}

impl OpenAiClient {
    /// Creates an [`OpenAiClient`] targeting `https://api.openai.com/v1`.
    #[must_use]
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::with_base_url(model, api_key, "https://api.openai.com/v1")
    }

    /// Creates an [`OpenAiClient`] targeting a custom OpenAI-compatible base URL.
    #[must_use]
    pub fn with_base_url(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            temperature: 0.0,
            http: Client::new(),
        }
    }
}

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[async_trait]
impl LlmClient for OpenAiClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat(
        &self,
        messages: &[Message],
        options: &ChatOptions,
    ) -> Result<ChatResponse, MinionError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = OpenAiRequest {
            model: &self.model,
            messages,
            temperature: options.temperature.unwrap_or(self.temperature),
            max_tokens: options.max_tokens,
            n: options.num_samples,
        };

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
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

        let parsed: OpenAiResponse = resp
            .json()
            .await
            .map_err(|e| MinionError::Http(e.to_string()))?;

        let responses = parsed
            .choices
            .iter()
            .map(|c| c.message.content.clone().unwrap_or_default())
            .collect();
        let finish_reasons = parsed
            .choices
            .iter()
            .map(|c| {
                c.finish_reason
                    .clone()
                    .unwrap_or_else(|| "stop".to_string())
            })
            .collect();
        let usage = parsed.usage.map_or_else(Usage::default, |u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            cached_prompt_tokens: u.prompt_tokens_details.map_or(0, |d| d.cached_tokens),
            seen_prompt_tokens: 0,
        });

        Ok(ChatResponse {
            responses,
            usage,
            finish_reasons,
        })
    }
}
