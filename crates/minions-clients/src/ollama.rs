//! Local Ollama HTTP client (`/api/chat`).

use async_trait::async_trait;
use minions_core::{ChatOptions, ChatResponse, LlmClient, Message, MinionError, Usage};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Client for a local or LAN Ollama server.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    model: String,
    base_url: String,
    temperature: f32,
    http: Client,
}

impl OllamaClient {
    /// Creates an [`OllamaClient`] targeting `http://localhost:11434`.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base_url(model, "http://localhost:11434")
    }

    /// Creates an [`OllamaClient`] targeting a custom Ollama base URL.
    #[must_use]
    pub fn with_base_url(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            temperature: 0.0,
            http: Client::new(),
        }
    }
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    done_reason: Option<String>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    content: String,
}

#[async_trait]
impl LlmClient for OllamaClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat(
        &self,
        messages: &[Message],
        options: &ChatOptions,
    ) -> Result<ChatResponse, MinionError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = OllamaRequest {
            model: &self.model,
            messages,
            stream: false,
            options: OllamaOptions {
                temperature: options.temperature.unwrap_or(self.temperature),
                num_predict: options.max_tokens,
            },
        };

        let resp = self
            .http
            .post(&url)
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

        let parsed: OllamaResponse = resp
            .json()
            .await
            .map_err(|e| MinionError::Http(e.to_string()))?;

        Ok(ChatResponse {
            responses: vec![parsed.message.content],
            usage: Usage::new(parsed.prompt_eval_count, parsed.eval_count),
            finish_reasons: vec![parsed.done_reason.unwrap_or_else(|| "stop".to_string())],
        })
    }
}
