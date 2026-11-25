//! Google Gemini `generateContent` HTTP client.

use async_trait::async_trait;
use minions_core::{ChatOptions, ChatResponse, LlmClient, Message, MinionError, Role, Usage};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Client for the Google Gemini REST API.
#[derive(Debug, Clone)]
pub struct GeminiClient {
    model: String,
    api_key: String,
    base_url: String,
    http: Client,
}

impl GeminiClient {
    /// Creates a [`GeminiClient`] targeting Google's Generative Language API.
    #[must_use]
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            http: Client::new(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest<'a> {
    contents: Vec<GeminiContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent<'a>>,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    parts: Vec<GeminiPart<'a>>,
}

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiCandidateContent,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiCandidateContent {
    #[serde(default)]
    parts: Vec<GeminiCandidatePart>,
}

#[derive(Deserialize)]
struct GeminiCandidatePart {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    #[serde(default)]
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    #[serde(default)]
    cached_content_token_count: u64,
}

#[async_trait]
impl LlmClient for GeminiClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat(
        &self,
        messages: &[Message],
        _options: &ChatOptions,
    ) -> Result<ChatResponse, MinionError> {
        let mut system_parts = Vec::new();
        let mut contents = Vec::new();
        for msg in messages {
            match msg.role {
                Role::System => system_parts.push(GeminiPart { text: &msg.content }),
                Role::User | Role::Tool => contents.push(GeminiContent {
                    role: Some("user"),
                    parts: vec![GeminiPart { text: &msg.content }],
                }),
                Role::Assistant => contents.push(GeminiContent {
                    role: Some("model"),
                    parts: vec![GeminiPart { text: &msg.content }],
                }),
            }
        }

        let system_instruction = (!system_parts.is_empty()).then_some(GeminiContent {
            role: None,
            parts: system_parts,
        });

        let body = GeminiRequest {
            contents,
            system_instruction,
        };
        let url = format!("{}/models/{}:generateContent", self.base_url, self.model);
        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
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

        let parsed: GeminiResponse = resp
            .json()
            .await
            .map_err(|e| MinionError::Http(e.to_string()))?;

        let responses = parsed
            .candidates
            .iter()
            .map(|c| {
                c.content
                    .parts
                    .iter()
                    .map(|p| p.text.as_str())
                    .collect::<String>()
            })
            .collect();
        let finish_reasons = parsed
            .candidates
            .iter()
            .map(|c| {
                c.finish_reason
                    .clone()
                    .unwrap_or_else(|| "STOP".to_string())
            })
            .collect();
        let usage = parsed
            .usage_metadata
            .map_or_else(Usage::default, |u| Usage {
                prompt_tokens: u.prompt_token_count,
                completion_tokens: u.candidates_token_count,
                cached_prompt_tokens: u.cached_content_token_count,
                seen_prompt_tokens: 0,
            });

        Ok(ChatResponse {
            responses,
            usage,
            finish_reasons,
        })
    }
}
