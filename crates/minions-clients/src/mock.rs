//! Deterministic mock client for offline protocol testing and benchmarks.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use minions_core::{ChatOptions, ChatResponse, LlmClient, Message, MinionError, Usage};

/// Scripted [`LlmClient`] that returns pre-configured responses in FIFO order.
#[derive(Debug, Clone)]
pub struct MockClient {
    model: String,
    queue: Arc<Mutex<VecDeque<ChatResponse>>>,
    calls: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl MockClient {
    /// Creates a new mock client with a model name and scripted responses.
    #[must_use]
    pub fn new(model: impl Into<String>, responses: Vec<ChatResponse>) -> Self {
        Self {
            model: model.into(),
            queue: Arc::new(Mutex::new(VecDeque::from(responses))),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Convenience constructor from plain text strings and synthetic token counts.
    #[must_use]
    pub fn from_texts(model: impl Into<String>, texts: Vec<&str>) -> Self {
        let responses = texts
            .into_iter()
            .map(|text| ChatResponse {
                responses: vec![text.to_string()],
                usage: Usage::new(64, 32),
                finish_reasons: vec!["stop".to_string()],
            })
            .collect();
        Self::new(model, responses)
    }

    /// Returns all message slices received by [`LlmClient::chat`].
    #[must_use]
    pub fn recorded_calls(&self) -> Vec<Vec<Message>> {
        self.calls
            .lock()
            .expect("mock calls mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl LlmClient for MockClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn chat(
        &self,
        messages: &[Message],
        _options: &ChatOptions,
    ) -> Result<ChatResponse, MinionError> {
        self.calls
            .lock()
            .expect("mock calls mutex poisoned")
            .push(messages.to_vec());

        self.queue
            .lock()
            .expect("mock queue mutex poisoned")
            .pop_front()
            .ok_or_else(|| {
                MinionError::Protocol(format!(
                    "MockClient({}) ran out of scripted responses",
                    self.model
                ))
            })
    }
}
