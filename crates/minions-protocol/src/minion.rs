//! Iterative Minion (singular) supervisor-worker protocol.

use std::sync::Arc;
use std::time::Instant;

use minions_core::prompts::{
    SUPERVISOR_CONVERSATION_PROMPT, SUPERVISOR_FINAL_PROMPT, SUPERVISOR_INITIAL_PROMPT,
    WORKER_PRIVACY_SHIELD_PROMPT, WORKER_SYSTEM_PROMPT,
};
use minions_core::{extract_json, ChatOptions, LlmClient, Message, MinionError, Usage};
use serde::{Deserialize, Serialize};

/// Controls whether the local worker enforces privacy-shielded summarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrivacyMode {
    /// Standard local-remote context sharing.
    #[default]
    Standard,
    /// Instructs the local worker not to leak raw PII or verbatim document dumps.
    Shielded,
}

/// Configuration for [`Minion`].
#[derive(Debug, Clone)]
pub struct MinionConfig {
    /// Maximum supervisor-worker dialogue rounds before forced synthesis.
    pub max_rounds: usize,
    /// Privacy mode for local worker responses.
    pub privacy: PrivacyMode,
}

impl Default for MinionConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            privacy: PrivacyMode::Standard,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SupervisorDecision {
    decision: String,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    answer: Option<String>,
}

/// Result returned from a [`Minion::run`] execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinionReport {
    /// Final synthesized answer.
    pub final_answer: String,
    /// Number of rounds executed.
    pub rounds: usize,
    /// Total token usage by the local model.
    pub local_usage: Usage,
    /// Total token usage by the remote frontier model.
    pub remote_usage: Usage,
    /// Elapsed wall-clock milliseconds for local calls.
    pub local_time_ms: u64,
    /// Elapsed wall-clock milliseconds for remote calls.
    pub remote_time_ms: u64,
}

/// Iterative local-remote Minion protocol runner.
pub struct Minion {
    local: Arc<dyn LlmClient>,
    remote: Arc<dyn LlmClient>,
    config: MinionConfig,
}

impl Minion {
    /// Creates a new [`Minion`] runner with local and remote [`LlmClient`]s.
    #[must_use]
    pub fn new(
        local: Arc<dyn LlmClient>,
        remote: Arc<dyn LlmClient>,
        config: MinionConfig,
    ) -> Self {
        Self {
            local,
            remote,
            config,
        }
    }

    /// Runs the iterative Minion protocol over `task` and `context`.
    ///
    /// # Errors
    /// Returns [`MinionError`] if any LLM request fails or returns unparseable
    /// supervisor JSON.
    pub async fn run(&self, task: &str, context: &[String]) -> Result<MinionReport, MinionError> {
        let joined_context = context.join("\n\n");
        let mut local_usage = Usage::default();
        let mut remote_usage = Usage::default();
        let mut local_time_ms = 0u64;
        let mut remote_time_ms = 0u64;

        let mut worker_system = format!("{WORKER_SYSTEM_PROMPT}\n\nContext:\n{joined_context}");
        if self.config.privacy == PrivacyMode::Shielded {
            worker_system.push_str("\n\n");
            worker_system.push_str(WORKER_PRIVACY_SHIELD_PROMPT);
        }

        let mut supervisor_msgs = vec![
            Message::system(SUPERVISOR_INITIAL_PROMPT),
            Message::user(format!(
                "Task: {task}\n\nContext length: {} characters across {} section(s).",
                joined_context.len(),
                context.len()
            )),
        ];
        let mut worker_msgs = vec![Message::system(worker_system)];

        let max_rounds = self.config.max_rounds.max(1);
        let opts = ChatOptions::default();

        for round in 1..=max_rounds {
            let t0 = Instant::now();
            let remote_resp = self.remote.chat(&supervisor_msgs, &opts).await?;
            remote_time_ms += t0.elapsed().as_millis() as u64;
            remote_usage += remote_resp.usage;

            let raw_supervisor = remote_resp.first_text().unwrap_or_default();
            supervisor_msgs.push(Message::assistant(raw_supervisor));
            let parsed: SupervisorDecision = extract_json(raw_supervisor)?;

            if parsed.decision == "provide_final_answer" {
                let answer = parsed.answer.unwrap_or_default();
                return Ok(MinionReport {
                    final_answer: answer,
                    rounds: round,
                    local_usage,
                    remote_usage,
                    local_time_ms,
                    remote_time_ms,
                });
            }

            let question = parsed.message.unwrap_or_else(|| task.to_string());
            worker_msgs.push(Message::user(&question));

            let t1 = Instant::now();
            let local_resp = self.local.chat(&worker_msgs, &opts).await?;
            local_time_ms += t1.elapsed().as_millis() as u64;
            local_usage += local_resp.usage;

            let worker_reply = local_resp.first_text().unwrap_or_default();
            worker_msgs.push(Message::assistant(worker_reply));

            let next_prompt = if round == max_rounds {
                format!("Local worker response:\n{worker_reply}\n\n{SUPERVISOR_FINAL_PROMPT}")
            } else {
                format!(
                    "Local worker response:\n{worker_reply}\n\n{SUPERVISOR_CONVERSATION_PROMPT}"
                )
            };
            supervisor_msgs.push(Message::user(next_prompt));
        }

        let t0 = Instant::now();
        let final_resp = self.remote.chat(&supervisor_msgs, &opts).await?;
        remote_time_ms += t0.elapsed().as_millis() as u64;
        remote_usage += final_resp.usage;

        let raw_final = final_resp.first_text().unwrap_or_default();
        let parsed: SupervisorDecision = extract_json(raw_final)?;

        Ok(MinionReport {
            final_answer: parsed.answer.unwrap_or_else(|| raw_final.to_string()),
            rounds: max_rounds,
            local_usage,
            remote_usage,
            local_time_ms,
            remote_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minions_clients::MockClient;

    #[tokio::test]
    async fn minion_completes_two_round_loop() {
        let local = Arc::new(MockClient::from_texts(
            "local-3b",
            vec!["The invoice total is $420.00 due Nov 30."],
        ));
        let remote = Arc::new(MockClient::from_texts(
            "frontier-cloud",
            vec![
                r#"{"decision": "request_additional_info", "message": "What is the invoice total and due date?"}"#,
                r#"{"decision": "provide_final_answer", "answer": "The invoice total is $420.00, due Nov 30."}"#,
            ],
        ));

        let runner = Minion::new(local, remote, MinionConfig::default());
        let report = runner
            .run(
                "Find the invoice total and due date",
                &["Invoice #1042 ... Total: $420.00 ... Due: Nov 30".to_string()],
            )
            .await
            .expect("minion protocol succeeds");

        assert_eq!(
            report.final_answer,
            "The invoice total is $420.00, due Nov 30."
        );
        assert_eq!(report.rounds, 2);
        assert!(report.local_usage.total_tokens() > 0);
        assert!(report.remote_usage.total_tokens() > 0);
    }
}
