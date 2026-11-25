//! Parallel decompose-execute-aggregate Minions (plural) protocol.

use std::sync::Arc;
use std::time::Instant;

use minions_core::prompts::{DECOMPOSE_TASK_PROMPT, REMOTE_SYNTHESIS_FINAL, WORKER_PROMPT_SHORT};
use minions_core::{
    bm25_retrieve_top_k_chunks, chunk_by_code, chunk_by_paragraph, chunk_by_section, extract_json,
    ChatOptions, LlmClient, Message, MinionError, Usage,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Strategy for splitting long contexts into local worker chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkStrategy {
    /// Fixed-size character sections with overlap.
    #[default]
    Section,
    /// Paragraph-aware chunking with sentence fallback.
    Paragraph,
    /// Function/class-level source code chunking.
    Code,
}

/// Configuration for [`Minions`].
#[derive(Debug, Clone)]
pub struct MinionsConfig {
    /// Chunking strategy for context documents.
    pub chunk_strategy: ChunkStrategy,
    /// Maximum character length per chunk.
    pub max_chunk_size: usize,
    /// Character overlap between adjacent section chunks.
    pub chunk_overlap: usize,
    /// Optional BM25 top-k pre-filter before dispatching local workers.
    pub bm25_top_k: Option<usize>,
    /// Maximum concurrent local worker requests.
    pub max_concurrency: usize,
}

impl Default for MinionsConfig {
    fn default() -> Self {
        Self {
            chunk_strategy: ChunkStrategy::Section,
            max_chunk_size: 3000,
            chunk_overlap: 100,
            bm25_top_k: None,
            max_concurrency: 8,
        }
    }
}

/// Work specification dispatched to a single local worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobManifest {
    /// Document chunk text.
    pub chunk: String,
    /// Focused sub-task instruction.
    pub task: String,
    /// Additional extraction guidance from the supervisor.
    pub advice: String,
    /// Zero-based chunk index.
    pub chunk_id: usize,
    /// Zero-based sub-task index.
    pub task_id: usize,
    /// Global job identifier.
    pub job_id: usize,
}

/// Structured output returned by a local worker for one [`JobManifest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobOutput {
    /// Brief reasoning or justification.
    #[serde(default)]
    pub explanation: String,
    /// Supporting quote or reference, if found.
    #[serde(default)]
    pub citation: Option<String>,
    /// Extracted answer, or `None` if the chunk was irrelevant.
    #[serde(default)]
    pub answer: Option<String>,
}

/// Completed local worker job with raw output and inclusion flag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Input manifest.
    pub manifest: JobManifest,
    /// Parsed structured output.
    pub output: JobOutput,
    /// Raw text response from the local model.
    pub sample: String,
    /// Whether this job produced a non-empty answer.
    pub include: bool,
}

#[derive(Debug, Deserialize)]
struct DecomposePlan {
    sub_tasks: Vec<String>,
    #[serde(default)]
    advice: String,
}

#[derive(Debug, Deserialize)]
struct SynthesisResponse {
    #[serde(default)]
    explanation: String,
    answer: String,
}

/// Result returned from [`Minions::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinionsReport {
    /// Final synthesized answer from the frontier supervisor.
    pub final_answer: String,
    /// Supervisor synthesis rationale.
    pub synthesis_explanation: String,
    /// All executed local worker jobs.
    pub jobs: Vec<Job>,
    /// Number of chunks processed.
    pub chunks_processed: usize,
    /// Total local model token usage.
    pub local_usage: Usage,
    /// Total remote frontier model token usage.
    pub remote_usage: Usage,
    /// Total wall-clock runtime in milliseconds.
    pub elapsed_ms: u64,
}

/// Parallel decompose-execute-aggregate Minions protocol runner.
pub struct Minions {
    local: Arc<dyn LlmClient>,
    remote: Arc<dyn LlmClient>,
    config: MinionsConfig,
}

impl Minions {
    /// Creates a new [`Minions`] runner.
    #[must_use]
    pub fn new(
        local: Arc<dyn LlmClient>,
        remote: Arc<dyn LlmClient>,
        config: MinionsConfig,
    ) -> Self {
        Self {
            local,
            remote,
            config,
        }
    }

    fn build_chunks(&self, task: &str, context: &[String]) -> Vec<String> {
        let mut all_chunks = Vec::new();
        for doc in context {
            let doc_chunks = match self.config.chunk_strategy {
                ChunkStrategy::Section => {
                    chunk_by_section(doc, self.config.max_chunk_size, self.config.chunk_overlap)
                }
                ChunkStrategy::Paragraph => chunk_by_paragraph(doc, self.config.max_chunk_size, 1),
                ChunkStrategy::Code => chunk_by_code(doc, 2),
            };
            all_chunks.extend(doc_chunks);
        }

        if let Some(top_k) = self.config.bm25_top_k {
            bm25_retrieve_top_k_chunks(&all_chunks, task, top_k)
                .into_iter()
                .map(|sc| sc.chunk)
                .collect()
        } else {
            all_chunks
        }
    }

    /// Executes the Minions decompose-execute-aggregate pipeline.
    ///
    /// # Errors
    /// Returns [`MinionError`] if remote decomposition/synthesis fails or if
    /// local worker execution encounters an unrecoverable error.
    pub async fn run(&self, task: &str, context: &[String]) -> Result<MinionsReport, MinionError> {
        let start = Instant::now();
        let mut local_usage = Usage::default();
        let mut remote_usage = Usage::default();
        let opts = ChatOptions::default();

        let chunks = self.build_chunks(task, context);
        let chunks_processed = chunks.len();

        // Step 1: Remote supervisor decomposes task into focused sub-tasks.
        let decompose_msgs = vec![
            Message::system(DECOMPOSE_TASK_PROMPT),
            Message::user(format!(
                "Task: {task}\nTotal chunks: {chunks_processed}\nTotal chars: {}",
                context.iter().map(String::len).sum::<usize>()
            )),
        ];
        let plan_resp = self.remote.chat(&decompose_msgs, &opts).await?;
        remote_usage += plan_resp.usage;
        let plan: DecomposePlan = extract_json(plan_resp.first_text().unwrap_or_default())?;
        let sub_tasks = if plan.sub_tasks.is_empty() {
            vec![task.to_string()]
        } else {
            plan.sub_tasks
        };

        // Step 2: Build job manifests (chunk x sub_task).
        let mut manifests = Vec::new();
        let mut job_id = 0usize;
        for (chunk_id, chunk) in chunks.into_iter().enumerate() {
            for (task_id, sub_task) in sub_tasks.iter().enumerate() {
                manifests.push(JobManifest {
                    chunk: chunk.clone(),
                    task: sub_task.clone(),
                    advice: plan.advice.clone(),
                    chunk_id,
                    task_id,
                    job_id,
                });
                job_id += 1;
            }
        }

        // Step 3: Execute local worker jobs in parallel with bounded concurrency.
        let sem = Arc::new(Semaphore::new(self.config.max_concurrency.max(1)));
        let mut join_set = JoinSet::new();
        for manifest in manifests {
            let local = Arc::clone(&self.local);
            let permit = Arc::clone(&sem);
            join_set.spawn(async move {
                let _guard = permit
                    .acquire_owned()
                    .await
                    .map_err(|e| MinionError::Protocol(format!("semaphore closed: {e}")))?;
                let prompt = format!(
                    "{WORKER_PROMPT_SHORT}\n\nSub-task: {}\nAdvice: {}\n\nChunk:\n{}",
                    manifest.task, manifest.advice, manifest.chunk
                );
                let resp = local
                    .chat(&[Message::user(prompt)], &ChatOptions::default())
                    .await?;
                let sample = resp.first_text().unwrap_or_default().to_string();
                let output: JobOutput = extract_json(&sample).unwrap_or(JobOutput {
                    explanation: "unstructured response".to_string(),
                    citation: None,
                    answer: (!sample.trim().is_empty()).then(|| sample.clone()),
                });
                let include = output
                    .answer
                    .as_deref()
                    .is_some_and(|a| !a.trim().is_empty() && !a.eq_ignore_ascii_case("none"));
                Ok::<(Job, Usage), MinionError>((
                    Job {
                        manifest,
                        output,
                        sample,
                        include,
                    },
                    resp.usage,
                ))
            });
        }

        let mut jobs = Vec::new();
        while let Some(joined) = join_set.join_next().await {
            let (job, usage) = joined
                .map_err(|e| MinionError::Protocol(format!("worker task panicked: {e}")))??;
            local_usage += usage;
            jobs.push(job);
        }
        jobs.sort_by_key(|j| j.manifest.job_id);

        // Step 4: Filter relevant findings and synthesize remotely.
        let aggregated: String = jobs
            .iter()
            .filter(|j| j.include)
            .map(|j| {
                format!(
                    "[Chunk {} | Task {}]\nAnswer: {}\nCitation: {}\nExplanation: {}",
                    j.manifest.chunk_id,
                    j.manifest.task_id,
                    j.output.answer.as_deref().unwrap_or(""),
                    j.output.citation.as_deref().unwrap_or("none"),
                    j.output.explanation
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let synth_msgs = vec![
            Message::system(REMOTE_SYNTHESIS_FINAL),
            Message::user(format!(
                "Original Task: {task}\n\nRelevant Local Worker Findings:\n{aggregated}"
            )),
        ];
        let synth_resp = self.remote.chat(&synth_msgs, &opts).await?;
        remote_usage += synth_resp.usage;
        let synth: SynthesisResponse = extract_json(synth_resp.first_text().unwrap_or_default())?;

        Ok(MinionsReport {
            final_answer: synth.answer,
            synthesis_explanation: synth.explanation,
            jobs,
            chunks_processed,
            local_usage,
            remote_usage,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use minions_clients::MockClient;

    #[tokio::test]
    async fn minions_parallel_decompose_execute_synthesize() {
        let local = Arc::new(MockClient::from_texts(
            "local-3b",
            vec![
                r#"{"explanation": "Found Q1", "citation": "Q1: $1.2M", "answer": "$1.2M"}"#,
                r#"{"explanation": "Nothing here", "citation": null, "answer": null}"#,
            ],
        ));
        let remote = Arc::new(MockClient::from_texts(
            "frontier-cloud",
            vec![
                r#"{"sub_tasks": ["Extract Q1 revenue"], "advice": "Look for dollar figures."}"#,
                r#"{"explanation": "Synthesized from Chunk 0", "answer": "Q1 revenue was $1.2M."}"#,
            ],
        ));

        let runner = Minions::new(
            local,
            remote,
            MinionsConfig {
                max_chunk_size: 60,
                chunk_overlap: 0,
                ..MinionsConfig::default()
            },
        );
        let docs = vec![
            "Q1 revenue reached $1.2M in March.     ".to_string(),
            "Unrelated appendix notes about office supplies.".to_string(),
        ];
        let report = runner
            .run("What was Q1 revenue?", &docs)
            .await
            .expect("minions protocol succeeds");

        assert_eq!(report.final_answer, "Q1 revenue was $1.2M.");
        assert_eq!(report.chunks_processed, 2);
        assert_eq!(report.jobs.iter().filter(|j| j.include).count(), 1);
    }
}
