//! Prompt templates for the Minion (singular) and Minions (plural) protocols.

/// System prompt for the on-device worker in the Minion (singular) protocol.
pub const WORKER_SYSTEM_PROMPT: &str = "\
You are a helpful local assistant with exclusive access to the user's context documents. \
Answer questions from the remote supervisor concisely, accurately, and strictly grounded in the context. \
Do not fabricate facts that are not present in the context.";

/// Privacy-shielded worker instruction preventing raw PII/document leakage.
pub const WORKER_PRIVACY_SHIELD_PROMPT: &str = "\
PRIVACY SHIELD ENABLED: Never copy verbatim sensitive PII, account numbers, SSNs, or raw document dumps. \
Summarize and answer only the specific question asked using minimal necessary facts.";

/// Initial supervisor prompt for the Minion (singular) protocol.
pub const SUPERVISOR_INITIAL_PROMPT: &str = "\
You are a frontier reasoning supervisor collaborating with a small local assistant that holds the full context. \
You do NOT have direct access to the context. Break down the task and ask focused questions. \
Respond with JSON: {\"decision\": \"request_additional_info\" | \"provide_final_answer\", \"message\": \"...\", \"answer\": \"...\"}.";

/// Follow-up supervisor prompt for later rounds of Minion (singular).
pub const SUPERVISOR_CONVERSATION_PROMPT: &str = "\
Review the local worker's findings. If sufficient to answer the user's task with high confidence, \
respond with JSON {\"decision\": \"provide_final_answer\", \"answer\": \"...\"}. \
Otherwise respond with JSON {\"decision\": \"request_additional_info\", \"message\": \"...\"}.";

/// Final synthesis prompt when `max_rounds` is reached in Minion (singular).
pub const SUPERVISOR_FINAL_PROMPT: &str = "\
You have reached the maximum number of rounds. Synthesize the best possible final answer \
from the conversation so far and respond with JSON {\"decision\": \"provide_final_answer\", \"answer\": \"...\"}.";

/// Worker prompt template for parallel chunk jobs in Minions (plural).
pub const WORKER_PROMPT_SHORT: &str = "\
Read the following document chunk and perform the assigned sub-task. \
Return ONLY valid JSON: {\"explanation\": \"...\", \"citation\": \"...\" or null, \"answer\": \"...\" or null}. \
Set \"answer\" to null if the chunk contains no relevant information.";

/// Remote supervisor decomposition prompt for Minions (plural).
pub const DECOMPOSE_TASK_PROMPT: &str = "\
You are a frontier supervisor orchestrating parallel local workers over document chunks. \
Given the user task and document metadata, return JSON: \
{\"sub_tasks\": [\"focused instruction 1\", ...], \"advice\": \"concise extraction guidance\"}.";

/// Remote supervisor synthesis prompt for Minions (plural).
pub const REMOTE_SYNTHESIS_FINAL: &str = "\
Synthesize a complete, accurate, well-cited final answer from the aggregated local worker outputs. \
Return JSON: {\"explanation\": \"...\", \"answer\": \"...\"}.";
