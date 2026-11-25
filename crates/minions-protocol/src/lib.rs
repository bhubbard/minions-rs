//! Execution protocols for local-cloud LLM collaboration.
//!
//! - [`minion::Minion`]: iterative supervisor/worker conversation loop where the
//!   local model holds the full context and answers focused remote queries.
//! - [`minions::Minions`]: decompose-execute-aggregate protocol that chunks
//!   long contexts, dispatches parallel local worker jobs, filters relevant
//!   findings, and synthesizes the final answer remotely.

pub mod minion;
pub mod minions;

pub use minion::{Minion, MinionConfig, MinionReport, PrivacyMode};
pub use minions::{
    ChunkStrategy, Job, JobManifest, JobOutput, Minions, MinionsConfig, MinionsReport,
};
