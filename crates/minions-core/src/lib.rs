//! Core primitives for the Minions local-cloud LLM collaboration protocol.
//!
//! This crate provides token usage accounting, the async [`LlmClient`] trait,
//! document chunkers, an Okapi BM25 retriever, resilient JSON extraction, and
//! prompt templates ported from HazyResearch/minions.

pub mod bm25;
pub mod chunking;
pub mod client;
pub mod error;
pub mod json_extract;
pub mod prompts;
pub mod usage;

pub use bm25::{bm25_retrieve_top_k_chunks, ScoredChunk};
pub use chunking::{
    chunk_by_code, chunk_by_page, chunk_by_paragraph, chunk_by_section, chunk_sentences,
};
pub use client::{ChatOptions, ChatResponse, LlmClient, Message, Role};
pub use error::MinionError;
pub use json_extract::extract_json;
pub use usage::{estimate_cost_usd, ModelPricing, Usage};
