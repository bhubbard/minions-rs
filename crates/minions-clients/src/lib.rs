//! Asynchronous LLM clients for local and frontier cloud providers.
//!
//! Includes [`OllamaClient`] (local on-device), [`OpenAiClient`] (OpenAI and
//! compatible APIs such as Together, Groq, DeepSeek, vLLM, and `anemll-server`),
//! [`AnthropicClient`], [`GeminiClient`], and [`MockClient`] for deterministic testing.

pub mod anthropic;
pub mod gemini;
pub mod mock;
pub mod ollama;
pub mod openai;

pub use anthropic::AnthropicClient;
pub use gemini::GeminiClient;
pub use mock::MockClient;
pub use ollama::OllamaClient;
pub use openai::OpenAiClient;
