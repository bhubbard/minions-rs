# minions-rs

Pure-Rust port of [HazyResearch/minions](https://github.com/HazyResearch/minions) — the local-cloud LLM collaboration protocol where an on-device small language model reads and extracts from long contexts while a frontier cloud model orchestrates reasoning, cutting cloud token costs dramatically.

Zero Python runtime required.

## Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| `minions-core` | `Usage` & cost estimator, `LlmClient` trait, UTF-8-safe chunkers (`section`, `page`, `paragraph`, `code`), Okapi BM25 retriever, resilient JSON extractor, prompt templates |
| `minions-clients` | Async HTTP clients for `OllamaClient`, `OpenAiClient` (OpenAI, Together, Groq, vLLM, `anemll-server`), `AnthropicClient`, `GeminiClient`, and `MockClient` |
| `minions-protocol` | `Minion` (iterative supervisor-worker loop with `PrivacyMode::Shielded`) and `Minions` (parallel decompose-execute-aggregate pipeline with bounded concurrency) |
| `minions-cli` | `minions` binary (`minion`, `minions`, `chunk`, `estimate`) |

## Quickstart

```bash
cargo test --workspace

# Estimate cloud-only vs Minions cost on a 200k-token context
cargo run -p minions-cli -- estimate --context-tokens 200000

# Chunk a long document locally
cargo run -p minions-cli -- chunk --file README.md --strategy paragraph --chunk-size 1500
```
