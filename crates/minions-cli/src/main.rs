//! CLI for running Minion/Minions local-cloud LLM workflows and cost estimation.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use minions_clients::{AnthropicClient, GeminiClient, OllamaClient, OpenAiClient};
use minions_core::{
    chunk_by_code, chunk_by_paragraph, chunk_by_section, estimate_cost_usd, LlmClient,
    ModelPricing, Usage,
};
use minions_protocol::{ChunkStrategy, Minion, MinionConfig, Minions, MinionsConfig, PrivacyMode};

#[derive(Parser)]
#[command(
    name = "minions",
    about = "Pure-Rust Minions local-cloud LLM collaboration CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, ValueEnum)]
enum RemoteProvider {
    Openai,
    Anthropic,
    Gemini,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the iterative Minion (singular) protocol.
    Minion {
        #[arg(long)]
        task: String,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value = "llama3.2")]
        local_model: String,
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,
        #[arg(long, value_enum, default_value = "openai")]
        remote_provider: RemoteProvider,
        #[arg(long, default_value = "gpt-4o-mini")]
        remote_model: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: String,
        #[arg(long, default_value_t = 3)]
        max_rounds: usize,
        #[arg(long)]
        privacy_shield: bool,
    },
    /// Run the parallel Minions (plural) decompose-execute-aggregate protocol.
    Minions {
        #[arg(long)]
        task: String,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value = "llama3.2")]
        local_model: String,
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,
        #[arg(long, value_enum, default_value = "openai")]
        remote_provider: RemoteProvider,
        #[arg(long, default_value = "gpt-4o-mini")]
        remote_model: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        api_key: String,
        #[arg(long, default_value_t = 3000)]
        chunk_size: usize,
        #[arg(long)]
        bm25_top_k: Option<usize>,
    },
    /// Chunk a local file and inspect chunk counts without calling any LLM.
    Chunk {
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value = "section")]
        strategy: String,
        #[arg(long, default_value_t = 3000)]
        chunk_size: usize,
    },
    /// Estimate cloud-only vs Minions local+cloud cost savings.
    Estimate {
        #[arg(long)]
        context_tokens: u64,
        #[arg(long, default_value_t = 500)]
        remote_prompt_tokens: u64,
        #[arg(long, default_value_t = 300)]
        remote_completion_tokens: u64,
    },
}

fn build_remote(provider: RemoteProvider, model: String, api_key: String) -> Arc<dyn LlmClient> {
    match provider {
        RemoteProvider::Openai => Arc::new(OpenAiClient::new(model, api_key)),
        RemoteProvider::Anthropic => Arc::new(AnthropicClient::new(model, api_key)),
        RemoteProvider::Gemini => Arc::new(GeminiClient::new(model, api_key)),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Minion {
            task,
            file,
            local_model,
            ollama_url,
            remote_provider,
            remote_model,
            api_key,
            max_rounds,
            privacy_shield,
        } => {
            let content = fs::read_to_string(&file)
                .with_context(|| format!("reading context file {}", file.display()))?;
            let local: Arc<dyn LlmClient> =
                Arc::new(OllamaClient::with_base_url(local_model, ollama_url));
            let remote = build_remote(remote_provider, remote_model, api_key);
            let runner = Minion::new(
                local,
                remote,
                MinionConfig {
                    max_rounds,
                    privacy: if privacy_shield {
                        PrivacyMode::Shielded
                    } else {
                        PrivacyMode::Standard
                    },
                },
            );
            let report = runner.run(&task, &[content]).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::Minions {
            task,
            file,
            local_model,
            ollama_url,
            remote_provider,
            remote_model,
            api_key,
            chunk_size,
            bm25_top_k,
        } => {
            let content = fs::read_to_string(&file)
                .with_context(|| format!("reading context file {}", file.display()))?;
            let local: Arc<dyn LlmClient> =
                Arc::new(OllamaClient::with_base_url(local_model, ollama_url));
            let remote = build_remote(remote_provider, remote_model, api_key);
            let runner = Minions::new(
                local,
                remote,
                MinionsConfig {
                    chunk_strategy: ChunkStrategy::Section,
                    max_chunk_size: chunk_size,
                    chunk_overlap: 100,
                    bm25_top_k,
                    max_concurrency: 8,
                },
            );
            let report = runner.run(&task, &[content]).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::Chunk {
            file,
            strategy,
            chunk_size,
        } => {
            let content = fs::read_to_string(&file)
                .with_context(|| format!("reading file {}", file.display()))?;
            let chunks = match strategy.as_str() {
                "paragraph" => chunk_by_paragraph(&content, chunk_size, 1),
                "code" => chunk_by_code(&content, 2),
                _ => chunk_by_section(&content, chunk_size, 100),
            };
            println!("Produced {} chunks (strategy={strategy})", chunks.len());
        }
        Commands::Estimate {
            context_tokens,
            remote_prompt_tokens,
            remote_completion_tokens,
        } => {
            let pricing = ModelPricing::frontier_default();
            let cloud_only = estimate_cost_usd(
                Usage::new(
                    context_tokens + remote_prompt_tokens,
                    remote_completion_tokens,
                ),
                pricing,
            );
            let with_minions = estimate_cost_usd(
                Usage::new(remote_prompt_tokens, remote_completion_tokens),
                pricing,
            );
            let savings_pct = if cloud_only > 0.0 {
                (1.0 - with_minions / cloud_only) * 100.0
            } else {
                0.0
            };
            println!(
                "Cloud-only: ${cloud_only:.4} | Minions: ${with_minions:.4} | Savings: {savings_pct:.1}%"
            );
        }
    }
    Ok(())
}
