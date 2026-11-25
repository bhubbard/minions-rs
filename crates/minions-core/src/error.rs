//! Error types for `minions-core` and downstream protocol crates.

use thiserror::Error;

/// Errors returned by Minions clients, chunkers, and protocol runners.
#[derive(Debug, Error)]
pub enum MinionError {
    /// HTTP transport or client request failure.
    #[error("http request failed: {0}")]
    Http(String),

    /// Upstream API returned an error status or payload.
    #[error("api error ({status}): {message}")]
    Api {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Provider error message body.
        message: String,
    },

    /// JSON serialization or deserialization failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Regex compilation failure.
    #[error("invalid regex pattern: {0}")]
    Regex(#[from] regex::Error),

    /// Protocol failed to extract a valid structured decision or answer.
    #[error("protocol parse error: {0}")]
    Protocol(String),

    /// Configuration or input validation error.
    #[error("invalid configuration: {0}")]
    Config(String),
}
