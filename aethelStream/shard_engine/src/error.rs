use thiserror::Error;

/// ShardEngine error types.
#[derive(Error, Debug)]
pub enum ShardEngineError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse error.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Layer index not found in registry.
    #[error("layer index {0} not found in layer registry")]
    LayerNotFound(u32),

    /// Parameter not found in shard index.
    #[error("parameter '{0}' not found in shard index")]
    ParamNotFound(String),

    /// Malformed index entry.
    #[error("malformed index entry '{0}': {1}")]
    MalformedIndex(String, String),
}

/// Result type for shard engine operations.
pub type Result<T> = std::result::Result<T, ShardEngineError>;
