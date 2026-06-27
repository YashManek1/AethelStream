#![deny(clippy::unwrap_used, clippy::panic)]

//! Shard Engine: Rust runtime loader for quantized model shards.

pub mod error;
pub mod index;
pub mod loader;
pub mod nf4;

pub use error::{Result, ShardEngineError};
pub use index::{IndexStore, LayerRegistry, ShardIndex, TensorInfo};
pub use loader::{LayerBuffer, ShardLoader, TensorBuffer};

#[cfg(feature = "python-ffi")]
pub mod ffi;

#[cfg(feature = "python-ffi")]
pub use ffi::PyShardLoader;
