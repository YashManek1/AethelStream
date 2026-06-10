//! A5: I/O backend abstraction -- `IoBackend` trait and capability probing.
//!
//! FlowCast drives one backend at a time.  The backend selection runs at
//! startup and is not changed at runtime.

pub mod file_read;
pub mod gds;
pub mod mock;
pub mod super_shard;
pub mod uring;

use crate::Result;
use ramflow::PinnedBuffer;

/// Abstraction over the raw I/O path: io_uring, GDS, or mock.
///
/// # Thread safety
/// All methods that take `&self` are called from the completion-router thread
/// as well as the prefetch-dispatch thread.  Implementations must be
/// `Send + Sync`.
pub trait IoBackend: Send + Sync {
    /// Start background threads (e.g. io_uring CQE poller).
    ///
    /// Must be called exactly once before any `prefetch` or `write_async`.
    fn start(&mut self) -> Result<()>;

    /// Submit an asynchronous read of a shard range into `dst`.
    ///
    /// `token` is an opaque `u64` that will appear in the next `Completion`
    /// returned by `poll_completions` when the read finishes.
    ///
    /// # Errors
    /// Returns `FlowCastError::BackendIo` on submission failure.
    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()>;

    /// Submit an asynchronous write from `src` to a shard range.
    ///
    /// Returns `FlowCastError::BackendIo` on submission failure.
    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()>;

    /// Drain completed I/O operations; return the list of completions.
    ///
    /// Returns `FlowCastError::BackendIo` if the ring is in a bad state.
    fn poll_completions(&self) -> Result<Vec<Completion>>;

    /// Whether the backend is paused due to memory pressure.
    fn is_paused(&self) -> bool;

    /// Instruct the backend to pause or resume submissions.
    fn set_pause(&self, paused: bool);

    /// Advertise which optional features this backend supports.
    fn capabilities(&self) -> BackendCapabilities;

    /// Graceful shutdown: drain in-flight I/O and close resources.
    ///
    /// Returns `FlowCastError::BackendIo` if shutdown fails.
    fn shutdown(&mut self) -> Result<()>;
}

/// A completed I/O operation returned by `IoBackend::poll_completions`.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The opaque token supplied to `prefetch` or `write_async`.
    pub token: u64,

    /// Bytes transferred (>= 0) or negative errno on failure.
    pub result: i32,
}

/// Feature flags reported by each backend.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Zero-copy GPU direct storage (cuFile) is available.
    pub supports_gds: bool,

    /// Super-shard grouping of small tensors is supported.
    pub supports_super_shard: bool,

    /// Write-skip optimisation (A9) is supported.
    pub supports_write_skip: bool,

    /// Multi-GPU peer-transfer is supported.
    pub supports_multi_gpu: bool,

    /// Human-readable backend name for logging.
    pub name: &'static str,
}

/// Probe system capabilities and return the best available backend.
///
/// Priority: GDS (`gds` feature + hardware) → SuperShard (io_uring, Linux)
///           → Uring (Linux) → MockBackend (fallback / mock-cuda).
///
/// `override_name` forces a specific backend: `"gds"`, `"super-shard"`,
/// `"uring"`, or `"mock"`.
pub fn select_backend(
    shard_dir: &std::path::Path,
    num_shards: u32,
) -> Result<Box<dyn IoBackend>> {
    select_backend_with_override(shard_dir, num_shards, None)
}

/// Select backend with an optional name override (`"gds"`, `"super-shard"`, `"uring"`, `"mock"`).
pub fn select_backend_with_override(
    shard_dir: &std::path::Path,
    num_shards: u32,
    override_name: Option<&str>,
) -> Result<Box<dyn IoBackend>> {
    match override_name {
        Some("mock") => return Ok(Box::new(mock::MockBackend::new())),
        Some("gds") => {
            let backend = gds::GdsBackend::new()?;
            return Ok(Box::new(backend));
        }
        Some("uring") => {
            let backend = uring::UringBackend::new(shard_dir, num_shards, 0)?;
            return Ok(Box::new(backend));
        }
        Some("super-shard") => {
            let base = uring::UringBackend::new(shard_dir, num_shards, 0)?;
            return Ok(Box::new(super_shard::SuperShardBackend::new(
                Box::new(base),
                super_shard::SuperShardConfig::default(),
            )));
        }
        Some(other) => {
            return Err(crate::FlowCastError::Config(format!(
                "unknown backend override '{other}'"
            )));
        }
        None => {}
    }

    // Auto-detect: GDS → SuperShard/Uring → Mock
    #[cfg(feature = "gds")]
    if let Ok(backend) = gds::GdsBackend::new() {
        return Ok(Box::new(backend));
    }

    #[cfg(target_os = "linux")]
    if let Ok(base) = uring::UringBackend::new(shard_dir, num_shards, 0) {
        let ss = super_shard::SuperShardBackend::new(
            Box::new(base),
            super_shard::SuperShardConfig::default(),
        );
        return Ok(Box::new(ss));
    }

    Ok(Box::new(mock::MockBackend::new()))
}
