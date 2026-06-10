//! cuFile GDS backend — GPU-direct storage, zero host-copy reads.
//!
//! When `feature = "gds"` is inactive, falls back to `MockBackend` so the
//! rest of the codebase can always reference `GdsBackend` without `#[cfg]`.

use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
use ramflow::PinnedBuffer;

/// cuFile-based GPU-direct storage backend.
///
/// Falls back to `MockBackend` when the `gds` feature is disabled, so tests
/// and non-CUDA builds always compile without conditional imports.
pub struct GdsBackend {
    inner: GdsInner,
}

enum GdsInner {
    /// Real cuFile path (only reachable when `feature = "gds"`).
    #[allow(dead_code)]
    Real,
    /// Mock fallback: instant zero-filled completions.
    Mock(super::mock::MockBackend),
}

impl GdsBackend {
    /// Open the GDS backend.
    ///
    /// Returns `Ok(GdsBackend { inner: Mock })` when `feature = "gds"` is
    /// absent so capability probing in `select_backend` always succeeds.
    pub fn new() -> Result<Self> {
        #[cfg(feature = "gds")]
        {
            // Real cuFile initialisation would go here.
            // For now, compile-time gate is present but implementation is
            // deferred until the cuFile FFI layer is wired (Module 5).
            return Ok(GdsBackend { inner: GdsInner::Real });
        }
        #[cfg(not(feature = "gds"))]
        {
            Err(crate::FlowCastError::BackendIo(
                "GDS backend requires the 'gds' feature".to_string(),
            ))
        }
    }

    /// Create a GDS backend backed by the mock path (test helper).
    pub fn new_mock() -> Self {
        GdsBackend { inner: GdsInner::Mock(super::mock::MockBackend::new()) }
    }
}

impl Default for GdsBackend {
    fn default() -> Self {
        Self::new_mock()
    }
}

impl IoBackend for GdsBackend {
    fn start(&mut self) -> Result<()> {
        match &mut self.inner {
            GdsInner::Mock(m) => m.start(),
            GdsInner::Real => Ok(()),
        }
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        match &self.inner {
            GdsInner::Mock(m) => m.prefetch(shard_id, byte_offset, length, dst, token),
            GdsInner::Real => Err(crate::FlowCastError::BackendIo(
                "cuFile prefetch not yet implemented".to_string(),
            )),
        }
    }

    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        match &self.inner {
            GdsInner::Mock(m) => m.write_async(shard_id, byte_offset, length, src, token),
            GdsInner::Real => Err(crate::FlowCastError::BackendIo(
                "cuFile write_async not yet implemented".to_string(),
            )),
        }
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        match &self.inner {
            GdsInner::Mock(m) => m.poll_completions(),
            GdsInner::Real => Ok(Vec::new()),
        }
    }

    fn is_paused(&self) -> bool {
        match &self.inner {
            GdsInner::Mock(m) => m.is_paused(),
            GdsInner::Real => false,
        }
    }

    fn set_pause(&self, paused: bool) {
        if let GdsInner::Mock(m) = &self.inner {
            m.set_pause(paused);
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_gds: true,
            supports_super_shard: true,
            supports_write_skip: true,
            supports_multi_gpu: true,
            name: "gds",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        match &mut self.inner {
            GdsInner::Mock(m) => m.shutdown(),
            GdsInner::Real => Ok(()),
        }
    }
}
