//! Mock I/O backend — completes all reads instantly with zero-filled buffers.
//!
//! Used by `cargo test --features mock-cuda` so no NVMe or GPU hardware is
//! required.  Every `prefetch` call records a pending completion; the next
//! `poll_completions` drains them all and reports success.

use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
use ramflow::PinnedBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// In-memory backend that simulates instant I/O completion.
pub struct MockBackend {
    paused: AtomicBool,
    pending: Mutex<Vec<Completion>>,
}

impl MockBackend {
    /// Create a new mock backend.
    pub fn new() -> Self {
        Self {
            paused: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl IoBackend for MockBackend {
    fn start(&mut self) -> Result<()> {
        Ok(())
    }

    fn prefetch(
        &self,
        _shard_id: u32,
        _byte_offset: u64,
        length: u64,
        _dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        let mut pending = self.pending.lock().map_err(|poison| {
            crate::FlowCastError::BackendIo(format!("mock pending lock poisoned: {poison}"))
        })?;
        pending.push(Completion {
            token,
            result: length as i32,
        });
        Ok(())
    }

    fn write_async(
        &self,
        _shard_id: u32,
        _byte_offset: u64,
        length: u64,
        _src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        let mut pending = self.pending.lock().map_err(|poison| {
            crate::FlowCastError::BackendIo(format!("mock pending lock poisoned: {poison}"))
        })?;
        pending.push(Completion {
            token,
            result: length as i32,
        });
        Ok(())
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        let mut pending = self.pending.lock().map_err(|poison| {
            crate::FlowCastError::BackendIo(format!("mock pending lock poisoned: {poison}"))
        })?;
        Ok(std::mem::take(&mut *pending))
    }

    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    fn set_pause(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_gds: false,
            supports_super_shard: false,
            supports_write_skip: false,
            supports_multi_gpu: false,
            name: "mock",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
