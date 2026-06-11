//! Mock I/O backend — completes all reads instantly with zero-filled buffers.
//!
//! Used by `cargo test --features mock-cuda` so no NVMe or GPU hardware is
//! required.  Every `prefetch` call records a pending completion; the next
//! `poll_completions` drains them all and reports success.
//!
//! A4-e fix: `write_async` stores the written bytes by `shard_id` so tests
//! can verify byte-identical round-trips via `last_written_bytes`.

use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
use ramflow::PinnedBuffer;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// In-memory backend that simulates instant I/O completion.
pub struct MockBackend {
    paused: AtomicBool,
    pending: Mutex<Vec<Completion>>,
    /// Stores the last bytes written per shard_id (A4-e fix).
    written: Mutex<HashMap<u32, Vec<u8>>>,
}

impl MockBackend {
    /// Create a new mock backend.
    pub fn new() -> Self {
        Self {
            paused: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
            written: Mutex::new(HashMap::new()),
        }
    }

    /// Return a copy of the last bytes written for `shard_id`, or `None` if
    /// no write has been submitted for that shard.
    ///
    /// Used by tests to verify byte-identical round-trips (A4-e / INT-e).
    pub fn last_written_bytes(&self, shard_id: u32) -> Option<Vec<u8>> {
        self.written
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&shard_id)
            .cloned()
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
        pending.push(Completion { token, result: length as i32 });
        Ok(())
    }

    fn write_async(
        &self,
        shard_id: u32,
        _byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        // Store the written bytes for later verification (A4-e fix).
        {
            let mut written = self.written.lock().map_err(|poison| {
                crate::FlowCastError::BackendIo(format!("mock written lock poisoned: {poison}"))
            })?;
            written.insert(shard_id, src.as_slice().to_vec());
        }
        let mut pending = self.pending.lock().map_err(|poison| {
            crate::FlowCastError::BackendIo(format!("mock pending lock poisoned: {poison}"))
        })?;
        pending.push(Completion { token, result: length as i32 });
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
