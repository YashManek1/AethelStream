//! FileRead backend -- synchronous file-reading backend for tests.
//!
//! Reads shard content directly via `std::fs` and fills the destination
//! PinnedBuffer in-place.  Intended exclusively for `cargo test` on
//! platforms without io_uring (e.g., Windows CI).
//!
//! # Safety contract
//! The write into the PinnedBuffer via raw pointer is safe because:
//! 1. The PoolSlot was just claimed; no other owner exists.
//! 2. The DMA engine is not involved (no io_uring, no cuFile).
//! 3. The lifetime of the buffer exceeds the duration of the write
//!    (the InFlightEntry in the state machine keeps the slot alive).

use super::{BackendCapabilities, Completion, IoBackend};
use crate::{FlowCastError, Result};
use ramflow::PinnedBuffer;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// File-reading backend: fills PinnedBuffers from real shard files.
///
/// Completions are queued synchronously inside `prefetch` and drained by
/// the next call to `poll_completions`.
pub struct FileReadBackend {
    paths: Vec<PathBuf>,
    paused: AtomicBool,
    pending: Mutex<Vec<Completion>>,
}

impl FileReadBackend {
    /// Create a backend backed by the given shard file paths.
    ///
    /// `paths[shard_id]` is the file read when `prefetch(shard_id, ...)` is called.
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            paused: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
        }
    }
}

impl IoBackend for FileReadBackend {
    fn start(&mut self) -> Result<()> {
        Ok(())
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        let path = self.paths.get(shard_id as usize).ok_or_else(|| {
            FlowCastError::BackendIo(format!(
                "FileReadBackend: shard_id {shard_id} out of range (have {})",
                self.paths.len()
            ))
        })?;

        let mut file = std::fs::File::open(path).map_err(|e| {
            FlowCastError::BackendIo(format!("open {}: {e}", path.display()))
        })?;

        file.seek(SeekFrom::Start(byte_offset)).map_err(|e| {
            FlowCastError::BackendIo(format!("seek {}: {e}", path.display()))
        })?;

        let available = file
            .metadata()
            .map(|m| m.len().saturating_sub(byte_offset))
            .unwrap_or(0);

        let read_len = (length.min(available) as usize).min(dst.len());
        let mut buf = vec![0u8; read_len];
        file.read_exact(&mut buf).map_err(|e| {
            FlowCastError::BackendIo(format!("read {}: {e}", path.display()))
        })?;

        // SAFETY: dst was just claimed from the pool ring; no other thread
        // holds a reference to its backing memory.  We write exactly
        // `read_len` bytes starting at offset 0 -- within the allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), dst.as_ptr() as *mut u8, read_len);
        }

        let mut pending = self.pending.lock().map_err(|p| {
            FlowCastError::BackendIo(format!("FileReadBackend pending lock poisoned: {p}"))
        })?;
        pending.push(Completion { token, result: read_len as i32 });
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
        let mut pending = self.pending.lock().map_err(|p| {
            FlowCastError::BackendIo(format!("FileReadBackend pending lock poisoned: {p}"))
        })?;
        pending.push(Completion { token, result: length as i32 });
        Ok(())
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        let mut pending = self.pending.lock().map_err(|p| {
            FlowCastError::BackendIo(format!("FileReadBackend pending lock poisoned: {p}"))
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
            name: "file-read",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
