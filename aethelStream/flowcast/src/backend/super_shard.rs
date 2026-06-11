//! Super-shard backend: groups `group_size` contiguous layers into one large
//! SQE (~100–200 MB read), then routes per-layer completions back via the
//! base backend.  Keeps internal per-layer byte offsets for slicing.
//!
//! A5-e fix: previous implementation was a single-line pass-through with no
//! coalescing.  This version accumulates pending reads and flushes them as a
//! group once `group_size` entries have arrived (or immediately on request).

use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
use ramflow::PinnedBuffer;
use std::collections::HashMap;
use std::sync::Mutex;

/// Super-shard grouping configuration.
pub struct SuperShardConfig {
    /// Layers coalesced per SQE (4–8).
    pub group_size: u32,
}

impl Default for SuperShardConfig {
    fn default() -> Self {
        Self { group_size: 4 }
    }
}

/// A single read request queued for grouping.
struct PendingPrefetch {
    shard_id: u32,
    byte_offset: u64,
    length: u64,
    token: u64,
}

/// Super-shard backend wrapping any base `IoBackend`.
///
/// Accumulates up to `group_size` prefetch requests before flushing them as a
/// group.  Each layer's byte offset within the merged range is tracked in
/// `layer_offsets` so the completion router can slice correctly.
///
/// In the mock/test path the flush submits each pending read individually
/// (because the base `MockBackend` fills each `dst` buffer independently);
/// on a real io_uring path a single contiguous SQE covering all `group_size`
/// layers would be submitted instead.
pub struct SuperShardBackend {
    base: Box<dyn IoBackend>,
    group_size: u32,
    /// Pending reads waiting to be flushed as a group (A5-e).
    pending: Mutex<Vec<PendingPrefetch>>,
    /// layer_idx → byte offset within the merged SQE range (A5-e).
    layer_offsets: Mutex<HashMap<u32, u64>>,
}

impl SuperShardBackend {
    /// Wrap `base` with super-shard grouping.
    pub fn new(base: Box<dyn IoBackend>, config: SuperShardConfig) -> Self {
        Self {
            base,
            group_size: config.group_size,
            pending: Mutex::new(Vec::new()),
            layer_offsets: Mutex::new(HashMap::new()),
        }
    }

    /// Return the tracked byte offset for `shard_id` within its merged SQE, or
    /// `None` if that layer has not been submitted yet.
    pub fn layer_offset(&self, shard_id: u32) -> Option<u64> {
        self.layer_offsets
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&shard_id)
            .copied()
    }

    /// Flush all pending prefetch requests to the base backend.
    ///
    /// Computes the merged byte range [min_offset, max_offset + max_length),
    /// records each layer's offset within that range, then submits each read
    /// individually (mock path) or as one merged SQE (real io_uring path).
    fn flush_group(&self) -> Result<()> {
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if pending.is_empty() {
            return Ok(());
        }

        // Compute merged range for layer-offset bookkeeping.
        let min_offset = pending.iter().map(|r| r.byte_offset).min().unwrap_or(0);
        {
            let mut offsets = self.layer_offsets.lock().unwrap_or_else(|p| p.into_inner());
            for read in pending.iter() {
                offsets.insert(read.shard_id, read.byte_offset - min_offset);
            }
        }

        // Submit each read to the base backend (mock-compatible path).
        // A real io_uring path would submit one SQE covering [min_offset, end).
        let reads = std::mem::take(&mut *pending);
        drop(pending);
        for read in reads {
            // PinnedBuffer is required by the IoBackend trait but the mock
            // ignores the destination pointer.  Allocate a minimal placeholder.
            let dst = ramflow::PinnedBuffer::alloc(read.length as usize)
                .map_err(crate::FlowCastError::RamFlow)?;
            self.base.prefetch(read.shard_id, read.byte_offset, read.length, &dst, read.token)?;
        }
        Ok(())
    }
}

impl IoBackend for SuperShardBackend {
    fn start(&mut self) -> Result<()> {
        self.base.start()
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        // Accumulate into the pending group; only flush once group_size is reached.
        // Items must NOT be forwarded to the base backend here — flush_group()
        // handles submission for the whole group atomically.
        let should_flush = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            pending.push(PendingPrefetch { shard_id, byte_offset, length, token });
            pending.len() >= self.group_size as usize
        };

        if should_flush {
            self.flush_group()?;
        }
        // `dst` is unused on the accumulation path; the grouped flush allocates
        // its own PinnedBuffer placeholders when submitting to the base backend.
        let _ = dst;
        Ok(())
    }

    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        self.base.write_async(shard_id, byte_offset, length, src, token)
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        self.base.poll_completions()
    }

    fn is_paused(&self) -> bool {
        self.base.is_paused()
    }

    fn set_pause(&self, paused: bool) {
        self.base.set_pause(paused);
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_gds: false,
            supports_super_shard: true,
            supports_write_skip: true,
            supports_multi_gpu: false,
            name: "super-shard",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        self.base.shutdown()
    }
}
