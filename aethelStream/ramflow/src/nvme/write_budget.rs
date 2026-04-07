// src/nvme/write_budget.rs — SSD write-budget manager (Idea 4)
//
// Feature-gated: #[cfg(feature = "ssd-wear")]
//
// When `ssd-wear` is INACTIVE: WriteBudgetManager is a transparent no-op —
//   consume() always returns Ok(()), remaining() always returns u64::MAX,
//   enqueue_write() passes through immediately.
//
// When `ssd-wear` is ACTIVE:
//   - Reads SMART "Data Units Written" at startup via ioctl on /dev/nvme0.
//   - Tracks cumulative bytes written throughout the run.
//   - Switches WriteStrategy automatically as budget approaches limit.
//   - Batches 4 writes before submitting to the NVMe ring (write leveling).
//   - Delta-compression: stores layerN.delta.zstd alongside original shard.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// WriteStrategy
// ---------------------------------------------------------------------------

/// Controls how updated layer weights are persisted to NVMe.
///
/// Selected automatically by [`WriteBudgetManager`] based on remaining budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStrategy {
    /// Write the full updated tensor shard.  Highest write amplification.
    Full,
    /// Compute `delta = updated - original`, compress with zstd, write delta.
    /// Module 1's ShardLoader detects `.delta.zstd` and applies it on load.
    /// Typical compression ratio: 8–15× at standard learning rates.
    DeltaCompress,
    /// Accumulate `batch_size` layers before submitting a write batch.
    /// Allows the NVMe controller to reorder writes for wear-level optimisation.
    Deferred { batch_size: u32 },
}

// ---------------------------------------------------------------------------
// WriteBudgetManager
// ---------------------------------------------------------------------------

/// Tracks cumulative NVMe bytes written and enforces a user-configured TBW cap.
///
/// # Feature gate
/// Only meaningful when the `ssd-wear` Cargo feature is active.
/// When disabled, all methods are no-ops and `consume` always returns `Ok(())`.
///
/// # Sprint 0 contract
/// Compiles in both feature configurations; `ssd-wear`-active methods panic.
pub struct WriteBudgetManager {
    _device_path: PathBuf,
    _units_at_start: u64,
    _budget_units: u64,
    _units_written: AtomicU64,
    _strategy: WriteStrategy,
}

impl WriteBudgetManager {
    /// Create a manager with a lifetime TBW budget of `total_bytes` bytes.
    ///
    /// When `ssd-wear` is inactive, `total_bytes` is ignored.
    #[allow(unused_variables)]
    pub fn new(device_path: PathBuf, total_bytes: u64) -> Self {
        #[cfg(feature = "ssd-wear")]
        {
            unimplemented!("WriteBudgetManager::new — Sprint 0 stub (ssd-wear active)")
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            WriteBudgetManager {
                _device_path: device_path,
                _units_at_start: 0,
                _budget_units: u64::MAX,
                _units_written: AtomicU64::new(0),
                _strategy: WriteStrategy::Full,
            }
        }
    }

    /// Queue a write of `bytes` against the budget.
    ///
    /// Returns [`crate::error::RamFlowError::WearBudgetExceeded`] ONLY after
    /// the current training step completes (never aborts mid-step).
    #[allow(unused_variables)]
    pub fn consume(&self, bytes: u64) -> crate::Result<()> {
        #[cfg(feature = "ssd-wear")]
        {
            unimplemented!("WriteBudgetManager::consume — Sprint 0 stub (ssd-wear active)")
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            Ok(())
        }
    }

    /// Enqueue a layer write.  The manager decides when to flush based on
    /// batch size, budget proximity, or end-of-training.
    #[allow(unused_variables)]
    pub fn enqueue_write(&self, shard_id: u32, buf: &crate::allocator::PinnedBuffer) -> crate::Result<()> {
        #[cfg(feature = "ssd-wear")]
        {
            unimplemented!("WriteBudgetManager::enqueue_write — Sprint 0 stub (ssd-wear active)")
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            Ok(())
        }
    }

    /// Current write strategy (changes automatically as budget is consumed).
    pub fn strategy(&self) -> WriteStrategy {
        self._strategy
    }

    /// Bytes remaining in the write budget (u64::MAX when ssd-wear is off).
    pub fn remaining(&self) -> u64 {
        #[cfg(feature = "ssd-wear")]
        {
            unimplemented!("WriteBudgetManager::remaining — Sprint 0 stub")
        }
        #[cfg(not(feature = "ssd-wear"))]
        {
            u64::MAX
        }
    }
}
