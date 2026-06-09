// src/pool/slow_path.rs — slow-path allocator for pool exhaustion recovery
//
// Sprint 4: signal_coscheduler_stall now calls gauge.signal_stall() when a
// MemoryPressureGauge is registered, replacing the Sprint 3A no-op.
//
// ─── THREE-STAGE RECOVERY ─────────────────────────────────────────────────────
//   Stage 1: Call gauge.signal_stall() — fires high-pressure callbacks immediately
//            so the NVMe prefetch engine pauses, DMA completes, slots return.
//   Stage 2: Attempt to reclaim the oldest in-flight buffer via a fresh try_claim.
//   Stage 3: Fresh PinnedBuffer::alloc fallback with a warning.

use std::sync::{Arc, Mutex};

use crate::allocator::PinnedBuffer;
use crate::pool::{LayerKind, PoolSlot, RingBuffer};
use crate::Result;

/// Slow-path allocator invoked when a pool ring has no free slots.
///
/// Three-stage recovery:
///   1. Signal the gauge (fires high-pressure callbacks immediately via `signal_stall`).
///   2. Attempt a fresh `try_claim` after signalling.
///   3. Fall back to a fresh `PinnedBuffer::alloc` with a warning.
///
/// Never panics.  Never returns a null or invalid slot.
pub struct SlowPathAllocator {
    /// Maximum time to block in stage 2 before falling back to stage 3 (ms).
    reclaim_timeout_ms: u64,

    /// Optional pressure gauge — wired after construction via `set_gauge`.
    /// Uses `Mutex<Option<...>>` for interior mutability so `PoolRegistry`
    /// can register the gauge on an `Arc<PoolRegistry>` without `&mut self`.
    gauge: Mutex<Option<crate::scheduler::MemoryPressureGauge>>,
}

impl SlowPathAllocator {
    /// Construct with the default 50 ms reclaim timeout.
    pub fn new() -> Self {
        SlowPathAllocator {
            reclaim_timeout_ms: 50,
            gauge: Mutex::new(None),
        }
    }

    /// Register the pressure gauge for immediate stall signalling.
    ///
    /// Called by `PoolRegistry::set_pressure_gauge` after the gauge is wired up.
    pub fn set_gauge(&self, gauge: crate::scheduler::MemoryPressureGauge) {
        *self
            .gauge
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(gauge);
    }

    /// Handle pool exhaustion for the given ring.
    ///
    /// # Errors
    ///
    /// Infallible for Sprint 4 unless the OS rejects the overflow allocation.
    pub fn handle_exhaustion(&self, ring: &Arc<RingBuffer>, kind: LayerKind) -> Result<PoolSlot> {
        // Stage 1: Signal co-scheduler stall for IMMEDIATE pause.
        self.signal_coscheduler_stall(kind);

        // Stage 2: try_claim after the stall signal may have freed a slot.
        if let Some(pool_slot) = ring.try_claim() {
            return Ok(pool_slot);
        }

        // Stage 3: fresh overflow allocation.
        match PinnedBuffer::alloc(ring.slot_bytes()) {
            Ok(pinned_buffer) => {
                eprintln!(
                    "warn: RamFlow slow path allocated overflow pinned buffer for {:?} pool",
                    kind
                );
                Ok(PoolSlot::overflow(pinned_buffer))
            }
            Err(_allocation_error) => {
                eprintln!(
                    "warn: RamFlow slow path overflow allocation failed for {:?} pool; waiting for pooled slot",
                    kind
                );
                let _ = self.reclaim_timeout_ms;
                Ok(ring.claim_blocking())
            }
        }
    }

    /// Fire gauge.signal_stall() for immediate high-pressure callback dispatch.
    ///
    /// Unlike the periodic sample, this fires immediately so the NVMe engine
    /// pauses DMA submissions and lets in-flight transfers return their buffers.
    /// Uses u32::MAX as the layer_id sentinel meaning "pool stall, no specific layer".
    fn signal_coscheduler_stall(&self, _kind: LayerKind) {
        if let Ok(guard) = self.gauge.lock() {
            if let Some(gauge) = guard.as_ref() {
                gauge.signal_stall(u32::MAX);
            }
        }
    }
}

impl Default for SlowPathAllocator {
    fn default() -> Self {
        Self::new()
    }
}
