// src/cuda_bridge/zero_copy.rs — hybrid zero-copy router (Algorithm 3)
//
// Sprint 0: all types declared, all logic unimplemented!.
//
// Decision tree at runtime (threshold tuned by WarmupProfiler):
//
//   tensor.size_bytes < ZERO_COPY_THRESHOLD
//     → ZeroCopy: cudaHostGetDevicePointer → GPU reads pinned CPU mem via UVA
//       (requires buf registered with cudaHostRegisterMapped flag)
//   tensor.size_bytes ≥ ZERO_COPY_THRESHOLD
//     → DmaCopy:  cudaMallocAsync + cudaMemcpyAsync → full PCIe DMA to VRAM
//       (requires buf registered with cudaHostRegisterDefault flag)
//
// The crossover threshold defaults to 4 MiB on PCIe Gen4 x16, but the
// WarmupProfiler measures it on the actual machine and stores it in
// hardware_profile.json.

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::cuda_bridge::stream::CudaStream;

// ---------------------------------------------------------------------------
// Transfer strategy types
// ---------------------------------------------------------------------------

/// Opaque GPU-side device pointer returned by cudaHostGetDevicePointer.
///
/// Only valid while the underlying `PinnedBuffer` is alive and registered
/// with `cudaHostRegisterMapped`.
#[derive(Debug)]
pub struct DevicePointer {
    _raw: *mut u8,
}

// Safety: DevicePointer is a plain address that can be sent across threads.
// The underlying memory it points to is either pinned host memory (coherent
// via UVA) or VRAM — both are multi-GPU-accessible.
unsafe impl Send for DevicePointer {}
unsafe impl Sync for DevicePointer {}

/// The two transfer strategies the router selects between.
#[derive(Debug)]
pub enum TransferStrategy {
    /// GPU reads pinned CPU memory directly via CUDA UVA.
    ///
    /// No PCIe copy is issued.  Latency is lower for sub-threshold tensors.
    /// The `device_ptr` is valid for the lifetime of the source `PinnedBuffer`.
    ZeroCopy { device_ptr: DevicePointer },

    /// Full asynchronous PCIe DMA copy to a VRAM staging buffer.
    ///
    /// Preferred for large tensors (≥ threshold bytes) where sustained PCIe
    /// bandwidth outweighs the latency benefit of UVA reads.
    DmaCopy {
        /// The CUDA stream on which the `cudaMemcpyAsync` was issued.
        stream: CudaStream,
    },
}

// ---------------------------------------------------------------------------
// Zero-copy threshold (globally configurable, set by WarmupProfiler)
// ---------------------------------------------------------------------------

/// Default threshold: 4 MiB (calibrated for PCIe Gen4 x16).
/// Overwritten at runtime by `ZeroCopyRouter::set_threshold`.
static ZERO_COPY_THRESHOLD: AtomicUsize = AtomicUsize::new(4 * 1024 * 1024);

// ---------------------------------------------------------------------------
// ZeroCopyRouter — the routing engine itself
// ---------------------------------------------------------------------------

/// Selects [`TransferStrategy`] for each tensor at runtime.
///
/// Instantiated once by [`PoolRegistry`].  Holds no mutable state — all
/// configuration is stored in atomics so callbacks can update the threshold
/// without taking a lock.
///
/// # Sprint 0 contract
/// Compiles; `route` and `set_threshold` are `unimplemented!`.
pub struct ZeroCopyRouter {
    _opaque: (),
}

impl ZeroCopyRouter {
    /// Construct a router.  The threshold is initially `ZERO_COPY_THRESHOLD`.
    pub fn new() -> Self {
        unimplemented!("ZeroCopyRouter::new — Sprint 0 stub")
    }

    /// Route `buf` to the appropriate transfer strategy.
    ///
    /// `buf.is_mapped` must be `true` for `ZeroCopy` to be returned;
    /// if it is `false` this always falls back to `DmaCopy`.
    #[allow(unused_variables)]
    pub fn route(
        &self,
        buf: &crate::allocator::PinnedBuffer,
        stream: &CudaStream,
    ) -> crate::Result<TransferStrategy> {
        unimplemented!("ZeroCopyRouter::route — Sprint 0 stub")
    }

    /// Update the crossover threshold (bytes).
    ///
    /// Called by the warm-up profiler after measuring the actual crossover
    /// on this machine.  Uses `Ordering::Relaxed` — the threshold is a
    /// best-effort hint, not a sequentially-consistent decision point.
    pub fn set_threshold(bytes: usize) {
        ZERO_COPY_THRESHOLD.store(bytes, Ordering::Relaxed);
    }

    /// Current threshold in bytes.
    pub fn threshold() -> usize {
        ZERO_COPY_THRESHOLD.load(Ordering::Relaxed)
    }
}

impl Default for ZeroCopyRouter {
    fn default() -> Self {
        Self::new()
    }
}
