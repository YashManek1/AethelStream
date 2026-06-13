// src/lib.rs — RamFlow crate root
//
// Sprint 0 skeleton: all submodules declared, all types are stubs that
// compile but panic if called.  Every downstream module can immediately
// depend on this crate and reference any type below.
//
// Type import cheat-sheet for other modules:
//
//   use ramflow::{PinnedBuffer, PoolRegistry, DirectNvmeEngine, MemoryPressureGauge};
//   use ramflow::{CoScheduler, PerLayerScaleTable, TensorSlab};
//   use ramflow::pool::{LayerKind, PoolSlot, TensorLocationDict};
//   use ramflow::phase::{TrainingPhase, Direction, PhaseClassifier, WarmupProfiler};
//   use ramflow::cuda_bridge::zero_copy::{TransferStrategy, ZeroCopyRouter};
//   use ramflow::nvme::write_budget::{WriteBudgetManager, WriteStrategy};
//   use ramflow::{Result, RamFlowError};

#![warn(missing_docs, clippy::all)]
#![deny(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

//! **ramflow** — high-throughput memory orchestration for AethelStream.
//!
//! Implements five novel algorithms:
//! 1. Phase-aware predictive pool allocation
//! 2. Tensor-size-aware hybrid zero-copy routing
//! 3. Memory/I/O co-scheduler with pressure feedback
//! 4. Tensor slab packing for co-traveling small tensors
//! 5. Per-layer exponentially weighted overflow density scaling

// ---------------------------------------------------------------------------
// Error / Result — declared first so all submodules can use ramflow::Result
// ---------------------------------------------------------------------------
/// Structured error types and `Result` alias for the entire crate.
pub mod error;
pub use error::{RamFlowError, Result};

// ---------------------------------------------------------------------------
// Allocator — pinned host memory, RAII drop guard
// ---------------------------------------------------------------------------
/// Page-locked host-memory allocator: `PinnedBuffer` and its `Drop` guard.
pub mod allocator;

// ---------------------------------------------------------------------------
// Pool — ring buffers, slab packer, slow path, TensorLocationDict
// ---------------------------------------------------------------------------
/// Pre-allocated pool rings, tensor slab packer, and tensor location index.
pub mod pool;

// ---------------------------------------------------------------------------
// CUDA bridge — streams, zero-copy router, raw bindings
// ---------------------------------------------------------------------------
/// CUDA runtime bridge: stream handle, zero-copy router, and FFI bindings.
pub mod cuda_bridge;

// ---------------------------------------------------------------------------
// NVMe I/O — io_uring, fd table, prefetch engine, write budget
// ---------------------------------------------------------------------------
/// Zero-syscall NVMe I/O engine: io_uring, fd table, prefetch, write budget.
pub mod nvme;

// ---------------------------------------------------------------------------
// Phase manager — TrainingPhase, PhaseClassifier, Warmup profiler, rebalancer
// ---------------------------------------------------------------------------
/// Training-phase management: classifier, warm-up profiler, pool rebalancer.
pub mod phase;

// ---------------------------------------------------------------------------
// Kernel wrappers — thin Rust wrappers around compiled .cu kernels
// ---------------------------------------------------------------------------
/// Thin Rust wrappers around compiled CUDA kernels.
pub mod kernels;

// ---------------------------------------------------------------------------
// Scheduler — co-scheduler, pressure gauge, per-layer scale table
// ---------------------------------------------------------------------------
/// Memory-pressure gauge, I/O co-scheduler, and per-layer loss scale table.
pub mod scheduler;

// ---------------------------------------------------------------------------
// Emergency checkpointing — opt-in signal hook
// ---------------------------------------------------------------------------
/// Opt-in emergency checkpoint hook for SIGTERM/SIGINT shutdown paths.
pub mod emergency;

// ---------------------------------------------------------------------------
// Primary re-exports — what 95% of callers need
// ---------------------------------------------------------------------------

/// Page-locked (pinned) host buffer visible to both CPU and GPU.
///
/// Has two registration modes: Default (DMA only) and Mapped (DMA + UVA).
/// When the `mmap-fallback` feature is active, `PinnedBuffer::alloc_mmap` creates
/// a pageable variant (`is_pinned() == false`) for low-RAM machines.
pub use allocator::PinnedBuffer;

/// Allocation kind discriminant: Standard (posix_memalign) or Huge (mmap + hugepages).
pub use allocator::AllocKind;

/// Minimum slot size in bytes for hugepage-backed allocation (2 MiB, feature = "hugepages").
#[cfg(feature = "hugepages")]
pub use allocator::HUGEPAGE_THRESHOLD;

/// NUMA topology config returned by [llocator::numa::detect].
pub use allocator::NumaConfig;

/// Central registry of all pool shards. Routes `claim(LayerKind)` to the
/// correct ring buffer and owns all per-layer `TensorSlab`s.
pub use pool::PoolRegistry;

/// Zero-syscall NVMe engine built on io_uring.
/// Integrates with the co-scheduler via `pause_signal`.
pub use nvme::DirectNvmeEngine;

/// Real-time memory-pressure sensor.  Drives the co-scheduler's prefetch
/// window adjustments and triggers the slow-path stall handler.
pub use scheduler::MemoryPressureGauge;

/// CPU/GPU co-scheduler — eviction orchestration + prefetch control.
pub use scheduler::CoScheduler;

/// Per-layer loss scale table with EWA overflow density tracking (Alg 6).
pub use scheduler::PerLayerScaleTable;

/// Per-layer packed single allocation for all small tensors (Alg 5).
pub use pool::TensorSlab;

// ---------------------------------------------------------------------------
// lz4-cache re-exports — available when feature = "lz4-cache"
// ---------------------------------------------------------------------------

/// Telemetry snapshot for the LZ4 eviction cache (feature = "lz4-cache").
///
/// Retrieved via [`PoolRegistry::lz4_cache_telemetry`].
#[cfg(feature = "lz4-cache")]
pub use pool::eviction_cache::Lz4CacheTelemetry;

/// Precision discriminant stored per LZ4 cache entry (feature = "lz4-cache").
#[cfg(feature = "lz4-cache")]
pub use pool::eviction_cache::CachePrecision;

// ---------------------------------------------------------------------------
// mmap-fallback re-exports — available when feature = "mmap-fallback"
// ---------------------------------------------------------------------------

/// mmap-backed host buffer for graceful-degradation streaming (feature = "mmap-fallback").
///
/// Allocated via `mmap(MAP_PRIVATE|MAP_ANONYMOUS)` + `madvise(MADV_SEQUENTIAL)`.
/// `is_pinned()` is always `false` for this buffer type.
#[cfg(feature = "mmap-fallback")]
pub use allocator::MmapBuffer;

/// Unified buffer enum covering pinned and mmap-backed allocations.
///
/// Internal type used by callers that must handle both allocation tiers.
#[cfg(feature = "mmap-fallback")]
pub use allocator::AnyBuffer;

/// Common access interface implemented by [`PinnedBuffer`], [`MmapBuffer`], and [`AnyBuffer`].
#[cfg(feature = "mmap-fallback")]
pub use allocator::BufferAccess;


// ---------------------------------------------------------------------------
// DirectStorage re-exports — available when feature = "direct-storage"
// ---------------------------------------------------------------------------

/// Capability probe result for Windows DirectStorage (feature = "direct-storage").
///
/// Returned by [`probe_direct_storage`]. On non-Windows platforms or when
/// `dstorage.dll` is absent, always `Unavailable`.
#[cfg(feature = "direct-storage")]
pub use nvme::direct_storage::DirectStorageCapability;

/// Probe for Windows DirectStorage availability at runtime (feature = "direct-storage").
///
/// Loads `dstorage.dll` transiently and checks for the `DStorageGetFactory` export.
/// Returns [`DirectStorageCapability::Unavailable`] on non-Windows or when the DLL
/// is not installed; never panics.
#[cfg(feature = "direct-storage")]
pub use nvme::direct_storage::probe_direct_storage;

/// Allocate a 4 096-byte-aligned [`PinnedBuffer`] for DirectStorage GPU transfers.
///
/// The `DSTORAGE_REQUEST_DESTINATION_BUFFER` path requires 4 096-byte alignment;
/// the standard [`PinnedBuffer::alloc`] guarantees only 512-byte alignment.
#[cfg(feature = "direct-storage")]
pub use nvme::direct_storage::alloc_windows_ds_compatible;
