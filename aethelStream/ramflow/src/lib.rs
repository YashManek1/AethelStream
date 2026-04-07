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
pub mod error;
pub use error::{RamFlowError, Result};

// ---------------------------------------------------------------------------
// Allocator — pinned host memory, RAII drop guard
// ---------------------------------------------------------------------------
pub mod allocator;

// ---------------------------------------------------------------------------
// Pool — ring buffers, slab packer, slow path, TensorLocationDict
// ---------------------------------------------------------------------------
pub mod pool;

// ---------------------------------------------------------------------------
// CUDA bridge — streams, zero-copy router, raw bindings
// ---------------------------------------------------------------------------
pub mod cuda_bridge;

// ---------------------------------------------------------------------------
// NVMe I/O — io_uring, fd table, prefetch engine, write budget
// ---------------------------------------------------------------------------
pub mod nvme;

// ---------------------------------------------------------------------------
// Phase manager — TrainingPhase, PhaseClassifier, Warmup profiler, rebalancer
// ---------------------------------------------------------------------------
pub mod phase;

// ---------------------------------------------------------------------------
// Kernel wrappers — thin Rust wrappers around compiled .cu kernels
// ---------------------------------------------------------------------------
pub mod kernels;

// ---------------------------------------------------------------------------
// Scheduler — co-scheduler, pressure gauge, per-layer scale table
// ---------------------------------------------------------------------------
pub mod scheduler;

// ---------------------------------------------------------------------------
// Primary re-exports — what 95% of callers need
// ---------------------------------------------------------------------------

/// Page-locked (pinned) host buffer visible to both CPU and GPU.
///
/// Has two registration modes: Default (DMA only) and Mapped (DMA + UVA).
pub use allocator::PinnedBuffer;

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
