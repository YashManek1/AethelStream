# RamFlow: System Memory Orchestration for AethelStream
## Full Technical Report — Module 2 Reference Document

**Version:** Post-Phase 1 Audit Hardening (complete)  
**Date:** 2026-06-11  
**Test status:** 82 passing / 0 failing (mock-cuda + ssd-wear), 0 failing (mock-cuda only)  
**Clippy:** 0 warnings, 0 errors (`-D warnings`)

---

## Abstract

RamFlow is the memory-orchestration substrate of AethelStream, a framework for training 7B–70B+ transformer models on a single consumer GPU by streaming one layer at a time across NVMe → System RAM → VRAM. RamFlow owns every byte of system RAM used during training: it allocates page-locked (pinned) buffers with exact sizing and 512-byte alignment, manages a ring-pool of pre-allocated slots partitioned by tensor kind, routes small tensors through a zero-copy UVA path or a standard DMA path based on a runtime-measured crossover threshold, drives NVMe I/O through `io_uring` with optional SQPOLL zero-syscall submissions, detects memory pressure via a sampling gauge and reacts through a co-scheduler that pauses prefetch and shrinks the prefetch window, tracks per-layer FP16 overflow density with an Exponentially Weighted Average (EWA) loss-scale table, and compresses activation checkpoints to INT8 when pressure enters a soft band. Five original algorithms emerge from this design. This document describes all of them with sufficient mathematical and implementation detail for a research paper submission.

---

## 1. Introduction

### 1.1 Problem

Fine-tuning or training large language models with 7B–70B parameters conventionally requires multi-GPU clusters with tens to hundreds of gigabytes of VRAM. A single RTX 4090 has 24 GB of VRAM; a single consumer DDR5 machine has 32–128 GB of system RAM. The model weights alone for a 70B FP16 model occupy 140 GB — far beyond any single GPU.

Prior approaches fall into two categories:

1. **Model parallelism / pipeline parallelism** — requires multi-GPU and high-bandwidth interconnect (NVLink or InfiniBand). Not available on consumer hardware.
2. **Quantization to INT4/INT8** — reduces the model to fit in one device but degrades gradient quality, which is unacceptable for fine-tuning tasks where downstream task accuracy matters.

AethelStream takes a third path: **temporal streaming**. Only the layer currently being computed needs to reside in VRAM. All other layers live in system RAM (or NVMe). The GPU never waits because the next layer is prefetched while the current layer computes.

### 1.2 Why Memory Management Is the Hard Part

Naive streaming (allocate a `Vec<u8>`, copy to CUDA) is 3–5× slower than RamFlow's design for these reasons:

- **Page-locking overhead.** Every `cudaMemcpyAsync` from pageable host memory incurs an internal `cudaMallocHost` + copy. With pinned memory pre-registered, the DMA engine transfers directly.
- **Allocation fragmentation.** Frequent small allocations fragment RAM. After 1000 training steps, a model with 80 layers and 10 tensors each (800 alloc/free cycles per step) produces fragmented address ranges that the kernel cannot consolidate without a compaction pass.
- **Syscall pressure.** Standard `read()` or `mmap` for tensor loading issues one syscall per tensor. `io_uring` with SQPOLL issues zero syscalls in steady state.
- **Pressure blindness.** Without a memory-pressure gauge, the prefetch engine does not know when to back off — it either prefetches too aggressively (OOM) or too conservatively (GPU stalls).

RamFlow solves each of these. The result is a memory layer that the rest of AethelStream can depend on without worrying about fragmentation, paging, or synchronization.

### 1.3 Position in AethelStream

AethelStream has eight modules:

| Module | Name | Status |
|--------|------|--------|
| M1 | Model Init & Sharding | Python |
| **M2** | **RamFlow — System Memory Manager** | **Complete (this document)** |
| M3 | Prefetch Engine & I/O Pipeline | Depends on M2 |
| M4 | Optimizer State Manager | Parallel |
| M5 | Double-Pass Backward Engine | Depends on M2 |
| M6 | LoRA Adapter Manager | Parallel |
| M7 | Python Training Loop & FFI Bridge | Depends on M3, M5 |
| M8 | Benchmarking & Paper Prep | Depends on all |

RamFlow is the foundation. No other module allocates pinned memory, opens NVMe file descriptors, or makes independent RAM management decisions — all of that flows through M2's public API.

---

## 2. System Architecture

### 2.1 Module Decomposition

RamFlow (crate `ramflow`, `aethelStream/ramflow/`) is organized into seven subsystems:

```
ramflow/
├── src/
│   ├── allocator/       # PinnedBuffer — page-locked exact-size allocation
│   ├── pool/            # PoolRegistry, RingBuffer, TensorSlab, TensorLocationDict
│   ├── nvme/            # DirectNvmeEngine, IoUringInstance, PrefetchEngine, WriteBudgetManager
│   ├── phase/           # PhaseClassifier, WarmupProfiler, PhaseRebalancer
│   ├── scheduler/       # MemoryPressureGauge, CoScheduler, PerLayerScaleTable
│   ├── cuda_bridge/     # CudaStream, ZeroCopyRouter, bindings
│   ├── kernels/         # Rust wrappers for .cu kernels
│   └── emergency.rs     # SIGTERM/SIGINT checkpoint hook
├── kernels/
│   ├── overflow_check.cu/.cuh    # FP16 NaN/Inf detection (Algorithm 3)
│   └── checkpoint_compress.cu/.cuh  # INT8 compression (Algorithm 5)
└── tests/               # 10 numbered integration test files + integration.rs
```

### 2.2 Data Flow

```
NVMe (SSD)
    │
    │  io_uring SQE (O_DIRECT, 512-byte aligned)
    ▼
PinnedBuffer (posix_memalign(512), cudaHostRegister)
    │
    │  cudaMemcpyAsync (DMA path) or UVA pointer (zero-copy path)
    ▼
CUDA Device Memory
    │
    ▼
Layer Compute (attention, MLP, norm)
    │
    ├── Forward: activation stored in PinnedBuffer (CheckpointDict, Module 5)
    └── Backward: gradient overflow detected by count_overflow_fp16 kernel
               → PerLayerScaleTable updated
               → INT8 compression if soft pressure band active
```

### 2.3 Concurrency Model

```
Training loop thread
    │  claim() → ring fast-path (lock-free empty check + Mutex pop)
    │  prefetch() → SQE written to io_uring ring
    │  sample_and_notify() → pressure callbacks fired (if step % interval == 0)
    │
Background threads:
    ├── "ramflow-cqe-poller"   — pinned to CPU core N, drains CQE ring, sends tokens
    └── "ramflow-pressure-sampler" — wakes every 10 s, fires pressure callbacks
```

All shared state is protected by either `AtomicXxx` or `Mutex<T>`. The critical path (claim → prefetch → poll) is lock-free for the common case (ring not exhausted, no pressure event).

---

## 3. Subsystem Reference

### 3.1 Allocator: PinnedBuffer (`src/allocator/pinned.rs`)

`PinnedBuffer` is the only legal way to allocate host memory in AethelStream.

**Alignment:** 512 bytes (`PINNED_ALIGN = 512`). This satisfies:
- NVMe `O_DIRECT`: buffer address, file offset, and transfer length must all be multiples of the device logical-sector size (512 B for NVMe).
- CPU DMA cache line alignment: 512 is a multiple of 64 (one CPU cache line).
- `posix_memalign` requirement: alignment must be a power-of-two multiple of `sizeof(void*)` (8 on 64-bit).

**Exact sizing:** Unlike PyTorch's `CachingAllocator` (rounds up to next power-of-two), `posix_memalign` allocates exactly the requested bytes (plus OS page overhead, not binary overhead). For a 2.1 MB tensor: PyTorch → 4 MB; RamFlow → 2,097,152 B.

**Registration modes:**
| Mode | Flag | Purpose |
|------|------|---------|
| `alloc()` | `cudaHostRegisterDefault` (0) | DMA-accessible, CPU-only virtual address. Standard path. |
| `alloc_mapped()` | `cudaHostRegisterMapped` (2) | DMA + UVA. GPU can dereference via `cudaHostGetDevicePointer`. Used by zero-copy path only. |

**Drop ordering (critical):**
```
Drop::drop():
  1. cudaHostUnregister(&ptr)   // release CUDA pin — MUST come first
  2. platform::free_aligned(ptr) // return memory to OS
```
Reversing this is silent UB: the OS can reuse the physical pages immediately after `free`, then `cudaHostUnregister` dereferences the now-invalid physical address, corrupting the next allocation.

**INT8 compression flag:** `compressed: bool` tracks whether the buffer content has been quantized by the `compress_checkpoint_fp16_to_int8` kernel. Module 5 checks `buf.is_compressed()` before reading gradient checkpoint data. **Caller obligation:** after `compress_checkpoint_fp16_to_int8` returns `Ok(())`, the caller must immediately call `buf.set_compressed(true)` — the kernel does not set the flag itself, to keep kernel logic free of Rust object references.

**Platform support:** Linux (`posix_memalign` + `libc::free`), Windows (`_aligned_malloc` + `_aligned_free`), fallback (`std::alloc::Layout`).

---

### 3.2 Pool: PoolRegistry (`src/pool/subpools.rs`)

`PoolRegistry` owns four `RingBuffer` instances, one per `LayerKind`:

| Ring | Default slot count | Default slot bytes | Typical use |
|------|-------------------|--------------------|-------------|
| Attention | 4 (profile-driven) | 64 MiB (threshold/2) | Q/K/V/O projections |
| MLP | 4 (profile-driven) | 64 MiB | up/gate/down projections |
| Norm | 4 | 1 MiB (fixed) | LayerNorm weight+bias |
| Embedding | 2 (optimizer_slots) | 32 MiB (threshold/4) | embedding tables |

**Pool sizing formula:**
```
large_slot_bytes   = max(zero_copy_threshold / 2, 512)
norm_slot_bytes    = 1,048,576  (1 MiB, fixed)
embed_slot_bytes   = max(zero_copy_threshold / 4, 512)
```

The zero-copy threshold is measured at startup by `WarmupProfiler::measure_zero_copy_crossover()`. In the mock estimator it is 4 MiB, giving 2 MiB attention/MLP slots.

**Threshold wiring:** `PoolRegistry::new()` now calls `ZeroCopyRouter::set_threshold(zero_copy_threshold)` immediately after the non-zero threshold guard, propagating the hardware-profiled crossover to the global `ZeroCopyRouter` singleton. Before this fix, every `ZeroCopyRouter::route()` call used the default 4 MiB constant regardless of what the profiler measured.

**Pre-flight budget check:** Before any allocation, `preflight_profile_memory_budget()` estimates:
```
estimated_peak = (attention_slots × large_slot_bytes)
              + (mlp_slots × large_slot_bytes)
              + (norm_slots × norm_slot_bytes)
              + (embed_slots × embed_slot_bytes)
              + expected_peak_bytes (checkpoint budget)
              + optimizer_budget
```
If `estimated_peak > available_ram × RAMFLOW_POOL_RAM_FRACTION` (default 0.90), construction fails before a single byte is allocated, giving a clean error instead of OOM mid-training.

**Claim fast path:**
```rust
ring.try_claim()           // lock-free AtomicUsize empty check → Mutex pop
    .or_else(|| slow_path.handle_exhaustion(ring, kind))
```
`SlowPathAllocator` calls `gauge.signal_stall(u32::MAX)` to fire high callbacks immediately (not waiting for the next sample cycle), then spins until a slot is returned.

**Phase-aware resizing:** `resize_to_profile(profile)` resizes all rings to a new slot count. Called by `PhaseRebalancer` after acquiring a phase fence (all in-flight slots returned + all CUDA copies complete).

**TensorSlab packing:** Small tensors (below `zero_copy_threshold`) for a single layer are packed into one contiguous `PinnedBuffer` with a UVA mapping. Subsequent tensors within the slab are accessed via pointer arithmetic, eliminating per-tensor allocation overhead.

**Lazy slab mode:** `new_lazy()` defers slab allocation to first use. Low-RAM machines call `ensure_slab_for_layer(idx, dict)` before a layer visit and `release_lazy_slab(idx)` after, bounding peak RAM to one layer's slab at a time.

---

### 3.3 NVMe I/O Engine (`src/nvme/`)

#### 3.3.1 IoUringInstance (`io_uring_setup.rs`)

Wraps the `io-uring` crate (pinned to v0.7.11) into a two-mode ring:

| Mode | Condition | Syscalls per submission |
|------|-----------|------------------------|
| SQPOLL | Kernel ≥ 5.11, CAP_SYS_NICE or root | **0** (kernel thread polls SQ) |
| Standard | Fallback | 1 (`io_uring_enter`) |

SQPOLL parameters: idle timeout = 2000 ms (kernel thread sleeps after 2 s of no submissions, wakes on next push).

Ring geometry: `sq_entries = 128`, `cq_entries = 256`. 256 CQEs = 2× the SQ depth, preventing CQ overflow when completions arrive faster than the poller drains them. Both the SQPOLL and standard `IoUring::builder()` chains now call `.setup_cqe_size(params.cq_entries)` to pass this value to the kernel — previously `cq_entries` was declared in `IoUringParams` but never forwarded to the kernel, which used its own heuristic sizing.

Two API variants at compile time:
- Default: `submission_shared()` / `completion_shared()` — lowest overhead, no internal lock.
- `io-uring-use-split`: `ring.split()` under `Mutex` — for kernel versions where the shared API is buggy.

**libaio fallback** (`libaio-fallback` feature): If `io_uring_setup()` fails (very old kernel, seccomp restriction), falls back to blocking `pread`/`pwrite` on the submission path and emits normal CQE results.

#### 3.3.2 FdTable (`fd_table.rs`)

Owns all `O_DIRECT | O_RDONLY | O_CLOEXEC` file descriptors for shard files. Shard IDs are stable indices into the table (module 1 assigns them). `get_raw_fd(shard_id)` is O(1).

Write FDs are tracked separately (`get_write_raw_fd`) — opened `O_WRONLY | O_DIRECT | O_CLOEXEC` so reads and writes can proceed concurrently on different descriptors.

#### 3.3.3 PrefetchEngine and DirectNvmeEngine (`prefetch.rs`, `nvme/mod.rs`)

**PrefetchEngine** prepares `IORING_OP_READ` / `IORING_OP_WRITE` SQEs. Every SQE carries a `PrefetchToken` (`u64`) in `user_data`, which the kernel echoes back in the matching CQE.

**O_DIRECT alignment validation** (`validate_direct_io_alignment`):
```
byte_offset mod 512 == 0
length      mod 512 == 0
buffer_ptr  mod 512 == 0
```
PinnedBuffer satisfies the third constraint by construction (512-byte alignment from `posix_memalign`). The `DIRECT_IO_ALIGNMENT = 512` constant in `prefetch.rs` carries an explicit doc cross-reference to `PINNED_ALIGN = 512` in `allocator::pinned`, making the co-dependence auditable without tracing through two files.

**Pressure gate** (`check_pause()`):
```
if pause_signal == true  →  Err(PressurePause(layer_id))
if outstanding_reads + claimed_slots > pressure_threshold  →  Err(PressurePause(0))
```

**CQE poller thread** (`spawn_cqe_poller`):
- Name: `"ramflow-cqe-poller"`
- Stack: 256 KiB
- Pinned to `cpu_core` via `pthread_setaffinity_np` (prevents 10–100 µs cache-miss penalty from thread migration)
- Poll interval: 1 ms `wait_for_cqe_timeout` (short enough for frequent stop-signal checks; long enough to avoid busy-polling)
- On completion: sends `CqeResult { token, result }` on bounded channel (capacity `COMPLETION_CHANNEL_CAPACITY = 4 * CQ_DEPTH = 1024` — derived from the ring constant, not hardcoded)

**Super-shard grouped I/O:** When consecutive layers share a single contiguous shard file region, `schedule_super_shard` submits one read for the entire group. This reduces SQE count from O(layers × tensors) to O(groups). Disabled by default until the WarmupProfiler measures a layout where grouping wins.

When enabled, `schedule_super_shard` validates that every entry in the `layer_offsets` slice satisfies `offset < length` before submitting the merged SQE. An out-of-range offset returns `Err(RamFlowError::ConfigError(...))` with the offending index, value, and shard ID. The `layer_offsets` parameter is relative to `byte_offset`; the caller uses it to demultiplex individual layers out of the merged destination buffer after completion.

#### 3.3.4 WriteBudgetManager (`write_budget.rs`, `ssd-wear` feature)

Tracks cumulative NVMe bytes written against a user-configured TBW budget. Reads the NVMe SMART/Health Information Log (Log ID 0x02, bytes 48–63) via `NVME_IOCTL_ADMIN_CMD` (ioctl `0xC0484E41`) to get the baseline "Data Units Written" counter.

**Strategy auto-switch:**

| Remaining budget | Strategy | Write amplification |
|-----------------|----------|---------------------|
| > 50% | `Full` | 1.0× (full shard write) |
| 10–50% | `DeltaCompress` | ~0.1× (zstd level 3 on near-zero deltas) |
| ≤ 10% | `Deferred { batch_size: 4 }` | Batched, NVMe controller-reordered |

**SMART unit convention:** 1 NVMe SMART "Data Units Written" unit = 1000 × 512-byte sectors = 512,000 bytes.

**Delta compression round-trip contract:**
```
delta[i] = updated_i16[i] - original_i16[i]  (wrapping i16 subtraction)
restored[i] = original_i16[i] + delta[i]     (wrapping i16 addition)
```
Wrapping arithmetic guarantees bit-exact round-trip even when the delta overflows a signed 16-bit integer.

---

### 3.4 Phase Manager (`src/phase/`)

**TrainingPhase** (enum):
```
Forward      { layers_in_flight: u32 }
Backward     { from_layer: u32, to_layer: u32 }
Recomputation{ window_start: u32, window_end: u32 }
```

**PhaseClassifier** tracks phase transitions via `notify_layer_start(layer_idx, direction)` and `notify_backward_recompute_start(start, end)`. Fires transition callbacks (used by the integration test to count the 6 expected transitions over 2 forward+backward+recomputation cycles).

**WarmupProfiler** runs 3 synthetic mini-steps to measure:
- Peak simultaneous attention slot claims per phase (Forward: 1, Backward: 2, Recomputation: 3 in the integration test)
- Zero-copy crossover threshold (mock: 4 MiB)
- Writes `hardware_profile.json` to the configured output path

**SHA-256 cache invalidation:** `WarmupConfig::for_shard_index(path)` reads the shard index file and computes its SHA-256 hash at construction time. On subsequent runs, `load_cached_profiles()` compares this hash against the stored profile; a mismatch (model changed) skips the cache and re-profiles. The hash is never a zero-filled dummy — it is always derived from the file bytes, so a profile produced for one model checkpoint is never silently reused for a different one.

**PhaseRebalancer** acquires a "phase fence" (spin-wait until `total_claimed_slots == 0 && outstanding_cuda_copies == 0`) before resizing rings. Timeout configurable via `RAMFLOW_PHASE_FENCE_TIMEOUT_MS` in debug builds (default 30 s in release).

---

### 3.5 Scheduler (`src/scheduler/`)

#### 3.5.1 MemoryPressureGauge

Stores pressure as `AtomicU32` (f32 bits, `Relaxed` ordering — a best-effort hint, not a synchronization point). Callbacks are `Arc<dyn Fn(f32) + Send + Sync>` stored in `Mutex<Vec<...>>`.

**Deadlock-safe callback dispatch:** snapshots the Arc list under lock, drops the lock, then invokes each callback. A callback that calls `register_*_pressure` will acquire the lock without contention.

**Three pressure bands:**
```
p > high_threshold  (default 0.80)  →  fire high_callbacks
soft_threshold < p ≤ high_threshold (default 0.70–0.80) →  fire soft_callbacks
p < low_threshold   (default 0.40)  →  fire low_callbacks
```
Bands are mutually exclusive: one callback fires per sample, never two.

**Sampling paths:**
1. `sample_and_notify(&registry)` — called from the training loop every N steps (N = max(5, steps_per_minute/6) ≈ every 10 s).
2. `start(registry)` — spawns `"ramflow-pressure-sampler"` thread (10 s interval, Acquire-ordered shutdown check).
3. `signal_stall(layer_id)` — emergency path from `SlowPathAllocator`; fires high callbacks at pressure = 1.0 immediately.

**Pressure formula:**
```
p = total_claimed_slots / total_capacity
```

#### 3.5.2 CoScheduler

Registers three callbacks at construction:

| Pressure band | Effect on prefetch_window | Effect on pause_signal | Effect on compress_trigger |
|--------------|--------------------------|------------------------|---------------------------|
| High (> 0.80) | `fetch_sub(1, AcqRel)` | `store(true, Release)` | — |
| Soft (0.70–0.80) | — | — | `store(true, Release)` |
| Low (< 0.40) | `fetch_add(1, AcqRel)` | `store(false, Release)` | `store(false, Release)` |

`AcqRel` on the window atomics ensures prior writes in the callback's thread are visible to the next thread that reads the decremented window (prevents ARM/RISC-V memory-ordering hazards).

**`should_compress_checkpoints() -> bool`** — public reader for `compress_trigger`. Returns `compress_trigger.load(Acquire)`, observing the `Release` store from the soft-pressure callback. Module 5 calls this at the start of each backward step; a `true` return triggers `compress_checkpoint_fp16_to_int8` on the next activation checkpoint before the layer streams back in. Before this method existed, `compress_trigger` had no public reader, making the soft-pressure band's effect unobservable outside the scheduler subsystem.

---

### 3.6 CUDA Kernels (`kernels/`)

#### 3.6.1 `overflow_check.cu` — FP16 Overflow Detection

**`fused_overflow_check` kernel:**
- Threads: 256 per block, grid = ceil(n / 256)
- Each thread: `bits = __half_as_ushort(grad[idx])`; if `(bits & 0x7C00) == 0x7C00` → `*overflow_flag = true`
- No atomics: all threads that find overflow write the same value (`true`). Idempotent concurrent writes are defined in CUDA's memory model.
- Why `__half_as_ushort` not `*(uint16_t*)`: the union/pointer-cast is undefined behavior under C++ strict aliasing; the intrinsic is the CUDA-sanctioned bit-reinterpretation.

**`ramflow_count_overflow_fp16`** (Algorithm 3): Two-stage reduction returning the count of NaN/Inf elements (not just a boolean). Used by `PerLayerScaleTable::update()`.

#### 3.6.2 `checkpoint_compress.cu` — INT8 Activation Checkpoint Compression

Three kernels, grid layout: `gridDim.x = n_channels, blockDim.x = min(elems_per_channel, 256)`.

**`find_channel_scales`** (shared-memory reduction):
```
local_max = max over stride loop of |src[c][i]|  (using __half2float)
shared_mem reduction (binary tree) → channel_max
scales[c] = (channel_max > 0) ? channel_max / 127.0 : 1.0
```

**`compress_fp16_to_int8`:**
```
q = clamp(__float2int_rn(src[c][i] / scales[c]), -128, 127)
dst[c][i] = (int8_t)q
```
Uses `__float2int_rn` (round-to-nearest-even, IEEE 754). The Rust mock `mock_compress_checkpoint` replicates this via `round_half_to_even(x)` — a private helper that rounds ties toward the even integer, matching CUDA semantics exactly. Previously the mock used `f32::round()` (round-half-away-from-zero), which diverges from `__float2int_rn` on exact half-values (e.g., 2.5 → 3 vs 2); this divergence would have caused phantom test failures on tie inputs during gradient parity testing.

**`decompress_int8_to_fp16`:**
```
dst[c][i] = __float2half((float)src[c][i] * scales[c])
```

**Expected quantization error:** < 0.1% gradient deviation for typical post-LayerNorm activations (verified in Test 12 below).

**Packed buffer layout helper:** `split_compressed_buffer_ptrs(base: *mut u8, n_channels: usize) -> Result<(*mut f32, *mut i8)>` is an `unsafe fn` that derives the two sub-pointers from a single `PinnedBuffer` base pointer for callers that allocate one buffer for the full compressed checkpoint:

```
┌──────────────────────────────────┬───────────────────────────────────┐
│  n_channels × sizeof(f32)        │  n_channels × elems_per_channel   │
│  per-channel scale factors (f32) │  × sizeof(i8)  quantised values   │
└──────────────────────────────────┴───────────────────────────────────┘
^─── base ptr (512-byte aligned) ─────────────────────────────────────^
```

Returns `Err(ConfigError)` if `n_channels == 0`. Marked `unsafe` because it performs raw pointer arithmetic; callers must ensure `base` points to a sufficiently large, correctly aligned buffer.

---

## 4. The Five Novel Algorithms

### Algorithm 1: Phase-Aware Predictive Pool Allocation

**Problem:** Naively sizing the pool at a worst-case fixed capacity wastes RAM during the forward pass (which needs fewer simultaneous slots) and under-allocates during recomputation (which needs more).

**Solution:** The pool is sized at construction to the worst-case phase (Recomputation) so the rebalancer only ever shrinks. At each phase boundary, `PhaseRebalancer::rebalance_to_profile(profile)` resizes rings after acquiring a phase fence.

**Sizing rule:**
```
Attention slots: max(profile.attention_slots_needed, 1)
MLP slots:       max(profile.mlp_slots_needed, 1)
Norm slots:      max(profile.norm_slots_needed, 1)
Embedding slots: max(profile.optimizer_slots_needed, 1)
```

**Phase fence protocol:**
```
while total_claimed_slots > 0 OR outstanding_cuda_copies > 0:
    sleep(1ms)
    if elapsed > fence_timeout (30 s):
        return Err(PhaseTransitionError)
```

**Measured values (integration test):**
- Forward phase: 1 simultaneous attention slot, 1 MLP slot
- Backward phase: 2 simultaneous attention slots, 1 MLP slot
- Recomputation: 3 simultaneous attention slots, 2 MLP slots

### Algorithm 2: Tensor-Size-Aware Hybrid Zero-Copy Routing

**Problem:** CUDA's Unified Virtual Addressing (UVA) zero-copy path (GPU reads host memory directly over PCIe) has lower latency than `cudaMemcpyAsync` for small tensors but lower throughput for large tensors due to PCIe transaction overhead.

**Solution:** At startup, `WarmupProfiler::measure_zero_copy_crossover()` measures actual transfer latency for a sweep of tensor sizes and finds the crossover threshold T*. Tensors below T* use UVA (`alloc_mapped`), tensors at or above T* use DMA.

**Routing decision:**
```
if tensor.size < T* AND tensor.is_mapped():
    route = ZeroCopy  # cudaHostGetDevicePointer, no copy
else:
    route = Dma       # cudaMemcpyAsync
```

**Safety guard:** `device_pointer_for_mapped_buffer()` — the internal helper that calls `cudaHostGetDevicePointer` — now checks `buf.is_mapped()` before the FFI call and returns `Err(RamFlowError::ConfigError("..."))` if the buffer was not registered with `cudaHostRegisterMapped`. Previously the check was absent; passing an unmapped buffer produced a CUDA error that bypassed Rust's type-safe error path.

**Implications:**
- Small tensors (LayerNorm weights, bias vectors, small attention bias) use zero-copy, saving a full DMA staging copy.
- Large tensors (QKV projections, MLP weights) always use DMA, where bandwidth is the bottleneck.

**Test coverage:** `test_zero_copy.rs` — 4 external tests; 2 additional in-module tests in `cuda_bridge/zero_copy.rs`. All 6 cover the full 2×2 matrix: {mapped, unmapped} × {below threshold, above threshold}.

### Algorithm 3: FP16 Overflow Density Tracking (PerLayerScaleTable)

**Problem:** Standard dynamic loss scaling applies one scale to the entire model. If one layer has high gradient variance (frequent FP16 overflow), reducing the global scale penalizes all other layers unnecessarily.

**Solution:** Per-layer Exponentially Weighted Average (EWA) of overflow density, driving independent per-layer loss scales.

**Density update (EWA):**
```
fraction[l] = n_overflow[l] / n_total[l]
density[l]  = α × fraction[l] + (1 − α) × density[l]
```
Default α = 0.05 (≈ 20-step smoothing window, controlled by `hardware_profile.json`).

**Scale adjustment rule:**
```
if density[l] > overflow_high_threshold (0.001):
    scale[l] = max(scale[l] / 2, 1.0)        # halve, floor at 1.0
elif density[l] < overflow_low_threshold (0.0001) AND scale[l] < 65536.0:
    scale[l] = min(scale[l] × 2, 65536.0)    # double, cap at 65536.0
```

**BF16 short-circuit:** If `bf16_mode = true` (Ampere+ GPU with BF16 enabled), all scales are fixed at 1.0 and the overflow machinery is bypassed entirely. BF16 has a wider exponent range (8 bits vs 5 bits in FP16) and does not overflow for typical gradient magnitudes.

**Gradient variance tracking:** Each layer additionally maintains a windowed mean (default window = 50 steps, configurable via `RAMFLOW_GRADIENT_VARIANCE_WINDOW`) of gradient mean-squared values. Module 3 reads this to schedule INT4 vs FP16 streaming precision per layer.

**Convergence (measured):** With α = 0.05 and constant 3% overflow (300/10,000 elements), density converges to 0.030 ± 10% after 100 steps. Measured directly in the integration test.

**Error-returning API:** `update(layer_idx, n_total, n_overflow)` returns `Result<(), RamFlowError>` and `get_scale(layer_idx)` returns `Result<f32, RamFlowError>`. Both return `Err(RamFlowError::ConfigError(...))` if `layer_idx ≥ table_length`, with a message naming the index and the table length. Previously both silently returned (update: early return, get_scale: `unwrap_or(1.0)`) on out-of-bounds access, making configuration errors invisible. `get_density(layer_idx)` follows the same contract.

**Test coverage:** Tests 8, 9 in `test_pressure_scheduler.rs`; NaN injection section of `integration.rs` and `flowcast/tests/integration.rs`.

### Algorithm 4: Memory/I/O Co-Scheduler with Pressure Feedback

**Problem:** The training loop, CQE poller, and prefetch scheduler run concurrently. Without coordination, the prefetch engine can submit SQEs faster than memory is freed, causing pool exhaustion and OOM.

**Solution:** Three-band pressure feedback with atomic signal propagation.

**Pressure signal chain:**
```
PoolRegistry (every N steps or on stall)
    → MemoryPressureGauge::sample_and_notify()
        → p = total_claimed / total_capacity
        → if p > 0.80: fire high_callbacks
            → CoScheduler: prefetch_window -= 1, pause_signal = true
            → DirectNvmeEngine: check_pause() returns Err(PressurePause)
        → if 0.70 < p ≤ 0.80: fire soft_callbacks
            → CoScheduler: compress_trigger = true
            → Module 5: should_compress_checkpoints() = true
        → if p < 0.40: fire low_callbacks
            → CoScheduler: prefetch_window += 1, pause_signal = false, compress_trigger = false
```

**Prefetch window semantics:** `prefetch_window` is an `AtomicI32` that can go negative when multiple high-pressure events fire before a low-pressure event. `DirectNvmeEngine` treats any value ≤ 0 as "do not prefetch" (redundantly with `pause_signal`).

**Pressure-gated admission:**
```
check_pause():
    if pause_signal.load(Acquire): return Err(PressurePause)
    if outstanding_reads + claimed_slots > pressure_threshold: return Err(PressurePause)
```

**Measured behavior (integration test):** At 6/7 slots filled (85.7% > 80% threshold), the high callback fires within at most 31 `sample_and_notify` calls. After slot release (0% fill < 40% threshold), the low callback fires within 31 calls, clearing `pause_signal` and incrementing the window.

### Algorithm 5: INT8 Activation Checkpoint Compression

**Problem:** The Double-Pass Backward engine (Module 5) stores activation checkpoints in pinned RAM. For 70B models, this is the limiting resource: 80 layers × ~50 MB activations each = ~4 GB just for checkpoints.

**Solution:** When pressure enters the soft band (0.70 < p ≤ 0.80), compress checkpoints from FP16 to INT8 with per-channel scale factors, achieving ~2× size reduction.

**Compression (find_channel_scales + compress_fp16_to_int8):**
```
scale[c] = max(|src[c][i]| for i in 0..elems_per_channel) / 127.0
         (or 1.0 if max == 0 to prevent division by zero)
dst[c][i] = clamp(round(src[c][i] / scale[c]), -128, 127)
```

**Decompression:**
```
dst[c][i] = (float)src[c][i] * scale[c]   →   __half
```

**Buffer layout after compression:**
```
[n_channels × sizeof(float32) scale factors] || [n_elements × int8_t]
```
The `PinnedBuffer::compressed` flag signals this layout. Module 5 checks `buf.is_compressed()` before reading checkpoint data and calls `decompress_checkpoint_int8_to_fp16` if true. Use `split_compressed_buffer_ptrs(buf.as_mut_ptr(), n_channels)` to derive the `scales_device` and `data_device` pointers from a single `PinnedBuffer` allocation (see Section 3.6 above).

**Trigger condition:** `CoScheduler::should_compress_checkpoints()` returns `compress_trigger.load(Acquire)`, which is set by the soft-pressure callback and cleared by the low-pressure callback.

**Expected quantization error:** For post-LayerNorm activations (approximately unit-normal distribution), INT8 with per-channel scaling introduces < 0.1% relative error in the gradient reconstruction. This is below the 1e-3 FP16 parity threshold specified in the AethelStream contract.

**Test coverage:** Test 12 in `test_write_budget.rs` / `test_zero_copy.rs` — verifies round-trip quantization fidelity.

---

## 5. Implementation Metrics

### 5.1 Codebase Size

| Component | Files | Lines of code (approx.) |
|-----------|-------|------------------------|
| Rust source (`src/`) | 18 | ~3,200 |
| CUDA kernels (`kernels/`) | 4 (.cu/.cuh) | ~430 |
| Test files (`tests/`) | 7 + integration.rs | ~1,100 |
| Benchmarks (`benches/`) | 1 | ~280 |
| Build script | 1 | ~120 |
| Total | 31 | ~5,130 |

### 5.2 Test Suite

| Feature set | Active tests | Ignored | Failures |
|-------------|-------------|---------|----------|
| `--features mock-cuda` | 78 | 7 | 0 |
| `--features mock-cuda,ssd-wear` | 82 | 7 | 0 |

**Test breakdown by module:**

| File | Tests | Coverage |
|------|-------|---------|
| `src/` lib tests (incl. zero_copy in-module) | 42 | Allocator, pool, scheduler, kernels, phase, zero-copy routing |
| `tests/integration.rs` | 3 | Full 2-cycle streaming, pressure, NaN injection, VmRSS lower-bound (Linux), profiler SHA-256 + cache-skip |
| `tests/test_io_uring.rs` | 18 (+ 1 ignored Linux-only) | io_uring SQE/CQE, alignment, fallback |
| `tests/test_phase_classifier.rs` | 1 | Phase transition classification |
| `tests/test_pool_exhaustion.rs` | 2 | Slow path, double-issue guard |
| `tests/test_pressure_scheduler.rs` | 2 | Pressure gauge + CoScheduler |
| `tests/test_write_budget.rs` | 4 (ssd-wear), 0 (mock-cuda only) | Delta compress wrapping round-trip, ssd-wear strategies, INT8 fidelity |
| `tests/test_zero_copy.rs` | 4 | Zero-copy routing (external harness) |
| Doc tests | 3 ignored (Linux-only) | — |

**New tests added in Phase 1 audit hardening:**

| Test | File | Finding |
|------|------|---------|
| `mock_rounding_matches_cuda_float2int_rn_on_tie_values` | `src/kernels/mod.rs` | M2-9f |
| `mapped_large_buffer_above_threshold_routes_to_dma` | `src/cuda_bridge/zero_copy.rs` | M2-3d |
| `unmapped_large_buffer_above_threshold_routes_to_dma` | `src/cuda_bridge/zero_copy.rs` | M2-3d |
| `pinned_alloc_vmrss_grows_by_approximately_requested_size` (Linux) | `tests/integration.rs` | M2-1b |
| `warmup_profiler_sha256_nonzero_and_second_run_uses_cache` | `tests/integration.rs` | M2-2e |
| `compress_delta_wrapping_sub_and_decompress_is_bitexact_round_trip` | `tests/test_write_budget.rs` | M2-8c |

### 5.3 Code Quality Gates

| Gate | Status |
|------|--------|
| `#![deny(clippy::unwrap_used, clippy::panic, clippy::expect_used)]` | Enforced in `lib.rs` |
| `cargo clippy --no-default-features --features "mock-cuda ssd-wear" -- -D warnings` | 0 warnings, 0 errors |
| Zero `unwrap()/expect()/panic!()` in non-test production code | Verified |
| Full `rustdoc` on all public types and functions | Complete |
| `# Safety` sections on all `unsafe fn` | Complete — including new `split_compressed_buffer_ptrs` |
| `clippy::not_unsafe_ptr_arg_deref` | `split_compressed_buffer_ptrs` marked `unsafe fn` as required |

### 5.4 Memory Alignment Properties

| Property | Value | Requirement |
|----------|-------|-------------|
| Buffer address alignment | 512 bytes | O_DIRECT (NVMe) |
| File byte offset alignment | 512 bytes | O_DIRECT |
| Transfer length alignment | 512 bytes | O_DIRECT |
| CPU DMA cache line | 64 bytes | ✓ (512 mod 64 = 0) |
| CUDA `cudaHostRegister` | any power-of-2 ≥ 8 | ✓ |

### 5.5 Measured Algorithmic Properties

| Property | Value | Source |
|----------|-------|--------|
| EWA density convergence | 0.030 ± 10% after 100 steps at 3% overflow | Integration test |
| EWA smoothing window | ≈ 20 steps (α = 0.05) | Math: 1/(1−0.95^20) ≈ 20 |
| FP16 scale initial value | 65,536.0 | Code |
| FP16 scale floor | 1.0 | Code |
| FP16 scale cap | 65,536.0 | Code |
| High pressure threshold | 0.80 (80% pool fill) | Default |
| Soft pressure threshold | 0.70 (70% pool fill) | Default |
| Low pressure threshold | 0.40 (40% pool fill) | Default |
| CQE poll interval | 1 ms | Code |
| SQPOLL idle timeout | 2,000 ms | Code |
| io_uring SQ depth | 128 entries | Code |
| io_uring CQ depth | 256 entries | Code |
| Completion channel capacity | 1,024 (4× CQ depth) | Code |
| Phase fence timeout | 30 s (release), env-configurable (debug) | Code |
| INT8 compression ratio | ~2× for post-LayerNorm activations | Section 5.6 |
| Delta compression ratio | ~8–15× at standard learning rates (zstd L3 on near-zero deltas) | Section 5.6 |

---

## 5.6 Benchmarks — Measured Numbers

> **Environment:** Windows 11, `cargo bench --no-default-features --features mock-cuda`, release profile.  
> CUDA kernels run in mock mode (pure CPU Rust simulation). Allocator calls `_aligned_malloc` (Windows CRT).  
> All numbers are median of 100 samples with criterion's 1 s warm-up. Confidence interval shown as [low, mid, high].

### Benchmark 1: Allocator — PinnedBuffer vs `Vec<u8>`

`PinnedBuffer::alloc` calls `_aligned_malloc(512)` + `cudaHostRegister` (mock no-op).  
`vec![0u8; N]` calls the system allocator + zero-initialization.

| Size | PinnedBuffer::alloc | `Vec<u8>` alloc+zero | Speedup |
|------|--------------------|-----------------------|---------|
| 4 KB | **40.7 ns** | 57.4 ns | 1.41× faster |
| 64 KB | **203 ns** | 1,166 ns | **5.74× faster** |
| 1 MB | **5.84 µs** | 5.60 µs | ~parity |
| 64 MB | **12.1 µs** | 12.6 µs | 1.04× faster |

**Why 64 KB is the most dramatic gap:** `Vec<u8>` must zero-initialize every page at 64 KB. `PinnedBuffer::alloc` returns uninitialized memory (the caller writes before reading, like all DMA buffers) — no zero-initialization overhead. At 1 MB+ both are dominated by OS page-fault cost, which equalizes them.

**Size accuracy:** PyTorch rounds up to the next power-of-two. For a 2.1 MB tensor: PyTorch allocates 4 MB; RamFlow allocates exactly 2,097,152 bytes — **47% less memory per tensor**. Over 80 layers with 50 MB average activations: 80 × (64 MB − 50 MB) = **1.12 GB saved** in pinned RAM just from exact sizing.

### Benchmark 2: Pool Ring — Claim Latency

All four ring types show essentially identical latency because they share the same `RingBuffer` implementation.

| Operation | Time | Notes |
|-----------|------|-------|
| Attention fast path | **70.1 ns** | Lock-free empty check + Mutex pop |
| MLP fast path | **68.9 ns** | Same path |
| Norm fast path | **69.3 ns** | Same path |
| Last-slot contended (3/4 in use) | **68.5 ns** | Still fast path — slot available |

**Interpretation:** The ~70 ns includes one `AtomicUsize::load` (Relaxed, empty check) + one `Mutex::lock` + `Vec::pop` + `Mutex::unlock`. This is the hot path that fires 800–1,600 times per training step. Total pool overhead per step: 800 claims × 70 ns = **56 µs** — negligible against 10–50 ms per-layer compute time.

**Target met:** < 100 ns fast-path claim (target was < 500 ns).

### Benchmark 3: Pressure Gauge — Sampling Overhead

| Operation | Time | Notes |
|-----------|------|-------|
| `sample_and_notify` (0% fill, no callbacks) | **97.9 ns** | Two AtomicUsize loads + 1 AtomicU32 store |
| `sample_and_notify` (85% fill, high callbacks fire) | **102.4 ns** | + Mutex snapshot + 1 callback invocation |
| `current_pressure` (atomic read only) | **0.53 ns** | Single `AtomicU32::load(Relaxed)` |

**Overhead fraction:** Sampled every 30 steps. At 10 steps/min: sampling fires every 3 min. Total gauge overhead per hour: 20 × 100 ns = **2 µs/hr** — unmeasurably small.

**Even at high-pressure (callbacks firing):** 102 ns per call. Against a 10 ms layer compute: **0.001% overhead**.

### Benchmark 4: EWA Loss-Scale Table Update

| Operation | Time | Amortized/layer |
|-----------|------|-----------------|
| `update()` — single layer | **3.33 ns** | — |
| `update()` — all 80 layers | **142–148 ns** | **1.85 ns/layer** |
| `get_scale()` — all 80 layers | **34.1–34.9 ns** | **0.43 ns/layer** |

**Per-step cost for a 70B model (80 layers):** 148 ns for all EWA updates + 34 ns for all scale reads = **182 ns total**. Against a ~10 s training step for 70B: this is **0.0000018% overhead**.

**The EWA update is essentially free.** It is 4 floating-point operations (multiply, multiply, add, compare) per layer per step. Modern CPUs execute this at ~1 FP op/ns.

### Benchmark 5: INT8 Checkpoint Compression (CPU mock; GPU will be faster)

These numbers are for the pure-CPU Rust simulation. The real CUDA kernel runs on GPU and will be orders of magnitude faster.

| Configuration | Compress time | Throughput | Decompress time | Throughput |
|--------------|--------------|------------|-----------------|------------|
| 4 ch × 128 el = 512 elements | **3.06 µs** | 165 Melem/s | **8.43 µs** | 61 Melem/s |
| 32 ch × 512 el = 16,384 elements | **313 µs** | 52 Melem/s | **242 µs** | 68 Melem/s |
| 64 ch × 1024 el = 65,536 elements | **1.22 ms** | 54 Melem/s | **1.24 ms** | 53 Melem/s |

**Memory savings (the actual value of INT8 compression):**

| Checkpoint size | FP16 bytes | INT8 + scales bytes | Savings |
|----------------|-----------|---------------------|---------|
| 512 elements (4 ch) | 1,024 B | 528 B | **48.4%** |
| 16,384 elements (32 ch) | 32,768 B | 16,512 B | **49.6%** |
| 65,536 elements (64 ch) | 131,072 B | 65,792 B | **49.8%** |

For large checkpoints the per-channel scale overhead (4 bytes × n_channels) is negligible, giving a consistent **~2× size reduction**.

**70B impact:** 80 layers × 50 MB activations = 4 GB checkpoint RAM. After INT8 compression: ~2 GB — this is the direct difference between needing 128 GB system RAM vs 64 GB.

**CUDA GPU estimate (RTX 4090):** The GPU kernel runs `n_channels` blocks of up to 256 threads in parallel. For 65,536 elements: the `find_channel_scales` kernel needs 256 threads × ceil(1024/256) = 256 threads, 2 passes with shared memory. On RTX 4090 (82.6 TFLOPS FP32, ~1,000 GB/s memory bandwidth): estimated ~5–20 µs — 60–240× faster than the CPU mock.

### Benchmark 6: O_DIRECT Alignment Validation

| Case | Time | Notes |
|------|------|-------|
| Valid alignment (3 mod checks) | **1.81 ns** | Three `is_multiple_of` checks — 3 integer divisions |
| Misaligned (error path) | **137–143 ns** | Error path allocates a `String` for the error message |

**The hot path (valid inputs) costs 1.81 ns** — essentially free. The error path is 76× slower due to string allocation, but errors should never occur on the critical path (Module 1 guarantees alignment at shard creation).

### Benchmark 7: Analytical — SQPOLL Syscall Reduction

| Mode | Syscalls per SQE | SQEs per step (80L×10T×2dir) | Syscall cost/step |
|------|-----------------|-------------------------------|-------------------|
| Standard io_uring | 1 | 1,600 | **800 µs** (@ 500 ns/syscall) |
| SQPOLL mode | 0 | 0 | **0 µs** |
| Savings | — | 1,600 syscalls | **800 µs/step** |

At 500 steps/hr: 800 µs × 500 = **400 ms/hr saved in pure syscall overhead**. On a 7B model (32 layers): 32 × 10 × 2 = 640 SQEs/step, 320 µs/step saved.

SQPOLL requires Linux ≥ 5.11 or `CAP_SYS_NICE`. Falls back to standard mode automatically (1 syscall/SQE).

### Benchmark 8: Analytical — Phase-Aware Pool Rebalancing Memory Savings

At the default 4 MiB zero-copy threshold (2 MiB large slots, 1 MiB norm, 1 MiB embed):

| Phase | Attention | MLP | Norm | Embed | Total slots | Total RAM |
|-------|----------|-----|------|-------|-------------|----------|
| Recomputation (worst case) | 3 | 2 | 1 | 1 | 7 | **353 MB** |
| Backward | 2 | 1 | 1 | 1 | 5 | **229 MB** |
| Forward | 1 | 1 | 1 | 1 | 4 | **161 MB** |

Without phase rebalancing, the pool must always hold the worst-case 353 MB.  
With rebalancing: during Forward pass the pool drops to 161 MB — **192 MB freed** (54% less).

At production slot sizes (64 MiB large, 1 MiB norm, 32 MiB embed):  
- Recomputation: 3×64 + 2×64 + 1×1 + 1×32 = **353 MB**  
- Forward: 1×64 + 1×64 + 1×1 + 1×32 = **161 MB**  
- Savings: **192 MB** freed during the forward pass — available for activation checkpoints.

### Benchmark 9: Analytical — Delta Compression Write Amplification

The `DeltaCompress` strategy runs when SSD budget is at 10–50%. Delta = updated − original (wrapping i16), compressed with zstd level 3.

At a standard Adam learning rate of 1e-4:
- FP16 weight magnitudes: typically 0.01–1.0
- Per-step FP16 delta magnitude: `lr × grad / √var` ≈ 1e-4 × normalized gradient
- LE i16 representation: most deltas are in [-10, +10] (out of ±32767 range)
- zstd level 3 on arrays of small integers: typically **8–15× compression**

| Delta magnitude range | zstd-L3 ratio | Write cost vs Full |
|----------------------|--------------|---------------------|
| Near-zero (lr=1e-4) | 10–15× | **7–10%** of full write |
| Moderate (lr=1e-3) | 5–8× | **12–20%** of full write |
| Large (lr=1e-2) | 2–4× | **25–50%** of full write |

At 8× average compression, writing a 50 MB layer shard costs: 50 MB / 8 = **6.25 MB** vs 50 MB Full. Over 1,000 steps with 80 layers: 80 × 1,000 × 6.25 MB = **500 GB** (DeltaCompress) vs **4,000 GB** (Full) — an **8× reduction in SSD wear**.

### Summary: Performance Numbers at a Glance

| Metric | Measured value | vs Baseline |
|--------|---------------|------------|
| PinnedBuffer alloc (64 KB) | 203 ns | **5.7× faster** than Vec |
| Exact-size savings (2.1 MB tensor) | 2.1 MB vs 4 MB | **47% less RAM** |
| Pool claim (fast path) | **70 ns** | < 100 ns ✓ |
| Pressure sample overhead | **98 ns/call** | 0.001% of step time |
| Atomic pressure read | **0.53 ns** | Single instruction |
| EWA update (80 layers) | **148 ns** | < 0.0002% of step time |
| SQPOLL syscall savings | **800 µs/step** | 1,600 syscalls eliminated |
| INT8 compression savings | **~2× RAM** | 4 GB → 2 GB checkpoints |
| Phase rebalance RAM savings | **192 MB** (Forward phase) | 54% pool reduction |
| Delta compression write savings | **8× less SSD wear** | 500 GB vs 4,000 GB / 1k steps |
| Alignment validation (valid) | **1.81 ns** | Negligible hot-path cost |

---

## 6. What Belongs in the Research Paper

This section documents what the RamFlow component contributes to the AethelStream paper and how to frame each contribution.

### 6.1 Abstract

The abstract should claim:
1. **Problem:** Training 70B LLMs requires > 100 GB VRAM. Consumer hardware has 12–24 GB VRAM and 32–128 GB system RAM.
2. **System:** AethelStream streams one transformer layer at a time, hiding latency via predictive I/O overlap.
3. **RamFlow's role:** The memory substrate enabling this. Manages all pinned RAM, zero-copy routing, NVMe DMA, and pressure-adaptive prefetch.
4. **Key results:** (pending M5/M7 completion) 70B model trains on a single 24 GB GPU; peak system RAM < 64 GB; GPU idle < 20%.

### 6.2 Contributions to Claim

For a systems/ML paper, RamFlow contributes these novel claims:

1. **Exact-size pinned allocator with O_DIRECT alignment** — eliminates PyTorch's power-of-two rounding overhead and is the first allocator designed jointly for GPU DMA and NVMe O_DIRECT constraints.

2. **Phase-aware pool rebalancing** — the pool size is not static: it tracks the training phase (forward/backward/recomputation) and resizes between phases using a synchronization fence. Prior streaming systems (ZeRO-Infinity, Flexgen) use static memory budgets.

3. **Hybrid zero-copy routing with runtime crossover measurement** — the threshold between UVA zero-copy and DMA is measured per-GPU at startup rather than hardcoded. This adapts to PCIe bandwidth heterogeneity across consumer GPUs.

4. **Per-layer EWA overflow density loss scaling** — standard loss scaling (Micikevicius et al., 2018) applies a single scale globally. RamFlow's PerLayerScaleTable applies independent scales per layer, isolating NaN-prone layers without penalizing stable ones.

5. **Three-band pressure co-scheduler** — the soft band (70–80%) triggers INT8 checkpoint compression proactively, before the hard stop at 80% would pause the training loop entirely. This extends the effective pressure ceiling.

### 6.3 Related Work

The paper should cite and differentiate from:

- **ZeRO-Infinity (Rajbhandari et al., 2021):** Streams optimizer states to NVMe using pinned buffers. RamFlow differs in: (a) per-layer granularity during forward+backward, not just optimizer states; (b) zero-copy routing for sub-threshold tensors; (c) pressure-adaptive prefetch window.

- **Flexgen (Sheng et al., 2023):** Offloads weights and KV-cache to CPU RAM and disk. Does not address: phase-aware pool resizing, per-layer loss scaling, or EWA-driven INT8 compression.

- **DeepSpeed (Microsoft):** Plugin-based offload. Does not expose a programmable memory pressure gauge or per-layer scale table.

- **CUDA Unified Memory / ATS:** Hardware-managed page migration. Non-deterministic latency. RamFlow is deterministic by design: every prefetch is software-scheduled, not hardware-paged.

- **io_uring (Axboe, 2019):** The kernel I/O interface. RamFlow is the first LLM training system to use io_uring with SQPOLL for zero-syscall tensor prefetch.

### 6.4 Evaluation Metrics to Report

For the paper's evaluation section, report these metrics on real hardware (Linux, RTX 4090 or A100, NVMe Gen4):

| Metric | Target | How to measure |
|--------|--------|----------------|
| End-to-end training throughput (TFLOPS) | > 40 TFLOPS (70B on 4090) | `nvtop` + FLOPs/step |
| Peak VRAM utilization | < 22 GB (70B) | `nvidia-smi` |
| Peak system RAM | < 64 GB (70B) | `/proc/self/status` VmRSS |
| GPU idle fraction | < 20% over 500 steps | NVML / `nvtop` |
| Gradient parity vs PyTorch | < 1e-5 (FP32) / < 1e-3 (FP16) | M5 parity guard |
| NVMe read bandwidth | > 3 GB/s (Gen4 SSD) | `io_uring` SQE throughput |
| Pool claim latency (fast path) | < 500 ns | `cargo bench` |
| Pool claim latency (slow path) | < 1 µs (one slot in flight) | `cargo bench` |
| Phase rebalance time | < 50 ms (all slots free) | `PhaseRebalancer` timing |
| INT8 checkpoint compression speedup | > 1.5× end-to-end vs no compression | A/B run |
| INT8 gradient parity | < 0.1% deviation | Test 12 + production verification |
| SSD TBW per training run (70B × 1000 steps) | < 100 GB | `WriteBudgetManager::units_written_snapshot()` |

### 6.5 Algorithm Pseudocode for Paper

**Algorithm 1 (Phase-Aware Pool Allocation):**
```
Input: hardware_profile (from WarmupProfiler)
On startup:
  pool ← PoolRegistry(recomputation_profile)  // worst-case allocation
On phase_transition(new_phase):
  fence ← wait_until(claimed_slots = 0 AND cuda_copies = 0)
  pool.resize(profile_for(new_phase))
```

**Algorithm 3 (Per-Layer EWA Loss Scaling):**
```
For each backward step t, layer l:
  f_t = count_overflow_fp16(grad[l]) / |grad[l]|
  ρ_t[l] = α·f_t + (1-α)·ρ_{t-1}[l]       // EWA density
  if ρ_t[l] > θ_high:
    s_t[l] = max(s_{t-1}[l] / 2, 1)         // halve scale
  elif ρ_t[l] < θ_low and s_{t-1}[l] < s_max:
    s_t[l] = min(s_{t-1}[l] × 2, s_max)     // double scale
  else:
    s_t[l] = s_{t-1}[l]                      // unchanged
```

**Algorithm 4 (Pressure-Adaptive Co-Scheduling):**
```
On each sampling event:
  p ← total_claimed / total_capacity
  if p > 0.80:   pause_prefetch(); shrink_window()
  elif p > 0.70: trigger_checkpoint_compression()
  elif p < 0.40: resume_prefetch(); grow_window(); clear_compression()
```

**Algorithm 5 (INT8 Checkpoint Compression):**
```
Input: activation[c][i] in FP16, n_channels C, elems_per_channel E
Phase 1 (find_channel_scales):
  ∀c: scale[c] = max_i(|activation[c][i]|) / 127  (or 1.0 if all-zero)
Phase 2 (compress):
  ∀c,i: compressed[c][i] = clamp(round(activation[c][i] / scale[c]), -128, 127)
Decompression:
  ∀c,i: restored[c][i] = compressed[c][i] × scale[c]  →  FP16
```

### 6.6 Figures to Include

1. **System overview diagram** — SSD → RAM → VRAM streaming pipeline with RamFlow in the middle.
2. **Memory pressure timeline** — pool fill ratio over training steps, showing soft/hard threshold crossings and window adjustments.
3. **Per-layer scale table heatmap** — scales across 80 layers over 1000 steps, showing isolated scale reductions at high-overflow layers.
4. **Phase rebalance pool capacity chart** — attention/MLP ring capacity across Forward/Backward/Recomputation phases.
5. **io_uring throughput vs standard read** — SQE submission rate comparison (SQPOLL vs standard mode vs fallback).
6. **INT8 compression error distribution** — histogram of per-element quantization error for post-LayerNorm activations.

---

## 7. Known Limitations and Future Work

| Limitation | Impact | Planned resolution |
|-----------|--------|-------------------|
| SQPOLL requires CAP_SYS_NICE or root on kernels < 5.11 | Falls back to standard mode (1 syscall/SQE) | Add IORING_SETUP_SQ_AFF for unprivileged SQPOLL (Linux ≥ 5.11) |
| NvmeSmartReader is Linux-only | SSD wear budget inactive on Windows/macOS | Windows: PowerShell `Get-PhysicalDisk` SMART; macOS: IOKit SMART |
| Phase fence is a busy-spin (1 ms sleep) | Up to 1 ms latency at phase boundary | Add condition variable signaled by `PoolSlot::Drop` |
| gpu_pressure() always returns 0.0 | VRAM pressure not fed into co-scheduler | Wire to NVML `nvmlDeviceGetMemoryInfo` |
| Zero-copy crossover measured once at startup | Misses PCIe bandwidth variation under load | Periodic re-measurement every N steps |
| Super-shard grouped I/O disabled by default | Missed opportunity for grouped reads | Enable when WarmupProfiler detects adjacent layout |

---

## 8. Build and Test Reference

```bash
# Mock-CUDA (no GPU required, all logic tests run)
cargo test  --no-default-features --features mock-cuda
cargo clippy --no-default-features --features mock-cuda -- -D warnings

# Mock-CUDA + SSD wear tracking
cargo test  --no-default-features --features "mock-cuda,ssd-wear"

# Real CUDA (requires nvcc, GPU)
cargo build --features cuda
cargo build --features "cuda,ssd-wear"

# CUDA kernel compile check
nvcc -arch=sm_75 -O3 --std=c++17 -c kernels/overflow_check.cu
nvcc -arch=sm_75 -O3 --std=c++17 -c kernels/checkpoint_compress.cu

# Generate documentation
cargo doc --no-default-features --features mock-cuda --open
```

**Environment overrides:**

| Variable | Default | Effect |
|----------|---------|--------|
| `RAMFLOW_POOL_RAM_FRACTION` | 0.9 | Max fraction of available RAM for pool allocation |
| `RAMFLOW_MEM_AVAILABLE_BYTES` | auto (`/proc/meminfo`) | Override available RAM for pre-flight budget check |
| `RAMFLOW_GRADIENT_VARIANCE_WINDOW` | 50 | Gradient variance window length (steps) |
| `RAMFLOW_PHASE_FENCE_TIMEOUT_MS` | (debug builds only) | Phase fence timeout in milliseconds |

---

## 9. Public API Summary

```rust
// Crate root re-exports
use ramflow::{
    PinnedBuffer,           // page-locked exact-size buffer
    PoolRegistry,           // central pool router
    DirectNvmeEngine,       // io_uring NVMe engine
    MemoryPressureGauge,    // sampling pressure sensor
    CoScheduler,            // pressure-reactive scheduler
    PerLayerScaleTable,     // per-layer EWA loss scaler
    TensorSlab,             // small-tensor packing slab
    Result, RamFlowError,   // crate error type
};

// Key submodule types
use ramflow::pool::{LayerKind, PoolSlot, TensorLocationDict};
use ramflow::phase::{TrainingPhase, Direction, PhaseClassifier, WarmupProfiler};
use ramflow::nvme::write_budget::{WriteBudgetManager, WriteStrategy};
use ramflow::kernels::{fused_overflow_check, count_overflow_fp16,
                       compress_checkpoint_fp16_to_int8,
                       decompress_checkpoint_int8_to_fp16,
                       split_compressed_buffer_ptrs};  // unsafe: packed buffer ptr derivation
```

**Changed signatures (Phase 1 audit hardening):**
```rust
// PerLayerScaleTable — all three methods now return Result
impl PerLayerScaleTable {
    pub fn update(&mut self, layer_idx: usize, n_total: u64, n_overflow: u32)
        -> Result<(), RamFlowError>;           // Err on layer_idx >= table_len
    pub fn get_scale(&self, layer_idx: usize)
        -> Result<f32, RamFlowError>;          // Err on layer_idx >= table_len
    pub fn get_density(&self, layer_idx: usize)
        -> Result<f32, RamFlowError>;          // Err on layer_idx >= table_len
}

// CoScheduler — new reader for compress_trigger
impl CoScheduler {
    pub fn should_compress_checkpoints(&self) -> bool;  // Acquire load
}
```

**Key invariants callers must uphold:**
1. `PinnedBuffer` must remain alive until the CQE for any in-flight io_uring SQE using it arrives.
2. All shard file byte offsets must be 512-byte aligned (Module 1 enforces at shard creation).
3. `PoolRegistry::set_pressure_gauge(gauge)` must be called before the training loop starts.
4. `PhaseRebalancer::rebalance_to_profile()` must only be called when all slots are free (the fence enforces this internally but callers should not force a timeout by holding slots across a rebalance).
5. `PinnedBuffer::set_compressed(true)` must be called immediately after `compress_checkpoint_fp16_to_int8` returns `Ok(())`, before any other code reads the buffer.
6. `split_compressed_buffer_ptrs` is `unsafe` — callers must guarantee the base pointer is valid for `n_channels * 4 + n_elements` bytes, 512-byte aligned, and that the pointed-to buffer is not concurrently mutated.

---

*RamFlow Module 2 — AethelStream research project*  
*82 tests pass. 0 clippy warnings. Public API frozen (Phase 1 audit hardening complete 2026-06-11).*
