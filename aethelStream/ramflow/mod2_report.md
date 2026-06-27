# AethelStream Module 2: RamFlow Reference

**Version**: 0.1.0  
**Edition**: Rust 2021  
**Purpose**: High-throughput memory orchestration for 7B–70B streaming LLM inference and training

## 1. Module Purpose

RamFlow is the memory management layer of AethelStream. It orchestrates pinned host RAM, NVMe secondary storage, and CUDA GPU memory for streaming transformer inference. The module implements five core algorithms:

1. **Phase-aware predictive pool allocation** — resizes ring buffers based on training phase transitions (forward/backward/recomputation)
2. **Tensor-size-aware hybrid zero-copy routing** — selects between CUDA UVA (for small tensors) and asynchronous DMA (for large) dynamically
3. **Memory/I/O co-scheduler with pressure feedback** — adjusts NVMe prefetch window and triggers checkpoint compression based on pool fill
4. **Tensor slab packing** — merges many small tensors into co-allocated buffers to reduce ring fragmentation
5. **Per-layer exponentially weighted overflow density scaling** — tracks gradient overflow per layer and adjusts loss scales automatically

**Key invariant**: RamFlow maintains predictable peak RAM usage across training phases by profiling each phase once during warmup, caching the profile in `hardware_profile.json`, and rebalancing pool rings at phase boundaries.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Loop (Module 5)                    │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ├─► notify_phase() ──────────────────────────────────────────────┐
     │                                                                  │
     ├─► claim(LayerKind) ────► PoolRegistry ────┬─► RingBuffer ──┐   │
     │                          ├─ Lazy slabs │   │   (Attention)  │   │
     │                          ├─ Scale table│   │                │   │
     │                          └─ Pressure   │   ├─ RingBuffer ──┤   │
     │                                         │   │   (MLP)        │   │
     │                                         │   ├─ RingBuffer ──┤   │
     │                                         │   │   (Norm)       │   │
     │                                         │   └─ RingBuffer ──┘   │
     │                                         │                       │
     ├─► zero_copy_route(buf) ──► ZeroCopyRouter                      │
     │         │                      │                                │
     │         └─ (small + mapped) ──► UVA zero-copy                  │
     │         └─ (else) ───────────► cudaMemcpyAsync DMA            │
     │                                                                  │
     ├─► prefetch(shard_id) ─────► DirectNvmeEngine ◄─────────────┐   │
     │         │                      │                            │   │
     │         │ poll_completions()   │ io_uring ring              │   │
     │         └─────────────────────► ├─ SQE submission           │   │
     │                                 ├─ CQE poller thread        │   │
     │                                 ├─ Checksum verify (xxH3)   │   │
     │                                 └─ Retry/fallback           │   │
     │                                                               │   │
     ├─► update_scale(layer, overflow) ◄─ PerLayerScaleTable       │   │
     │                                      (EWA density tracker)    │   │
     │                                                               │   │
     └─► sample_and_notify() ────────► MemoryPressureGauge ────────┘
                                         │
                                         ├─ High (>0.80)
                                         │  └─► CoScheduler callbacks
                                         │      ├─ pause NVMe prefetch
                                         │      ├─ trigger INT8 compression
                                         │      └─ adjust prefetch_window
                                         │
                                         ├─ Low (<0.40)
                                         │  └─► Resume prefetch
                                         │
                                         └─ Pressure signals back-pressure
```

**Key interactions**:
- **PoolRegistry**: Central registry; owns four pre-allocated RingBuffers (Attention, MLP, Norm, Embedding), lazy slabs, and fallback allocators
- **RingBuffer**: Lock-free ring with atomic claim/release; supports mmap fallback for low-RAM machines
- **TensorSlab**: Packed single allocation for all small tensors in a layer; reduces fragmentation
- **DirectNvmeEngine**: io_uring-backed prefetch engine; submits SQEs, polls CQEs, verifies checksums
- **MemoryPressureGauge**: Real-time pool fill sensor; invokes callbacks at pressure thresholds
- **CoScheduler**: Registers callbacks on gauge; manages pause_signal, prefetch_window, and compression trigger
- **ZeroCopyRouter**: Routes tensors to zero-copy (UVA) or DMA based on size and registration mode
- **PerLayerScaleTable**: EWA overflow density per layer; adjusts loss scales to prevent gradient clipping

---

## 3. Public API

### 3.1 Core Types

#### **PoolRegistry**

Central registry of all pool shards. Routes claim requests to the correct ring.

```rust
pub struct PoolRegistry {
    // Internal: four RingBuffer instances (Attention, MLP, Norm, Embedding)
    // Internal: lazy slab map for low-RAM machines
}

impl PoolRegistry {
    // Construction
    pub fn new(tensor_location_dict: &TensorLocationDict, ...) -> Result<Self>;
    pub fn new_lazy(config: SlabInitMode, embedding_fallback: EmbeddingFallbackMode) -> Result<Self>;
    pub fn with_defaults() -> Result<Self>;

    // Claiming tensors
    pub fn claim(&self, kind: LayerKind) -> Result<PoolSlot>;
    pub fn claim_for_layer(&self, kind: LayerKind, layer_idx: u32) -> Result<PoolSlot>;

    // Slab management
    pub fn ensure_slab_for_layer(&self, layer_idx: u32, dict: &TensorLocationDict) -> Result<()>;
    pub fn slab_for_layer(&self, layer_idx: u32) -> Option<&TensorSlab>;
    pub fn release_lazy_slab(&self, layer_idx: u32);

    // LZ4 eviction cache (feature = "lz4-cache")
    pub fn enable_lz4_cache(&self, max_compressed_bytes: usize);
    pub fn set_recompute_window(&self, window_start: u32, window_end: u32);
    pub fn clear_recompute_window(&self);
    pub fn offer_for_lz4_eviction(&self, layer_idx: u32, slot: PoolSlot, kind: LayerKind, precision: CachePrecision);
    pub fn lz4_cache_telemetry(&self) -> Option<Lz4CacheTelemetry>;

    // Ring inspection
    pub fn claimed_slots_for(&self, kind: LayerKind) -> usize;
    pub fn capacity_for(&self, kind: LayerKind) -> usize;
    pub fn total_claimed_slots(&self) -> usize;
    pub fn total_capacity(&self) -> usize;
    pub fn bytes_allocated(&self) -> usize;

    // Resizing
    pub fn resize_to_profile(&self, profiles: &[PhaseMemoryProfile]) -> Result<()>;

    // Pressure integration
    pub fn set_pressure_gauge(&self, gauge: MemoryPressureGauge);

    // NUMA topology
    pub fn numa_config(&self) -> NumaConfig;
}
```

#### **PoolSlot**

RAII token representing a claimed buffer. Automatically returned to its ring on drop.

```rust
pub struct PoolSlot {
    // Fields are private; manipulated via methods
}

impl PoolSlot {
    // Access
    pub fn buffer(&self) -> &PinnedBuffer;
    pub fn buffer_mut(&mut self) -> &mut PinnedBuffer;
    pub unsafe fn buffer_ptr(&self) -> *mut u8;  // For CUDA/io_uring FFI
    pub fn buffer_len(&self) -> usize;
    pub fn slot_index(&self) -> usize;
}

impl Drop for PoolSlot {
    // Automatically returns buffer to ring
}
```

#### **TensorLocationDict**

Maps `(layer_index, tensor_name)` to on-disk shard location and metadata.

```rust
pub struct TensorLocationDict {
    // Internal: HashMap<(u32, String), TensorInfo>
}

impl TensorLocationDict {
    pub fn load(path: &Path) -> Result<Self>;
    pub fn empty() -> Self;
    pub fn from_json_bytes(bytes: &[u8], base_dir: Option<&Path>) -> Result<Self>;
    pub fn get(&self, layer_idx: u32, name: &str) -> Option<&TensorInfo>;
    pub fn tensors_for_layer(&self, layer_idx: u32) -> impl Iterator<Item = &TensorInfo>;
    pub fn num_layers(&self) -> usize;
}

pub struct TensorInfo {
    pub name: String,
    pub path: PathBuf,           // Absolute path to shard file
    pub byte_offset: u64,        // Offset within shard (must be multiple of 512)
    pub byte_length: usize,      // Length in bytes
    pub shape: Vec<usize>,       // Tensor shape
    pub dtype: String,           // "f16", "bf16", "i8", etc.
    pub xxhash3: Option<u64>,   // xxHash3 digest for integrity check
}

pub enum TensorLocation {
    PinnedHost,
    GpuVram { device: u32 },
    NvmeSpill,
    Transitioning,
}
```

#### **CoScheduler**

Orchestrates tensor evictions and prefetch window control based on memory pressure.

```rust
pub struct CoScheduler {
    // Internal: registered on MemoryPressureGauge callbacks
}

impl CoScheduler {
    pub fn new(gauge: MemoryPressureGauge) -> Result<Self>;

    // Scheduling
    pub fn tick(&self) -> Result<()>;
    pub fn register_tensor(&self, tensor_id: u64, priority: u8);
    pub fn deregister_tensor(&self, tensor_id: u64);

    // State queries
    pub fn is_paused(&self) -> bool;               // High pressure signal
    pub fn prefetch_window(&self) -> i32;          // Current window size
    pub fn should_compress_checkpoints(&self) -> bool;  // Soft pressure signal
    pub fn gauge(&self) -> &MemoryPressureGauge;
}
```

#### **MemoryPressureGauge**

Real-time pool fill sensor with threshold-based callbacks.

```rust
pub struct MemoryPressureGauge {
    // Internal: tracks (claimed_slots + outstanding_reads) / capacity
}

impl MemoryPressureGauge {
    pub fn new(capacity: usize) -> Self;

    // Sampling and feedback
    pub fn sample_and_notify(&self, claimed_slots: usize, outstanding_reads: usize) -> f64;

    // Callback registration
    pub fn register_high_pressure<F: Fn(f64) + Send + Sync + 'static>(&self, callback: F);
    pub fn register_soft_pressure<F: Fn(f64) + Send + Sync + 'static>(&self, callback: F);
    pub fn register_low_pressure<F: Fn(f64) + Send + Sync + 'static>(&self, callback: F);

    // State queries
    pub fn current_pressure(&self) -> f64;
}
```

#### **PerLayerScaleTable**

Per-layer loss scale tracker with exponentially weighted average overflow density.

```rust
pub struct PerLayerScaleTable {
    // Internal: per-layer density, scale, gradient variance, resident flags
}

impl PerLayerScaleTable {
    pub fn new(num_layers: usize, alpha: f32) -> Self;
    pub fn with_thresholds(num_layers: usize, alpha: f32, high: f32, low: f32) -> Self;
    pub fn enable_bf16_mode(&mut self);

    // Updates
    pub fn update(&mut self, layer_idx: usize, n_total: usize, n_overflow: u32) -> Result<()>;
    pub fn update_gradient_variance(&mut self, layer_idx: usize, grad_mean_sq: f32);
    pub fn mark_resident(&mut self, layer_idx: usize, resident: bool);
    pub fn reset_all_scales(&mut self);

    // Queries
    pub fn get_scale(&self, layer_idx: usize) -> Result<f32>;
    pub fn get_density(&self, layer_idx: usize) -> Result<f32>;
    pub fn gradient_variance(&self, layer_idx: usize) -> f32;
    pub fn is_resident(&self, layer_idx: usize) -> bool;
}
```

#### **PinnedBuffer**

Page-locked host memory, visible to both CPU and GPU.

```rust
pub struct PinnedBuffer {
    // Internal: raw pointer, size, registration mode, compression flag
}

impl PinnedBuffer {
    // Constructors
    pub fn alloc(bytes: usize) -> Result<Self>;
    pub fn alloc_mapped(bytes: usize) -> Result<Self>;  // For UVA/zero-copy
    pub fn external_view(ptr: *mut u8, len: usize, is_mapped: bool) -> Self;

    // Allocation strategies (feature-gated)
    #[cfg(feature = "hugepages")]
    pub fn alloc_pinned_huge(bytes: usize) -> Result<Self>;

    #[cfg(feature = "mmap-fallback")]
    pub fn alloc_mmap(bytes: usize) -> Result<Self>;

    // Access
    pub fn as_ptr(&self) -> *const u8;
    pub fn as_slice(&self) -> &[u8];
    pub fn as_mut_slice(&mut self) -> &mut [u8];
    pub fn len(&self) -> usize;
    pub fn is_pinned(&self) -> bool;
    pub fn is_mapped(&self) -> bool;
    pub fn is_compressed(&self) -> bool;
    pub fn set_compressed(&mut self, compressed: bool);

    // NUMA binding (feature = "numa")
    pub fn apply_numa_binding(&self, numa_node: u32) -> Result<()>;
}

pub enum AllocKind {
    Standard,  // posix_memalign
    #[cfg(feature = "hugepages")]
    Huge { mmap_size: usize },
    #[cfg(feature = "mmap-fallback")]
    Mmap { mmap_size: usize },
    External,  // Borrowed; Drop is no-op
}
```

#### **DirectNvmeEngine**

Zero-syscall NVMe I/O engine built on io_uring.

```rust
pub struct DirectNvmeEngine {
    // Internal: io_uring ring, fd table, prefetch engine, CQE poller thread
}

impl DirectNvmeEngine {
    pub fn open(shard_dir: &Path, n_shards: u32) -> Result<Self>;
    pub fn open_with_paths(paths: &[&Path]) -> Result<Self>;

    // Execution
    pub fn start_cqe_poller(&mut self, cpu_core: usize) -> Result<()>;

    // Prefetch
    pub fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: usize,
        dst_buf: &PinnedBuffer,
        token: &PrefetchToken,
    ) -> Result<()>;

    #[cfg(feature = "checksums")]
    pub fn prefetch_with_checksum(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: usize,
        dst_buf: &PinnedBuffer,
        expected_xxh3: u64,
        token: &PrefetchToken,
    ) -> Result<()>;

    // Completion handling
    pub fn completion_rx(&self) -> &Receiver<CqeResult>;
    pub fn poll_completions(&self) -> Result<u32>;

    // Pressure control
    pub fn is_paused(&self) -> bool;
    pub fn set_pause(&self, paused: bool);
    pub fn is_pressure_relieved(&self) -> bool;

    // Warmup
    pub fn prewarm_first_n(&self, n: u32) -> Result<()>;

    // Super-shard coalescing
    pub fn set_super_shard_config(&self, config: SuperShardConfig);
    pub fn super_shard_config(&self) -> SuperShardConfig;
    pub fn prefetch_super_shard(...) -> Result<()>;

    // Write operations (SSD wear budget)
    pub fn write_async(
        &self,
        dst_shard: u32,
        byte_offset: u64,
        src_buf: &PinnedBuffer,
        token: &PrefetchToken,
    ) -> Result<()>;

    // Diagnostics
    pub fn outstanding_reads(&self) -> usize;
    pub fn shard_count(&self) -> usize;
}
```

#### **PhaseClassifier**

Observer interface for training-loop phase transitions.

```rust
pub trait PhaseClassifier: Send + Sync {
    fn current_phase(&self) -> TrainingPhase;
    fn notify_layer_start(&self, layer_idx: u32, direction: Direction);
    fn notify_backward_recompute_start(&self, from_checkpoint: u32, to_layer: u32);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingPhase {
    Forward { layers_in_flight: u32 },
    Backward { checkpoint_interval: u32 },
    Recomputation { window_start: u32, window_end: u32 },
}

pub enum Direction { Forward, Backward }

pub struct DefaultPhaseClassifier {
    // Concrete implementation
}

impl DefaultPhaseClassifier {
    pub fn new(profile_path: PathBuf) -> Result<Self>;
    pub fn with_transition_callback(
        profile_path: PathBuf,
        callback: Arc<dyn Fn(TrainingPhase, TrainingPhase) + Send + Sync>,
    ) -> Result<Self>;
}
```

#### **ZeroCopyRouter**

Selects transfer strategy (UVA vs. DMA) based on tensor size and registration mode.

```rust
pub struct ZeroCopyRouter;

pub enum TransferStrategy {
    ZeroCopy { device_ptr: DevicePointer },
    DmaCopy { stream: CudaStream },
}

impl ZeroCopyRouter {
    pub fn new() -> Self;
    pub fn route(&self, buf: &PinnedBuffer, stream: &CudaStream) -> Result<TransferStrategy>;
    pub fn set_threshold(bytes: usize);
    pub fn threshold() -> usize;
}
```

---

## 4. TensorLocationDict Schema

**Source**: `shard_index.json`, produced by Module 1 during shard creation

**Contract**:
- Loaded at startup by `TensorLocationDict::load(path)` or `from_json_bytes()`
- Maps every tensor to its on-disk shard file, byte offset, and metadata
- Linked to Module 1's `shard_index.json` schema

**Example JSON structure**:

```json
{
  "layers": [
    {
      "index": 0,
      "tensors": [
        {
          "tensor_name": "q_proj",
          "path": "shard_0000.bin",
          "byte_offset": 0,
          "byte_length": 2097152,
          "shape": [32, 64, 1024],
          "dtype": "f16",
          "xxh3": "0x1a2b3c4d5e6f7g8h"
        },
        {
          "tensor_name": "k_proj",
          "path": "shard_0000.bin",
          "byte_offset": 2097152,
          "byte_length": 2097152,
          "shape": [32, 64, 1024],
          "dtype": "f16",
          "xxh3": "0x9a8b7c6d5e4f3g2h"
        }
      ]
    },
    {
      "index": 1,
      "tensors": { ... }  // Object or array format supported
    }
  ]
}
```

**Fields**:
- `index` — Layer index (0-indexed)
- `tensor_name` or map key — Tensor identifier within the layer
- `path` — Relative path to shard file (resolved against JSON directory)
- `byte_offset` — Byte offset within the shard file (must be multiple of 512 for O_DIRECT)
- `byte_length` — Length of the tensor in bytes
- `shape` — Tensor shape as array of dimensions
- `dtype` — Data type string: "f16", "bf16", "i8", "f32", etc.
- `xxh3` — (Optional) xxHash3-64 digest for integrity verification when `checksums` feature is active

**Linking to M1**: Module 1's `shard_index.json` and RamFlow's `TensorLocationDict` are **identical**. The dict is a zero-copy parsed view of that JSON file, used by all downstream stages to locate tensors without re-parsing.

---

## 5. Allocator Tiers

RamFlow implements a **multi-tier allocator** that selects the best strategy per allocation size and hardware capability.

```
┌────────────────────────────────────────────────────────────┐
│              Allocator Tier Selection                       │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Size >= HUGEPAGE_THRESHOLD (2 MiB) ┐                      │
│  AND hugepages feature active        ├─► Huge:             │
│  AND size >= PAGE_SIZE (4 KiB)      │    mmap + MADV_HUG.. │
│                                      │    (Linux only)      │
│                                      ┘                      │
│                                                              │
│  Size < HUGEPAGE_THRESHOLD ───────────────────┐            │
│  OR hugepages inactive                         ├─► Standard:│
│  OR not on Linux                               │  posix_.. │
│                                                 ┘           │
│  (On Windows: _aligned_malloc)                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ If mmap-fallback feature active & pool stalled:     │   │
│  │                                                      │   │
│  │ Allocate via mmap + MADV_SEQUENTIAL                 │   │
│  │ (pageable; DMA needs staging copy)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ NUMA binding (if numa feature active):              │   │
│  │                                                      │   │
│  │ Call mbind(MPOL_BIND) on pinned allocation          │   │
│  │ Binds buffer to GPU's NUMA node (default: -1=any)   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ CUDA registration:                                  │   │
│  │                                                      │   │
│  │ cudaHostRegisterDefault (DMA only)                  │   │
│  │   OR                                                 │   │
│  │ cudaHostRegisterMapped (DMA + UVA)                  │   │
│  │   for zero-copy slab packer (Algorithm 3)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

**Allocation functions**:

| Function | Alignment | Registration | Use case |
|----------|-----------|--------------|----------|
| `PinnedBuffer::alloc(size)` | 512 bytes | DMA only | Standard ring slots |
| `PinnedBuffer::alloc_mapped(size)` | 512 bytes | DMA + UVA | Slab packer, zero-copy |
| `PinnedBuffer::alloc_pinned_huge(size)` | 512 bytes | DMA only | Large slots >= 2 MiB (hugepages) |
| `PinnedBuffer::alloc_mmap(size)` | 512 bytes | None (pageable) | Low-RAM fallback (feature = "mmap-fallback") |

**Why 512-byte alignment?**
- O_DIRECT requires buffer address, offset, and length to be multiples of the NVMe sector size (512 bytes)
- Every `io_uring` read can be issued with O_DIRECT without EINVAL
- 512 is a power of two and satisfies both `posix_memalign` and Windows `_aligned_malloc` constraints

**Deallocator order (CRITICAL)**:
1. `cudaHostUnregister(&ptr)` — tell CUDA driver to release pin
2. `libc::free(ptr)` or platform-specific free — release to OS

**Reversing this order is undefined behavior**: the OS releases pages to the OS after CUDA still holds a reference, leading to silent corruption.

---

## 6. Pool Subsystem

The pool consists of four pre-allocated **RingBuffer** instances (Attention, MLP, Norm, Embedding) and optional **TensorSlab** packers.

### 6.1 RingBuffer

Lock-free ring of pre-allocated buffers. Uses atomic head/tail counters to track claims.

```rust
pub struct RingBuffer {
    head: AtomicUsize,        // Total claims ever made
    tail: AtomicUsize,        // Total returns ever made
    active_claims: AtomicUsize,  // Real-time gauge (Relaxed ordering)
    state: Mutex<ResizableState>,
}

struct ResizableState {
    free_queue: VecDeque<(usize, ManuallyDrop<PinnedBuffer>)>,
    capacity: usize,
    slot_bytes: usize,
}
```

**Invariant**: `available_slots = tail - head`. When `available_slots > 0`, there are free buffers ready to claim.

**Operations**:
- `try_claim()` — fast path; returns `Some(PoolSlot)` if a buffer is available, else `None`
- `claim_blocking()` — blocks the calling thread until a buffer is available (used by slow path)
- `release(slot_index, buffer)` — called by `PoolSlot::drop`; re-inserts buffer into free queue
- `resize(new_capacity)` — atomically replaces the entire allocation (used at phase boundaries)

**Synchronization**:
- `head` CAS: `AcqRel` on success (makes prior writes visible to new claimer)
- `tail` fetch_add: `Release` (makes returned buffer visible to next claimer)
- `active_claims` fetch_add/fetch_sub: `Relaxed` (best-effort gauge only)
- `free_queue`: guarded by `Mutex`; resizes happen under the lock

### 6.2 PoolSlot

RAII token representing a claimed buffer. Automatically returned to its ring on `drop`.

```rust
pub struct PoolSlot {
    ring: Option<Arc<RingBuffer>>,
    slot_index: usize,
    buffer: ManuallyDrop<PinnedBuffer>,
}
```

**Two construction paths**:
1. `PoolSlot::pooled(ring, slot_index, buffer)` — from a ring claim
2. `PoolSlot::overflow(buffer)` — slow-path overflow; freed directly on drop (not returned to ring)

**Drop implementation**:
```rust
impl Drop for PoolSlot {
    fn drop(&mut self) {
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        if let Some(ring) = &self.ring {
            ring.release(self.slot_index, ManuallyDrop::new(buffer));
        } else {
            drop(buffer);  // Overflow: free immediately
        }
    }
}
```

### 6.3 TensorSlab

Packed single allocation for all small tensors in a layer. Reduces fragmentation by co-allocating many tensors in one buffer.

```rust
pub struct TensorSlab {
    slot: PoolSlot,
    offsets: HashMap<String, usize>,  // tensor_name -> offset within slot
    total_size: usize,
}

impl TensorSlab {
    pub fn get(&self, tensor_name: &str) -> Option<&[u8]>;
    pub fn get_mut(&mut self, tensor_name: &str) -> Option<&mut [u8]>;
}
```

**Layout**:
```
┌─────────────────────────────────────────────┐
│ PoolSlot buffer                             │
├─────────────────────────────────────────────┤
│ Tensor A (offset 0)   | Tensor B | Tensor C │ ...
└─────────────────────────────────────────────┘
```

**Allocation strategy**:
- Built during phase startup from `TensorLocationDict`
- All small tensors in layer 0 are merged into slab 0
- Lookup is O(1) via HashMap
- Reduces peak ring pressure during startup (one claim instead of 100 claims for 100 small tensors)

### 6.4 Pool Lifecycle

```
1. STARTUP
   ├─ Load TensorLocationDict from shard_index.json
   ├─ Profiler runs 5 training steps → hardware_profile.json
   │  (or load cached profile if model SHA256 matches)
   ├─ PoolRegistry constructed with ring capacities from profile
   ├─ Four RingBuffers allocated (Attention, MLP, Norm, Embedding)
   ├─ Lazy slabs initialized (Eager or Lazy mode)
   └─ Pressure gauge created; registered with CoScheduler

2. FORWARD PASS
   ├─ PhaseClassifier notifies "Forward { layers_in_flight: 2 }"
   ├─ Training loop: claim(LayerKind::Attention) for layer 0
   │  ├─ Fast path: RingBuffer::try_claim() succeeds
   │  └─ If fails: SlowPathAllocator::borrow_or_allocate()
   ├─ Prefetch layer 1 while layer 0 computes
   ├─ release() when layer 0 done
   └─ Repeat for layers 1..L-1

3. BACKWARD PASS
   ├─ PhaseClassifier notifies "Backward { checkpoint_interval: 1 }"
   ├─ claim() for recomputation / optimizer state tensors
   ├─ Different ring load profile (more MLP/Norm, fewer Attention)
   ├─ RingBuffer::resize() invoked if profile recommends rebalancing
   └─ Release as layers are freed

4. RECOMPUTATION WINDOW
   ├─ PhaseClassifier notifies "Recomputation { window_start: i, window_end: j }"
   ├─ Peak RAM: window_end - window_start + 2 slots in flight
   ├─ CoScheduler triggers LZ4 eviction (if feature = "lz4-cache")
   ├─ Checkpoint compression triggered (soft pressure signal)
   └─ After window: PhaseClassifier transitions back to Backward

5. SHUTDOWN
   ├─ All PoolSlots dropped
   ├─ RingBuffers deallocated
   └─ ResizableState drop → all PinnedBuffers freed (CUDA unregister + libc::free)
```

---

## 7. Scheduler & CoScheduler

### 7.1 MemoryPressureGauge

Real-time pool fill sensor. Samples at each training step and invokes threshold-based callbacks.

```rust
pub struct MemoryPressureGauge {
    capacity: usize,
    callbacks_high: Arc<Mutex<Vec<...>>>,
    callbacks_soft: Arc<Mutex<Vec<...>>>,
    callbacks_low: Arc<Mutex<Vec<...>>>,
    current_pressure: AtomicU32,  // Stores pressure * 1_000 (to avoid floats)
}

impl MemoryPressureGauge {
    pub fn sample_and_notify(&self, claimed_slots: usize, outstanding_reads: usize) -> f64 {
        let fill_ratio = (claimed_slots + outstanding_reads) as f64 / self.capacity as f64;
        // Invoke callbacks if crossing thresholds
        if fill_ratio > 0.80 { call_high_callbacks(fill_ratio); }
        else if fill_ratio > 0.70 { call_soft_callbacks(fill_ratio); }
        else if fill_ratio < 0.40 { call_low_callbacks(fill_ratio); }
        fill_ratio
    }
}
```

**Threshold definitions**:
- **High** (> 0.80): Pool is nearly full; pause NVMe prefetch, prepare for emergency offload
- **Soft** (0.70–0.80): Pressure rising; trigger INT8 checkpoint compression
- **Low** (< 0.40): Pool has recovered; resume prefetch, clear compression flag

### 7.2 CoScheduler

Registers callbacks on the gauge to manage prefetch window and compression trigger.

```rust
pub struct CoScheduler {
    gauge: MemoryPressureGauge,
    prefetch_window: Arc<AtomicI32>,      // Initial: 2
    pause_signal: Arc<AtomicBool>,        // Set by high-pressure
    compress_trigger: Arc<AtomicBool>,    // Set by soft-pressure
    tensor_registry: Mutex<BTreeMap<u64, u8>>,  // Priority registry
}

impl CoScheduler {
    pub fn new(gauge: MemoryPressureGauge) -> Result<Self> {
        // Register callbacks:
        gauge.register_high_pressure(|_pressure| {
            prefetch_window.fetch_sub(1, AcqRel);
            pause_signal.store(true, Release);
        });

        gauge.register_soft_pressure(|_pressure| {
            compress_trigger.store(true, Release);
        });

        gauge.register_low_pressure(|_pressure| {
            prefetch_window.fetch_add(1, AcqRel);
            pause_signal.store(false, Release);
            compress_trigger.store(false, Release);
        });

        Ok(...)
    }
}
```

**Prefetch window policy**:
- Starts at 2 (prefetch up to 2 layers ahead)
- Decrements on high pressure (minimum 0)
- Increments on low pressure (maximum unbounded, typically 4–8)
- `DirectNvmeEngine::prefetch()` reads this before submission and refuses if `window <= 0` or `pause_signal` is set

**Compression trigger**:
- Set to `true` by soft-pressure callback (0.70–0.80)
- Cleared by low-pressure callback (< 0.40)
- Module 5 reads this via `should_compress_checkpoints()` every backward step
- If `true`, activation checkpoints are compressed to INT8 before the next layer streams in

### 7.3 PerLayerScaleTable (Algorithm 6)

Tracks per-layer exponentially weighted average (EWA) of gradient overflow density. Adjusts loss scales to prevent NaN/Inf propagation.

```rust
pub struct PerLayerScaleTable {
    density: Vec<f32>,                  // EWA overflow fraction
    scale: Vec<f32>,                    // Current loss scale per layer
    gradient_variance: Vec<GradientVarianceWindow>,  // Windowed mean for Module 3
    resident: Vec<bool>,                // Hot-set flags
    alpha: f32,                         // EWA decay (default 0.05 ≈ 20-step window)
    overflow_high_threshold: f32,       // Halve scale if > (default 0.001)
    overflow_low_threshold: f32,        // Double scale if < (default 0.0001)
    bf16_mode: bool,                    // If true, fix all scales at 1.0
}

impl PerLayerScaleTable {
    pub fn update(&mut self, layer_idx: usize, n_total: usize, n_overflow: u32) -> Result<()> {
        if n_total == 0 { return Ok(()); }
        if self.bf16_mode { return Ok(()); }  // BF16: skip scaling

        let fraction = n_overflow as f32 / n_total as f32;
        let new_density = self.alpha * fraction + (1.0 - self.alpha) * self.density[layer_idx];
        self.density[layer_idx] = new_density;

        if new_density > self.overflow_high_threshold {
            self.scale[layer_idx] = (self.scale[layer_idx] * 0.5).max(1.0);
        } else if new_density < self.overflow_low_threshold && self.scale[layer_idx] < 65536.0 {
            self.scale[layer_idx] = (self.scale[layer_idx] * 2.0).min(65536.0);
        }
        Ok(())
    }

    pub fn reset_all_scales(&mut self) {
        if !self.bf16_mode {
            for scale in &mut self.scale {
                *scale = 65536.0;
            }
        }
    }
}
```

**EWA formula**:
```
density[t] = α * fraction[t] + (1 - α) * density[t-1]
```
With α = 0.05, a sudden spike in overflow contributes only 5% to the new density; historical average contributes 95%.

**Scale adjustment**:
- If `new_density > 0.001` (0.1% overflow): halve scale (e.g., 65536 → 32768)
- If `new_density < 0.0001` (0.01% overflow) and scale < 65536: double scale (e.g., 1 → 2)
- Bounds: scale stays in [1.0, 65536.0]

**BF16 short-circuit**:
- When GPU is Ampere+ and BF16 mode is active, fix all scales at 1.0
- BF16 has native overflow immunity (NaN/Inf → 0 silently); scaling is unnecessary

---

## 8. Phase Classifier

Tracks training-phase transitions (Forward → Backward → Recomputation → Backward …).

```rust
pub enum TrainingPhase {
    Forward { layers_in_flight: u32 },  // 2 layers max
    Backward { checkpoint_interval: u32 },
    Recomputation { window_start: u32, window_end: u32 },
}

pub enum Direction { Forward, Backward }

pub trait PhaseClassifier: Send + Sync {
    fn current_phase(&self) -> TrainingPhase;
    fn notify_layer_start(&self, layer_idx: u32, direction: Direction);
    fn notify_backward_recompute_start(&self, from_checkpoint: u32, to_layer: u32);
}

pub struct DefaultPhaseClassifier {
    current_phase: Mutex<TrainingPhase>,
    checkpoint_interval: u32,
    transition_callback: Option<Arc<dyn Fn(TrainingPhase, TrainingPhase) + Send + Sync>>,
}
```

**Module 5 responsibilities**:
- Call `notify_layer_start(layer_idx, Direction::Forward)` at the start of each forward layer
- Call `notify_layer_start(layer_idx, Direction::Backward)` at the start of each backward layer
- Call `notify_backward_recompute_start(checkpoint_idx, to_layer)` when a recomputation window opens

**Transition callback**:
- Receives `(old_phase, new_phase)` when phase changes
- Used by pool rebalancer to resize rings at phase boundaries
- Can be `None` (no callback needed for simple use cases)

---

## 9. NVMe Passthrough

### 9.1 DirectNvmeEngine

Zero-syscall I/O engine using `io_uring` for batch submission and completion handling.

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│ Training Loop (Module 5)                            │
│  ├─ prefetch(shard_id, offset, size, dst, token)  │
│  └─ poll_completions() → CqeResult                  │
└───────────────────┬─────────────────────────────────┘
                    │
      ┌─────────────┴──────────────┐
      │                            │
  ┌───▼──────────────┐   ┌────────▼──────────┐
  │ Submission Queue │   │ Completion Ring   │
  │ (SQE)            │   │ (CQE)             │
  │ - read SQE       │   │ - result (bytes)  │
  │ - write SQE      │   │ - error code      │
  │ - uring_cmd (NVMe)   │ - user_data       │
  └────────┬─────────┘   └──────────────────┘
           │ io_uring system call
           │
  ┌────────▼──────────────────────────────────┐
  │ Linux Kernel                              │
  │ ├─ io_uring subsystem                     │
  │ ├─ NVMe block layer                       │
  │ └─ NVMe driver                            │
  └───────────────────────────────────────────┘
```

### 9.2 SQE128 Ring & io_uring Setup

**Ring configuration**:
- **SQ (Submission Queue)**: 128 entries
- **CQ (Completion Queue)**: 256 entries (2× SQ, per io_uring convention)
- **Flags**: `IORING_SETUP_CQSIZE` (explicit CQ size)

**Optional SQPOLL** (kernel >= 5.4):
- Polls the SQ without sleeping; kernel thread handles submissions
- Reduces context switches but increases CPU usage
- Fallback to regular mode if SQPOLL unavailable

**CQE poller thread**:
- Spawned after `start_cqe_poller(cpu_core)` is called
- Pins to `cpu_core` (e.g., core 2 for low-latency, off-core for training core)
- Drains CQE ring and sends results via completion channel to training loop

### 9.3 Prefetch Submission

```rust
pub fn prefetch(
    &self,
    shard_id: u32,
    byte_offset: u64,
    length: usize,
    dst_buf: &PinnedBuffer,
    token: &PrefetchToken,
) -> Result<()> {
    // Check pause signal
    if self.pause_signal.load(Acquire) {
        return Err(RamFlowError::PressurePause(shard_id));
    }

    // Check prefetch window
    if self.prefetch_engine.should_prefetch_window_close() {
        return Err(RamFlowError::PressurePause(shard_id));
    }

    // Validate O_DIRECT alignment
    if byte_offset % 512 != 0 || length % 512 != 0 {
        return Err(RamFlowError::ConfigError("misaligned I/O"));
    }

    // Submit read SQE
    self.prefetch_engine.submit_read(shard_id, byte_offset, length, dst_buf, token)?;
    self.outstanding_reads.fetch_add(1, Relaxed);
    Ok(())
}
```

### 9.4 CQE Completion Handling

**Completion result codes**:
- `result > 0`: Successful read; `result` = bytes transferred
- `result < 0`: I/O error; `result` = negative errno

**Error classification**:
```rust
pub fn classify_cqe_error(res: i32) -> CqeErrorKind {
    match -res {
        4 | 11 | 16 => CqeErrorKind::Transient,   // EINTR, EAGAIN, EBUSY
        5 | 19 => CqeErrorKind::MediaError,        // EIO, ENODEV
        n => CqeErrorKind::Unknown(n),
    }
}
```

**Retry policy**:
- **Transient** errors (EAGAIN, EINTR): exponential backoff retry (up to 3 times)
- **Media** errors (EIO, ENODEV): fail immediately with `RamFlowError::MediaError`
- Unknown errors: fail with best guess (non-retriable)

### 9.5 Write-Budget Manager (Feature: ssd-wear)

Limits NVMe write amplification by tracking SMART "Data Units Written" counter and switching write strategies.

**SMART unit convention**:
- 1 unit = 1000 × 512-byte LBA sectors = 512,000 bytes
- NVMe spec § 6.1.4 defines this standard

**Strategy selection**:
| Remaining Budget | Strategy | Notes |
|------------------|----------|-------|
| > 50% | `Full` | Write complete shard update |
| 50% to 10% | `DeltaCompress` | Compute delta, compress with zstd, write delta only |
| < 10% | `Deferred{batch_size}` | Accumulate N layers; batch-flush for wear leveling |

**Delta compression**:
- Compute `delta = updated - original`
- Compress delta with zstd (typical ratio 8–15×)
- Store as `layer_NNNN.delta.zstd` alongside original shard
- Module 1's ShardLoader detects `.delta.zstd` files and applies them on load
- Saves up to 15× in write bytes vs. full write

**Write-budget callback**:
- Invoked when crossing thresholds (50%, 10%)
- Can trigger alerts or graceful slowdown

### 9.6 NVMe Block-Layer Bypass (Feature: nvme-passthrough)

When kernel >= 6.0 and `nvme-passthrough` feature is active: use `IORING_OP_URING_CMD` to bypass the NVMe block layer and submit commands directly to the NVMe controller.

**Benefits**:
- Lower latency (no block-layer stack)
- Higher throughput (direct command path)

**Fallback**:
- If kernel < 6.0 or character device unavailable: transparently fall back to standard `IORING_OP_READ`

### 9.7 DirectStorage (Feature: direct-storage, Windows only)

Windows DirectStorage API for zero-copy GPU reads via `IDStorageQueue`.

```rust
#[cfg(feature = "direct-storage")]
pub enum DirectStorageCapability {
    Available { version: u32 },
    Unavailable,
}

#[cfg(feature = "direct-storage")]
pub fn probe_direct_storage() -> DirectStorageCapability;

#[cfg(feature = "direct-storage")]
pub fn alloc_windows_ds_compatible(size: usize) -> Result<PinnedBuffer>;
```

**4096-byte alignment requirement**:
- `DSTORAGE_REQUEST_DESTINATION_BUFFER` path requires 4 KiB alignment
- `alloc_windows_ds_compatible()` enforces this; standard `alloc()` gives 512-byte alignment only

---

## 10. CUDA Bridge

### 10.1 Stream Management

```rust
pub struct CudaStream {
    // Private: raw CUDA stream handle
}

impl CudaStream {
    pub fn new() -> Result<Self>;
    pub fn as_raw(&self) -> *mut std::os::raw::c_void;
    pub fn synchronize(&self) -> Result<()>;
}
```

**Used by**:
- ZeroCopyRouter for UVA pointer lookup and DMA copy submissions
- Kernel wrappers for overflow checking and checkpoint compression
- Module 5 for training kernel launches

### 10.2 Zero-Copy Routing (Algorithm 3)

```rust
pub struct ZeroCopyRouter;

pub enum TransferStrategy {
    ZeroCopy { device_ptr: DevicePointer },
    DmaCopy { stream: CudaStream },
}

impl ZeroCopyRouter {
    pub fn route(&self, buf: &PinnedBuffer, stream: &CudaStream) -> Result<TransferStrategy> {
        // If mapped AND small: zero-copy (UVA)
        if buf.is_mapped() && buf.len() < Self::threshold() {
            let device_ptr = unsafe { cuda_host_get_device_pointer(buf.as_ptr())? };
            return Ok(TransferStrategy::ZeroCopy {
                device_ptr: unsafe { DevicePointer::from_raw(device_ptr as *mut u8) },
            });
        }

        // Otherwise: DMA copy
        let device_ptr = unsafe { cuda_malloc_async(buf.len(), stream.as_raw())? };
        unsafe {
            cuda_memcpy_async_host_to_device(
                device_ptr,
                buf.as_ptr() as *const c_void,
                buf.len(),
                stream.as_raw(),
            )?;
        }
        Ok(TransferStrategy::DmaCopy { stream: CudaStream::new()? })
    }

    pub fn set_threshold(bytes: usize);  // Default: 4 MiB
}
```

**Decision logic**:
- **Small + Mapped** (< 4 MiB, registered with `cudaHostRegisterMapped`): zero-copy via UVA
  - GPU reads pinned host memory directly
  - Latency: ~8 μs + 1 ns/byte
  - Best for <4 MiB tensors
- **Large or Unmapped**: async DMA to VRAM
  - GPU allocates temporary VRAM buffer
  - Async copy: host → GPU (cudaMemcpyAsync)
  - Latency: ~30 μs + 0.5 ns/byte (better throughput)
  - Best for >4 MiB tensors

### 10.3 Kernel Wrappers

**Overflow checking**:
```rust
pub fn fused_overflow_check(
    grad_device: *const u16,
    n_elements: usize,
    stream: &CudaStream,
) -> Result<bool>;

pub fn count_overflow_fp16(
    grad_device: *const u16,
    n_elements: usize,
    stream: &CudaStream,
) -> Result<u32>;
```

**Checkpoint compression (INT8)**:
```rust
pub fn compress_checkpoint_fp16_to_int8(
    src_device: *const u16,
    dst_device: *mut i8,
    scales_device: *mut f32,
    n_channels: usize,
    elems_per_channel: usize,
    stream: &CudaStream,
) -> Result<()>;

pub fn decompress_checkpoint_int8_to_fp16(
    src_device: *const i8,
    dst_device: *mut u16,
    scales_device: *const f32,
    n_channels: usize,
    elems_per_channel: usize,
    stream: &CudaStream,
) -> Result<()>;
```

**Packed buffer layout**:
```
┌──────────────────────────────┬───────────────────────────┐
│ n_channels × f32 scales      │ n_channels × elems × i8   │
│ (per-channel factors)        │ (quantised values)        │
└──────────────────────────────┴───────────────────────────┘
```

---

## 11. Checksums

**Feature**: `checksums`

**Integration**: xxHash3-64 per-shard integrity verification

```rust
pub fn prefetch_with_checksum(
    &self,
    shard_id: u32,
    byte_offset: u64,
    length: usize,
    dst_buf: &PinnedBuffer,
    expected_xxh3: u64,  // From TensorLocationDict
    token: &PrefetchToken,
) -> Result<()>;
```

**Verification flow**:
1. Caller provides expected digest from `TensorInfo::xxhash3`
2. `prefetch_with_checksum()` registers digest in registry
3. After successful read, `poll_completions()` computes xxHash3 of received bytes
4. Mismatch triggers `RamFlowError::ShardCorrupted { shard_id, expected, got }`

**Overhead**:
- xxHash3 throughput: ~30 GB/s on modern CPUs
- Negligible vs. NVMe read latency (~1 GB/s)

**Error handling**:
```rust
pub enum RamFlowError {
    ShardCorrupted {
        shard_id: u32,
        expected: u64,
        got: u64,
    },
    MediaError {
        shard_id: u32,
        errno: i32,
    },
}
```

---

## 12. Feature Flags

| Feature | Default | Purpose |
|---------|---------|---------|
| `cuda` | Yes | Real CUDA GPU path; requires `nvcc` at build time |
| `mock-cuda` | No | No-op stubs for CI; mutually exclusive with `cuda` |
| `hugepages` | No | mmap + MADV_HUGEPAGE for allocations ≥ 2 MiB (Linux) |
| `numa` | No | NUMA-aware allocation via mbind(MPOL_BIND) |
| `nvme-passthrough` | No | NVMe block-layer bypass via IORING_OP_URING_CMD (Linux ≥ 6.0) |
| `io-uring-use-split` | No | io_uring split() fallback (compatibility mode; slower) |
| `libaio-fallback` | No | Blocking pread/pwrite fallback (when io_uring unavailable) |
| `mmap-fallback` | No | Graceful degradation: mmap pool slots when pinned RAM exhausted |
| `lz4-cache` | No | LZ4-compressed eviction tier for Recomputation windows |
| `ssd-wear` | No | SMART-tracked write-budget manager with delta compression |
| `checksums` | No | xxHash3 per-shard integrity verification |
| `direct-storage` | No | Windows DirectStorage API (zero-copy GPU reads) |
| `python-ffi` | No | Python bindings (PyO3) |

---

## 13. Benchmarks

**Command**: `cargo bench --no-default-features --features mock-cuda`

**Groups** (in `benches/ramflow_bench.rs`):

| Benchmark | Sizes | Metric |
|-----------|-------|--------|
| `allocator` | 4 KB, 64 KB, 1 MB, 64 MB | PinnedBuffer vs. Vec allocation time |
| `pool/claim` | 1–128 slots | Fast-path claim latency |
| `pressure/sample` | — | MemoryPressureGauge::sample_and_notify() overhead |
| `ewa/update` | 80 layers | PerLayerScaleTable::update() throughput |
| `int8/compress` | 512, 16384 elements | Checkpoint compression throughput |
| `delta/compress` | 64 KB weights | zstd delta compression ratio and speed |
| `alignment/validate` | — | Direct I/O alignment guard overhead |
| `hugepage/alloc` | 2 MB, 64 MB | Hugepage allocation speed (feature-gated) |
| `numa/binding` | — | mbind() overhead (feature-gated) |
| `lz4/evict` | 1 MB, 10 MB | LZ4 cache eviction throughput (feature-gated) |
| `xxh3/throughput` | 1 KB, 64 KB, 1 MB | xxHash3 checksum speed (feature-gated) |

**Typical results** (on Ryzen 7950X, RTX 4090):
- `PinnedBuffer::alloc` 64 MB: ~50 μs (vs. `Vec`: ~3 μs; higher because of CUDA registration)
- `RingBuffer::try_claim` (hit): ~50 ns
- `MemoryPressureGauge::sample_and_notify`: ~200 ns
- `PerLayerScaleTable::update`: ~50 ns per layer
- xxHash3 throughput: ~30 GB/s (1-2% of peak NVMe read bandwidth)

---

## 14. Integration

### 14.1 Upstream: Module 1 (Shard Engine)

**What M1 provides**:
- `shard_index.json` (loaded by RamFlow as `TensorLocationDict`)
- Shard files: `shard_NNNN.bin` (tensor data, aligned to 512 bytes)
- Optional xxHash3 digests in `shard_index.json` (when `checksums` feature is active in M1)

**RamFlow's contract**:
- Load `shard_index.json` at startup
- Validate that shard paths exist and are readable
- Prefetch shards on demand via `DirectNvmeEngine::prefetch()`
- Return shards to NVMe after use

### 14.2 Downstream: Module 3 (FlowCast)

**What M3 expects from RamFlow**:
- Consistent pool semantics: `claim(LayerKind)` → `PoolSlot`
- `PoolSlot::buffer()` is live and valid until dropped
- Phase notifications (Forward/Backward/Recomputation) for prefetch scheduling
- Pressure signals (pause_signal, prefetch_window) to throttle prefetch
- Per-layer scale table (loss scale, overflow density, gradient variance, resident flag)

**Module 3 responsibilities**:
- Call `PhaseClassifier::notify_*()` at phase boundaries
- Call `MemoryPressureGauge::sample_and_notify()` every training step
- Read `PerLayerScaleTable::get_scale()` before gradient scaling
- Call `DirectNvmeEngine::prefetch()` for layer prefetch
- Poll `DirectNvmeEngine::completion_rx()` and call `poll_completions()`

### 14.3 Downstream: Module 5 (DoublePass)

**What M5 expects**:
- `PerLayerScaleTable::get_scale()`, `update()`, reset
- `CoScheduler::should_compress_checkpoints()` signal
- `ZeroCopyRouter` for tensor routing strategy
- Kernel wrappers: overflow check, INT8 compression/decompression

**M5 responsibilities**:
- Call `PerLayerScaleTable::update(layer, n_overflow)` after each layer's forward/backward
- Check `should_compress_checkpoints()` before gradient checkpointing
- Use `ZeroCopyRouter::route()` to select DMA vs. zero-copy per tensor
- Call INT8 kernel wrappers for checkpoint compression

---

## 15. Test Coverage

**Test count**: 112 tests green (including integration tests)

**Categories**:

| Category | Tests | Examples |
|----------|-------|----------|
| Allocator | 12 | `pinned_buffer_alloc_registers_with_cuda`, `hugepage_alloc_works`, `mmap_fallback_returns_pageable` |
| Pool / RingBuffer | 18 | `new_ring_has_all_slots_available`, `try_claim_reduces_available_count`, `resize_rejected_when_slots_in_flight` |
| PoolRegistry | 8 | `registry_claim_routes_to_correct_ring`, `lz4_cache_eviction_works` |
| TensorLocationDict | 5 | `load_shard_index_json`, `parse_grouped_layer_format`, `duplicate_tensors_rejected` |
| MemoryPressureGauge | 6 | `callbacks_fire_at_thresholds`, `soft_pressure_independent_of_high` |
| PerLayerScaleTable | 8 | `overflow_halves_scale`, `gradient_variance_uses_windowed_mean` |
| Scheduler | 4 | `co_scheduler_registers_callbacks`, `prefetch_window_adjusts_on_pressure` |
| Phase Classifier | 3 | `phase_transitions_invoke_callback`, `forward_phase_updates_layers_in_flight` |
| CUDA Bridge | 10 | `zero_copy_routes_mapped_small_buffer`, `unmapped_small_buffer_routes_to_dma` |
| Kernels | 8 | `count_overflow_mock_counts_nan_and_inf`, `checkpoint_mock_wrappers_round_trip` |
| DirectNvmeEngine | 14 | `engine_open_initializes_ring`, `prefetch_pauses_on_pressure_signal`, `checksum_corruption_detection` |
| Write Budget | 6 | `smart_source_read_succeeds`, `strategy_switches_at_budget_thresholds` |
| Integration | 10 | `pool_registry_with_phase_classifier`, `direct_nvme_with_pressure_gauge_end_to_end` |

**Run tests**:
```bash
cargo test --no-default-features --features mock-cuda
cargo test --features cuda,hugepages,numa,nvme-passthrough
cargo test --features mock-cuda,lz4-cache,ssd-wear,checksums
```

---

## Summary Table

| Subsystem | File(s) | Key Type | Main Responsibility |
|-----------|---------|----------|---------------------|
| Allocator | `allocator/mod.rs`, `pinned.rs`, `huge.rs`, `mmap_tier.rs`, `numa.rs` | `PinnedBuffer`, `AllocKind` | Page-locked memory allocation with CUDA registration |
| Pool | `pool/mod.rs`, `subpools.rs`, `ring_buffer.rs`, `slab.rs` | `PoolRegistry`, `RingBuffer`, `PoolSlot`, `TensorSlab` | Ring-based buffer pooling; lazy slab packing |
| TensorIndex | `pool/tensor_location.rs` | `TensorLocationDict`, `TensorInfo` | Map tensor names to shard locations |
| Scheduler | `scheduler/mod.rs`, `coscheduler.rs`, `pressure_gauge.rs` | `CoScheduler`, `MemoryPressureGauge`, `PerLayerScaleTable` | Prefetch orchestration and loss scaling |
| Phase Manager | `phase/mod.rs`, `classifier.rs`, `profiler.rs`, `rebalancer.rs` | `PhaseClassifier`, `TrainingPhase`, `DefaultPhaseClassifier` | Training phase tracking and pool resizing |
| NVMe Engine | `nvme/mod.rs`, `prefetch.rs`, `passthrough.rs`, `write_budget.rs` | `DirectNvmeEngine`, `WriteBudgetManager` | Asynchronous I/O and wear management |
| CUDA Bridge | `cuda_bridge/mod.rs`, `zero_copy.rs`, `stream.rs`, `bindings.rs` | `ZeroCopyRouter`, `CudaStream`, `TransferStrategy` | CUDA runtime integration and zero-copy routing |
| Kernels | `kernels/mod.rs` | (kernel wrappers) | Overflow checking, INT8 compression |
| Emergency | `emergency.rs` | (emergency hooks) | SIGTERM/SIGINT checkpoint path |

---

**Last updated**: June 2025  
**Repository**: AethelStream (BTech Research Paper)  
**Maintainers**: Yash Manek
