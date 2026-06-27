# AethelStream Module 3 (FlowCast) — Comprehensive Reference

**Version**: 0.1.0  
**Language**: Rust 2021 edition  
**Total LOC**: ~7600 (src + backends + scheduler)  
**Tests**: 17 test files, 3224 LOC  
**Codebase**: `aethelStream/flowcast/`

---

## Table of Contents

1. [Module Purpose](#module-purpose)
2. [Architecture Overview](#architecture-overview)
3. [Public API](#public-api)
4. [SARP Algorithm (A1)](#sarp-algorithm-a1)
5. [VRAM Double-Buffering (A11)](#vram-double-buffering-a11)
6. [IoBackend Trait & Implementations](#iobackend-trait--implementations)
7. [Super-Shard Adaptive Coalescing (A5-e)](#super-shard-adaptive-coalescing-a5-e)
8. [Scheduler: EDF & DuplexBudget (A5)](#scheduler-edf--duplexbudget-a5)
9. [Hot-Set & Ready Queue (A6)](#hot-set--ready-queue-a6)
10. [SSD Thermal Throttling (A3-T)](#ssd-thermal-throttling-a3-t)
11. [CQE Retry Mechanism](#cqe-retry-mechanism)
12. [Telemetry](#telemetry)
13. [Feature Flags](#feature-flags)
14. [Integration Points](#integration-points)
15. [Benchmarks](#benchmarks)
16. [Test Coverage](#test-coverage)

---

## Module Purpose

**FlowCast** is a prefetch engine and I/O pipeline that overlaps NVMe→RAM reads with GPU computation during LLM training. It decides *what* to prefetch, *when*, at *which precision*, and whether to skip write-back, delegating all I/O mechanism to RamFlow (Module 2).

**Key responsibilities**:
- **Bidirectional prefetch**: Forward pass (layers 0→L) and backward pass (layers L→0) with direction-aware ready maps
- **Adaptive window sizing**: EWMA-smoothed lookahead depth (Algorithm A2) capped by memory pressure
- **Selective precision**: Per-layer FP16, INT8, INT4 modes determined by importance scores
- **Write-back skipping**: Gradient-threshold gating (A9) to reduce SSD wear
- **VRAM double-buffering**: Overlaps RAM→VRAM DMA with GPU compute via dedicated copy stream (when enabled)
- **Smart scheduling**: EDF (Earliest-Deadline-First) ordering, DuplexBudget token bucket, thermal re-profiling

**Output to Module 5**: `ReadyLayer` objects containing pinned-RAM buffers, precision tags, and optional CUDA copy events.

---

## Architecture Overview

### High-Level Pipeline

```
FlowCast Facade
    ├── PrefetchStateMachine (A1)
    │   ├── Phase: Idle / Forward / Backward / Recompute
    │   ├── Direction-keyed ready maps (ready_forward, ready_backward)
    │   ├── In-flight token tracking
    │   └── Deferred reads (DuplexBudget backpressure)
    ├── CompletionRouter (dedicated thread)
    │   └── Polls backend.poll_completions() every ~50 µs
    ├── IoBackend trait implementations
    │   ├── FileReadBackend (fallback, sync reads)
    │   ├── UringBackend (io_uring on Linux)
    │   ├── DirectStorageBackend (Windows IDStorageQueue)
    │   ├── GdsBackend (NVIDIA GPU Direct Storage)
    │   └── SuperShardBackend (adaptive grouping wrapper)
    ├── AdaptiveWindow (A2)
    │   └── EWMA-smoothed lookahead, pressure-capped
    ├── WritebackScheduler (A4/A9)
    │   ├── Deferred queue (DuplexBudget backpressure)
    │   └── Gradient-threshold skip detection
    ├── EdfScheduler (A5-EDF)
    │   └── Deadline-ordered SQE submission
    ├── DuplexBudget (A5-DB)
    │   └── Token bucket: 60% reads / 40% writes
    ├── HotSet (A6)
    │   └── Resident layer cache (LFU promotion)
    ├── Profiler (A3)
    │   └── Warm-up: measure t_ssd, t_pcie, t_gpu
    ├── SmartMonitor (A3-T, Linux + ssd-thermal)
    │   └── NVMe SMART periodic re-profiling
    └── VramDoubleBuffer (A11, cuda-double-buffer)
        └── Two alternating VRAM slots + copy stream
```

### Data Flow: on_layer_start → take_ready

```
1. M5 calls on_layer_start(layer_idx, Direction::Forward)
   ├─ Phase classifier notified → RamFlow rebalancer aware
   ├─ DuplexBudget.refill() → read/write tokens replenished
   ├─ Drain deferred reads (token-starved SQEs from prior step)
   └─ Submit prefetch window
      ├─ Compute targets via prefetch_targets()
      ├─ Filter by hotset residency
      └─ For each target:
         ├─ Claim pool slot (blocks if exhausted)
         ├─ Check DuplexBudget.take_read() → defer if tokens exhausted
         └─ Submit backend.prefetch()

2. CompletionRouter thread polls backend every ~50 µs
   ├─ backend.poll_completions() → Vec<Completion>
   └─ route_completions() into direction-appropriate ready map
      └─ Signal condvar to wake take_ready()

3. M5 calls take_ready(layer_idx, timeout)
   ├─ Wait on condvar (with timeout) for layer to arrive in ready map
   ├─ Remove from ready map
   ├─ Determine precision + decode requirement from shard_index
   ├─ If cuda-double-buffer enabled:
   │  ├─ Copy pinned RAM → VRAM slot via dedicated stream
   │  ├─ Record CUDA event
   │  └─ Return device pointer in slab_device_ptrs
   └─ Return ReadyLayer (PoolSlot auto-freed on drop)

4. M5 calls on_weights_updated(layer_idx, &pinned_src)
   ├─ Check gradient norm via scale_table
   ├─ Apply gradient-threshold skip (A9)
   └─ Enqueue write via WritebackScheduler
      ├─ Check DuplexBudget.take_write() → defer if exhausted
      └─ Submit backend.write_async() when tokens available
```

---

## Public API

### `FlowCast` Facade

```rust
impl FlowCast {
    /// Initialize pipeline: start backend, prime window, spawn completion router.
    pub fn new(config: FlowCastConfig, backend: Box<dyn IoBackend>) 
        -> Result<Self>

    /// Profile num_layers representative layers, measure bandwidth, select W_max.
    /// Installs EDF scheduler and DuplexBudget from measured timings.
    pub fn warmup(&mut self, num_layers: u32) -> Result<HardwareProfile>

    /// Notify GPU started executing layer_idx in direction; submit next window.
    /// Honors co-scheduler pause signal (memory pressure).
    pub fn on_layer_start(&self, layer_idx: u32, direction: Direction) 
        -> Result<()>

    /// Block until layer_idx resident in pinned RAM, return ReadyLayer.
    /// Optionally issues async RAM→VRAM copy (cuda-double-buffer).
    pub fn take_ready(&self, layer_idx: u32, timeout: Duration) 
        -> Result<ReadyLayer>

    /// Notify optimizer updated layer_idx weights; enqueue async write.
    /// Applies gradient-threshold skip (A9), checks DuplexBudget tokens.
    pub fn on_weights_updated(&mut self, layer_idx: u32, src: &PinnedBuffer) 
        -> Result<()>

    /// Return current telemetry snapshot (counters, rates, thermal state).
    pub fn telemetry(&self) -> TelemetrySnapshot

    /// Graceful shutdown: stop router thread, shut down backend.
    pub fn shutdown(&mut self) -> Result<()>

    // Legacy API (compat with M5):
    pub fn advance_step(&mut self, direction: Direction, current_layer: u32) -> Result<()>
    pub fn wait_for_layer(&mut self, layer_idx: u32) -> Result<ReadyLayer>
    pub fn retire_layer(&mut self, _layer: ReadyLayer) -> Result<()>
}
```

### `FlowCastConfig`

```rust
pub struct FlowCastConfig {
    pub shard_dir: PathBuf,
    pub num_shards: u32,
    pub initial_lookahead: u32,
    pub ewma_alpha: f32,
    pub pressure_threshold: f32,
    pub default_precision: Precision,
    pub hardware_profile: Option<HardwareProfile>,
    pub io_poller_cpu_core: usize,
    pub completion_router_cpu_core: usize,
    pub target_gpu_utilisation: f32,
    pub reprofiling_interval_steps: u64,  // Default: 1000
    pub max_cqe_retries: u8,              // Default: 3
    pub base_backoff_ms: u64,             // Default: 10
}
```

### `ReadyLayer`

```rust
pub struct ReadyLayer {
    pub layer_idx: u32,
    pub precision: Precision,
    pub weight: PoolSlot,              // Freed on drop
    pub slab_device_ptrs: Vec<(u32, DevicePointer)>,  // VRAM slots
    pub needs_decode: bool,            // INT4/INT8 require decode
    #[cfg(feature = "cuda-double-buffer")]
    pub copy_event: Option<CudaEvent>, // DMA completion event
}

impl ReadyLayer {
    pub fn as_slice(&self) -> &[u8]
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
}
```

### `Precision` Enum

```rust
pub enum Precision {
    FP32,   // Full precision (warmup profiling only)
    FP16,   // Half precision (edge layers, activations)
    BF16,   // Brain float (Ampere/Ada preferred)
    INT8,   // 8-bit integer (mid-network)
    INT4,   // 4-bit integer (highest compression)
}
```

---

## SARP Algorithm (A1)

**SARP = Selective Activation Re-use Policy**: bidirectional prefetch state machine tracking which layers are in-flight, ready, and which pass (forward/backward/recompute) is active.

### State Machine States

```
Phase enum:
  ├─ Idle: not started
  ├─ Forward: executing layers 0→L (prefetch layers i+1..i+W)
  ├─ Backward: executing layers L→0 (prefetch layers i-W..i-1)
  └─ Recompute { window_start, window_end }:
     └─ Mini-forward recomputation within backward
        (isolated prefetch window for checkpoint reload)
```

### Direction-Aware State Tracking

```
MachineInner {
    phase: Phase,
    direction: Direction,
    in_flight: HashMap<token: u64 → InFlightEntry>,
    ready_forward: HashMap<layer_idx: u32 → PoolSlot>,
    ready_backward: HashMap<layer_idx: u32 → PoolSlot>,
}
```

**Why two ready maps?** (API-j fix)  
When the same layer_idx appears in both forward and backward windows (e.g., layer 0 near a phase boundary), a single combined map would overwrite the forward entry when the backward completion arrives. Two maps prevent silent loss.

### State Transitions

```
Idle
  ├─→ Forward: prime_window(Forward) submits layers 0..W-1
  └─→ Backward: prime_window(Backward) submits layers L-W..L-1

Forward
  ├─→ Backward: on_layer_start(L-1, Backward) switches direction
  └─→ Recompute { start, end }: on_layer_start() called within backward

Backward
  └─→ Forward: on_layer_start(0, Forward) wraps to next epoch

Recompute
  └─→ Backward: when recomputation window closes
```

### Token Lifecycle

```
1. submit_prefetch_for()
   ├─ Allocate token: next_token.fetch_add(1, Relaxed)
   └─ Insert into in_flight[token] = { layer_idx, direction, PoolSlot }

2. backend.prefetch() issued with token

3. CompletionRouter polls backend
   ├─ Receives Completion { token, result }
   ├─ route_completions() removes from in_flight[token]
   └─ If result >= 0:
      └─ Insert into ready_forward or ready_backward (direction-keyed)

4. take_ready() waits on condvar
   └─ Removes from ready_{forward|backward} and returns to caller
```

### Age-Based Tiebreak (SARP ordering)

When multiple layers are in-flight, EDF scheduler (A5-EDF) orders them by ascending deadline computed from `layer_plan` timings:

```
deadline[i] = current_time + t_ssd[i] + t_gpu[i]
```

On first run (no profiler data), falls back to **scalar window order**: prefetch targets in sequence.

### Deferred Reads (DuplexBudget backpressure)

```
submit_prefetch_for(target, direction, ...)
  ├─ Read shard_index for compressed byte length
  ├─ Check DuplexBudget.take_read(byte_len)
  │  ├─ OK: insert into in_flight, call backend.prefetch()
  │  └─ EXHAUSTED: push to deferred_reads queue, return Ok(())
  └─ Caller continues to next target

on_layer_start_with_residency()
  ├─ DuplexBudget.refill() for elapsed time
  └─ drain_deferred_reads()
     ├─ Peek front of queue
     ├─ Check token availability
     └─ Dequeue and resubmit via submit_prefetch_for_no_budget()
```

---

## VRAM Double-Buffering (A11)

**Purpose**: Overlap RAM→VRAM DMA with GPU compute using two alternating device-memory slots and a dedicated CUDA copy stream.

### Design

```
Module 3 (FlowCast)                Module 5 (Compute Loop)
────────────────────────────────────────────────────────

take_ready(layer_idx)              [GPU executes compute kernel
  ├─ Wait for pinned-RAM ready     on VRAM_SLOT_A with event
  ├─ Copy pinned → VRAM_SLOT_A     signalled from prior layer]
  │  via copy_stream
  ├─ Record event_A                cuda_stream_wait_event(
  └─ Return event_A                   compute_stream, event_A)
                                   [Ensures DMA complete]

take_ready(layer_idx+1)            [While compute runs,
  ├─ Wait for pinned-RAM ready     copy_stream DMA's
  ├─ Copy pinned → VRAM_SLOT_B     next layer to SLOT_B]
  │  via copy_stream (independent)
  ├─ Record event_B
  └─ Return event_B
```

### Architecture

```
CudaStream
  ├─ Real CUDA: cudaStream_t handle (independent work queue)
  └─ Mock CUDA: zero-sized type (no-op)

CudaEvent
  ├─ Real CUDA: cudaEvent_t, .record() timestamps completion
  ├─ .is_ready() polls event status
  └─ .synchronize() waits for completion

VramSlot
  ├─ Real CUDA: device memory (allocated externally)
  └─ Mock: heap-allocated Vec<u8>

VramDoubleBuffer { slot_a, slot_b, current_slot, copy_stream }
  └─ advance(layer_idx, src_bytes)
     ├─ Select next_slot = !current_slot
     ├─ Async memcpy: src → VRAM next_slot via copy_stream
     ├─ Record event on copy_stream
     ├─ Toggle current_slot
     └─ Return (device_ptr, event)
```

### Invariants (enforced by M5)

1. **Two slots**: M5 must pre-allocate VRAM_SLOT_A and VRAM_SLOT_B.
2. **Event-wait before compute**: M5 must call `cuda_stream_wait_event(compute_stream, event)` before dispatching the GPU kernel.
3. **Single-slot read window**: M5 must not reference call-N's slot after requesting call-N+2 (ping-pong reuses buffers).

### Feature Gate

```toml
[features]
cuda-double-buffer = []
```

When disabled (default):
- `take_ready()` returns `copy_event = None`
- M5 is responsible for H→D copy
- No VRAM device pointers populated in `slab_device_ptrs`

---

## IoBackend Trait & Implementations

### `IoBackend` Trait

```rust
pub trait IoBackend: Send + Sync {
    fn start(&mut self) -> Result<()>;
    
    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()>;
    
    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()>;
    
    fn poll_completions(&self) -> Result<Vec<Completion>>;
    
    fn is_paused(&self) -> bool;
    fn set_pause(&self, paused: bool);
    
    fn capabilities(&self) -> BackendCapabilities;
    fn shutdown(&mut self) -> Result<()>;
}

pub struct Completion {
    pub token: u64,
    pub result: i32,  // errno on failure, >= 0 on success
}

pub struct BackendCapabilities {
    pub supports_gds: bool,
    pub supports_super_shard: bool,
    pub supports_write_skip: bool,
    pub supports_multi_gpu: bool,
    pub name: &'static str,
}
```

### Backend Selection

```rust
pub fn select_backend() -> Result<Box<dyn IoBackend>>
  // Probe system capabilities:
  // 1. Windows + feature=direct-storage → DirectStorageBackend
  // 2. Linux + NVME CAPABLE → UringBackend
  // 3. Feature=gds (NVIDIA GDS) → GdsBackend
  // 4. Fallback → FileReadBackend

pub fn select_backend_with_override(name: &str) -> Result<Box<dyn IoBackend>>
  // Force a specific backend by name string
```

### Backend Implementations

#### FileReadBackend

- **Use case**: Linux/Windows fallback; mock-cuda CI.
- **Mechanism**: Synchronous `std::fs::File::read_at()` calls.
- **Completion**: Results pushed to internal queue inside `prefetch()`; no async completion; router drains queue every ~50 µs.
- **Advantage**: Works everywhere (no kernel driver required).
- **Disadvantage**: Blocks prefetch thread until read completes (latency jitter).

```rust
pub struct FileReadBackend {
    shard_path: PathBuf,
    completions: Mutex<Vec<Completion>>,
}

impl FileReadBackend::new(shard_path: PathBuf) -> Self
```

#### UringBackend

- **Use case**: Linux production path; io_uring SQE submission and CQE polling.
- **Mechanism**: 
  - `prefetch()` → submit SQE to io_uring ring
  - Router polls CQE ring every ~50 µs via `poll_completions()`
  - Returns Completion { token, result }
- **Capabilities**: Supports multi-GPU, super-shard coalescing.
- **Pause/resume**: SQE submission gated by `is_paused()` flag.

```rust
pub struct UringBackend {
    ring: io_uring::IoUring,
    io_poller_thread: Option<JoinHandle<()>>,
}
```

#### DirectStorageBackend

- **Use case**: Windows; Windows DirectStorage COM API (IDStorageQueue).
- **Mechanism**:
  - Submit SQE to Windows DirectStorage queue
  - Wait on Win32 Event object signalled by DirectStorage completion thread
  - Poll for ready CQEs
- **Capabilities**: Zero-copy SSD reads on NVMe (Windows only).
- **Feature gate**: `feature = "direct-storage"`; always compiles cleanly on Linux (stub path only).

```rust
pub struct DirectStorageBackend {
    queue: *const IDStorageQueue,
    completion_event: HANDLE,
}
```

#### GdsBackend

- **Use case**: NVIDIA GPU Direct Storage (NVIDIA DGX systems).
- **Mechanism**: Routes `prefetch` calls through NVIDIA GDS kernel driver.
- **Capabilities**: Supports GDS, multi-GPU.
- **Feature gate**: `feature = "gds"`

```rust
pub struct GdsBackend {
    inner: Box<dyn IoBackend>,  // Wrapped backend (UringBackend typically)
}
```

#### SuperShardBackend

- **Wrapper around any base backend**.
- **Purpose**: Group `group_size` contiguous layers into one large SQE (~100–200 MB) to reduce ring contention and optimize for I/O knee point.
- **See section 7 below**.

#### CqeRetryBackend

- **Wraps any backend**.
- **Purpose**: Classify CQE errors (transient vs. fatal); auto-retry transient errors with exponential backoff.
- **See "CQE Retry Mechanism" below**.

---

## Super-Shard Adaptive Coalescing (A5-e)

**Purpose**: Group contiguous layers into one large SQE to optimize I/O throughput, especially on high-bandwidth NVMe (PCIe 5.0+).

### Adaptive Group Size Computation

```rust
pub fn compute_group_size(optimal_bytes: u64, layer_sizes: &[u64]) -> u32
  // Input: optimal_super_shard_bytes from profiler (byte budget)
  //        layer_sizes: shard bytes for each layer
  // Process:
  //   1. Sort layer_sizes
  //   2. Compute median shard size
  //   3. group_size = optimal_bytes / median (clamped to [1, num_layers])
  // Output: number of layers per super-shard SQE
  //
  // Examples:
  //   optimal_bytes=100MiB, median=32MiB → group_size = 3 (96 MiB transfer)
  //   optimal_bytes=0 → group_size = 4 (default, no measurement)
```

### Architecture

```
SuperShardBackend {
    base: Box<dyn IoBackend>,
    pending_prefetch: Mutex<VecDeque<PendingPrefetch>>,
    group_size: AtomicU32,
    optimal_super_shard_bytes: AtomicU64,
    layer_offsets: HashMap<u32, u64>,
}

PendingPrefetch {
    shard_id: u32,
    byte_offset: u64,
    length: u64,
    token: u64,
}
```

### Prefetch Flow

```
prefetch(shard_id, byte_offset, length, dst, token)
  ├─ Allocate contiguous VRAM slot for the super-shard
  ├─ Push { shard_id, byte_offset, length, token } to pending queue
  ├─ If pending.len() >= group_size:
  │  └─ flush_group()
  └─ Return Ok(())

flush_group()
  ├─ Dequeue group_size pending entries
  ├─ Compute concatenated byte_offset + byte_length
  ├─ Call base.prefetch(group_id, start_offset, total_len, superVram, group_token)
  └─ Map token → list of original layer tokens for routing back
```

### Completion Routing

```
base.poll_completions() → Completion { group_token, result }
  ├─ Look up per-layer tokens from group_token mapping
  └─ Emit Completion { token: layer_token, result } for each layer
```

### Adaptive Updates

```rust
pub fn update_group_size(&self, new_optimal_bytes: u64, layer_sizes: &[u64])
  // Called by:
  //   1. Profiler after warmup (A3-e measurement)
  //   2. SmartMonitor after thermal re-profiling (A3-T)
  // Updates:
  //   - optimal_super_shard_bytes
  //   - group_size (recomputed from new byte budget)
```

---

## Scheduler: EDF & DuplexBudget (A5)

### EDF Scheduler (A5-EDF)

**Earliest-Deadline-First**: reorder prefetch SQEs by deadline within the current window.

```rust
pub struct EdfScheduler {
    deadlines: HashMap<u32, u32>,  // layer_idx → deadline (milliseconds)
    available: bool,                // is_available() ⟺ deadlines populated
}

impl EdfScheduler {
    pub fn new(layer_plan: &[LayerTiming], pcie_bandwidth_gbs: f32, num_layers: u32) 
        -> Self
      // Compute deadline[i] = t_ssd[i] + t_pcie[i] + t_gpu[i]
      // using measured layer_plan from profiler

    pub fn is_available(&self) -> bool
      // Returns true if deadline map was successfully populated
      
    pub fn sort_by_deadline(&self, targets: &mut Vec<u32>)
      // In-place sort: ascending deadline order (earliest first)
}
```

**Integration**: Called from `prefetch_targets()` after computing window:

```
prefetch_targets(current_layer, direction)
  ├─ Compute sequential targets in direction
  ├─ If EDF available:
  │  └─ Sort targets by deadline (in place)
  └─ Return sorted targets
```

### DuplexBudget Token Bucket (A5-DB)

**Purpose**: Split NVMe bandwidth between prefetch reads (60%) and write-back (40%), preventing write bursts from starving reads.

```rust
pub struct DuplexBudget {
    read_bucket: Arc<TokenBucket>,
    write_bucket: Arc<TokenBucket>,
}

pub const DEFAULT_READ_FRACTION: f32 = 0.6;

impl DuplexBudget {
    pub fn new(nvme_bandwidth_gbs: f32, read_fraction: f32, _scale: f32) -> Self
      // Allocate read and write token buckets
      // Tokens = bytes; max capacity = nvme_bandwidth_gbs * 1e9 / refill_interval_hz

    pub fn refill(&self)
      // Called at start of on_layer_start_with_residency()
      // Adds tokens proportional to elapsed wall time

    pub fn take_read(&self, byte_len: u64) -> std::result::Result<(), BandwidthExhausted>
      // Consume read tokens; fails if bucket exhausted

    pub fn take_write(&self, byte_len: u64) -> std::result::Result<(), BandwidthExhausted>
      // Consume write tokens; fails if bucket exhausted
}
```

**Backpressure Flow**:

```
prefetch flow:
  submit_prefetch_for()
    ├─ Check DuplexBudget.take_read(read_len)
    ├─ If Ok: submit to backend
    └─ If Err: push to deferred_reads queue
    
  on_layer_start_with_residency()
    ├─ DuplexBudget.refill() for elapsed time
    └─ drain_deferred_reads() (retries deferred SQEs)

write flow:
  on_weights_updated()
    ├─ Check DuplexBudget.take_write(write_len)
    ├─ If Ok: submit to backend
    └─ If Err: push to deferred_writes queue
    
  on_layer_start() or manual drain_deferred()
    ├─ DuplexBudget already refilled
    └─ Retry deferred writes (via WritebackScheduler.drain_deferred())
```

---

## Hot-Set & Ready Queue (A6)

### HotSet: Resident Layer Cache

**Purpose**: Track layers permanently resident in pinned RAM (embeddings, LoRA adapters, static blocks) so I/O is skipped for them.

```rust
pub struct HotSet {
    capacity: usize,
    entries: Vec<Entry>,  // Current resident layers
    access_counts: Vec<u64>,  // Per-layer access frequency
}

#[derive(Debug, Clone)]
struct Entry {
    layer_idx: u32,
    access_count: u64,  // For LFU promotion
}

impl HotSet {
    pub fn new(capacity: usize) -> Self

    pub fn seed_static(
        &mut self,
        num_layers: u32,
        k: u32,  // First/last k blocks to pin
        lora_layer_indices: &[u32],  // LoRA adapter indices
        scale_table: &mut PerLayerScaleTable,
    )
      // Pin layers:
      //   - Layer 0 (embedding / first block)
      //   - Layer num_layers-1 (LM head / last block)
      //   - First k and last k transformer blocks
      //   - All LoRA adapters
      // Mark as resident in scale_table

    pub fn is_resident(&self, layer_idx: u32, scale_table: &PerLayerScaleTable) -> bool
      // Returns true if layer in entries OR marked resident in scale_table

    pub fn record_access(&mut self, layer_idx: u32, scale_table: &PerLayerScaleTable)
      // Increment access_count[layer_idx]
      // Consider LFU promotion if resident capacity available

    pub fn evict_lru(&mut self, scale_table: &PerLayerScaleTable)
      // Remove least-frequently-used non-static entry
}
```

**Usage in state_machine**:

```
on_layer_start_with_residency(layer_idx, direction, pool, backend, resident_fn)
  // resident_fn = |idx| hotset.is_resident(idx, scale_table)
  ├─ For each target:
  │  ├─ If resident_fn(target): skip I/O
  │  └─ Else: submit prefetch_for(target, ...)
  └─ Update access counts via resident_fn calls
```

### Ready Queue

**Not a separate data structure**; implemented via:
- `state_machine.ready_forward` / `ready_backward`: HashMap<u32, PoolSlot>
- `router` thread polls `backend.poll_completions()` and populates these maps
- `take_ready()` blocks on condvar until layer appears in the correct map

**Depth snapshot**: `telemetry.ready_queue_depth` (count of entries in both ready maps).

---

## SSD Thermal Throttling (A3-T)

**Purpose**: Periodically read NVMe SMART data, detect thermal state (Normal/Warm/Throttling), and trigger re-profiling when temperature approaches throttle point.

**Platform**: Linux only + `feature = "ssd-thermal"`.

### SmartMonitor Architecture

```rust
pub struct ThermalMonitor {
    device_path: PathBuf,
    shard_dir: PathBuf,
    num_layers: u32,
    ssd_temp_celsius: AtomicF32,
    thermal_state: AtomicU8,  // 0=Normal, 1=Warm, 2=Throttling
    reprofiling_events: AtomicU64,
    pending_outcome: Mutex<Option<ReprofileOutcome>>,
}

pub struct ReprofileOutcome {
    pub w_max: u32,  // New W_max from adaptive profiling
    pub optimal_super_shard_bytes: u64,
}

pub enum ThermalState {
    Normal = 0,
    Warm = 1,      // Approaching throttle; reduce W_max by 10%
    Throttling = 2,  // Active throttling; reduce W_max by 25%
}
```

### Lifecycle

```
FlowCast::new()
  └─ #[cfg(ssd-thermal)]
     └─ ThermalMonitor::new(device_path: /dev/nvme0, shard_dir, num_layers)

FlowCast::on_layer_start()
  ├─ steps_since_reprofile += 1
  └─ If steps % reprofiling_interval == 0:
     ├─ thermal_monitor.tick(step, reprofiling_interval)
     │  ├─ Read SMART via ioctl(fd, NVME_IOCTL_ADMIN_CMD, ...)
     │  ├─ Parse temperature from SMART data
     │  ├─ Classify thermal_state
     │  └─ If state changed to Warm/Throttling:
     │     └─ Spawn background profiler thread
     │        ├─ Measure new latency-vs-size curve
     │        ├─ Compute adjusted w_max (reduced by 10–25%)
     │        └─ Store in pending_outcome
     └─ poll_outcome() returns Some(ReprofileOutcome) if complete
        └─ AdaptiveWindow.apply_w_max_update(new_w_max)
        └─ SuperShardBackend.update_group_size(...)

Telemetry snapshot includes:
  ssd_temp_celsius: f32 (latest SMART reading)
  thermal_state: u8 (Normal | Warm | Throttling)
  reprofiling_events: u64 (count of triggered re-profiles)
```

### SMART Parsing

```
NVMe SMART page 0x02:
  Bytes [1-2]: Composite temperature (in Kelvin)
  Conversion: T_celsius = composite_temp_k - 273.15

Thresholds (configurable):
  Normal: T < 60°C
  Warm: 60°C ≤ T < 75°C
  Throttling: T ≥ 75°C
```

---

## CQE Retry Mechanism

**Purpose**: Classify I/O errors from backend CQEs as transient (retryable) vs. fatal (media errors), retry transient with exponential backoff.

### CqeRetryBackend Wrapper

```rust
pub struct CqeRetryBackend {
    inner: Arc<dyn IoBackend>,
    config: RetryConfig,
    telemetry: Telemetry,
    pending: Mutex<VecDeque<PendingRead>>,  // In-flight retries
}

pub struct RetryConfig {
    pub max_retries: u8,      // Default: 3
    pub base_backoff_ms: u64, // Default: 10
}

pub struct PendingRead {
    pub token: u64,
    pub shard_id: u32,
    pub byte_offset: u64,
    pub length: u64,
    pub dst: *mut u8,
    pub retry_count: u8,
    pub next_retry_at: Option<Instant>,
}
```

### Error Classification

```rust
pub fn classify_cqe_error(result: i32) -> CqeErrorKind
  // Returns:
  //   CqeErrorKind::Transient: e.g., EAGAIN, ENOMEM, EBUSY
  //   CqeErrorKind::MediaError: e.g., EIO, EBADMSG (bad CRC)
  //   CqeErrorKind::Unknown: other error codes
```

### Retry Flow

```
CqeRetryBackend::poll_completions()
  ├─ Call inner.poll_completions() → Vec<Completion>
  ├─ For each Completion { token, result }:
  │  ├─ If result < 0:
  │  │  ├─ Classify error via classify_cqe_error(result)
  │  │  ├─ If Transient && retry_count < max_retries:
  │  │  │  ├─ Compute backoff: 2^(retry_count) * base_backoff_ms
  │  │  │  ├─ Push to pending queue with next_retry_at timestamp
  │  │  │  └─ Increment retry_count
  │  │  │  └─ Record telemetry.retry_count++
  │  │  └─ Else (fatal or max retries):
  │  │     └─ Emit Completion { token, result } downstream
  │  │     └─ Record telemetry.media_error_count++
  │  └─ Else (result >= 0): emit success
  └─ drain_retry_queue(): check pending queue timers, resubmit ready entries
```

### Telemetry

- `retry_count`: cumulative transient errors that were auto-retried
- `media_error_count`: fatal errors (media errors, exhausted retries)

---

## Telemetry

**All counters are `AtomicU64` (Relaxed ordering); best-effort gauges; sampled by completion router thread.**

### TelemetrySnapshot

```rust
pub struct TelemetrySnapshot {
    pub prefetch_submitted: u64,     // Total SQEs issued
    pub prefetch_completed: u64,     // Total CQEs (success + failure)
    pub prefetch_errors: u64,        // CQEs with result < 0
    pub miss_count: u64,             // PrefetchMiss errors to M5
    pub gpu_idle_us: u64,            // Total GPU idle time (µs)
    pub hotset_hits: u64,            // Layers found resident (no I/O)
    pub hotset_misses: u64,          // Layers not resident (I/O issued)
    pub nvme_bytes_read: u64,        // Total bytes transferred
    pub queue_depth: u64,            // Current io_uring SQ depth
    pub ready_queue_depth: u64,      // Layers completed, not yet consumed
    pub window_grow_events: u64,     // Adaptive window size increases
    pub window_shrink_events: u64,   // Adaptive window size decreases
    pub decode_ns: u64,              // Total time in decode kernels (nanosecs)
    pub write_skip_count: u64,       // Layers skipped via gradient threshold
    pub write_submitted: u64,        // Write SQEs issued
    pub nvme_throughput_mbps: f32,   // Instantaneous throughput (MB/s)
    pub ssd_temp_celsius: f32,       // Latest SMART temperature
    pub thermal_state: u8,           // 0=Normal, 1=Warm, 2=Throttling
    pub reprofiling_events: u64,     // Thermal re-profiling triggers
    pub retry_count: u64,            // Transient CQE errors (auto-retried)
    pub media_error_count: u64,      // Fatal CQE errors
}

impl TelemetrySnapshot {
    pub fn prefetch_hit_rate(&self) -> f32
      // (prefetch_completed - prefetch_errors) / prefetch_submitted
      
    pub fn hotset_hit_rate(&self) -> f32
      // hotset_hits / (hotset_hits + hotset_misses)
      
    pub fn gpu_idle_fraction(&self, total_elapsed_us: u64) -> f32
      // gpu_idle_us / total_elapsed_us
      
    pub fn write_skip_rate(&self) -> f32
      // write_skip_count / (write_skip_count + write_submitted)
      
    pub fn to_json(&self) -> Result<String>
      // Serialize to JSON for paper reports
}
```

### Recording Methods

```rust
pub struct Telemetry;

impl Telemetry {
    pub fn record_prefetch_submitted(&self, byte_length: u64)
    pub fn record_prefetch_completed(&self, success: bool)
    pub fn record_miss(&self)
    pub fn record_gpu_idle_us(&self, micros: u64)
    pub fn record_hotset(&self, hit: bool)
    pub fn set_queue_depth(&self, depth: u64)
    pub fn set_ready_queue_depth(&self, depth: u64)
    pub fn record_window_grow(&self)
    pub fn record_window_shrink(&self)
    pub fn record_decode_ns(&self, nanos: u64)
    pub fn record_write_skip(&self)
    pub fn record_write_submitted(&self)
    pub fn record_thermal_state(&self, temp_celsius: f32, state_u8: u8)
    pub fn set_reprofiling_events(&self, count: u64)
    pub fn record_cqe_retry(&self)
    pub fn record_media_error(&self)
    pub fn record_nvme_throughput_mbps(&self, mbps: f32)
    pub fn snapshot(&self) -> TelemetrySnapshot
}
```

---

## Feature Flags

| Flag | Purpose | Platform | Default |
|------|---------|----------|---------|
| `cuda` | Real CUDA runtime (not mock) | All | Off |
| `mock-cuda` | Mock CUDA (CI, GPU-less) | All | Off |
| `direct-storage` | Windows IDStorageQueue backend | Windows | Off |
| `gds` | NVIDIA GPU Direct Storage | All | Off |
| `super-shard` | Adaptive super-shard coalescing | All | Off |
| `quantized-stream` | INT4/INT8 mixed-precision support | All | Off |
| `write-skip` | Gradient-threshold write skipping (A9) | All | Off |
| `multi-gpu` | Multi-GPU support (NVLink awareness) | All | Off |
| `ssd-wear` | WriteBudgetManager SSD wear tracking | All | Off |
| `ssd-thermal` | NVMe SMART thermal throttling | Linux | Off |
| `cuda-double-buffer` | VRAM ping-pong with copy stream | All | Off |

**Default features** (no flags): ramflow with no GPU, file-read backend.

**Recommended production** (7B–70B model):
```toml
[features]
default = ["cuda", "direct-storage", "super-shard", "quantized-stream", "ssd-wear", "ssd-thermal"]
```

---

## Integration Points

### Upstream: Module 2 (RamFlow)

**FlowCast uses**:
- `PoolRegistry`: claim / release pinned-RAM slots
- `PerLayerScaleTable`: gradient variance lookup for write-skip (A9)
- `MemoryPressureGauge`: memory pressure callbacks (high/soft/low)
- `CoScheduler`: pause/resume prefetch based on pool pressure
- `DefaultPhaseClassifier`: notify RamFlow which layer GPU is executing (for pool rebalancing)
- `PinnedBuffer`: source/destination for I/O operations

**FlowCast provides**:
- Layer shard reading (via IoBackend backends)
- Precision selection and decoding (A7, A8)
- Write-back scheduling (A4, A9)
- Thermal throttling (A3-T)

### Downstream: Module 5 (DoublePass / Training Loop)

**M5 calls**:
```rust
fc.warmup(num_layers)  // Before training
fc.on_layer_start(layer_idx, direction)  // At GPU step start
layer = fc.take_ready(layer_idx, timeout)  // Block until ready
fc.on_weights_updated(layer_idx, &src)  // After optimizer update
fc.shutdown()  // At training end
```

**M5 receives**:
- `ReadyLayer`: pinned-RAM buffer with precision tag, decode flag
- CUDA copy event (if `cuda-double-buffer` enabled)
- Telemetry snapshots for logging

---

## Benchmarks

**Location**: `benches/flowcast_bench.rs`

```rust
// Synthetic benchmarks measuring:
//   - Prefetch submission latency (SQE enqueue time)
//   - Completion routing latency (CQE poll → ready map update)
//   - State machine throughput (layers/sec in steady state)
//   - Backend selection overhead
//   - Adaptive window update overhead
//   - Scheduler sort overhead (EDF vs. scalar)
```

**Run**:
```bash
cargo bench --bench flowcast_bench --features "mock-cuda"
```

**Output**: `target/criterion/` (HTML reports).

---

## Test Coverage

**Total**: 17 test files, 3224 LOC, organized by feature/component.

### Integration Tests

- **`integration.rs`**: End-to-end pipeline (warmup → forward → backward → take_ready → write)

### Component Tests

| Test File | Purpose |
|-----------|---------|
| `test_backend_selection.rs` | Backend probe logic, fallback chain |
| `test_cqe_retry.rs` | CQE error classification, exponential backoff |
| `test_determinism.rs` | Reproducibility of window updates, EDF ordering |
| `test_direct_storage.rs` | DirectStorage COM queue (Windows-only) |
| `test_gpu_idle.rs` | GPU idle time measurement |
| `test_hotset.rs` | Static + LFU resident layer tracking |
| `test_prefetch_correctness.rs` | Layers arrive in-order, no duplicates |
| `test_priority.rs` | Priority queue operations |
| `test_profiler_accuracy.rs` | Warmup timings, W_max selection |
| `test_quantized_stream.rs` | INT4/INT8 precision mode handling |
| `test_super_shard_adaptive.rs` | Group size computation, knee detection |
| `test_thermal_monitor.rs` | SMART parsing, thermal state transitions |
| `test_vram_double_buffer.rs` | Ping-pong slot cycling, event recording |
| `test_window_sizing.rs` | Adaptive window EWMA, pressure capping |
| `test_write_skip.rs` | Gradient threshold gating |
| `test_writeback.rs` | Deferred write queue, DuplexBudget interaction |

### Test Execution

```bash
# All tests
cargo test --all-features

# Specific test
cargo test --lib test_hotset -- --nocapture

# With mock-cuda + ssd-thermal (CI path)
cargo test --features "mock-cuda,ssd-thermal"
```

**Coverage**: 80%+ of core paths (state machine, backends, scheduler, telemetry).

---

## Key Algorithms & Fixes

### SARP State Machine (A1)

- **API-j fix**: Direction-aware ready maps prevent silent layer overwrite at phase boundaries.
- **A1-c2 fix**: `on_layer_start_with_residency()` unifies two prior code paths.

### VRAM Double-Buffer (A11)

- **Copy-stream isolation**: Independent stream allows DMA to overlap with GPU compute.
- **Event-wait invariant**: M5 must call `cuda_stream_wait_event()` before compute kernel.

### Super-Shard (A5-e)

- **Adaptive knee detection**: Profiler runs latency-vs-size curve to find optimal transfer size.
- **Online group size updates**: Thermal re-profiling can adjust group size live.

### DuplexBudget (A5-DB)

- **Deferred read/write queues**: Token exhaustion defers SQEs to next refill window.
- **60/40 split**: Conservative reads (60%), aggressive writes (40%).

### Thermal Throttling (A3-T)

- **Periodic re-profiling**: Every N steps (default 1000) read NVMe SMART.
- **Adaptive W_max**: Temperature-based adjustment (10–25% reduction when throttling).

### Gradient-Threshold Write-Skip (A9)

- **Per-layer skipping**: Skip write if `|lr·grad| < threshold`.
- **Max-skip-rate guard**: Prevent excessive deferral; enforce flush after threshold fraction of layers.

---

## Summary Table

| Component | Lines | Purpose | Key Files |
|-----------|-------|---------|-----------|
| **State Machine (A1)** | 706 | Bidirectional prefetch, direction-aware maps | `state_machine.rs` |
| **Completion Router** | 200+ | Dedicated CQE polling thread | `completion_router.rs` |
| **IoBackend Trait** | 100+ | Abstraction: file, io_uring, DirectStorage, GDS | `backend/mod.rs` + implementations |
| **SuperShard (A5-e)** | 250+ | Adaptive coalescing, knee detection | `backend/super_shard.rs` |
| **EDF Scheduler (A5-EDF)** | 150+ | Deadline-ordered SQE submission | `scheduler/edf.rs` |
| **DuplexBudget (A5-DB)** | 200+ | Token bucket: 60% reads / 40% writes | `scheduler/token_bucket.rs` |
| **Adaptive Window (A2)** | 250+ | EWMA lookahead, pressure capping | `window.rs` |
| **HotSet (A6)** | 200+ | Resident layer cache, LFU promotion | `hotset.rs` |
| **Writeback (A4/A9)** | 350+ | Deferred write queue, gradient skip | `writeback.rs` |
| **Profiler (A3)** | 500+ | Warmup measurements, hardware profile | `profiler.rs` |
| **SmartMonitor (A3-T)** | 200+ | NVMe thermal throttling (Linux) | `smart_monitor.rs` |
| **VRAM Double-Buffer (A11)** | 200+ | Overlapped H→D DMA with compute | `vram_double_buffer.rs` |
| **Telemetry** | 150+ | Counter aggregation, JSON export | `telemetry.rs` |

---

## Running the Codebase

### Build

```bash
# Development (mock CUDA, file backend)
cargo build

# Production (real CUDA, io_uring, DirectStorage)
cargo build --release --features "cuda,direct-storage,super-shard,ssd-thermal"

# CI (mock CUDA, GDS probe, ssd-thermal)
cargo build --features "mock-cuda,gds,ssd-thermal"
```

### Tests

```bash
# Unit + integration
cargo test --lib

# All tests (long)
cargo test --all-features

# Specific test with output
cargo test test_hotset -- --nocapture --test-threads=1
```

### Benchmarks

```bash
cargo bench --bench flowcast_bench -- --verbose
```

---

## Error Handling

All FlowCast APIs return `Result<T>` where `Result` is an alias for `std::result::Result<T, FlowCastError>`.

```rust
pub enum FlowCastError {
    RamFlow(ramflow::RamFlowError),  // Upstream pool/scale table error
    PrefetchMiss { layer_idx: u32 }, // Timeout waiting for layer
    InvalidTransition(String),       // Illegal state transition
    BackendIo(String),              // I/O backend failure
    ProfileIo(String),              // Profile file I/O failure
    Config(String),                 // Invalid config field
}
```

**Module 5 must handle `PrefetchMiss`**: it signals that the prefetch did not complete in time and M5 should retry or adjust lookahead.

---

## Design Constraints & Invariants

1. **No mutations to global pool while a PoolSlot is alive**: PoolSlots returned to pool only on drop (RAII).
2. **Completion router is the sole drainer of backend CQEs** (production): exposing manual draining creates race conditions.
3. **Direction-keyed ready maps**: Forward and backward layers must be stored separately to prevent overwrites.
4. **State machine lock never held while calling backend methods**: prevents deadlock with CompletionRouter.
5. **VRAM double-buffer invariant (M5's responsibility)**: only one slot readable at a time; event-wait enforced.
6. **DuplexBudget tokens are best-effort**: transient failures may cause SQE deferral (not stalls).

---

**Last Updated**: 2026-06-27  
**Module Status**: Production-ready (245+ tests passing, 0 clippy warnings)

