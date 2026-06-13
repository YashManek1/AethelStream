# Graph Report - Code  (2026-06-13)

## Corpus Check
- 98 files · ~105,215 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1793 nodes · 3127 edges · 91 communities
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f54ce717`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]

## God Nodes (most connected - your core abstractions)
1. `PoolRegistry` - 46 edges
2. `DirectNvmeEngine` - 40 edges
3. `FlowCast` - 29 edges
4. `NvmePassthroughEngine` - 29 edges
5. `PrefetchEngine` - 28 edges
6. `WritebackScheduler` - 26 edges
7. `Telemetry` - 25 edges
8. `PinnedBuffer` - 23 edges
9. `WriteBudgetManager` - 21 edges
10. `PrefetchStateMachine` - 20 edges

## Surprising Connections (you probably didn't know these)
- `auto_select_returns_working_backend()` --calls--> `select_backend_with_override()`  [INFERRED]
  aethelStream/flowcast/tests/test_backend_selection.rs → aethelStream/flowcast/src/backend/mod.rs
- `override_mock_selects_mock_backend()` --calls--> `select_backend_with_override()`  [INFERRED]
  aethelStream/flowcast/tests/test_backend_selection.rs → aethelStream/flowcast/src/backend/mod.rs
- `override_super_shard_on_non_linux_errors_gracefully()` --calls--> `select_backend_with_override()`  [INFERRED]
  aethelStream/flowcast/tests/test_backend_selection.rs → aethelStream/flowcast/src/backend/mod.rs
- `unknown_override_returns_config_error()` --calls--> `select_backend_with_override()`  [INFERRED]
  aethelStream/flowcast/tests/test_backend_selection.rs → aethelStream/flowcast/src/backend/mod.rs
- `low_variance_layers_get_int4()` --calls--> `precision_for_variance()`  [INFERRED]
  aethelStream/flowcast/tests/test_priority.rs → aethelStream/flowcast/src/priority.rs

## Import Cycles
- 1-file cycle: `aethelStream/flowcast/benches/flowcast_bench.rs -> aethelStream/flowcast/benches/flowcast_bench.rs`
- 1-file cycle: `aethelStream/flowcast/src/backend/file_read.rs -> aethelStream/flowcast/src/backend/file_read.rs`
- 1-file cycle: `aethelStream/flowcast/src/backend/gds.rs -> aethelStream/flowcast/src/backend/gds.rs`
- 1-file cycle: `aethelStream/flowcast/src/backend/mock.rs -> aethelStream/flowcast/src/backend/mock.rs`
- 1-file cycle: `aethelStream/flowcast/src/backend/mod.rs -> aethelStream/flowcast/src/backend/mod.rs`
- 1-file cycle: `aethelStream/flowcast/src/backend/super_shard.rs -> aethelStream/flowcast/src/backend/super_shard.rs`
- 1-file cycle: `aethelStream/flowcast/src/backend/uring.rs -> aethelStream/flowcast/src/backend/uring.rs`
- 1-file cycle: `aethelStream/ramflow/src/nvme/mod.rs -> aethelStream/ramflow/src/nvme/mod.rs`
- 1-file cycle: `aethelStream/flowcast/src/completion_router.rs -> aethelStream/flowcast/src/completion_router.rs`
- 1-file cycle: `aethelStream/flowcast/src/config.rs -> aethelStream/flowcast/src/config.rs`
- 1-file cycle: `aethelStream/flowcast/src/decode.rs -> aethelStream/flowcast/src/decode.rs`
- 1-file cycle: `aethelStream/ramflow/src/bin/checksum_shard.rs -> aethelStream/ramflow/src/bin/checksum_shard.rs`
- 1-file cycle: `aethelStream/flowcast/src/hotset.rs -> aethelStream/flowcast/src/hotset.rs`
- 1-file cycle: `aethelStream/flowcast/src/lib.rs -> aethelStream/flowcast/src/lib.rs`
- 1-file cycle: `aethelStream/flowcast/src/state_machine.rs -> aethelStream/flowcast/src/state_machine.rs`
- 1-file cycle: `aethelStream/flowcast/tests/test_profiler_accuracy.rs -> aethelStream/flowcast/tests/test_profiler_accuracy.rs`
- 1-file cycle: `aethelStream/flowcast/src/peer.rs -> aethelStream/flowcast/src/peer.rs`
- 1-file cycle: `aethelStream/flowcast/src/priority.rs -> aethelStream/flowcast/src/priority.rs`
- 1-file cycle: `aethelStream/flowcast/src/profiler.rs -> aethelStream/flowcast/src/profiler.rs`
- 1-file cycle: `aethelStream/flowcast/src/ready.rs -> aethelStream/flowcast/src/ready.rs`

## Communities (91 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (38): Drop, Result, Self, Send, c_void, Result, CudaStream, Default (+30 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (35): Arc, AtomicBool, AtomicU32, HashMap, LayerKind, MemoryPressureGauge, Mutex, Option (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (35): AtomicU32, Default, Drop, HashMap, Instant, LayerKind, Mutex, Path (+27 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (39): Arc, AtomicBool, AtomicUsize, Default, FdTable, IoUringInstance, JoinHandle, Mutex (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (29): Arc, AtomicBool, AtomicUsize, CqeResult, Drop, FdTable, HashMap, IoUringInstance (+21 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (30): AdaptiveWindow, Arc, Box, Direction, Duration, HardwareProfile, IoBackend, MemoryPressureGauge (+22 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (37): c_void, Drop, Option, Result, Self, Send, async_overflow_token_matches_sync_check(), check_overflow_fp16() (+29 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (34): HardwareProfile, Item, Iterator, Option, Path, PathBuf, Result, Self (+26 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (45): 1. ABSTRACT, 2.1 Data Flow Diagram, 2.2 File Map, 2.3 Concurrency Model, 2.4 Frozen Public API (M3-API v1.0), 2. ARCHITECTURE, 3. ALGORITHMS A1-A9 + TELEMETRY, 4.1 Connection Matrix C1-C10 (+37 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (25): AtomicU64, Box, Fn, Mutex, Option, Path, PathBuf, PinnedBuffer (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.10
Nodes (26): Arc, AtomicI32, AtomicUsize, Condvar, Drop, ManuallyDrop, Mutex, Option (+18 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (26): Arc, AtomicU64, Completion, Condvar, Direction, Duration, Fn, HashMap (+18 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (16): Arc, AtomicBool, AtomicI32, MemoryPressureGauge, Mutex, Result, Self, Vec (+8 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (19): Arc, AtomicU32, Default, HashMap, IoBackend, Option, PathBuf, PinnedBuffer (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (33): Criterion, CudaStream, Result, T, bench_alignment_validation(), bench_allocator(), bench_delta_compression(), bench_ewa_scale_table() (+25 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (12): Arc, AtomicU32, AtomicU64, Default, Instant, Mutex, Option, Result (+4 more)

### Community 16 - "Community 16"
Cohesion: 0.10
Nodes (17): Arc, AtomicBool, AtomicU32, Default, Drop, Fn, JoinHandle, Mutex (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (20): Arc, AtomicUsize, Default, Duration, Option, PhaseMemoryProfile, PoolRegistry, Result (+12 more)

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (18): Default, Item, Iterator, Option, PerLayerScaleTable, Precision, Result, Self (+10 more)

### Community 19 - "Community 19"
Cohesion: 0.13
Nodes (19): Drop, HashMap, Mutex, Option, Path, PathBuf, RawFd, Result (+11 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (16): AtomicU64, HashMap, Result, Self, Vec, VecDeque, CachePrecision, compress_empty_src_returns_error() (+8 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (15): BackendCapabilities, Box, Completion, Default, HashMap, IoBackend, Mutex, Option (+7 more)

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (17): Default, HashMap, Mutex, Option, PathBuf, Result, Self, Send (+9 more)

### Community 23 - "Community 23"
Cohesion: 0.16
Nodes (21): HashMap, Option, Path, PathBuf, Result, Self, String, Vec (+13 more)

### Community 24 - "Community 24"
Cohesion: 0.08
Nodes (11): Option, capability_probe_does_not_panic(), capability_probe_returns_valid_state(), engine_passthrough_available_matches_probe(), nvme_get_nsid(), PassthroughCapability, probe_passthrough_capability(), engine_passthrough_flag_matches_probe() (+3 more)

### Community 25 - "Community 25"
Cohesion: 0.09
Nodes (7): Option, measure_rss(), print_rss_row(), rss_overhead_under_one_page(), rss_pinned_leq_vec(), rss_proportional_growth(), vm_rss_kb()

### Community 26 - "Community 26"
Cohesion: 0.08
Nodes (25): 1. Sequential Layer Streaming Engine, 2. Double-Pass Backward Algorithm, 3. Predictive I/O Overlap, 4. GaLore Optimizer Compression, 5. MemAscend-Inspired Memory Virtualization, 6. Precision-Aware Streaming, AethelStream, Architecture Overview (+17 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (13): AtomicBool, BackendCapabilities, Completion, Default, HashMap, IoBackend, Mutex, Option (+5 more)

### Community 28 - "Community 28"
Cohesion: 0.20
Nodes (21): Arc, AtomicBool, Drop, F, JoinHandle, Mutex, Option, Result (+13 more)

### Community 29 - "Community 29"
Cohesion: 0.14
Nodes (11): BackendCapabilities, Completion, Default, IoBackend, PinnedBuffer, Result, Self, Vec (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.16
Nodes (10): Arc, AtomicBool, AtomicU32, IoBackend, MemoryPressureGauge, Option, Result, Self (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.13
Nodes (12): Arc, AtomicBool, BackendCapabilities, Completion, IoBackend, Mutex, Path, PinnedBuffer (+4 more)

### Community 32 - "Community 32"
Cohesion: 0.13
Nodes (5): Send, AnyBuffer, BufferAccess, MmapBuffer, PinnedBuffer

### Community 33 - "Community 33"
Cohesion: 0.15
Nodes (11): AtomicBool, BackendCapabilities, Completion, IoBackend, Mutex, PathBuf, PinnedBuffer, Result (+3 more)

### Community 34 - "Community 34"
Cohesion: 0.14
Nodes (15): Box, Option, Path, Result, Send, Sync, BackendCapabilities, Completion (+7 more)

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (9): Path, PathBuf, make_temp_dir(), remove_temp_dir(), run_integration_body(), test_integration_module2_full_streaming_cycle(), vmrss_kb(), warmup_profiler_sha256_nonzero_and_second_run_uses_cache() (+1 more)

### Community 36 - "Community 36"
Cohesion: 0.14
Nodes (13): Arc, AtomicBool, AtomicUsize, CqeResult, Drop, FdTable, JoinHandle, RawFd (+5 more)

### Community 37 - "Community 37"
Cohesion: 0.15
Nodes (12): HashMap, Option, PinnedBuffer, Result, Self, Send, String, Sync (+4 more)

### Community 38 - "Community 38"
Cohesion: 0.15
Nodes (9): Drop, Result, Self, Send, alloc_and_round_trip(), is_pinned_is_always_false(), mmap_alloc(), MmapBuffer (+1 more)

### Community 39 - "Community 39"
Cohesion: 0.18
Nodes (10): Default, F, IoUring, Mutex, Option, Result, Self, T (+2 more)

### Community 40 - "Community 40"
Cohesion: 0.19
Nodes (9): Arc, Drop, ManuallyDrop, Option, PinnedBuffer, RingBuffer, Self, LayerKind (+1 more)

### Community 41 - "Community 41"
Cohesion: 0.20
Nodes (12): Arc, AtomicBool, AtomicU64, Drop, IoBackend, JoinHandle, Option, PrefetchStateMachine (+4 more)

### Community 42 - "Community 42"
Cohesion: 0.24
Nodes (9): HashMap, Mutex, Precision, Result, Self, Vec, DecodeState, f32_to_f16() (+1 more)

### Community 43 - "Community 43"
Cohesion: 0.20
Nodes (6): Option, PerLayerScaleTable, Self, Vec, Entry, HotSet

### Community 44 - "Community 44"
Cohesion: 0.19
Nodes (8): Default, Result, Self, Send, Sync, NvlinkPeerSync, PeerSync, SingleGpuSync

### Community 45 - "Community 45"
Cohesion: 0.16
Nodes (11): Arc, Default, LayerKind, MemoryPressureGauge, Mutex, Option, PoolSlot, Result (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.17
Nodes (12): Box, Option, Path, PathBuf, Result, String, main(), parse_args() (+4 more)

### Community 47 - "Community 47"
Cohesion: 0.20
Nodes (11): build_nvm_read_cmd(), engine_open_empty_paths_succeeds(), engine_pause_roundtrip(), engine_prefetch_returns_pressure_pause_when_paused(), nvm_read_cmd_high_slba_split(), nvm_read_cmd_is_80_bytes(), nvm_read_cmd_rejects_non_page_aligned_buffer(), nvm_read_cmd_rejects_unaligned_offset() (+3 more)

### Community 48 - "Community 48"
Cohesion: 0.21
Nodes (10): Default, Option, PathBuf, Result, Self, Vec, FlowCastConfig, HardwareProfile (+2 more)

### Community 50 - "Community 50"
Cohesion: 0.32
Nodes (12): AdaptiveWindow, MemoryPressureGauge, fire_high(), fire_low(), high_pressure_caps_window_overriding_a2_growth(), increase_decrease_respect_bounds(), increase_lookahead_suppressed_by_pressure_cap(), low_pressure_lifts_cap_and_allows_growth() (+4 more)

### Community 51 - "Community 51"
Cohesion: 0.18
Nodes (5): Drop, Self, Send, free_aligned(), PinnedDropGuard

### Community 52 - "Community 52"
Cohesion: 0.17
Nodes (12): 3.1 Allocator: PinnedBuffer (`src/allocator/pinned.rs`), 3.2 Pool: PoolRegistry (`src/pool/subpools.rs`), 3.3.1 IoUringInstance (`io_uring_setup.rs`), 3.3.2 FdTable (`fd_table.rs`), 3.3.3 PrefetchEngine and DirectNvmeEngine (`prefetch.rs`, `nvme/mod.rs`), 3.3.4 WriteBudgetManager (`write_budget.rs`, `ssd-wear` feature), 3.3 NVMe I/O Engine (`src/nvme/`), 3.4 Phase Manager (`src/phase/`) (+4 more)

### Community 53 - "Community 53"
Cohesion: 0.27
Nodes (10): Arc, PathBuf, PoolRegistry, PrefetchStateMachine, TempDir, Vec, FileReadBackend, create_temp_shards() (+2 more)

### Community 54 - "Community 54"
Cohesion: 0.35
Nodes (7): Default, Option, Self, detect(), NumaConfig, read_numa_node_for(), scan_gpu_numa_node()

### Community 55 - "Community 55"
Cohesion: 0.33
Nodes (6): Path, Self, c_int, open_char_device(), passthrough_latency_not_worse_than_odirect(), passthrough_read_matches_odirect()

### Community 56 - "Community 56"
Cohesion: 0.18
Nodes (11): 5.6 Benchmarks — Measured Numbers, Benchmark 1: Allocator — PinnedBuffer vs `Vec<u8>`, Benchmark 2: Pool Ring — Claim Latency, Benchmark 3: Pressure Gauge — Sampling Overhead, Benchmark 4: EWA Loss-Scale Table Update, Benchmark 5: INT8 Checkpoint Compression (CPU mock; GPU will be faster), Benchmark 6: O_DIRECT Alignment Validation, Benchmark 7: Analytical — SQPOLL Syscall Reduction (+3 more)

### Community 58 - "Community 58"
Cohesion: 0.38
Nodes (9): Criterion, bench_completion_routing(), bench_decode(), bench_pressure_response(), bench_priority(), bench_seam_end_to_end(), bench_state_machine(), bench_window() (+1 more)

### Community 59 - "Community 59"
Cohesion: 0.38
Nodes (9): PinnedBuffer, accumulated_delta_increments_per_step(), epoch_end_written_bytes_match_source(), flush_epoch_end_writes_accumulated_layers(), make_buf(), make_buf_filled(), max_skip_rate_guard_forces_write_when_exhausted(), skip_rate_never_exceeds_cap() (+1 more)

### Community 60 - "Community 60"
Cohesion: 0.58
Nodes (3): PinnedBuffer, PrefetchToken, Result

### Community 61 - "Community 61"
Cohesion: 0.25
Nodes (5): PoolSlot, Precision, Vec, DevicePointer, ReadyLayer

### Community 62 - "Community 62"
Cohesion: 0.32
Nodes (4): Arc, PoolRegistry, integration_slow_path_uses_lz4_cache_instead_of_blocking(), single_slot_registry()

### Community 64 - "Community 64"
Cohesion: 0.48
Nodes (5): PinnedBuffer, above_threshold_write_never_skipped(), inflight_write_cap_respected(), make_buf(), write_submitted_and_data_matches()

### Community 65 - "Community 65"
Cohesion: 0.52
Nodes (6): Option, PathBuf, find_cuda_lib_dir(), find_nvcc(), main(), nvcc_exe()

### Community 66 - "Community 66"
Cohesion: 0.29
Nodes (5): F, IoUring, Send, Sync, PassthroughRing

### Community 67 - "Community 67"
Cohesion: 0.29
Nodes (6): 7. Known Limitations and Future Work, 8. Build and Test Reference, 9. Public API Summary, Abstract, Full Technical Report — Module 2 Reference Document, RamFlow: System Memory Orchestration for AethelStream

### Community 68 - "Community 68"
Cohesion: 0.29
Nodes (7): 6.1 Abstract, 6.2 Contributions to Claim, 6.3 Related Work, 6.4 Evaluation Metrics to Report, 6.5 Algorithm Pseudocode for Paper, 6.6 Figures to Include, 6. What Belongs in the Research Paper

### Community 71 - "Community 71"
Cohesion: 0.40
Nodes (5): PathBuf, TempDir, Vec, create_temp_shards(), test_prefetch_byte_identical()

### Community 72 - "Community 72"
Cohesion: 0.33
Nodes (6): 4. The Five Novel Algorithms, Algorithm 1: Phase-Aware Predictive Pool Allocation, Algorithm 2: Tensor-Size-Aware Hybrid Zero-Copy Routing, Algorithm 3: FP16 Overflow Density Tracking (PerLayerScaleTable), Algorithm 4: Memory/I/O Co-Scheduler with Pressure Feedback, Algorithm 5: INT8 Activation Checkpoint Compression

### Community 73 - "Community 73"
Cohesion: 0.33
Nodes (6): 5.1 Codebase Size, 5.2 Test Suite, 5.3 Code Quality Gates, 5.4 Memory Alignment Properties, 5.5 Measured Algorithmic Properties, 5. Implementation Metrics

### Community 74 - "Community 74"
Cohesion: 0.50
Nodes (3): Result, mmap_huge(), round_to_huge()

### Community 75 - "Community 75"
Cohesion: 0.67
Nodes (3): Option, test_fragmentation_stability_80_layers_x_10_passes(), vm_rss_kb()

### Community 76 - "Community 76"
Cohesion: 0.50
Nodes (4): 1.1 Problem, 1.2 Why Memory Management Is the Hard Part, 1.3 Position in AethelStream, 1. Introduction

### Community 77 - "Community 77"
Cohesion: 0.50
Nodes (4): 2.1 Module Decomposition, 2.2 Data Flow, 2.3 Concurrency Model, 2. System Architecture

### Community 78 - "Community 78"
Cohesion: 0.67
Nodes (3): 3.5.1 MemoryPressureGauge, 3.5.2 CoScheduler, 3.5 Scheduler (`src/scheduler/`)

## Knowledge Gaps
- **319 isolated node(s):** `AtomicBool`, `Self`, `IoBackend`, `BackendCapabilities`, `GdsInner` (+314 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RamFlowError` connect `Community 12` to `Community 17`, `Community 4`, `Community 6`, `Community 39`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `DirectNvmeEngine` connect `Community 1` to `Community 4`, `Community 31`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `spawn_cqe_poller()` connect `Community 3` to `Community 4`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **What connects `AtomicBool`, `Self`, `IoBackend` to the rest of the system?**
  _319 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07075873827791987 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.0689484126984127 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.06666666666666667 - nodes in this community are weakly interconnected._