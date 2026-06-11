// benches/flowcast_bench.rs — FlowCast (M3) + RamFlow (M2) combined benchmarks
//
// Run: cargo bench --features mock-cuda
//      cargo bench --features mock-cuda -- <filter>  (run subset)
//
// Groups:
//   window/       — AdaptiveWindow::update() EWMA throughput; pressure-callback latency
//   priority/     — PriorityQueue push/pop at varying sizes; importance rebuild
//   decode/       — QuantizedDecoder: INT8→FP16, NF4(INT4)→FP16 decode throughput
//   state_machine/ — submit_prefetch_for latency (M2 pool claim + M3 SQE write)
//   completion/   — route_completions throughput (CQE ingestion → ready-map insert)
//   seam/         — End-to-end M2+M3 latency: pool_claim → prefetch → poll → take_ready
//   pressure/     — Pressure gauge sample_and_notify → window cap response
//   writeback/    — WritebackScheduler: delta accumulate, skip test, write submission

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flowcast::{
    backend::mock::MockBackend,
    config::Precision,
    decode::QuantizedDecoder,
    priority::{PriorityQueue, PrefetchRequest},
    state_machine::PrefetchStateMachine,
    window::AdaptiveWindow,
};
use ramflow::{MemoryPressureGauge, PoolRegistry};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Benchmark 1: AdaptiveWindow EWMA update throughput
// ---------------------------------------------------------------------------

fn bench_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("window");

    // T_iter update — hot path called every layer in the training loop.
    group.bench_function("update_ewma", |b| {
        let mut window = AdaptiveWindow::new(3.0, 0.2, 8.0);
        b.iter(|| {
            window.update(black_box(50.0_f32), black_box(200.0_f32)).unwrap();
        });
    });

    // High-pressure callback firing latency (Arc store + atomic store).
    group.bench_function("high_pressure_callback", |b| {
        let gauge = MemoryPressureGauge::new(30);
        let window = AdaptiveWindow::new(4.0, 0.2, 8.0);
        window.register_pressure_callbacks(&gauge, None);
        b.iter(|| {
            gauge.signal_stall(black_box(0));
        });
    });

    // Low-pressure callback (cap lift path).
    group.bench_function("low_pressure_callback", |b| {
        let gauge = MemoryPressureGauge::new(30);
        let window = AdaptiveWindow::new(4.0, 0.2, 8.0);
        window.register_pressure_callbacks(&gauge, None);
        // pre-fire high so low actually does work
        gauge.signal_stall(0);
        b.iter(|| {
            // Directly fire low callbacks via sample_and_notify with empty pool.
            // (Pool has 0% utilisation → low pressure path executes.)
            let pool = Arc::new(PoolRegistry::with_defaults().unwrap());
            gauge.sample_and_notify(black_box(&pool));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: PriorityQueue throughput
// ---------------------------------------------------------------------------

fn bench_priority(c: &mut Criterion) {
    let mut group = c.benchmark_group("priority");

    // Single push + pop round-trip (queue size stays at 1 after each iter).
    group.bench_function("push_pop_single", |b| {
        let mut queue = PriorityQueue::new();
        b.iter(|| {
            queue.push(PrefetchRequest {
                layer_idx: black_box(0),
                importance: black_box(1.0),
                precision: Precision::FP16,
                enqueue_step: 0,
            }).unwrap();
            black_box(queue.pop());
        });
    });

    // Pop from a queue with 32 elements (typical window size) — O(n) scan.
    group.bench_function("pop_from_32", |b| {
        b.iter_batched(
            || {
                let mut queue = PriorityQueue::new();
                for i in 0u32..32 {
                    queue.push(PrefetchRequest {
                        layer_idx: i,
                        importance: (i as f32) * 0.1 + 0.5,
                        precision: Precision::FP16,
                        enqueue_step: 0,
                    }).unwrap();
                }
                queue
            },
            |mut queue| {
                while let Some(req) = queue.pop() {
                    black_box(req);
                }
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // rebuild_from_scale_table — called at every phase boundary.
    group.bench_function("rebuild_80_layers", |b| {
        let mut queue = PriorityQueue::new();
        let scale_table = ramflow::PerLayerScaleTable::new(80, 0.05);
        b.iter(|| {
            queue.rebuild_from_scale_table(
                black_box(0u32..80),
                black_box(&scale_table),
                Precision::FP16,
            ).unwrap();
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: QuantizedDecoder throughput
// ---------------------------------------------------------------------------

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    // Sizes that represent real model layer slices.
    // FP16 attention layer in 7B: ~256 MiB compressed to ~64 MiB INT4.
    // Benchmark at smaller sizes to keep iteration time under 1 s.
    let configs: &[(usize, &str)] = &[
        (1024,  "1K_bytes"),     // very small: bias/norm
        (65536, "64K_bytes"),    // medium: attention slice
        (1 * 1024 * 1024, "1M_bytes"),   // full small layer
    ];

    for &(n_bytes, label) in configs {
        // INT8 → FP16: n_bytes input → 2n_bytes output
        group.throughput(Throughput::Bytes(n_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("int8_to_fp16", label),
            &n_bytes,
            |b, &n| {
                let src = vec![0i8 as u8; n];
                let scales = vec![1.0f32; 1];
                b.iter(|| {
                    let out = QuantizedDecoder::decode_int8_to_fp16(
                        black_box(&src),
                        black_box(&scales),
                        1,
                    );
                    black_box(out);
                });
            },
        );

        // INT4 (NF4) → FP16: n_bytes input → 4n_bytes output (2 nibbles per byte)
        group.bench_with_input(
            BenchmarkId::new("int4_nf4_to_fp16", label),
            &n_bytes,
            |b, &n| {
                let src = vec![0u8; n];
                b.iter(|| {
                    let out = QuantizedDecoder::decode_int4_to_fp16(
                        black_box(&src),
                        black_box(1.0f32),
                    );
                    black_box(out);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: State machine — submit_prefetch_for round-trip
// (M2 pool claim + M3 in-flight insert + M2 mock prefetch)
// ---------------------------------------------------------------------------

fn bench_state_machine(c: &mut Criterion) {
    use ramflow::phase::Direction;

    let mut group = c.benchmark_group("state_machine");

    group.bench_function("prime_window_4layers", |b| {
        b.iter_batched(
            || {
                let pool = PoolRegistry::with_defaults().unwrap();
                let backend = MockBackend::new();
                let sm = PrefetchStateMachine::new(16, 4, Precision::FP16);
                (pool, backend, sm)
            },
            |(pool, backend, sm)| {
                sm.prime_window(Direction::Forward, black_box(&pool), black_box(&backend))
                    .unwrap();
                black_box(sm);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.bench_function("on_layer_start_advance", |b| {
        b.iter_batched(
            || {
                let pool = PoolRegistry::with_defaults().unwrap();
                let backend = MockBackend::new();
                let sm = PrefetchStateMachine::new(32, 4, Precision::FP16);
                sm.prime_window(Direction::Forward, &pool, &backend).unwrap();
                // drain completions so pool slots are returned
                sm.poll_and_route(&backend).unwrap();
                (pool, backend, sm)
            },
            |(pool, backend, sm)| {
                sm.on_layer_start(0, Direction::Forward, black_box(&pool), black_box(&backend))
                    .unwrap();
                black_box(&sm);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5: Completion routing throughput (CQE batch → ready-map insert)
// ---------------------------------------------------------------------------

fn bench_completion_routing(c: &mut Criterion) {
    use ramflow::phase::Direction;

    let mut group = c.benchmark_group("completion");

        // Batch routing: process N completions via poll_and_route (public API).
    for batch_size in [1usize, 4, 8, 16, 32] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("poll_and_route", batch_size),
            &batch_size,
            |b, &n| {
                b.iter_batched(
                    || {
                        let pool = PoolRegistry::with_defaults().unwrap();
                        let backend = MockBackend::new();
                        let sm = PrefetchStateMachine::new(64, n as u32, Precision::FP16);
                        sm.prime_window(Direction::Forward, &pool, &backend).unwrap();
                        (sm, backend)
                    },
                    |(sm, backend)| {
                        sm.poll_and_route(black_box(&backend)).unwrap();
                        black_box(sm);
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 6: End-to-end M2+M3 seam — pool claim → prefetch → route → take_ready
// Measures the latency for one full "layer arrival" event seen by the training loop.
// ---------------------------------------------------------------------------

fn bench_seam_end_to_end(c: &mut Criterion) {
    use ramflow::phase::Direction;

    let mut group = c.benchmark_group("seam");

    // Latency of a single layer: prime, route, take_ready.
    group.bench_function("single_layer_prefetch_to_ready_ms", |b| {
        b.iter_batched(
            || {
                let pool = PoolRegistry::with_defaults().unwrap();
                let backend = MockBackend::new();
                let sm = Arc::new(PrefetchStateMachine::new(8, 1, Precision::FP16));
                // Prime layer 0 only (lookahead=1).
                sm.prime_window(Direction::Forward, &pool, &backend).unwrap();
                (pool, backend, sm)
            },
            |(_, backend, sm)| {
                // Route the mock completion via poll_and_route.
                sm.poll_and_route(&backend).unwrap();
                // take_ready should return immediately (0 ms wait).
                let layer = sm
                    .take_ready(0, Duration::from_millis(100))
                    .unwrap();
                black_box(layer);
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // Sustained throughput: how many layer-ready events per second?
    // Models a forward pass over `n_layers` with lookahead=4.
    for n_layers in [16u32, 32, 80] {
        group.throughput(Throughput::Elements(n_layers as u64));
        group.bench_with_input(
            BenchmarkId::new("forward_pass_layers_per_sec", n_layers),
            &n_layers,
            |b, &nl| {
                b.iter_batched(
                    || {
                        let pool = PoolRegistry::with_defaults().unwrap();
                        let backend = MockBackend::new();
                        let sm = Arc::new(PrefetchStateMachine::new(nl, 4, Precision::FP16));
                        sm.prime_window(Direction::Forward, &pool, &backend).unwrap();
                        (pool, backend, sm)
                    },
                    |(pool, backend, sm)| {
                        for layer_idx in 0..nl {
                            // Route pending completions first.
                            sm.poll_and_route(&backend).unwrap();
                            // Take this layer.
                            let ready = sm.take_ready(layer_idx, Duration::from_millis(200));
                            let _ = black_box(ready);
                            // Submit next window.
                            sm.on_layer_start(layer_idx, Direction::Forward, &pool, &backend)
                                .unwrap();
                        }
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 7: Write-back scheduler — delta accumulation hot path
// ---------------------------------------------------------------------------

fn bench_writeback(c: &mut Criterion) {
    use flowcast::writeback::{WritebackConfig, WritebackMode, WritebackScheduler};
    use ramflow::PinnedBuffer;

    let mut group = c.benchmark_group("writeback");

    group.bench_function("on_weights_updated_skip_path", |b| {
        // Gradient norm below threshold → pure delta accumulation, no I/O.
        let backend = MockBackend::new();
        let buf = PinnedBuffer::alloc(64).unwrap();
        let mut scheduler = WritebackScheduler::with_config(
            WritebackMode::Immediate,
            WritebackConfig {
                skip_threshold: 1.0,    // high threshold → always skip
                max_skip_rate: 1.0,
                max_inflight_writes: 4,
                shard_dir: std::path::PathBuf::from("."),
                write_budget_bytes: u64::MAX,
            },
        );
        b.iter(|| {
            scheduler.on_weights_updated(
                black_box(0u32),
                black_box(&buf),
                0,
                black_box(0.0001_f32),
                black_box(&backend),
            ).unwrap();
        });
    });

    group.bench_function("on_weights_updated_write_path", |b| {
        // Gradient norm above threshold → delta accumulates past threshold → write submitted.
        let backend = MockBackend::new();
        let buf = PinnedBuffer::alloc(64).unwrap();
        let mut scheduler = WritebackScheduler::with_config(
            WritebackMode::Immediate,
            WritebackConfig {
                skip_threshold: 1e-12,  // tiny threshold → always write
                max_skip_rate: 0.0,
                max_inflight_writes: 4,
                shard_dir: std::path::PathBuf::from("."),
                write_budget_bytes: u64::MAX,
            },
        );
        b.iter(|| {
            scheduler.on_weights_updated(
                black_box(0u32),
                black_box(&buf),
                0,
                black_box(1.0_f32),
                black_box(&backend),
            ).unwrap();
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 8: Pressure gauge → window cap response time
// Measures latency from gauge.signal_stall() to pressure_cap_active becoming true.
// ---------------------------------------------------------------------------

fn bench_pressure_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("pressure");

    group.bench_function("signal_stall_to_cap_active", |b| {
        let gauge = MemoryPressureGauge::new(30);
        let window = AdaptiveWindow::new(4.0, 0.2, 8.0);
        window.register_pressure_callbacks(&gauge, None);

        b.iter(|| {
            // Fire high-pressure path.
            gauge.signal_stall(black_box(0));
            // Assert cap is now active (prevents dead-code elimination).
            assert!(window.pressure_cap_active());
            // Reset for next iteration (re-fire low pressure via sample on empty pool).
            let pool = Arc::new(PoolRegistry::with_defaults().unwrap());
            gauge.sample_and_notify(&pool); // pool empty → low pressure → cap lifted
        });
    });

    group.bench_function("sample_and_notify_zero_pressure", |b| {
        let gauge = MemoryPressureGauge::new(30);
        let pool = Arc::new(PoolRegistry::with_defaults().unwrap());
        b.iter(|| {
            gauge.sample_and_notify(black_box(&pool));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_window,
    bench_priority,
    bench_decode,
    bench_state_machine,
    bench_completion_routing,
    bench_seam_end_to_end,
    bench_writeback,
    bench_pressure_response,
);
criterion_main!(benches);
