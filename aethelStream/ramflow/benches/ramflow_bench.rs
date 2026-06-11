// benches/ramflow_bench.rs — RamFlow performance benchmarks
//
// Run: cargo bench --no-default-features --features mock-cuda
//
// Groups:
//   allocator/   — PinnedBuffer vs Vec<u8> allocation at 4 KB, 64 KB, 1 MB, 64 MB
//   pool/        — ring claim fast-path and slow-path (1 in-flight slot) latency
//   pressure/    — MemoryPressureGauge::sample_and_notify overhead
//   ewa/         — PerLayerScaleTable::update() throughput (80 layers)
//   int8/        — mock INT8 compress + decompress throughput at 512 and 16384 elements
//   delta/       — zstd delta compress + decompress on 64 KB synthetic FP16 weights
//   alignment/   — validate_direct_io_alignment() overhead (hot path guard)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ramflow::{
    allocator::PinnedBuffer,
    cuda_bridge::CudaStream,
    kernels::{compress_checkpoint_fp16_to_int8, decompress_checkpoint_int8_to_fp16},
    phase::{PhaseMemoryProfile, TrainingPhase},
    pool::{LayerKind, PoolRegistry, TensorLocationDict},
    scheduler::{CoScheduler, MemoryPressureGauge, PerLayerScaleTable},
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Benchmark 1: Allocator — PinnedBuffer vs Vec
// ---------------------------------------------------------------------------

fn bench_allocator(c: &mut Criterion) {
    let sizes: &[usize] = &[
        4 * 1024,          // 4 KB  — small bias/norm tensor
        64 * 1024,         // 64 KB — medium attention slice
        1 * 1024 * 1024,   // 1 MB  — norm pool slot
        64 * 1024 * 1024,  // 64 MB — full attention/MLP slot
    ];

    let mut group = c.benchmark_group("allocator");

    for &size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("PinnedBuffer_alloc", format!("{}_bytes", size)),
            &size,
            |b, &sz| {
                b.iter(|| {
                    let buf = PinnedBuffer::alloc(black_box(sz)).expect("alloc failed");
                    black_box(buf.len());
                    // buf dropped here — cudaHostUnregister + free
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Vec_u8_alloc", format!("{}_bytes", size)),
            &size,
            |b, &sz| {
                b.iter(|| {
                    let v = vec![0u8; black_box(sz)];
                    black_box(v.len());
                    // v dropped here
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: Pool — fast-path claim/return latency
// ---------------------------------------------------------------------------

fn bench_pool_claim(c: &mut Criterion) {
    let profile = PhaseMemoryProfile {
        phase: TrainingPhase::Recomputation {
            window_start: 0,
            window_end: 2,
        },
        expected_peak_bytes: 0,
        attention_slots_needed: 4,
        mlp_slots_needed: 4,
        norm_slots_needed: 4,
        optimizer_slots_needed: 2,
    };
    let dict = TensorLocationDict::empty();
    let threshold = 4 * 1024 * 1024; // 4 MB zero-copy threshold
    let registry = Arc::new(PoolRegistry::new(&profile, &dict, threshold).expect("registry"));

    let mut group = c.benchmark_group("pool");

    // Fast path: pool has free slots — lock-free AtomicUsize check + Mutex pop
    group.bench_function("claim_attention_fast_path", |b| {
        b.iter(|| {
            let slot = registry.claim(black_box(LayerKind::Attention)).expect("claim");
            black_box(&slot);
            drop(slot); // returns slot to ring
        });
    });

    group.bench_function("claim_mlp_fast_path", |b| {
        b.iter(|| {
            let slot = registry.claim(black_box(LayerKind::Mlp)).expect("claim");
            black_box(&slot);
            drop(slot);
        });
    });

    group.bench_function("claim_norm_fast_path", |b| {
        b.iter(|| {
            let slot = registry.claim(black_box(LayerKind::Norm)).expect("claim");
            black_box(&slot);
            drop(slot);
        });
    });

    // Slow path: hold 3 of 4 attention slots, claim the 4th (last-slot contention)
    group.bench_function("claim_attention_last_slot", |b| {
        let _s1 = registry.claim(LayerKind::Attention).expect("hold1");
        let _s2 = registry.claim(LayerKind::Attention).expect("hold2");
        let _s3 = registry.claim(LayerKind::Attention).expect("hold3");
        b.iter(|| {
            let slot = registry.claim(black_box(LayerKind::Attention)).expect("last claim");
            black_box(&slot);
            drop(slot);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3: Pressure gauge — sampling overhead
// ---------------------------------------------------------------------------

fn bench_pressure_gauge(c: &mut Criterion) {
    let profile = PhaseMemoryProfile {
        phase: TrainingPhase::Forward { layers_in_flight: 1 },
        expected_peak_bytes: 0,
        attention_slots_needed: 4,
        mlp_slots_needed: 4,
        norm_slots_needed: 4,
        optimizer_slots_needed: 2,
    };
    let registry = Arc::new(
        PoolRegistry::new(&profile, &TensorLocationDict::empty(), 4 * 1024 * 1024).expect("registry"),
    );
    let gauge = MemoryPressureGauge::new(30);
    registry.set_pressure_gauge(gauge.clone());
    let _ = CoScheduler::new(gauge.clone()).expect("coscheduler");

    let mut group = c.benchmark_group("pressure");

    // Zero pressure: pool empty, no callbacks fire
    group.bench_function("sample_notify_zero_pressure", |b| {
        b.iter(|| {
            gauge.sample_and_notify(black_box(&registry));
        });
    });

    // High pressure: fill 85% of slots → high callback fires every sample
    group.bench_function("sample_notify_high_pressure", |b| {
        let slots: Vec<_> = (0..3)
            .filter_map(|_| registry.claim(LayerKind::Attention).ok())
            .collect();
        b.iter(|| {
            gauge.sample_and_notify(black_box(&registry));
        });
        drop(slots);
    });

    // Current pressure read (no sampling — just atomic load)
    group.bench_function("current_pressure_read", |b| {
        b.iter(|| {
            black_box(gauge.current_pressure());
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4: EWA loss-scale table — update throughput over 80 layers
// ---------------------------------------------------------------------------

fn bench_ewa_scale_table(c: &mut Criterion) {
    const NUM_LAYERS: usize = 80;
    const ELEMENTS: usize = 10_000;

    let mut group = c.benchmark_group("ewa");

    // Single-layer update
    group.bench_function("scale_table_update_one_layer", |b| {
        let mut table = PerLayerScaleTable::new(NUM_LAYERS, 0.05);
        b.iter(|| {
            let _ = table.update(black_box(0), black_box(ELEMENTS), black_box(100u32));
        });
    });

    // Full model update: one update per layer per step (80 calls)
    group.bench_function("scale_table_update_all_80_layers", |b| {
        let mut table = PerLayerScaleTable::new(NUM_LAYERS, 0.05);
        b.iter(|| {
            for layer in 0..NUM_LAYERS {
                let _ = table.update(black_box(layer), black_box(ELEMENTS), black_box(50u32));
            }
        });
    });

    // Scale read (hot path in training loop for every grad scale application)
    group.bench_function("scale_table_get_scale", |b| {
        let table = PerLayerScaleTable::new(NUM_LAYERS, 0.05);
        b.iter(|| {
            for layer in 0..NUM_LAYERS {
                black_box(table.get_scale(layer).unwrap_or(1.0));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5: INT8 checkpoint compression (mock CUDA — runs on CPU)
// ---------------------------------------------------------------------------

fn bench_int8_compression(c: &mut Criterion) {
    let stream = CudaStream::new().expect("stream");

    let configs: &[(usize, usize, &str)] = &[
        (4, 128, "4ch_128el"),     // 512 FP16 elements — small checkpoint
        (32, 512, "32ch_512el"),   // 16384 elements — medium activation
        (64, 1024, "64ch_1024el"), // 65536 elements — layer activation
    ];

    let mut group = c.benchmark_group("int8");

    for &(n_channels, elems_per_channel, label) in configs {
        let n_total = n_channels * elems_per_channel;
        group.throughput(Throughput::Elements(n_total as u64));

        let src: Vec<u16> = (0..n_total).map(|i| (i as u16) & 0x7BFF).collect();
        let mut compressed = vec![0i8; n_total];
        let mut scales = vec![0.0f32; n_channels];
        let mut restored = vec![0u16; n_total];

        group.bench_with_input(
            BenchmarkId::new("compress_fp16_to_int8", label),
            &(n_channels, elems_per_channel),
            |b, &(ch, el)| {
                b.iter(|| {
                    compress_checkpoint_fp16_to_int8(
                        black_box(src.as_ptr()),
                        black_box(compressed.as_mut_ptr()),
                        black_box(scales.as_mut_ptr()),
                        black_box(ch),
                        black_box(el),
                        &stream,
                    )
                    .expect("compress");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("decompress_int8_to_fp16", label),
            &(n_channels, elems_per_channel),
            |b, &(ch, el)| {
                // Ensure scales are initialized
                for s in scales.iter_mut() {
                    *s = 0.01;
                }
                b.iter(|| {
                    decompress_checkpoint_int8_to_fp16(
                        black_box(compressed.as_ptr()),
                        black_box(restored.as_mut_ptr()),
                        black_box(scales.as_ptr()),
                        black_box(ch),
                        black_box(el),
                        &stream,
                    )
                    .expect("decompress");
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 6: zstd delta compression (ssd-wear path, CPU)
// ---------------------------------------------------------------------------

#[cfg(feature = "ssd-wear")]
fn bench_delta_compression(c: &mut Criterion) {
    use ramflow::nvme::write_budget::{compress_delta, decompress_and_apply_delta};

    let sizes: &[(usize, &str)] = &[
        (64 * 1024, "64KB"),
        (512 * 1024, "512KB"),
        (4 * 1024 * 1024, "4MB"),
    ];

    let tmp_dir = std::path::PathBuf::from(std::env::temp_dir()).join("ramflow_bench_delta");
    std::fs::create_dir_all(&tmp_dir).ok();

    let mut group = c.benchmark_group("delta");

    for &(size, label) in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        // Near-zero deltas (typical at standard learning rates)
        let original: Vec<u8> = (0..size)
            .map(|i| ((i * 7919 + 13337) & 0xFF) as u8)
            .collect();
        let updated: Vec<u8> = original
            .iter()
            .enumerate()
            .map(|(i, &b)| b.wrapping_add(if i % 64 == 0 { 1 } else { 0 }))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("compress_delta", label),
            &size,
            |b, _| {
                b.iter(|| {
                    compress_delta(
                        black_box(0),
                        black_box(&updated),
                        black_box(&original),
                        black_box(&tmp_dir),
                    )
                    .expect("compress_delta");
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("decompress_delta", label),
            &size,
            |b, _| {
                compress_delta(0, &updated, &original, &tmp_dir).expect("setup compress");
                b.iter(|| {
                    decompress_and_apply_delta(
                        black_box(0),
                        black_box(&original),
                        black_box(&tmp_dir),
                    )
                    .expect("decompress_delta");
                });
            },
        );
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    group.finish();
}

#[cfg(not(feature = "ssd-wear"))]
fn bench_delta_compression(_c: &mut Criterion) {}

// ---------------------------------------------------------------------------
// Benchmark 7: O_DIRECT alignment validation — hot-path guard overhead
// ---------------------------------------------------------------------------

fn bench_alignment_validation(c: &mut Criterion) {
    use ramflow::nvme::prefetch::validate_direct_io_alignment;

    let buf = PinnedBuffer::alloc(4096).expect("alloc");

    let mut group = c.benchmark_group("alignment");

    group.bench_function("validate_direct_io_alignment_valid", |b| {
        b.iter(|| {
            validate_direct_io_alignment(
                black_box(512),  // aligned offset
                black_box(4096), // aligned length
                black_box(buf.as_ptr()),
            )
            .expect("valid alignment");
        });
    });

    group.bench_function("validate_direct_io_alignment_reject", |b| {
        b.iter(|| {
            let _ = validate_direct_io_alignment(
                black_box(100),  // misaligned offset
                black_box(4096),
                black_box(buf.as_ptr()),
            );
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_allocator,
    bench_pool_claim,
    bench_pressure_gauge,
    bench_ewa_scale_table,
    bench_int8_compression,
    bench_delta_compression,
    bench_alignment_validation,
);
criterion_main!(benches);
