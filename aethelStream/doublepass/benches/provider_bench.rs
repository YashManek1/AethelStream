//! Criterion benchmarks for the WeightProvider→Model→forward_pass pipeline.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use doublepass::{
    forward::{full_forward, BlockConfig, Model},
    plan::TrainingPlan,
    weight_provider::SyntheticProvider,
};

fn make_model(n_layers: u32, d_model: usize, d_ff: usize, n_heads: usize) -> (Model, BlockConfig) {
    let cfg = BlockConfig {
        d_model,
        n_heads,
        d_ff,
        seq_len: 16,
        batch: 1,
        dropout_p: 0.0,
    };
    let mut p = SyntheticProvider { num_layers: n_layers, d_model, d_ff, n_heads };
    let m = Model::from_provider(&mut p, cfg).unwrap();
    (m, cfg)
}

fn default_plan() -> TrainingPlan {
    TrainingPlan {
        checkpoint_freq: 2,
        ..TrainingPlan::default()
    }
}

/// Measure time to load all layers from SyntheticProvider into a Model.
fn bench_provider_load(c: &mut Criterion) {
    c.bench_function("SyntheticProvider load 8×d256", |b| {
        b.iter(|| {
            let cfg = BlockConfig {
                d_model: 256,
                n_heads: 8,
                d_ff: 512,
                seq_len: 8,
                batch: 1,
                dropout_p: 0.0,
            };
            let mut p = SyntheticProvider { num_layers: 8, d_model: 256, d_ff: 512, n_heads: 8 };
            black_box(Model::from_provider(&mut p, cfg).unwrap())
        })
    });
}

/// Measure time for a full forward pass on a 4-layer d=128 model.
fn bench_full_forward_4l(c: &mut Criterion) {
    let (model, cfg) = make_model(4, 128, 256, 4);
    let input = vec![0.1_f32; cfg.d_model * cfg.batch * cfg.seq_len];
    let plan = default_plan();

    c.bench_function("full_forward 4-layer d=128 seq=16", |b| {
        b.iter(|| {
            black_box(
                full_forward(black_box(&model), &[input.clone()], &plan, false).unwrap(),
            )
        })
    });
}

/// Measure time for a full forward pass on an 8-layer d=256 model.
fn bench_full_forward_8l(c: &mut Criterion) {
    let (model, cfg) = make_model(8, 256, 512, 8);
    let input = vec![0.1_f32; cfg.d_model * cfg.batch * cfg.seq_len];
    let plan = default_plan();

    c.bench_function("full_forward 8-layer d=256 seq=16", |b| {
        b.iter(|| {
            black_box(
                full_forward(black_box(&model), &[input.clone()], &plan, false).unwrap(),
            )
        })
    });
}

criterion_group!(benches, bench_provider_load, bench_full_forward_4l, bench_full_forward_8l);
criterion_main!(benches);
