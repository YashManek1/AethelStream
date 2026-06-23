//! Shared training harness for GaLore validation tests (Tests 3–5).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use doublepass::forward::{BlockConfig, BlockWeights};
pub use doublepass::forward::Model;
use doublepass::plan::TrainingPlan;
use doublepass::train_step::{full_training_step, StepConfig};
use doublepass::OptimizerBackend;
use galore::{GaLoreConfig, GaLoreOptimizer, StandardAdamW, AdamWConfig};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// 125M-scale hyperparameters (2-layer reduction; d_model/d_ff/n_heads match GPT-125M ratios).
pub fn config_125m() -> BlockConfig {
    BlockConfig {
        d_model: 64,
        n_heads: 4,
        d_ff: 256,
        seq_len: 4,
        batch: 1,
        dropout_p: 0.0,
    }
}

/// Smaller config for scale-independence test (500 steps).
pub fn config_scale_test() -> BlockConfig {
    BlockConfig {
        d_model: 32,
        n_heads: 2,
        d_ff: 64,
        seq_len: 4,
        batch: 1,
        dropout_p: 0.0,
    }
}

/// Approximate parameter count for a model config.
pub fn count_params(cfg: &BlockConfig, num_layers: usize, vocab_size: usize) -> usize {
    let per_layer = 2 * cfg.d_model + 4 * cfg.d_model * cfg.d_model + 3 * cfg.d_model * cfg.d_ff;
    num_layers * per_layer + vocab_size * cfg.d_model
}

pub fn temp_state_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("galore_{name}_{nanos}.bin"))
}

pub fn make_lm_head(vocab_size: usize, d_model: usize) -> Vec<f32> {
    (0..vocab_size * d_model)
        .map(|i| {
            let x = i as f32 * 0.001 - 0.5;
            x.sin() * 0.05
        })
        .collect()
}

pub fn make_inputs(d_model: usize, seq_len: usize, batch: usize) -> Vec<Vec<f32>> {
    let n = seq_len * batch * d_model;
    vec![(0..n).map(|i| ((i as f32 * 0.013).sin() * 0.1)).collect()]
}

pub fn make_labels(seq_len: usize, batch: usize, vocab_size: usize) -> Vec<u32> {
    (0..seq_len * batch)
        .map(|i| (i as u32 * 7 + 3) % vocab_size as u32)
        .collect()
}

/// Build `(layer_idx, param_name, m, n)` specs for GaLore registration.
pub fn param_specs(model: &Model) -> Vec<(u32, &'static str, usize, usize)> {
    let d = model.cfg.d_model;
    let ff = model.cfg.d_ff;
    let mut specs = Vec::new();
    for li in 0..model.layers.len() {
        specs.push((li as u32, "d_rms1_w", 1, d));
        specs.push((li as u32, "d_wq", d, d));
        specs.push((li as u32, "d_wk", d, d));
        specs.push((li as u32, "d_wv", d, d));
        specs.push((li as u32, "d_wo", d, d));
        specs.push((li as u32, "d_rms2_w", 1, d));
        specs.push((li as u32, "d_wg", ff, d));
        specs.push((li as u32, "d_wu", ff, d));
        specs.push((li as u32, "d_wd", d, ff));
    }
    specs
}

pub fn trainable_layers(model: &Model) -> Vec<(u32, String)> {
    param_specs(model)
        .into_iter()
        .map(|(li, name, _, _)| (li, name.to_string()))
        .collect()
}

/// Flat specs for StandardAdamW: `(layer_idx, param_name, num_elements)`.
pub fn flat_param_specs(model: &Model) -> Vec<(u32, &'static str, usize)> {
    param_specs(model)
        .into_iter()
        .map(|(li, name, m, n)| (li, name, m * n))
        .collect()
}

pub fn apply_weight_delta(weights: &mut [f32], delta: &[f32]) {
    assert_eq!(weights.len(), delta.len());
    for (w, d) in weights.iter_mut().zip(delta.iter()) {
        *w -= d;
    }
}

pub fn apply_galore_deltas(model: &mut Model, opt: &GaLoreOptimizer) {
    for li in 0..model.layers.len() {
        let layer = &mut model.layers[li];
        let idx = li as u32;
        if let Some(d) = opt.take_weight_delta(idx, "d_rms1_w") {
            apply_weight_delta(&mut layer.rms1_w, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wq") {
            apply_weight_delta(&mut layer.wq, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wk") {
            apply_weight_delta(&mut layer.wk, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wv") {
            apply_weight_delta(&mut layer.wv, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wo") {
            apply_weight_delta(&mut layer.wo, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_rms2_w") {
            apply_weight_delta(&mut layer.rms2_w, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wg") {
            apply_weight_delta(&mut layer.wg, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wu") {
            apply_weight_delta(&mut layer.wu, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wd") {
            apply_weight_delta(&mut layer.wd, &d);
        }
    }
}

pub fn apply_standard_deltas(model: &mut Model, opt: &StandardAdamW) {
    for li in 0..model.layers.len() {
        let layer = &mut model.layers[li];
        let idx = li as u32;
        if let Some(d) = opt.take_weight_delta(idx, "d_rms1_w") {
            apply_weight_delta(&mut layer.rms1_w, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wq") {
            apply_weight_delta(&mut layer.wq, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wk") {
            apply_weight_delta(&mut layer.wk, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wv") {
            apply_weight_delta(&mut layer.wv, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wo") {
            apply_weight_delta(&mut layer.wo, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_rms2_w") {
            apply_weight_delta(&mut layer.rms2_w, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wg") {
            apply_weight_delta(&mut layer.wg, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wu") {
            apply_weight_delta(&mut layer.wu, &d);
        }
        if let Some(d) = opt.take_weight_delta(idx, "d_wd") {
            apply_weight_delta(&mut layer.wd, &d);
        }
    }
}

pub fn clone_model(model: &Model) -> Model {
    Model {
        cfg: model.cfg,
        layers: model.layers.clone(),
    }
}

pub fn step_config(vocab_size: usize) -> StepConfig {
    StepConfig {
        vocab_size,
        chunk_size: 256,
        keep_resident: false,
        compress_checkpoints: false,
    }
}

pub fn training_plan() -> TrainingPlan {
    TrainingPlan {
        max_grad_norm: 1.0,
        ..TrainingPlan::default()
    }
}

/// Run `steps` training iterations, returning per-step losses.
pub fn train_with_galore(
    model: &mut Model,
    lm_head: &[f32],
    inputs: &[Vec<f32>],
    labels: &[u32],
    vocab_size: usize,
    opt: &GaLoreOptimizer,
    steps: usize,
    start_step: u64,
) -> Vec<f32> {
    let plan = training_plan();
    let step_cfg = step_config(vocab_size);
    let trainable = trainable_layers(model);
    let mut losses = Vec::with_capacity(steps);

    for s in 0..steps {
        let step = start_step + s as u64 + 1;
        let out = full_training_step(
            model,
            lm_head,
            inputs,
            labels,
            &plan,
            &step_cfg,
            opt,
            &trainable,
        )
        .expect("training step");
        apply_galore_deltas(model, opt);
        opt.notify_step(step);
        losses.push(out.loss);
    }
    let _ = opt.flush_all_to_file();
    losses
}

/// Run `steps` with StandardAdamW baseline.
pub fn train_with_adamw(
    model: &mut Model,
    lm_head: &[f32],
    inputs: &[Vec<f32>],
    labels: &[u32],
    vocab_size: usize,
    opt: &StandardAdamW,
    steps: usize,
    start_step: u64,
) -> Vec<f32> {
    let plan = training_plan();
    let step_cfg = step_config(vocab_size);
    let trainable = trainable_layers(model);
    let mut losses = Vec::with_capacity(steps);

    for s in 0..steps {
        let step = start_step + s as u64 + 1;
        let out = full_training_step(
            model,
            lm_head,
            inputs,
            labels,
            &plan,
            &step_cfg,
            opt,
            &trainable,
        )
        .expect("training step");
        apply_standard_deltas(model, opt);
        opt.notify_step(step);
        losses.push(out.loss);
    }
    losses
}

/// Compute loss at the next step without applying optimizer update (for resume check).
pub fn eval_loss(
    model: &Model,
    lm_head: &[f32],
    inputs: &[Vec<f32>],
    labels: &[u32],
    vocab_size: usize,
) -> f32 {
    struct NoOp;
    impl OptimizerBackend for NoOp {
        fn project_and_accumulate(&self, _: &[f32], _: u32, _: &str) {}
        fn lowrank_grad_sqnorm(&self, _: u32, _: &str) -> f64 {
            0.0
        }
        fn apply_update(&self, _: u32, _: &str, _: f32) {}
        fn zero_accum(&self, _: u32, _: &str) {}
        fn notify_step(&self, _: u64) {}
        fn projector_kind(&self, _: u32, _: &str) -> doublepass::hook::ProjectorKind {
            doublepass::hook::ProjectorKind::None
        }
    }

    let plan = training_plan();
    let step_cfg = step_config(vocab_size);
    let noop = NoOp;
    full_training_step(
        model,
        lm_head,
        inputs,
        labels,
        &plan,
        &step_cfg,
        &noop,
        &[],
    )
    .expect("eval")
    .loss
}

pub fn write_loss_curves(path: &Path, curves: &[(&str, Vec<f32>)]) {
    let mut obj = serde_json::Map::new();
    for (name, losses) in curves {
        obj.insert(
            name.to_string(),
            serde_json::Value::Array(
                losses
                    .iter()
                    .map(|l| serde_json::json!(*l))
                    .collect(),
            ),
        );
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, serde_json::Value::Object(obj).to_string());
}

pub fn default_adam(lr: f32) -> AdamWConfig {
    AdamWConfig {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        scale_lr_by_rank: false,
    }
}

pub fn galore_config(state_path: PathBuf, switch_interval: u64, adam: AdamWConfig) -> GaLoreConfig {
    GaLoreConfig {
        rank: 16,
        switch_interval,
        adam,
        state_file_path: state_path,
        ..GaLoreConfig::default()
    }
}
