//! Test 5 — Optimizer state file integrity (save/resume).
//!
//! After 500 training steps, save optimizer states, reload into a fresh run,
//! and assert loss at step 501 matches within 1e-4.
//!
//! Run: cargo test --no-default-features --features mock-cuda -p galore test_optimizer_state_resume -- --nocapture

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/mod.rs"]
mod common;

use common::*;
use doublepass::OptimizerBackend;
use galore::GaLoreOptimizer;

const TRAIN_STEPS: usize = 500;
const LR: f32 = 1e-3;
const VOCAB: usize = 64;

#[test]
fn test_optimizer_state_resume_loss_matches_at_step_501() {
    let cfg = config_scale_test();
    let num_layers = 2;
    let inputs = make_inputs(cfg.d_model, cfg.seq_len, cfg.batch);
    let labels = make_labels(cfg.seq_len, cfg.batch, VOCAB);
    let lm_head = make_lm_head(VOCAB, cfg.d_model);

    let state_path = temp_state_path("resume");

    // Original run: 500 steps then save
    let mut model_orig = Model::new(num_layers, cfg);
    let adam = default_adam(LR);
    let galore_cfg = galore_config(state_path.clone(), 0, adam);
    let opt_orig = GaLoreOptimizer::new(galore_cfg.clone(), &param_specs(&model_orig)).expect("create");
    train_with_galore(
        &mut model_orig,
        &lm_head,
        &inputs,
        &labels,
        VOCAB,
        &opt_orig,
        TRAIN_STEPS,
        0,
    );
    opt_orig.flush_all_to_file().expect("flush");

    // Step 501 loss in original run (continue one more step)
    let mut model_orig_cont = clone_model(&model_orig);
    let loss_501_orig = {
        let plan = training_plan();
        let step_cfg = step_config(VOCAB);
        let trainable = trainable_layers(&model_orig_cont);
        let out = doublepass::train_step::full_training_step(
            &model_orig_cont,
            &lm_head,
            &inputs,
            &labels,
            &plan,
            &step_cfg,
            &opt_orig,
            &trainable,
        )
        .expect("step 501");
        apply_galore_deltas(&mut model_orig_cont, &opt_orig);
        opt_orig.notify_step(TRAIN_STEPS as u64 + 1);
        out.loss
    };

    // Release mmap lock before reopening state file (required on Windows).
    drop(opt_orig);

    // Resumed run: reload state at step 500, evaluate step 501
    let mut model_resumed = clone_model(&model_orig);
    let opt_resumed =
        GaLoreOptimizer::open(galore_cfg, &param_specs(&model_resumed)).expect("open");
    assert_eq!(
        opt_resumed.step_count(),
        TRAIN_STEPS as u64,
        "resumed step count must match saved checkpoint"
    );

    let loss_501_resumed = {
        let plan = training_plan();
        let step_cfg = step_config(VOCAB);
        let trainable = trainable_layers(&model_resumed);
        let out = doublepass::train_step::full_training_step(
            &model_resumed,
            &lm_head,
            &inputs,
            &labels,
            &plan,
            &step_cfg,
            &opt_resumed,
            &trainable,
        )
        .expect("resumed step 501");
        apply_galore_deltas(&mut model_resumed, &opt_resumed);
        opt_resumed.notify_step(TRAIN_STEPS as u64 + 1);
        out.loss
    };

    println!(
        "Resume integrity: step 501 loss original={loss_501_orig:.8}, resumed={loss_501_resumed:.8}, diff={:.2e}",
        (loss_501_orig - loss_501_resumed).abs()
    );

    assert!(
        (loss_501_orig - loss_501_resumed).abs() < 1e-4,
        "Step 501 loss mismatch after resume: orig={loss_501_orig}, resumed={loss_501_resumed}"
    );

    // Verify β1, β2, ε loaded from header
    let file = galore::OptimizerStateFile::open(&state_path).expect("open file");
    assert!((file.header().beta1 - adam.beta1).abs() < 1e-6);
    assert!((file.header().beta2 - adam.beta2).abs() < 1e-6);
    assert!((file.header().eps - adam.eps).abs() < 1e-8);

    let _ = std::fs::remove_file(state_path);
}

#[test]
fn test_optimizer_state_header_hyperparams_portable() {
    let path = temp_state_path("header_portable");
    let dims = vec![(32u32, 32u32)];
    let ranks = vec![8u32];
    let adam = galore::AdamWConfig {
        beta1: 0.88,
        beta2: 0.995,
        eps: 1e-7,
        ..galore::AdamWConfig::default()
    };
    let mut file = galore::OptimizerStateFile::create(&path, &dims, &ranks, &adam).expect("create");
    file.write_header(&adam, 500).expect("write header");
    file.flush().expect("flush");
    drop(file);

    let loaded = galore::OptimizerStateFile::open(&path).expect("open");
    assert!((loaded.header().beta1 - 0.88).abs() < 1e-6);
    assert!((loaded.header().beta2 - 0.995).abs() < 1e-6);
    assert!((loaded.header().eps - 1e-7).abs() < 1e-8);
    assert_eq!(loaded.step_count(), 500);

    let _ = std::fs::remove_file(path);
}
