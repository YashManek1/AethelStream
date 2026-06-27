//! Test 4 — Scale factor independence (α/r vs no division).
//!
//! Train for 500 steps; assert the version without r division converges faster
//! or at equal rate compared to α/r scaling.
//!
//! Run: cargo test --no-default-features --features mock-cuda -p galore test_galore_scale -- --nocapture

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/mod.rs"]
mod common;

use common::*;
use galore::{AdamWConfig, GaLoreOptimizer};

const STEPS: usize = 500;
const LR: f32 = 1e-3;
const VOCAB: usize = 64;

#[test]
fn test_galore_scale_factor_independence() {
    let cfg = config_scale_test();
    let num_layers = 2;
    let inputs = make_inputs(cfg.d_model, cfg.seq_len, cfg.batch);
    let labels = make_labels(cfg.seq_len, cfg.batch, VOCAB);
    let lm_head = make_lm_head(VOCAB, cfg.d_model);

    // Without α/r division (preferred — decoupled scale factor)
    let mut model_no_div = Model::new(num_layers, cfg);
    let path_no_div = temp_state_path("scale_no_div");
    let adam_no_div = AdamWConfig {
        lr: LR,
        scale_lr_by_rank: false,
        ..AdamWConfig::default()
    };
    let cfg_no_div = galore_config(path_no_div.clone(), 0, adam_no_div);
    let opt_no_div = GaLoreOptimizer::new(cfg_no_div, &param_specs(&model_no_div)).expect("no div");
    let losses_no_div = train_with_galore(
        &mut model_no_div,
        &lm_head,
        &inputs,
        &labels,
        VOCAB,
        &opt_no_div,
        STEPS,
        0,
    );

    // With α/r division (ablation)
    let mut model_with_div = Model::new(num_layers, cfg);
    let path_with_div = temp_state_path("scale_with_div");
    let adam_with_div = AdamWConfig {
        lr: LR,
        scale_lr_by_rank: true,
        ..AdamWConfig::default()
    };
    let cfg_with_div = galore_config(path_with_div.clone(), 0, adam_with_div);
    let opt_with_div =
        GaLoreOptimizer::new(cfg_with_div, &param_specs(&model_with_div)).expect("with div");
    let losses_with_div = train_with_galore(
        &mut model_with_div,
        &lm_head,
        &inputs,
        &labels,
        VOCAB,
        &opt_with_div,
        STEPS,
        0,
    );

    let final_no_div = *losses_no_div.last().expect("no_div");
    let final_with_div = *losses_with_div.last().expect("with_div");
    let initial_no_div = losses_no_div[0];
    let initial_with_div = losses_with_div[0];

    println!(
        "Scale independence: no α/r division final={final_no_div:.6}, with α/r final={final_with_div:.6}"
    );

    write_loss_curves(
        std::path::Path::new("pyref/galore_scale_independence_curves.json"),
        &[
            ("no_rank_division", losses_no_div),
            ("with_rank_division", losses_with_div),
        ],
    );

    // Both variants must converge
    assert!(
        final_no_div < initial_no_div,
        "no α/r division must converge: {final_no_div} vs {initial_no_div}"
    );
    assert!(
        final_with_div < initial_with_div,
        "α/r division must converge: {final_with_div} vs {initial_with_div}"
    );

    // Without r division must converge faster (lower loss) or at equal rate
    assert!(
        final_no_div <= final_with_div * 1.001,
        "No α/r division should converge faster or equally: no_div={final_no_div}, with_div={final_with_div}"
    );

    let _ = std::fs::remove_file(path_no_div);
    let _ = std::fs::remove_file(path_with_div);
}
