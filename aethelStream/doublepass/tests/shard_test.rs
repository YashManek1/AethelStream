//! T-SHARD: WeightProvider + Model::from_provider integration tests.
//!
//! Tests run against SyntheticProvider (no file I/O) so they pass in CI
//! without a real model checkpoint.

use doublepass::{
    forward::{full_forward, BlockConfig, Model},
    plan::TrainingPlan,
    weight_provider::SyntheticProvider,
};

fn small_cfg() -> BlockConfig {
    BlockConfig {
        d_model: 64,
        n_heads: 4,
        d_ff: 128,
        seq_len: 8,
        batch: 1,
        dropout_p: 0.0,
    }
}

fn default_plan() -> TrainingPlan {
    TrainingPlan {
        checkpoint_freq: 1,
        ..TrainingPlan::default()
    }
}

#[test]
fn t_shard_1_synthetic_provider_loads_correct_layer_count() {
    let cfg = small_cfg();
    let mut provider = SyntheticProvider {
        num_layers: 4,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let model = Model::from_provider(&mut provider, cfg).unwrap();
    assert_eq!(model.num_layers(), 4);
}

#[test]
fn t_shard_2_adjacent_layers_have_distinct_weights() {
    let cfg = small_cfg();
    let mut provider = SyntheticProvider {
        num_layers: 3,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let model = Model::from_provider(&mut provider, cfg).unwrap();
    // from_formula_layered uses layer*100 offset — adjacent layers must differ.
    assert_ne!(
        model.layers[0].wq[0],
        model.layers[1].wq[0],
        "layer 0 and 1 wq[0] must differ"
    );
    assert_ne!(
        model.layers[1].wq[0],
        model.layers[2].wq[0],
        "layer 1 and 2 wq[0] must differ"
    );
}

#[test]
fn t_shard_3_forward_runs_on_provider_model() {
    let cfg = small_cfg();
    let mut provider = SyntheticProvider {
        num_layers: 2,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let model = Model::from_provider(&mut provider, cfg).unwrap();

    let tokens = cfg.d_model * cfg.batch * cfg.seq_len;
    let input = vec![0.1_f32; tokens];
    let result = full_forward(&model, &[input], &default_plan(), false).unwrap();

    assert_eq!(result.outputs.len(), 1, "one micro-batch out");
    assert_eq!(result.outputs[0].len(), tokens, "output shape unchanged");
}

#[test]
fn t_shard_4_weight_bytes_scale_linearly_with_layers() {
    let cfg = small_cfg();
    let input = vec![0.1_f32; cfg.d_model * cfg.batch * cfg.seq_len];
    let plan = default_plan();

    let mut p2 = SyntheticProvider {
        num_layers: 2,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let m2 = Model::from_provider(&mut p2, cfg).unwrap();
    let r2 = full_forward(&m2, &[input.clone()], &plan, false).unwrap();

    let mut p4 = SyntheticProvider {
        num_layers: 4,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let m4 = Model::from_provider(&mut p4, cfg).unwrap();
    let r4 = full_forward(&m4, &[input.clone()], &plan, false).unwrap();

    assert_eq!(
        r4.weight_bytes_streamed,
        r2.weight_bytes_streamed * 2,
        "4-layer model streams exactly 2x bytes of 2-layer model"
    );
}

#[test]
fn t_shard_5_provider_output_is_finite() {
    let cfg = small_cfg();
    let mut provider = SyntheticProvider {
        num_layers: 2,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let model = Model::from_provider(&mut provider, cfg).unwrap();
    let input = vec![0.5_f32; cfg.d_model * cfg.batch * cfg.seq_len];
    let result = full_forward(&model, &[input], &default_plan(), false).unwrap();
    let all_finite = result.outputs[0].iter().all(|x| x.is_finite());
    assert!(all_finite, "all output activations must be finite");
}

#[test]
fn t_shard_6_from_provider_matches_new_weights() {
    // Verify that Model::from_provider with SyntheticProvider produces identical weights
    // to Model::new (which uses the same from_formula_layered internally).
    let cfg = small_cfg();
    let n = 3_usize;

    let model_new = Model::new(n, cfg);
    let mut provider = SyntheticProvider {
        num_layers: n as u32,
        d_model: cfg.d_model,
        d_ff: cfg.d_ff,
        n_heads: cfg.n_heads,
    };
    let model_prov = Model::from_provider(&mut provider, cfg).unwrap();

    for (i, (la, lb)) in model_new.layers.iter().zip(&model_prov.layers).enumerate() {
        assert_eq!(la.wq, lb.wq, "layer {} wq mismatch", i);
        assert_eq!(la.rms1_w, lb.rms1_w, "layer {} rms1_w mismatch", i);
    }
}
