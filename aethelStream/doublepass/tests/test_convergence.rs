//! T-CONV-5 — Integration test for M5 full_training_step convergence and reference matching.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "mock-cuda")]
mod tests {
    use doublepass::{
        forward::{BlockConfig, Model},
        plan::TrainingPlan,
        state::RngState,
        train_step::{full_training_step, StepConfig},
        OptimizerBackend,
    };
    use std::sync::Mutex;

    /// No-op optimizer for testing; all methods are no-ops.
    /// `projector_kind` returns `ProjectorKind::None` to trigger grouped fallback in A6.
    struct NoOpOpt;

    impl OptimizerBackend for NoOpOpt {
        fn project_and_accumulate(&self, _grad: &[f32], _layer_idx: u32, _param_name: &str) {}

        fn lowrank_grad_sqnorm(&self, _layer_idx: u32, _param_name: &str) -> f64 {
            0.0
        }

        fn apply_update(&self, _layer_idx: u32, _param_name: &str, _clip_scale: f32) {}

        fn zero_accum(&self, _layer_idx: u32, _param_name: &str) {}

        fn notify_step(&self, _step: u64) {}

        fn projector_kind(
            &self,
            _layer_idx: u32,
            _param_name: &str,
        ) -> doublepass::hook::ProjectorKind {
            doublepass::hook::ProjectorKind::None
        }
    }

    fn make_config() -> BlockConfig {
        BlockConfig {
            d_model: 32,
            n_heads: 2,
            d_ff: 64,
            seq_len: 4,
            batch: 1,
            dropout_p: 0.0,
        }
    }

    fn make_lm_head(vocab_size: usize, d_model: usize) -> Vec<f32> {
        (0..vocab_size * d_model)
            .map(|i| {
                let x = i as f32 * 0.01 - 0.32;
                x.sin()
            })
            .collect()
    }

    fn make_inputs(d_model: usize, seq_len: usize, batch: usize) -> Vec<Vec<f32>> {
        vec![vec![0.1; seq_len * batch * d_model]]
    }

    fn make_labels(seq_len: usize, batch: usize, vocab_size: usize) -> Vec<u32> {
        vec![1u32, 2, 3, 0]
            .iter()
            .take(seq_len * batch)
            .copied()
            .collect()
    }

    fn apply_sgd(model: &mut Model, grads: &[doublepass::ParamGrads], lr: f32) {
        for (layer_idx, layer_grad) in grads.iter().enumerate() {
            let layer = &mut model.layers[layer_idx];

            for (w, dw) in layer.rms1_w.iter_mut().zip(&layer_grad.d_rms1_w) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wq.iter_mut().zip(&layer_grad.d_wq) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wk.iter_mut().zip(&layer_grad.d_wk) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wv.iter_mut().zip(&layer_grad.d_wv) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wo.iter_mut().zip(&layer_grad.d_wo) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.rms2_w.iter_mut().zip(&layer_grad.d_rms2_w) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wg.iter_mut().zip(&layer_grad.d_wg) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wu.iter_mut().zip(&layer_grad.d_wu) {
                *w -= lr * dw;
            }
            for (w, dw) in layer.wd.iter_mut().zip(&layer_grad.d_wd) {
                *w -= lr * dw;
            }
        }
    }

    #[test]
    #[cfg(feature = "mock-cuda")]
    fn t_conv5_m5_converges() {
        let cfg = make_config();
        let mut model = Model::new(2, cfg);
        let vocab_size = 64;
        let d_model = cfg.d_model;
        let seq_len = cfg.seq_len;
        let batch = cfg.batch;
        let lm_head = make_lm_head(vocab_size, d_model);
        let inputs = make_inputs(d_model, seq_len, batch);
        let labels = make_labels(seq_len, batch, vocab_size);
        let plan = TrainingPlan::default();
        let step_cfg = StepConfig {
            vocab_size,
            chunk_size: 32,
            keep_resident: false,
            compress_checkpoints: false,
        };
        let optimizer = NoOpOpt;
        let trainable_layers = vec![
            (0u32, "d_rms1_w".to_string()),
            (0u32, "d_wq".to_string()),
            (0u32, "d_wk".to_string()),
            (0u32, "d_wv".to_string()),
            (0u32, "d_wo".to_string()),
            (0u32, "d_rms2_w".to_string()),
            (0u32, "d_wg".to_string()),
            (0u32, "d_wu".to_string()),
            (0u32, "d_wd".to_string()),
            (1u32, "d_rms1_w".to_string()),
            (1u32, "d_wq".to_string()),
            (1u32, "d_wk".to_string()),
            (1u32, "d_wv".to_string()),
            (1u32, "d_wo".to_string()),
            (1u32, "d_rms2_w".to_string()),
            (1u32, "d_wg".to_string()),
            (1u32, "d_wu".to_string()),
            (1u32, "d_wd".to_string()),
        ];

        const STEPS: usize = 500;
        const LR: f32 = 0.05;
        let mut losses = Vec::new();

        for _step in 0..STEPS {
            let result = full_training_step(
                &model,
                &lm_head,
                &inputs,
                &labels,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");

            losses.push(result.loss);
            apply_sgd(&mut model, &result.layer_grads, LR);
        }

        let first_loss = losses[0];
        let final_loss = losses[STEPS - 1];
        let improvement_ratio = final_loss / first_loss;

        println!(
            "T-CONV-5: loss[0]={:.6}, loss[499]={:.6}, ratio={:.4}",
            first_loss, final_loss, improvement_ratio
        );

        assert!(
            improvement_ratio < 0.99,
            "Loss must decrease by at least 1%: {:.4}",
            improvement_ratio
        );

        // Write convergence curve to JSON
        let json = serde_json::json!({
            "m5_losses": losses,
            "step_count": STEPS,
        });
        let path = "pyref/convergence_curves.json";
        std::fs::create_dir_all("pyref").expect("create pyref dir");
        std::fs::write(path, json.to_string()).expect("write convergence_curves.json");
    }

    #[test]
    #[cfg(feature = "mock-cuda")]
    fn t_conv5_reference_matches() {
        let cfg = make_config();
        let vocab_size = 64;
        let d_model = cfg.d_model;
        let seq_len = cfg.seq_len;
        let batch = cfg.batch;
        let lm_head = make_lm_head(vocab_size, d_model);
        let inputs = make_inputs(d_model, seq_len, batch);
        let labels = make_labels(seq_len, batch, vocab_size);
        let plan = TrainingPlan::default();
        let step_cfg = StepConfig {
            vocab_size,
            chunk_size: 32,
            keep_resident: false,
            compress_checkpoints: false,
        };
        let optimizer = NoOpOpt;
        let trainable_layers = vec![
            (0u32, "d_rms1_w".to_string()),
            (0u32, "d_wq".to_string()),
            (0u32, "d_wk".to_string()),
            (0u32, "d_wv".to_string()),
            (0u32, "d_wo".to_string()),
            (0u32, "d_rms2_w".to_string()),
            (0u32, "d_wg".to_string()),
            (0u32, "d_wu".to_string()),
            (0u32, "d_wd".to_string()),
            (1u32, "d_rms1_w".to_string()),
            (1u32, "d_wq".to_string()),
            (1u32, "d_wk".to_string()),
            (1u32, "d_wv".to_string()),
            (1u32, "d_wo".to_string()),
            (1u32, "d_rms2_w".to_string()),
            (1u32, "d_wg".to_string()),
            (1u32, "d_wu".to_string()),
            (1u32, "d_wd".to_string()),
        ];

        const LR: f32 = 0.05;
        const STEPS: usize = 50;

        // Path 1: Full M5 via full_training_step
        let mut model_m5 = Model::new(2, cfg);
        let mut losses_m5 = Vec::new();

        for _step in 0..STEPS {
            let result = full_training_step(
                &model_m5,
                &lm_head,
                &inputs,
                &labels,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");

            losses_m5.push(result.loss);
            apply_sgd(&mut model_m5, &result.layer_grads, LR);
        }

        // Path 2: Explicit reference path (full_forward_with_retention → streaming_cut_ce → full_backward_sarp)
        let mut model_ref = Model::new(2, cfg);
        let mut losses_ref = Vec::new();

        for _step in 0..STEPS {
            let fwd =
                doublepass::forward::full_forward_with_retention(&model_ref, &inputs, &plan, false)
                    .expect("full_forward_with_retention failed");

            let hidden: Vec<f32> = fwd.outputs.iter().flat_map(|o| o.iter().copied()).collect();

            let loss_out = doublepass::loss::streaming_cut_ce(
                &hidden, &lm_head, d_model, vocab_size, 32, &labels, None,
            )
            .expect("streaming_cut_ce failed");

            losses_ref.push(loss_out.loss);

            let tokens_per_mb = if inputs.len() == 0 {
                0
            } else {
                loss_out.grad_hidden.len() / inputs.len()
            };
            let upstream_grads: Vec<Vec<f32>> = (0..inputs.len())
                .map(|i| loss_out.grad_hidden[i * tokens_per_mb..(i + 1) * tokens_per_mb].to_vec())
                .collect();

            let bwd = doublepass::backward::full_backward_sarp(
                &model_ref,
                &fwd,
                &inputs,
                &upstream_grads,
                &plan,
                false,
                &optimizer,
            )
            .expect("full_backward_sarp failed");

            apply_sgd(&mut model_ref, &bwd.layer_grads, LR);
        }

        // Assert exact match (bit-identical since dropout=0 and same code paths)
        for (i, (m5_loss, ref_loss)) in losses_m5.iter().zip(losses_ref.iter()).enumerate() {
            assert_eq!(
                m5_loss, ref_loss,
                "Loss mismatch at step {}: M5={}, ref={}",
                i, m5_loss, ref_loss
            );
        }

        println!("T-CONV-5 reference: {} steps matched exactly", STEPS);
    }
}
