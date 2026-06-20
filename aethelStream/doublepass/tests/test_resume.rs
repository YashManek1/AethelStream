//! T-RESUME — Integration test for checkpoint/resume with bit-exact recovery.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "mock-cuda")]
mod tests {
    use doublepass::{
        forward::{BlockConfig, Model},
        plan::TrainingPlan,
        state::ConsistentState,
        train_step::{full_training_step, StepConfig},
        OptimizerBackend,
    };

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

    struct MockM10 {
        committed_state: Option<ConsistentState>,
        committed_weights: Option<Vec<LayerSnap>>,
    }

    #[derive(Clone)]
    struct LayerSnap {
        rms1_w: Vec<f32>,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        rms2_w: Vec<f32>,
        wg: Vec<f32>,
        wu: Vec<f32>,
        wd: Vec<f32>,
    }

    impl MockM10 {
        fn new() -> Self {
            Self {
                committed_state: None,
                committed_weights: None,
            }
        }

        fn snap_layer(layer: &doublepass::BlockWeights) -> LayerSnap {
            LayerSnap {
                rms1_w: layer.rms1_w.clone(),
                wq: layer.wq.clone(),
                wk: layer.wk.clone(),
                wv: layer.wv.clone(),
                wo: layer.wo.clone(),
                rms2_w: layer.rms2_w.clone(),
                wg: layer.wg.clone(),
                wu: layer.wu.clone(),
                wd: layer.wd.clone(),
            }
        }

        fn record_commit(&mut self, state: ConsistentState, model: &Model) {
            self.committed_state = Some(state);
            self.committed_weights = Some(model.layers.iter().map(Self::snap_layer).collect());
        }

        fn recover(&self) -> Option<(ConsistentState, Vec<LayerSnap>)> {
            match (&self.committed_state, &self.committed_weights) {
                (Some(state), Some(weights)) => Some((state.clone(), weights.clone())),
                _ => None,
            }
        }
    }

    fn restore_model(model: &mut Model, snaps: &[LayerSnap]) {
        for (layer, snap) in model.layers.iter_mut().zip(snaps) {
            layer.rms1_w = snap.rms1_w.clone();
            layer.wq = snap.wq.clone();
            layer.wk = snap.wk.clone();
            layer.wv = snap.wv.clone();
            layer.wo = snap.wo.clone();
            layer.rms2_w = snap.rms2_w.clone();
            layer.wg = snap.wg.clone();
            layer.wu = snap.wu.clone();
            layer.wd = snap.wd.clone();
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
    fn t_resume_bit_exact() {
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
        const N: usize = 5;
        const _CRASH_STEP: usize = 3;

        // Path 1: Run uninterrupted for N steps
        let mut model_uninterrupted = Model::new(2, cfg);
        let mut uninterrupted_losses = Vec::new();

        for _step in 0..N {
            let result = full_training_step(
                &model_uninterrupted,
                &lm_head,
                &inputs,
                &labels,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");

            uninterrupted_losses.push(result.loss);
            apply_sgd(&mut model_uninterrupted, &result.layer_grads, LR);
        }

        // Path 2: Run with simulated crash and resume
        let mut model_with_crash = Model::new(2, cfg);
        let mut m10 = MockM10::new();
        let mut crashed_losses = Vec::new();

        // Run steps 0, 1, 2 with commits
        for step in 0..3 {
            let result = full_training_step(
                &model_with_crash,
                &lm_head,
                &inputs,
                &labels,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");

            crashed_losses.push(result.loss);
            apply_sgd(&mut model_with_crash, &result.layer_grads, LR);

            // Commit after each step
            let state = ConsistentState {
                step: step as u64,
                optimizer_version: 0,
                rng_states: vec![],
                data_position: 0,
            };
            m10.record_commit(state, &model_with_crash);
        }

        // Simulate crash: step 3 begins but does NOT commit; step 4 never happens.
        // We'll skip step 3 in the simulation and resume from step 2's checkpoint.

        // Recover from checkpoint (state 2, weights at step 2)
        let (_recovered_state, recovered_weights) = m10.recover().expect("recovery failed");
        let mut model_resumed = Model::new(2, cfg);
        restore_model(&mut model_resumed, &recovered_weights);

        // Run steps 3 and 4 from recovered state
        let mut resumed_losses = Vec::new();
        for _step_in_resume in 0..2 {
            let result = full_training_step(
                &model_resumed,
                &lm_head,
                &inputs,
                &labels,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");

            resumed_losses.push(result.loss);
            apply_sgd(&mut model_resumed, &result.layer_grads, LR);
        }

        // Assert exact match: resumed step 0 should match uninterrupted step 3, etc.
        assert_eq!(
            resumed_losses[0], uninterrupted_losses[3],
            "resumed[0] != uninterrupted[3]: {:.15} != {:.15}",
            resumed_losses[0], uninterrupted_losses[3]
        );
        assert_eq!(
            resumed_losses[1], uninterrupted_losses[4],
            "resumed[1] != uninterrupted[4]: {:.15} != {:.15}",
            resumed_losses[1], uninterrupted_losses[4]
        );

        println!("T-RESUME: bit-exact recovery verified across crash boundary");
    }
}
