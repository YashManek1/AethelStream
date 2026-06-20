//! T-THRU — Roofline model benchmark (MOCK — CPU f32, NOT GPU).
//!
//! Run with: `cargo test --features mock-cuda -- roofline --ignored --nocapture`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(feature = "mock-cuda")]
mod tests {
    use doublepass::{
        forward::{BlockConfig, Model},
        plan::{ActivationAction, SegmentPlan, TrainingPlan},
        train_step::{full_training_step, StepConfig},
        OptimizerBackend,
    };
    use std::time::Instant;

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

    fn measure_cpu_bw_gbs() -> f64 {
        // Measure bandwidth by allocating 256MB and timing a read+write loop
        const SIZE: usize = 256 * 1024 * 1024 / 4; // 256MB of f32s
        let mut data: Vec<f32> = vec![0.0; SIZE];

        let start = Instant::now();
        for i in 0..SIZE {
            data[i] = data[i] + 1.0;
        }
        let elapsed = start.elapsed().as_secs_f64();

        // Two accesses per iteration (read + write)
        let bytes = (SIZE * 8) as f64;
        let gbs = bytes / (1e9 * elapsed);
        gbs.max(0.1) // Avoid division by zero
    }

    fn measure_cpu_gflops() -> f64 {
        // Measure throughput with 100 dot products of 1M floats
        const N: usize = 1_000_000;
        const REPS: usize = 100;

        let v1: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
        let v2: Vec<f32> = (0..N).map(|i| (i as f32 * 0.002).sin()).collect();

        let start = Instant::now();
        let mut acc = 0.0_f32;
        for _ in 0..REPS {
            for (a, b) in v1.iter().zip(&v2) {
                acc += a * b;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();

        // 2 FLOPs per multiply-accumulate, N MACs per dot product
        let flops = (2.0 * N as f64 * REPS as f64) / (1e9 * elapsed);
        flops.max(0.1) // Avoid division by zero
    }

    #[derive(Debug)]
    struct RoofEntry {
        label: &'static str,
        g: usize,
        sarp_retain: bool,
        t_iter_ms: f64,
        t_io_ms: f64,
        t_compute_ms: f64,
        t_roof_ms: f64,
        ratio: f64,
        tokens_per_sec: f64,
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
            .map(|i| (i as f32 * 0.01 - 0.32).sin())
            .collect()
    }

    fn make_inputs(d_model: usize, seq_len: usize, batch: usize, g: usize) -> Vec<Vec<f32>> {
        (0..g)
            .map(|_| vec![0.1; seq_len * batch * d_model])
            .collect()
    }

    fn make_labels(seq_len: usize, batch: usize, _vocab_size: usize, g: usize) -> Vec<u32> {
        let base: Vec<u32> = [1u32, 2, 3, 0]
            .iter()
            .copied()
            .cycle()
            .take(seq_len * batch)
            .collect();
        // Repeat once per micro-batch so labels.len() == g * batch * seq_len
        base.iter().copied().cycle().take(g * base.len()).collect()
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
    #[ignore]
    fn t_thru_roofline() {
        let cpu_bw_gbs = measure_cpu_bw_gbs();
        let cpu_gflops = measure_cpu_gflops();

        println!("\n=== AethelStream M5 Roofline (MOCK — CPU f32, NOT GPU) ===");
        println!("CPU bandwidth : {:.2} GB/s  [MOCK]", cpu_bw_gbs);
        println!("CPU throughput: {:.2} GFLOPS [MOCK]", cpu_gflops);
        println!();

        let cfg = make_config();
        let vocab_size = 64;
        let d_model = cfg.d_model;
        let seq_len = cfg.seq_len;
        let batch = cfg.batch;
        let lm_head = make_lm_head(vocab_size, d_model);
        let plan = TrainingPlan::default();
        let step_cfg = StepConfig {
            vocab_size,
            chunk_size: 32,
            keep_resident: false,
            compress_checkpoints: false,
        };
        let optimizer = NoOpOpt;
        let trainable_layers = (0..2)
            .flat_map(|li| {
                vec![
                    (li, "d_rms1_w".to_string()),
                    (li, "d_wq".to_string()),
                    (li, "d_wk".to_string()),
                    (li, "d_wv".to_string()),
                    (li, "d_wo".to_string()),
                    (li, "d_rms2_w".to_string()),
                    (li, "d_wg".to_string()),
                    (li, "d_wu".to_string()),
                    (li, "d_wd".to_string()),
                ]
            })
            .collect::<Vec<_>>();

        let mut results = Vec::new();

        // Test G=1,2,4 × {no-SARP, with-SARP}
        let test_configs = vec![
            (1, false, "G=1, greedy"),
            (1, true, "G=1, SARP"),
            (2, false, "G=2, greedy"),
            (2, true, "G=2, SARP"),
            (4, false, "G=4, greedy"),
            (4, true, "G=4, SARP"),
        ];

        for (g, sarp_retain, label) in test_configs {
            let inputs = make_inputs(d_model, seq_len, batch, g);
            let labels = make_labels(seq_len, batch, vocab_size, g);

            // Build plan with schedule if SARP
            let plan = if sarp_retain {
                let mut p = TrainingPlan::default();
                p.activation_schedule = (0..2)
                    .map(|i| SegmentPlan {
                        segment_index: i,
                        action: ActivationAction::RetainVram,
                        recompute_ops: vec![false; 7],
                    })
                    .collect();
                p
            } else {
                TrainingPlan::default()
            };

            // Warm up 3 steps
            let mut model = Model::new(2, cfg);
            for _ in 0..3 {
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
                apply_sgd(&mut model, &result.layer_grads, 0.05);
            }

            // Time 10 steps
            let start = Instant::now();
            for _ in 0..10 {
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
                apply_sgd(&mut model, &result.layer_grads, 0.05);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let t_iter_ms = (elapsed / 10.0) * 1000.0;

            // Compute roofline
            let n_layers = 2;
            let weight_bytes =
                (n_layers as u64) * (d_model * (d_model + d_model / 8 + d_model * 4)) as u64 * 4; // Rough estimate
            let params = weight_bytes / 4; // f32s
            let t_io_ms = (weight_bytes as f64) / (cpu_bw_gbs * 1e9) * 1000.0;
            let t_compute_ms =
                (2.0 * params as f64 * g as f64 * seq_len as f64) / (cpu_gflops * 1e9) * 1000.0;
            let t_roof_ms = t_io_ms.max(t_compute_ms);
            let ratio = t_iter_ms / t_roof_ms;

            let tokens_per_sec = (g * batch * seq_len) as f64 / (t_iter_ms / 1000.0);

            results.push(RoofEntry {
                label,
                g,
                sarp_retain,
                t_iter_ms,
                t_io_ms,
                t_compute_ms,
                t_roof_ms,
                ratio,
                tokens_per_sec,
            });
        }

        // Print table
        println!(
            "{:<15} {:>2} {:>5} {:>11} {:>10} {:>10} {:>11} {:>8} {:>12}",
            "label",
            "G",
            "SARP",
            "T_iter(ms)",
            "T_io(ms)",
            "T_comp(ms)",
            "T_roof(ms)",
            "ratio",
            "tok/s"
        );
        println!(
            "{:<15} {:>2} {:>5} {:>11} {:>10} {:>10} {:>11} {:>8} {:>12}",
            "───────────────",
            "──",
            "─────",
            "───────────",
            "──────────",
            "──────────",
            "───────────",
            "────────",
            "────────────"
        );

        for entry in &results {
            println!(
                "{:<15} {:>2} {:>5} {:>11.3} {:>10.3} {:>10.3} {:>11.3} {:>8.2} {:>12.0}",
                entry.label,
                entry.g,
                if entry.sarp_retain { "yes" } else { "no" },
                entry.t_iter_ms,
                entry.t_io_ms,
                entry.t_compute_ms,
                entry.t_roof_ms,
                entry.ratio,
                entry.tokens_per_sec
            );
        }

        println!();
        println!("NOTE: All numbers are MOCK (CPU f32). Do NOT cite as measured GPU throughput.");
        println!();

        // Verify ratios are within reasonable bounds
        for entry in &results {
            assert!(
                entry.ratio >= 0.05 && entry.ratio <= 200.0,
                "Ratio {} out of bounds for {}: {}",
                entry.ratio,
                entry.label,
                entry.ratio
            );
        }
    }

    #[test]
    #[cfg(feature = "mock-cuda")]
    #[ignore]
    fn t_thru_g_linear() {
        let cfg = make_config();
        let vocab_size = 64;
        let d_model = cfg.d_model;
        let seq_len = cfg.seq_len;
        let batch = cfg.batch;
        let lm_head = make_lm_head(vocab_size, d_model);
        let plan = TrainingPlan::default();
        let step_cfg = StepConfig {
            vocab_size,
            chunk_size: 32,
            keep_resident: false,
            compress_checkpoints: false,
        };
        let optimizer = NoOpOpt;
        let trainable_layers = (0..2)
            .flat_map(|li| {
                vec![
                    (li, "d_rms1_w".to_string()),
                    (li, "d_wq".to_string()),
                    (li, "d_wk".to_string()),
                    (li, "d_wv".to_string()),
                    (li, "d_wo".to_string()),
                    (li, "d_rms2_w".to_string()),
                    (li, "d_wg".to_string()),
                    (li, "d_wu".to_string()),
                    (li, "d_wd".to_string()),
                ]
            })
            .collect::<Vec<_>>();

        // Time G=1
        let inputs_g1 = make_inputs(d_model, seq_len, batch, 1);
        let labels_g1 = make_labels(seq_len, batch, vocab_size, 1);
        let mut model = Model::new(2, cfg);
        for _ in 0..3 {
            let result = full_training_step(
                &model,
                &lm_head,
                &inputs_g1,
                &labels_g1,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");
            apply_sgd(&mut model, &result.layer_grads, 0.05);
        }

        let start = Instant::now();
        for _ in 0..10 {
            let result = full_training_step(
                &model,
                &lm_head,
                &inputs_g1,
                &labels_g1,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");
            apply_sgd(&mut model, &result.layer_grads, 0.05);
        }
        let t_g1 = start.elapsed().as_secs_f64() / 10.0;

        // Time G=2
        let inputs_g2 = make_inputs(d_model, seq_len, batch, 2);
        let labels_g2 = make_labels(seq_len, batch, vocab_size, 2);
        let mut model = Model::new(2, cfg);
        for _ in 0..3 {
            let result = full_training_step(
                &model,
                &lm_head,
                &inputs_g2,
                &labels_g2,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");
            apply_sgd(&mut model, &result.layer_grads, 0.05);
        }

        let start = Instant::now();
        for _ in 0..10 {
            let result = full_training_step(
                &model,
                &lm_head,
                &inputs_g2,
                &labels_g2,
                &plan,
                &step_cfg,
                &optimizer,
                &trainable_layers,
            )
            .expect("full_training_step failed");
            apply_sgd(&mut model, &result.layer_grads, 0.05);
        }
        let t_g2 = start.elapsed().as_secs_f64() / 10.0;

        let ratio = t_g2 / t_g1;

        println!("T-THRU-G-LINEAR [MOCK]:");
        println!("  T(G=1) = {:.3} s", t_g1);
        println!("  T(G=2) = {:.3} s", t_g2);
        println!(
            "  T(G=2)/T(G=1) = {:.3} (target < 2.5 for sub-linear)",
            ratio
        );
        println!();

        assert!(
            ratio < 2.5,
            "G scaling not sub-linear: T(G=2) / T(G=1) = {:.3}",
            ratio
        );
    }
}
