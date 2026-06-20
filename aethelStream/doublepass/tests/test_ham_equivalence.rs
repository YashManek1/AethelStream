//! T-HAM: Hybrid Activation Materialization equivalence tests.
//!
//! Verifies:
//! 1. RECOMPUTE and OFFLOAD paths produce bit-identical gradients for the same segment.
//! 2. SARS picks RECOMPUTE under a simulated I/O-bound profile.
//! 3. SARS picks OFFLOAD under a simulated compute-bound profile.
//! 4. OFFLOAD stats: pcie_bytes > 0, recompute_flops == 0.
//! 5. RECOMPUTE stats: pcie_bytes == 0, recompute_flops > 0.

#[cfg(feature = "ham-offload")]
mod ham_tests {
    use doublepass::forward::{single_layer_forward, BlockConfig, BlockWeights};
    use doublepass::ham::{
        backward_offload, backward_recompute, HamAction, Sars, SegmentActivationStore,
    };
    use flowcast::HardwareProfile;

    // -----------------------------------------------------------------------
    // Test fixture helpers
    // -----------------------------------------------------------------------

    fn tiny_cfg() -> BlockConfig {
        BlockConfig {
            d_model: 8,
            n_heads: 2,
            d_ff: 16,
            seq_len: 2,
            batch: 1,
            dropout_p: 0.0, // no dropout -> recompute is deterministic without RNG
        }
    }

    fn make_weights(cfg: &BlockConfig) -> Vec<BlockWeights> {
        (0..2).map(|_| BlockWeights::from_formula(cfg)).collect()
    }

    fn make_input(cfg: &BlockConfig) -> Vec<f32> {
        let n = cfg.bs() * cfg.d_model;
        (0..n).map(|i| (i as f32 * 0.05 + 0.1).sin()).collect()
    }

    fn make_upstream(cfg: &BlockConfig) -> Vec<f32> {
        let n = cfg.bs() * cfg.d_model;
        (0..n)
            .map(|i| (i as f32 * 0.07 + 0.3).cos() * 0.01)
            .collect()
    }

    fn profile_io_bound() -> HardwareProfile {
        // Slow PCIe (0.5 GB/s), fast GPU (mean_forward+backward = 0.1 ms x 2 layers)
        // t_io   = (2 * 1024 bytes) / (0.5e9) = ~4 us
        // t_cmp  = 0.2 ms * 2 = 0.4 ms -> t_io << t_cmp ... that's compute-bound
        // To be I/O-bound we need t_io > t_cmp:
        //   pcie very slow: 1e-6 GB/s, weight_bytes=1M -> t_io = 1e6/(1e-6 * 1e9) = 1000 s >> 0.4 ms
        HardwareProfile {
            nvme_bandwidth_gbs: 1.0,
            pcie_bandwidth_gbs: 1e-6, // extremely slow PCIe -> I/O-bound
            gpu_bandwidth_gbs: 800.0,
            mean_forward_ms: 0.1,
            mean_backward_ms: 0.1,
            sample_count: 5,
            layer_plan: vec![],
            optimal_super_shard_bytes: 0,
        }
    }

    fn profile_compute_bound() -> HardwareProfile {
        // Very fast PCIe (100 GB/s), slow GPU (mean_fwd+bwd = 500 ms per layer)
        // t_io   = 1_000_000 / (100e9) = 1e-5 s = 0.01 ms
        // t_cmp  = 500 ms * 2 layers = 1000 ms -> t_io << t_cmp -> compute-bound -> OFFLOAD
        HardwareProfile {
            nvme_bandwidth_gbs: 1.0,
            pcie_bandwidth_gbs: 100.0, // fast PCIe
            gpu_bandwidth_gbs: 800.0,
            mean_forward_ms: 250.0, // very slow GPU
            mean_backward_ms: 250.0,
            sample_count: 5,
            layer_plan: vec![],
            optimal_super_shard_bytes: 0,
        }
    }

    // -----------------------------------------------------------------------
    // T-HAM-A: RECOMPUTE == OFFLOAD (bit-identical gradients)
    // -----------------------------------------------------------------------

    #[test]
    fn t_ham_a_bit_identical_gradients() {
        let cfg = tiny_cfg();
        let weights = make_weights(&cfg);
        let input = make_input(&cfg);
        let upstream = make_upstream(&cfg);

        let layer_indices: Vec<usize> = vec![0, 1];
        let num_micro_batches = 1;

        // Forward pass: collect full activations for OFFLOAD and checkpoint for RECOMPUTE.
        let mut fwd_all: Vec<Vec<doublepass::forward::SingleLayerFwdOut>> =
            vec![Vec::new(); num_micro_batches];

        let mut act = input.clone();
        for &li in &layer_indices {
            let fwd_out = single_layer_forward(&cfg, &weights[li], &act);
            act = fwd_out.output.clone();
            fwd_all[0].push(fwd_out);
        }

        // Segment checkpoint = input to layer 0.
        let checkpoint_bufs = vec![input.clone()];

        // OFFLOAD store: built from the full forward pass.
        let store = SegmentActivationStore::new(fwd_all);

        // Run RECOMPUTE backward.
        let mut ups_recompute = vec![upstream.clone()];
        let (grads_rc, stats_rc) = backward_recompute(
            &cfg,
            &weights,
            &layer_indices,
            &checkpoint_bufs,
            &[], // no dropout -> empty rng_states
            &mut ups_recompute,
            0,
        )
        .unwrap();

        // Run OFFLOAD backward.
        let mut ups_offload = vec![upstream.clone()];
        let (grads_ol, stats_ol) =
            backward_offload(&cfg, &weights, &layer_indices, &store, &mut ups_offload, 0).unwrap();

        // --- Bit-identical assertion ---
        assert_eq!(
            grads_rc.len(),
            grads_ol.len(),
            "number of layer grad sets must match"
        );
        for (li, (rc, ol)) in grads_rc.iter().zip(grads_ol.iter()).enumerate() {
            assert_eq!(rc.d_wq, ol.d_wq, "d_wq mismatch at layer {li}");
            assert_eq!(rc.d_wk, ol.d_wk, "d_wk mismatch at layer {li}");
            assert_eq!(rc.d_wv, ol.d_wv, "d_wv mismatch at layer {li}");
            assert_eq!(rc.d_wo, ol.d_wo, "d_wo mismatch at layer {li}");
            assert_eq!(rc.d_rms1_w, ol.d_rms1_w, "d_rms1_w mismatch at layer {li}");
            assert_eq!(rc.d_rms2_w, ol.d_rms2_w, "d_rms2_w mismatch at layer {li}");
            assert_eq!(rc.d_wg, ol.d_wg, "d_wg mismatch at layer {li}");
            assert_eq!(rc.d_wu, ol.d_wu, "d_wu mismatch at layer {li}");
            assert_eq!(rc.d_wd, ol.d_wd, "d_wd mismatch at layer {li}");
            assert_eq!(rc.d_input, ol.d_input, "d_input mismatch at layer {li}");
        }

        // Upstream gradients after backward must also match.
        assert_eq!(ups_recompute, ups_offload, "upstream grads must match");

        // --- Stats checks ---
        // OFFLOAD: pcie_bytes > 0, recompute_flops == 0
        assert!(
            stats_ol.pcie_bytes > 0,
            "OFFLOAD must report non-zero pcie_bytes; got {}",
            stats_ol.pcie_bytes
        );
        assert_eq!(
            stats_ol.recompute_flops, 0.0,
            "OFFLOAD must report zero recompute_flops"
        );
        assert_eq!(stats_ol.action, HamAction::Offload);

        // RECOMPUTE: pcie_bytes == 0, recompute_flops > 0
        assert_eq!(
            stats_rc.pcie_bytes, 0,
            "RECOMPUTE must report zero pcie_bytes"
        );
        assert!(
            stats_rc.recompute_flops > 0.0,
            "RECOMPUTE must report non-zero recompute_flops; got {}",
            stats_rc.recompute_flops
        );
        assert_eq!(stats_rc.action, HamAction::Recompute);
    }

    // -----------------------------------------------------------------------
    // T-HAM-B: SARS picks RECOMPUTE under I/O-bound profile
    // -----------------------------------------------------------------------

    #[test]
    fn t_ham_b_sars_picks_recompute_when_io_bound() {
        let sars = Sars::new(profile_io_bound());
        // Large weight bytes to ensure I/O-bound (1 MB with 1e-6 GB/s PCIe)
        let segment_weight_bytes = 1_000_000_u64;
        let num_layers = 2;

        assert!(
            sars.is_io_bound(segment_weight_bytes, num_layers),
            "profile with 1e-6 GB/s PCIe must be I/O-bound"
        );
        let action = sars.select(0, segment_weight_bytes, num_layers);
        assert_eq!(
            action,
            HamAction::Recompute,
            "I/O-bound profile must select RECOMPUTE"
        );
    }

    // -----------------------------------------------------------------------
    // T-HAM-C: SARS picks OFFLOAD under compute-bound profile
    // -----------------------------------------------------------------------

    #[test]
    fn t_ham_c_sars_picks_offload_when_compute_bound() {
        let sars = Sars::new(profile_compute_bound());
        let segment_weight_bytes = 1_000_000_u64;
        let num_layers = 2;

        assert!(
            !sars.is_io_bound(segment_weight_bytes, num_layers),
            "profile with 100 GB/s PCIe and 500 ms/layer compute must be compute-bound"
        );
        let action = sars.select(0, segment_weight_bytes, num_layers);
        assert_eq!(
            action,
            HamAction::Offload,
            "compute-bound profile must select OFFLOAD"
        );
    }

    // -----------------------------------------------------------------------
    // T-HAM-D: pcie_bytes scales with activation count
    // -----------------------------------------------------------------------

    #[test]
    fn t_ham_d_pcie_bytes_scales_with_activations() {
        let cfg = tiny_cfg();
        let weights = make_weights(&cfg);
        let input = make_input(&cfg);
        let layer_indices: Vec<usize> = vec![0, 1];

        let mut fwd_all: Vec<Vec<doublepass::forward::SingleLayerFwdOut>> = vec![Vec::new()];
        let mut act = input.clone();
        for &li in &layer_indices {
            let fwd_out = single_layer_forward(&cfg, &weights[li], &act);
            act = fwd_out.output.clone();
            fwd_all[0].push(fwd_out);
        }

        let store = SegmentActivationStore::new(fwd_all);
        // Each SingleLayerFwdOut holds many Vec<f32> buffers; total > 0.
        assert!(
            store.pcie_bytes() > 0,
            "store must report non-zero pcie_bytes; got {}",
            store.pcie_bytes()
        );
        // pcie_bytes must equal element_count * 4 (f32 size).
        assert_eq!(store.pcie_bytes(), store.element_count * 4);
    }

    // -----------------------------------------------------------------------
    // T-HAM-E: zero-weight-bytes segment is I/O-bound (infinite t_io / 0)
    // -----------------------------------------------------------------------

    #[test]
    fn t_ham_e_zero_bandwidth_is_infinite_io() {
        let mut profile = profile_compute_bound();
        profile.pcie_bandwidth_gbs = 0.0; // division by zero guard
        let sars = Sars::new(profile);
        assert!(
            sars.is_io_bound(1, 1),
            "zero PCIe bandwidth must always be I/O-bound"
        );
    }

    // -----------------------------------------------------------------------
    // T-HAM-F: select_action maps correctly to ActivationAction variants
    // -----------------------------------------------------------------------

    #[test]
    fn t_ham_f_select_action_maps_to_activation_action() {
        use doublepass::ham::select_action;
        use doublepass::plan::ActivationAction;

        let io_profile = profile_io_bound();
        let cmp_profile = profile_compute_bound();

        let action_io = select_action(0, &io_profile, 0.0, 1_000_000, 2);
        assert_eq!(
            action_io,
            ActivationAction::Recompute,
            "I/O-bound -> ActivationAction::Recompute"
        );

        let action_cmp = select_action(0, &cmp_profile, 0.0, 1_000_000, 2);
        assert_eq!(
            action_cmp,
            ActivationAction::PageCompressedRam,
            "compute-bound -> ActivationAction::PageCompressedRam"
        );
    }
}

// When the feature is not enabled, compile a trivial placeholder so cargo does
// not complain about an empty test binary.
#[cfg(not(feature = "ham-offload"))]
#[test]
fn t_ham_feature_not_enabled() {}
