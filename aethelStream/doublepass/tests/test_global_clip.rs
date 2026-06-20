//! T-CLIP — Exact global-norm gradient clipping (A6 / N3 / N3′).
//!
//! Tests both projector arms (Orthonormal and Random) plus the grouped/local-clip
//! fallback (No-projection). Covers:
//!
//! 1. Orthonormal arm: exact global clip, clip-triggers case and no-clip case.
//! 2. Frobenius norm preservation property: verifies numerically that `‖P·a‖_F = ‖a‖_F`
//!    when P has orthonormal columns — the mathematical justification for exact clipping
//!    without forming the full gradient (GaLore arXiv 2403.03507 §3.2).
//! 3. Random projector (N3′ JL path): clip uses JL-approximate norm; asserts the
//!    relative sq-norm error is bounded (JL lemma) and the clip differs from the exact
//!    reference (demonstrating the approximation).
//! 4. Random projector + O(1) Frobenius accumulator: restores exact match on
//!    clip-critical layers (APOLLO arXiv 2412.05270 mitigation).
//! 5. Grouped/local-clip fallback: engages ONLY when `ProjectorKind::None`; per-layer
//!    scales match the LOMO-style reference (Lv et al. 2023 arXiv 2306.09782 §3.3).
//! 6. Fallback does NOT engage for Orthonormal projectors (gate test).
//!
//! Run: cargo test --features mock-cuda -p doublepass test_global_clip -- --nocapture

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Mutex;

use doublepass::hook::{deferred_apply_with_clip, ClipResult, ProjectorKind};
use doublepass::{OptimizerBackend, TrainingPlan};

// ---------------------------------------------------------------------------
// MockOptimizer
// ---------------------------------------------------------------------------

struct ParamConfig {
    sq_norm: f64,
    kind: ProjectorKind,
    true_frob_sq: Option<f64>,
}

/// Configurable mock M4 optimizer that records `apply_update` and `zero_accum` calls.
struct MockOptimizer {
    params: HashMap<(u32, String), ParamConfig>,
    apply_calls: Mutex<Vec<(u32, String, f32)>>,
    zero_calls: Mutex<Vec<(u32, String)>>,
}

impl MockOptimizer {
    fn new() -> Self {
        Self {
            params: HashMap::new(),
            apply_calls: Mutex::new(Vec::new()),
            zero_calls: Mutex::new(Vec::new()),
        }
    }

    fn add(
        &mut self,
        layer_idx: u32,
        param_name: &str,
        sq_norm: f64,
        kind: ProjectorKind,
        true_frob_sq: Option<f64>,
    ) {
        self.params.insert(
            (layer_idx, param_name.to_string()),
            ParamConfig { sq_norm, kind, true_frob_sq },
        );
    }

    fn apply_calls_sorted(&self) -> Vec<(u32, String, f32)> {
        let mut calls = self.apply_calls.lock().expect("lock").clone();
        calls.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        calls
    }

    fn zero_count(&self) -> usize {
        self.zero_calls.lock().expect("lock").len()
    }
}

impl OptimizerBackend for MockOptimizer {
    fn project_and_accumulate(&self, _grad: &[f32], _layer_idx: u32, _param_name: &str) {}

    fn lowrank_grad_sqnorm(&self, layer_idx: u32, param_name: &str) -> f64 {
        self.params
            .get(&(layer_idx, param_name.to_string()))
            .map(|p| p.sq_norm)
            .unwrap_or(0.0)
    }

    fn apply_update(&self, layer_idx: u32, param_name: &str, clip_scale: f32) {
        self.apply_calls
            .lock()
            .expect("lock")
            .push((layer_idx, param_name.to_string(), clip_scale));
    }

    fn zero_accum(&self, layer_idx: u32, param_name: &str) {
        self.zero_calls
            .lock()
            .expect("lock")
            .push((layer_idx, param_name.to_string()));
    }

    fn notify_step(&self, _step: u64) {}

    fn projector_kind(&self, layer_idx: u32, param_name: &str) -> ProjectorKind {
        self.params
            .get(&(layer_idx, param_name.to_string()))
            .map(|p| p.kind)
            .unwrap_or(ProjectorKind::None)
    }

    fn true_frobenius_sqnorm(&self, layer_idx: u32, param_name: &str) -> Option<f64> {
        self.params
            .get(&(layer_idx, param_name.to_string()))
            .and_then(|p| p.true_frob_sq)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn plan_with_max_norm(max_norm: f32) -> TrainingPlan {
    TrainingPlan { max_grad_norm: max_norm, ..TrainingPlan::default() }
}

/// Reproduce the clip coefficient formula from `global_clip` for reference comparisons.
fn ref_clip_coeff(total_gsq: f64, max_norm: f64) -> f32 {
    ((max_norm / (total_gsq.sqrt() + 1e-6)) as f32).min(1.0_f32)
}

/// Frobenius norm of a flat f64 vector.
fn frob(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Matrix-vector product: P @ v where P is row-major [rows × cols], v is [cols].
fn matvec(p_rows: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    p_rows
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Orthonormal arm — clip triggers
// ---------------------------------------------------------------------------

/// T-CLIP-1a: Orthonormal projector, gradient norm > max_norm.
///
/// 4 params (2 layers × 2 each), sq_norm = 1.0 each.
/// total_sq = 4.0, gnorm = 2.0, max_norm = 1.0 → clip_coeff = 1/(2+ε) ≈ 0.5.
///
/// Asserts: exact gnorm, clip_coeff matches reference within 1e-5, `clipped = true`,
/// all `apply_update` calls carry the same scale, all `zero_accum` calls fired.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_clip_orthonormal_triggers() {
    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = [
        (0u32, "wq"), (0, "wk"), (1, "wq"), (1, "wk"),
    ]
    .iter()
    .map(|(li, pn)| (*li, pn.to_string()))
    .collect();

    for (li, pn) in &layers {
        mock.add(*li, pn, 1.0, ProjectorKind::Orthonormal, None);
    }

    let plan = plan_with_max_norm(1.0);
    let result: ClipResult = deferred_apply_with_clip(&mock, &plan, &layers)
        .expect("deferred_apply_with_clip");

    let total_sq  = 4.0_f64;
    let exp_gnorm = total_sq.sqrt();          // 2.0
    let exp_clip  = ref_clip_coeff(total_sq, 1.0);

    println!(
        "\nT-CLIP-1a: gnorm={:.6}  clip={:.6}  clipped={}",
        result.global_grad_norm, result.clip_coeff, result.clipped
    );

    assert!(
        (result.global_grad_norm - exp_gnorm).abs() < 1e-9,
        "gnorm: got {:.9} expected {:.9}", result.global_grad_norm, exp_gnorm
    );
    assert!(
        (result.clip_coeff - exp_clip).abs() < 1e-5,
        "clip_coeff: got {:.7} expected {:.7}", result.clip_coeff, exp_clip
    );
    assert!(result.clipped,              "expected clipping to trigger");
    assert!(!result.used_grouped_fallback);
    assert_eq!(result.frobenius_exact_layers, 0);

    let apply = mock.apply_calls_sorted();
    assert_eq!(apply.len(), 4, "expected 4 apply_update calls");
    for (_, pn, scale) in &apply {
        assert!(
            (scale - exp_clip).abs() < 1e-5,
            "param={pn}: scale {scale:.7} ≠ {exp_clip:.7}"
        );
    }
    assert_eq!(mock.zero_count(), 4, "expected 4 zero_accum calls");

    println!("T-CLIP-1a PASSED — orthonormal clip triggered, scale ≈ {exp_clip:.5}");
}

// ---------------------------------------------------------------------------
// 2. Orthonormal arm — no clip
// ---------------------------------------------------------------------------

/// T-CLIP-1b: Orthonormal projector, gradient norm < max_norm.
///
/// 2 params, sq_norm = 0.04 each → gnorm ≈ 0.283, max_norm = 1.0.
/// min(1, 1/0.283) > 1 → clip_coeff = 1.0 (no clipping).
/// `apply_update` must still be called with scale 1.0 (identity step).
#[test]
#[cfg(feature = "mock-cuda")]
fn test_clip_orthonormal_no_clip() {
    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = vec![
        (0u32, "wq".to_string()), (1u32, "wq".to_string()),
    ];
    for (li, pn) in &layers {
        mock.add(*li, pn, 0.04, ProjectorKind::Orthonormal, None);
    }

    let plan = plan_with_max_norm(1.0);
    let result = deferred_apply_with_clip(&mock, &plan, &layers).expect("ok");

    println!(
        "\nT-CLIP-1b: gnorm={:.6}  clip={:.6}  clipped={}",
        result.global_grad_norm, result.clip_coeff, result.clipped
    );

    assert!(!result.clipped, "clip must not trigger when gnorm < max_norm");
    assert!(
        (result.clip_coeff - 1.0_f32).abs() < 1e-6,
        "clip_coeff must be 1.0 when unclipped, got {}", result.clip_coeff
    );
    assert!(!result.used_grouped_fallback);

    let apply = mock.apply_calls_sorted();
    assert_eq!(apply.len(), 2);
    for (_, pn, scale) in &apply {
        assert!((scale - 1.0_f32).abs() < 1e-5, "param={pn}: scale={scale:.7} (expected 1.0)");
    }
    assert_eq!(mock.zero_count(), 2);

    println!("T-CLIP-1b PASSED — no clipping, scale = 1.0");
}

// ---------------------------------------------------------------------------
// 3. Frobenius norm preservation property (mathematical sanity check)
// ---------------------------------------------------------------------------

/// T-CLIP-FROB: For P with orthonormal columns, `‖P·a‖_F = ‖a‖_F` exactly.
///
/// This is the mathematical basis for why A6 achieves exact global-norm clipping
/// without forming the full gradient (GaLore arXiv 2403.03507 §3.2).
/// A6 stores `P^T G` in the low-rank accumulator; the squared norm it reads from
/// `lowrank_grad_sqnorm` equals `‖P^T G‖_F^2`. The back-projected gradient that
/// is actually applied to weights is `P · (P^T G)`, whose norm equals `‖P^T G‖_F`
/// exactly because P is an isometry on its column span (P^T P = I_r).
///
/// This test does NOT call M5's A6 code; it verifies the underlying mathematical
/// invariant that A6's "exact" claim depends on.
#[test]
fn test_frobenius_norm_preserved_orthonormal() {
    // 2×2 orthogonal rotation matrix (Pythagorean triple 3-4-5 → cos=3/5, sin=4/5).
    // P = [[0.6, -0.8], [0.8, 0.6]].  P^T P = I_2. ✓
    let p2x2: Vec<Vec<f64>> = vec![
        vec![ 0.6, -0.8],
        vec![ 0.8,  0.6],
    ];
    let a = vec![3.0_f64, 4.0_f64];   // ‖a‖ = 5.0 exactly
    let norm_a  = frob(&a);
    let norm_pa = frob(&matvec(&p2x2, &a));

    println!(
        "\nT-CLIP-FROB (2×2): ‖a‖={:.12}  ‖Pa‖={:.12}  |diff|={:.2e}",
        norm_a, norm_pa, (norm_pa - norm_a).abs()
    );
    assert!(
        (norm_pa - norm_a).abs() < 1e-10,
        "2×2 orthogonal: ‖Pa‖={norm_pa:.12} ≠ ‖a‖={norm_a:.12}"
    );
    assert!((norm_a - 5.0).abs() < 1e-12, "‖a‖ must be exactly 5.0");

    // 4×2 semi-orthogonal (tall-thin, as in GaLore): P^T P = I_2, P P^T ≠ I_4.
    // Columns: [1/√2, 1/√2, 0, 0] and [0, 0, 1/√2, 1/√2].
    let s = 2.0_f64.sqrt().recip();
    let p4x2: Vec<Vec<f64>> = vec![
        vec![s, 0.0],
        vec![s, 0.0],
        vec![0.0, s],
        vec![0.0, s],
    ];
    let norm_pa4 = frob(&matvec(&p4x2, &a));

    println!(
        "T-CLIP-FROB (4×2): ‖a‖={:.12}  ‖Pa‖={:.12}  |diff|={:.2e}",
        norm_a, norm_pa4, (norm_pa4 - norm_a).abs()
    );
    assert!(
        (norm_pa4 - norm_a).abs() < 1e-10,
        "4×2 semi-orthogonal: ‖Pa‖={norm_pa4:.12} ≠ ‖a‖={norm_a:.12}"
    );

    println!("T-CLIP-FROB PASSED — ‖P·a‖_F = ‖a‖_F for both 2×2 and 4×2 orthonormal P");
}

// ---------------------------------------------------------------------------
// 4. Random projector — JL-approximate path (N3′)
// ---------------------------------------------------------------------------

/// T-CLIP-JL-APPROX: Random projector, no Frobenius accumulator.
///
/// The mock reports a sq_norm that deliberately differs from the true gradient sq_norm
/// (simulating the output of a random projection). Verifies:
/// - M5 uses the JL-approximate norm from `lowrank_grad_sqnorm`.
/// - The resulting clip_coeff differs from the exact-reference clip_coeff.
/// - The relative sq-norm error is within a generous JL bound (ε < 0.90 for the
///   deterministic perturbation chosen here).
/// - `frobenius_exact_layers = 0`, `used_grouped_fallback = false`.
///
/// True sq_norm = 100.0/param (4 params) → true_gnorm = 20.0, exact_clip ≈ 0.5000.
/// JL-approx sq_norm = 64.0/param      → approx_gnorm = 16.0, approx_clip ≈ 0.6250.
/// JL relative error: |256 − 400| / 400 = 0.36 < ε=0.90. ✓
#[test]
#[cfg(feature = "mock-cuda")]
fn test_random_projector_jl_approximate() {
    let true_sq_per   = 100.0_f64;
    let approx_sq_per =  64.0_f64;

    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = vec![
        (0u32, "wq".to_string()), (0, "wk".to_string()),
        (1u32, "wq".to_string()), (1, "wk".to_string()),
    ];
    for (li, pn) in &layers {
        mock.add(*li, pn, approx_sq_per, ProjectorKind::Random, None);
    }

    let plan = plan_with_max_norm(10.0);
    let result = deferred_apply_with_clip(&mock, &plan, &layers).expect("ok");

    let n          = layers.len() as f64;
    let approx_gsq = approx_sq_per * n;  // 256.0
    let true_gsq   = true_sq_per   * n;  // 400.0
    let exp_approx = ref_clip_coeff(approx_gsq, 10.0);
    let exp_exact  = ref_clip_coeff(true_gsq,   10.0);
    let jl_rel     = ((approx_gsq - true_gsq) / true_gsq).abs();  // 0.36

    println!(
        "\nT-CLIP-JL-APPROX: approx_gnorm={:.4}  exact_gnorm={:.4}  \
         approx_clip={:.5}  exact_clip={:.5}  jl_rel={:.3}",
        approx_gsq.sqrt(), true_gsq.sqrt(), exp_approx, exp_exact, jl_rel
    );

    // A6 must use the JL-approximate norm.
    assert!(
        (result.clip_coeff - exp_approx).abs() < 1e-5,
        "JL clip_coeff: got {:.7} expected {:.7}", result.clip_coeff, exp_approx
    );

    // The approximation must introduce a measurable difference from the exact reference.
    assert!(
        (result.clip_coeff - exp_exact).abs() > 1e-4,
        "JL approx and exact clips are indistinguishable (both {:.7}); \
         test is not exercising the JL path correctly", result.clip_coeff
    );

    // JL relative error is within a generous bound for this deterministic setup.
    let jl_epsilon = 0.90_f64;
    assert!(
        jl_rel < jl_epsilon,
        "JL relative error {jl_rel:.4} exceeds ε={jl_epsilon}"
    );

    assert_eq!(result.frobenius_exact_layers, 0);
    assert!(!result.used_grouped_fallback);
    assert_eq!(mock.apply_calls_sorted().len(), 4);
    assert_eq!(mock.zero_count(), 4);

    println!(
        "T-CLIP-JL-APPROX PASSED — JL err={jl_rel:.3}, \
         Δclip={:.5} (approx ≠ exact)", (result.clip_coeff - exp_exact).abs()
    );
}

// ---------------------------------------------------------------------------
// 5. Random projector + O(1) Frobenius accumulator — restores exact clip
// ---------------------------------------------------------------------------

/// T-CLIP-JL-EXACT: Random projector with `true_frobenius_sqnorm` set for all params.
///
/// Same JL-perturbed `lowrank_grad_sqnorm` as the previous test, but the mock also
/// supplies the pre-projection true sq_norm via `true_frobenius_sqnorm`.
/// A6 must prefer the Frobenius accumulator and produce a clip_coeff matching the
/// exact reference (within 1e-5).
///
/// Verifies `frobenius_exact_layers == 4` (one per param).
/// Cite: APOLLO arXiv 2412.05270 — motivates the O(1) accumulator as correctness
/// mitigation under random projection.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_random_projector_frobenius_exact() {
    let true_sq_per  = 100.0_f64;
    let approx_sq_per =  64.0_f64;   // what lowrank_grad_sqnorm returns (JL-approx)
    let n = 4_usize;

    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = vec![
        (0u32, "wq".to_string()), (0, "wk".to_string()),
        (1u32, "wq".to_string()), (1, "wk".to_string()),
    ];
    for (li, pn) in &layers {
        mock.add(*li, pn, approx_sq_per, ProjectorKind::Random, Some(true_sq_per));
    }

    let plan = plan_with_max_norm(10.0);
    let result = deferred_apply_with_clip(&mock, &plan, &layers).expect("ok");

    let true_gsq  = true_sq_per * n as f64;  // 400.0
    let exp_gnorm = true_gsq.sqrt();           // 20.0
    let exp_clip  = ref_clip_coeff(true_gsq, 10.0);

    println!(
        "\nT-CLIP-JL-EXACT: gnorm={:.6}  expected={:.6}  clip={:.6}  exact={:.6}  \
         frob_exact_layers={}",
        result.global_grad_norm, exp_gnorm, result.clip_coeff,
        exp_clip, result.frobenius_exact_layers
    );

    assert!(
        (result.global_grad_norm - exp_gnorm).abs() < 1e-9,
        "gnorm: got {:.9} expected {:.9}", result.global_grad_norm, exp_gnorm
    );
    assert!(
        (result.clip_coeff - exp_clip).abs() < 1e-5,
        "clip_coeff: got {:.7} expected {:.7}", result.clip_coeff, exp_clip
    );
    assert_eq!(
        result.frobenius_exact_layers, n as u32,
        "expected frobenius_exact_layers={n}, got {}", result.frobenius_exact_layers
    );
    assert!(!result.used_grouped_fallback);

    println!("T-CLIP-JL-EXACT PASSED — O(1) accumulator restored exact clip");
}

// ---------------------------------------------------------------------------
// 6. Grouped/local-clip fallback — None projector
// ---------------------------------------------------------------------------

/// T-CLIP-GROUPED: `ProjectorKind::None` triggers the LOMO grouped-clip fallback.
///
/// Two layers with different gradient magnitudes:
/// - Layer 0: wq (sq=9.0), wk (sq=16.0) → layer_norm = 5.0 → clip ≈ 0.2
/// - Layer 1: wq (sq=0.04), wk (sq=0.04) → layer_norm ≈ 0.283 → clip = 1.0
///
/// max_norm = 1.0.  Clips are per-layer (LOMO §3.3, arXiv 2306.09782).
/// A global clip would incorrectly apply ≈0.2 to layer 1's small gradient —
/// verifying that the two scales differ demonstrates the fallback is correct.
///
/// Cite: Lv et al. 2023 arXiv 2306.09782 §3.3 — smooth-loss-surface justification
/// for grouped/local clipping when no projection is available.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_grouped_fallback_none_projector() {
    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = vec![
        (0u32, "wq".to_string()), (0, "wk".to_string()),
        (1u32, "wq".to_string()), (1, "wk".to_string()),
    ];
    mock.add(0, "wq",  9.0,  ProjectorKind::None, None);
    mock.add(0, "wk", 16.0,  ProjectorKind::None, None);
    mock.add(1, "wq",  0.04, ProjectorKind::None, None);
    mock.add(1, "wk",  0.04, ProjectorKind::None, None);

    let max_norm = 1.0_f64;
    let plan = plan_with_max_norm(max_norm as f32);
    let result = deferred_apply_with_clip(&mock, &plan, &layers).expect("grouped ok");

    // Reference per-layer clips (same formula as `grouped_clip_fallback`).
    let l0_sq    = 9.0_f64 + 16.0;                                            // 25.0
    let l0_clip  = ((max_norm / (l0_sq.sqrt() + 1e-6)) as f32).min(1.0); // ≈ 0.2
    let l1_sq    = 0.04_f64 + 0.04;
    let l1_clip  = ((max_norm / (l1_sq.sqrt() + 1e-6)) as f32).min(1.0); // = 1.0

    println!(
        "\nT-CLIP-GROUPED: l0_clip={:.6}  l1_clip={:.6}  used_fallback={}",
        l0_clip, l1_clip, result.used_grouped_fallback
    );

    assert!(result.used_grouped_fallback, "expected grouped fallback to engage");
    assert_eq!(result.frobenius_exact_layers, 0);

    let apply = mock.apply_calls_sorted();
    assert_eq!(apply.len(), 4, "expected 4 apply_update calls");

    let l0_calls: Vec<_> = apply.iter().filter(|(li, _, _)| *li == 0).collect();
    for (_, pn, scale) in &l0_calls {
        assert!(
            (scale - l0_clip).abs() < 1e-5,
            "layer=0 {pn}: scale={scale:.7} expected {l0_clip:.7}"
        );
    }
    let l1_calls: Vec<_> = apply.iter().filter(|(li, _, _)| *li == 1).collect();
    for (_, pn, scale) in &l1_calls {
        assert!(
            (scale - l1_clip).abs() < 1e-5,
            "layer=1 {pn}: scale={scale:.7} expected {l1_clip:.7}"
        );
    }

    // Demonstrate that grouped ≠ global for layer 1 (the key difference).
    let global_clip = ref_clip_coeff(l0_sq + l1_sq, max_norm);
    assert!(
        (l1_clip - global_clip).abs() > 1e-3,
        "Layer-1 clip ({l1_clip:.6}) should differ from global clip ({global_clip:.6})"
    );

    assert_eq!(mock.zero_count(), 4, "expected 4 zero_accum calls");

    println!(
        "T-CLIP-GROUPED PASSED — l0={l0_clip:.4}  l1={l1_clip:.4}  \
         (global would have been {global_clip:.4} for l1)"
    );
}

// ---------------------------------------------------------------------------
// 7. Fallback does NOT engage for Orthonormal projectors (gate test)
// ---------------------------------------------------------------------------

/// T-CLIP-GATE: `used_grouped_fallback` must be `false` for Orthonormal projectors.
///
/// Same gradient structure as T-CLIP-GROUPED but with `ProjectorKind::Orthonormal`.
/// A6 must take the global-norm path and NOT invoke per-layer clips.
/// All params must receive the same global clip scale.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_grouped_fallback_not_when_orthonormal() {
    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = vec![
        (0u32, "wq".to_string()), (0, "wk".to_string()),
        (1u32, "wq".to_string()), (1, "wk".to_string()),
    ];
    mock.add(0, "wq",  9.0,  ProjectorKind::Orthonormal, None);
    mock.add(0, "wk", 16.0,  ProjectorKind::Orthonormal, None);
    mock.add(1, "wq",  0.04, ProjectorKind::Orthonormal, None);
    mock.add(1, "wk",  0.04, ProjectorKind::Orthonormal, None);

    let plan = plan_with_max_norm(1.0);
    let result = deferred_apply_with_clip(&mock, &plan, &layers).expect("ok");

    println!(
        "\nT-CLIP-GATE: used_fallback={}  clip={:.6}  clipped={}",
        result.used_grouped_fallback, result.clip_coeff, result.clipped
    );

    assert!(
        !result.used_grouped_fallback,
        "Orthonormal projectors must NOT use grouped fallback"
    );

    // All params must receive the same global clip scale.
    let global_gsq  = 9.0 + 16.0 + 0.04 + 0.04;
    let global_clip = ref_clip_coeff(global_gsq, 1.0);
    let apply = mock.apply_calls_sorted();
    assert_eq!(apply.len(), 4);
    for (_, pn, scale) in &apply {
        assert!(
            (scale - global_clip).abs() < 1e-5,
            "param={pn}: scale={scale:.7} expected global {global_clip:.7}"
        );
    }

    println!("T-CLIP-GATE PASSED — no grouped fallback, global clip = {global_clip:.5}");
}

// ---------------------------------------------------------------------------
// 8. Empty trainable_layers → Config error
// ---------------------------------------------------------------------------

/// T-CLIP-EMPTY: An empty `trainable_layers` slice must return `DoublePassError::Config`.
#[test]
fn test_empty_trainable_layers_returns_error() {
    struct NullOpt;
    impl OptimizerBackend for NullOpt {
        fn project_and_accumulate(&self, _: &[f32], _: u32, _: &str) {}
        fn lowrank_grad_sqnorm(&self, _: u32, _: &str) -> f64 { 0.0 }
        fn apply_update(&self, _: u32, _: &str, _: f32) {}
        fn zero_accum(&self, _: u32, _: &str) {}
        fn notify_step(&self, _: u64) {}
    }

    let plan = plan_with_max_norm(1.0);
    let err = deferred_apply_with_clip(&NullOpt, &plan, &[]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("empty") || msg.contains("trainable"),
        "unexpected error message: {msg}"
    );
    println!("\nT-CLIP-EMPTY PASSED — Config error: {msg}");
}

// ---------------------------------------------------------------------------
// 9. Mixed arms: one Random with accumulator, one without
// ---------------------------------------------------------------------------

/// T-CLIP-MIXED: two Random params — one clip-critical (has Frobenius accumulator),
/// one not (JL path). A6 must use the exact sq_norm for the critical param and the
/// JL-approx sq_norm for the other.
///
/// Verifies `frobenius_exact_layers == 1` and the resulting gnorm/clip_coeff.
#[test]
#[cfg(feature = "mock-cuda")]
fn test_random_mixed_exact_and_jl() {
    // wq: clip-critical — has the O(1) Frobenius accumulator.
    // wk: not critical  — JL-approximate norm used.
    let true_sq_wq   = 100.0_f64;
    let approx_sq_wq =  81.0_f64;  // JL-approx for wq (ignored — exact path overrides)
    let approx_sq_wk =  36.0_f64;  // JL-approx for wk (used)

    let mut mock = MockOptimizer::new();
    let layers: Vec<(u32, String)> = vec![
        (0u32, "wq".to_string()),
        (0u32, "wk".to_string()),
    ];
    mock.add(0, "wq", approx_sq_wq, ProjectorKind::Random, Some(true_sq_wq));
    mock.add(0, "wk", approx_sq_wk, ProjectorKind::Random, None);

    let plan = plan_with_max_norm(1.0);
    let result = deferred_apply_with_clip(&mock, &plan, &layers).expect("ok");

    // Expected: true_sq_wq (exact) + approx_sq_wk (JL).
    let exp_gsq   = true_sq_wq + approx_sq_wk;  // 136.0
    let exp_gnorm = exp_gsq.sqrt();
    let exp_clip  = ref_clip_coeff(exp_gsq, 1.0);

    println!(
        "\nT-CLIP-MIXED: gnorm={:.6}  expected={:.6}  clip={:.6}  \
         frob_exact_layers={}",
        result.global_grad_norm, exp_gnorm, result.clip_coeff,
        result.frobenius_exact_layers
    );

    assert!(
        (result.global_grad_norm - exp_gnorm).abs() < 1e-9,
        "gnorm: got {:.9} expected {:.9}", result.global_grad_norm, exp_gnorm
    );
    assert!(
        (result.clip_coeff - exp_clip).abs() < 1e-5,
        "clip_coeff: got {:.7} expected {:.7}", result.clip_coeff, exp_clip
    );
    assert_eq!(result.frobenius_exact_layers, 1, "expected 1 frob-exact layer");
    assert!(!result.used_grouped_fallback);

    println!("T-CLIP-MIXED PASSED — mixed exact/JL arms, frobenius_exact_layers=1");
}
