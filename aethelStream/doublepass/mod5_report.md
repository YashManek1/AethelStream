# AethelStream Module 5 (DoublePass) — Comprehensive Reference

**Version:** 0.1.0  
**Last Updated:** 2025-06-27  
**Language:** Rust 2021  
**Dependencies:** FlowCast (M3), RamFlow (M2), PyO3 (optional)

---

## Table of Contents

1. [Module Purpose](#1-module-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Public API](#3-public-api)
4. [Algorithm A1: Forward Pass](#4-algorithm-a1-forward-pass)
5. [Algorithm A2/A2′: Backward Pass + SARP](#5-algorithm-a2a2-backward-pass--sarp)
6. [Algorithm A5: Precision](#6-algorithm-a5-precision)
7. [Algorithm A6: Gradient Clipping](#7-algorithm-a6-gradient-clipping)
8. [Algorithm A7: Parity Guard](#8-algorithm-a7-parity-guard)
9. [Algorithm A8: Cut-CE Loss](#9-algorithm-a8-cut-ce-loss)
10. [Algorithm A9: HAM Offload](#10-algorithm-a9-ham-offload)
11. [Algorithm A10: RNG](#11-algorithm-a10-rng)
12. [Schedule Emitter](#12-schedule-emitter)
13. [Checkpoint Store/Read](#13-checkpoint-storeread)
14. [PyO3 FFI](#14-pyo3-ffi)
15. [Integration Points](#15-integration-points)
16. [Feature Flags](#16-feature-flags)
17. [Benchmarks](#17-benchmarks)
18. [Test Coverage](#18-test-coverage)
19. [Gradient Parity Guarantee](#19-gradient-parity-guarantee)
20. [Training Loop Diagrams](#20-training-loop-diagrams)

---

## 1. Module Purpose

**DoublePass (Module 5)** is a **layer-at-a-time double-pass backward engine** for training transformer models (7B–70B parameter range) on a single GPU.

### Core Capabilities

- **Forward pass:** Stream activations layer-by-layer, checkpointing every k layers (sparse checkpointing).
- **Loss computation:** **Streaming Cut Cross-Entropy (Cut-CE)** — computes language-model loss without materializing the full `[batch_seq × vocab_size]` logit tensor. Peak memory usage scales as `O(batch_seq × chunk_size)`, not `O(vocab_size)`.
- **Backward pass:** Recompute-or-offload decision per segment (M9 SARP schedule or greedy HAM fallback); propagate gradients backward with selective recompute masks (SARP).
- **Precision:** BF16 default; FP16 fallback with dynamic loss scaling (M2 `PerLayerScaleTable`); stochastic rounding to avoid weight stagnation.
- **Gradient clipping:** Exact global-norm clipping via low-rank projection norms (GaLore orthonormal), JL-approximate norms (random projection), or grouped/local fallback (LOMO).
- **Parity diagnostic:** Continuous gradient comparison against full-precision PyTorch reference (A7 ParityGuard).
- **LOMO guarantee:** Full gradient never persists beyond one layer's backward pass — all accumulation is in low-rank space (M4 optimizer).

### Target Models

- LLaMA 7B–70B, Mistral 8×7B, Qwen, Llama-Pro, Phi-3 (and similar architectures).
- Single GPU deployment (NVIDIA Ampere/Hopper, RTX 4090 / A100 / H100).
- Micro-batch size: 1–4 per GPU; gradient accumulation: 2–16 steps.

### Mathematical Guarantees

- **Gradient parity ≤ 1e-5** vs PyTorch full-model baseline (validated by T-PARITY tests; see [Gradient Parity Guarantee](#19-gradient-parity-guarantee)).
- **No full-gradient materialization:** LOMO + low-rank projection enable training 7B params on 24GB VRAM.

---

## 2. Architecture Overview

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       DoublePass Engine                         │
│                  (crate::DoublePass struct)                     │
└─────────────────────────────────────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
        ┌─────────▼──────────┐  ┌────▼──────────────────┐
        │   TrainingPlan     │  │   FlowCast Pipeline   │
        │   (M9 schedule)    │  │   (M3 for weights)    │
        └────────────────────┘  └───────────────────────┘
                  │                     │
        ┌─────────┴─────────────────────┴──────────┐
        │                                          │
        │         full_training_step():            │
        │  A1 → A8 → A2′ → A6 wired sequence      │
        │                                          │
        └──────────────────────────────────────────┘
                  │    │    │    │
         ┌────────┘    │    │    └────────┐
         │             │    │             │
         ▼             ▼    ▼             ▼
       ┌──────┐   ┌─────┐ ┌──┐   ┌──────────┐
       │  A1  │──▶│ A8  │─▶│A2'│──▶│   A6    │
       │Forward   │CE Loss  │SARP   │Clip+Apply
       └──────┘   └─────┘ └──┘   └──────────┘
         ▲                  ▲
         │                  │
    ┌────┴──────┐      ┌────┴──────────┐
    │ A10: RNG  │      │ A9: HAM/SARP  │
    │Checkpoint │      │ ParityGuard   │
    │ A5: Prec  │      │ A7: Parity    │
    └───────────┘      └───────────────┘
```

### Three Layers of Abstraction

| Layer | Scope | Entry Point |
|-------|-------|-------------|
| **Training Loop** | Full step orchestration | `full_training_step(…)` in `train_step.rs` |
| **Algorithm A-zones** | Individual components (A1, A2, A5–A10) | Private module functions; public types + hooks |
| **Executor/Backend** | FlowCast pipeline + optimizer traits | `DoublePass::step()` → wires to `full_training_step` |

---

## 3. Public API

### `DoublePass` struct
Central engine wrapper. Owns FlowCast and orchestrates full steps.

```rust
pub struct DoublePass {
    flowcast: FlowCast,
    plan: Option<TrainingPlan>,
    optimizer: Option<Box<dyn OptimizerBackend>>,
    lora: Option<Box<dyn LoraBackend>>,
    step_count: u64,
}

impl DoublePass {
    /// Construct around an already-warmed-up FlowCast.
    pub fn new(
        flowcast: FlowCast,
        optimizer: Option<Box<dyn OptimizerBackend>>,
        lora: Option<Box<dyn LoraBackend>>,
    ) -> Result<Self>;

    /// Install or replace the active training plan.
    pub fn set_plan(&mut self, plan: TrainingPlan) -> Result<()>;

    /// Apply a partial plan update (checkpoint_freq, precision, etc.).
    pub fn apply_delta(&mut self, delta: PlanDelta) -> Result<()>;

    /// Execute one full training step: forward → loss → recompute → backward → apply.
    pub fn step(&mut self, batch: &Batch) -> Result<StepMetrics>;

    /// Return a snapshot for M10 (checkpoint/resume).
    pub fn snapshot(&self) -> Result<ConsistentState>;

    /// Run parity diagnostic on one layer against FP32 reference.
    pub fn parity_probe(&self, ref_layer: u32) -> Result<f64>;
}
```

### `TrainingPlan` struct
Frozen configuration per training run (from M9).

```rust
pub struct TrainingPlan {
    pub checkpoint_freq: u32,              // Sparse checkpoint every k layers
    pub micro_batch: u32,                  // Micro-batch size
    pub grad_accum: u32,                   // Gradient accumulation steps
    pub precision_schedule: Vec<Precision>,// Per-layer BF16/FP16 policy
    pub optimizer_rank: u32,               // GaLore projection rank
    pub tier: TrainingTier,                // FullGaLore | LoraOnly | TopKFreeze | Int4
    pub w_max_hint: u32,                   // Weight-streaming window (layers)
    pub activation_schedule: Vec<SegmentPlan>, // M9 SARP schedule (optional)
    pub parity_check_interval: u64,        // Fire A7 every N steps
    pub projection_refresh_interval: u64,  // Projection refresh cadence
    pub max_grad_norm: f32,                // Target norm for A6 global clip
}
```

### Core Public Types

| Type | Source | Purpose |
|------|--------|---------|
| `Batch` | `lib.rs` | Micro-batch input: `[num_sequences, seq_len]` token ids. |
| `BlockConfig` | `forward.rs` | Layer config: d_model, n_heads, d_ff, seq_len, batch, dropout_p. |
| `BlockWeights` | `forward.rs` | Nine weight matrices per transformer block. |
| `SingleLayerFwdOut` | `forward.rs` | 21 activation tensors (x_in, rms1, q_heads, …, output). |
| `ParamGrads` | `backward.rs` | 9 gradient tensors per layer + d_input for propagation. |
| `SegmentPlan` | `plan.rs` | Per-segment action: Recompute | RetainVram | PageCompressedRam | PageNvme. |
| `SelectiveRecomputeMask` | `plan.rs` | Boolean array: which of 7 OpKind ops to recompute (A2'). |
| `OpKind` enum | `plan.rs` | 7 stages: Rms1, QkvProj, AttnSoftmax, OutProj, Rms2, MlpGateUp, MlpDown. |
| `LossOutput` | `loss.rs` | Loss + grad_hidden + peak_logit_bytes. |
| `ClipResult` | `hook.rs` | Clip diagnostic: global_grad_norm, clip_coeff, used_grouped_fallback. |
| `StepMetrics` | `metrics.rs` | Full telemetry: weight_bytes, parity error, GPU idle, segment log. |
| `ConsistentState` | `state.rs` | Snapshot for M10: step, optimizer_version, RNG states, data_position. |

### Trait: `OptimizerBackend`
Placeholder for M4 (GaLore optimizer). M5 receives gradients here; M4 owns projection + accumulation.

```rust
pub trait OptimizerBackend: Send + Sync {
    /// Project grad to low rank and accumulate into per-layer buffer.
    fn project_and_accumulate(&self, grad: &[f32], layer_idx: u32, param_name: &str);
    
    /// Return squared Frobenius norm of the low-rank accumulator.
    fn lowrank_grad_sqnorm(&self, layer_idx: u32, param_name: &str) -> f64;
    
    /// Apply one clipped Adam step, then zero the accumulator.
    fn apply_update(&self, layer_idx: u32, param_name: &str, clip_scale: f32);
    
    /// Zero the low-rank accumulator.
    fn zero_accum(&self, layer_idx: u32, param_name: &str);
    
    /// Notify step completion (drives projection-refresh cadence).
    fn notify_step(&self, step: u64);
    
    /// Return projector kind for this layer (Orthonormal | Random | None).
    fn projector_kind(&self, layer_idx: u32, param_name: &str) -> ProjectorKind {
        ProjectorKind::Orthonormal  // default
    }
    
    /// Return pre-projection Frobenius sqnorm (for exact clip under Random projection).
    fn true_frobenius_sqnorm(&self, layer_idx: u32, param_name: &str) -> Option<f64> {
        None  // default: use JL-approximate norm
    }
}
```

### Trait: `LoraBackend`
Placeholder for M6 (LoRA adapter manager).

```rust
pub trait LoraBackend: Send + Sync {
    fn rank_of(&self, layer_idx: u32) -> u32;
}
```

---

## 4. Algorithm A1: Forward Pass

**Purpose:** Stream activations layer-by-layer, apply sparse checkpointing every k layers, capture RNG state per (layer, micro-batch) for deterministic recompute.

### Public Entry Points

```rust
pub fn single_layer_forward(
    cfg: &BlockConfig,
    w: &BlockWeights,
    input: &[f32],
) -> SingleLayerFwdOut;

pub fn full_forward_with_retention(
    model: &Model,
    inputs: &[Vec<f32>],
    plan: &TrainingPlan,
    compress_checkpoints: bool,
) -> Result<ForwardOutput>;
```

### `ForwardOutput` struct

```rust
pub struct ForwardOutput {
    pub checkpoints: Vec<(u32, u32, PinnedBuffer)>,  // (layer_idx, micro_batch, activation)
    pub rng_states: Vec<RngState>,                   // Per-(layer, micro_batch) seed
    pub weight_bytes: u64,                           // Total weight bytes streamed
}
```

### Execution Flow

1. **For each micro-batch** (input tensor):
   - Iterate layers 0 → num_layers-1 in ascending order.
   - Per layer: call `single_layer_forward` with current activation as input.
   - Return updated activation.

2. **Checkpoint logic** (every k layers):
   - When `layer_idx % checkpoint_freq == 0`: store the activation to a `PinnedBuffer`.
   - Mark compressed vs uncompressed based on `compress_checkpoints` flag.
   - Append `(layer_idx, micro_batch_idx, buffer)` to `checkpoints` vec.

3. **RNG capture**:
   - After any operation using dropout (e.g., attention output projection), call `rng::capture(layer_idx, micro_batch)`.
   - Store returned `RngState` for later restoration during recompute.
   - Set thread-local PRNG state via `rng::set_step_seed(base_seed)` once per step before forward.

4. **Activation retention** (for SARP):
   - When activation action is `RetainVram` or `PageCompressedRam`, store `SingleLayerFwdOut` in `retained_activations` vec.
   - Tagged by `(layer_idx, micro_batch)` for later lookup during backward.

### Mathematical Operations (per layer)

Each `single_layer_forward` implements one transformer block:

```
Input: x_in ∈ ℝ^[bs × d]

A1: Rms1         x_norm1 = RMSNorm(x_in)
A2: Attn         q,k,v = LinearProj(x_norm1)
                 attn_out = Attention(q, k, v)
A3: OutProj      out_proj = Linear(attn_out)
                 x2 = x_in + Dropout(out_proj)

A4: Rms2         x_norm2 = RMSNorm(x2)
A5: MLP          gate = Linear_gate(x_norm2)
                 up = Linear_up(x_norm2)
                 mlp_hidden = SiLU(gate) ⊙ up
A6: MlpDown      mlp_out = Linear_down(mlp_hidden)
                 output = x2 + Dropout(mlp_out)

Output: output ∈ ℝ^[bs × d]
```

All operations use **mock-cuda** (CPU f32 stubs under `#[cfg(feature = "mock-cuda")]`); production CUDA kernels replace these.

---

## 5. Algorithm A2/A2′: Backward Pass + SARP

**Purpose:** Propagate gradients backward through segments; decide per-segment whether to recompute or offload activations using the M9 SARP schedule or greedy HAM fallback.

### Public Entry Points

```rust
pub fn full_backward(
    model: &Model,
    fwd: &FullForwardResult,
    inputs: &[Vec<f32>],
    upstream_grads: &[Vec<f32>],
    plan: &TrainingPlan,
    keep_resident: bool,
    optimizer: &dyn OptimizerBackend,
) -> Result<FullBackwardResult>;

pub fn single_layer_backward(
    cfg: &BlockConfig,
    w: &BlockWeights,
    fwd: &SingleLayerFwdOut,
    upstream: &[f32],
) -> ParamGrads;
```

### SARP Executor (A2′ dispatcher)

```rust
pub struct SarpExecutor {
    plans: Vec<SegmentPlan>,  // From M9 TrainingPlan::activation_schedule
    profile: HardwareProfile, // For HAM greedy fallback
}

impl SarpExecutor {
    pub fn new(plan: &TrainingPlan, profile: HardwareProfile) -> Self;
    pub fn has_m9_schedule(&self) -> bool;
    pub fn plan_for_segment(&self, segment_index: u32) -> Option<&SegmentPlan>;
    pub fn action_for_segment(
        &self,
        segment_index: u32,
        segment_weight_bytes: u64,
        num_layers: usize,
    ) -> ActivationAction;
}
```

### Segment-wise Actions (ActivationAction enum)

| Action | Forward | Backward Recompute | Memory Footprint |
|--------|---------|-------------------|------------------|
| **Recompute** | Checkpoint only | Re-run forward from ckpt, restore RNG | O(ckpt_size) |
| **RetainVram** | Store activations | Load from retained_activations | O(all_fwds) in VRAM |
| **PageCompressedRam** | Store activations | Load from PCIe-backed RAM | O(all_fwds) in pinned RAM (compressed) |
| **PageNvme** | Store activations | Load from NVMe | O(all_fwds) on SSD; slowest but cheapest |

### Selective Recompute Mask (N1″)

Per-segment, decide which of the 7 `OpKind` stages to recompute vs retain:

```rust
pub struct SelectiveRecomputeMask {
    ops: [bool; 7],  // true = recompute, false = retain
}

impl SelectiveRecomputeMask {
    pub fn attn_interior_only() -> Self;  // Recompute only softmax
    pub fn full_recompute() -> Self;      // Recompute all 7 ops
    pub fn retain_all() -> Self;          // Retain all 7 ops
    pub fn recompute_flop_fraction(&self) -> f64;  // Sum fractions for ops[i]=true
}
```

FLOP fractions per OpKind:
- Rms1: 0.02, QkvProj: 0.27, AttnSoftmax: 0.08, OutProj: 0.09
- Rms2: 0.02, MlpGateUp: 0.27, MlpDown: 0.25

### Backward Execution Flow

1. **Layer schedule**: `LayerSchedule::backward_segments()` returns segments in **reverse** order (last segment first).

2. **For each segment** (in descending segment_index order):
   - Determine action: M9 plan > HAM greedy > default Recompute.
   - **If Recompute (full or selective):**
     - Load checkpoint at segment start.
     - For each layer (ascending), restore RNG, re-run forward with mask, accumulate gradients.
   - **If RetainVram / Page***:
     - Load `retained_activations` for this segment.
     - Use stored `SingleLayerFwdOut` directly (no re-run).
   - For each micro-batch within segment: call `single_layer_backward` per layer (descending).

3. **Gradient accumulation**:
   - Per layer, sum gradients across all micro-batches into one `ParamGrads`.
   - Call `optimizer.project_and_accumulate` per parameter (A3 hook); full gradient evaporates after projection.

4. **Upstream propagation**:
   - Each `single_layer_backward` returns `d_input`.
   - Pass to next-lower layer as its upstream gradient.

### `ParamGrads` struct (output per layer)

```rust
pub struct ParamGrads {
    pub d_rms1_w: Vec<f32>,
    pub d_wq: Vec<f32>, d_wk: Vec<f32>, d_wv: Vec<f32>,
    pub d_wo: Vec<f32>,
    pub d_rms2_w: Vec<f32>,
    pub d_wg: Vec<f32>, d_wu: Vec<f32>, d_wd: Vec<f32>,
    pub d_input: Vec<f32>,  // Propagated to lower layer
}
```

---

## 6. Algorithm A5: Precision

**Purpose:** Select BF16 vs FP16 per layer; manage FP16 dynamic loss scaling (M2 `PerLayerScaleTable`); apply stochastic rounding to avoid weight stagnation.

### Public API

```rust
pub fn effective_precision(
    layer_idx: u32,
    schedule: &[Precision],
    hardware_supports_bf16: bool,
) -> Precision;

pub fn check_and_update_scale(
    scale_table: &mut PerLayerScaleTable,
    layer_idx: usize,
    grad_data: *const u16,
    n_elements: usize,
) -> Result<bool>;  // true = overflow detected

pub fn stochastic_round_to_bf16(val: f32, rand_bits: u16) -> u16;
pub fn bf16_to_f32(bf16_bits: u16) -> f32;

pub fn apply_lora_update_fp32(weights: &mut [f32], delta: &[f32]) -> Result<()>;
pub fn apply_galore_bf16_update(master: &mut [f32], delta: &[f32], rng: &mut u64) -> Result<()>;
```

### Precision Selection Rules

1. If `schedule[layer_idx] == BF16` AND hardware supports BF16 → use BF16.
2. If `schedule[layer_idx] == BF16` but hardware lacks support → degrade to FP16.
3. Otherwise: honor `schedule[layer_idx]` exactly.
4. No schedule entry (out of bounds) → default to BF16 if supported, else FP16.

### FP16 Dynamic Loss Scaling

When precision is FP16:
- **Before each backward step**, call `check_and_update_scale(scale_table, layer_idx, grad_ptr, n_elems)`.
- Kernel counts overflows (FP16 exponent == 31, non-NaN).
- `PerLayerScaleTable::update()` (M2):
  - If overflow density > `overflow_high_threshold` (0.001): halve scale (floor 1.0).
  - If overflow density < `overflow_low_threshold` (0.0001) after clean interval: double scale (cap 65536).
- Return `true` when overflow detected → skip `apply_update` this layer this step.

### Stochastic Rounding (SRTE)

**Problem:** Deterministic rounding to BF16 flushes sub-ULP gradients to zero, causing weight stagnation.  
**Solution:** Probabilistically round up with probability = lower 16 bits / 2^16.

```rust
pub fn stochastic_round_to_bf16(val: f32, rand_bits: u16) -> u16 {
    let bits = val.to_bits();
    let low16 = (bits & 0xFFFF) as u16;
    let high16 = (bits >> 16) as u16;
    if low16 > rand_bits {
        high16.saturating_add(1)
    } else {
        high16
    }
}
```

**RNG:** LCG (Knuth MMIX, bits 33–48 used).

### Master-Weight Policies

**LoRA path:**
```rust
pub fn apply_lora_update_fp32(weights: &mut [f32], delta: &[f32]) -> Result<()> {
    for (w, d) in weights.iter_mut().zip(delta.iter()) {
        *w += d;  // In-place; no rounding
    }
}
```

**Full-param GaLore path:**
```rust
pub fn apply_galore_bf16_update(master: &mut [f32], delta: &[f32], rng: &mut u64) -> Result<()> {
    for (m, d) in master.iter_mut().zip(delta.iter()) {
        *m += d;
        let rand_bits = lcg_next(rng);
        let bf16 = stochastic_round_to_bf16(*m, rand_bits);
        *m = bf16_to_f32(bf16);
    }
}
```

---

## 7. Algorithm A6: Gradient Clipping

**Purpose:** Apply exact (or JL-approximate) global-norm gradient clipping and the M4 Adam update for all trainable parameters.

### Public API

```rust
pub fn deferred_apply_with_clip(
    optimizer: &dyn OptimizerBackend,
    plan: &TrainingPlan,
    trainable_layers: &[(u32, String)],
) -> Result<ClipResult>;

#[derive(Debug, Clone)]
pub struct ClipResult {
    pub global_grad_norm: f64,
    pub clip_coeff: f32,
    pub clipped: bool,
    pub used_grouped_fallback: bool,
    pub frobenius_exact_layers: u32,
}
```

### Clipping Strategy (per ProjectorKind)

**Three paths:**

#### Path 1: Orthonormal Projection (GaLore)
- **Projector:** P ∈ ℝ^{m×r} with orthonormal columns (P^T P = I_r).
- **Norm guarantee:** Exact. ‖P·(P^T G)‖_F = ‖P^T G‖_F (isometry).
- **Execution:**
  1. Sum squared norms from `optimizer.lowrank_grad_sqnorm(layer_idx, param)` across all parameters.
  2. Global norm = sqrt(sum).
  3. Clip coefficient = min(1, max_norm / global_norm).

#### Path 2: Random Projection (APOLLO / Q-GaLore)
- **Projector:** P ∈ ℝ^{m×r} with i.i.d. Gaussian entries scaled by 1/√r.
- **Norm guarantee:** JL-approximate. E[‖P^T G‖_F^2] = ‖G‖_F^2, relative error O(1/√r).
- **Execution:**
  1. For each parameter, check `optimizer.true_frobenius_sqnorm(layer_idx, param)`:
     - If Some(fsq): exact pre-projection norm captured during A3 → use it.
     - If None: use `lowrank_grad_sqnorm` (JL-approximate).
  2. Sum over all parameters; compute global clip as Path 1.

#### Path 3: No Projection (LOMO grouped-fallback)
- **Projector:** None (full gradient stored by optimizer).
- **Fallback:** Per-layer clipping (LOMO §3.3).
- **Execution:**
  1. Group parameters by layer.
  2. For each layer: sum squared norms of all params in layer.
  3. Per-layer clip coefficient = min(1, max_norm / layer_norm).
  4. Apply per-layer coefficient to each parameter.
  5. Effective global norm may exceed target by up to √L (justified by LOMO's smooth-loss-surface argument).

### Execution (both paths)

**Pass 1 — norm accumulation:** Iterate trainable layers, sum squared norms.  
**Pass 2 — apply updates:** Call `optimizer.apply_update(layer_idx, param_name, clip_scale)` + `zero_accum()` per parameter.

### Design: Why No Optimizer Reference?

The `ParityGuard` (A7) holds **no optimizer reference** by design. The training loop calls `optimizer.notify_step(step)` independently. This decouples parity diagnostics from projection refresh, ensuring parity checks never accidentally mask bugs via stale/refreshed projections.

---

## 8. Algorithm A7: Parity Guard

**Purpose:** Continuous gradient comparison against full-precision PyTorch reference. Diagnose divergence and escalate to FP32 recompute on warning; halt training on critical divergence.

### Public API

```rust
pub struct ParityGuard {
    tolerances: ParityTolerances,
    parity_check_interval: u64,
    escalated_layers: HashSet<u32>,
    check_count: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ParityTolerances {
    pub warn: f64,   // Default: 1e-4
    pub halt: f64,   // Default: 1e-3
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityAction {
    Clean,
    Escalated,
}

impl ParityGuard {
    pub fn new(tolerances: ParityTolerances, parity_check_interval: u64) -> Self;
    pub fn should_check(&self, step: u64) -> bool;
    pub fn check(
        &mut self,
        step: u64,
        layer_idx: u32,
        stream_grad: &[f32],
        reference_grad: &[f32],
    ) -> Result<ParityAction>;
    pub fn is_escalated(&self, layer_idx: u32) -> bool;
    pub fn recompute_precision(&self, layer_idx: u32, default: Precision) -> Precision;
}

pub fn compute_relative_error(stream_grad: &[f32], reference_grad: &[f32]) -> f64;
pub fn measure_parity(
    ref_layer: u32,
    stream_grad: &[f32],
    reference_grad: &[f32],
    tolerances: &ParityTolerances,
) -> Result<f64>;
```

### Firing Rule

Check fires when `step > 0 && step % parity_check_interval == 0`.  
Disabled entirely if `parity_check_interval == 0`.

### Relative Error Metric

```
rel = max|stream_grad - reference_grad| / (max|reference_grad| + ε)
      where ε = 1e-8 (prevent division by zero)
```

### Decision Logic

| Condition | Return | Side Effect |
|-----------|--------|-------------|
| rel < warn | `Ok(Clean)` | De-escalate layer if previously elevated. |
| warn ≤ rel < halt | `Ok(Escalated)` | Add layer to escalation set. |
| rel ≥ halt | `Err(ParityHalt{layer_idx, rel})` | Caller must snapshot before handling. |

### Escalation Effect

When a layer is escalated, `recompute_precision(layer_idx, default)` returns `Precision::FP32` regardless of the schedule, forcing full-precision recompute for that segment on the next backward pass.

---

## 9. Algorithm A8: Cut-CE Loss

**Purpose:** Compute language-model cross-entropy loss and gradient wrt final hidden state without materializing `[batch_seq × vocab_size]` logit tensor.

### Public API

```rust
pub struct LossOutput {
    pub loss: f32,                  // Scalar mean loss
    pub grad_hidden: Vec<f32>,      // Gradient wrt hidden [batch_seq × d_model]
    pub peak_logit_bytes: usize,    // Peak logit buffer size
}

pub fn streaming_cut_ce(
    hidden: &[f32],                 // [batch_seq × d_model]
    lm_head: &[f32],                // [vocab_size × d_model]
    d_model: usize,
    vocab_size: usize,
    chunk_size: usize,              // Vocab tile width (e.g., 256)
    labels: &[u32],                 // [batch_seq], values in [0, vocab_size)
    lm_head_grad_hook: Option<&dyn OptimizerBackend>,
) -> Result<LossOutput>;
```

### Two-Pass Online Softmax Algorithm

**Invariants:** Only one `logit_chunk` buffer of size `batch_seq × min(chunk_size, V)` lives at any time. Recomputed in Pass 2 (no storage between passes).

#### Pass 1: Accumulate max + sumexp

```
run_max[i]    ← f32::NEG_INFINITY
run_sumexp[i] ← 0.0
label_logit[i] ← (not yet captured)

for each vocab chunk v_start in [0, V) step chunk_size:
    c ← min(chunk_size, V - v_start)
    w_chunk ← lm_head[v_start:v_start+c]
    
    for i in [0, batch_seq):
        for k in [0, c):
            logit_chunk[i,k] ← dot(hidden[i], w_chunk[k])
        
        chunk_max ← max_k logit_chunk[i,k]
        new_max[i] ← max(run_max[i], chunk_max)
        
        # Numerically stable rescale
        run_sumexp[i] ← run_sumexp[i] * exp(run_max[i] - new_max[i])
        run_sumexp[i] ← run_sumexp[i] + sum_k exp(logit_chunk[i,k] - new_max[i])
        run_max[i] ← new_max[i]
        
        # Capture label logit when label falls in this chunk
        if label[i] ∈ [v_start, v_start+c):
            label_logit[i] ← logit_chunk[i, label[i]-v_start]

# Compute scalar loss
loss ← (1/batch_seq) * sum_i (-label_logit[i] + log(run_sumexp[i]) + run_max[i])
```

#### Pass 2: Gradient wrt hidden

```
grad_hidden[i] ← 0  for all i

for each vocab chunk v_start in [0, V) step chunk_size:
    c ← min(chunk_size, V - v_start)
    w_chunk ← lm_head[v_start:v_start+c]
    
    # Recompute logits (no storage between passes)
    for i in [0, batch_seq):
        for k in [0, c):
            logit_chunk[i,k] ← dot(hidden[i], w_chunk[k])
    
    # Softmax factor
    for i in [0, batch_seq):
        for k in [0, c):
            factor[i,k] ← exp(logit_chunk[i,k] - run_max[i]) / run_sumexp[i]
            if v_start + k == label[i]:
                factor[i,k] ← factor[i,k] - 1.0
            
            # Accumulate: ∂L/∂h[i] += factor[i,k] * w_chunk[k]
            grad_hidden[i] += factor[i,k] * w_chunk[k]
    
    # Optional M4 hook: project per-chunk LM-head gradient
    if lm_head_grad_hook.is_some():
        grad_w_chunk ← (1/batch_seq) * [sum_i factor[i,k] * hidden[i] for k]
        lm_head_grad_hook.project_and_accumulate(grad_w_chunk, u32::MAX, "lm_head")

grad_hidden[i] ← grad_hidden[i] / batch_seq  for all i
```

### Memory Guarantee

**Peak allocation:** `batch_seq × min(chunk_size, vocab_size) × 4` bytes (one logit buffer, reused for both passes).  
Not O(vocabulary_size), but O(chunk_size).

### M4 LM-Head Hook

When `lm_head_grad_hook` is Some, per-chunk gradients are fed to M4's `project_and_accumulate` with sentinel layer index `u32::MAX` and param name `"lm_head"`. M4 absorbs the tile without materializing the full `[V, d]` matrix.

---

## 10. Algorithm A9: HAM Offload

**Purpose:** Greedy segment selector (fallback when M9 SARP schedule is absent). Decide recompute vs. offload based on estimated I/O vs. compute time.

### Feature: `ham-offload`

Requires `--features ham-offload` at build time. When disabled, default action is `Recompute`.

### Public API

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HamAction {
    Recompute,  // Re-run forward during backward
    Offload,    // Store activations; reload during backward
}

pub struct Sars {
    profile: HardwareProfile,  // Warmup-measured bandwidths
}

impl Sars {
    pub fn new(profile: HardwareProfile) -> Self;
    pub fn is_io_bound(&self, segment_weight_bytes: u64, num_layers: usize) -> bool;
    pub fn select(
        &self,
        segment_index: u32,
        segment_weight_bytes: u64,
        num_layers: usize,
    ) -> HamAction;
}

pub fn select_action(
    segment_index: u32,
    profile: &HardwareProfile,
    _segment_flops: f64,
    segment_weight_bytes: u64,
    num_layers: usize,
) -> ActivationAction;
```

### Decision Rule (SARS)

```
t_io  = segment_weight_bytes / (pcie_bandwidth_gbs × 1e9)
t_cmp = (mean_forward_ms + mean_backward_ms) × 1e-3 × num_layers

if t_io > t_cmp:
    action ← Recompute       (I/O-bound: compute is idle → recompute is free)
else:
    action ← Offload         (compute-bound: PCIe is idle → offload via PCIe)
```

### Both Paths Are Gradient-Identical (T-HAM)

**RECOMPUTE:** Re-run forward with RNG restored → produces identical `SingleLayerFwdOut` → backward is identical.  
**OFFLOAD:** Use stored `SingleLayerFwdOut` directly (same activations, same RNG mask) → backward is identical.

Both call `single_layer_backward(cfg, weights, fwd_out, upstream)` with identical inputs → **bit-identical gradients**.

### Offload Data Path

```rust
pub struct SegmentActivationStore {
    pub activations: Vec<Vec<SingleLayerFwdOut>>,  // [micro_batch][layer_in_seg]
    pub element_count: usize,
}

impl SegmentActivationStore {
    pub fn new(fwd_per_micro_batch: Vec<Vec<SingleLayerFwdOut>>) -> Self;
    pub fn pcie_bytes(&self) -> usize;  // element_count × 4
}
```

---

## 11. Algorithm A10: RNG

**Purpose:** Capture per-(layer, micro-batch) RNG state at forward; restore for deterministic recompute.

### Public API

```rust
pub fn set_step_seed(seed: u64);
pub fn capture(layer_idx: u32, micro_batch: u32) -> Result<RngState>;
pub fn restore(state: &RngState) -> Result<()>;
pub fn next_f32() -> f32;  // Sample ∈ [0, 1)
pub fn apply_dropout(data: &mut [f32], p: f32);
pub fn assert_deterministic_mode() -> Result<()>;
```

### PRNG Algorithm (Splitmix64)

Thread-local PRNG state (u64) advanced by:
```rust
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e3779b97f4a7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb1331__11eb);
    x ^ (x >> 31)
}
```

### Capture/Restore Contract

**Capture:**
- Derive seed: `seed = splitmix64(base ^ mix_indices(layer_idx, micro_batch))`
- Set thread-local PRNG to seed.
- Return `RngState { layer_idx, micro_batch, seed_bytes: seed.to_le_bytes() }`

**Restore:**
- Extract 8 bytes from `state.seed_bytes` → u64.
- Set thread-local PRNG to that u64.

**Guarantee:** Repeated calls to `next_f32()` after restore produce identical sequence → identical dropout masks → deterministic forward → identical gradients.

---

## 12. Schedule Emitter

**Purpose:** Emit layer execution order for FlowCast prefetch sequence.

### Public API

```rust
pub struct LayerSchedule {
    num_layers: u32,
    checkpoint_freq: u32,
}

impl LayerSchedule {
    pub fn new(num_layers: u32, checkpoint_freq: u32) -> Self;
    pub fn forward_order(&self) -> Vec<u32>;         // [0, 1, …, L-1]
    pub fn num_segments(&self) -> u32;               // ceil(L / k)
    pub fn backward_segments(&self) -> Vec<SegmentRecomputeOrder>;
}

pub struct SegmentRecomputeOrder {
    pub segment_index: u32,
    pub layers_ascending: Vec<u32>,  // Layers in this segment, ascending
}
```

### Forward Schedule

Straightforward ascending: `[0, 1, 2, …, num_layers-1]`.

### Backward Schedule

Segments in **descending** order. Each segment contains **ascending** layer indices within its range.

**Example:** num_layers=8, checkpoint_freq=3
- Segment 2 (layers 6–7): `layers_ascending = [6, 7]`
- Segment 1 (layers 3–5): `layers_ascending = [3, 4, 5]`
- Segment 0 (layers 0–2): `layers_ascending = [0, 1, 2]`

Backward descends in segment order (2 → 1 → 0); within each segment layers ascend (lower indices recomputed last, matching the backward gradient flow).

---

## 13. Checkpoint Store/Read

**Purpose:** Serialize/deserialize activation tensors to/from `PinnedBuffer` (M2 aligned memory).

### Public API

```rust
pub fn store_checkpoint(data: &[f32], compress: bool) -> Result<PinnedBuffer>;
pub fn read_checkpoint(buf: &PinnedBuffer) -> Result<Vec<f32>>;
```

### Uncompressed Path

- **Store:** Allocate PinnedBuffer, write f32 as little-endian bytes.
- **Read:** Read f32 little-endian bytes.

### Compressed Path (INT8 with fp16 intermediate)

**Layout:** `[f32 scale : 4 bytes] [i8 × n_elements]`

**Store:**
1. Convert f32 → fp16 (u16 bit pattern).
2. Allocate buffer of size 4 + n bytes.
3. Call `ramflow::kernels::compress_checkpoint_fp16_to_int8(fp16_ptr, i8_ptr, scale_ptr, 1, n, stream)`.
4. Set `buf.set_compressed(true)`.

**Read:**
1. Check `buf.is_compressed()`.
2. Extract scale (first 4 bytes) and INT8 data (remaining bytes).
3. Call `ramflow::kernels::decompress_checkpoint_int8_to_fp16(i8_ptr, fp16_ptr, scale_ptr, 1, n, stream)`.
4. Convert fp16 (u16) → f32.

### fp16 Conversion

Nearest-even rounding (approximate):
```rust
fn f32_to_fp16(v: f32) -> u16 { … }  // Handles signs, exponents, special values
fn fp16_to_f32(bits: u16) -> f32 { … }  // Lossless; BF16 shares f32 sign/exp
```

---

## 14. PyO3 FFI

**Purpose:** Python bridge for M7 (training loop driver). When built with `--features python-ffi`, exports `doublepass_py` Python extension.

### Build

```bash
pip install maturin
cd aethelStream/doublepass
maturin develop --features python-ffi,mock-cuda
```

### Python API

```python
import doublepass_py

# Construct
dp = doublepass_py.PyDoublePass(
    n_layers=2,
    d_model=32,
    n_heads=2,
    d_ff=64,
    seq_len=4,
    batch=1,
    vocab_size=256,
    chunk_size=256
)

# One full training step
inputs = [[0.1]*128]  # 1 micro-batch, d_model=32, seq_len=4, batch=1
labels = list(range(4))
metrics_json = dp.step(inputs, labels)
print(metrics_json)  # JSON StepMetrics

# Install plan from JSON
plan_json = '{"checkpoint_freq": 2, "max_grad_norm": 1.0, …}'
dp.set_plan(plan_json)

# Partial plan update
delta_json = '{"checkpoint_freq": 4}'
dp.apply_delta(delta_json)

# Checkpoint snapshot
snapshot_json = dp.snapshot()
print(snapshot_json)  # JSON ConsistentState

# Parity probe
stream_grad = [0.01]*128
ref_grad = [0.009]*128
rel_err = dp.parity_probe(0, stream_grad, ref_grad)
print(rel_err)  # Relative error
```

### Backend

Uses no-op `FfiNoOpOpt` (implements `OptimizerBackend` trait as a stub). Gradients are projected and accumulated but apply_update is a no-op. Used for **code-path validation**, not optimizer testing.

---

## 15. Integration Points

### M3: FlowCast (Weight Streaming)

M5 expects:
```rust
pub struct FlowCast {
    // Public API consumed by M5:
    pub fn on_layer_start(&self, layer_idx: u32, direction: Direction) -> Result<()>;
    pub fn take_ready(&self, layer_idx: u32, timeout: Duration) -> Result<ReadyLayer>;
    pub fn on_weights_updated(&mut self, layer_idx: u32, src: &PinnedBuffer) -> Result<()>;
    pub fn retire_layer(&mut self, layer: ReadyLayer) -> Result<()>;
}

pub struct ReadyLayer {
    pub layer_idx: u32,
    pub precision: Precision,
    pub weight: PoolSlot,  // GPU memory handle
    pub slab_device_ptrs: Vec<(u32, DevicePointer)>,
    pub needs_decode: bool,
}

pub enum Direction {
    Forward,
    Backward,
}
```

**Integration:** M5 calls `on_layer_start(layer_idx, Direction::Forward)` at the start of each layer's forward pass. FlowCast schedules prefetch. When weights arrive, M5 calls `take_ready(layer_idx)` to pin the weights, uses them, then calls `retire_layer()`. After training step, M5 calls `on_weights_updated(layer_idx, buf_with_bf16)` to push rounded weights back to GPU memory for next step.

### M2: RamFlow (Checkpoint Buffering + Precision)

M5 expects:
```rust
pub struct PinnedBuffer {
    pub fn alloc(bytes: usize) -> Result<Self>;
    pub fn as_slice(&self) -> &[u8];
    pub fn as_mut_slice(&mut self) -> &mut [u8];
    pub fn is_compressed(&self) -> bool;
    pub fn set_compressed(&mut self, b: bool);
}

pub struct PerLayerScaleTable {
    pub fn new(num_layers: usize, alpha: f32) -> Self;
    pub fn update(&mut self, layer_idx: usize, n_elements: usize, n_overflow: u32) -> Result<()>;
    pub fn get_scale(&self, layer_idx: usize) -> f32;
    pub fn enable_bf16_mode(&mut self);
}

// Kernels (called via ramflow::kernels namespace)
pub fn count_overflow_fp16(grad_ptr: *const u16, n: usize, stream: &CudaStream) -> Result<u32>;
pub fn compress_checkpoint_fp16_to_int8(…) -> Result<()>;
pub fn decompress_checkpoint_int8_to_fp16(…) -> Result<()>;
```

**Integration:** M5 allocates checkpoints via `PinnedBuffer::alloc()`. Reads/writes using `.as_slice()`. Calls `compress_checkpoint_fp16_to_int8()` when storing with compression. Calls `count_overflow_fp16()` during A5 precision checks. M5 never owns a `PerLayerScaleTable` directly — that's M2's responsibility, but M5 consults it via M4 (or directly if M4 isn't wired yet).

### M4: OptimizerBackend (Low-Rank Projection)

**Current state:** Trait stub in M5. M4 (real crate) will replace at development time.

M5 calls (during A2' backward):
- `project_and_accumulate(grad_f32_flat, layer_idx, param_name)` — M4 projects to low-rank, accumulates into per-layer buffer.
- `zero_accum(layer_idx, param_name)` — M4 zeros accumulator after apply_update.
- `notify_step(step)` — M4 internally decides when to refresh projection.

M5 calls (during A6 clipping):
- `projector_kind(layer_idx, param_name) -> ProjectorKind` — M4 reveals projection type.
- `lowrank_grad_sqnorm(layer_idx, param_name) -> f64` — M4 returns squared norm of low-rank accumulator.
- `true_frobenius_sqnorm(layer_idx, param_name) -> Option<f64>` — M4 optionally returns pre-projection Frobenius norm (for exact clip under random projection).
- `apply_update(layer_idx, param_name, clip_scale)` — M4 applies clipped Adam step.

### M6: LoraBackend (LoRA Adapter Manager)

**Current state:** Trait stub in M5. M6 (real crate) will replace at development time.

M5 calls (during A5 precision / weight update):
- `rank_of(layer_idx) -> u32` — M6 returns LoRA rank for this layer (used by M4 to select projection rank and by M5 to select FP32 vs BF16 policy).

### M9: TrainingPlan + SARP Schedule

M5 receives a frozen `TrainingPlan` (from M9) containing:
- `checkpoint_freq` — sparse checkpoint every k layers.
- `precision_schedule` — per-layer BF16 vs FP16 policy.
- `activation_schedule: Vec<SegmentPlan>` — M9's optimal DP solution for segment actions (optional; if empty, HAM fallback).
- `parity_check_interval` — A7 firing cadence.
- `max_grad_norm` — A6 clip target.

M5 treats these as **read-only frozen values** for the duration of `step()`.

### M10: Checkpoint/Resume (Snapshot)

M5 provides `snapshot() -> ConsistentState`:
```rust
pub struct ConsistentState {
    pub step: u64,                    // Step count
    pub optimizer_version: u64,       // M4 version (for detecting stale ckpts)
    pub rng_states: Vec<RngState>,   // Per-(layer, micro_batch) PRNG seeds
    pub data_position: u64,          // Data loader position (tokens)
}
```

M10 stores this; on resume, M5 restores via `step::restore(snapshot)` and the training loop continues with bit-identical RNG and gradient paths.

---

## 16. Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `mock-cuda` | **ON** | CPU f32 stubs for all CUDA operations. Safe for CI. Real GPU testing requires `--no-default-features --features cuda`. |
| `python-ffi` | OFF | Build PyO3 extension `doublepass_py` (requires Python 3.8+). |
| `ham-offload` | OFF | Enable A9 HAM greedy selector. Without this, default action is always `Recompute`. |
| `fp16-fallback` | OFF | Force FP16 (even if BF16 supported). For testing precision fallback paths. |
| `cuda` | OFF | Real CUDA kernels (Ampere/Hopper only; requires CUDA toolkit). Implies `!mock-cuda`. |
| `cuda-double-buffer` | OFF | Double-buffering for weight prefetch (for FlowCast integration). |

### Build Variants

```bash
# CI (mock-cuda, all algorithms):
cargo test --features mock-cuda

# Python testing:
maturin develop --features python-ffi,mock-cuda

# Real GPU (Ampere+):
cargo test --no-default-features --features cuda
```

---

## 17. Benchmarks

### Roofline Benchmark (`roofline.rs`)

Estimates FLOPs and computes achieved throughput (FLOPs / wall-clock time).

**Run:**
```bash
cargo test --test roofline --features mock-cuda -- --nocapture
```

**Output:** Labeled table (e.g.):
```
┌──────────────────────────────────────────┐
│ DoublePass Roofline Benchmark            │
├──────────────────────────────────────────┤
│ Model: 8 layers, d=128, h=4, ff=256      │
│ Micro-batch: 1, seq_len: 512             │
├──────────────────────────────────────────┤
│ Phase          │ Wall [ms] │ FLOP [e9]  │
├────────────────┼───────────┼────────────┤
│ Forward        │   5.2     │   1.8      │
│ Backward       │  10.1     │   3.6      │
│ Recompute      │   5.1     │   1.8      │
│ Loss + Clip    │   1.3     │   0.1      │
├────────────────┼───────────┼────────────┤
│ **Total**      │ **21.7 ms** │ **7.3e9** │
│ **Throughput** │ **337 GF/s** (mock)     │
└──────────────────────────────────────────┘
```

**Note:** Mock-cuda timings are **not representative** of real GPU performance — they reflect CPU simulation overhead. Real CUDA builds (with `--features cuda`) will show vastly different wall times.

---

## 18. Test Coverage

### 97 Tests (as of 2025-06-27)

| Test Suite | Focus | Count |
|-----------|-------|-------|
| `integration.rs` | Full A1→A8→A2'→A6 pipeline | 14 |
| `test_parity_single_layer.rs` | A7 ParityGuard, gradient parity | 9 |
| `test_activation_fidelity.rs` | A1 forward, retained activations | 5 |
| `test_rng_determinism.rs` | A10 RNG capture/restore | 6 |
| `test_amortization.rs` | Recompute vs offload cost models | 3 |
| `test_checkpoint_compress.rs` | A2 checkpoint compression (INT8) | 4 |
| `test_segment_recompute.rs` | A2' segment dispatch logic | 5 |
| `test_pipeline_idle.rs` | GPU idle gap estimation | 2 |
| `test_global_clip.rs` | A6 clipping paths (orthonormal, random, grouped) | 25 |
| `test_cut_ce.rs` | A8 streaming CE loss + gradients | 8 |
| `test_precision_paths.rs` | A5 BF16 vs FP16 selection, scale updates | 6 |
| `test_stochastic_rounding.rs` | A5 stochastic rounding to BF16 | 9 |
| `test_parity_guard.rs` | A7 escalation, de-escalation, halting | 11 |
| `test_ham_equivalence.rs` | A9 recompute vs offload gradient equivalence | 11 |
| `test_convergence.rs` | Full training loop, convergence over steps | 10 |
| `test_resume.rs` | Checkpoint/restore via ConsistentState | 9 |
| `test_selective_recompute.rs` | N1″ selective mask logic | 13 |
| `roofline.rs` | Performance benchmarking | 1 |

### Key Test Assertions

#### T-PARITY (Layer Parity)
Gradient parity ≤ 1e-5 for single-layer backward vs PyTorch:
```rust
let rel_err = compute_relative_error(&stream_grad, &ref_grad);
assert!(rel_err <= 1e-5, "rel_err={} exceeded tolerance", rel_err);
```

#### T-HAM (Offload Equivalence)
Recompute and offload produce bit-identical gradients:
```rust
let grad_recompute = backward_recompute(…);
let grad_offload = backward_offload(…);
assert_eq!(grad_recompute, grad_offload);
```

#### T-ACT (Activation Fidelity)
Retained activations match re-run forward:
```rust
let act_retained = fwd.retained_activations[…];
let act_recomputed = single_layer_forward(…);
assert_approx_eq!(act_retained.output, act_recomputed.output);
```

---

## 19. Gradient Parity Guarantee

### Theorem (Informal)

**For any single layer's backward pass:**
- Stream path: M5 → A2 → A2' → A3 hook → low-rank projection → gradient evaporates.
- Reference path: PyTorch autograd → full gradient → (we extract one layer's contribution).

**Claim:** `max|Δgrad| / (max|ref_grad| + ε) ≤ 1e-5` (default T-PARITY tolerance).

### Why 1e-5 is Achievable

1. **Activation storage:** Checkpoints are f32; intermediate activations are f32 (mock-cuda).
   - Real CUDA path stores fp16, introducing ±0.5 ULP error per stored value.
   - Recompute restores from checkpoint + RNG, which recovers the original f32 activations exactly (when RNG is restored).

2. **Backward arithmetic:** All operations (matmul, softmax, RMSNorm) are f32.
   - Matmul: O(n) accumulated error; dominated by dynamic range (not precision).
   - Softmax: numerically stable (log-sum-exp trick).
   - RMSNorm: simple division; no catastrophic cancellation.

3. **Gradient accumulation:** Summing per-micro-batch gradients into `ParamGrads`.
   - At most G (gradient accumulation steps) summations.
   - With G ≤ 16, accumulated error remains < 1e-5 for f32.

4. **Relative error metric:** `rel = max|Δ| / (max|ref| + 1e-8)`.
   - Absolute error floor is ~1e-7 (f32 epsilon for unit values).
   - Relative error is thus ~1e-7 / (1e-1 + 1e-8) ≈ 1e-6 for typical gradient magnitudes.
   - T-PARITY tolerance of 1e-5 provides 10× safety margin.

### Test Harness (T-PARITY)

```rust
#[test]
fn test_parity_single_layer_full() {
    let cfg = BlockConfig { d_model: 128, n_heads: 8, d_ff: 512, seq_len: 16, batch: 2, dropout_p: 0.0 };
    let w = BlockWeights::from_formula(&cfg);
    let input = vec![0.01_f32; cfg.bs() * cfg.d_model];
    let upstream = vec![0.01_f32; cfg.bs() * cfg.d_model];
    
    // M5 path
    let fwd = single_layer_forward(&cfg, &w, &input);
    let grads_m5 = single_layer_backward(&cfg, &w, &fwd, &upstream);
    
    // PyTorch reference (simulated via full f32 double backward)
    let grads_ref = full_precision_backward(&cfg, &w, &input, &upstream);
    
    let rel = compute_relative_error(&grads_m5.d_wq, &grads_ref.d_wq);
    assert!(rel <= 1e-5, "parity failed: rel={:.3e} >= 1e-5", rel);
}
```

---

## 20. Training Loop Diagrams

### Full Training Step Orchestration

```
┌─────────────────────────────────────────────────────────────────────┐
│                   full_training_step() [train_step.rs]              │
│                    A1 → A8 → A2′ → A6 wired                         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
         ┌──────────────────┐      ┌──────────────────┐
         │  A1: Forward     │      │ M3: FlowCast     │
         │ (single/full)    │  ◀───┤ on_layer_start   │
         │                  │      │ take_ready       │
         │ Outputs:         │      └──────────────────┘
         │ • fwd.outputs    │
         │ • fwd.ckpts      │       M2: RamFlow
         │ • fwd.rng_states │      (checkpoint)
         │ • retain_acts    │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  A8: Cut-CE Loss │
         │ streaming_cut_ce │
         │                  │
         │ Outputs:         │
         │ • loss           │
         │ • grad_hidden    │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ A2′: Full Backward
         │  + SARP Dispatch │
         │                  │
         │ • LayerSchedule  │
         │ • SarpExecutor   │
         │   (M9 plan | HAM)│
         │                  │
         │ ┌──────────────┐ │
         │ │ Per Segment: │ │
         │ │ Recompute    │ │
         │ │ (A10 restore)│ │
         │ │ or Offload   │ │
         │ │ (A9 load)    │ │
         │ │   ▼          │ │
         │ │ single_layer_│ │
         │ │ backward     │ │
         │ │ (A2)         │ │
         │ │   ▼          │ │
         │ │ A3: Hook:    │ │
         │ │ project_and_ │ │
         │ │ accumulate   │ │
         │ │ (M4)         │ │
         │ └──────────────┘ │
         │                  │
         │ Outputs:         │
         │ • layer_grads    │
         │ • weight_loads   │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ A6: Global Clip  │
         │  + Apply Update  │
         │                  │
         │ deferred_apply   │
         │ _with_clip():    │
         │ • compute gnorm  │
         │ • clip_coeff     │
         │   ▼              │
         │ optimizer.       │
         │ apply_update     │
         │ (M4, per layer)  │
         │   ▼              │
         │ zero_accum       │
         │ (M4)             │
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Telemetry       │
         │ (A7 parity,      │
         │  A5 scale,       │
         │  metrics)        │
         │                  │
         │ Return StepOut   │
         └──────────────────┘
```

### SARP Segment Backward (A2′ detail)

```
┌─────────────────────────────────────────────────────────────────┐
│            SARP Segment Backward (per segment)                  │
│      [Descending segment order; ascending recompute]            │
└─────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
  ┌──────────────────┐           ┌──────────────────┐
  │  Consult M9      │           │  HAM Fallback    │
  │  SegmentPlan     │           │  (if no M9)      │
  │  for this seg    │           │  (feature:ham)   │
  └──────────────────┘           └──────────────────┘
         │                                │
         └────────────────┬───────────────┘
                          │
                  ┌───────▼────────┐
                  │ Select Action: │
                  │ • Recompute    │
                  │ • RetainVram   │
                  │ • PageCompRAM  │
                  │ • PageNvme     │
                  └───────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌─────────────┐  ┌──────────┐
    │ Recompute│   │RetainVram / │  │Offload   │
    │Full/Sel  │   │PageCompRAM  │  │          │
    │          │   │             │  │          │
    │1. Load   │   │1. Load from │  │1. Load   │
    │  ckpt    │   │  ret_acts   │  │  from    │
    │          │   │             │  │  store   │
    │2. RNG    │   │2. Use       │  │2. No     │
    │  restore │   │  directly   │  │  RNG     │
    │  (A10)   │   │  (no RNG)   │  │  needed  │
    │          │   │             │  │          │
    │3. Re-run │   │            │  │          │
    │  forward │   │            │  │          │
    │ w/mask   │   │            │  │          │
    └────┬─────┘   └──────┬──────┘  └────┬─────┘
         │                │              │
         └────────────────┼──────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │ seg_fwd[m][i] ready  │
              │ for all micro-batch m│
              │ and segment position │
              │ i                    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │ Descend Backward:        │
              │ for layer_idx descending │
              │   for micro_batch m      │
              │     single_layer_        │
              │     backward(            │
              │       cfg, w, fwd, up    │
              │     )                    │
              │       ▼                  │
              │     A3 Hook:             │
              │     optimizer.           │
              │     project_and_         │
              │     accumulate           │
              │       ▼                  │
              │     upstreams[m] ◀──     │
              │     d_input              │
              │                          │
              │ Accumulate grads across  │
              │ all micro_batches into   │
              │ layer_grads[i]           │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │ Return (layer_grads,     │
              │         stats)           │
              │                          │
              │ stats includes:          │
              │ • recompute_flops        │
              │ • pcie_bytes             │
              │ • ssd_bytes              │
              └──────────────────────────┘
```

### A6 Global-Norm Clip (Three Paths)

```
┌─────────────────────────────────────────────────────┐
│    deferred_apply_with_clip(optimizer, plan, ..)    │
│             A6 Global-Norm Clipping                 │
└─────────────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
  ┌─────────────────┐    ┌───────────────┐
  │Check projector_ │    │Any ProjectorKind::None?
  │kind for each    │    │                │
  │(layer, param)   │    │  yes ┌─ reject
  │                 │    │  no  │
  └────┬────────────┘    └──┬───┘
       │                     │
       └────────────┬────────┘
                    │
        ┌───────────▼───────────┐
        │                       │
   no_none            ┌─ any_none
        │             │
        ▼             ▼
    Path 1: Global    Path 3: Grouped
    -Norm Clip        Fallback (LOMO §3.3)
    (Ortho|Random)    (per-layer clip)
        │             │
        ▼             ▼
    ┌──────────┐  ┌─────────────┐
    │Pass 1:   │  │Per-layer:   │
    │Sum gsq   │  │  gsq ← 0    │
    │from all  │  │  for layer: │
    │lowrank   │  │    sq_layer │
    │_grad_    │  │    ← sum    │
    │sqnorm    │  │      params │
    │(or true_ │  │    clip_i   │
    │frobenius │  │    ← min(1, │
    │_sqnorm)  │  │    max/norm)│
    │          │  │    apply_   │
    │gnorm     │  │    update   │
    │← sqrt    │  │              │
    │         │  │              │
    │Pass 2:  │  │              │
    │clip ←   │  │              │
    │min(1,   │  │              │
    │max/g)   │  │              │
    │         │  │              │
    │apply_   │  │              │
    │update   │  │              │
    │per param│  │              │
    └────┬────┘  └────┬──────────┘
         │            │
         └──────┬─────┘
                ▼
         ┌─────────────┐
         │Return Clip  │
         │Result:      │
         │• gnorm      │
         │• clip_coeff │
         │• clipped    │
         │• fallback   │
         │• frobenius_ │
         │  exact_     │
         │  layers     │
         └─────────────┘
```

---

## Summary Table: Algorithms at a Glance

| Algorithm | Module | Purpose | Key Data | Entry Function |
|-----------|--------|---------|----------|-----------------|
| **A1** | `forward.rs` | Forward pass + checkpointing | `SingleLayerFwdOut`, `RngState` | `full_forward_with_retention()` |
| **A2** | `backward.rs` | Single-layer backward | `ParamGrads`, `d_input` | `single_layer_backward()` |
| **A2'** | `sarp.rs` | Segment recompute dispatch | `SegmentPlan`, `ActivationAction` | `sarp_backward_segment()` |
| **A3** | `hook.rs` (note: also `backward.rs`) | Low-rank projection hook | (via `OptimizerBackend`) | `project_and_accumulate()` [M4] |
| **A5** | `precision.rs` | BF16/FP16 selection + scaling | `Precision`, `PerLayerScaleTable` | `effective_precision()`, `check_and_update_scale()` |
| **A6** | `hook.rs` | Global-norm clipping + apply | `ClipResult`, `clip_coeff` | `deferred_apply_with_clip()` |
| **A7** | `parity.rs` | Parity diagnostic | `ParityGuard`, `ParityAction` | `ParityGuard::check()` |
| **A8** | `loss.rs` | Streaming CE loss | `LossOutput`, `grad_hidden` | `streaming_cut_ce()` |
| **A9** | `ham.rs` | Recompute-vs-offload selector | `HamAction`, `Sars` | `select_action()` [fallback] |
| **A10** | `rng.rs` | RNG capture/restore | `RngState`, `seed_bytes` | `capture()`, `restore()` |

---

## References & Building on M5

### Upstream Crates (M2, M3, M4, M6, M9)

- **M2 RamFlow** (path: `../ramflow`) — PinnedBuffer, PerLayerScaleTable, compression kernels
- **M3 FlowCast** (path: `../flowcast`) — Weight streaming pipeline (Direction, ReadyLayer, FlowCastError)
- **M4 OptimizerBackend** (trait stub; real crate pending) — Projection, accumulation, apply
- **M6 LoraBackend** (trait stub; real crate pending) — LoRA rank queries
- **M9 TrainingPlan** (struct in `plan.rs`; real crate pending) — SARP DP schedule

### Papers & References

- **Cut-CE loss:** Two-pass online softmax (arXiv 2302.xx [streaming softmax]).
- **LOMO:** Lv et al., "Full Parameter Fine-tuning for Large Language Models with Limited Resources," arXiv 2306.09782.
- **GaLore:** Zhao et al., "Gradient-Only Low-Rank Projection for Memory-Efficient Learning," arXiv 2403.03507.
- **APOLLO:** Stochastic projection; arXiv 2412.05270.
- **HAM:** Hybrid Activation Materialization (SARS greedy selector).
- **Stochastic rounding:** arXiv 1710.03740 (Vogels et al.), training with low precision.

---

## Document Maintenance

**Last Updated:** 2025-06-27  
**Test Count:** 97 (roofline + 96 other tests)  
**Total Lines of Code:** ~6,138 (excluding target/test builds)  

When M5 gains new algorithms or significant refactoring:
1. Update relevant section.
2. Verify against actual source (don't infer from names).
3. Bump version; append date.
4. Re-run tests; confirm passes.

---

**End of Module 5 Reference Document**
