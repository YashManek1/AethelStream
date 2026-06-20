//! A5 mixed precision: BF16 default; FP16 fallback using M2 PerLayerScaleTable + overflow kernels.
//!
//! **BF16 path (Ampere+):** `effective_precision` returns `Precision::BF16`; overflow checks
//! are skipped entirely because BF16 has native NaN/Inf immunity.  The `PerLayerScaleTable`
//! must be put in `bf16_mode` before training begins, which fixes all scales at 1.0.
//!
//! **FP16 fallback (Turing / no-BF16):** `check_and_update_scale` drives the M2
//! `PerLayerScaleTable` (Algorithm 6): calls `count_overflow_fp16`, updates the EWA
//! density, and returns `true` (skip apply) when overflow is detected.  The scale
//! table handles backoff and growth internally:
//! - density > `overflow_high_threshold` (0.001): scale halved, floor 1.0.
//! - density < `overflow_low_threshold` (0.0001) AND scale < 65536: scale doubled.
//!
//! **Master-weight policies (A5)**
//! - LoRA: FP32 adapter weights only; base weights are frozen (no write-back).
//!   Use `apply_lora_update_fp32`.
//! - Full-param GaLore: stochastic-rounded BF16 update in-place, then the caller
//!   must stream the result back via `FlowCast::on_weights_updated`.
//!   Use `apply_galore_bf16_update`.

use crate::{DoublePassError, Precision, Result};
use ramflow::PerLayerScaleTable;

// ---------------------------------------------------------------------------
// Precision selection
// ---------------------------------------------------------------------------

/// Return the effective compute precision for `layer_idx`.
///
/// Rules (in priority order):
/// 1. If `schedule[layer_idx]` is `BF16` and `hardware_supports_bf16` → `BF16`.
/// 2. If `schedule[layer_idx]` is `BF16` but hardware lacks support → degrade to `FP16`.
/// 3. Otherwise: honour `schedule[layer_idx]` exactly.
/// 4. No schedule entry for `layer_idx` (out of bounds) → default to `BF16` if supported,
///    else `FP16`.
pub fn effective_precision(
    layer_idx: u32,
    schedule: &[Precision],
    hardware_supports_bf16: bool,
) -> Precision {
    let scheduled = schedule
        .get(layer_idx as usize)
        .copied()
        .unwrap_or(Precision::BF16);
    match scheduled {
        Precision::BF16 => {
            if hardware_supports_bf16 {
                Precision::BF16
            } else {
                Precision::FP16
            }
        }
        other => other,
    }
}

// ---------------------------------------------------------------------------
// FP16 overflow check + scale update
// ---------------------------------------------------------------------------

/// Check FP16 gradient tensor for overflow and update the EWA scale table.
///
/// Returns `true` when overflow is detected — the caller should skip `apply_update`
/// for this layer this step.  The `PerLayerScaleTable` internally halves the scale
/// when the EWA density exceeds `overflow_high_threshold` and doubles it (up to
/// 65536) when the density falls below `overflow_low_threshold` after a clean interval.
///
/// When the scale table is in `bf16_mode` (enabled via `enable_bf16_mode()`), this
/// function still runs the kernel but `table.update()` is a no-op — safe to call
/// but not needed.  The BF16 fast path should omit the call entirely.
///
/// # Safety
/// `grad_data` must be a valid pointer to `n_elements` FP16 values (as `u16` bits).
/// In `mock-cuda` mode this is host memory; in real CUDA mode it is device memory.
///
/// # Errors
/// Returns `DoublePassError::Config` if the CUDA stream cannot be created or if
/// `layer_idx` is out of bounds for the scale table.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn check_and_update_scale(
    scale_table: &mut PerLayerScaleTable,
    layer_idx: usize,
    grad_data: *const u16,
    n_elements: usize,
) -> Result<bool> {
    if n_elements == 0 {
        return Ok(false);
    }

    let stream = ramflow::cuda_bridge::CudaStream::new()
        .map_err(|e| DoublePassError::Config(format!("CudaStream::new: {e}")))?;

    let n_overflow = ramflow::kernels::count_overflow_fp16(grad_data, n_elements, &stream)
        .map_err(|e| DoublePassError::Config(format!("count_overflow_fp16: {e}")))?;

    scale_table
        .update(layer_idx, n_elements, n_overflow)
        .map_err(|e| DoublePassError::Config(format!("PerLayerScaleTable::update: {e}")))?;

    Ok(n_overflow > 0)
}

// ---------------------------------------------------------------------------
// Stochastic rounding
// ---------------------------------------------------------------------------

/// Convert an f32 value to its BF16 representation using stochastic rounding (SRTE).
///
/// BF16 is f32 with the lower 16 mantissa bits discarded.  Deterministic truncation
/// flushes sub-epsilon updates to zero, causing weight stagnation near convergence.
/// SRTE preserves small gradients probabilistically: an update of magnitude `ε` has
/// probability `(bits[15:0] / 2^16)` of incrementing the BF16 mantissa by one ULP,
/// so `E[w_bf16] = w_f32` in expectation.
///
/// # Parameters
/// - `val`       — the f32 value to round.
/// - `rand_bits` — 16 bits of uniform randomness consumed per call (low 16 bits of
///   one LCG step; see `apply_galore_bf16_update`).
///
/// # Returns
/// The rounded BF16 value as a `u16` (identical to the high 16 bits of the f32
/// representation after rounding).
#[inline]
pub fn stochastic_round_to_bf16(val: f32, rand_bits: u16) -> u16 {
    let bits = val.to_bits();
    let low16 = (bits & 0xFFFF) as u16;
    let high16 = (bits >> 16) as u16;
    // Round up if the discarded fraction exceeds the random threshold.
    // saturating_add prevents wrap-around on ±Inf / NaN bit patterns.
    if low16 > rand_bits {
        high16.saturating_add(1)
    } else {
        high16
    }
}

/// Widen a BF16 value (as `u16`) back to f32 by zero-padding the lower 16 bits.
///
/// Lossless: BF16 shares the f32 sign and exponent fields (7 vs 23 mantissa bits).
#[inline]
pub fn bf16_to_f32(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

// ---------------------------------------------------------------------------
// Master-weight update policies
// ---------------------------------------------------------------------------

/// LoRA master-weight policy: apply FP32 adapter delta in-place.
///
/// Base model weights are frozen; only the low-rank A/B adapter parameters
/// (stored in FP32) receive updates.  No BF16 rounding; no write-back.
///
/// # Errors
/// Returns `DoublePassError::Config` on length mismatch.
pub fn apply_lora_update_fp32(weights: &mut [f32], delta: &[f32]) -> Result<()> {
    if weights.len() != delta.len() {
        return Err(DoublePassError::Config(format!(
            "apply_lora_update_fp32: weights.len()={} != delta.len()={}",
            weights.len(),
            delta.len(),
        )));
    }
    for (w, d) in weights.iter_mut().zip(delta.iter()) {
        *w += d;
    }
    Ok(())
}

/// GaLore full-param master-weight policy: stochastic-rounded BF16 update in-place.
///
/// Each element of `master` (FP32 master copy) is updated by `delta[i]`, then
/// stochastic-rounded to BF16 and stored back as a widened f32.  After this call
/// the caller must stream the rounded weights to device via
/// `FlowCast::on_weights_updated` with a `PinnedBuffer` containing the BF16 values.
///
/// Stochastic rounding avoids weight stagnation when gradient steps are smaller
/// than one BF16 ULP at the current weight magnitude (arXiv 1710.03740).
///
/// # Parameters
/// - `master` — FP32 master weights, updated in-place.
/// - `delta`  — FP32 optimizer step (already clipped and scaled).
/// - `rng`    — LCG state (`u64`), updated in-place; 16 bits consumed per element.
///
/// # Errors
/// Returns `DoublePassError::Config` on length mismatch.
pub fn apply_galore_bf16_update(master: &mut [f32], delta: &[f32], rng: &mut u64) -> Result<()> {
    if master.len() != delta.len() {
        return Err(DoublePassError::Config(format!(
            "apply_galore_bf16_update: master.len()={} != delta.len()={}",
            master.len(),
            delta.len(),
        )));
    }
    for (m, d) in master.iter_mut().zip(delta.iter()) {
        *m += d;
        let rand_bits = lcg_next(rng);
        let bf16 = stochastic_round_to_bf16(*m, rand_bits);
        *m = bf16_to_f32(bf16);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Private: LCG RNG
// ---------------------------------------------------------------------------

/// Advance a 64-bit multiplicative LCG (Knuth MMIX) and return bits 33–48.
///
/// Bits 33–48 are used (skips the low-entropy low bits of the LCG output).
#[inline]
fn lcg_next(state: &mut u64) -> u16 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) & 0xFFFF) as u16
}
