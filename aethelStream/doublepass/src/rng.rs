//! A10 RNG capture/restore: per-(layer, micro-batch) RNG capture at forward; restore at recompute.
//!
//! Deterministic kernels guard: cuDNN/cuBLAS deterministic flags must be set
//! on the recompute + backward path.
//!
//! Implementation: splitmix64 PRNG with per-(step, layer, micro) derived seeds.
//! Under mock-cuda: host PRNG only. Under cuda: curandState (bytes), cfg-gated.

use crate::error::DoublePassError;
use crate::state::RngState;
use crate::Result;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global per-step base seed. Set once per step before any capture calls.
static STEP_SEED: AtomicU64 = AtomicU64::new(0xdeadbeef_cafef00d);

thread_local! {
    /// Current PRNG state for the calling thread (splitmix64 internal state).
    static PRNG_STATE: Cell<u64> = const { Cell::new(0) };
}

/// One splitmix64 round — invertible bijection on u64.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Derive a unique seed from `(layer_idx, micro_batch)` pair.
#[inline]
fn mix_indices(layer_idx: u32, micro_batch: u32) -> u64 {
    let a = (layer_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
    let b = (micro_batch as u64).wrapping_mul(0x6c62_272e_07bb_0142);
    a ^ b ^ (a << 32)
}

/// Set the step-global base seed. Call once per training step before any [`capture`] calls.
pub fn set_step_seed(seed: u64) {
    STEP_SEED.store(seed, Ordering::Relaxed);
}

/// Advance the calling thread's PRNG by one step and return the new raw state.
#[inline]
fn advance() -> u64 {
    PRNG_STATE.with(|cell| {
        let next = splitmix64(cell.get());
        cell.set(next);
        next
    })
}

/// Capture the current thread RNG state for `(layer_idx, micro_batch)`.
///
/// Sets the thread-local PRNG to a seed derived deterministically from
/// `STEP_SEED`, `layer_idx`, and `micro_batch`. Call this immediately before
/// any operations that draw from the PRNG (e.g., dropout).
///
/// # Errors
/// Currently infallible under mock-cuda. Returns `Result` for API stability
/// (real CUDA path will capture `curandState`).
pub fn capture(layer_idx: u32, micro_batch: u32) -> Result<RngState> {
    let base = STEP_SEED.load(Ordering::Relaxed);
    let seed = splitmix64(base ^ mix_indices(layer_idx, micro_batch));
    PRNG_STATE.with(|cell| cell.set(seed));
    Ok(RngState {
        layer_idx,
        micro_batch,
        seed_bytes: seed.to_le_bytes().to_vec(),
    })
}

/// Restore the RNG state for `(layer_idx, micro_batch)` before recompute.
///
/// Guarantees bit-identical dropout masks between forward and recompute when
/// the same `RngState` returned by [`capture`] is passed here.
///
/// # Errors
/// Returns `RngStateMissing` if `state.seed_bytes` is not exactly 8 bytes.
pub fn restore(state: &RngState) -> Result<()> {
    if state.seed_bytes.len() != 8 {
        return Err(DoublePassError::RngStateMissing {
            layer_idx: state.layer_idx,
            micro_batch: state.micro_batch,
        });
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&state.seed_bytes);
    let seed = u64::from_le_bytes(bytes);
    PRNG_STATE.with(|cell| cell.set(seed));
    Ok(())
}

/// Sample a uniform `f32` in `[0, 1)` from the thread-local PRNG.
///
/// Advances the PRNG state by one step.
#[inline]
pub fn next_f32() -> f32 {
    let bits = advance();
    // Use top 23 bits as mantissa → uniform in [1, 2) → subtract 1 → [0, 1).
    let mantissa = ((bits >> 41) as u32) | 0x3f80_0000;
    f32::from_bits(mantissa) - 1.0
}

/// Apply in-place Bernoulli dropout to `data` with drop probability `p`.
///
/// When `p == 0.0` this is a **no-op** (no RNG draw). When `p > 0`, each
/// element is zeroed with probability `p` and scaled by `1 / (1 - p)` otherwise,
/// preserving the expected value.
pub fn apply_dropout(data: &mut [f32], p: f32) {
    if p <= 0.0 {
        return;
    }
    let p_clamped = p.clamp(0.0, 0.999_9);
    let scale = 1.0 / (1.0 - p_clamped);
    for v in data.iter_mut() {
        if next_f32() < p_clamped {
            *v = 0.0;
        } else {
            *v *= scale;
        }
    }
}

/// Assert that deterministic kernel mode is enabled.
///
/// Under mock-cuda: always `Ok(())`.
/// Under cuda: checks cuDNN / cuBLAS deterministic flags (cfg-gated, not yet wired).
pub fn assert_deterministic_mode() -> Result<()> {
    // Real CUDA: assert cuDNN / cuBLAS deterministic flags — landing in the
    // cuda-kernel sprint when cuDNN is first invoked. No-op under mock.
    Ok(())
}
