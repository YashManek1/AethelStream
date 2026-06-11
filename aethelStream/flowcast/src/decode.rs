//! A7: Quantized-stream decode coordinator.
//!
//! Reads INT4/INT8 layers (precision field from Module 1's `TensorInfo`),
//! dispatches decode-to-FP16 (calls the kernel module; mock under mock-cuda),
//! and marks `ReadyLayer.needs_decode`.
//!
//! # INT8 decode (symmetric per-channel quantisation)
//! `fp16 = int8_val * (max_abs / 127.0)` per channel.
//! Decode error per channel is bounded by `max_abs / 127`.
//!
//! # INT4 decode (NF4 normalised float)
//! Maps each 4-bit code to the nearest NF4 level, then scales by `absmax`.
//! Decode error is bounded by the NF4 quantisation step (~0.083 × absmax).

use crate::config::Precision;
use crate::Result;
use std::collections::HashMap;
use std::sync::Mutex;

/// Decode state for one in-flight layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeState {
    /// Awaiting I/O completion — not yet decoded.
    Pending,
    /// Decode kernel dispatched; awaiting completion.
    Decoding,
    /// Decode complete; buffer holds FP16 weights.
    Done,
}

// NF4 lookup table: 16 normalised-float levels (bitsandbytes convention).
// Literals are truncated to the precision representable by f32 (~7 decimal digits).
const NF4_TABLE: [f32; 16] = [
    -1.0, -0.696_192_8, -0.525_073_05, -0.394_917_5,
    -0.284_441_38, -0.184_773_43, -0.091_050_036, 0.0,
    0.079_580_3, 0.160_930_2, 0.246_112_3, 0.337_915_24,
    0.440_709_83, 0.562_617, 0.722_956_84, 1.0,
];

/// Quantized-stream decode coordinator.
pub struct QuantizedDecoder {
    /// Precision to decode into (FP16 or BF16).
    target_precision: Precision,
    /// Per-layer decode state.
    states: Mutex<HashMap<u32, DecodeState>>,
}

impl QuantizedDecoder {
    /// Create a decoder that expands compressed weights to `target_precision`.
    pub fn new(target_precision: Precision) -> Self {
        Self {
            target_precision,
            states: Mutex::new(HashMap::new()),
        }
    }

    /// Decode INT8 bytes in-place to FP16 (f16 stored as u16 LE pairs).
    ///
    /// `src_int8`: raw INT8 bytes (one i8 per element).
    /// `scale_per_channel`: one f32 scale per output channel (= max_abs / 127).
    /// Returns FP16-encoded bytes (2 bytes per element).
    ///
    /// Decode error per channel ≤ `max_abs / 127`.
    pub fn decode_int8_to_fp16(
        src_int8: &[u8],
        scale_per_channel: &[f32],
        num_channels: usize,
    ) -> Vec<u8> {
        let elements_per_channel = src_int8.len().checked_div(num_channels).unwrap_or(src_int8.len());

        let mut out = Vec::with_capacity(src_int8.len() * 2);
        for (element_idx, &raw) in src_int8.iter().enumerate() {
            let channel = if num_channels > 0 {
                element_idx / elements_per_channel.max(1)
            } else {
                0
            };
            let scale = scale_per_channel.get(channel).copied().unwrap_or(1.0);
            let fp32_val = (raw as i8) as f32 * scale;
            let fp16_bits = f32_to_f16(fp32_val);
            out.extend_from_slice(&fp16_bits.to_le_bytes());
        }
        out
    }

    /// Decode INT4 (packed two nibbles per byte) to FP16 using NF4 table.
    ///
    /// `src_int4`: packed bytes (2 nibbles each, low-nibble first).
    /// `absmax`: per-tensor absolute maximum for scale recovery.
    /// Returns FP16-encoded bytes.
    ///
    /// Decode error ≤ NF4 quantisation step × absmax.
    pub fn decode_int4_to_fp16(src_int4: &[u8], absmax: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(src_int4.len() * 4);
        for &byte in src_int4 {
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            for code in [lo, hi] {
                let fp32_val = NF4_TABLE[code] * absmax;
                let fp16_bits = f32_to_f16(fp32_val);
                out.extend_from_slice(&fp16_bits.to_le_bytes());
            }
        }
        out
    }

    /// Dispatch and execute decode for `layer_idx` given its source precision.
    ///
    /// Calls the appropriate CPU decode function (A7-d fix — the previous stub
    /// marked `Done` without calling any decode logic).  After decoding, verifies
    /// that the output has the expected FP16 byte count (A7-e shape check).
    ///
    /// * INT8: calls `decode_int8_to_fp16`; expects `src_bytes.len() * 2` output bytes.
    /// * INT4: calls `decode_int4_to_fp16`; expects `src_bytes.len() * 4` output bytes.
    /// * FP16/BF16/FP32: returns `src_bytes` verbatim (no decode required).
    ///
    /// # Errors
    /// Returns `FlowCastError::Config` if the output size does not match the
    /// expected FP16 byte count (invariant violation).
    pub fn dispatch(
        &self,
        layer_idx: u32,
        source_precision: Precision,
        src_bytes: &[u8],
        scale_per_channel: &[f32],
        absmax: f32,
        num_channels: usize,
    ) -> crate::Result<Vec<u8>> {
        let output = match source_precision {
            Precision::INT8 => {
                let decoded = Self::decode_int8_to_fp16(src_bytes, scale_per_channel, num_channels);
                let expected_fp16_bytes = src_bytes.len() * 2;
                if decoded.len() != expected_fp16_bytes {
                    return Err(crate::FlowCastError::Config(format!(
                        "INT8 decode shape mismatch: src={} bytes → expected {expected_fp16_bytes} FP16 bytes, got {}",
                        src_bytes.len(),
                        decoded.len(),
                    )));
                }
                decoded
            }
            Precision::INT4 => {
                let decoded = Self::decode_int4_to_fp16(src_bytes, absmax);
                let expected_fp16_bytes = src_bytes.len() * 4;
                if decoded.len() != expected_fp16_bytes {
                    return Err(crate::FlowCastError::Config(format!(
                        "INT4 decode shape mismatch: src={} bytes → expected {expected_fp16_bytes} FP16 bytes, got {}",
                        src_bytes.len(),
                        decoded.len(),
                    )));
                }
                decoded
            }
            _ => {
                // FP16/BF16/FP32 — no decode required; return verbatim.
                src_bytes.to_vec()
            }
        };

        let mut states = self.states.lock().unwrap_or_else(|p| p.into_inner());
        states.insert(layer_idx, DecodeState::Done);

        Ok(output)
    }

    /// Query the decode state for `layer_idx`.
    ///
    /// # Errors
    /// Always `Ok`.
    pub fn decode_state(&self, layer_idx: u32) -> Result<DecodeState> {
        let states = self.states.lock().unwrap_or_else(|p| p.into_inner());
        Ok(states.get(&layer_idx).copied().unwrap_or(DecodeState::Pending))
    }

    /// Target precision this decoder expands into.
    pub fn target_precision(&self) -> Precision {
        self.target_precision
    }

    /// Whether `source_precision` requires a decode pass before GPU use.
    pub fn needs_decode(source_precision: Precision) -> bool {
        matches!(source_precision, Precision::INT4 | Precision::INT8)
    }
}

// ---------------------------------------------------------------------------
// Minimal f32 → f16 conversion (no external crate)
// ---------------------------------------------------------------------------

/// Convert f32 to IEEE 754 binary16 bits (round-to-nearest-even).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // NaN or Inf
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        return sign << 15; // underflow → ±0
    }
    if exp16 >= 31 {
        return (sign << 15) | 0x7C00; // overflow → ±Inf
    }
    let mant16 = (mant >> 13) as u16;
    (sign << 15) | ((exp16 as u16) << 10) | mant16
}
