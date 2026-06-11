// src/kernels/mod.rs — typed Rust wrappers around compiled CUDA kernels.

use crate::cuda_bridge::CudaStream;
use crate::{RamFlowError, Result};

const FP16_ALIGNMENT: usize = std::mem::align_of::<u16>();
const U32_ALIGNMENT: usize = std::mem::align_of::<u32>();
const F32_ALIGNMENT: usize = std::mem::align_of::<f32>();

/// Check whether any FP16 element in `grad_device` is NaN or Inf.
///
/// `grad_device` must point to `n_elements` FP16 values on the device in real
/// CUDA mode, or host memory in `mock-cuda` mode.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn fused_overflow_check(
    grad_device: *const u16,
    n_elements: usize,
    stream: &CudaStream,
) -> Result<bool> {
    validate_ptr("grad_device", grad_device, FP16_ALIGNMENT)?;
    let n_i32 = validate_element_count(n_elements)?;
    if n_i32 == 0 {
        return Ok(false);
    }

    #[cfg(not(feature = "mock-cuda"))]
    {
        // Safety: validation above checks pointer shape and launch bound; caller
        // owns the device allocation lifetime for the submitted CUDA work.
        let overflow = unsafe {
            ramflow_check_overflow_fp16(
                grad_device as *const std::os::raw::c_void,
                n_i32,
                stream.as_raw(),
            )
        };
        Ok(overflow)
    }

    #[cfg(feature = "mock-cuda")]
    {
        let _ = stream;
        // Safety: in mock mode the API contract says this is host memory.
        let values = unsafe { std::slice::from_raw_parts(grad_device, n_elements) };
        Ok(values.iter().any(|bits| (*bits & 0x7C00) == 0x7C00))
    }
}

/// Count FP16 NaN/Inf elements for Module 5 compression-priority decisions.
///
/// This is the real Rust entry point for consumers of
/// `CoScheduler::should_compress_checkpoints()`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn count_overflow_fp16(
    grad_device: *const u16,
    n_elements: usize,
    stream: &CudaStream,
) -> Result<u32> {
    validate_ptr("grad_device", grad_device, FP16_ALIGNMENT)?;
    let n_i32 = validate_element_count(n_elements)?;
    if n_i32 == 0 {
        return Ok(0);
    }

    #[cfg(not(feature = "mock-cuda"))]
    {
        // Safety: validation above checks pointer shape and launch bound; caller
        // owns the device allocation lifetime for the submitted CUDA work.
        Ok(unsafe {
            ramflow_count_overflow_fp16(
                grad_device as *const std::os::raw::c_void,
                n_i32,
                stream.as_raw(),
            )
        })
    }

    #[cfg(feature = "mock-cuda")]
    {
        let _ = stream;
        // Safety: in mock mode the API contract says this is host memory.
        let values = unsafe { std::slice::from_raw_parts(grad_device, n_elements) };
        Ok(values
            .iter()
            .filter(|bits| (**bits & 0x7C00) == 0x7C00)
            .count() as u32)
    }
}

/// Round x to the nearest integer using IEEE 754 round-to-nearest-even
/// (ties go to the even integer), matching CUDA's `__float2int_rn`.
///
/// `f32::round()` uses round-half-away-from-zero (C `roundf` semantics),
/// which diverges from `__float2int_rn` on exact half-values (e.g. 2.5).
/// This helper preserves parity between the mock and the real CUDA kernel.
fn round_half_to_even(x: f32) -> f32 {
    let floor = x.floor();
    let diff = x - floor;
    // Exact tie: x = n + 0.5 for some integer n.
    if (diff - 0.5_f32).abs() < f32::EPSILON * 4.0 {
        // Round toward the even integer.
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

/// Split a packed INT8-compressed checkpoint buffer into its scale and data sub-pointers.
///
/// # Packed layout contract
/// When a [`crate::allocator::PinnedBuffer`] holds an INT8-compressed activation checkpoint
/// (i.e. `buf.is_compressed() == true`), its bytes are arranged as:
///
/// ```text
/// ┌─────────────────────────────────┬──────────────────────────────────┐
/// │  n_channels × sizeof(f32)       │  n_channels × elems_per_channel  │
/// │  per-channel scale factors (f32)│  × sizeof(i8)  quantised values  │
/// └─────────────────────────────────┴──────────────────────────────────┘
/// ^─── base ptr (512-byte aligned) ──────────────────────────────────^
/// ```
///
/// This helper computes the two sub-pointers from a single base pointer, matching
/// the layout assumed by [`compress_checkpoint_fp16_to_int8`] and
/// [`decompress_checkpoint_int8_to_fp16`] when the caller uses a single buffer.
///
/// # Safety
/// `base` must point to a buffer of at least
/// `n_channels * size_of::<f32>() + n_channels * elems_per_channel` bytes,
/// aligned to 512 bytes.
///
/// # Returns
/// `(scales_ptr, data_ptr)` — `scales_ptr` is the start of the buffer cast to
/// `*mut f32`, and `data_ptr` is offset by `n_channels * 4` bytes cast to `*mut i8`.
///
/// # Errors
/// Returns [`crate::RamFlowError::ConfigError`] if `n_channels` is zero.
pub unsafe fn split_compressed_buffer_ptrs(
    base: *mut u8,
    n_channels: usize,
) -> crate::Result<(*mut f32, *mut i8)> {
    if n_channels == 0 {
        return Err(crate::RamFlowError::ConfigError(
            "split_compressed_buffer_ptrs: n_channels must be > 0".into(),
        ));
    }
    let scale_bytes = n_channels * core::mem::size_of::<f32>();
    // SAFETY: base is valid for at least scale_bytes + data bytes (caller guarantee).
    let scales_ptr = base as *mut f32;
    let data_ptr = unsafe { base.add(scale_bytes) } as *mut i8;
    Ok((scales_ptr, data_ptr))
}

/// Compress FP16 activation checkpoint data to INT8 with one scale per channel.
///
/// # Packed buffer convention
/// Callers that allocate a single [`crate::allocator::PinnedBuffer`] for the compressed
/// result must place scales at the start of the buffer and INT8 data immediately after:
/// use [`split_compressed_buffer_ptrs`] to derive `dst_device` and `scales_device` from
/// the buffer's base pointer.
///
/// # After completion
/// On `Ok(())`, call [`crate::allocator::PinnedBuffer::set_compressed`]`(true)` on the
/// destination buffer **before** passing it to any consumer.  This flag tells downstream
/// stages (Module 5 checkpoint decompressor) that the buffer holds the packed INT8 format
/// rather than raw FP16 weights.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn compress_checkpoint_fp16_to_int8(
    src_device: *const u16,
    dst_device: *mut i8,
    scales_device: *mut f32,
    n_channels: usize,
    elems_per_channel: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_checkpoint_args(
        src_device,
        dst_device,
        scales_device,
        n_channels,
        elems_per_channel,
    )?;
    let _n_channels_i32 = validate_element_count(n_channels)?;
    let _elems_i32 = validate_element_count(elems_per_channel)?;

    #[cfg(not(feature = "mock-cuda"))]
    {
        // Safety: pointers and launch bounds are validated above.
        let rc = unsafe {
            ramflow_compress_checkpoint_fp16_to_int8(
                src_device as *const std::os::raw::c_void,
                dst_device,
                scales_device,
                _n_channels_i32,
                _elems_i32,
                stream.as_raw(),
            )
        };
        translate_kernel_status(rc)
    }

    #[cfg(feature = "mock-cuda")]
    {
        let _ = stream;
        mock_compress_checkpoint(
            src_device,
            dst_device,
            scales_device,
            n_channels,
            elems_per_channel,
        );
        Ok(())
    }
}

/// Decompress INT8 checkpoint data back to FP16 using stored channel scales.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn decompress_checkpoint_int8_to_fp16(
    src_device: *const i8,
    dst_device: *mut u16,
    scales_device: *const f32,
    n_channels: usize,
    elems_per_channel: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_ptr("src_device", src_device, std::mem::align_of::<i8>())?;
    validate_ptr("dst_device", dst_device, FP16_ALIGNMENT)?;
    validate_ptr("scales_device", scales_device, F32_ALIGNMENT)?;
    let _n_channels_i32 = validate_positive_count("n_channels", n_channels)?;
    let _elems_i32 = validate_positive_count("elems_per_channel", elems_per_channel)?;

    #[cfg(not(feature = "mock-cuda"))]
    {
        // Safety: pointers and launch bounds are validated above.
        let rc = unsafe {
            ramflow_decompress_checkpoint_int8_to_fp16(
                src_device,
                dst_device as *mut std::os::raw::c_void,
                scales_device,
                _n_channels_i32,
                _elems_i32,
                stream.as_raw(),
            )
        };
        translate_kernel_status(rc)
    }

    #[cfg(feature = "mock-cuda")]
    {
        let _ = stream;
        mock_decompress_checkpoint(
            src_device,
            dst_device,
            scales_device,
            n_channels,
            elems_per_channel,
        );
        Ok(())
    }
}

fn validate_checkpoint_args(
    src_device: *const u16,
    dst_device: *mut i8,
    scales_device: *mut f32,
    n_channels: usize,
    elems_per_channel: usize,
) -> Result<()> {
    validate_ptr("src_device", src_device, FP16_ALIGNMENT)?;
    validate_ptr("dst_device", dst_device, std::mem::align_of::<i8>())?;
    validate_ptr("scales_device", scales_device, F32_ALIGNMENT)?;
    let _ = validate_positive_count("n_channels", n_channels)?;
    let _ = validate_positive_count("elems_per_channel", elems_per_channel)?;
    Ok(())
}

fn validate_ptr<T>(name: &str, ptr: *const T, alignment: usize) -> Result<()> {
    if ptr.is_null() {
        return Err(RamFlowError::ConfigError(format!(
            "{name} must be non-null"
        )));
    }
    if !(ptr as usize).is_multiple_of(alignment) {
        return Err(RamFlowError::ConfigError(format!(
            "{name} address {ptr:p} is not {alignment}-byte aligned"
        )));
    }
    Ok(())
}

fn validate_element_count(n_elements: usize) -> Result<i32> {
    i32::try_from(n_elements)
        .map_err(|_| RamFlowError::ConfigError(format!("element count {n_elements} exceeds i32")))
}

fn validate_positive_count(name: &str, count: usize) -> Result<i32> {
    if count == 0 {
        return Err(RamFlowError::ConfigError(format!(
            "{name} must be non-zero"
        )));
    }
    validate_element_count(count)
}

fn translate_kernel_status(rc: i32) -> Result<()> {
    if rc < 0 {
        return Err(RamFlowError::CudaError(rc));
    }
    Ok(())
}

#[cfg(feature = "mock-cuda")]
fn mock_compress_checkpoint(
    src_device: *const u16,
    dst_device: *mut i8,
    scales_device: *mut f32,
    n_channels: usize,
    elems_per_channel: usize,
) {
    // Safety: public wrapper validates pointer alignment, non-nullness, and
    // non-zero dimensions before calling this mock implementation.
    let src = unsafe { std::slice::from_raw_parts(src_device, n_channels * elems_per_channel) };
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_device, n_channels * elems_per_channel) };
    let scales = unsafe { std::slice::from_raw_parts_mut(scales_device, n_channels) };

    for (channel_idx, scale_slot) in scales.iter_mut().enumerate().take(n_channels) {
        let start = channel_idx * elems_per_channel;
        let end = start + elems_per_channel;
        let channel = &src[start..end];
        let max_abs = channel
            .iter()
            .map(|bits| fp16_bits_to_f32(*bits).abs())
            .fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        *scale_slot = scale;
        for (offset, bits) in channel.iter().enumerate() {
            let quantized = round_half_to_even(fp16_bits_to_f32(*bits) / scale)
                .clamp(-128.0, 127.0);
            dst[start + offset] = quantized as i8;
        }
    }
}

#[cfg(feature = "mock-cuda")]
fn mock_decompress_checkpoint(
    src_device: *const i8,
    dst_device: *mut u16,
    scales_device: *const f32,
    n_channels: usize,
    elems_per_channel: usize,
) {
    // Safety: public wrapper validates pointer alignment, non-nullness, and
    // non-zero dimensions before calling this mock implementation.
    let src = unsafe { std::slice::from_raw_parts(src_device, n_channels * elems_per_channel) };
    let dst = unsafe { std::slice::from_raw_parts_mut(dst_device, n_channels * elems_per_channel) };
    let scales = unsafe { std::slice::from_raw_parts(scales_device, n_channels) };

    for (channel_idx, scale) in scales.iter().enumerate().take(n_channels) {
        let start = channel_idx * elems_per_channel;
        let end = start + elems_per_channel;
        for offset in start..end {
            dst[offset] = f32_to_fp16_bits(src[offset] as f32 * *scale);
        }
    }
}

#[cfg(feature = "mock-cuda")]
fn fp16_bits_to_f32(bits: u16) -> f32 {
    let sign = if (bits & 0x8000) == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as u32;
    if exponent == 0 {
        return sign * (mantissa as f32) * 2_f32.powi(-24);
    }
    if exponent == 0x1F {
        return if mantissa == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + mantissa as f32 / 1024.0) * 2_f32.powi(exponent - 15)
}

#[cfg(feature = "mock-cuda")]
fn f32_to_fp16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let sign = if value.is_sign_negative() { 0x8000 } else { 0 };
    let abs_value = value.abs();
    if !abs_value.is_finite() {
        return sign | 0x7C00;
    }
    let exponent = abs_value.log2().floor() as i32;
    let biased = (exponent + 15).clamp(1, 30) as u16;
    let mantissa = ((abs_value / 2_f32.powi(exponent) - 1.0) * 1024.0).round() as u16;
    sign | (biased << 10) | (mantissa & 0x03FF)
}

#[cfg(not(feature = "mock-cuda"))]
extern "C" {
    fn ramflow_check_overflow_fp16(
        grad_device: *const std::os::raw::c_void,
        n: i32,
        stream: *mut std::os::raw::c_void,
    ) -> bool;

    fn ramflow_count_overflow_fp16(
        grad_device: *const std::os::raw::c_void,
        n: i32,
        stream: *mut std::os::raw::c_void,
    ) -> u32;

    fn ramflow_compress_checkpoint_fp16_to_int8(
        src_device: *const std::os::raw::c_void,
        dst_device: *mut i8,
        scales_device: *mut f32,
        n_channels: i32,
        elems_per_channel: i32,
        stream: *mut std::os::raw::c_void,
    ) -> i32;

    fn ramflow_decompress_checkpoint_int8_to_fp16(
        src_device: *const i8,
        dst_device: *mut std::os::raw::c_void,
        scales_device: *const f32,
        n_channels: i32,
        elems_per_channel: i32,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

#[cfg(test)]
#[cfg(feature = "mock-cuda")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn count_overflow_mock_counts_nan_and_inf() {
        let stream = CudaStream::new().expect("stream");
        let values = [0x3C00_u16, 0x7C00, 0x7E00, 0x0000];
        let count = count_overflow_fp16(values.as_ptr(), values.len(), &stream).expect("count");
        assert_eq!(count, 2);
    }

    #[test]
    fn kernel_wrappers_reject_null_and_bad_bounds() {
        let stream = CudaStream::new().expect("stream");
        let result = fused_overflow_check(std::ptr::null(), 4, &stream);
        assert!(matches!(result, Err(RamFlowError::ConfigError(_))));

        let value = 0_u16;
        let too_large = count_overflow_fp16(&value, i32::MAX as usize + 1, &stream);
        assert!(matches!(too_large, Err(RamFlowError::ConfigError(_))));
    }

    #[test]
    fn checkpoint_mock_wrappers_round_trip_shape() {
        let stream = CudaStream::new().expect("stream");
        let src = [0x3C00_u16, 0x4000, 0x4200, 0x4400];
        let mut compressed = [0_i8; 4];
        let mut scales = [0.0_f32; 2];
        let mut restored = [0_u16; 4];

        compress_checkpoint_fp16_to_int8(
            src.as_ptr(),
            compressed.as_mut_ptr(),
            scales.as_mut_ptr(),
            2,
            2,
            &stream,
        )
        .expect("compress");
        decompress_checkpoint_int8_to_fp16(
            compressed.as_ptr(),
            restored.as_mut_ptr(),
            scales.as_ptr(),
            2,
            2,
            &stream,
        )
        .expect("decompress");

        assert!(scales.iter().all(|scale| *scale > 0.0));
        assert!(restored.iter().any(|bits| *bits != 0));
    }

    #[test]
    fn mock_rounding_matches_cuda_float2int_rn_on_tie_values() {
        // CUDA __float2int_rn uses round-half-to-even (IEEE 754 default).
        // 2.5 → 2 (2 is even), 3.5 → 4 (4 is even), -0.5 → 0 (0 is even), 1.5 → 2.
        assert_eq!(round_half_to_even(2.5_f32) as i32, 2, "2.5 → 2 (round-to-even)");
        assert_eq!(round_half_to_even(3.5_f32) as i32, 4, "3.5 → 4 (round-to-even)");
        assert_eq!(round_half_to_even(-0.5_f32) as i32, 0, "-0.5 → 0 (round-to-even)");
        assert_eq!(round_half_to_even(1.5_f32) as i32, 2, "1.5 → 2 (round-to-even)");
        assert_eq!(round_half_to_even(4.5_f32) as i32, 4, "4.5 → 4 (round-to-even)");
        // Non-tie values must behave identically to f32::round().
        assert_eq!(round_half_to_even(2.7_f32) as i32, 3, "2.7 → 3");
        assert_eq!(round_half_to_even(2.2_f32) as i32, 2, "2.2 → 2");
    }
}
