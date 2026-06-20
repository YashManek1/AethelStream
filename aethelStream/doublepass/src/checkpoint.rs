//! A2/A10 checkpoint store/read via M2 `PinnedBuffer`.
//!
//! Uncompressed path: raw f32 little-endian bytes stored verbatim.
//! Compressed path: f32 → fp16 → INT8 via `ramflow::kernels::compress_checkpoint_fp16_to_int8`,
//! with one channel covering the whole activation tensor.
//!
//! Buffer layout when compressed (`n_channels = 1`):
//! ```text
//! [ f32 scale : 4 bytes ] [ i8 × n_elements bytes ]
//! ```

use crate::error::DoublePassError;
use crate::Result;
use ramflow::PinnedBuffer;

/// Store an activation tensor into a freshly allocated [`PinnedBuffer`].
///
/// When `compress` is `true`, the buffer is stored in packed INT8 format
/// (one f32 scale followed by n INT8 values) and `set_compressed(true)` is called.
/// When `compress` is `false`, raw f32 LE bytes are stored.
pub fn store_checkpoint(data: &[f32], compress: bool) -> Result<PinnedBuffer> {
    if compress {
        store_compressed(data)
    } else {
        store_uncompressed(data)
    }
}

/// Read a checkpoint back into f32 activations.
///
/// Automatically detects compression via [`PinnedBuffer::is_compressed`].
pub fn read_checkpoint(buf: &PinnedBuffer) -> Result<Vec<f32>> {
    if buf.is_compressed() {
        read_compressed(buf)
    } else {
        read_uncompressed(buf)
    }
}

// ── Uncompressed path ────────────────────────────────────────────────────────

fn store_uncompressed(data: &[f32]) -> Result<PinnedBuffer> {
    let byte_len = data
        .len()
        .checked_mul(4)
        .ok_or_else(|| DoublePassError::Checkpoint("activation too large".into()))?;
    let mut buf = PinnedBuffer::alloc(byte_len)?;
    let dst = buf.as_mut_slice();
    for (i, &v) in data.iter().enumerate() {
        dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    Ok(buf)
}

fn read_uncompressed(buf: &PinnedBuffer) -> Result<Vec<f32>> {
    let raw = buf.as_slice();
    if !raw.len().is_multiple_of(4) {
        return Err(DoublePassError::Checkpoint(
            "uncompressed checkpoint length not a multiple of 4".into(),
        ));
    }
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0u8; 4];
        b.copy_from_slice(&raw[i * 4..i * 4 + 4]);
        out.push(f32::from_le_bytes(b));
    }
    Ok(out)
}

// ── Compressed path ──────────────────────────────────────────────────────────
// Layout: [ f32 scale (4 bytes) ] [ i8 × n_elements ]
// n_channels = 1 (whole activation treated as one channel).

fn store_compressed(data: &[f32]) -> Result<PinnedBuffer> {
    let n = data.len();
    if n == 0 {
        return Err(DoublePassError::Checkpoint("cannot compress empty activation".into()));
    }

    // Convert f32 → fp16 (u16) for the compress kernel's source.
    let fp16: Vec<u16> = data.iter().map(|&v| f32_to_fp16(v)).collect();

    // Compressed buffer: 4 bytes (1 scale) + n bytes (INT8 data).
    let compressed_size = core::mem::size_of::<f32>() + n;
    let mut buf = PinnedBuffer::alloc(compressed_size)?;

    // Safety: PinnedBuffer::alloc guarantees 64-byte alignment (posix_memalign).
    // split_compressed_buffer_ptrs returns scales at base (f32-aligned) and
    // INT8 data after the scale region (i8-aligned = always ok).
    let (scales_ptr, data_ptr) = unsafe {
        ramflow::kernels::split_compressed_buffer_ptrs(buf.as_mut_ptr(), 1)
            .map_err(|e| DoublePassError::Checkpoint(e.to_string()))?
    };

    let stream = ramflow::cuda_bridge::CudaStream::new()?;
    ramflow::kernels::compress_checkpoint_fp16_to_int8(
        fp16.as_ptr(),
        data_ptr,
        scales_ptr,
        1, // n_channels
        n, // elems_per_channel
        &stream,
    )?;

    buf.set_compressed(true);
    Ok(buf)
}

fn read_compressed(buf: &PinnedBuffer) -> Result<Vec<f32>> {
    let total = buf.len();
    let scale_bytes = core::mem::size_of::<f32>();
    if total <= scale_bytes {
        return Err(DoublePassError::Checkpoint(
            "compressed checkpoint buffer too small".into(),
        ));
    }
    let n = total - scale_bytes; // n_elements

    // Layout: base[0..4] = f32 scale, base[4..4+n] = i8 data.
    // Safety: PinnedBuffer is 64-byte aligned; f32 needs 4-byte alignment (ok);
    //         i8 needs 1-byte alignment (ok). We cast *const u8 to typed pointers.
    let base = buf.as_ptr();
    let scales_ptr = base as *const f32;
    // SAFETY: base + 4 is within the allocation (total > scale_bytes).
    let data_ptr = unsafe { base.add(scale_bytes) } as *const i8;

    let mut fp16_out: Vec<u16> = vec![0u16; n];

    let stream = ramflow::cuda_bridge::CudaStream::new()?;
    ramflow::kernels::decompress_checkpoint_int8_to_fp16(
        data_ptr,
        fp16_out.as_mut_ptr(),
        scales_ptr,
        1, // n_channels
        n, // elems_per_channel
        &stream,
    )?;

    Ok(fp16_out.iter().map(|&bits| fp16_to_f32(bits)).collect())
}

// ── fp16 conversion helpers ───────────────────────────────────────────────────

/// f32 → IEEE 754 fp16 bit pattern. Nearest-even rounding (approximate).
#[inline]
fn f32_to_fp16(v: f32) -> u16 {
    if v == 0.0 {
        return if v.is_sign_negative() { 0x8000 } else { 0 };
    }
    if v.is_nan() {
        return 0x7e00; // quiet NaN
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if v.is_infinite() {
        return sign | 0x7c00;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00; // overflow → inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow → ±0
        }
        let man = ((bits & 0x007f_ffff) | 0x0080_0000) >> (14 - exp);
        return sign | (man as u16);
    }
    let man = (bits & 0x007f_ffff) >> 13;
    sign | ((exp as u16) << 10) | (man as u16)
}

/// IEEE 754 fp16 bit pattern → f32.
#[inline]
fn fp16_to_f32(bits: u16) -> f32 {
    let sign: f32 = if (bits & 0x8000) != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x03ff) as u32;
    if exp == 0 {
        return sign * (man as f32) * (2.0f32.powi(-24));
    }
    if exp == 31 {
        return if man == 0 { sign * f32::INFINITY } else { f32::NAN };
    }
    sign * (1.0 + man as f32 / 1024.0) * 2.0f32.powi(exp - 15)
}
