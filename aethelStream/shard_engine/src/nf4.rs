//! NF4 (4-bit normal float) dequantization.
//!
//! Matches bitsandbytes NF4 format exactly.

/// NF4 quantization codebook: 16 normalized values spanning [-1.0, 1.0].
pub const NF4_CODES: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Dequantize NF4-packed tensor into f16 output buffer.
///
/// # Arguments
///
/// - `packed`: Packed NF4 data (2 elements per byte, high nibble = even idx, low nibble = odd idx)
/// - `absmax`: Absolute max values per block (f32 slice)
/// - `output`: Pre-allocated f16 output buffer
/// - `block_size`: Number of elements per block
///
/// # Errors
///
/// Returns an error if absmax or packed are too small.
pub fn dequant_nf4_into(
    packed: &[u8],
    absmax: &[f32],
    output: &mut [half::f16],
    block_size: usize,
) -> crate::error::Result<()> {
    let num_elements = output.len();
    let num_blocks = num_elements.div_ceil(block_size);

    if absmax.len() < num_blocks {
        return Err(crate::error::ShardEngineError::MalformedIndex(
            "nf4_absmax".to_owned(),
            format!("absmax too small: {} < {}", absmax.len(), num_blocks),
        ));
    }
    if packed.len() * 2 < num_elements {
        return Err(crate::error::ShardEngineError::MalformedIndex(
            "nf4_packed".to_owned(),
            format!("packed too small: {} < {}", packed.len() * 2, num_elements),
        ));
    }

    for (block_idx, block_absmax) in absmax.iter().enumerate() {
        let block_start = block_idx * block_size;
        let block_end = (block_start + block_size).min(num_elements);
        let block_len = block_end - block_start;

        for elem_in_block in 0..block_len {
            let elem_idx = block_start + elem_in_block;
            let byte_idx = elem_idx / 2;
            let is_high = elem_idx.is_multiple_of(2);

            let byte_val = packed[byte_idx];
            let nibble = if is_high {
                (byte_val >> 4) & 0x0F
            } else {
                byte_val & 0x0F
            };

            let code_val = NF4_CODES[nibble as usize];
            let dequant_val = code_val * block_absmax;
            output[elem_idx] = half::f16::from_f32(dequant_val);
        }
    }

    Ok(())
}

/// Dequantize NF4-packed tensor and return allocated f16 vec.
///
/// # Arguments
///
/// - `packed`: Packed NF4 data
/// - `absmax`: Absolute max values per block
/// - `block_size`: Number of elements per block
///
/// Returns the number of elements as `packed.len() * 2`.
///
/// # Errors
///
/// Returns an error if absmax or packed are too small.
pub fn dequant_nf4_alloc(
    packed: &[u8],
    absmax: &[f32],
    block_size: usize,
) -> crate::error::Result<Vec<half::f16>> {
    let num_elements = packed.len() * 2;
    let mut output = vec![half::f16::ZERO; num_elements];
    dequant_nf4_into(packed, absmax, &mut output, block_size)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_at_index_7() {
        assert_eq!(NF4_CODES[7], 0.0);
    }

    #[test]
    fn test_roundtrip_accuracy() {
        // Pack some known nibbles: 0x7 (zero) and 0xF (1.0)
        let packed = [0x7F]; // high=7 (0.0), low=15 (1.0)
        let absmax = [1.0];
        let mut output = vec![half::f16::ZERO; 2];
        let _ = dequant_nf4_into(&packed, &absmax, &mut output, 2);

        let zero = output[0].to_f32();
        let one = output[1].to_f32();
        assert!(zero.abs() < 1e-6, "Expected ~0.0, got {}", zero);
        assert!((one - 1.0).abs() < 1e-5, "Expected ~1.0, got {}", one);
    }

    #[test]
    fn test_partial_block_handling() {
        // Pack 3 elements, block_size=2 (so last element is alone in block 2)
        let packed = [0x7F, 0x70]; // elements: 0.0, 1.0, 0.0
        let absmax = [1.0, 1.0]; // block 0 has 2 elems, block 1 has 1 elem
        let mut output = vec![half::f16::ZERO; 3];
        let _ = dequant_nf4_into(&packed, &absmax, &mut output, 2);

        assert!(output[0].to_f32().abs() < 1e-6);
        assert!((output[1].to_f32() - 1.0).abs() < 1e-5);
        assert!(output[2].to_f32().abs() < 1e-6);
    }
}
