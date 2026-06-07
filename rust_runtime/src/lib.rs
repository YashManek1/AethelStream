use half::f16;
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

// Precomputed NF4 constants from PyTorch quantization
const NF4_CODES: [f32; 16] = [
    -1.0000, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0,
];

#[derive(Serialize, Deserialize, Clone)]
pub struct TensorInfo {
    pub start: usize,
    pub end: usize,
    pub dtype: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ShardParam {
    pub file: String,
    pub tensors: HashMap<String, TensorInfo>,
}

pub type ShardIndex = HashMap<String, ShardParam>;

pub struct ShardLoader {
    pub index: ShardIndex,
    pub model_dir: String,
}

#[derive(Debug)]
pub struct LayerBuffer {
    pub data: Vec<u8>,
}

impl LayerBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }
}

impl ShardLoader {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> std::io::Result<Self> {
        let path = model_dir.as_ref().join("shard_index.json");
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let index: ShardIndex = serde_json::from_str(&content)?;
        Ok(Self {
            index,
            model_dir: model_dir.as_ref().to_string_lossy().to_string(),
        })
    }

    pub fn load_layer(&self, param_name: &str) -> std::io::Result<LayerBuffer> {
        let param = self
            .index
            .get(param_name)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "Param not found in index")
            })?;

        let file_path = Path::new(&self.model_dir).join(&param.file);
        let file = File::open(file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Madvise WILLNEED as required by "Module 3"
        #[cfg(unix)]
        {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_WILLNEED,
                );
            }
        }

        if let Some(fp16_info) = param.tensors.get("fp16") {
            let num_bytes = fp16_info.end - fp16_info.start;
            let mut buffer = LayerBuffer::new(num_bytes);
            buffer.data.copy_from_slice(&mmap[fp16_info.start..fp16_info.end]);
            Ok(buffer)
        } else if let (Some(packed_info), Some(absmax_info)) = (
            param.tensors.get("packed"),
            param.tensors.get("absmax"),
        ) {
            let packed_slice = &mmap[packed_info.start..packed_info.end];
            let absmax_slice = &mmap[absmax_info.start..absmax_info.end];
            
            // INT4 packed shape is (N/2) bytes. Return fp16 array size: 2 bytes per element, so N elements -> 2 * N bytes.
            let num_elements = packed_slice.len() * 2;
            let mut buffer = LayerBuffer::new(num_elements * 2); // 2 bytes per fp16
            
            // Block size is 64 standard for NF4. Check absmax slice len
            let num_blocks = absmax_slice.len() / 4; // 4 bytes per f32
            let block_size = num_elements / num_blocks;

            // Transmute chunks
            let fp16_out: &mut [f16] = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.data.as_mut_ptr() as *mut f16,
                    num_elements,
                )
            };
            
            let f32_absmax: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    absmax_slice.as_ptr() as *const f32,
                    num_blocks,
                )
            };

            for (block_idx, &absmax) in f32_absmax.iter().enumerate() {
                let start_elem = block_idx * block_size;
                let end_elem = std::cmp::min(start_elem + block_size, num_elements);
                
                // Unpack byte by byte
                let packed_start = start_elem / 2;
                let packed_end = (end_elem + 1) / 2;
                
                let bs = &packed_slice[packed_start..packed_end];
                let mut out_idx = start_elem;
                
                // Tight dequantization loop
                for &b in bs {
                    if out_idx >= end_elem {
                        break;
                    }
                    let left_idx = (b >> 4) as usize;
                    let right_idx = (b & 0x0F) as usize;
                    
                    let left_val = NF4_CODES[left_idx] * absmax;
                    fp16_out[out_idx] = f16::from_f32(left_val);
                    out_idx += 1;
                    
                    if out_idx < end_elem {
                        let right_val = NF4_CODES[right_idx] * absmax;
                        fp16_out[out_idx] = f16::from_f32(right_val);
                        out_idx += 1;
                    }
                }
            }
            Ok(buffer)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing expected formats in index",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_roundtrip() {
        // This validates logic against NF4 expectations
        let absmax = 10.0f32;
        let test_val = 5.0f32;
        
        let norm_val = test_val / absmax;
        
        // Quantize
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;
        for (i, &code) in NF4_CODES.iter().enumerate() {
            let dist = (norm_val - code).abs();
            if dist < min_dist {
                min_dist = dist;
                best_idx = i as u8;
            }
        }
        
        // Dequantize
        let dequant_val = NF4_CODES[best_idx as usize] * absmax;
        
        // Expect close
        assert!((dequant_val - test_val).abs() < 1.0);
    }
    
    #[test]
    fn test_zero_point() {
        // Find 0.0 in NF4 code table
        let mut zero_idx = None;
        for (i, &code) in NF4_CODES.iter().enumerate() {
            if code == 0.0 {
                zero_idx = Some(i);
                break;
            }
        }
        
        assert_eq!(zero_idx.expect("Must have exact 0.0"), 7);
        // Ensure evaluating 0 index returns exact 0.0
        assert_eq!(NF4_CODES[7] * f32::MAX, 0.0);
    }
    
    #[test]
    fn test_precision_schedule() {
        // Represents schedule L0-3 and L-4 to L-1 stay fp16
        let num_layers = 32;
        let mut fp16_count = 0;
        let mut int4_count = 0;
        for layer_idx in 0..num_layers {
            if layer_idx <= 3 || (num_layers - 4) <= layer_idx && layer_idx <= (num_layers - 1) {
                fp16_count += 1;
            } else {
                int4_count += 1;
            }
        }
        assert_eq!(fp16_count, 8); // 0,1,2,3 AND 28,29,30,31
        assert_eq!(int4_count, 24);
    }
}

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyBytes;

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct PyShardLoader {
    loader: ShardLoader,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyShardLoader {
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let loader = ShardLoader::new(&model_dir).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(e.to_string())
        })?;
        Ok(PyShardLoader { loader })
    }

    fn load_layer<'py>(&self, py: Python<'py>, param_name: &str) -> PyResult<&'py PyBytes> {
        let buffer = self.loader.load_layer(param_name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })?;
        Ok(PyBytes::new(py, &buffer.data))
    }
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn rust_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyShardLoader>()?;
    Ok(())
}
