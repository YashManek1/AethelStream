use crate::error::Result;
use crate::index::IndexStore;
use std::collections::HashMap;
use std::path::Path;

/// Raw bytes of a single tensor (always FP16 after dequantization).
#[derive(Clone, Debug)]
pub struct TensorBuffer {
    /// FP16 bytes (2 bytes per element).
    pub data: Vec<u8>,

    /// Shape of the tensor.
    pub shape: Vec<usize>,

    /// Data type (always "F16").
    pub dtype: String,

    /// Parameter name.
    pub param_name: String,
}

/// Raw bytes of an entire layer shard file (safetensors format).
#[derive(Clone, Debug)]
pub struct LayerBuffer {
    /// Raw safetensors bytes.
    pub data: Vec<u8>,

    /// Layer index.
    pub layer_index: u32,

    /// File path.
    pub file_path: String,
}

/// Shard loader with lazy mmap caching.
pub struct ShardLoader {
    /// Index store containing tensor and layer metadata.
    pub store: IndexStore,
    mmap_cache: HashMap<String, memmap2::Mmap>,
}

impl ShardLoader {
    /// Create a new shard loader for the given model directory.
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let store = IndexStore::load(model_dir)?;
        Ok(ShardLoader {
            store,
            mmap_cache: HashMap::new(),
        })
    }

    /// Load an entire layer shard file as raw safetensors bytes.
    pub fn load_layer(&mut self, layer_index: u32) -> Result<LayerBuffer> {
        let filename = self.store.shard_file_for_layer(layer_index)?;
        let filename_owned = filename.to_owned();

        // Ensure mmap is cached
        self.get_or_open_mmap(&filename_owned)?;

        let mmap = self
            .mmap_cache
            .get(&filename_owned)
            .ok_or(crate::error::ShardEngineError::LayerNotFound(layer_index))?;

        Ok(LayerBuffer {
            data: mmap.to_vec(),
            layer_index,
            file_path: self
                .store
                .model_dir
                .join(&filename_owned)
                .to_string_lossy()
                .to_string(),
        })
    }

    /// Load a single parameter tensor, dequantizing NF4 on the fly if needed.
    pub fn load_param(&mut self, param_name: &str) -> Result<TensorBuffer> {
        // Clone info to avoid borrow conflicts
        let info = self.store.tensor_info(param_name)?.clone();
        let filename = info.file_path.clone();

        // Ensure mmap is cached
        self.get_or_open_mmap(&filename)?;

        let mmap =
            self.mmap_cache
                .get(&filename)
                .ok_or(crate::error::ShardEngineError::ParamNotFound(
                    param_name.to_owned(),
                ))?;

        let data = if info.precision == "nf4" {
            // NF4 dequantization path
            let packed_offset = info.byte_offset;
            let packed_length = info.byte_length;

            // Bounds check before slicing
            if packed_offset + packed_length > mmap.len() {
                return Err(crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    format!(
                        "packed data out of bounds: offset {} + length {} > mmap {}",
                        packed_offset,
                        packed_length,
                        mmap.len()
                    ),
                ));
            }

            let packed = &mmap[packed_offset..packed_offset + packed_length];

            let absmax_offset = info.nf4_absmax_offset.ok_or_else(|| {
                crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    "missing nf4_absmax_offset".to_owned(),
                )
            })?;
            let absmax_length = info.nf4_absmax_length.ok_or_else(|| {
                crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    "missing nf4_absmax_length".to_owned(),
                )
            })?;
            let block_size = info.nf4_block_size.ok_or_else(|| {
                crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    "missing nf4_block_size".to_owned(),
                )
            })?;

            // Bounds check for absmax
            if absmax_offset + absmax_length > mmap.len() {
                return Err(crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    format!(
                        "absmax data out of bounds: offset {} + length {} > mmap {}",
                        absmax_offset,
                        absmax_length,
                        mmap.len()
                    ),
                ));
            }

            // Check alignment: absmax must be 4-byte aligned for f32 access
            if absmax_offset % std::mem::align_of::<f32>() != 0 {
                return Err(crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    format!("absmax offset {} is not 4-byte aligned", absmax_offset),
                ));
            }

            // SAFETY: We have verified:
            // 1. absmax_offset + absmax_length <= mmap.len() (bounds check above)
            // 2. absmax_offset is 4-byte aligned for f32 access (alignment check above)
            // 3. absmax_length is validated by dequant_nf4_alloc to be correct size
            // Therefore, creating a slice of f32 from these bytes is sound.
            let absmax_ptr =
                mmap[absmax_offset..absmax_offset + absmax_length].as_ptr() as *const f32;
            let absmax_count = absmax_length / std::mem::size_of::<f32>();
            let absmax = unsafe { std::slice::from_raw_parts(absmax_ptr, absmax_count) };

            // Dequantize to f16
            let dequant = crate::nf4::dequant_nf4_alloc(packed, absmax, block_size)?;

            // Reinterpret Vec<f16> as Vec<u8>
            let mut f16_vec = dequant;
            let ptr = f16_vec.as_mut_ptr() as *mut u8;
            let len = f16_vec.len() * std::mem::size_of::<half::f16>();
            std::mem::forget(f16_vec);
            // SAFETY: We are converting Vec<f16> to Vec<u8> by reinterpreting the memory.
            // This is sound because:
            // 1. f16 is a 2-byte type, u8 is 1 byte
            // 2. The allocation and capacity are correct: len * 2 bytes becomes len * 2 u8s
            // 3. We forget the original vector to prevent double-free
            // 4. The new Vec owns the same heap allocation
            // 5. f16 has no drop semantics that need to run (it's just bits)
            unsafe { Vec::from_raw_parts(ptr, len, len) }
        } else {
            // FP16 or other precision: direct copy
            let offset = info.byte_offset;
            let length = info.byte_length;

            // Bounds check before slicing
            if offset + length > mmap.len() {
                return Err(crate::error::ShardEngineError::MalformedIndex(
                    param_name.to_owned(),
                    format!(
                        "tensor data out of bounds: offset {} + length {} > mmap {}",
                        offset,
                        length,
                        mmap.len()
                    ),
                ));
            }

            mmap[offset..offset + length].to_vec()
        };

        Ok(TensorBuffer {
            data,
            shape: info.shape,
            dtype: "F16".to_owned(),
            param_name: param_name.to_owned(),
        })
    }

    /// Get or open an mmap for the given filename (cached).
    fn get_or_open_mmap(&mut self, filename: &str) -> Result<()> {
        if !self.mmap_cache.contains_key(filename) {
            let path = self.store.model_dir.join(filename);
            let file = std::fs::File::open(&path)?;
            // SAFETY: MmapOptions::map ensures the mmap is valid for the duration of the
            // Mmap object's lifetime. We keep the Mmap in the cache, so the file must remain
            // open. memmap2 holds a handle to the file, so this is safe.
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            self.mmap_cache.insert(filename.to_owned(), mmap);
        }
        Ok(())
    }
}
