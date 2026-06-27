//! Memory-mapped optimizer state file layout (Algorithm 4).
//!
//! File structure:
//! ```text
//! Header (64 bytes):
//!   magic, version, num_layers, default_rank, precision, header_size, total_size,
//!   beta1, beta2, eps, step_count, reserved
//! Layer descriptor table (num_layers × 20 bytes):
//!   m (u32), n (u32), rank (u32), byte_offset (u64)
//! Per layer at fixed offset:
//!   [P FP16 m×r][Q FP16 n×r][momentum INT8 r×r][variance INT8 r×r][scale_m f32][scale_v f32]
//! ```

use crate::adamw::{AdamWConfig, LowRankAdamState};
use crate::error::{GaLoreError, Result};
use crate::project::{f16_bits_to_f32, f32_to_f16_bits};
use memmap2::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Magic bytes "GALR" (GaLore).
pub const MAGIC: u32 = 0x4741_4C52;
/// Current file format version (v2: per-layer rank + header hyperparams).
pub const VERSION: u32 = 2;
/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 64;
/// Bytes per layer descriptor entry (v2).
pub const LAYER_DESC_SIZE: usize = 20;

/// Precision metadata flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PrecisionMeta {
    /// Projection matrices stored as FP16; m/v as INT8.
    Fp16ProjInt8State = 0,
}

/// On-disk header (64 bytes, little-endian).
#[derive(Debug, Clone, Copy)]
pub struct OptimizerStateHeader {
    /// Magic number [`MAGIC`].
    pub magic: u32,
    /// Format version.
    pub version: u32,
    /// Number of layer entries.
    pub num_layers: u32,
    /// Default projection rank r (legacy; per-layer rank in descriptor table).
    pub rank: u32,
    /// Precision metadata ([`PrecisionMeta`] as u32).
    pub precision: u32,
    /// Header size (always 64).
    pub header_size: u32,
    /// Total file size in bytes.
    pub total_size: u64,
    /// AdamW β1 stored in header for checkpoint portability.
    pub beta1: f32,
    /// AdamW β2 stored in header for checkpoint portability.
    pub beta2: f32,
    /// AdamW ε stored in header for checkpoint portability.
    pub eps: f32,
    /// Training step count at last flush (for resume).
    pub step_count: u64,
}

impl OptimizerStateHeader {
    /// Encode header to 64 bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.num_layers.to_le_bytes());
        buf[12..16].copy_from_slice(&self.rank.to_le_bytes());
        buf[16..20].copy_from_slice(&self.precision.to_le_bytes());
        buf[20..24].copy_from_slice(&self.header_size.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_size.to_le_bytes());
        buf[32..36].copy_from_slice(&self.beta1.to_le_bytes());
        buf[36..40].copy_from_slice(&self.beta2.to_le_bytes());
        buf[40..44].copy_from_slice(&self.eps.to_le_bytes());
        buf[44..52].copy_from_slice(&self.step_count.to_le_bytes());
        buf
    }

    /// Decode header from first 64 bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(GaLoreError::StateFile("header too short".into()));
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().map_err(|_| GaLoreError::StateFile("magic".into()))?);
        if magic != MAGIC {
            return Err(GaLoreError::StateFile(format!("bad magic {magic:#x}")));
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().map_err(|_| GaLoreError::StateFile("version".into()))?);
        let beta1 = if version >= 2 {
            f32::from_le_bytes(buf[32..36].try_into().map_err(|_| GaLoreError::StateFile("beta1".into()))?)
        } else {
            AdamWConfig::default().beta1
        };
        let beta2 = if version >= 2 {
            f32::from_le_bytes(buf[36..40].try_into().map_err(|_| GaLoreError::StateFile("beta2".into()))?)
        } else {
            AdamWConfig::default().beta2
        };
        let eps = if version >= 2 {
            f32::from_le_bytes(buf[40..44].try_into().map_err(|_| GaLoreError::StateFile("eps".into()))?)
        } else {
            AdamWConfig::default().eps
        };
        let step_count = if version >= 2 {
            u64::from_le_bytes(buf[44..52].try_into().map_err(|_| GaLoreError::StateFile("step_count".into()))?)
        } else {
            0
        };
        Ok(Self {
            magic,
            version,
            num_layers: u32::from_le_bytes(buf[8..12].try_into().map_err(|_| GaLoreError::StateFile("num_layers".into()))?),
            rank: u32::from_le_bytes(buf[12..16].try_into().map_err(|_| GaLoreError::StateFile("rank".into()))?),
            precision: u32::from_le_bytes(buf[16..20].try_into().map_err(|_| GaLoreError::StateFile("precision".into()))?),
            header_size: u32::from_le_bytes(buf[20..24].try_into().map_err(|_| GaLoreError::StateFile("header_size".into()))?),
            total_size: u64::from_le_bytes(buf[24..32].try_into().map_err(|_| GaLoreError::StateFile("total_size".into()))?),
            beta1,
            beta2,
            eps,
            step_count,
        })
    }

    /// Write AdamW hyperparameters and step counter into the header.
    pub fn set_adam_hyperparams(&mut self, cfg: &AdamWConfig, step_count: u64) {
        self.beta1 = cfg.beta1;
        self.beta2 = cfg.beta2;
        self.eps = cfg.eps;
        self.step_count = step_count;
    }
}

/// Per-layer dimensions, rank, and file offset.
#[derive(Debug, Clone, Copy)]
pub struct LayerDescriptor {
    /// Gradient matrix rows m.
    pub m: u32,
    /// Gradient matrix cols n.
    pub n: u32,
    /// Per-layer projection rank.
    pub rank: u32,
    /// Byte offset from file start.
    pub byte_offset: u64,
}

impl LayerDescriptor {
    fn to_bytes(self) -> [u8; LAYER_DESC_SIZE] {
        let mut buf = [0u8; LAYER_DESC_SIZE];
        buf[0..4].copy_from_slice(&self.m.to_le_bytes());
        buf[4..8].copy_from_slice(&self.n.to_le_bytes());
        buf[8..12].copy_from_slice(&self.rank.to_le_bytes());
        buf[12..20].copy_from_slice(&self.byte_offset.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8], file_version: u32, default_rank: u32) -> Result<Self> {
        if buf.len() < LAYER_DESC_SIZE {
            // v1 compatibility: 16-byte descriptors without per-layer rank.
            if buf.len() >= 16 && file_version < 2 {
                return Ok(Self {
                    m: u32::from_le_bytes(buf[0..4].try_into().map_err(|_| GaLoreError::StateFile("m".into()))?),
                    n: u32::from_le_bytes(buf[4..8].try_into().map_err(|_| GaLoreError::StateFile("n".into()))?),
                    rank: default_rank,
                    byte_offset: u64::from_le_bytes(buf[8..16].try_into().map_err(|_| GaLoreError::StateFile("offset".into()))?),
                });
            }
            return Err(GaLoreError::StateFile("layer desc too short".into()));
        }
        Ok(Self {
            m: u32::from_le_bytes(buf[0..4].try_into().map_err(|_| GaLoreError::StateFile("m".into()))?),
            n: u32::from_le_bytes(buf[4..8].try_into().map_err(|_| GaLoreError::StateFile("n".into()))?),
            rank: u32::from_le_bytes(buf[8..12].try_into().map_err(|_| GaLoreError::StateFile("rank".into()))?),
            byte_offset: u64::from_le_bytes(buf[12..20].try_into().map_err(|_| GaLoreError::StateFile("offset".into()))?),
        })
    }
}

/// Compute byte size of one layer's on-disk section.
pub fn layer_state_size(m: u32, n: u32, r: u32) -> u64 {
    let m = u64::from(m);
    let n = u64::from(n);
    let r = u64::from(r);
    // P (m×r FP16) + Q (n×r FP16) + m (r×r INT8) + v (r×r INT8) + 2×f32 scales
    2 * m * r + 2 * n * r + 2 * r * r + 8
}

/// Layout offsets within a layer section (relative to layer byte_offset).
#[derive(Debug, Clone, Copy)]
pub struct LayerLayout {
    /// Byte offset of P matrix (FP16, m×r) within the layer section.
    pub p_offset: u64,
    /// Byte offset of Q matrix (FP16, n×r).
    pub q_offset: u64,
    /// Byte offset of momentum INT8 tensor (r×r).
    pub momentum_offset: u64,
    /// Byte offset of variance INT8 tensor (r×r).
    pub variance_offset: u64,
    /// Byte offset of momentum absmax scale (f32).
    pub scale_m_offset: u64,
    /// Byte offset of variance absmax scale (f32).
    pub scale_v_offset: u64,
    /// Total byte size of this layer section.
    pub section_size: u64,
}

/// Compute field offsets within a layer section.
pub fn layer_layout(m: u32, n: u32, r: u32) -> LayerLayout {
    let m64 = u64::from(m);
    let n64 = u64::from(n);
    let r64 = u64::from(r);
    let p_off = 0;
    let q_off = 2 * m64 * r64;
    let m_off = q_off + 2 * n64 * r64;
    let v_off = m_off + r64 * r64;
    let sm_off = v_off + r64 * r64;
    let sv_off = sm_off + 4;
    LayerLayout {
        p_offset: p_off,
        q_offset: q_off,
        momentum_offset: m_off,
        variance_offset: v_off,
        scale_m_offset: sm_off,
        scale_v_offset: sv_off,
        section_size: layer_state_size(m, n, r),
    }
}

/// Build layer descriptors with O(1) seek offsets.
///
/// `layer_ranks[i]` is the rank for layer `i`; must match `dims.len()`.
pub fn build_layer_descriptors(dims: &[(u32, u32)], layer_ranks: &[u32]) -> Vec<LayerDescriptor> {
    let table_start = (HEADER_SIZE + dims.len() * LAYER_DESC_SIZE) as u64;
    let mut offset = table_start;
    dims.iter()
        .zip(layer_ranks.iter())
        .map(|(&(m, n), &rank)| {
            let desc = LayerDescriptor {
                m,
                n,
                rank,
                byte_offset: offset,
            };
            offset += layer_state_size(m, n, rank);
            desc
        })
        .collect()
}

/// Total file size for given layers with per-layer ranks.
pub fn total_file_size(dims: &[(u32, u32)], layer_ranks: &[u32]) -> u64 {
    let descs = build_layer_descriptors(dims, layer_ranks);
    descs
        .last()
        .map(|d| d.byte_offset + layer_state_size(d.m, d.n, d.rank))
        .unwrap_or((HEADER_SIZE + dims.len() * LAYER_DESC_SIZE) as u64)
}

/// Memory-mapped optimizer state file.
pub struct OptimizerStateFile {
    path: PathBuf,
    header: OptimizerStateHeader,
    descriptors: Vec<LayerDescriptor>,
    /// Underlying file handle (must stay open for the mmap on Windows).
    _file: File,
    mmap: MmapMut,
}

impl OptimizerStateFile {
    /// Create a new optimizer_states.bin with per-layer ranks and AdamW hyperparams.
    pub fn create(
        path: impl AsRef<Path>,
        dims: &[(u32, u32)],
        layer_ranks: &[u32],
        adam: &AdamWConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let num_layers = dims.len() as u32;
        let default_rank = layer_ranks.first().copied().unwrap_or(16);
        let total = total_file_size(dims, layer_ranks);
        let descriptors = build_layer_descriptors(dims, layer_ranks);

        let header = OptimizerStateHeader {
            magic: MAGIC,
            version: VERSION,
            num_layers,
            rank: default_rank,
            precision: PrecisionMeta::Fp16ProjInt8State as u32,
            header_size: HEADER_SIZE as u32,
            total_size: total,
            beta1: adam.beta1,
            beta2: adam.beta2,
            eps: adam.eps,
            step_count: 0,
        };

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| GaLoreError::StateFile(e.to_string()))?;
        file.set_len(total)
            .map_err(|e| GaLoreError::StateFile(e.to_string()))?;

        let mut mmap = unsafe {
            MmapOptions::new()
                .len(total as usize)
                .map_mut(&file)
                .map_err(|e| GaLoreError::StateFile(e.to_string()))?
        };

        mmap[0..HEADER_SIZE].copy_from_slice(&header.to_bytes());
        let desc_base = HEADER_SIZE;
        for (i, desc) in descriptors.iter().enumerate() {
            let start = desc_base + i * LAYER_DESC_SIZE;
            mmap[start..start + LAYER_DESC_SIZE].copy_from_slice(&desc.to_bytes());
        }

        Ok(Self {
            path,
            header,
            descriptors,
            _file: file,
            mmap,
        })
    }

    /// Open an existing optimizer state file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| GaLoreError::StateFile(e.to_string()))?;
        let len = file.metadata().map_err(|e| GaLoreError::StateFile(e.to_string()))?.len() as usize;
        let mmap = unsafe {
            MmapOptions::new()
                .len(len)
                .map_mut(&file)
                .map_err(|e| GaLoreError::StateFile(e.to_string()))?
        };

        let header = OptimizerStateHeader::from_bytes(&mmap[0..HEADER_SIZE])?;
        let desc_stride = if header.version >= 2 {
            LAYER_DESC_SIZE
        } else {
            16
        };
        let mut descriptors = Vec::with_capacity(header.num_layers as usize);
        let desc_base = HEADER_SIZE;
        for i in 0..header.num_layers as usize {
            let start = desc_base + i * desc_stride;
            let end = start + desc_stride;
            descriptors.push(LayerDescriptor::from_bytes(
                &mmap[start..end],
                header.version,
                header.rank,
            )?);
        }

        Ok(Self {
            path,
            header,
            descriptors,
            _file: file,
            mmap,
        })
    }

    /// Per-layer rank at index.
    pub fn layer_rank(&self, layer_index: usize) -> Result<u32> {
        self.descriptors
            .get(layer_index)
            .map(|d| d.rank)
            .ok_or_else(|| GaLoreError::StateFile(format!("layer index {layer_index} out of range")))
    }

    /// O(1) byte offset for layer index.
    pub fn layer_byte_offset(&self, layer_index: usize) -> Result<u64> {
        self.descriptors
            .get(layer_index)
            .map(|d| d.byte_offset)
            .ok_or_else(|| GaLoreError::StateFile(format!("layer index {layer_index} out of range")))
    }

    /// Layer dimensions at index.
    pub fn layer_dims(&self, layer_index: usize) -> Result<(u32, u32)> {
        self.descriptors
            .get(layer_index)
            .map(|d| (d.m, d.n))
            .ok_or_else(|| GaLoreError::StateFile(format!("layer index {layer_index} out of range")))
    }

    /// Read P matrix (m×r FP16) as f32 vector.
    pub fn read_p_f32(&self, layer_index: usize) -> Result<Vec<f32>> {
        let desc = self.descriptors.get(layer_index).ok_or_else(|| {
            GaLoreError::StateFile(format!("layer index {layer_index} out of range"))
        })?;
        let (m, n, r) = (desc.m, desc.n, desc.rank);
        let layout = layer_layout(m, n, r);
        let base = desc.byte_offset + layout.p_offset;
        let count = (m * r) as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = base as usize + i * 2;
            let bits = u16::from_le_bytes([self.mmap[off], self.mmap[off + 1]]);
            out.push(f16_bits_to_f32(bits));
        }
        Ok(out)
    }

    /// Read Q matrix (n×r FP16) as f32 vector.
    pub fn read_q_f32(&self, layer_index: usize) -> Result<Vec<f32>> {
        let desc = self.descriptors.get(layer_index).ok_or_else(|| {
            GaLoreError::StateFile(format!("layer index {layer_index} out of range"))
        })?;
        let (m, n, r) = (desc.m, desc.n, desc.rank);
        let layout = layer_layout(m, n, r);
        let base = desc.byte_offset + layout.q_offset;
        let count = (n * r) as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let off = base as usize + i * 2;
            let bits = u16::from_le_bytes([self.mmap[off], self.mmap[off + 1]]);
            out.push(f16_bits_to_f32(bits));
        }
        Ok(out)
    }

    /// Load full layer state into a [`LowRankAdamState`].
    pub fn load_layer_state(&self, layer_index: usize) -> Result<LowRankAdamState> {
        let desc = self.descriptors.get(layer_index).ok_or_else(|| {
            GaLoreError::StateFile(format!("layer index {layer_index} out of range"))
        })?;
        let (m, n, r) = (desc.m, desc.n, desc.rank);
        let layout = layer_layout(m, n, r);
        let base = desc.byte_offset;
        let r_usize = r as usize;
        let rr = r_usize * r_usize;

        let mut state = LowRankAdamState::new(r_usize);

        let m_base = base + layout.momentum_offset;
        for i in 0..rr {
            state.momentum_i8[i] = self.mmap[(m_base + i as u64) as usize] as i8;
        }
        let v_base = base + layout.variance_offset;
        for i in 0..rr {
            state.variance_i8[i] = self.mmap[(v_base + i as u64) as usize] as i8;
        }

        let sm_off = (base + layout.scale_m_offset) as usize;
        state.scale_m = f32::from_le_bytes([
            self.mmap[sm_off],
            self.mmap[sm_off + 1],
            self.mmap[sm_off + 2],
            self.mmap[sm_off + 3],
        ]);
        let sv_off = (base + layout.scale_v_offset) as usize;
        state.scale_v = f32::from_le_bytes([
            self.mmap[sv_off],
            self.mmap[sv_off + 1],
            self.mmap[sv_off + 2],
            self.mmap[sv_off + 3],
        ]);

        state.dequantize_from_ram();
        let _ = (m, n);
        Ok(state)
    }

    /// Write P, Q, and 8-bit m/v state back to mmap at layer offset.
    pub fn write_layer_state(
        &mut self,
        layer_index: usize,
        p: &[f32],
        q: &[f32],
        state: &LowRankAdamState,
    ) -> Result<()> {
        let desc = self.descriptors.get(layer_index).ok_or_else(|| {
            GaLoreError::StateFile(format!("layer index {layer_index} out of range"))
        })?;
        let (m, n, r) = (desc.m, desc.n, desc.rank);
        let layout = layer_layout(m, n, r);
        let base = desc.byte_offset;

        let p_expected = (m * r) as usize;
        let q_expected = (n * r) as usize;
        if p.len() != p_expected || q.len() != q_expected {
            return Err(GaLoreError::Shape(format!(
                "P/Q size mismatch: P {} expected {}, Q {} expected {}",
                p.len(),
                p_expected,
                q.len(),
                q_expected
            )));
        }

        let p_base = base + layout.p_offset;
        for (i, &v) in p.iter().enumerate() {
            let bits = f32_to_f16_bits(v);
            let off = (p_base as usize) + i * 2;
            self.mmap[off..off + 2].copy_from_slice(&bits.to_le_bytes());
        }

        let q_base = base + layout.q_offset;
        for (i, &v) in q.iter().enumerate() {
            let bits = f32_to_f16_bits(v);
            let off = (q_base as usize) + i * 2;
            self.mmap[off..off + 2].copy_from_slice(&bits.to_le_bytes());
        }

        let m_base = base + layout.momentum_offset;
        for (i, &v) in state.momentum_i8.iter().enumerate() {
            self.mmap[(m_base as usize) + i] = v as u8;
        }
        let v_base = base + layout.variance_offset;
        for (i, &v) in state.variance_i8.iter().enumerate() {
            self.mmap[(v_base as usize) + i] = v as u8;
        }

        let sm_off = (base + layout.scale_m_offset) as usize;
        self.mmap[sm_off..sm_off + 4].copy_from_slice(&state.scale_m.to_le_bytes());
        let sv_off = (base + layout.scale_v_offset) as usize;
        self.mmap[sv_off..sv_off + 4].copy_from_slice(&state.scale_v.to_le_bytes());

        Ok(())
    }

    /// Update header hyperparams and step counter, then write to mmap.
    pub fn write_header(&mut self, adam: &AdamWConfig, step_count: u64) -> Result<()> {
        self.header.set_adam_hyperparams(adam, step_count);
        self.mmap[0..HEADER_SIZE].copy_from_slice(&self.header.to_bytes());
        Ok(())
    }

    /// File path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Header metadata.
    pub fn header(&self) -> &OptimizerStateHeader {
        &self.header
    }

    /// Default projection rank from header.
    pub fn default_rank(&self) -> u32 {
        self.header.rank
    }

    /// Resumed step count from header.
    pub fn step_count(&self) -> u64 {
        self.header.step_count
    }

    /// AdamW hyperparameters stored in header.
    pub fn adam_from_header(&self) -> AdamWConfig {
        AdamWConfig {
            beta1: self.header.beta1,
            beta2: self.header.beta2,
            eps: self.header.eps,
            ..AdamWConfig::default()
        }
    }

    /// Flush mmap to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.mmap
            .flush()
            .map_err(|e| GaLoreError::StateFile(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adamw::LowRankAdamState;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("galore_{name}_{nanos}.bin"))
    }

    #[test]
    fn layer_offset_is_o1() {
        let dims = vec![(512, 512), (256, 1024)];
        let ranks = vec![16u32, 32];
        let descs = build_layer_descriptors(&dims, &ranks);
        assert_eq!(descs[0].byte_offset, (HEADER_SIZE + 2 * LAYER_DESC_SIZE) as u64);
        assert_eq!(
            descs[1].byte_offset,
            descs[0].byte_offset + layer_state_size(512, 512, 16)
        );
    }

    #[test]
    fn header_stores_adam_hyperparams() {
        let path = temp_path("header");
        let dims = vec![(32, 32)];
        let ranks = vec![4u32];
        let adam = AdamWConfig {
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-6,
            ..AdamWConfig::default()
        };
        let file = OptimizerStateFile::create(&path, &dims, &ranks, &adam).expect("create");
        assert!((file.header().beta1 - 0.85).abs() < 1e-6);
        assert!((file.header().beta2 - 0.99).abs() < 1e-6);
        assert!((file.header().eps - 1e-6).abs() < 1e-8);
        drop(file);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn roundtrip_state_file() {
        let path = temp_path("state");
        let dims = vec![(64, 64)];
        let ranks = vec![4u32];
        let adam = AdamWConfig::default();
        let mut file = OptimizerStateFile::create(&path, &dims, &ranks, &adam).expect("create");
        let rank = 4u32;

        let p: Vec<f32> = (0..64 * rank).map(|i| 0.01 * (i as f32)).collect();
        let q: Vec<f32> = (0..64 * rank).map(|i| 0.02 * (i as f32)).collect();
        let mut state = LowRankAdamState::new(rank as usize);
        state.momentum[0] = 0.5;
        state.quantize_to_ram();
        let expected_m = state.momentum_i8.clone();
        let expected_scale = state.scale_m;

        file.write_layer_state(0, &p, &q, &state).expect("write");
        file.flush().expect("flush");

        let loaded = file.load_layer_state(0).expect("load");
        assert_eq!(loaded.momentum_i8, expected_m);
        assert!((loaded.scale_m - expected_scale).abs() < 1e-6);

        let p_read = file.read_p_f32(0).expect("read p");
        assert!((p_read[0] - p[0]).abs() < 1e-3);

        drop(file);
        let _ = std::fs::remove_file(path);
    }
}

