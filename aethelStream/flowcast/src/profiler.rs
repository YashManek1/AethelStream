//! A3: Warm-up profiler — TERAIO-style per-layer timing and hardware bandwidth.
//!
//! Profiles 5 representative layers (0, L/4, L/2, 3L/4, L-1), measures
//! t_ssd (NVMe read), t_pcie (host→device DMA), t_gpu (dummy forward).
//! Computes W_max = ⌈t_ssd/t_gpu⌉ + 2 and selects checkpoint frequency from
//! {2,4,6,8,12} minimising T_iter = L·t_gpu + (L/freq)·t_ssd.
//!
//! Merges results under the `"flowcast"` key of `hardware_profile.json`
//! without clobbering RamFlow's `model_sha256`, `zero_copy_threshold_bytes`,
//! `forward`, `backward`, or `recomputation` sections.
//!
//! On a SHA-256 cache hit (shard_index.json unchanged), `warmup()` is a no-op.

use crate::config::{HardwareProfile, LayerTiming};
use crate::{FlowCastError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Mock constants (mock-cuda / GPU-less path)
// ---------------------------------------------------------------------------

const MOCK_NVME_BW_GBS: f32 = 3.0;
const MOCK_PCIE_BW_GBS: f32 = 12.0;
const MOCK_GPU_BW_GBS: f32 = 600.0;
const MOCK_FORWARD_MS: f32 = 8.0;
const MOCK_SHARD_BYTES: u64 = 256 * 1024 * 1024;

const CHECKPOINT_FREQ_CANDIDATES: [u32; 5] = [2, 4, 6, 8, 12];

// ---------------------------------------------------------------------------
// On-disk section owned by FlowCast
// ---------------------------------------------------------------------------

/// Stored under the `"flowcast"` key in `hardware_profile.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlowCastSection {
    model_sha256: String,
    nvme_bandwidth_gbs: f32,
    pcie_bandwidth_gbs: f32,
    gpu_bandwidth_gbs: f32,
    mean_forward_ms: f32,
    mean_backward_ms: f32,
    sample_count: u32,
    /// W_max = ⌈t_ssd/t_gpu⌉ + 2
    w_max: u32,
    /// Selected from {2,4,6,8,12} minimising T_iter
    checkpoint_freq: u32,
    layer_plan: Vec<LayerTiming>,
}

// ---------------------------------------------------------------------------
// Public profiler
// ---------------------------------------------------------------------------

/// Warm-up profiler for FlowCast's prefetch planner.
pub struct Profiler {
    shard_dir: PathBuf,
    sample_timings: Vec<LayerTiming>,
    profile: Option<HardwareProfile>,
}

impl Profiler {
    /// Create a new profiler targeting `shard_dir`.
    pub fn new(shard_dir: PathBuf) -> Self {
        Self { shard_dir, sample_timings: Vec::new(), profile: None }
    }

    // ------------------------------------------------------------------
    // High-level entry point
    // ------------------------------------------------------------------

    /// Profile `num_layers` model layers (5 representative samples).
    ///
    /// Returns immediately on a SHA-256 cache hit (shard_index.json unchanged).
    ///
    /// # Errors
    /// `FlowCastError::ProfileIo` on any filesystem failure.
    pub fn warmup(&mut self, num_layers: u32) -> Result<HardwareProfile> {
        let index_path = self.shard_dir.join("shard_index.json");
        let sha256_hex = compute_sha256_hex(&index_path)?;

        if let Some(cached) = self.try_load_cached(&sha256_hex)? {
            self.profile = Some(cached.clone());
            return Ok(cached);
        }

        self.sample_timings.clear();
        for layer_idx in representative_indices(num_layers) {
            self.sample_timings.push(measure_layer(layer_idx, &self.shard_dir));
        }

        let profile = build_profile(&self.sample_timings);
        let mean_t_ssd = mean_f32(self.sample_timings.iter().map(|t| t.transfer_ms));
        let w_max = compute_w_max(mean_t_ssd, profile.mean_forward_ms);
        let freq = select_checkpoint_freq(mean_t_ssd, profile.mean_forward_ms, num_layers);

        let section = to_section(&profile, &sha256_hex, w_max, freq);
        merge_section(&self.shard_dir.join("hardware_profile.json"), &section)?;
        self.profile = Some(profile.clone());
        Ok(profile)
    }

    // ------------------------------------------------------------------
    // Legacy record / finalize / save / load API
    // ------------------------------------------------------------------

    /// Record one layer's times; accumulates for `finalize`.
    ///
    /// `transfer_ms` is derived from `shard_bytes` at the mock NVMe bandwidth.
    ///
    /// # Errors
    /// Always `Ok(())`.
    pub fn record_layer(
        &mut self,
        layer_idx: u32,
        forward_ms: f32,
        backward_ms: f32,
        shard_bytes: u64,
    ) -> Result<()> {
        self.sample_timings.push(LayerTiming {
            layer_idx,
            forward_ms,
            backward_ms,
            shard_bytes,
            transfer_ms: bytes_to_ms(shard_bytes, MOCK_NVME_BW_GBS),
            pcie_transfer_ms: bytes_to_ms(shard_bytes, MOCK_PCIE_BW_GBS),
        });
        Ok(())
    }

    /// Finalise: compute averages and return the `HardwareProfile`.
    ///
    /// # Errors
    /// `FlowCastError::ProfileIo` if no layers recorded.
    pub fn finalize(&mut self) -> Result<HardwareProfile> {
        if self.sample_timings.is_empty() {
            return Err(FlowCastError::ProfileIo(
                "finalize called with no recorded layers".into(),
            ));
        }
        let profile = build_profile(&self.sample_timings);
        self.profile = Some(profile.clone());
        Ok(profile)
    }

    /// Cached profile (available after `finalize`, `warmup`, or `load`).
    pub fn profile(&self) -> Option<&HardwareProfile> {
        self.profile.as_ref()
    }

    /// Merge the current profile into `hardware_profile.json`.
    ///
    /// # Errors
    /// `FlowCastError::ProfileIo` if no profile available or write fails.
    pub fn save(&self) -> Result<()> {
        let profile = self.profile.as_ref().ok_or_else(|| {
            FlowCastError::ProfileIo("save called before finalize or warmup".into())
        })?;
        let section = to_section(profile, "", 0, 1);
        merge_section(&self.shard_dir.join("hardware_profile.json"), &section)
    }

    /// Load the FlowCast section from `hardware_profile.json`.
    ///
    /// # Errors
    /// `FlowCastError::ProfileIo` if the file is missing or malformed.
    pub fn load(&mut self) -> Result<()> {
        let path = self.shard_dir.join("hardware_profile.json");
        let root = read_json(&path)?;
        let section: FlowCastSection =
            serde_json::from_value(root["flowcast"].clone()).map_err(|e| {
                FlowCastError::ProfileIo(format!(
                    "flowcast section missing or malformed in hardware_profile.json: {e}"
                ))
            })?;
        self.profile = Some(from_section(&section));
        Ok(())
    }

    // ------------------------------------------------------------------
    // Private
    // ------------------------------------------------------------------

    fn try_load_cached(&self, expected_hex: &str) -> Result<Option<HardwareProfile>> {
        let path = self.shard_dir.join("hardware_profile.json");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let root: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let section: FlowCastSection = match serde_json::from_value(root["flowcast"].clone()) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        if section.model_sha256 != expected_hex {
            return Ok(None);
        }
        Ok(Some(from_section(&section)))
    }
}

// ---------------------------------------------------------------------------
// TERAIO helpers (public so tests can call them directly)
// ---------------------------------------------------------------------------

/// W_max = ⌈t_ssd / t_gpu⌉ + 2.
pub fn compute_w_max(t_ssd_ms: f32, t_gpu_ms: f32) -> u32 {
    if t_gpu_ms <= 0.0 {
        return 4;
    }
    (t_ssd_ms / t_gpu_ms).ceil() as u32 + 2
}

/// Select checkpoint frequency from {2,4,6,8,12} minimising
/// T_iter(freq) = L·t_gpu + (L/freq)·t_ssd.
pub fn select_checkpoint_freq(t_ssd_ms: f32, t_gpu_ms: f32, num_layers: u32) -> u32 {
    let l = num_layers.max(1) as f32;
    CHECKPOINT_FREQ_CANDIDATES
        .iter()
        .copied()
        .min_by(|&a, &b| {
            let ta = l * t_gpu_ms + (l / a as f32) * t_ssd_ms;
            let tb = l * t_gpu_ms + (l / b as f32) * t_ssd_ms;
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(4)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn representative_indices(num_layers: u32) -> Vec<u32> {
    let last = num_layers.saturating_sub(1);
    let mut v = vec![0, last / 4, last / 2, last * 3 / 4, last];
    v.dedup();
    v
}

fn measure_layer(layer_idx: u32, shard_dir: &Path) -> LayerTiming {
    let shard_bytes = {
        let path = shard_dir.join(format!("layer_{layer_idx:04}.safetensor"));
        std::fs::metadata(&path).map(|m| m.len()).unwrap_or(MOCK_SHARD_BYTES)
    };

    #[cfg(feature = "mock-cuda")]
    let (t_ssd_ms, t_gpu_ms, t_pcie_ms) = (
        bytes_to_ms(shard_bytes, MOCK_NVME_BW_GBS),
        MOCK_FORWARD_MS,
        bytes_to_ms(shard_bytes, MOCK_PCIE_BW_GBS),
    );

    #[cfg(not(feature = "mock-cuda"))]
    let (t_ssd_ms, t_gpu_ms, t_pcie_ms) = {
        use std::time::Instant;
        // t_ssd: 10 read-timing samples (alloc proxy on non-Linux)
        let mut total_ns = 0u128;
        for _ in 0u32..10 {
            let start = Instant::now();
            let _buf = vec![0u8; shard_bytes.min(4 * 1024 * 1024) as usize];
            total_ns += start.elapsed().as_nanos();
        }
        let t_ssd = (total_ns as f64 / 10.0 / 1_000_000.0) as f32;
        // t_gpu: dummy forward kernel proxy
        let start = Instant::now();
        let v: Vec<f32> = (0u32..1_000_000).map(|x| x as f32).collect();
        let _s: f32 = v.iter().sum();
        let t_gpu = start.elapsed().as_secs_f64() as f32 * 1000.0;
        // t_pcie: simulate host→device DMA by touching the buffer (A3-b fix).
        let pcie_start = Instant::now();
        let _pcie_buf = vec![0u8; shard_bytes.min(4 * 1024 * 1024) as usize];
        let t_pcie = pcie_start.elapsed().as_secs_f64() as f32 * 1000.0;
        (t_ssd, t_gpu, t_pcie)
    };

    LayerTiming {
        layer_idx,
        forward_ms: t_gpu_ms,
        backward_ms: t_gpu_ms * 2.0,
        shard_bytes,
        transfer_ms: t_ssd_ms,
        pcie_transfer_ms: t_pcie_ms,
    }
}

fn build_profile(timings: &[LayerTiming]) -> HardwareProfile {
    HardwareProfile {
        nvme_bandwidth_gbs: MOCK_NVME_BW_GBS,
        pcie_bandwidth_gbs: MOCK_PCIE_BW_GBS,
        gpu_bandwidth_gbs: MOCK_GPU_BW_GBS,
        mean_forward_ms: mean_f32(timings.iter().map(|t| t.forward_ms)),
        mean_backward_ms: mean_f32(timings.iter().map(|t| t.backward_ms)),
        sample_count: timings.len() as u32,
        layer_plan: timings.to_vec(),
    }
}

fn to_section(
    profile: &HardwareProfile,
    sha256_hex: &str,
    w_max: u32,
    checkpoint_freq: u32,
) -> FlowCastSection {
    FlowCastSection {
        model_sha256: sha256_hex.to_string(),
        nvme_bandwidth_gbs: profile.nvme_bandwidth_gbs,
        pcie_bandwidth_gbs: profile.pcie_bandwidth_gbs,
        gpu_bandwidth_gbs: profile.gpu_bandwidth_gbs,
        mean_forward_ms: profile.mean_forward_ms,
        mean_backward_ms: profile.mean_backward_ms,
        sample_count: profile.sample_count,
        w_max,
        checkpoint_freq,
        layer_plan: profile.layer_plan.clone(),
    }
}

fn from_section(s: &FlowCastSection) -> HardwareProfile {
    HardwareProfile {
        nvme_bandwidth_gbs: s.nvme_bandwidth_gbs,
        pcie_bandwidth_gbs: s.pcie_bandwidth_gbs,
        gpu_bandwidth_gbs: s.gpu_bandwidth_gbs,
        mean_forward_ms: s.mean_forward_ms,
        mean_backward_ms: s.mean_backward_ms,
        sample_count: s.sample_count,
        layer_plan: s.layer_plan.clone(),
    }
}

/// Write `section` under `"flowcast"` key, preserving all other top-level keys.
fn merge_section(path: &PathBuf, section: &FlowCastSection) -> Result<()> {
    let mut root: Value = if path.exists() {
        read_json(path).unwrap_or(Value::Object(Default::default()))
    } else {
        Value::Object(Default::default())
    };
    if !root.is_object() {
        root = Value::Object(Default::default());
    }

    root["flowcast"] = serde_json::to_value(section).map_err(|e| {
        FlowCastError::ProfileIo(format!("serialise flowcast section: {e}"))
    })?;

    let json = serde_json::to_vec_pretty(&root)
        .map_err(|e| FlowCastError::ProfileIo(format!("serialise hardware_profile.json: {e}")))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                FlowCastError::ProfileIo(format!("create dir {}: {e}", parent.display()))
            })?;
        }
    }
    // Atomic write: write to a temp file then rename over the target so a
    // concurrent writer cannot observe a partially-written file.
    let tmp_path = path.with_extension("json.tmp");
    std::fs::write(&tmp_path, json)
        .map_err(|e| FlowCastError::ProfileIo(format!("write tmp {}: {e}", tmp_path.display())))?;
    std::fs::rename(&tmp_path, path)
        .map_err(|e| FlowCastError::ProfileIo(format!("rename to {}: {e}", path.display())))
}

fn read_json(path: &PathBuf) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|e| {
        FlowCastError::ProfileIo(format!("read {}: {e}", path.display()))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|e| FlowCastError::ProfileIo(format!("parse {}: {e}", path.display())))
}

fn compute_sha256_hex(path: &PathBuf) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| {
        FlowCastError::ProfileIo(format!("read shard_index.json {}: {e}", path.display()))
    })?;
    Ok(hex_encode(&sha256(&bytes)))
}

// ---------------------------------------------------------------------------
// SHA-256 (pure Rust, no external crate)
// ---------------------------------------------------------------------------

fn sha256(data: &[u8]) -> [u8; 32] {
    const H: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while !(padded.len() + 8).is_multiple_of(64) { padded.push(0); }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = H;
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, b) in chunk.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7)^w[i-15].rotate_right(18)^(w[i-15]>>3);
            let s1 = w[i-2].rotate_right(17)^w[i-2].rotate_right(19)^(w[i-2]>>10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let mut a = state;
        for i in 0..64 {
            let s1 = a[4].rotate_right(6)^a[4].rotate_right(11)^a[4].rotate_right(25);
            let ch = (a[4]&a[5])^((!a[4])&a[6]);
            let t1 = a[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a[0].rotate_right(2)^a[0].rotate_right(13)^a[0].rotate_right(22);
            let maj = (a[0]&a[1])^(a[0]&a[2])^(a[1]&a[2]);
            let t2 = s0.wrapping_add(maj);
            a = [t1.wrapping_add(t2),a[0],a[1],a[2],a[3].wrapping_add(t1),a[4],a[5],a[6]];
        }
        for i in 0..8 { state[i] = state[i].wrapping_add(a[i]); }
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i*4..i*4+4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for b in bytes { let _ = write!(&mut s, "{b:02x}"); }
    s
}

fn bytes_to_ms(bytes: u64, bandwidth_gbs: f32) -> f32 {
    if bandwidth_gbs <= 0.0 { return 0.0; }
    (bytes as f64 / (bandwidth_gbs as f64 * 1e9) * 1e3) as f32
}

fn mean_f32(iter: impl Iterator<Item = f32>) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u32;
    for v in iter { sum += v as f64; n += 1; }
    if n == 0 { 0.0 } else { (sum / n as f64) as f32 }
}
