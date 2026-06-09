// src/phase/profiler.rs — warm-up profiler and access-pattern profiler

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::phase::classifier::{PhaseMemoryProfile, TrainingPhase};
use crate::pool::LayerKind;
use crate::{RamFlowError, Result};

/// Runs instrumented training steps to measure peak pool usage per phase.
///
/// The profiler records per-phase peak claim counters and writes
/// `hardware_profile.json`. On later runs, a matching model SHA-256 lets the
/// profiler return cached profiles without rerunning warm-up work.
pub struct WarmupProfiler {
    config: WarmupConfig,
    counters: PoolClaimCounters,
}

/// Configuration for the warm-up profiling pass.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of mini-training steps to run (default: 5).
    pub steps: u32,
    /// Path to write the resulting hardware profile JSON.
    pub output_path: PathBuf,
    /// SHA-256 of `shard_index.json`; used to skip profiling on cache hit.
    pub model_sha256: [u8; 32],
}

impl WarmupConfig {
    /// Build a profiler config and compute the SHA-256 of `shard_index.json`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::ConfigError`] if the shard index cannot be read.
    pub fn for_shard_index(
        steps: u32,
        output_path: PathBuf,
        shard_index_path: &Path,
    ) -> Result<Self> {
        let bytes = std::fs::read(shard_index_path).map_err(|read_error| {
            RamFlowError::ConfigError(format!(
                "failed to read shard index {}: {read_error}",
                shard_index_path.display()
            ))
        })?;
        Ok(WarmupConfig {
            steps,
            output_path,
            model_sha256: sha256(&bytes),
        })
    }
}

impl WarmupProfiler {
    /// Create a new profiler. Does not start profiling; call [`Self::run`].
    ///
    /// # Errors
    /// Returns [`RamFlowError::ConfigError`] if `steps` is zero.
    pub fn new(config: WarmupConfig) -> Result<Self> {
        if config.steps == 0 {
            return Err(RamFlowError::ConfigError(
                "WarmupProfiler steps must be non-zero".into(),
            ));
        }
        Ok(WarmupProfiler {
            config,
            counters: PoolClaimCounters::default(),
        })
    }

    /// Check whether a valid `hardware_profile.json` exists for this model.
    pub fn is_cache_valid(&self) -> bool {
        self.load_cached_profiles().is_ok()
    }

    /// Record one instrumented pool claim and return a guard that releases it.
    pub fn record_pool_claim(&self, phase: ProfilePhase, kind: LayerKind) -> ClaimCounterGuard<'_> {
        self.counters.claim(phase, kind)
    }

    /// Execute the warm-up profiling pass.
    ///
    /// Returns `[forward_profile, backward_profile, recomputation_profile]`.
    ///
    /// # Errors
    /// Returns [`RamFlowError::ConfigError`] if cached JSON is malformed or the
    /// output profile cannot be written.
    pub fn run(&self) -> Result<[PhaseMemoryProfile; 3]> {
        if let Ok(cached_profiles) = self.load_cached_profiles() {
            return Ok(cached_profiles);
        }

        for _step_index in 0..self.config.steps {
            self.simulate_mini_step();
        }

        let profiles = self.counters.to_profiles();
        self.write_profiles(&profiles)?;
        Ok(profiles)
    }

    /// Measure the zero-copy/DMA crossover threshold.
    ///
    /// Sprint 4A sweeps 512 KiB, 1 MiB, 2 MiB, 4 MiB, and 8 MiB. In mock-cuda
    /// this uses deterministic timing estimates so the routing policy remains
    /// testable without a GPU.
    pub fn measure_zero_copy_crossover(&self) -> Result<usize> {
        let sample_sizes = [
            512 * 1024usize,
            1024 * 1024,
            2 * 1024 * 1024,
            4 * 1024 * 1024,
            8 * 1024 * 1024,
        ];
        let mut threshold = sample_sizes[0];
        for size in sample_sizes {
            let zero_copy_score = estimated_zero_copy_score(size);
            let dma_score = estimated_dma_copy_score(size);
            if zero_copy_score <= dma_score {
                threshold = size;
            }
        }
        Ok(threshold)
    }

    /// Measure pressure sampling interval in steps.
    ///
    /// Sprint 3B returns `1` so pressure is sampled every step in warm-up and
    /// tests. Later profiling can widen this interval.
    pub fn measure_pressure_sample_interval(&self) -> Result<u32> {
        Ok(1)
    }

    fn simulate_mini_step(&self) {
        {
            let _attention = self.record_pool_claim(ProfilePhase::Forward, LayerKind::Attention);
            let _mlp = self.record_pool_claim(ProfilePhase::Forward, LayerKind::Mlp);
            let _norm = self.record_pool_claim(ProfilePhase::Forward, LayerKind::Norm);
        }
        {
            let _attention_a = self.record_pool_claim(ProfilePhase::Backward, LayerKind::Attention);
            let _attention_b = self.record_pool_claim(ProfilePhase::Backward, LayerKind::Attention);
            let _mlp = self.record_pool_claim(ProfilePhase::Backward, LayerKind::Mlp);
            let _embedding = self.record_pool_claim(ProfilePhase::Backward, LayerKind::Embedding);
        }
        {
            let _attention_a =
                self.record_pool_claim(ProfilePhase::Recomputation, LayerKind::Attention);
            let _attention_b =
                self.record_pool_claim(ProfilePhase::Recomputation, LayerKind::Attention);
            let _attention_c =
                self.record_pool_claim(ProfilePhase::Recomputation, LayerKind::Attention);
            let _mlp_a = self.record_pool_claim(ProfilePhase::Recomputation, LayerKind::Mlp);
            let _mlp_b = self.record_pool_claim(ProfilePhase::Recomputation, LayerKind::Mlp);
            let _norm = self.record_pool_claim(ProfilePhase::Recomputation, LayerKind::Norm);
        }
    }

    fn load_cached_profiles(&self) -> Result<[PhaseMemoryProfile; 3]> {
        let bytes = std::fs::read(&self.config.output_path).map_err(|read_error| {
            RamFlowError::ConfigError(format!(
                "failed to read hardware profile {}: {read_error}",
                self.config.output_path.display()
            ))
        })?;
        let cache: HardwareProfileCache =
            serde_json::from_slice(&bytes).map_err(|parse_error| {
                RamFlowError::ConfigError(format!(
                    "failed to parse hardware profile {}: {parse_error}",
                    self.config.output_path.display()
                ))
            })?;

        if cache.model_sha256 != hex_encode(&self.config.model_sha256) {
            return Err(RamFlowError::ConfigError(
                "hardware profile SHA-256 does not match shard_index.json".into(),
            ));
        }

        Ok([
            cache.forward.to_profile(ProfilePhase::Forward),
            cache.backward.to_profile(ProfilePhase::Backward),
            cache.recomputation.to_profile(ProfilePhase::Recomputation),
        ])
    }

    fn write_profiles(&self, profiles: &[PhaseMemoryProfile; 3]) -> Result<()> {
        let cache = HardwareProfileCache {
            model_sha256: hex_encode(&self.config.model_sha256),
            zero_copy_threshold_bytes: self.measure_zero_copy_crossover()?,
            forward: CachedPhaseProfile::from_profile(&profiles[0]),
            backward: CachedPhaseProfile::from_profile(&profiles[1]),
            recomputation: CachedPhaseProfile::from_profile(&profiles[2]),
        };
        let json = serde_json::to_vec_pretty(&cache).map_err(|serialize_error| {
            RamFlowError::ConfigError(format!(
                "failed to serialize hardware profile: {serialize_error}"
            ))
        })?;
        if let Some(parent) = self.config.output_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|create_error| {
                    RamFlowError::ConfigError(format!(
                        "failed to create profile directory {}: {create_error}",
                        parent.display()
                    ))
                })?;
            }
        }
        std::fs::write(&self.config.output_path, json).map_err(|write_error| {
            RamFlowError::ConfigError(format!(
                "failed to write hardware profile {}: {write_error}",
                self.config.output_path.display()
            ))
        })
    }
}

/// Phase key used by warm-up claim counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilePhase {
    /// Forward mini-step section.
    Forward,
    /// Backward mini-step section.
    Backward,
    /// Recomputation mini-forward section.
    Recomputation,
}

#[derive(Default)]
struct PoolClaimCounters {
    forward: PhaseCounters,
    backward: PhaseCounters,
    recomputation: PhaseCounters,
}

impl PoolClaimCounters {
    fn claim(&self, phase: ProfilePhase, kind: LayerKind) -> ClaimCounterGuard<'_> {
        self.phase(phase).claim(kind)
    }

    fn to_profiles(&self) -> [PhaseMemoryProfile; 3] {
        [
            self.forward.to_profile(ProfilePhase::Forward),
            self.backward.to_profile(ProfilePhase::Backward),
            self.recomputation.to_profile(ProfilePhase::Recomputation),
        ]
    }

    fn phase(&self, phase: ProfilePhase) -> &PhaseCounters {
        match phase {
            ProfilePhase::Forward => &self.forward,
            ProfilePhase::Backward => &self.backward,
            ProfilePhase::Recomputation => &self.recomputation,
        }
    }
}

#[derive(Default)]
struct PhaseCounters {
    attention_current: AtomicU32,
    attention_peak: AtomicU32,
    mlp_current: AtomicU32,
    mlp_peak: AtomicU32,
    norm_current: AtomicU32,
    norm_peak: AtomicU32,
    embedding_current: AtomicU32,
    embedding_peak: AtomicU32,
}

impl PhaseCounters {
    fn claim(&self, kind: LayerKind) -> ClaimCounterGuard<'_> {
        let (current, peak) = self.counter_pair(kind);
        let active = current.fetch_add(1, Ordering::AcqRel).saturating_add(1);
        update_peak(peak, active);
        ClaimCounterGuard { current }
    }

    fn to_profile(&self, phase: ProfilePhase) -> PhaseMemoryProfile {
        let attention = self.attention_peak.load(Ordering::Acquire).max(1);
        let mlp = self.mlp_peak.load(Ordering::Acquire).max(1);
        let norm = self.norm_peak.load(Ordering::Acquire).max(1);
        let embedding = self.embedding_peak.load(Ordering::Acquire).max(1);
        PhaseMemoryProfile {
            phase: match phase {
                ProfilePhase::Forward => TrainingPhase::Forward {
                    layers_in_flight: attention,
                },
                ProfilePhase::Backward => TrainingPhase::Backward {
                    checkpoint_interval: 1,
                },
                ProfilePhase::Recomputation => TrainingPhase::Recomputation {
                    window_start: 0,
                    window_end: attention.saturating_sub(1),
                },
            },
            expected_peak_bytes: estimated_peak_bytes(attention, mlp, norm, embedding),
            attention_slots_needed: attention,
            mlp_slots_needed: mlp,
            norm_slots_needed: norm,
            optimizer_slots_needed: embedding,
        }
    }

    fn counter_pair(&self, kind: LayerKind) -> (&AtomicU32, &AtomicU32) {
        match kind {
            LayerKind::Attention => (&self.attention_current, &self.attention_peak),
            LayerKind::Mlp => (&self.mlp_current, &self.mlp_peak),
            LayerKind::Norm => (&self.norm_current, &self.norm_peak),
            LayerKind::Embedding => (&self.embedding_current, &self.embedding_peak),
        }
    }
}

/// RAII guard for one instrumented warm-up pool claim.
pub struct ClaimCounterGuard<'a> {
    current: &'a AtomicU32,
}

impl Drop for ClaimCounterGuard<'_> {
    fn drop(&mut self) {
        self.current.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct HardwareProfileCache {
    model_sha256: String,
    #[serde(default = "default_zero_copy_threshold_bytes")]
    zero_copy_threshold_bytes: usize,
    forward: CachedPhaseProfile,
    backward: CachedPhaseProfile,
    recomputation: CachedPhaseProfile,
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedPhaseProfile {
    expected_peak_bytes: usize,
    attention_slots_needed: u32,
    mlp_slots_needed: u32,
    norm_slots_needed: u32,
    optimizer_slots_needed: u32,
}

impl CachedPhaseProfile {
    fn from_profile(profile: &PhaseMemoryProfile) -> Self {
        CachedPhaseProfile {
            expected_peak_bytes: profile.expected_peak_bytes,
            attention_slots_needed: profile.attention_slots_needed,
            mlp_slots_needed: profile.mlp_slots_needed,
            norm_slots_needed: profile.norm_slots_needed,
            optimizer_slots_needed: profile.optimizer_slots_needed,
        }
    }

    fn to_profile(&self, phase: ProfilePhase) -> PhaseMemoryProfile {
        PhaseMemoryProfile {
            phase: match phase {
                ProfilePhase::Forward => TrainingPhase::Forward {
                    layers_in_flight: self.attention_slots_needed,
                },
                ProfilePhase::Backward => TrainingPhase::Backward {
                    checkpoint_interval: 1,
                },
                ProfilePhase::Recomputation => TrainingPhase::Recomputation {
                    window_start: 0,
                    window_end: self.attention_slots_needed.saturating_sub(1),
                },
            },
            expected_peak_bytes: self.expected_peak_bytes,
            attention_slots_needed: self.attention_slots_needed,
            mlp_slots_needed: self.mlp_slots_needed,
            norm_slots_needed: self.norm_slots_needed,
            optimizer_slots_needed: self.optimizer_slots_needed,
        }
    }
}

fn update_peak(peak: &AtomicU32, active: u32) {
    let mut observed = peak.load(Ordering::Acquire);
    while active > observed {
        match peak.compare_exchange(observed, active, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(next_observed) => observed = next_observed,
        }
    }
}

fn estimated_peak_bytes(attention: u32, mlp: u32, norm: u32, embedding: u32) -> usize {
    let large_slot_bytes = 64usize * 1024 * 1024;
    let norm_slot_bytes = 1024usize * 1024;
    let embedding_slot_bytes = 32usize * 1024 * 1024;
    (attention as usize + mlp as usize) * large_slot_bytes
        + norm as usize * norm_slot_bytes
        + embedding as usize * embedding_slot_bytes
}

fn default_zero_copy_threshold_bytes() -> usize {
    4 * 1024 * 1024
}

fn estimated_zero_copy_score(size_bytes: usize) -> usize {
    8_000usize.saturating_add(size_bytes / 128)
}

fn estimated_dma_copy_score(size_bytes: usize) -> usize {
    30_000usize.saturating_add(size_bytes / 256)
}

/// Records per-tensor access statistics used by the phase classifier.
pub struct AccessProfiler {
    started_at: Instant,
    counts: Mutex<HashMap<u64, u64>>,
}

impl AccessProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        AccessProfiler {
            started_at: Instant::now(),
            counts: Mutex::new(HashMap::new()),
        }
    }

    /// Record an access to `tensor_id` at the current monotonic timestamp.
    pub fn record_access(&self, tensor_id: u64) {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let counter = counts.entry(tensor_id).or_insert(0);
        *counter = counter.saturating_add(1);
    }

    /// Return the access frequency (accesses/sec) for `tensor_id`.
    pub fn frequency(&self, tensor_id: u64) -> f64 {
        let elapsed = self.started_at.elapsed().as_secs_f64().max(f64::EPSILON);
        let counts = self
            .counts
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        counts.get(&tensor_id).copied().unwrap_or(0) as f64 / elapsed
    }
}

impl Default for AccessProfiler {
    fn default() -> Self {
        Self::new()
    }
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    const INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while !(padded.len() + 8).is_multiple_of(64) {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL_STATE;
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (word_index, word_bytes) in chunk.chunks_exact(4).enumerate().take(16) {
            schedule[word_index] =
                u32::from_be_bytes([word_bytes[0], word_bytes[1], word_bytes[2], word_bytes[3]]);
        }
        for word_index in 16..64 {
            let small_sigma_0 = schedule[word_index - 15].rotate_right(7)
                ^ schedule[word_index - 15].rotate_right(18)
                ^ (schedule[word_index - 15] >> 3);
            let small_sigma_1 = schedule[word_index - 2].rotate_right(17)
                ^ schedule[word_index - 2].rotate_right(19)
                ^ (schedule[word_index - 2] >> 10);
            schedule[word_index] = schedule[word_index - 16]
                .wrapping_add(small_sigma_0)
                .wrapping_add(schedule[word_index - 7])
                .wrapping_add(small_sigma_1);
        }

        let mut working = state;
        for round_index in 0..64 {
            let big_sigma_1 = working[4].rotate_right(6)
                ^ working[4].rotate_right(11)
                ^ working[4].rotate_right(25);
            let choose = (working[4] & working[5]) ^ ((!working[4]) & working[6]);
            let temp1 = working[7]
                .wrapping_add(big_sigma_1)
                .wrapping_add(choose)
                .wrapping_add(ROUND_CONSTANTS[round_index])
                .wrapping_add(schedule[round_index]);
            let big_sigma_0 = working[0].rotate_right(2)
                ^ working[0].rotate_right(13)
                ^ working[0].rotate_right(22);
            let majority =
                (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
            let temp2 = big_sigma_0.wrapping_add(majority);
            working = [
                temp1.wrapping_add(temp2),
                working[0],
                working[1],
                working[2],
                working[3].wrapping_add(temp1),
                working[4],
                working[5],
                working[6],
            ];
        }

        for state_index in 0..8 {
            state[state_index] = state[state_index].wrapping_add(working[state_index]);
        }
    }

    let mut output = [0u8; 32];
    for (state_index, word) in state.iter().enumerate() {
        output[state_index * 4..state_index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}
