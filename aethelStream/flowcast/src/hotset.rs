//! A6: Hot-set resident cache.
//!
//! Manages a static resident set: embeddings, LM head, first/last K blocks,
//! and LoRA adapters. When RAM headroom is available, LFU promotion pins
//! the most-accessed non-resident layers into the hot-set.
//!
//! Resident layers are registered in `PerLayerScaleTable::mark_resident` so
//! `PrefetchStateMachine` can skip I/O for them (they are already in RAM).

use std::cmp::Reverse;

use ramflow::PerLayerScaleTable;

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// One slot in the hot-set.
#[derive(Debug, Clone)]
struct Entry {
    layer_idx: u32,
    access_count: u64,
}

// ---------------------------------------------------------------------------
// HotSet
// ---------------------------------------------------------------------------

/// Resident hot-set manager.
///
/// # Residency semantics
/// A layer is *resident* when it is in `entries`. Resident layers must never
/// be submitted to the I/O backend; they are already in pinned RAM.
pub struct HotSet {
    /// Maximum resident slots (static + LFU promoted).
    capacity: usize,
    /// Current resident entries.
    entries: Vec<Entry>,
    /// Per-layer access frequency table (indexed by layer_idx); grows lazily.
    access_counts: Vec<u64>,
}

impl HotSet {
    /// Create a hot-set with `capacity` resident slots.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            access_counts: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Static seeding
    // ------------------------------------------------------------------

    /// Seed the static resident set for a model with `num_layers` total layers.
    ///
    /// Always pins (up to capacity):
    /// - Layer 0 (embedding/first block)
    /// - Layer `num_layers - 1` (LM head/last block)
    /// - First `k` and last `k` transformer blocks
    /// - LoRA adapter layers if `lora_layer_indices` is non-empty
    ///
    /// Marks each pinned layer resident in `scale_table`.
    pub fn seed_static(
        &mut self,
        num_layers: u32,
        k: u32,
        lora_layer_indices: &[u32],
        scale_table: &mut PerLayerScaleTable,
    ) {
        let mut candidates: Vec<u32> = Vec::new();

        // Embedding + LM head
        candidates.push(0);
        if num_layers > 1 {
            candidates.push(num_layers - 1);
        }
        // First K blocks
        for index in 1..k.min(num_layers) {
            candidates.push(index);
        }
        // Last K blocks
        let last_start = num_layers.saturating_sub(k + 1);
        for index in last_start..num_layers.saturating_sub(1) {
            candidates.push(index);
        }
        // LoRA adapters
        for &index in lora_layer_indices {
            if index < num_layers {
                candidates.push(index);
            }
        }

        // Deduplicate preserving order.
        candidates.sort_unstable();
        candidates.dedup();

        for layer_idx in candidates {
            if self.entries.len() >= self.capacity {
                break;
            }
            if !scale_table.is_resident(layer_idx as usize) {
                self.entries.push(Entry { layer_idx, access_count: 0 });
                scale_table.mark_resident(layer_idx as usize, true);
            }
        }
    }

    // ------------------------------------------------------------------
    // Access tracking and LFU promotion
    // ------------------------------------------------------------------

    /// Record an access to `layer_idx`.
    ///
    /// If the layer is resident its counter is bumped. If not resident and
    /// `ram_headroom_bytes >= headroom_threshold_bytes`, promotes it (LFU).
    pub fn record_access(
        &mut self,
        layer_idx: u32,
        ram_headroom_bytes: u64,
        headroom_threshold_bytes: u64,
        scale_table: &mut PerLayerScaleTable,
    ) {
        // Grow access_counts table if necessary.
        let needed = layer_idx as usize + 1;
        if self.access_counts.len() < needed {
            self.access_counts.resize(needed, 0);
        }
        self.access_counts[layer_idx as usize] += 1;
        let new_count = self.access_counts[layer_idx as usize];

        // Bump counter if already resident.
        if let Some(entry) = self.entries.iter_mut().find(|e| e.layer_idx == layer_idx) {
            entry.access_count = new_count;
            return;
        }

        // LFU promotion: only if headroom available.
        if ram_headroom_bytes < headroom_threshold_bytes {
            return;
        }

        if self.entries.len() < self.capacity {
            self.entries.push(Entry { layer_idx, access_count: new_count });
            scale_table.mark_resident(layer_idx as usize, true);
        } else {
            // Evict the LFU entry if new layer is accessed more.
            let (lfu_pos, lfu_count) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(pos, e)| (pos, e.access_count))
                .unwrap_or((0, u64::MAX));
            if new_count > lfu_count {
                let evicted = self.entries[lfu_pos].layer_idx;
                scale_table.mark_resident(evicted as usize, false);
                self.entries[lfu_pos] = Entry { layer_idx, access_count: new_count };
                scale_table.mark_resident(layer_idx as usize, true);
            }
        }
    }

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Whether `layer_idx` is currently resident in pinned RAM.
    ///
    /// Delegates to `scale_table.is_resident` (A6-c fix: the previous
    /// implementation read from `self.entries`, which diverged from the
    /// authoritative residency store in `PerLayerScaleTable`).
    pub fn is_resident(&self, layer_idx: u32, scale_table: &PerLayerScaleTable) -> bool {
        scale_table.is_resident(layer_idx as usize)
    }

    /// Resident layer indices ordered by descending access count.
    ///
    /// Uses `self.entries` for the ordering (LFU counts); residency truth is in
    /// `PerLayerScaleTable` and checked via `is_resident`.
    pub fn resident_layers(&self) -> Vec<u32> {
        let mut sorted = self.entries.clone();
        sorted.sort_unstable_by_key(|entry| Reverse(entry.access_count));
        sorted.iter().map(|e| e.layer_idx).collect()
    }

    /// Whether `layer_idx` is in the hot-set (delegates to `is_resident`).
    pub fn is_hot(&self, layer_idx: u32, scale_table: &PerLayerScaleTable) -> bool {
        self.is_resident(layer_idx, scale_table)
    }

    /// Evict the least-frequently-used entry.
    ///
    /// Returns the evicted layer index, or `None` if empty.
    pub fn evict_lfu(
        &mut self,
        scale_table: &mut PerLayerScaleTable,
    ) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }
        let pos = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.access_count)
            .map(|(pos, _)| pos)
            .unwrap_or(0);
        let evicted = self.entries.remove(pos);
        scale_table.mark_resident(evicted.layer_idx as usize, false);
        Some(evicted.layer_idx)
    }

    /// Capacity of the hot-set.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of resident layers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the hot-set is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
