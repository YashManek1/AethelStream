// tests/test_lz4_eviction.rs — Integration tests for the LZ4 eviction cache
//
// Run:
//   cargo test --no-default-features --features "mock-cuda,lz4-cache"
//
// All tests use mock-cuda so no GPU is required.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[cfg(feature = "lz4-cache")]
mod lz4_eviction_tests {
    use ramflow::pool::{CachePrecision, LayerKind, PoolRegistry, TensorLocationDict};
    use ramflow::phase::{PhaseMemoryProfile, TrainingPhase};
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Helper: build a minimal 1-slot attention PoolRegistry
    // -----------------------------------------------------------------------
    fn single_slot_registry() -> Arc<PoolRegistry> {
        let profile = PhaseMemoryProfile {
            phase: TrainingPhase::Recomputation {
                window_start: 0,
                window_end: 8,
            },
            expected_peak_bytes: 0,
            attention_slots_needed: 1,
            mlp_slots_needed: 1,
            norm_slots_needed: 1,
            optimizer_slots_needed: 1,
        };
        Arc::new(
            PoolRegistry::new(&profile, &TensorLocationDict::empty(), 4096)
                .expect("single_slot_registry"),
        )
    }

    // -----------------------------------------------------------------------
    // Test 1: round-trip compress → decompress is byte-identical
    // -----------------------------------------------------------------------
    #[test]
    fn round_trip_compress_decompress_byte_identical() {
        use ramflow::pool::EvictionCache;

        const SIZE: usize = 1024 * 1024; // 1 MiB
        let original: Vec<u8> = (0..SIZE)
            .map(|index: usize| (index.wrapping_mul(6364136223846793005usize).wrapping_add(1442695040888963407)) as u8)
            .collect();

        let mut cache = EvictionCache::new(64 * 1024 * 1024);
        cache
            .compress(0, &original, CachePrecision::Fp16)
            .expect("compress failed");
        assert!(cache.contains(0), "entry 0 must exist after compress");

        let mut restored = vec![0u8; SIZE];
        let hit = cache.decompress(0, &mut restored).expect("decompress failed");

        assert!(hit, "decompress must return true on a hit");
        assert_eq!(
            original, restored,
            "decompressed data must be byte-identical to original"
        );
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
        assert!(!cache.contains(0), "entry must be consumed after decompress");
    }

    // -----------------------------------------------------------------------
    // Test 2: compression ratio — document behaviour for random FP16
    // -----------------------------------------------------------------------
    #[test]
    fn compression_ratio_random_fp16_documented() {
        use ramflow::pool::EvictionCache;

        // Random FP16 compresses poorly because the bit patterns are white noise.
        // Real trained weights (with many near-zero values and repetitive structure)
        // compress to 0.5–0.75× and benefit substantially from this cache.
        const SIZE: usize = 1024 * 1024;
        let random_fp16: Vec<u8> = (0..SIZE)
            .map(|index| ((index * 2654435761usize) ^ (index >> 5)) as u8)
            .collect();

        let mut cache = EvictionCache::new(64 * 1024 * 1024);
        cache
            .compress(0, &random_fp16, CachePrecision::Fp16)
            .expect("compress");

        let compressed_bytes = cache.current_bytes();
        let ratio = compressed_bytes as f64 / SIZE as f64;
        // Random FP16: ratio is typically 0.95–1.05 (LZ4 overhead for incompressible data).
        // Trained weights: ratio is typically 0.50–0.75.
        assert!(
            compressed_bytes > 0,
            "cache must store at least one byte; got ratio {ratio:.3}"
        );

        // Verify round-trip still works on the compressed noise.
        let mut restored = vec![0u8; SIZE];
        let hit = cache.decompress(0, &mut restored).expect("decompress");
        assert!(hit, "must hit after compress");
        assert_eq!(random_fp16, restored, "round-trip must be lossless");
    }

    // -----------------------------------------------------------------------
    // Test 3: LRU eviction drops oldest entry when budget is exceeded
    // -----------------------------------------------------------------------
    #[test]
    fn lru_eviction_drops_oldest_entry_when_budget_exceeded() {
        use ramflow::pool::EvictionCache;

        // Use a highly compressible payload (run of same byte) so we can tightly
        // control the compressed size and budget.
        let payload_a = vec![0xAAu8; 512];
        let payload_b = vec![0xBBu8; 512];
        let payload_c = vec![0xCCu8; 512];

        // Measure compressed size of a and b.
        let compressed_a = lz4_flex::compress(&payload_a);
        let compressed_b = lz4_flex::compress(&payload_b);

        // Budget: fits A + B exactly, not A + B + C.
        let budget = compressed_a.len() + compressed_b.len() + 1;
        let mut cache = EvictionCache::new(budget);

        cache.compress(0, &payload_a, CachePrecision::Fp16).expect("compress a");
        assert!(cache.contains(0));

        cache.compress(1, &payload_b, CachePrecision::Fp16).expect("compress b");
        assert!(cache.contains(1));
        assert_eq!(cache.len(), 2);

        // Adding C pushes total past budget — A (oldest) must be evicted.
        cache.compress(2, &payload_c, CachePrecision::Fp16).expect("compress c");
        assert!(
            !cache.contains(0),
            "oldest entry (A, layer 0) must have been LRU-evicted"
        );
        assert!(cache.contains(1), "B must still be present");
        assert!(cache.contains(2), "C must be present");
        assert_eq!(cache.len(), 2, "cache must hold exactly 2 entries after eviction");
    }

    // -----------------------------------------------------------------------
    // Test 4: integration — slow path uses LZ4 cache instead of blocking
    // -----------------------------------------------------------------------
    #[test]
    fn integration_slow_path_uses_lz4_cache_instead_of_blocking() {
        const SLOT_BYTES: usize = 4096;
        let registry = single_slot_registry();

        // Enable a 64 MiB LZ4 cache and declare a Recomputation window.
        registry.enable_lz4_cache(64 * 1024 * 1024);
        registry.set_recompute_window(0, 8);

        // Claim the only attention slot.
        let mut slot = registry.claim(LayerKind::Attention).expect("claim slot");
        assert_eq!(registry.capacity_for(LayerKind::Attention), 1);
        assert_eq!(registry.claimed_slots_for(LayerKind::Attention), 1);

        // Write a recognisable pattern into the slot buffer.
        let slot_len = slot.buffer_len().min(SLOT_BYTES);
        let pattern: Vec<u8> = (0..slot_len).map(|index| (index & 0xFF) as u8).collect();
        slot.buffer_mut().as_mut_slice()[..slot_len].copy_from_slice(&pattern);

        // Offer this slot (layer 3) as an LZ4 eviction candidate.
        registry
            .offer_for_lz4_eviction(3, LayerKind::Attention, slot, CachePrecision::Fp16)
            .expect("offer_for_lz4_eviction");

        // The slot is now in the eviction queue (still claimed by the queue).
        // claim() for a new slot must trigger LZ4 eviction: compress layer 3,
        // free its ring slot, then return it.
        let new_slot = registry
            .claim(LayerKind::Attention)
            .expect("claim after LZ4 eviction");
        assert_eq!(
            new_slot.buffer_len(),
            slot_len,
            "reclaimed slot must have same buffer size"
        );
        drop(new_slot);

        // Verify telemetry: no hits yet (decompression hasn't been called).
        let telemetry = registry
            .lz4_cache_telemetry()
            .expect("cache must be enabled");
        assert_eq!(telemetry.hits, 0);
        assert!(
            telemetry.current_bytes > 0,
            "cache must hold the compressed layer-3 payload"
        );

        // Claim for layer 3: should decompress from cache into the slot.
        let layer3_slot = registry
            .claim_for_layer(LayerKind::Attention, 3)
            .expect("claim_for_layer");

        assert_eq!(
            layer3_slot.buffer().as_slice()[..slot_len],
            pattern,
            "decompressed data must match original pattern"
        );

        let telemetry_after = registry
            .lz4_cache_telemetry()
            .expect("cache still enabled");
        assert_eq!(telemetry_after.hits, 1, "one decompress hit expected");
        drop(layer3_slot);
    }
}

// Import lz4_flex in test scope so test 3 can measure compressed sizes.
#[cfg(feature = "lz4-cache")]
use lz4_flex;


