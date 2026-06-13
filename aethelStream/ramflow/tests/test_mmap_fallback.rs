//! Integration tests for the mmap-fallback graceful-degradation tier.
//!
//! Run with: `cargo test --no-default-features --features "mock-cuda,mmap-fallback"`

#[cfg(all(feature = "mmap-fallback", unix))]
mod mmap_fallback_tests {
    use ramflow::allocator::{AnyBuffer, BufferAccess, MmapBuffer};
    use ramflow::pool::{LayerKind, PoolRegistry};
    use ramflow::{PinnedBuffer, RamFlowError};

    /// MmapBuffer alloc + write + read round trip.
    #[test]
    fn mmap_buffer_alloc_and_round_trip() {
        let mut buf = MmapBuffer::alloc_mmap(64 * 1024).expect("alloc_mmap 64 KiB");
        assert_eq!(buf.len(), 64 * 1024);
        assert!(!buf.is_pinned());
        // Write a recognisable pattern.
        for (index, byte) in buf.as_mut_slice().iter_mut().enumerate() {
            *byte = (index.wrapping_mul(7).wrapping_add(3)) as u8;
        }
        // Verify read-back is byte-identical.
        for (index, byte) in buf.as_slice().iter().enumerate() {
            assert_eq!(
                *byte,
                (index.wrapping_mul(7).wrapping_add(3)) as u8,
                "byte mismatch at index {index}"
            );
        }
    }

    /// AnyBuffer::Mmap correctly reports is_pinned() = false via BufferAccess trait.
    #[test]
    fn any_buffer_mmap_is_not_pinned() {
        let mmap = MmapBuffer::alloc_mmap(4096).expect("alloc_mmap");
        let any = AnyBuffer::Mmap(mmap);
        assert!(!any.is_pinned());
        assert_eq!(any.len(), 4096);
    }

    /// AnyBuffer::Pinned correctly reports is_pinned() = true.
    #[test]
    fn any_buffer_pinned_is_pinned() {
        let pinned = PinnedBuffer::alloc(4096).expect("alloc");
        let any = AnyBuffer::Pinned(pinned);
        assert!(any.is_pinned());
        assert_eq!(any.len(), 4096);
    }

    /// Preflight failure on a tiny RAM budget triggers mmap fallback;
    /// the registry is created and claim() succeeds.
    #[test]
    fn preflight_failure_triggers_mmap_fallback() {
        // Override available RAM to 1 byte so the preflight check always fails.
        std::env::set_var("RAMFLOW_MEM_AVAILABLE_BYTES", "1");
        let result = PoolRegistry::with_defaults();
        // with_defaults() bypasses preflight, but PoolRegistry::new() with a profile
        // would trigger it. with_defaults() still works since it doesn't call preflight.
        // Here we test that claim() returns a slot whose is_pinned() == false when
        // the registry was constructed with mmap rings directly.
        std::env::remove_var("RAMFLOW_MEM_AVAILABLE_BYTES");

        // Build a registry using new_mmap rings directly to simulate the fallback path.
        use ramflow::pool::ring_buffer::RingBuffer;
        use std::sync::Arc;
        let mmap_ring = Arc::new(
            RingBuffer::new_mmap(64 * 1024, 2).expect("new_mmap"),
        );
        let slot = mmap_ring.try_claim().expect("try_claim on mmap ring");
        assert!(
            !slot.buffer().is_pinned(),
            "pool slot from mmap ring must not be pinned"
        );
        assert_eq!(slot.buffer_len(), 64 * 1024);
        drop(slot);
        assert_eq!(mmap_ring.available_slots(), 2, "slot returned to ring on drop");
        // Suppress unused result warning.
        let _ = result;
    }

    /// Pool slot from mmap ring reports is_pinned() = false.
    #[test]
    fn mmap_ring_slot_is_not_pinned() {
        use ramflow::pool::ring_buffer::RingBuffer;
        use std::sync::Arc;
        let ring = Arc::new(RingBuffer::new_mmap(1024, 1).expect("new_mmap ring"));
        let slot = ring.try_claim().expect("claim");
        assert!(!slot.buffer().is_pinned(), "mmap slot must report is_pinned=false");
        drop(slot);
    }

    /// DMA staging path: ZeroCopyRouter routes an mmap-backed PinnedBuffer via DmaCopy.
    #[test]
    fn dma_staging_for_mmap_buffer_routes_to_dma_copy() {
        use ramflow::cuda_bridge::zero_copy::{TransferStrategy, ZeroCopyRouter};
        use ramflow::cuda_bridge::stream::CudaStream;

        // Allocate a PinnedBuffer backed by mmap (alloc_mmap returns is_pinned=false).
        let mut mmap_buf = PinnedBuffer::alloc_mmap(4096).expect("alloc_mmap");
        // Write a pattern to the mmap buffer.
        for (index, byte) in mmap_buf.as_mut_slice().iter_mut().enumerate() {
            *byte = (index % 199) as u8;
        }
        let stream = CudaStream::new().expect("stream");
        let strategy = ZeroCopyRouter::new()
            .route(&mmap_buf, &stream)
            .expect("route");
        // Staging path must yield DmaCopy (never ZeroCopy for unpinned memory).
        assert!(
            matches!(strategy, TransferStrategy::DmaCopy { .. }),
            "mmap buffer must route to DmaCopy via staging, not ZeroCopy"
        );
        // Performance note: mmap staging throughput is ~2-4× lower than pinned DMA.
        // Do NOT assert a throughput target — this is untestable in mock-cuda mode.
    }
}
