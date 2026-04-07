// src/pool/slow_path.rs — slow-path allocator (fallback when slab is full)

/// Invoked by PoolRegistry when all slab slots are occupied.
/// May attempt eviction, compaction, or return PoolExhausted.
pub struct SlowPathAllocator {
    _opaque: (),
}

impl SlowPathAllocator {
    pub fn new() -> Self {
        unimplemented!("SlowPathAllocator::new — Sprint 0 stub")
    }
}

impl Default for SlowPathAllocator {
    fn default() -> Self {
        Self::new()
    }
}
