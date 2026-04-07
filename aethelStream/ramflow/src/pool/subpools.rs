// src/pool/subpools.rs — per-NUMA / per-GPU sub-pool shards

/// One pool shard, bound to a specific NUMA node or GPU device index.
pub struct SubPool {
    _opaque: (),
}

impl SubPool {
    #[allow(unused_variables)]
    pub fn new(device_index: u32) -> crate::Result<Self> {
        unimplemented!("SubPool::new — Sprint 0 stub")
    }
}
