// src/nvme/prefetch.rs — predictive NVMe prefetch engine stub

/// Issues read-ahead requests for tensors predicted to enter the hot tier.
pub struct PrefetchEngine {
    _opaque: (),
}

impl PrefetchEngine {
    pub fn new() -> crate::Result<Self> {
        unimplemented!("PrefetchEngine::new — Sprint 0 stub")
    }

    /// Schedule a prefetch of `tensor_id` from NVMe into a pinned buffer.
    #[allow(unused_variables)]
    pub fn schedule(&self, tensor_id: u64) -> crate::Result<()> {
        unimplemented!("PrefetchEngine::schedule — Sprint 0 stub")
    }

    /// Preparation for Sprint 2: I/O logic for non-Linux platforms.
    ///
    /// io_uring is Linux-only. Use standard synchronous reads for dev/test on Windows.
    #[cfg(not(target_os = "linux"))]
    #[allow(unused_variables)]
    pub fn prefetch(&self, path: &std::path::Path, offset: u64, size: usize) -> crate::Result<()> {
        // io_uring not available outside Linux.
        // Use synchronous read for dev/test on Windows/macOS.
        todo!("implement fallback sync read for non-Linux")
    }
}
