// src/cuda_bridge/stream.rs — CUDA stream RAII wrapper stub

/// Owned handle to a `cudaStream_t`.
///
/// Automatically calls `cudaStreamDestroy` on drop (real CUDA path) or is a
/// no-op (mock-cuda path).
#[derive(Debug)]
pub struct CudaStream {
    _opaque: (),
}

impl CudaStream {
    /// Create a new non-blocking CUDA stream.
    pub fn new() -> crate::Result<Self> {
        unimplemented!("CudaStream::new — Sprint 0 stub")
    }

    /// Synchronise: wait for all work enqueued on this stream to finish.
    pub fn synchronize(&self) -> crate::Result<()> {
        unimplemented!("CudaStream::synchronize — Sprint 0 stub")
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // Sprint 0: nothing to destroy
    }
}
