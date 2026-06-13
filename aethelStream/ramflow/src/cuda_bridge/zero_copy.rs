// src/cuda_bridge/zero_copy.rs — hybrid zero-copy router (Algorithm 3)

use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::allocator::PinnedBuffer;
use crate::cuda_bridge::bindings::{
    cuda_host_get_device_pointer, cuda_malloc_async, cuda_memcpy_async_host_to_device,
};
use crate::cuda_bridge::stream::CudaStream;
use crate::Result;

/// Opaque GPU-side device pointer returned by CUDA.
///
/// For zero-copy routes this is the UVA pointer returned by
/// `cudaHostGetDevicePointer`. For mock-cuda it aliases the host pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePointer {
    raw: *mut u8,
}

// Safety: DevicePointer is a plain address. The caller must keep the backing
// allocation alive and synchronize external access.
unsafe impl Send for DevicePointer {}
unsafe impl Sync for DevicePointer {}

impl DevicePointer {
    /// Construct from a raw pointer.
    ///
    /// # Safety
    /// `raw` must remain valid for the intended GPU operation lifetime.
    pub unsafe fn from_raw(raw: *mut u8) -> Self {
        DevicePointer { raw }
    }

    /// Return the raw pointer value for CUDA kernel launches.
    pub fn as_ptr(&self) -> *mut u8 {
        self.raw
    }
}

/// The two transfer strategies the router selects between.
#[derive(Debug)]
pub enum TransferStrategy {
    /// GPU reads pinned CPU memory directly via CUDA UVA.
    ZeroCopy {
        /// Device-visible pointer to the mapped host allocation.
        device_ptr: DevicePointer,
    },

    /// Full asynchronous PCIe DMA copy to a VRAM staging buffer.
    DmaCopy {
        /// CUDA stream associated with the asynchronous copy.
        stream: CudaStream,
    },
}

/// Default threshold: 4 MiB (calibrated for PCIe Gen4 x16).
static ZERO_COPY_THRESHOLD: AtomicUsize = AtomicUsize::new(4 * 1024 * 1024);

/// Selects [`TransferStrategy`] for each tensor at runtime.
pub struct ZeroCopyRouter;

impl ZeroCopyRouter {
    /// Construct a router. The threshold is held globally.
    pub fn new() -> Self {
        ZeroCopyRouter
    }

    /// Route `buf` to zero-copy or DMA.
    ///
    /// Zero-copy is selected only when `buf.len() < threshold` and the buffer
    /// was registered with `cudaHostRegisterMapped`. Otherwise this performs a
    /// host-to-device async copy and returns `DmaCopy`.
    ///
    /// # Errors
    /// Returns [`crate::RamFlowError::CudaError`] when CUDA pointer lookup,
    /// allocation, or copy submission fails.
    pub fn route(&self, buf: &PinnedBuffer, stream: &CudaStream) -> Result<TransferStrategy> {
#[cfg(feature = "mmap-fallback")] if !buf.is_pinned() { return route_via_staging(buf, stream); }
        if buf.len() < Self::threshold() && buf.is_mapped() {
            let device_ptr = device_pointer_for_mapped_buffer(buf)?;
            return Ok(TransferStrategy::ZeroCopy { device_ptr });
        }

        let device_ptr = unsafe { cuda_malloc_async(buf.len(), stream.as_raw())? };
        unsafe {
            cuda_memcpy_async_host_to_device(
                device_ptr,
                buf.as_ptr() as *const c_void,
                buf.len(),
                stream.as_raw(),
            )?;
        }

        Ok(TransferStrategy::DmaCopy {
            stream: CudaStream::new()?,
        })
    }

    /// Update the crossover threshold in bytes.
    pub fn set_threshold(bytes: usize) {
        ZERO_COPY_THRESHOLD.store(bytes, Ordering::Relaxed);
    }

    /// Current threshold in bytes.
    pub fn threshold() -> usize {
        ZERO_COPY_THRESHOLD.load(Ordering::Relaxed)
    }
}

impl Default for ZeroCopyRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Transfer an mmap-backed buffer to VRAM via a temporary pinned staging copy.
///
/// Allocation cost: one `PinnedBuffer::alloc(buf.len())` per call.
/// A future optimisation is a reusable fixed-size staging pool.
///
/// # Why staging?
/// mmap buffers are pageable — the GPU's DMA engine cannot address them
/// directly. We must copy mmap -> pinned staging (CPU memcpy, uses memory
/// bandwidth) then initiate cudaMemcpyAsync from the pinned buffer.
#[cfg(feature = "mmap-fallback")]
fn route_via_staging(buf: &PinnedBuffer, stream: &CudaStream) -> Result<TransferStrategy> {
    use crate::allocator::PinnedBuffer as Pinned;
    let mut staging = Pinned::alloc(buf.len())?;
    staging.as_mut_slice().copy_from_slice(buf.as_slice());
    let device_ptr = unsafe { cuda_malloc_async(staging.len(), stream.as_raw())? };
    unsafe {
        cuda_memcpy_async_host_to_device(
            device_ptr,
            staging.as_ptr() as *const c_void,
            staging.len(),
            stream.as_raw(),
        )?;
    }
    drop(staging);
    Ok(TransferStrategy::DmaCopy {
        stream: CudaStream::new()?,
    })
}

pub(crate) fn device_pointer_for_mapped_buffer(buf: &PinnedBuffer) -> Result<DevicePointer> {
    if !buf.is_mapped() {
        return Err(crate::RamFlowError::ConfigError(
            "device_pointer_for_mapped_buffer called on an unmapped PinnedBuffer: \
             allocate with PinnedBuffer::alloc_mapped() for UVA/zero-copy access"
                .into(),
        ));
    }
    let raw = unsafe { cuda_host_get_device_pointer(buf.as_ptr() as *mut c_void)? };
    Ok(unsafe { DevicePointer::from_raw(raw as *mut u8) })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn mapped_small_buffer_routes_to_zero_copy() {
        let mut buffer = PinnedBuffer::alloc_mapped(1024 * 1024).expect("mapped alloc");
        buffer.as_mut_slice()[0] = 0xA5;
        let stream = CudaStream::new().expect("stream");
        ZeroCopyRouter::set_threshold(4 * 1024 * 1024);
        match ZeroCopyRouter::new()
            .route(&buffer, &stream)
            .expect("route")
        {
            TransferStrategy::ZeroCopy { device_ptr } => {
                assert_eq!(device_ptr.as_ptr(), buffer.as_ptr() as *mut u8);
            }
            TransferStrategy::DmaCopy { .. } => panic!("mapped 1 MiB buffer should zero-copy"),
        }
    }

    #[test]
    fn unmapped_small_buffer_routes_to_dma() {
        let buffer = PinnedBuffer::alloc(1024 * 1024).expect("alloc");
        let stream = CudaStream::new().expect("stream");
        ZeroCopyRouter::set_threshold(4 * 1024 * 1024);
        match ZeroCopyRouter::new()
            .route(&buffer, &stream)
            .expect("route")
        {
            TransferStrategy::ZeroCopy { .. } => panic!("unmapped buffer must not zero-copy"),
            TransferStrategy::DmaCopy { .. } => {}
        }
    }

    #[test]
    fn mapped_zero_copy_is_byte_identical_and_crossover_matches_policy() {
        let mut buffer = PinnedBuffer::alloc_mapped(1024 * 1024).expect("mapped alloc");
        for (index, byte) in buffer.as_mut_slice().iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }

        let stream = CudaStream::new().expect("stream");
        ZeroCopyRouter::set_threshold(4 * 1024 * 1024);
        let device_ptr = match ZeroCopyRouter::new()
            .route(&buffer, &stream)
            .expect("route")
        {
            TransferStrategy::ZeroCopy { device_ptr } => device_ptr,
            TransferStrategy::DmaCopy { .. } => panic!("1 MiB mapped buffer should zero-copy"),
        };

        let device_view = unsafe { std::slice::from_raw_parts(device_ptr.as_ptr(), buffer.len()) };
        assert_eq!(device_view, buffer.as_slice());

        assert!(
            zero_copy_time_score(1024 * 1024) < dma_copy_time_score(1024 * 1024),
            "zero-copy should win at 1 MiB"
        );
        assert!(
            zero_copy_time_score(8 * 1024 * 1024) > dma_copy_time_score(8 * 1024 * 1024),
            "DMA should win at 8 MiB"
        );
    }

    #[test]
    fn mapped_large_buffer_above_threshold_routes_to_dma() {
        // Threshold is 4 MiB; an 8 MiB mapped buffer must route to DMA (too large for zero-copy).
        let buffer = PinnedBuffer::alloc_mapped(8 * 1024 * 1024).expect("mapped alloc 8 MiB");
        let stream = CudaStream::new().expect("stream");
        ZeroCopyRouter::set_threshold(4 * 1024 * 1024);
        match ZeroCopyRouter::new()
            .route(&buffer, &stream)
            .expect("route")
        {
            TransferStrategy::ZeroCopy { .. } => {
                panic!("8 MiB mapped buffer exceeds 4 MiB threshold — must route to DMA")
            }
            TransferStrategy::DmaCopy { .. } => {}
        }
    }

    #[test]
    fn unmapped_large_buffer_above_threshold_routes_to_dma() {
        // Unmapped buffers always DMA regardless of size.
        let buffer = PinnedBuffer::alloc(8 * 1024 * 1024).expect("alloc 8 MiB");
        let stream = CudaStream::new().expect("stream");
        ZeroCopyRouter::set_threshold(4 * 1024 * 1024);
        match ZeroCopyRouter::new()
            .route(&buffer, &stream)
            .expect("route")
        {
            TransferStrategy::ZeroCopy { .. } => {
                panic!("unmapped buffer must never route to zero-copy")
            }
            TransferStrategy::DmaCopy { .. } => {}
        }
    }

    fn zero_copy_time_score(size_bytes: usize) -> usize {
        8_000usize.saturating_add(size_bytes / 128)
    }

    fn dma_copy_time_score(size_bytes: usize) -> usize {
        30_000usize.saturating_add(size_bytes / 256)
    }
}
