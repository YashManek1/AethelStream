use ramflow::cuda_bridge::stream::CudaStream;
use ramflow::cuda_bridge::zero_copy::{TransferStrategy, ZeroCopyRouter};
use ramflow::PinnedBuffer;

fn zero_copy_time_score(size_bytes: usize) -> usize {
    8_000usize.saturating_add(size_bytes / 128)
}

fn dma_copy_time_score(size_bytes: usize) -> usize {
    30_000usize.saturating_add(size_bytes / 256)
}

#[test]
fn t7_mapped_sub_threshold_buffer_routes_to_usable_zero_copy_pointer() {
    let mut buffer = PinnedBuffer::alloc_mapped(1024 * 1024).expect("mapped allocation failed");
    let last_index = buffer.len() - 1;
    let buffer_slice = buffer.as_mut_slice();
    buffer_slice[0] = 0xA5;
    buffer_slice[last_index] = 0x5A;

    let stream = CudaStream::new().expect("stream creation failed");
    ZeroCopyRouter::set_threshold(4 * 1024 * 1024);

    match ZeroCopyRouter::new()
        .route(&buffer, &stream)
        .expect("route failed")
    {
        TransferStrategy::ZeroCopy { device_ptr } => {
            assert!(!device_ptr.as_ptr().is_null(), "device pointer is null");
            #[cfg(feature = "mock-cuda")]
            {
                assert_eq!(
                    device_ptr.as_ptr(),
                    buffer.as_ptr() as *mut u8,
                    "mock-cuda mapped pointer should alias host memory"
                );
                let device_view =
                    unsafe { std::slice::from_raw_parts(device_ptr.as_ptr(), buffer.len()) };
                assert_eq!(device_view[0], 0xA5);
                assert_eq!(device_view[buffer.len() - 1], 0x5A);
            }
        }
        TransferStrategy::DmaCopy { .. } => panic!("mapped sub-threshold buffer must zero-copy"),
    }
}

#[test]
fn t7_mapped_threshold_or_larger_buffer_routes_to_dma() {
    let buffer = PinnedBuffer::alloc_mapped(4 * 1024 * 1024).expect("mapped allocation failed");
    let stream = CudaStream::new().expect("stream creation failed");
    ZeroCopyRouter::set_threshold(4 * 1024 * 1024);

    match ZeroCopyRouter::new()
        .route(&buffer, &stream)
        .expect("route failed")
    {
        TransferStrategy::ZeroCopy { .. } => panic!("threshold-sized buffer must DMA-copy"),
        TransferStrategy::DmaCopy { .. } => {}
    }
}

#[test]
fn t7_unmapped_sub_threshold_buffer_falls_back_to_dma() {
    let buffer = PinnedBuffer::alloc(1024 * 1024).expect("allocation failed");
    let stream = CudaStream::new().expect("stream creation failed");
    ZeroCopyRouter::set_threshold(4 * 1024 * 1024);

    match ZeroCopyRouter::new()
        .route(&buffer, &stream)
        .expect("route failed")
    {
        TransferStrategy::ZeroCopy { .. } => panic!("unmapped buffer must not zero-copy"),
        TransferStrategy::DmaCopy { .. } => {}
    }
}

#[test]
fn t7_latency_shape_prefers_zero_copy_below_crossover_and_dma_above() {
    assert!(
        zero_copy_time_score(1024 * 1024) < dma_copy_time_score(1024 * 1024),
        "mock latency shape should prefer zero-copy for sub-threshold tensors"
    );
    assert!(
        zero_copy_time_score(8 * 1024 * 1024) > dma_copy_time_score(8 * 1024 * 1024),
        "mock latency shape should prefer DMA for large tensors"
    );
}
