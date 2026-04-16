// tests/test_pool_exhaustion.rs — Sprint 2 pressure control integration

#[cfg(target_os = "linux")]
use ramflow::{DirectNvmeEngine, PinnedBuffer, RamFlowError};

#[cfg(target_os = "linux")]
#[test]
fn test_pressure_control_stops_submissions_before_overflow() {
    use std::io::Write;

    let path = std::path::PathBuf::from(format!(
        "/tmp/ramflow_pressure_test_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock error")
            .as_nanos()
    ));

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .expect("failed to create temp shard file");
    let content = vec![0xA5u8; 8192];
    file.write_all(&content).expect("failed to write shard");
    file.sync_all().expect("failed to sync shard");

    let engine =
        DirectNvmeEngine::open_with_paths(&[path.as_path()]).expect("failed to open engine");

    // Force pressure gate to trip before submission.
    engine.set_pressure_threshold(0);
    engine.set_claimed_slots(1);

    let mut dst = PinnedBuffer::alloc(1536).expect("PinnedBuffer alloc failed");
    let result = engine.prefetch(0, 512, 1536, &dst, 0xBEEF);

    match result {
        Err(RamFlowError::PressurePause(_)) => {}
        other => panic!("expected PressurePause, got {other:?}"),
    }

    assert_eq!(
        engine.outstanding_reads(),
        0,
        "no IO should be in-flight when gate blocks before submission"
    );

    let _ = std::fs::remove_file(path);
}

#[cfg(not(target_os = "linux"))]
use ramflow::nvme::fd_table::FdTable;
#[cfg(not(target_os = "linux"))]
use ramflow::nvme::io_uring_setup::{IoUringInstance, IoUringParams};
#[cfg(not(target_os = "linux"))]
use ramflow::nvme::prefetch::PrefetchEngine;
#[cfg(not(target_os = "linux"))]
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(not(target_os = "linux"))]
use std::sync::Arc;

#[cfg(not(target_os = "linux"))]
#[test]
fn test_pressure_control_stops_submissions_before_overflow() {
    let ring = Arc::new(IoUringInstance::setup(IoUringParams::default()).expect("ring setup"));
    let fd_table = Arc::new(FdTable::new().expect("fd table"));
    let pause = Arc::new(AtomicBool::new(false));
    let outstanding = Arc::new(AtomicUsize::new(128));
    let claimed = Arc::new(AtomicUsize::new(4));
    let threshold = Arc::new(AtomicUsize::new(128));

    let prefetch = PrefetchEngine::new(ring, fd_table, pause, outstanding, claimed, threshold)
        .expect("prefetch engine setup");

    assert!(
        !prefetch.submission_allowed(),
        "submission should be blocked when outstanding+claimed exceeds threshold"
    );
}
