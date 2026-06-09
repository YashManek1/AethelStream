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

#[test]
fn test_t3_capacity_plus_one_simultaneous_claim_has_no_double_issue_or_deadlock() {
    use ramflow::phase::{PhaseMemoryProfile, TrainingPhase};
    use ramflow::pool::{LayerKind, PoolRegistry, TensorLocationDict};
    use std::collections::HashSet;
    use std::sync::{mpsc, Arc, Barrier};
    use std::time::Duration;

    const CAPACITY: usize = 4;
    let profile = PhaseMemoryProfile {
        phase: TrainingPhase::Forward {
            layers_in_flight: CAPACITY as u32,
        },
        expected_peak_bytes: 0,
        attention_slots_needed: CAPACITY as u32,
        mlp_slots_needed: 1,
        norm_slots_needed: 1,
        optimizer_slots_needed: 1,
    };
    let registry = Arc::new(
        PoolRegistry::new(&profile, &TensorLocationDict::empty(), 1024)
            .expect("registry construction failed"),
    );
    let barrier = Arc::new(Barrier::new(CAPACITY + 1));
    let (sender, receiver) = mpsc::channel();

    for thread_index in 0..(CAPACITY + 1) {
        let thread_registry = Arc::clone(&registry);
        let thread_barrier = Arc::clone(&barrier);
        let thread_sender = sender.clone();
        std::thread::spawn(move || {
            thread_barrier.wait();
            let slot = thread_registry
                .claim(LayerKind::Attention)
                .expect("claim failed");
            let slot_index = slot.slot_index();
            let buffer_ptr = slot.buffer().as_ptr() as usize;
            thread_sender
                .send((thread_index, slot_index, buffer_ptr, slot))
                .expect("receiver dropped");
        });
    }
    drop(sender);

    let mut results = Vec::new();
    for _ in 0..(CAPACITY + 1) {
        results.push(
            receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("claim race deadlocked"),
        );
    }

    let pooled: Vec<_> = results
        .iter()
        .filter(|(_thread_index, slot_index, _buffer_ptr, _slot)| *slot_index != usize::MAX)
        .collect();
    let slow_path: Vec<_> = results
        .iter()
        .filter(|(_thread_index, slot_index, _buffer_ptr, _slot)| *slot_index == usize::MAX)
        .collect();

    assert_eq!(
        pooled.len(),
        CAPACITY,
        "exactly the fixed ring capacity should receive pooled slots"
    );
    assert_eq!(
        slow_path.len(),
        1,
        "capacity+1 claim should route exactly one claimant through slow path overflow"
    );

    let pooled_slot_indices: HashSet<usize> = pooled
        .iter()
        .map(|(_thread_index, slot_index, _buffer_ptr, _slot)| *slot_index)
        .collect();
    assert_eq!(
        pooled_slot_indices.len(),
        CAPACITY,
        "pooled slot index double-issued during simultaneous claim"
    );

    let pooled_buffer_ptrs: HashSet<usize> = pooled
        .iter()
        .map(|(_thread_index, _slot_index, buffer_ptr, _slot)| *buffer_ptr)
        .collect();
    assert_eq!(
        pooled_buffer_ptrs.len(),
        CAPACITY,
        "pooled buffer pointer double-issued during simultaneous claim"
    );

    drop(results);
    assert_eq!(
        registry.claimed_slots_for(LayerKind::Attention),
        0,
        "all pooled slots should return after guards drop"
    );
}
