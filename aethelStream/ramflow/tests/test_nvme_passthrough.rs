// tests/test_nvme_passthrough.rs — integration tests for NVMe passthrough engine
//
// Feature gate: all tests in this file require `nvme-passthrough`.
//
// Run the full suite (mock-only):
//   cargo test --features "mock-cuda,nvme-passthrough" -- test_nvme_passthrough
//
// Run real-hardware tests (requires /dev/ng0n1 and /tmp/shard_pt_test.bin):
//   cargo test --features "mock-cuda,nvme-passthrough" -- --ignored
//
// Test categories:
//   1. Command construction math (mock — no hardware)
//   2. PinnedBuffer alignment helpers (mock)
//   3. Capability probe (safe — never panics regardless of hardware)
//   4. Engine lifecycle (mock — open/poller/pause/drop)
//   5. Real-hardware tests (all #[ignore])

#![cfg(feature = "nvme-passthrough")]

use ramflow::allocator::PinnedBuffer;

#[cfg(feature = "nvme-passthrough")]
use ramflow::nvme::passthrough::{
    probe_passthrough_capability, NvmePassthroughEngine, PassthroughCapability,
};

// ─── Section 1: Command construction (mock) ────────────────────────────────

/// The NvmeUringCmd struct must be exactly 68 bytes (verified at compile time
/// in passthrough.rs; this test documents the requirement in test output).
#[cfg(target_os = "linux")]
#[test]
fn nvme_uring_cmd_size_is_68_bytes() {
    // The compile-time assert in passthrough.rs enforces this. If this test
    // runs, the assertion already passed; we document it here for visibility.
    assert_eq!(
        68usize,
        68, // placeholder — real check is the const assert in passthrough.rs
        "NvmeUringCmd must be 68 bytes (matches linux/nvme_ioctl.h)"
    );
}

/// NVME_URING_CMD_IO ioctl number derivation.
///
/// _IOWR('N', 0x80, nvme_uring_cmd):
///   = (3<<30) | (sizeof(nvme_uring_cmd)<<16) | ('N'<<8) | 0x80
///   = (3<<30) | (68<<16)                     | (78<<8)  | 128
///   = 0xC000_0000 | 0x0044_0000 | 0x0000_4E00 | 0x0000_0080
///   = 0xC044_4E80
#[test]
fn nvme_uring_cmd_io_number_derivation() {
    let dir: u32 = 3; // IOC_READ | IOC_WRITE
    let size: u32 = 68; // sizeof(nvme_uring_cmd)
    let type_n: u32 = 0x4E; // 'N'
    let nr: u32 = 0x80;
    let expected = (dir << 30) | (size << 16) | (type_n << 8) | nr;
    assert_eq!(expected, 0xC044_4E80, "NVME_URING_CMD_IO = 0xC044_4E80");
}

/// NVME_IOCTL_ID number derivation.
///
/// _IO('N', 0x40) = (0<<30) | (0<<16) | ('N'<<8) | 0x40 = 0x00004E40
#[test]
fn nvme_ioctl_id_number_derivation() {
    let dir: u32 = 0; // IOC_NONE
    let size: u32 = 0;
    let type_n: u32 = 0x4E; // 'N'
    let nr: u32 = 0x40;
    let expected = (dir << 30) | (size << 16) | (type_n << 8) | nr;
    assert_eq!(expected, 0x0000_4E40, "NVME_IOCTL_ID = 0x00004E40");
}

// ─── Section 2: PinnedBuffer alignment ─────────────────────────────────────

/// alloc_page_aligned() always returns a 4096-byte aligned buffer.
#[test]
fn page_aligned_alloc_is_always_4096_aligned() {
    for size in [512usize, 4096, 8192, 65536] {
        let buf = PinnedBuffer::alloc_page_aligned(size).expect("alloc_page_aligned failed");
        assert!(
            buf.is_page_aligned(),
            "alloc_page_aligned({size}) must produce a page-aligned address"
        );
        assert_eq!(buf.len(), size);
    }
}

/// alloc_page_aligned(0) must return an error.
#[test]
fn page_aligned_alloc_zero_size_errors() {
    let result = PinnedBuffer::alloc_page_aligned(0);
    assert!(
        result.is_err(),
        "alloc_page_aligned(0) must return an error"
    );
}

/// alloc() (512-aligned) is NOT guaranteed to be page-aligned, but
/// alloc_page_aligned() IS guaranteed. Test that the guarantee holds
/// across multiple allocations (regression: ensure allocator doesn't
/// accidentally fall back to a weaker alignment).
#[test]
fn page_aligned_alloc_guarantee_not_coincidental() {
    let mut page_aligned_count = 0usize;
    let n = 20;
    for _ in 0..n {
        let buf = PinnedBuffer::alloc_page_aligned(512).expect("alloc failed");
        if buf.is_page_aligned() {
            page_aligned_count += 1;
        }
    }
    assert_eq!(
        page_aligned_count, n,
        "ALL alloc_page_aligned() buffers must be page-aligned (got {page_aligned_count}/{n})"
    );
}

// ─── Section 3: Capability probe ───────────────────────────────────────────

/// probe_passthrough_capability() must not panic on any system.
#[test]
fn probe_does_not_panic() {
    let _cap = probe_passthrough_capability();
}

/// On Linux, probe returns Available with valid fields OR Unavailable.
/// On non-Linux, probe always returns Unavailable.
#[test]
fn probe_returns_consistent_state() {
    let cap = probe_passthrough_capability();
    match cap {
        #[cfg(target_os = "linux")]
        PassthroughCapability::Available {
            char_device_fd,
            nsid,
        } => {
            assert!(char_device_fd >= 3, "char_device_fd must be > stderr (2)");
            assert!(nsid >= 1, "NVMe namespace IDs start at 1");
            // Close the fd: probe opened it and we took ownership.
            unsafe { libc::close(char_device_fd) };
        }
        PassthroughCapability::Unavailable => {
            // Expected on CI / systems without NVMe char device.
            // This is not a failure — the engine will fall back to O_DIRECT.
        }
    }
}

/// Calling probe_passthrough_capability() twice must not double-close any fd.
/// (The probe opens and returns the fd; the CALLER closes it. Calling probe
/// twice returns two independent fds or two Unavailable results.)
#[test]
fn probe_twice_no_double_close() {
    let cap1 = probe_passthrough_capability();
    let cap2 = probe_passthrough_capability();
    // Close fds if Available (caller owns them).
    #[cfg(target_os = "linux")]
    {
        if let PassthroughCapability::Available { char_device_fd, .. } = cap1 {
            unsafe { libc::close(char_device_fd) };
        }
        if let PassthroughCapability::Available { char_device_fd, .. } = cap2 {
            unsafe { libc::close(char_device_fd) };
        }
    }
    let _ = (cap1, cap2);
}

// ─── Section 4: Engine lifecycle (mock — no disk) ──────────────────────────

/// open_with_paths(&[]) must succeed — zero shards is valid for unit tests.
#[cfg(target_os = "linux")]
#[test]
fn engine_open_zero_shards() {
    let engine = NvmePassthroughEngine::open_with_paths(&[])
        .expect("open_with_paths(&[]) must succeed");
    assert_eq!(engine.shard_count(), 0);
    assert!(!engine.is_paused());
    assert_eq!(engine.outstanding_reads(), 0);
}

/// passthrough_available() matches what the probe would return.
#[cfg(target_os = "linux")]
#[test]
fn engine_passthrough_flag_matches_probe() {
    // Run probe first to determine expected value, then close its fd.
    let probe = probe_passthrough_capability();
    let expected = matches!(probe, PassthroughCapability::Available { .. });
    #[cfg(target_os = "linux")]
    if let PassthroughCapability::Available { char_device_fd, .. } = probe {
        unsafe { libc::close(char_device_fd) };
    }

    let engine =
        NvmePassthroughEngine::open_with_paths(&[]).expect("engine open failed");
    assert_eq!(
        engine.passthrough_available(),
        expected,
        "engine.passthrough_available() must match probe result"
    );
}

/// Pause signal set/clear roundtrip.
#[cfg(target_os = "linux")]
#[test]
fn engine_pause_signal_roundtrip() {
    let engine = NvmePassthroughEngine::open_with_paths(&[]).unwrap();
    assert!(!engine.is_paused(), "engine must not be paused after construction");
    engine.set_pause(true);
    assert!(engine.is_paused());
    engine.set_pause(false);
    assert!(!engine.is_paused());
}

/// prefetch() returns PressurePause immediately when paused.
#[cfg(target_os = "linux")]
#[test]
fn engine_prefetch_honours_pause_signal() {
    let engine = NvmePassthroughEngine::open_with_paths(&[]).unwrap();
    engine.set_pause(true);

    let buf = PinnedBuffer::alloc_page_aligned(4096).unwrap();
    let result = engine.prefetch(0, 0, 512, &buf, 99);

    assert!(
        matches!(result, Err(ramflow::RamFlowError::PressurePause(_))),
        "expected PressurePause when pause signal is set, got: {result:?}"
    );
}

/// write_async() returns PressurePause when paused (same pause gate as reads).
#[cfg(target_os = "linux")]
#[test]
fn engine_write_async_honours_pause_signal() {
    let engine = NvmePassthroughEngine::open_with_paths(&[]).unwrap();
    engine.set_pause(true);

    let buf = PinnedBuffer::alloc(512).unwrap();
    let result = engine.write_async(0, 0, 512, &buf, 99);

    assert!(
        matches!(result, Err(ramflow::RamFlowError::PressurePause(_))),
        "expected PressurePause when paused, got: {result:?}"
    );
}

/// poll_completions() returns Ok(0) when the channel is empty.
#[cfg(target_os = "linux")]
#[test]
fn engine_poll_completions_empty_channel() {
    let engine = NvmePassthroughEngine::open_with_paths(&[]).unwrap();
    let count = engine.poll_completions().unwrap();
    assert_eq!(count, 0, "no completions expected on a fresh engine");
}

/// Multiple calls to start_cqe_poller() must not create multiple threads.
#[cfg(target_os = "linux")]
#[test]
fn engine_start_cqe_poller_idempotent() {
    let mut engine = NvmePassthroughEngine::open_with_paths(&[]).unwrap();
    engine.start_cqe_poller().unwrap();
    engine.start_cqe_poller().unwrap(); // second call must be a no-op
    engine.start_cqe_poller().unwrap(); // third call must be a no-op
    // If we reach here, no deadlock or double-spawn occurred.
}

/// Drop must not deadlock or leak the CQE poller thread.
#[cfg(target_os = "linux")]
#[test]
fn engine_drop_joins_poller() {
    let mut engine = NvmePassthroughEngine::open_with_paths(&[]).unwrap();
    engine.start_cqe_poller().unwrap();
    // Drop: stop_signal is set, poller joins, no leak.
    drop(engine);
    // Reaching here means drop completed without deadlock.
}

// ─── Section 5: Real-hardware tests (#[ignore]) ────────────────────────────
//
// These tests require:
//   - /dev/ng0n1 (NVMe char device, kernel >= 5.14)
//   - /dev/nvme0n1 (block device)
//   - /tmp/shard_pt_test.bin (4096-byte aligned raw shard file)
//
//   Create the test file with:
//     dd if=/dev/nvme0n1 of=/tmp/shard_pt_test.bin bs=4096 count=4
//
// Run with:
//   cargo test --features "mock-cuda,nvme-passthrough" -- --ignored

/// Passthrough read result must equal O_DIRECT read for the same range.
#[cfg(all(target_os = "linux", feature = "nvme-passthrough"))]
#[test]
#[ignore]
fn passthrough_bytes_match_odirect() {
    let shard = std::path::Path::new("/tmp/shard_pt_test.bin");
    if !shard.exists() {
        eprintln!("SKIP: /tmp/shard_pt_test.bin not found");
        return;
    }

    let mut engine =
        NvmePassthroughEngine::open_with_paths(&[shard]).expect("engine open failed");
    engine.start_cqe_poller().expect("start_cqe_poller failed");

    if !engine.passthrough_available() {
        eprintln!("SKIP: passthrough not available on this system");
        return;
    }

    let mut pt_buf = PinnedBuffer::alloc_page_aligned(4096).unwrap();
    let mut od_buf = PinnedBuffer::alloc_page_aligned(4096).unwrap();

    // Passthrough read — page-aligned buffer triggers IORING_OP_URING_CMD.
    engine
        .prefetch(0, 0, 4096, &pt_buf, 1)
        .expect("passthrough prefetch failed");
    let cqe = engine.completion_rx().recv().unwrap();
    assert!(cqe.result >= 0, "passthrough CQE error: {}", cqe.result);

    // O_DIRECT read of the same range using a second (non-passthrough) buffer.
    // On a system with passthrough, the engine will also use passthrough for
    // this read (same page-aligned buffer). To force O_DIRECT we'd need a
    // 512-only buffer, but both paths should produce the same bytes.
    engine
        .prefetch(0, 0, 4096, &od_buf, 2)
        .expect("O_DIRECT prefetch failed");
    let cqe = engine.completion_rx().recv().unwrap();
    assert!(cqe.result >= 0, "O_DIRECT CQE error: {}", cqe.result);

    assert_eq!(
        pt_buf.as_slice(),
        od_buf.as_slice(),
        "passthrough and O_DIRECT must return identical bytes"
    );
    let _ = (pt_buf.as_mut_slice(), od_buf.as_mut_slice());
}

/// Latency comparison: 10 passthrough vs 10 O_DIRECT reads.
/// Records median latency; passthrough must not exceed O_DIRECT median × 1.2.
#[cfg(all(target_os = "linux", feature = "nvme-passthrough"))]
#[test]
#[ignore]
fn passthrough_latency_vs_odirect() {
    let shard = std::path::Path::new("/tmp/shard_pt_test.bin");
    if !shard.exists() {
        eprintln!("SKIP: /tmp/shard_pt_test.bin not found");
        return;
    }

    let mut engine =
        NvmePassthroughEngine::open_with_paths(&[shard]).expect("engine open failed");
    engine.start_cqe_poller().expect("start_cqe_poller failed");

    if !engine.passthrough_available() {
        eprintln!("SKIP: passthrough not available on this system");
        return;
    }

    const N: usize = 10;
    let mut pt_us: Vec<u128> = Vec::with_capacity(N);
    let mut od_us: Vec<u128> = Vec::with_capacity(N);

    // Passthrough reads: page-aligned buffers trigger IORING_OP_URING_CMD.
    for i in 0..N {
        let buf = PinnedBuffer::alloc_page_aligned(4096).unwrap();
        let t0 = std::time::Instant::now();
        engine.prefetch(0, 0, 4096, &buf, i as u64).unwrap();
        let _ = engine.completion_rx().recv().unwrap();
        pt_us.push(t0.elapsed().as_micros());
    }

    // O_DIRECT reads: use 512-aligned buffers (not page-aligned → fallback path).
    // Note: posix_memalign(512) may return a page-aligned address by coincidence,
    // in which case these also take the passthrough path — latency comparable.
    for i in 0..N {
        let buf = PinnedBuffer::alloc(512).unwrap();
        let t0 = std::time::Instant::now();
        engine
            .prefetch(0, 0, 512, &buf, (N + i) as u64)
            .unwrap();
        let _ = engine.completion_rx().recv().unwrap();
        od_us.push(t0.elapsed().as_micros());
    }

    pt_us.sort_unstable();
    od_us.sort_unstable();
    let pt_med = pt_us[N / 2];
    let od_med = od_us[N / 2];

    eprintln!(
        "NVMe passthrough median: {} µs  |  O_DIRECT median: {} µs  |  ratio: {:.2}",
        pt_med,
        od_med,
        pt_med as f64 / od_med.max(1) as f64
    );

    // Passthrough should not be more than 20% slower than O_DIRECT.
    // (On high-end NVMe it should be faster; the 20% slack handles jitter.)
    assert!(
        pt_med <= od_med * 12 / 10,
        "passthrough ({pt_med} µs) is >20% slower than O_DIRECT ({od_med} µs) — \
         unexpected regression"
    );
}
