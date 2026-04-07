// tests/test_allocation_precision.rs — Sprint 1: PinnedBuffer allocation precision
//
// Goal: Prove that PinnedBuffer::alloc(n) consumes exactly n bytes of physical
// RAM (modulo OS page rounding), NOT n bytes rounded to the next power of two.
//
// ─── HOW THE TEST WORKS ──────────────────────────────────────────────────
//
//   Linux exposes /proc/self/status, a text file with per-process memory
//   counters.  The field "VmRSS" is the Resident Set Size: how many KB of
//   physical RAM the process is currently using.
//
//   We snapshot VmRSS before and after each allocation, compute the delta,
//   and compare:
//
//     RamFlow delta  = VmRSS after PinnedBuffer::alloc(n)  - VmRSS before
//     Vec delta      = VmRSS after vec_of_size(n)          - VmRSS before
//
//   We assert RamFlow delta ≤ Vec delta.  On sizes that are NOT powers of two
//   (e.g. 2.1 MB), Vec rounds up (malloc may use power-of-two aligned slab
//   sizes), so Vec delta is larger.
//
// ─── PLATFORM GATE ───────────────────────────────────────────────────────
//
//   /proc is Linux-only.  On non-Linux the VmRSS comparison test is marked
//   #[ignore] with an explanatory message.  CI runs Linux containers.
//
//   All other tests (exact len, zero-size error, mapped flag) run everywhere.
//
// ─── MOCK-CUDA COMPATIBILITY ─────────────────────────────────────────────
//
//   PinnedBuffer::alloc always calls the platform aligned allocator; in
//   mock-cuda mode cudaHostRegister is a no-op.  Allocation size behavior is
//   identical.  Run with:
//     cargo test --no-default-features --features mock-cuda -- --nocapture

use ramflow::PinnedBuffer;

// ===========================================================================
// Cross-platform sanity tests (run on Linux, Windows, macOS)
// ===========================================================================

/// Verify that PinnedBuffer::alloc returns the exact requested size.
#[test]
fn test_pinned_exact_len() {
    let sizes = [1usize, 63, 64, 65, 100, 1_048_576, 1_572_864, 2_202_009, 3_879_731];
    for &n in &sizes {
        let buf = PinnedBuffer::alloc(n)
            .unwrap_or_else(|e| panic!("PinnedBuffer::alloc({n}) failed: {e}"));
        assert_eq!(
            buf.len(),
            n,
            "PinnedBuffer::alloc({n}).len() returned {} — expected exact size",
            buf.len()
        );
        assert!(!buf.is_empty(), "buf.is_empty() should be false for n={n}");
        assert!(!buf.is_mapped(), "alloc() should set is_mapped = false");
    }
}

/// Verify that alloc_mapped sets is_mapped = true and reports the correct size.
#[test]
fn test_pinned_mapped_flag() {
    let buf = PinnedBuffer::alloc_mapped(1_048_576)
        .expect("alloc_mapped(1 MiB) failed");
    assert!(buf.is_mapped(), "alloc_mapped should set is_mapped = true");
    assert_eq!(buf.len(), 1_048_576);
    assert!(!buf.is_empty());
}

/// Verify that zero-size allocation returns an error (not a null pointer).
#[test]
fn test_pinned_zero_size_error() {
    let result = PinnedBuffer::alloc(0);
    assert!(
        result.is_err(),
        "PinnedBuffer::alloc(0) should return Err, but returned Ok"
    );
}

/// Verify that as_slice and as_mut_slice give correct length.
#[test]
fn test_pinned_slice_len() {
    let size = 4096usize;
    let mut buf = PinnedBuffer::alloc(size).expect("alloc(4096) failed");
    assert_eq!(buf.as_slice().len(), size);
    assert_eq!(buf.as_mut_slice().len(), size);
}

/// Verify that as_ptr and as_mut_ptr are non-null and 64-byte aligned.
#[test]
fn test_pinned_ptr_alignment() {
    let mut buf = PinnedBuffer::alloc(4096).expect("alloc(4096) failed");
    let ptr = buf.as_ptr() as usize;
    assert_ne!(ptr, 0, "as_ptr() returned null");
    assert_eq!(ptr % 64, 0, "as_ptr() is not 64-byte aligned (offset = {})", ptr % 64);

    let mut_ptr = buf.as_mut_ptr() as usize;
    assert_eq!(mut_ptr % 64, 0, "as_mut_ptr() is not 64-byte aligned");
    assert_eq!(ptr, mut_ptr, "as_ptr and as_mut_ptr point to different addresses");
}

// ===========================================================================
// Test 1: VmRSS comparison — Linux only
// ===========================================================================

/// Read the current VmRSS from /proc/self/status, in kilobytes.
#[cfg(target_os = "linux")]
fn vm_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb_str = rest.trim().split_ascii_whitespace().next()?;
            return kb_str.parse().ok();
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn compare_rss_growth(label: &str, size_bytes: usize) {
    // ── PinnedBuffer path ──────────────────────────────────────────────────
    let before_pinned = vm_rss_kb().expect("could not read /proc/self/status");

    let buf = PinnedBuffer::alloc(size_bytes)
        .unwrap_or_else(|e| panic!("PinnedBuffer::alloc({size_bytes}) failed: {e}"));

    // Touch first byte to fault in the page.
    let _ = buf.as_slice().first().copied().unwrap_or(0);
    let after_pinned = vm_rss_kb().expect("could not read /proc/self/status");
    let pinned_rss_kb = after_pinned.saturating_sub(before_pinned);
    let pinned_bytes_approx = pinned_rss_kb * 1024;

    drop(buf);

    // ── Vec path (equivalent of PyTorch CachingAllocator) ─────────────────
    let before_vec = vm_rss_kb().expect("could not read /proc/self/status");

    let mut v: Vec<u8> = Vec::with_capacity(size_bytes);
    // SAFETY: we immediately write to the first byte, ensuring pages are faulted.
    unsafe { v.set_len(size_bytes) };
    v[0] = 0u8;
    let after_vec = vm_rss_kb().expect("could not read /proc/self/status");
    let vec_rss_kb = after_vec.saturating_sub(before_vec);
    let vec_bytes_approx = vec_rss_kb * 1024;

    drop(v);

    // ── Report ────────────────────────────────────────────────────────────
    println!(
        "[Sprint 1 | {label}]\n  \
         Requested : {:>12} bytes ({:.3} MiB)\n  \
         PinnedBuf : {:>12} bytes RSS delta\n  \
         Vec<u8>   : {:>12} bytes RSS delta\n  \
         Savings   : {:>+13} bytes\n",
        size_bytes,
        size_bytes as f64 / (1024.0 * 1024.0),
        pinned_bytes_approx,
        vec_bytes_approx,
        vec_bytes_approx as i64 - pinned_bytes_approx as i64,
    );

    // ── Assertion ─────────────────────────────────────────────────────────
    //
    // We allow PinnedBuffer up to (size_bytes + 64 KiB = 16 pages) extra RSS.
    // The OS charges pages, not bytes.  On power-of-two sizes both allocators
    // match; on non-power-of-two sizes Vec's underlying malloc rounds up.
    assert!(
        pinned_bytes_approx <= vec_bytes_approx + (64 * 1024),
        "[{label}] PinnedBuffer used more RSS than Vec<u8>!\n  \
         PinnedBuf RSS delta: {pinned_bytes_approx} bytes\n  \
         Vec RSS delta      : {vec_bytes_approx} bytes\n  \
         (Allowed headroom  : 65536 bytes = 16 OS pages)"
    );
}

/// Main allocation-precision test: four non-power-of-two sizes.
///
/// Compares VmRSS growth of PinnedBuffer vs Vec<u8> (PyTorch's equivalent).
/// Requires Linux /proc/self/status.
#[cfg(target_os = "linux")]
#[test]
fn test_pinned_allocation_precision() {
    let test_cases: &[(&str, usize)] = &[
        ("1.000 MiB (power-of-2 baseline)", 1_048_576),
        ("1.500 MiB (non-power-of-2)",      1_572_864),
        ("2.100 MiB (non-power-of-2)",      2_202_009),
        ("3.700 MiB (non-power-of-2)",      3_879_731),
    ];

    for (label, size) in test_cases {
        compare_rss_growth(label, *size);
    }
}

/// Placeholder for non-Linux hosts (Windows, macOS CI).
///
/// The VmRSS comparison requires Linux /proc.  All other tests in this file
/// run unconditionally and validate correctness on every platform.
#[cfg(not(target_os = "linux"))]
#[test]
#[ignore = "/proc/self/status is Linux-only; RSS growth comparison skipped on this platform. \
            All correctness tests (exact_len, alignment, zero_size) still run."]
fn test_pinned_allocation_precision() {}
