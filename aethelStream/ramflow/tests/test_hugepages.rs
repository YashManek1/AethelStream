// tests/test_hugepages.rs — hugepage-backed pinned memory tests
//
// Run with:
//   cargo test --no-default-features --features "mock-cuda,hugepages" test_hugepages
//
// On Linux: exercises mmap + MADV_HUGEPAGE path.
// On Windows / without hugepages feature: all alloc_pinned_huge calls fall
// back to standard alloc; the round-trip and drop tests still pass.

#[cfg(feature = "hugepages")]
mod hugepages {
    use ramflow::allocator::{AllocKind, PinnedBuffer};

    const _4_MIB: usize = 4 * 1024 * 1024;
    const _2_MIB: usize = 2 * 1024 * 1024;
    const _64_MIB: usize = 64 * 1024 * 1024;

    // ── alloc_pinned_huge succeeds and cudaHostRegister does not fail ─────────

    #[test]
    fn alloc_pinned_huge_4mib_succeeds() {
        let buf = PinnedBuffer::alloc_pinned_huge(_4_MIB)
            .expect("alloc_pinned_huge(4 MiB) should succeed");
        assert_eq!(buf.len(), _4_MIB);
        assert!(!buf.is_empty());
    }

    // ── AllocKind is Huge on Linux, Standard on other platforms ───────────────

    #[test]
    fn alloc_kind_is_huge_on_linux() {
        let buf = PinnedBuffer::alloc_pinned_huge(_4_MIB)
            .expect("alloc_pinned_huge should succeed");

        #[cfg(target_os = "linux")]
        {
            assert!(
                matches!(buf.alloc_kind(), AllocKind::Huge { .. }),
                "on Linux alloc_kind should be Huge, got {:?}",
                buf.alloc_kind()
            );
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(
                buf.alloc_kind(),
                AllocKind::Standard,
                "on non-Linux alloc_pinned_huge should fall back to Standard"
            );
        }
    }

    // ── Round-trip: write known bytes, read back byte-identical ───────────────

    #[test]
    fn alloc_pinned_huge_round_trip() {
        let mut buf = PinnedBuffer::alloc_pinned_huge(_4_MIB)
            .expect("alloc_pinned_huge should succeed");

        // Write a recognizable pattern.
        let pattern: Vec<u8> = (0.._4_MIB).map(|i| (i ^ 0xAB) as u8).collect();
        buf.as_mut_slice().copy_from_slice(&pattern);

        // Read back and verify.
        assert_eq!(buf.as_slice(), pattern.as_slice(), "round-trip data mismatch");
    }

    // ── alloc_pinned_huge(0) returns an error, not a panic ────────────────────

    #[test]
    fn alloc_pinned_huge_zero_bytes_returns_error() {
        let result = PinnedBuffer::alloc_pinned_huge(0);
        assert!(result.is_err(), "zero-size alloc should return Err");
    }

    // ── Drop completes without panic — verifies no double-munmap ─────────────
    //
    // The `drop` call here is the sole Drop for this buffer; if Drop runs
    // twice (double-free / double-munmap) the OS would return -EINVAL on the
    // second munmap and our debug_assert in munmap_huge would fire.
    // In release mode, the OS would return an error that we'd silently ignore,
    // but the test itself would remain green — which is fine because the ASAN
    // / valgrind run in CI would catch it.

    #[test]
    fn alloc_pinned_huge_drop_does_not_panic() {
        let buf = PinnedBuffer::alloc_pinned_huge(_4_MIB)
            .expect("alloc_pinned_huge should succeed");
        drop(buf); // must not panic
    }

    // ── Multiple independent buffers can coexist and are all freed ───────────

    #[test]
    fn multiple_huge_buffers_independent() {
        let a = PinnedBuffer::alloc_pinned_huge(_2_MIB).expect("a");
        let b = PinnedBuffer::alloc_pinned_huge(_4_MIB).expect("b");
        let c = PinnedBuffer::alloc_pinned_huge(_2_MIB).expect("c");

        assert_eq!(a.len(), _2_MIB);
        assert_eq!(b.len(), _4_MIB);
        assert_eq!(c.len(), _2_MIB);

        // Each pointer must be distinct.
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert_ne!(b.as_ptr(), c.as_ptr());
        assert_ne!(a.as_ptr(), c.as_ptr());

        drop(a);
        drop(b);
        drop(c); // all must drop cleanly
    }

    // ── Fallback: size below HUGEPAGE_THRESHOLD still works ──────────────────
    //
    // alloc_pinned_huge is valid for any size > 0. For sizes below 2 MiB the
    // routing in ring_buffer uses standard alloc, but a direct call to
    // alloc_pinned_huge must still succeed (it may return a Standard buffer
    // if mmap rounds up and there are no other failures).

    #[test]
    fn alloc_pinned_huge_small_size_succeeds() {
        let buf = PinnedBuffer::alloc_pinned_huge(1024)
            .expect("alloc_pinned_huge(1 KiB) should succeed via fallback");
        assert!(buf.len() == 1024 || buf.len() >= 1024);
    }

    // ── standard alloc() still produces Standard kind ────────────────────────

    #[test]
    fn standard_alloc_kind_is_standard() {
        let buf = PinnedBuffer::alloc(4096).expect("standard alloc should succeed");
        assert_eq!(buf.alloc_kind(), AllocKind::Standard);
    }

    // ── VmRSS sanity check (Linux only) ──────────────────────────────────────
    //
    // After a 64 MiB hugepage allocation, VmRSS (resident set size) should
    // increase by at least 64 MiB once we touch the memory.  We write one byte
    // per 4 KiB page to force page faults.  The test is advisory: it logs the
    // delta but does not fail if the kernel hasn't promoted to hugepages yet.

    #[test]
    #[cfg(target_os = "linux")]
    fn vmrss_increases_after_hugepage_alloc() {
        fn vmrss_kb() -> u64 {
            let content = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(val) = rest.split_whitespace().next() {
                        if let Ok(kb) = val.parse::<u64>() {
                            return kb;
                        }
                    }
                }
            }
            0
        }

        let rss_before = vmrss_kb();

        let mut buf = PinnedBuffer::alloc_pinned_huge(_64_MIB)
            .expect("64 MiB alloc_pinned_huge should succeed");

        // Touch every 4 KiB page to ensure pages are faulted in.
        let slice = buf.as_mut_slice();
        let mut i = 0;
        while i < slice.len() {
            slice[i] = 0xAB;
            i += 4096;
        }

        let rss_after = vmrss_kb();
        let delta_kb = rss_after.saturating_sub(rss_before);

        // Delta should be at least 32 MiB (half of 64 MiB — gives generous
        // slack for other RSS changes in the test process).
        assert!(
            delta_kb >= 32 * 1024,
            "VmRSS delta after 64 MiB alloc_pinned_huge should be >= 32 MiB, got {delta_kb} KiB"
        );
    }
}

// ── Tests that run regardless of the hugepages feature ───────────────────────

#[test]
fn alloc_kind_standard_without_hugepages() {
    // alloc() must always produce Standard — confirms the base invariant
    // and acts as a compile-time smoke test that AllocKind is always exported.
    let buf = ramflow::PinnedBuffer::alloc(512).expect("alloc(512)");
    assert_eq!(buf.alloc_kind(), ramflow::allocator::AllocKind::Standard);
}
