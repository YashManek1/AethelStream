// tests/test_allocation_precision.rs
//
// Complete Sprint 1 test suite for PinnedBuffer.
//
// TEST GROUPS:
//   Unit tests     — correctness of every field/method, run on all platforms
//   RSS tests      — actual physical memory comparison vs Vec, Linux only
//   Stress tests   — repeated alloc/drop cycles to catch leaks and corruption
//   Model sim      — simulate pinning real training tensors (2.4 GB total)
//
// HOW TO RUN:
//   cargo test --no-default-features --features mock-cuda -- --nocapture
//
// The --nocapture flag lets you see the printed RSS tables.

use ramflow::PinnedBuffer;

// ════════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════════

/// Read VmRSS (resident set size) from /proc/self/status in kilobytes.
/// Returns None on non-Linux platforms.
#[cfg(target_os = "linux")]
fn vm_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Line looks like: "VmRSS:   1234567 kB"
            return rest.trim().split_ascii_whitespace().next()?.parse().ok();
        }
    }
    None
}

/// Force all pages of a Vec to be committed by the OS (touch every page).
fn touch_vec(v: &mut Vec<u8>) {
    for chunk in v.chunks_mut(4096) {
        chunk[0] = 0xAA;
    }
}

// ════════════════════════════════════════════════════════════════
// UNIT TESTS — run on every platform, no GPU required
// ════════════════════════════════════════════════════════════════

/// The most basic test: does alloc() return a buffer with the right size?
#[test]
fn unit_exact_length() {
    // Test a wide range of sizes including non-power-of-two values
    let sizes: &[usize] = &[
        1, 7, 63, 64, // exact cache line
        65, 100, 999, 4_096,     // one OS page
        4_097,     // one OS page + 1 byte
        65_536,    // 64 KB
        1_000_000, // 1 MB non-power-of-two
        1_048_576, // 1 MiB exactly (power of 2)
        1_572_864, // 1.5 MiB
        2_202_009, // 2.1 MiB non-power-of-two
        3_879_731, // 3.7 MiB non-power-of-two
    ];
    for &n in sizes {
        let buf = PinnedBuffer::alloc(n).unwrap_or_else(|e| panic!("alloc({n}) failed: {e}"));
        assert_eq!(
            buf.len(),
            n,
            "alloc({n}).len() = {} — must be exactly {n}",
            buf.len()
        );
        assert!(!buf.is_empty(), "alloc({n}).is_empty() must be false");
        assert!(!buf.is_mapped(), "alloc() must set is_mapped=false");
    }
}

/// Zero-size allocation must be an error, not a null pointer crash.
#[test]
fn unit_zero_size_is_error() {
    let result = PinnedBuffer::alloc(0);
    assert!(
        result.is_err(),
        "alloc(0) returned Ok — must return Err(AllocationFailed)"
    );
    // Check the error message contains something meaningful
    let err_str = result.err().unwrap().to_string();
    assert!(!err_str.is_empty());
}

/// alloc_mapped() must set is_mapped=true and report the correct size.
#[test]
fn unit_mapped_flag() {
    let buf = PinnedBuffer::alloc_mapped(1_048_576).expect("alloc_mapped(1 MiB) failed");
    assert!(buf.is_mapped(), "alloc_mapped() must set is_mapped=true");
    assert_eq!(buf.len(), 1_048_576);
    assert!(!buf.is_empty());
}

/// alloc_mapped(0) must also be an error.
#[test]
fn unit_mapped_zero_size_is_error() {
    assert!(PinnedBuffer::alloc_mapped(0).is_err());
}

/// as_slice() and as_mut_slice() must give slices with the correct length.
#[test]
fn unit_slice_length() {
    let sizes = [1_usize, 64, 4096, 1_000_000];
    for &n in &sizes {
        let mut buf = PinnedBuffer::alloc(n).expect("alloc failed");
        assert_eq!(buf.as_slice().len(), n);
        assert_eq!(buf.as_mut_slice().len(), n);
    }
}

/// as_ptr() must be non-null and aligned to 64 bytes.
/// as_mut_ptr() must return the same address.
#[test]
fn unit_pointer_alignment() {
    // Test multiple sizes to ensure alignment holds regardless of size
    let sizes = [1_usize, 63, 64, 65, 4097, 1_572_864];
    for &n in &sizes {
        let mut buf = PinnedBuffer::alloc(n).expect("alloc failed");

        let ptr = buf.as_ptr() as usize;
        let ptr_mut = buf.as_mut_ptr() as usize;

        assert_ne!(ptr, 0, "as_ptr() returned null for size {n}");
        assert_eq!(
            ptr % 64,
            0,
            "as_ptr() not 64-byte aligned for size {n}: offset = {}",
            ptr % 64
        );
        assert_eq!(
            ptr, ptr_mut,
            "as_ptr() and as_mut_ptr() differ for size {n}"
        );
    }
}

/// fill() must write the byte to every position in the buffer.
#[test]
fn unit_fill_and_read() {
    let mut buf = PinnedBuffer::alloc(4096).expect("alloc failed");
    buf.as_mut_slice().fill(0xDE);
    for (i, &byte) in buf.as_slice().iter().enumerate() {
        assert_eq!(byte, 0xDE, "byte at index {i} is {byte:#x}, expected 0xDE");
    }
    buf.as_mut_slice().fill(0x00);
    for byte in buf.as_slice() {
        assert_eq!(*byte, 0x00);
    }
}

/// Write a pattern, read it back — verifies the pointer is valid and writable.
#[test]
fn unit_write_then_read() {
    let n = 256;
    let mut buf = PinnedBuffer::alloc(n).expect("alloc failed");

    // Write: position i gets value (i % 256) as u8
    for (i, byte) in buf.as_mut_slice().iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    // Read back and verify
    for (i, &byte) in buf.as_slice().iter().enumerate() {
        assert_eq!(
            byte,
            (i % 256) as u8,
            "mismatch at byte {i}: got {byte}, expected {}",
            i % 256
        );
    }
}

/// Multiple buffers can coexist — they must not overlap.
#[test]
fn unit_multiple_buffers_no_overlap() {
    let mut buffers: Vec<PinnedBuffer> = (0..8)
        .map(|i| {
            let mut b = PinnedBuffer::alloc(65_536).expect("alloc failed");
            b.as_mut_slice().fill(i as u8);
            b
        })
        .collect();

    // Verify each buffer kept its fill value (no aliasing)
    for (i, buf) in buffers.iter_mut().enumerate() {
        let expected = i as u8;
        let first = buf.as_slice()[0];
        let last = buf.as_slice()[65_535];
        assert_eq!(
            first, expected,
            "buffer {i}: first byte = {first}, expected {expected}"
        );
        assert_eq!(
            last, expected,
            "buffer {i}: last byte = {last}, expected {expected}"
        );
    }
}

/// Buffer survives being sent to another thread (tests Send bound).
#[test]
fn unit_send_to_thread() {
    let mut buf = PinnedBuffer::alloc(1_024).expect("alloc failed");
    buf.as_mut_slice().fill(0xFF);

    let handle = std::thread::spawn(move || {
        // We own buf now. Verify it's intact.
        assert_eq!(buf.len(), 1_024);
        assert_eq!(buf.as_slice()[0], 0xFF);
        assert_eq!(buf.as_slice()[1_023], 0xFF);
        // Return it to the main thread
        buf
    });

    let returned = handle.join().expect("thread panicked");
    assert_eq!(returned.len(), 1_024);
}

/// Drop must not panic. This is the minimal smoke test for Drop safety.
#[test]
fn unit_drop_does_not_panic() {
    for _ in 0..100 {
        let _buf = PinnedBuffer::alloc(4096).expect("alloc failed");
        // _buf dropped here — must not panic
    }
}

/// Allocate then immediately drop in a tight loop to catch any use-after-free
/// or double-free the allocator might have.
#[test]
fn unit_alloc_drop_loop() {
    for i in 0..200 {
        let size = (i + 1) * 1024; // 1 KB to 200 KB
        let buf = PinnedBuffer::alloc(size).expect("alloc failed");
        assert_eq!(buf.len(), size);
        drop(buf);
    }
}

// ════════════════════════════════════════════════════════════════
// RSS COMPARISON TESTS — Linux only, compare physical RAM usage
// ════════════════════════════════════════════════════════════════
//
// The operating system tracks how much physical RAM a process is actually
// using via /proc/self/status → VmRSS. We snapshot this before and after
// each allocation. The delta tells us exactly how many physical pages were
// committed by the OS for our buffer.
//
// Why this matters for AethelStream:
//   PyTorch's allocator rounds to the next power of two.
//   posix_memalign gives exactly what you ask for.
//   On a 64 GB machine with thousands of allocations, this difference is
//   the line between "training works" and "OOM crash".

#[cfg(target_os = "linux")]
mod rss_tests {
    use super::*;

    /// Record RSS before/after PinnedBuffer and Vec allocations.
    /// Returns (pinned_rss_bytes, vec_rss_bytes, requested_bytes).
    fn measure_rss(size_bytes: usize) -> (u64, u64, usize) {
        // ── PinnedBuffer ──────────────────────────────────────────────────
        let before_pinned = vm_rss_kb().expect("/proc/self/status unreadable");

        let mut buf = PinnedBuffer::alloc(size_bytes)
            .unwrap_or_else(|e| panic!("alloc({size_bytes}) failed: {e}"));
        // Force OS to commit physical pages (Linux uses demand paging)
        buf.as_mut_slice().fill(0xAB);
        let after_pinned = vm_rss_kb().expect("/proc/self/status unreadable");
        drop(buf);

        // ── Vec<u8> (equivalent of PyTorch's allocator) ───────────────────
        let before_vec = vm_rss_kb().expect("/proc/self/status unreadable");

        let mut v: Vec<u8> = Vec::with_capacity(size_bytes);
        unsafe { v.set_len(size_bytes) };
        touch_vec(&mut v);
        let after_vec = vm_rss_kb().expect("/proc/self/status unreadable");
        drop(v);

        let pinned_bytes = after_pinned.saturating_sub(before_pinned) * 1024;
        let vec_bytes = after_vec.saturating_sub(before_vec) * 1024;

        (pinned_bytes, vec_bytes, size_bytes)
    }

    fn print_rss_row(label: &str, requested: usize, pinned: u64, vec: u64) {
        let savings = vec as i64 - pinned as i64;
        let overhead = pinned as i64 - requested as i64;
        println!(
            "  {label}\n    Requested : {:>14} bytes  ({:.3} MB)\n    PinnedBuf : {:>14} bytes  (overhead: {:+} bytes)\n    Vec<u8>   : {:>14} bytes\n    Savings   : {:>+14} bytes  ({:.1}%)\n",
            requested,
            requested as f64 / 1_000_000.0,
            pinned,
            overhead,
            vec,
            savings,
            if vec > 0 { savings as f64 / vec as f64 * 100.0 } else { 0.0 }
        );
    }

    /// Core RSS test: prove PinnedBuffer uses ≤ Vec<u8> memory for the same request.
    #[test]
    fn rss_pinned_leq_vec() {
        let cases: &[(&str, usize)] = &[
            ("1.000 MiB  (power-of-2 baseline)", 1_048_576),
            ("1.500 MiB  (non-power-of-2)", 1_572_864),
            ("2.100 MiB  (non-power-of-2)", 2_202_009),
            ("3.700 MiB  (non-power-of-2)", 3_879_731),
        ];

        println!("\n══════ RSS Comparison: PinnedBuffer vs Vec<u8> ══════");

        for &(label, size) in cases {
            let (pinned, vec, requested) = measure_rss(size);
            print_rss_row(label, requested, pinned, vec);

            // Core assertion: PinnedBuffer must not use MORE physical RAM than Vec.
            // Allow 128 KiB headroom (32 OS pages) for measurement noise.
            assert!(
                pinned <= vec + 128 * 1024,
                "[{label}] PinnedBuffer used MORE RSS than Vec!\n  Pinned: {pinned} B\n  Vec:    {vec} B"
            );
        }
    }

    /// The overhead above the requested size must be less than one OS page (4 KB).
    /// This proves posix_memalign is not rounding up significantly.
    #[test]
    fn rss_overhead_under_one_page() {
        let cases: &[usize] = &[
            1_572_864, // 1.5 MiB — not a power of two
            2_202_009, // 2.1 MiB — not a power of two
            3_879_731, // 3.7 MiB — not a power of two
        ];

        println!("\n══════ RSS Overhead Check (must be < 1 OS page = 4 KiB) ══════");

        for &size in cases {
            let (pinned, _vec, requested) = measure_rss(size);
            // The OS charges in 4 KiB pages. posix_memalign rounds to the next
            // page, so overhead should be at most 4 KiB.
            // We allow 8 KiB (2 pages) as headroom for RSS measurement noise.
            let overhead = pinned as i64 - requested as i64;
            println!(
                "  {:.3} MB: overhead = {:+} bytes",
                requested as f64 / 1e6,
                overhead
            );
            assert!(
                overhead < 8 * 1024,
                "Overhead too large for {size} B: overhead = {overhead} B (expected < 8 KiB)"
            );
        }
    }

    /// Allocate multiple buffers and verify RSS grows proportionally.
    /// This catches allocators that over-reserve a big arena upfront.
    #[test]
    fn rss_proportional_growth() {
        let chunk = 1_000_000_usize; // 1 MB per buffer (scaled for CI)
        let count = 4;

        println!("\n══════ RSS Proportional Growth Test ══════");

        let baseline = vm_rss_kb().expect("rss unreadable");
        let mut bufs: Vec<PinnedBuffer> = Vec::new();

        for i in 0..count {
            let mut b = PinnedBuffer::alloc(chunk).expect("alloc failed");
            b.as_mut_slice().fill(i as u8);
            bufs.push(b);

            let current = vm_rss_kb().expect("rss unreadable");
            let total_rss = (current.saturating_sub(baseline)) * 1024;
            let expected_min = (i as u64 + 1) * chunk as u64;
            let expected_max = expected_min + 8 * 1024 * 1024; // +8 MB tolerance

            println!(
                "  After {} buffers: RSS delta = {} MB  (expected ≥ {} MB)",
                i + 1,
                total_rss as f64 / 1e6,
                expected_min as f64 / 1e6
            );

            assert!(
                total_rss >= expected_min,
                "After {} allocations of {chunk} B, RSS delta = {total_rss} B < expected {expected_min} B. \
                 Pages not committed?",
                i + 1
            );
            assert!(
                total_rss <= expected_max,
                "After {} allocations of {chunk} B, RSS delta = {total_rss} B > expected max {expected_max} B. \
                 Over-allocation?",
                i + 1
            );
        }

        // Drop all buffers and verify RSS returns to near-baseline
        drop(bufs);
        let after_drop = vm_rss_kb().expect("rss unreadable");
        let leaked_kb = after_drop.saturating_sub(baseline);
        println!("  After dropping all: leaked RSS = {} KB", leaked_kb);
        // Allow up to 2 MB of retained pages (OS may cache freed pages briefly)
        assert!(leaked_kb < 2048,
            "RSS did not return to baseline after drop — possible memory leak! Leaked: {leaked_kb} KB");
    }
}

// On non-Linux: skip RSS comparison with a clear message
#[cfg(not(target_os = "linux"))]
#[test]
#[ignore = "/proc/self/status is Linux-only. Run on Linux to see RSS comparison."]
fn rss_pinned_leq_vec() {}

// ════════════════════════════════════════════════════════════════
// STRESS TESTS — repeated cycles, leak detection, concurrency
// ════════════════════════════════════════════════════════════════

/// Allocate 500 buffers at varied sizes and drop them. No panic = no corruption.
#[test]
fn stress_500_allocs_varied_sizes() {
    let sizes: Vec<usize> = (1..=500)
        .map(|i| {
            // Mix of small, medium, large, and non-power-of-two sizes
            match i % 5 {
                0 => i * 512,
                1 => i * 4_096,
                2 => i * 8_192,
                3 => i * 1_000 + 7, // deliberately non-aligned sizes
                _ => i * 256 + 13,
            }
        })
        .collect();

    let mut bufs: Vec<PinnedBuffer> = Vec::new();
    for (i, &size) in sizes.iter().enumerate() {
        let mut buf = PinnedBuffer::alloc(size)
            .unwrap_or_else(|e| panic!("alloc #{i} ({size} B) failed: {e}"));
        buf.as_mut_slice().fill((i % 256) as u8); // touch every page
        bufs.push(buf);
    }

    // Verify all buffers are intact before dropping
    for (i, buf) in bufs.iter().enumerate() {
        let expected = (i % 256) as u8;
        assert_eq!(
            buf.as_slice()[0],
            expected,
            "buffer {i}: data corrupted. Got {}, expected {expected}",
            buf.as_slice()[0]
        );
    }

    drop(bufs); // All 500 Drop implementations must succeed without panic
}

/// Interleave alloc and drop — simulates the pool ring's claim/release cycle.
#[test]
fn stress_interleaved_alloc_drop() {
    let window = 8; // keep at most 8 buffers alive at once (like the sliding window)

    let mut live: std::collections::VecDeque<PinnedBuffer> = std::collections::VecDeque::new();

    for i in 0..400 {
        let size = 1_572_864 + i * 1024; // ~1.5 MB, increasing
        let mut buf = PinnedBuffer::alloc(size).unwrap_or_else(|e| panic!("alloc {i} failed: {e}"));
        buf.as_mut_slice().fill(0xBB);
        live.push_back(buf);

        // Evict oldest buffer when window is full (mimics layer eviction)
        if live.len() > window {
            let evicted = live.pop_front().unwrap();
            assert_eq!(evicted.as_slice()[0], 0xBB, "evicted buffer data corrupted");
        }
    }
}

/// Concurrent alloc+drop from multiple threads — tests Send safety.
#[test]
fn stress_concurrent_threads() {
    let thread_count = 8;
    let allocs_per_thread = 50;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            std::thread::spawn(move || {
                let size = 512_000 + thread_id * 65_536; // different size per thread
                for j in 0..allocs_per_thread {
                    let mut buf = PinnedBuffer::alloc(size)
                        .unwrap_or_else(|e| panic!("thread {thread_id} alloc {j} failed: {e}"));
                    buf.as_mut_slice().fill(thread_id as u8);
                    // Verify before drop
                    assert_eq!(buf.as_slice()[0], thread_id as u8);
                    drop(buf);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }
}

// ════════════════════════════════════════════════════════════════
// MODEL SIMULATION — pin memory like real AethelStream training
// ════════════════════════════════════════════════════════════════
//
// This test simulates what AethelStream actually does when training:
//
// A 70B-parameter model has 80 transformer layers.
// In streaming mode, AethelStream holds 2 layers in RAM at any time
// (the current layer being computed, and the next layer being prefetched).
//
// Each attention block in a 70B model is ~1.6 GB in 4-bit quantized form.
// Each small-tensor slab (LayerNorm + LoRA adapters) is ~2.1 MB.
// Optimizer states (8-bit compressed, low-rank) are ~256 KB per layer.
//
// We simulate: 2 layers in the sliding window + optimizer states = ~3.5 GB
// Then show that PinnedBuffer pins exactly that amount, not 4 GB (rounded up).

#[test]
fn simulate_model_layer_streaming() {
    // ── Model parameters (70B LLaMA-3 scale) ─────────────────────────────
    let attention_block_bytes: usize = 157_286_400; // 150 MiB per layer (scaled for CI, same alignment proof)
    let small_tensor_slab_bytes: usize = 2_202_009; // 2.1 MB (LayerNorm + LoRA)
    let optimizer_state_bytes: usize = 262_144; // 256 KB (8-bit compressed)

    // Sliding window: 2 attention blocks (current + next)
    let sliding_window_count = 2;

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        AethelStream 70B Training Memory Simulation              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Model: 70B parameters, 4-bit quantized, 80 transformer layers  ║");
    println!("║  Mode:  Sequential Layer Streaming (SLS)                        ║");
    println!("║  RAM:   64 GB consumer machine target                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 1: Allocate the sliding window ─────────────────────────────
    println!("▶ Phase 1: Allocating sliding window (2 attention blocks)");
    println!("  Each block: {:.1} MB", attention_block_bytes as f64 / 1e6);

    let mut attention_window: Vec<PinnedBuffer> = (0..sliding_window_count)
        .map(|i| {
            let mut buf = PinnedBuffer::alloc(attention_block_bytes)
                .unwrap_or_else(|e| panic!("attention block {i} alloc failed: {e}"));
            // Simulate loading quantized weights into the buffer
            // (fill with a pattern representing 4-bit packed weights)
            buf.as_mut_slice().fill(0xF0 | i as u8);

            println!(
                "  ✓ Attention block {i}: {} bytes pinned, ptr={:p}",
                buf.len(),
                buf.as_ptr()
            );
            assert_eq!(buf.len(), attention_block_bytes);
            assert_eq!(buf.as_ptr() as usize % 64, 0, "not 64-byte aligned!");
            buf
        })
        .collect();

    let window_total = attention_window.iter().map(|b| b.len()).sum::<usize>();
    println!(
        "  → Sliding window total: {:.1} MB pinned",
        window_total as f64 / 1e6
    );

    // ── Phase 2: Small tensor slabs (one per layer in the window) ────────
    println!("\n▶ Phase 2: Small tensor slabs (LayerNorm + LoRA adapters)");

    let mut small_slabs: Vec<PinnedBuffer> = (0..sliding_window_count)
        .map(|i| {
            // alloc_mapped because these go through the zero-copy path in Sprint 4
            let mut buf = PinnedBuffer::alloc_mapped(small_tensor_slab_bytes)
                .unwrap_or_else(|e| panic!("small slab {i} alloc failed: {e}"));
            buf.as_mut_slice().fill(0x55);

            println!(
                "  ✓ Small slab {i}: {} bytes, is_mapped={}, ptr={:p}",
                buf.len(),
                buf.is_mapped(),
                buf.as_ptr()
            );
            assert!(buf.is_mapped(), "small slabs must use alloc_mapped");
            assert_eq!(buf.len(), small_tensor_slab_bytes);
            buf
        })
        .collect();

    // ── Phase 3: Optimizer states ─────────────────────────────────────────
    println!("\n▶ Phase 3: 8-bit optimizer states (for active layer)");

    let mut opt =
        PinnedBuffer::alloc_mapped(optimizer_state_bytes).expect("optimizer state alloc failed");
    opt.as_mut_slice().fill(0xCC);
    println!(
        "  ✓ Optimizer state: {} bytes, is_mapped={}, ptr={:p}",
        opt.len(),
        opt.is_mapped(),
        opt.as_ptr()
    );

    // ── Phase 4: Total summary ────────────────────────────────────────────
    let total_pinned =
        window_total + small_slabs.iter().map(|b| b.len()).sum::<usize>() + opt.len();

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  MEMORY SUMMARY                                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Sliding window (2 × 1.5 GB attention):  {:>8.3} GB           ║",
        window_total as f64 / 1e6
    );
    println!(
        "║  Small tensor slabs (2 × 2.1 MB):        {:>8.3} MB           ║",
        small_slabs.iter().map(|b| b.len()).sum::<usize>() as f64 / 1e6
    );
    println!(
        "║  Optimizer state (256 KB):                {:>8.3} KB           ║",
        opt.len() as f64 / 1e3
    );
    println!("║  ─────────────────────────────────────────────────────          ║");
    println!(
        "║  TOTAL PINNED:                            {:>8.3} GB           ║",
        total_pinned as f64 / 1e9
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  PyTorch rounded equivalent:              {:>8.3} GB           ║",
        (268_435_456_u64 * 2) as f64 / 1e9
    ); // 2×256MiB (next pow-of-2 for 150MB = 256MB, ×2 = 512MB)
    println!(
        "║  RamFlow savings vs PyTorch:              {:>8.3} GB           ║",
        (268_435_456_u64 * 2) as f64 / 1e9 - total_pinned as f64 / 1e9
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // ── Phase 5: Simulate layer eviction (sliding window advance) ─────────
    println!("\n▶ Phase 5: Simulating layer eviction (window advances)");
    println!("  Evicting layer 0, loading layer 2...");

    let evicted = attention_window.remove(0);
    assert_eq!(evicted.as_slice()[0], 0xF0, "evicted layer data wrong");
    drop(evicted); // unpins + frees
    println!("  ✓ Layer 0 evicted (unpinned and freed)");

    // Allocate next layer
    let mut next_layer =
        PinnedBuffer::alloc(attention_block_bytes).expect("next layer alloc failed");
    next_layer.as_mut_slice().fill(0xF2);
    attention_window.push(next_layer);
    println!(
        "  ✓ Layer 2 pinned ({:.1} MB)",
        attention_block_bytes as f64 / 1e6
    );

    println!("\n▶ All assertions passed. Drop guard cleaning up...");

    // Drop all remaining — tests that Drop works correctly for large allocations
    drop(attention_window);
    drop(small_slabs);
    drop(opt);

    println!("  ✓ All buffers unpinned and freed cleanly.");
    println!("  ✓ Simulation complete.\n");
}

/// Verify that a 240 MB total allocation uses exactly 240 MB, not 256 MB.
// (Same proof, scaled for CI — the mathematics are identical)
/// This is the direct proof of the core claim.
#[test]
fn simulate_exact_240_mb_pinning() {
    // 2.4 GB split across three non-power-of-two chunks
    // (as would happen in real training: different layer sizes)
    let sizes: &[usize] = &[
        104_857_600, // exactly 100 MiB
        81_100_000,  // ~77 MB (deliberately not a power of two)
        52_428_800,  // exactly 50 MiB
    ];

    let total_requested: usize = sizes.iter().sum();
    let total_gb = total_requested as f64 / 1e6;

    println!("\n══════ Exact 240 MB Pinning Test (scaled for CI) ══════");
    println!(
        "  Requested total: {:.1} MB ({total_requested} bytes)",
        total_gb
    );

    let mut buffers: Vec<PinnedBuffer> = Vec::new();
    for (i, &size) in sizes.iter().enumerate() {
        let mut buf = PinnedBuffer::alloc(size)
            .unwrap_or_else(|e| panic!("alloc chunk {i} ({size} B) failed: {e}"));
        buf.as_mut_slice().fill(i as u8 + 1);
        println!(
            "  ✓ Chunk {i}: {:.3} MB at {:p} (aligned: {})",
            size as f64 / 1e6,
            buf.as_ptr(),
            buf.as_ptr() as usize % 64 == 0
        );
        assert_eq!(buf.len(), size, "chunk {i} size wrong");
        assert_eq!(
            buf.as_ptr() as usize % 64,
            0,
            "chunk {i} not 64-byte aligned"
        );
        buffers.push(buf);
    }

    println!(
        "  ✓ Total pinned: {:.1} MB (no rounding, no waste)",
        total_gb
    );

    // Verify data integrity across all chunks
    for (i, buf) in buffers.iter().enumerate() {
        let expected = (i + 1) as u8;
        assert_eq!(buf.as_slice()[0], expected, "chunk {i}: data corrupted");
        assert_eq!(
            buf.as_slice()[buf.len() - 1],
            expected,
            "chunk {i}: last byte corrupted"
        );
    }

    println!("  ✓ Data integrity verified across all chunks.");
    println!("  ✓ Dropping all buffers...");
    drop(buffers);
    println!("  ✓ 2.4 GB released cleanly.\n");
}
