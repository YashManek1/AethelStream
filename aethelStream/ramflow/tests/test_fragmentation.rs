// tests/test_fragmentation.rs — Sprint 2 fragmentation / stability test

use ramflow::PinnedBuffer;

#[cfg(target_os = "linux")]
fn vm_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest.trim().split_ascii_whitespace().next()?.parse().ok();
        }
    }
    None
}

#[test]
fn test_fragmentation_stability_80_layers_x_10_passes() {
    let baseline_rss = {
        #[cfg(target_os = "linux")]
        {
            vm_rss_kb().unwrap_or(0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            0u64
        }
    };

    // Simulate 80 layers * 10 passes with mixed-size allocations.
    for pass in 0..10 {
        let mut pass_buffers: Vec<PinnedBuffer> = Vec::with_capacity(80);
        for layer in 0..80 {
            let size = 64 * 1024 + ((layer + pass) % 7) * 8 * 1024;
            let mut buf = PinnedBuffer::alloc(size).expect("PinnedBuffer alloc failed");
            buf.as_mut_slice()[0] = (layer as u8) ^ (pass as u8);
            pass_buffers.push(buf);
        }
        // Dropping here exercises allocator churn each pass.
        drop(pass_buffers);
    }

    #[cfg(target_os = "linux")]
    {
        let end_rss = vm_rss_kb().unwrap_or(baseline_rss);
        if baseline_rss > 0 {
            let drift = end_rss.saturating_sub(baseline_rss) as f64 / baseline_rss as f64;
            assert!(
                drift < 0.02,
                "RSS drift exceeded 2%: baseline={}KB end={}KB drift={:.2}%",
                baseline_rss,
                end_rss,
                drift * 100.0
            );
        }
    }
}
