// tests/test_numa.rs — NUMA-aware pool allocation tests
//
// Run with:
//   cargo test --no-default-features --features "mock-cuda,numa" test_numa
//
// All tests compile and pass on Windows (where NUMA stubs always return
// `disabled()` / `false`).  Linux-only behaviour is guarded with
// `#[cfg(target_os = "linux")]`.

#[cfg(feature = "numa")]
mod numa_tests {
    use ramflow::allocator::{NumaConfig, PinnedBuffer};
    use ramflow::pool::PoolRegistry;

    // ── detect() never panics and returns a structurally valid NumaConfig ─────

    #[test]
    fn detect_none_does_not_panic() {
        let config = ramflow::allocator::numa::detect(None);
        // Structural invariant: if gpu_node is Some, available must be true.
        if config.gpu_node.is_some() {
            assert!(
                config.available,
                "NumaConfig with gpu_node set must have available == true"
            );
        }
    }

    #[test]
    fn detect_with_explicit_addr_does_not_panic() {
        // Pass a nonexistent PCI address — must fall back gracefully, not panic.
        let config = ramflow::allocator::numa::detect(Some("ffff:ff:ff.7"));
        assert!(
            !config.available || config.gpu_node.is_some(),
            "detect with bogus PCI addr must return disabled or valid config"
        );
    }

    // ── NumaConfig::disabled() invariants ─────────────────────────────────────

    #[test]
    fn disabled_config_has_no_node() {
        let config = NumaConfig::disabled();
        assert!(config.gpu_node.is_none(), "disabled config must have no gpu_node");
        assert!(!config.available, "disabled config must not be available");
    }

    #[test]
    fn default_config_equals_disabled() {
        let default = NumaConfig::default();
        let disabled = NumaConfig::disabled();
        assert_eq!(default.gpu_node, disabled.gpu_node);
        assert_eq!(default.available, disabled.available);
    }

    // ── mbind_buffer: never panics; returns false on non-Linux ────────────────

    #[test]
    fn mbind_buffer_on_valid_allocation_does_not_panic() {
        let buf = PinnedBuffer::alloc(4 * 1024 * 1024).expect("alloc 4 MiB");
        // On Windows: always returns false (stub).
        // On Linux without a real NUMA node 0: may return false (EPERM or no node).
        let _ = ramflow::allocator::numa::mbind_buffer(buf.as_ptr() as *mut u8, buf.len(), 0);
    }

    #[test]
    fn mbind_buffer_zero_size_does_not_panic() {
        // A zero-length bind is a no-op; must not panic on any platform.
        // On Linux the real impl returns true; the non-Linux stub returns false.
        let _ = ramflow::allocator::numa::mbind_buffer(std::ptr::null_mut::<u8>(), 0, 0);
    }

    #[test]
    fn mbind_buffer_high_node_does_not_panic() {
        // Node 63 is the maximum supported by the nodemask.  Must not panic on
        // any platform, even if the bind silently fails.
        let buf = PinnedBuffer::alloc(4096).expect("alloc 4 KiB");
        let _ = ramflow::allocator::numa::mbind_buffer(buf.as_ptr() as *mut u8, buf.len(), 63);
    }

    // ── PoolRegistry NUMA integration ─────────────────────────────────────────

    #[test]
    fn pool_registry_with_defaults_has_disabled_numa() {
        let registry = PoolRegistry::with_defaults().expect("with_defaults");
        let config = registry.numa_config();
        // with_defaults() always uses NumaConfig::disabled() —
        // no sysfs probing, so single-socket machines stay unaffected.
        assert!(
            !config.available,
            "with_defaults() must return disabled NUMA config, got available=true"
        );
        assert!(
            config.gpu_node.is_none(),
            "with_defaults() must return disabled NUMA config, got gpu_node={:?}",
            config.gpu_node
        );
    }

    // ── Linux-only: non-existent PCI address falls back gracefully ───────────

    #[test]
    #[cfg(target_os = "linux")]
    fn detect_with_bogus_pci_addr_returns_disabled() {
        // An address that does not exist in /sys must return disabled, not panic.
        // This exercises the I/O error path in read_numa_node_for().
        let config = ramflow::allocator::numa::detect(Some("0000:ff:ff.7"));
        // A missing sysfs entry => disabled (gpu_node == None, available == false).
        assert!(
            !config.available,
            "non-existent PCI addr must yield disabled config"
        );
        assert!(
            config.gpu_node.is_none(),
            "non-existent PCI addr must yield no gpu_node"
        );
    }

    // ── Regression: multiple detect() calls are idempotent ───────────────────

    #[test]
    fn detect_is_idempotent() {
        let first = ramflow::allocator::numa::detect(None);
        let second = ramflow::allocator::numa::detect(None);
        assert_eq!(
            first.available, second.available,
            "detect() must return consistent results across calls"
        );
        assert_eq!(
            first.gpu_node, second.gpu_node,
            "detect() must return consistent gpu_node across calls"
        );
    }
}
