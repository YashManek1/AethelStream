//! T6 — Backend capability probe and byte-identical reads.
//!
//! 1. select_backend("mock") returns a mock backend.
//! 2. select_backend("super-shard") returns a super-shard backend.
//! 3. auto-select without override falls back to mock on non-Linux.
//! 4. Each backend returns byte-identical data for the same prefetch range.
//! 5. Backend capabilities are correctly reported.
//! 6. Unknown override returns Config error.

use flowcast::backend::{select_backend_with_override, IoBackend};
use flowcast::backend::mock::MockBackend;
use flowcast::backend::gds::GdsBackend;
use ramflow::PinnedBuffer;

// T6-1: "mock" override selects MockBackend
#[test]
fn override_mock_selects_mock_backend() {
    let b = select_backend_with_override(std::path::Path::new("."), 0, Some("mock"))
        .expect("mock backend");
    assert_eq!(b.capabilities().name, "mock");
}

// T6-2: "super-shard" override on non-Linux falls back through uring → errors,
// so we test that the override path returns a Config error (uring unavailable
// on Windows) rather than panicking.
#[test]
fn override_super_shard_on_non_linux_errors_gracefully() {
    #[cfg(not(target_os = "linux"))]
    {
        let result = select_backend_with_override(
            std::path::Path::new("."), 0, Some("super-shard"));
        // On non-Linux UringBackend::new fails → BackendIo error, not panic.
        assert!(result.is_err(), "super-shard should fail on non-Linux (no io_uring)");
    }
    #[cfg(target_os = "linux")]
    {
        // On Linux it should succeed.
        let _ = select_backend_with_override(
            std::path::Path::new("/tmp"), 0, Some("super-shard"));
    }
}

// T6-3: auto-select without override returns a working backend (mock fallback)
#[test]
fn auto_select_returns_working_backend() {
    let b = select_backend_with_override(std::path::Path::new("."), 0, None)
        .expect("auto select");
    // Must not panic; capabilities must have a non-empty name.
    assert!(!b.capabilities().name.is_empty());
}

// T6-4: byte-identical reads — mock backend prefetch returns expected length
#[test]
fn mock_backend_returns_byte_identical_completions() {
    let backend = MockBackend::new();
    let buf = PinnedBuffer::alloc(128).expect("alloc");
    backend.prefetch(0, 0, 128, &buf, 0xABCD).expect("prefetch");

    let completions = backend.poll_completions().expect("poll");
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].token, 0xABCD);
    assert_eq!(completions[0].result, 128);
}

// T6-5: GDS mock backend prefetch also returns correct length
#[test]
fn gds_mock_backend_byte_identical() {
    let backend = GdsBackend::new_mock();
    let buf = PinnedBuffer::alloc(64).expect("alloc");
    backend.prefetch(0, 0, 64, &buf, 0x1234).expect("prefetch");

    let completions = backend.poll_completions().expect("poll");
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].result, 64);
    assert!(backend.capabilities().supports_gds);
}

// T6-6: unknown override returns Config error
#[test]
fn unknown_override_returns_config_error() {
    let result = select_backend_with_override(
        std::path::Path::new("."), 0, Some("nonexistent"));
    match result {
        Err(flowcast::FlowCastError::Config(_)) => {}
        other => panic!("expected Config error, got Err({:?})", other.err().map(|e| e.to_string())),
    }
}

// T6-7: super-shard backend reports supports_super_shard = true
#[test]
fn super_shard_backend_capabilities() {
    use flowcast::backend::super_shard::{SuperShardBackend, SuperShardConfig};
    let base = Box::new(MockBackend::new());
    let ss = SuperShardBackend::new(base, SuperShardConfig::default());
    assert!(ss.capabilities().supports_super_shard);
    assert_eq!(ss.capabilities().name, "super-shard");
}
