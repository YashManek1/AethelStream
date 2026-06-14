//! Tests for CQE retry classification and backoff in `CqeRetryBackend`.
//!
//! All tests run under `--features mock-cuda` (no real NVMe or io_uring).
//! The `MockBackend::inject_completion_for_test` helper injects synthetic CQE
//! results (negative errno values) to exercise the retry path end-to-end.

use flowcast::backend::mock::MockBackend;
use flowcast::backend::IoBackend;
use flowcast::completion_router::{CqeRetryBackend, PendingRead, RetryConfig};
use flowcast::telemetry::Telemetry;
use ramflow::{CqeErrorKind, classify_cqe_error};
use std::sync::Arc;

// ===========================================================================
// Helper
// ===========================================================================

/// Build a `CqeRetryBackend` wrapping a `MockBackend`.
///
/// Returns the inner mock (already started) and the decorator.
fn make_retry_backend(max_retries: u8, base_backoff_ms: u64) -> (Arc<MockBackend>, CqeRetryBackend) {
    let mut mock = MockBackend::new();
    mock.start().expect("mock start");
    let mock_arc = Arc::new(mock);
    let inner: Arc<dyn IoBackend> = Arc::clone(&mock_arc) as Arc<dyn IoBackend>;
    let backend = CqeRetryBackend::new(
        inner,
        RetryConfig { max_retries, base_backoff_ms },
        Telemetry::new(),
    );
    (mock_arc, backend)
}

// ===========================================================================
// T1 — classify_cqe_error returns correct variants
// ===========================================================================

/// Verifies that the three errno categories map to the correct `CqeErrorKind`.
#[test]
fn classify_cqe_error_returns_correct_kind() {
    // Transient: EINTR(4), EAGAIN(11), EBUSY(16)
    assert_eq!(classify_cqe_error(-4), CqeErrorKind::Transient);
    assert_eq!(classify_cqe_error(-11), CqeErrorKind::Transient);
    assert_eq!(classify_cqe_error(-16), CqeErrorKind::Transient);

    // Media: EIO(5), ENODEV(19)
    assert_eq!(classify_cqe_error(-5), CqeErrorKind::MediaError);
    assert_eq!(classify_cqe_error(-19), CqeErrorKind::MediaError);

    // Unknown
    assert!(matches!(classify_cqe_error(-22), CqeErrorKind::Unknown(_)));
    assert!(matches!(classify_cqe_error(-1), CqeErrorKind::Unknown(_)));
}

// ===========================================================================
// T2 — transient error queued for retry, not forwarded immediately
// ===========================================================================

/// A single EAGAIN result must be queued for retry and NOT appear in the
/// completed list returned by `poll_completions`.
#[test]
fn transient_error_is_queued_not_forwarded() {
    let (mock_arc, backend) = make_retry_backend(3, 5);

    // Register a pending read so the backend knows the I/O params for token 1.
    let buf = ramflow::PinnedBuffer::alloc(4096).expect("alloc");
    backend.insert_pending_for_test(PendingRead {
        token: 1,
        shard_id: 0,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 0,
        next_retry_at: None,
    });

    // Inject -EAGAIN for token 1.
    mock_arc.inject_completion_for_test(1, -11);

    let completions = backend.poll_completions().expect("poll");
    // Should NOT appear in the success/forwarded list.
    assert!(
        completions.iter().all(|c| c.token != 1),
        "transient error must not be forwarded immediately"
    );
    // Should be in the retry queue.
    assert_eq!(backend.retry_queue_len(), 1, "entry must be in retry queue");
}

// ===========================================================================
// T3 — media error forwarded immediately with negative result
// ===========================================================================

/// An EIO result must be forwarded immediately (negative CQE) so the state
/// machine can free the slot.
#[test]
fn media_error_forwarded_immediately() {
    let (mock_arc, backend) = make_retry_backend(3, 5);

    let buf = ramflow::PinnedBuffer::alloc(4096).expect("alloc");
    backend.insert_pending_for_test(PendingRead {
        token: 2,
        shard_id: 0,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 0,
        next_retry_at: None,
    });

    // Inject -EIO for token 2.
    mock_arc.inject_completion_for_test(2, -5);

    let completions = backend.poll_completions().expect("poll");
    let found = completions.iter().find(|c| c.token == 2);
    assert!(found.is_some(), "media error must be forwarded immediately");
    assert_eq!(found.unwrap().result, -5, "result must preserve the negative errno");
    assert_eq!(backend.retry_queue_len(), 0, "retry queue must be empty for media errors");
}

// ===========================================================================
// T4 — unknown error forwarded immediately
// ===========================================================================

/// An errno with no classification (e.g. EINVAL = 22) must be forwarded
/// immediately just like a media error.
#[test]
fn unknown_error_forwarded_immediately() {
    let (mock_arc, backend) = make_retry_backend(3, 5);

    let buf = ramflow::PinnedBuffer::alloc(4096).expect("alloc");
    backend.insert_pending_for_test(PendingRead {
        token: 3,
        shard_id: 0,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 0,
        next_retry_at: None,
    });

    // EINVAL = -22
    mock_arc.inject_completion_for_test(3, -22);

    let completions = backend.poll_completions().expect("poll");
    let found = completions.iter().find(|c| c.token == 3);
    assert!(found.is_some(), "unknown error must be forwarded immediately");
    assert_eq!(backend.retry_queue_len(), 0);
}

// ===========================================================================
// T5 — transient error exhausts budget and is forwarded as terminal failure
// ===========================================================================

/// When the retry count reaches `max_retries`, the next Transient error must
/// be forwarded (negative) instead of queued again.
#[test]
fn transient_error_exhausted_forwarded_as_terminal() {
    let (mock_arc, backend) = make_retry_backend(2, 0);

    let buf = ramflow::PinnedBuffer::alloc(4096).expect("alloc");

    // Pre-fill the pending entry with retry_count == max_retries.
    backend.insert_pending_for_test(PendingRead {
        token: 4,
        shard_id: 0,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 2, // already exhausted
        next_retry_at: None,
    });

    // Inject another EAGAIN — budget is exhausted, must be forwarded.
    mock_arc.inject_completion_for_test(4, -11);

    let completions = backend.poll_completions().expect("poll");
    let found = completions.iter().find(|c| c.token == 4);
    assert!(found.is_some(), "exhausted transient must be forwarded");
    assert_eq!(backend.retry_queue_len(), 0, "retry queue must be empty after exhaustion");
}

// ===========================================================================
// T6 — successful completion removed from pending_reads and forwarded
// ===========================================================================

/// A positive CQE (bytes transferred ≥ 0) must be forwarded unchanged and
/// must not remain in the pending-reads map.
#[test]
fn successful_completion_forwarded_and_removed() {
    let (mock_arc, backend) = make_retry_backend(3, 5);

    let buf = ramflow::PinnedBuffer::alloc(4096).expect("alloc");
    backend.insert_pending_for_test(PendingRead {
        token: 5,
        shard_id: 0,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 0,
        next_retry_at: None,
    });

    // Inject a successful completion (4096 bytes read).
    mock_arc.inject_completion_for_test(5, 4096);

    let completions = backend.poll_completions().expect("poll");
    let found = completions.iter().find(|c| c.token == 5);
    assert!(found.is_some(), "successful completion must be forwarded");
    assert_eq!(found.unwrap().result, 4096, "positive result preserved");
    assert_eq!(backend.retry_queue_len(), 0);
}

// ===========================================================================
// T7 — telemetry counters updated correctly
// ===========================================================================

/// `record_cqe_retry` must fire for each queued transient error;
/// `record_media_error` must fire for each terminal failure.
#[test]
fn telemetry_counters_updated_on_retry_and_error() {
    let telemetry = Telemetry::new();

    let mut mock = MockBackend::new();
    mock.start().expect("start");
    let mock_arc = Arc::new(mock);
    let inner: Arc<dyn IoBackend> = Arc::clone(&mock_arc) as Arc<dyn IoBackend>;
    let backend = CqeRetryBackend::new(
        inner,
        RetryConfig { max_retries: 3, base_backoff_ms: 5 },
        telemetry.clone(),
    );

    let buf = ramflow::PinnedBuffer::alloc(4096).expect("alloc");

    // Token 10 → transient (EAGAIN, retry_count 0 < 3 → queued).
    backend.insert_pending_for_test(PendingRead {
        token: 10,
        shard_id: 0,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 0,
        next_retry_at: None,
    });
    mock_arc.inject_completion_for_test(10, -11); // EAGAIN

    // Token 11 → media error (EIO → terminal immediately).
    backend.insert_pending_for_test(PendingRead {
        token: 11,
        shard_id: 1,
        byte_offset: 0,
        length: 4096,
        dst: buf.as_ptr() as *mut u8,
        retry_count: 0,
        next_retry_at: None,
    });
    mock_arc.inject_completion_for_test(11, -5); // EIO

    backend.poll_completions().expect("poll");

    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.retry_count, 1, "one retry recorded for EAGAIN");
    assert_eq!(snapshot.media_error_count, 1, "one media_error recorded for EIO");
}
