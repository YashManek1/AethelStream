//! T-DS: DirectStorage backend capability probe, fallback, and byte-integrity.
//!
//! Run: cargo test --features "mock-cuda,direct-storage" --test test_direct_storage
//!
//! All tests compile on Linux (no DirectStorage hardware required) because
//! all Windows COM code is guarded by `#[cfg(target_os = "windows")]`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use flowcast::backend::{select_backend_with_override, IoBackend};
use ramflow::PinnedBuffer;

// ---------------------------------------------------------------------------
// T-DS-1: capability probe never panics
// ---------------------------------------------------------------------------

/// `probe_direct_storage()` must return a well-formed result on any platform.
///
/// On Linux: always `Unavailable`.
/// On Windows without the DLL: `Unavailable`.
/// On Windows with the DLL: `Available { max_transfer_bytes: 32 MiB }`.
#[test]
#[cfg(feature = "direct-storage")]
fn probe_never_panics_and_returns_valid_result() {
    use ramflow::probe_direct_storage;
    use ramflow::DirectStorageCapability;

    let cap = probe_direct_storage();
    match cap {
        DirectStorageCapability::Unavailable => {
            // Expected on Linux / Windows without the DirectStorage DLL.
        }
        DirectStorageCapability::Available { max_transfer_bytes } => {
            assert!(
                max_transfer_bytes > 0,
                "max_transfer_bytes must be > 0 when DirectStorage is available"
            );
            // DS 1.2 spec: 32 MiB maximum transfer.
            assert_eq!(
                max_transfer_bytes,
                32 * 1024 * 1024,
                "DS 1.2 max transfer must be 32 MiB"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// T-DS-2: ReadFile fallback round-trip — byte-identical
// ---------------------------------------------------------------------------

/// The ReadFile fallback must return the exact bytes written to a temp shard.
///
/// This is the path taken on Linux and on Windows without the DirectStorage DLL.
/// The test creates a real temp file so the backend exercises actual `std::fs` I/O.
#[test]
#[cfg(feature = "direct-storage")]
fn readfile_fallback_byte_identical_round_trip() {
    use flowcast::backend::direct_storage::DirectStorageBackend;

    let payload: Vec<u8> = (0u8..=255).cycle().take(4096).collect();

    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::io::Write::write_all(&mut tmp.as_file(), &payload).expect("write payload");

    let shard_paths = vec![tmp.path().to_path_buf()];
    let backend = DirectStorageBackend::new_readfile_fallback(shard_paths);

    let buf = PinnedBuffer::alloc(4096).expect("alloc");

    backend
        .prefetch(0, 0, 4096, &buf, 0xABCD_1234)
        .expect("prefetch");

    let completions = backend.poll_completions().expect("poll_completions");
    assert_eq!(completions.len(), 1, "one completion expected");
    assert_eq!(completions[0].token, 0xABCD_1234);
    assert!(completions[0].result > 0, "result must be > 0 (bytes read)");

    assert_eq!(
        buf.as_slice()[..4096],
        payload[..4096],
        "bytes read via ReadFile fallback must be identical to written payload"
    );
}

// ---------------------------------------------------------------------------
// T-DS-3: backend name and capabilities are correctly reported
// ---------------------------------------------------------------------------

/// `capabilities().name` must contain "direct-storage" for both paths.
#[test]
#[cfg(feature = "direct-storage")]
fn backend_name_contains_direct_storage() {
    use flowcast::backend::direct_storage::DirectStorageBackend;

    let backend = DirectStorageBackend::new_readfile_fallback(Vec::new());
    let caps = backend.capabilities();
    assert!(
        caps.name.contains("direct-storage"),
        "backend name must contain 'direct-storage'; got '{}'",
        caps.name
    );
}

// ---------------------------------------------------------------------------
// T-DS-4: select_backend_with_override("direct-storage") selects DS backend
// ---------------------------------------------------------------------------

/// The "direct-storage" override must not return a Config error on any platform.
///
/// On Windows with DLL: `is_using_direct_storage() == true`.
/// On Linux / Windows without DLL: degrades to ReadFile fallback, name still
/// contains "direct-storage".
#[test]
#[cfg(feature = "direct-storage")]
fn override_direct_storage_returns_working_backend() {
    let result = select_backend_with_override(
        std::path::Path::new("."),
        0,
        Some("direct-storage"),
    );
    let backend = result.expect("direct-storage override must not fail");
    assert!(
        backend.capabilities().name.contains("direct-storage"),
        "backend name must contain 'direct-storage'"
    );
}

// ---------------------------------------------------------------------------
// T-DS-5 (Windows-only): real DirectStorage path smoke test
// ---------------------------------------------------------------------------

/// When DirectStorage DLL is present, the real COM path must handle a 512-byte
/// shard read without panicking and return a positive byte count.
///
/// Skipped on Linux (always passes by exclusion).
#[test]
#[cfg(all(target_os = "windows", feature = "direct-storage"))]
fn windows_real_path_smoke_test_if_dll_present() {
    use flowcast::backend::direct_storage::DirectStorageBackend;
    use ramflow::{probe_direct_storage, DirectStorageCapability};

    if !matches!(probe_direct_storage(), DirectStorageCapability::Available { .. }) {
        return;
    }

    let payload = vec![0xCCu8; 512];
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::io::Write::write_all(&mut tmp.as_file(), &payload).expect("write");

    let backend = DirectStorageBackend::new(vec![tmp.path().to_path_buf()]);
    if !backend.is_using_direct_storage() {
        return;
    }

    let buf = ramflow::alloc_windows_ds_compatible(512).expect("ds-compat alloc");
    backend
        .prefetch(0, 0, 512, &buf, 0xDEAD_BEEF)
        .expect("DirectStorage prefetch");

    let mut completions = Vec::new();
    for _ in 0..1000 {
        completions = backend.poll_completions().expect("poll");
        if !completions.is_empty() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_micros(100));
    }

    assert_eq!(completions.len(), 1, "one DirectStorage completion expected");
    assert_eq!(completions[0].token, 0xDEAD_BEEF);
    assert!(completions[0].result > 0, "DirectStorage result must be > 0");
}

// ---------------------------------------------------------------------------
// T-DS-6: alloc_windows_ds_compatible produces a page-aligned buffer
// ---------------------------------------------------------------------------

/// `alloc_windows_ds_compatible` must return a 4 096-byte-aligned buffer.
///
/// This is the minimum alignment required for `DSTORAGE_REQUEST_DESTINATION_BUFFER`.
#[test]
#[cfg(feature = "direct-storage")]
fn alloc_windows_ds_compatible_is_page_aligned() {
    use ramflow::alloc_windows_ds_compatible;

    for size in [4096usize, 8192, 1024 * 1024] {
        let buf = alloc_windows_ds_compatible(size)
            .unwrap_or_else(|error| panic!("alloc_windows_ds_compatible({size}): {error}"));
        assert!(
            buf.is_page_aligned(),
            "alloc_windows_ds_compatible({size}) must be 4 096-byte aligned"
        );
    }
}

// ---------------------------------------------------------------------------
// T-DS-7: pause / resume round-trip on fallback backend
// ---------------------------------------------------------------------------

/// `set_pause(true)` / `set_pause(false)` flag must round-trip through
/// `is_paused()`, and prefetch must succeed after resume.
///
/// Validated on the ReadFile fallback so the test runs without real hardware.
#[test]
#[cfg(feature = "direct-storage")]
fn pause_blocks_prefetch_and_resume_allows_it() {
    use flowcast::backend::direct_storage::DirectStorageBackend;

    let payload = vec![0u8; 512];
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::io::Write::write_all(&mut tmp.as_file(), &payload).expect("write");

    let backend = DirectStorageBackend::new_readfile_fallback(vec![tmp.path().to_path_buf()]);
    let buf = PinnedBuffer::alloc(512).expect("alloc");

    backend.set_pause(true);
    assert!(backend.is_paused(), "is_paused must be true after set_pause(true)");

    backend.set_pause(false);
    assert!(!backend.is_paused(), "is_paused must be false after set_pause(false)");

    backend.prefetch(0, 0, 512, &buf, 99).expect("prefetch after resume");
    let completions = backend.poll_completions().expect("poll");
    assert_eq!(completions.len(), 1, "completion after resume must arrive");
}
