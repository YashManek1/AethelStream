// tests/test_io_uring.rs — Sprint 2 integration checks (Linux only)
//
// These tests exercise:
// 1) io_uring read correctness (byte-identical comparison)
// 2) CQE pipeline behavior (submit async, receive completion token)

#[cfg(target_os = "linux")]
use ramflow::{DirectNvmeEngine, PinnedBuffer};

#[cfg(target_os = "linux")]
fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn make_tmp_file_with_data() -> (std::path::PathBuf, Vec<u8>) {
    use std::fs::OpenOptions;
    use std::io::Write;

    let content = pseudo_random_bytes(4 * 1024 * 1024, 0xD00D_BEEF_1337_5EED);
    let path = std::path::PathBuf::from(format!(
        "/tmp/ramflow_io_uring_it_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock error")
            .as_nanos()
    ));

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .expect("failed to create temp file");
    file.write_all(&content).expect("failed to write content");
    file.sync_all().expect("failed to sync temp file");

    (path, content)
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Linux io_uring + O_DIRECT-compatible filesystem/alignment"]
fn test_io_uring_read_correctness_byte_identical() {
    use std::io::{Read, Seek, SeekFrom};

    let (path, _content) = make_tmp_file_with_data();

    let mut engine =
        DirectNvmeEngine::open_with_paths(&[path.as_path()]).expect("engine open failed");
    engine
        .start_cqe_poller(0)
        .expect("failed to start cqe poller");

    let mut dst = PinnedBuffer::alloc(1536).expect("PinnedBuffer::alloc failed");
    engine
        .prefetch(0, 512, 1536, &dst, 0xABCD_1234)
        .expect("prefetch submit failed");

    let cqe = engine
        .completion_rx()
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("timeout waiting for cqe");
    assert_eq!(cqe.token, 0xABCD_1234, "token mismatch");
    assert!(cqe.result > 0, "read failed with errno {}", -cqe.result);

    let expected = {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open temp file for reference read failed");
        file.seek(SeekFrom::Start(512))
            .expect("seek reference file failed");
        let mut buf = vec![0u8; 1536];
        file.read_exact(&mut buf)
            .expect("reference read_exact failed");
        buf
    };

    assert_eq!(
        dst.as_slice(),
        expected.as_slice(),
        "byte mismatch detected"
    );

    let _ = std::fs::remove_file(path);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires Linux io_uring + O_DIRECT-compatible filesystem/alignment"]
fn test_cqe_pipeline_submit_then_token() {
    let (path, _content) = make_tmp_file_with_data();

    let mut engine =
        DirectNvmeEngine::open_with_paths(&[path.as_path()]).expect("engine open failed");
    engine
        .start_cqe_poller(0)
        .expect("failed to start cqe poller");

    let mut dst = PinnedBuffer::alloc(1536).expect("PinnedBuffer::alloc failed");

    let token_a = 0x1111_u64;
    let token_b = 0x2222_u64;
    engine
        .prefetch(0, 512, 1536, &dst, token_a)
        .expect("prefetch A failed");
    engine
        .prefetch(0, 2048, 1536, &dst, token_b)
        .expect("prefetch B failed");

    let r1 = engine
        .completion_rx()
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("missing cqe for first submission");
    let r2 = engine
        .completion_rx()
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("missing cqe for second submission");

    assert!(
        (r1.token == token_a && r2.token == token_b)
            || (r1.token == token_b && r2.token == token_a),
        "expected near-order token completion; got {} then {}",
        r1.token,
        r2.token
    );
    assert!(
        r1.result > 0 && r2.result > 0,
        "unexpected CQE read failure"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn test_io_uring_write_async_round_trip_byte_identical() {
    #[cfg(not(target_os = "linux"))]
    return;

    #[cfg(target_os = "linux")]
    {
        use std::fs::OpenOptions;
        use std::os::unix::fs::FileExt;

        const FILE_SIZE: usize = 4 * 1024 * 1024;
        let path = std::path::PathBuf::from(format!(
            "/tmp/ramflow_io_uring_write_it_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock error")
                .as_nanos()
        ));

        {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .expect("failed to create temp file");
            file.set_len(FILE_SIZE as u64)
                .expect("failed to size temp file");
            file.sync_all().expect("failed to sync temp file");
        }

        let expected = pseudo_random_bytes(FILE_SIZE, 0x5155_EE7A_5A17_E123);
        let mut src = PinnedBuffer::alloc(FILE_SIZE).expect("PinnedBuffer::alloc failed");
        src.as_mut_slice().copy_from_slice(&expected);

        let mut engine =
            DirectNvmeEngine::open_with_paths(&[path.as_path()]).expect("engine open failed");
        engine
            .start_cqe_poller(0)
            .expect("failed to start cqe poller");

        let token = 0x5752_4954_455F_0001;
        engine
            .write_async(0, 0, FILE_SIZE as u64, &src, token)
            .expect("write_async submit failed");

        let cqe = engine
            .completion_rx()
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("timeout waiting for write cqe");
        assert_eq!(cqe.token, token, "token mismatch");
        assert_eq!(
            cqe.result, FILE_SIZE as i32,
            "write completed with unexpected result {}",
            cqe.result
        );

        let file = OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("open temp file for pread failed");
        let mut actual = vec![0u8; FILE_SIZE];
        file.read_exact_at(&mut actual, 0)
            .expect("plain pread failed");
        assert_eq!(actual, expected, "write_async payload mismatch");

        let _ = std::fs::remove_file(path);
    }
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn test_write_async_rejects_misaligned_inputs_without_panic() {
    #[cfg(not(target_os = "linux"))]
    return;

    #[cfg(target_os = "linux")]
    {
        use ramflow::nvme::prefetch::validate_direct_io_alignment;
        use ramflow::RamFlowError;

        let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine open failed");
        let src = PinnedBuffer::alloc(512).expect("PinnedBuffer::alloc failed");

        let misaligned_offset = engine.write_async(0, 1, 512, &src, 0xBAD0);
        assert!(
            matches!(misaligned_offset, Err(RamFlowError::IoUringError(ref error)) if error.kind() == std::io::ErrorKind::InvalidInput),
            "expected InvalidInput for misaligned offset, got {misaligned_offset:?}"
        );

        let misaligned_length = engine.write_async(0, 0, 511, &src, 0xBAD1);
        assert!(
            matches!(misaligned_length, Err(RamFlowError::IoUringError(ref error)) if error.kind() == std::io::ErrorKind::InvalidInput),
            "expected InvalidInput for misaligned length, got {misaligned_length:?}"
        );

        let misaligned_ptr = src.as_ptr().wrapping_add(1);
        let misaligned_buffer = validate_direct_io_alignment(0, 512, misaligned_ptr);
        assert!(
            matches!(misaligned_buffer, Err(RamFlowError::IoUringError(ref error)) if error.kind() == std::io::ErrorKind::InvalidInput),
            "expected InvalidInput for misaligned buffer, got {misaligned_buffer:?}"
        );
    }
}

#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn test_write_async_pause_returns_without_submission() {
    #[cfg(not(target_os = "linux"))]
    return;

    #[cfg(target_os = "linux")]
    {
        use ramflow::RamFlowError;

        let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine open failed");
        let src = PinnedBuffer::alloc(512).expect("PinnedBuffer::alloc failed");
        engine.set_pause(true);

        let result = engine.write_async(0, 0, 512, &src, 0x5041_5553_45);
        assert!(
            matches!(result, Err(RamFlowError::PressurePause(0))),
            "expected PressurePause before submission, got {result:?}"
        );
        assert_eq!(
            engine.outstanding_reads(),
            0,
            "paused write must not increment outstanding I/O"
        );
    }
}

#[cfg(not(target_os = "linux"))]
#[test]
fn test_io_uring_linux_only() {
    // Intentionally empty: io_uring integration is Linux-only.
}
