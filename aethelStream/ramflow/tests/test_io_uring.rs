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

#[cfg(not(target_os = "linux"))]
#[test]
fn test_io_uring_linux_only() {
    // Intentionally empty: io_uring integration is Linux-only.
}
