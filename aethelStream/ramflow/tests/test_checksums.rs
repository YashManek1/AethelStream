// tests/test_checksums.rs — per-shard xxHash3 integrity verification
//
// Run: cargo test --features "mock-cuda,checksums" --test test_checksums
//
// All tests in this file are conditional on the `checksums` feature so that
// the CI matrix without that feature compiles and passes without changes.

#![cfg(feature = "checksums")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ramflow::{
    allocator::PinnedBuffer,
    nvme::DirectNvmeEngine,
    pool::TensorLocationDict,
    RamFlowError,
};

// ---------------------------------------------------------------------------
// 1. Happy path — correct checksum passes
// ---------------------------------------------------------------------------

#[test]
fn checksum_happy_path_no_error() {
    let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");

    let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
    let expected = xxhash_rust::xxh3::xxh3_64(&data);

    let mut buf = PinnedBuffer::alloc(4096).expect("alloc");
    buf.as_mut_slice().copy_from_slice(&data);

    engine
        .prefetch_with_checksum(0, 0, 4096, &buf, 10u64, Some(expected))
        .expect("schedule");

    engine.inject_completion_for_test(10u64, 4096);

    let n = engine.poll_completions().expect("poll_completions must succeed");
    assert_eq!(n, 1, "one completion must be returned");
}

// ---------------------------------------------------------------------------
// 2. Corruption detection — single flipped bit → ShardCorrupted
// ---------------------------------------------------------------------------

#[test]
fn checksum_corruption_surfaces_shard_corrupted() {
    let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");

    let data: Vec<u8> = vec![0xABu8; 512];
    let expected = xxhash_rust::xxh3::xxh3_64(&data);

    let mut buf = PinnedBuffer::alloc(512).expect("alloc");
    buf.as_mut_slice().copy_from_slice(&data);

    engine
        .prefetch_with_checksum(7, 0, 512, &buf, 20u64, Some(expected))
        .expect("schedule");

    // Flip one byte after the checksum was registered to simulate bit-rot.
    buf.as_mut_slice()[255] ^= 0x01;

    engine.inject_completion_for_test(20u64, 512);

    match engine.poll_completions() {
        Err(RamFlowError::ShardCorrupted {
            shard_id,
            expected: exp,
            got,
        }) => {
            assert_eq!(shard_id, 7, "shard_id must match the registered value");
            assert_eq!(exp, expected, "expected digest must be the original");
            assert_ne!(got, expected, "computed digest must differ on corruption");
        }
        other => panic!("expected ShardCorrupted, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 3. No checksum — TensorInfo::xxhash3 == None skips verification
// ---------------------------------------------------------------------------

#[test]
fn checksum_none_skips_verification_and_succeeds() {
    let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");

    // Buffer deliberately filled with zeros — no digest computed.
    let buf = PinnedBuffer::alloc(512).expect("alloc");

    engine
        .prefetch_with_checksum(0, 0, 512, &buf, 30u64, None)
        .expect("schedule");

    engine.inject_completion_for_test(30u64, 512);

    let n = engine.poll_completions().expect("must succeed without verification");
    assert_eq!(n, 1);
}

// ---------------------------------------------------------------------------
// 4. TensorLocationDict round-trip — xxh3 field survives JSON parse
// ---------------------------------------------------------------------------

#[test]
fn tensor_location_dict_parses_xxh3_field() {
    let json = br#"{
        "layers": [
            {
                "index": 0,
                "tensors": [
                    {
                        "name": "q_proj",
                        "path": "shard_0000.bin",
                        "byte_offset": 0,
                        "byte_length": 512,
                        "shape": [32, 16],
                        "dtype": "f16",
                        "xxh3": 12345678901234567890
                    }
                ]
            }
        ]
    }"#;

    let dict = TensorLocationDict::from_json_bytes(json, None).expect("parse");
    let info = dict.get(0, "q_proj").expect("tensor must be present");
    assert_eq!(
        info.xxhash3,
        Some(12345678901234567890u64),
        "xxhash3 must round-trip through JSON"
    );
}

// ---------------------------------------------------------------------------
// 5. TensorLocationDict — missing xxh3 field → None (backward compat)
// ---------------------------------------------------------------------------

#[test]
fn tensor_location_dict_missing_xxh3_is_none() {
    let json = br#"{
        "layers": [
            {
                "index": 0,
                "tensors": [
                    {
                        "name": "v_proj",
                        "path": "shard_0000.bin",
                        "byte_offset": 512,
                        "byte_length": 512,
                        "shape": [32, 16],
                        "dtype": "f16"
                    }
                ]
            }
        ]
    }"#;

    let dict = TensorLocationDict::from_json_bytes(json, None).expect("parse");
    let info = dict.get(0, "v_proj").expect("tensor must be present");
    assert!(
        info.xxhash3.is_none(),
        "missing xxh3 JSON field must deserialise to None"
    );
}

// ---------------------------------------------------------------------------
// 6. Multiple in-flight tokens — each verified against its own checksum
// ---------------------------------------------------------------------------

#[test]
fn checksum_multiple_tokens_each_verified_independently() {
    let engine = DirectNvmeEngine::open_with_paths(&[]).expect("engine init");

    // Two distinct 512-byte buffers (O_DIRECT minimum alignment) with different contents.
    let data_a: Vec<u8> = (0u8..=255).cycle().take(512).collect();
    let data_b: Vec<u8> = (0u8..=255).rev().cycle().take(512).collect();
    let expected_a = xxhash_rust::xxh3::xxh3_64(&data_a);
    let expected_b = xxhash_rust::xxh3::xxh3_64(&data_b);

    let mut buf_a = PinnedBuffer::alloc(512).expect("alloc a");
    buf_a.as_mut_slice().copy_from_slice(&data_a);
    let mut buf_b = PinnedBuffer::alloc(512).expect("alloc b");
    buf_b.as_mut_slice().copy_from_slice(&data_b);

    engine
        .prefetch_with_checksum(0, 0, 512, &buf_a, 40u64, Some(expected_a))
        .expect("schedule a");
    engine
        .prefetch_with_checksum(1, 0, 512, &buf_b, 41u64, Some(expected_b))
        .expect("schedule b");

    engine.inject_completion_for_test(40u64, 512);
    engine.inject_completion_for_test(41u64, 512);

    let n = engine.poll_completions().expect("poll_completions");
    assert_eq!(n, 2, "both completions must be returned");
}

// ---------------------------------------------------------------------------
// 7. ShardCorrupted error message format
// ---------------------------------------------------------------------------

#[test]
fn shard_corrupted_display_contains_hex_digests() {
    let err = RamFlowError::ShardCorrupted {
        shard_id: 3,
        expected: 0xDEAD_BEEF_CAFE_0001,
        got: 0xBAD0_F00D_1234_5678,
    };
    let msg = err.to_string();
    // {:#018x} produces lowercase hex with 0x prefix, no underscores, zero-padded to 18 chars
    assert!(
        msg.contains("0xdeadbeefcafe0001"),
        "display must include expected digest in hex; got: {msg}"
    );
    assert!(
        msg.contains("shard 3"),
        "display must include shard_id; got: {msg}"
    );
}
