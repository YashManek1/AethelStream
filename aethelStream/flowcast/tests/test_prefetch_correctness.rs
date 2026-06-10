//! T2: Prefetch byte-identity correctness test.
//!
//! For 20 consecutive forward layers, asserts that the buffer delivered by
//! the state machine is byte-identical to a direct synchronous read of the
//! same shard file.  Uses FileReadBackend so the test runs on Windows/CI
//! without io_uring.

use std::sync::Arc;
use std::time::Duration;

use flowcast::{
    backend::file_read::FileReadBackend,
    completion_router::CompletionRouter,
    state_machine::PrefetchStateMachine,
    Direction, Precision,
};
use ramflow::PoolRegistry;

const TOTAL_LAYERS: u32 = 20;
const LOOKAHEAD: u32 = 2;
/// 512-byte payload -- smallest O_DIRECT-aligned size; always fits in any pool slot.
const SHARD_BYTES: usize = 512;

/// Create N temp shard files each filled with `[i as u8; SHARD_BYTES]`.
fn create_temp_shards(n: usize) -> (tempfile::TempDir, Vec<std::path::PathBuf>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut paths = Vec::with_capacity(n);
    for i in 0..n {
        let path = dir.path().join(format!("shard_{:04}.bin", i));
        let content = vec![(i % 256) as u8; SHARD_BYTES];
        std::fs::write(&path, &content).expect("write shard");
        paths.push(path);
    }
    (dir, paths)
}

#[test]
fn test_prefetch_byte_identical() {
    let (_dir, paths) = create_temp_shards(TOTAL_LAYERS as usize);

    let pool = PoolRegistry::with_defaults().expect("PoolRegistry::with_defaults");
    let backend = Arc::new(FileReadBackend::new(paths.clone()));
    let sm = Arc::new(PrefetchStateMachine::new(TOTAL_LAYERS, LOOKAHEAD, Precision::FP16));

    let _router = CompletionRouter::spawn(backend.clone(), sm.clone())
        .expect("CompletionRouter::spawn");

    // Prime: submit layers 0..LOOKAHEAD before the loop.
    sm.prime_window(Direction::Forward, &pool, &*backend)
        .expect("prime_window");

    for layer_idx in 0..TOTAL_LAYERS {
        // Advance window: submit layer_idx+1..=layer_idx+W.
        sm.on_layer_start(layer_idx, Direction::Forward, &pool, &*backend)
            .expect("on_layer_start");

        // Wait for the current layer to arrive (primed or submitted by previous iteration).
        let ready = sm
            .take_ready(layer_idx, Duration::from_millis(500))
            .unwrap_or_else(|e| panic!("layer {layer_idx}: {e}"));

        // Compare buffer bytes with direct file read.
        let expected = std::fs::read(&paths[layer_idx as usize]).expect("read shard file");
        let actual = ready.as_slice();
        assert_eq!(
            &actual[..expected.len()],
            expected.as_slice(),
            "layer {layer_idx}: buffer content does not match shard file"
        );
    }
}
