//! T-DET: Determinism test — same inputs produce same ready order.
//!
//! Runs two identical forward sequences (on_layer_start + take_ready for
//! layers 0..N) and asserts both produce ready buffers in the same layer_idx
//! order with identical byte content.

use std::sync::Arc;
use std::time::Duration;

use flowcast::{
    backend::file_read::FileReadBackend,
    completion_router::CompletionRouter,
    state_machine::PrefetchStateMachine,
    Direction, Precision,
};
use ramflow::PoolRegistry;

const TOTAL_LAYERS: u32 = 8;
const LOOKAHEAD: u32 = 2;
const SHARD_BYTES: usize = 512;

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

fn run_forward_sequence(
    sm: &Arc<PrefetchStateMachine>,
    pool: &PoolRegistry,
    backend: &Arc<FileReadBackend>,
) -> Vec<(u32, Vec<u8>)> {
    let _router = CompletionRouter::spawn(backend.clone(), sm.clone())
        .expect("router spawn");

    sm.prime_window(Direction::Forward, pool, backend.as_ref())
        .expect("prime_window");

    let mut results = Vec::new();
    for layer_idx in 0..TOTAL_LAYERS {
        sm.on_layer_start(layer_idx, Direction::Forward, pool, backend.as_ref())
            .expect("on_layer_start");
        let ready = sm
            .take_ready(layer_idx, Duration::from_millis(500))
            .unwrap_or_else(|e| panic!("layer {layer_idx}: {e}"));
        let bytes = ready.as_slice().to_vec();
        results.push((ready.layer_idx, bytes));
    }
    results
}

#[test]
fn test_same_inputs_same_ready_order() {
    let (_dir, paths) = create_temp_shards(TOTAL_LAYERS as usize);
    let backend = Arc::new(FileReadBackend::new(paths.clone()));
    let pool = PoolRegistry::with_defaults().expect("pool");

    // Run 1
    let sm1 = Arc::new(PrefetchStateMachine::new(TOTAL_LAYERS, LOOKAHEAD, Precision::FP16));
    let run1 = run_forward_sequence(&sm1, &pool, &backend);

    // Run 2 — fresh state machine, same backend and pool
    let sm2 = Arc::new(PrefetchStateMachine::new(TOTAL_LAYERS, LOOKAHEAD, Precision::FP16));
    let run2 = run_forward_sequence(&sm2, &pool, &backend);

    // Assert identical order and byte content
    assert_eq!(run1.len(), run2.len(), "result counts must match");
    for (i, ((idx1, bytes1), (idx2, bytes2))) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(idx1, idx2, "step {i}: layer_idx mismatch ({idx1} vs {idx2})");
        assert_eq!(
            bytes1, bytes2,
            "step {i}: layer {idx1} bytes differ between run 1 and run 2"
        );
    }
}
