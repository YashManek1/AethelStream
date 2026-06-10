//! io_uring backend wrapping `ramflow::DirectNvmeEngine`.
//!
//! Implemented for Linux (io_uring is a Linux kernel interface).  On other
//! platforms, every method returns a `BackendIo` error so the rest of the
//! codebase can still reference the type without conditional imports.
//!
//! # Thread safety
//! `DirectNvmeEngine` contains a `Receiver<CqeResult>` which is `!Sync`.
//! We wrap it in `Arc<Mutex<_>>` so it is accessible from both the prefetch
//! thread and the completion-router thread.  The mutex is held only for the
//! duration of a single SQE submission or CQE batch drain -- negligible
//! compared with NVMe I/O latency.

use super::{BackendCapabilities, Completion, IoBackend};
use crate::{FlowCastError, Result};
use ramflow::PinnedBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// io_uring backend -- thin wrapper over `ramflow::DirectNvmeEngine`.
#[allow(dead_code)]
pub struct UringBackend {
    engine: Arc<Mutex<ramflow::DirectNvmeEngine>>,
    paused: Arc<AtomicBool>,
    /// CPU core for the io_uring CQE poller thread (used in `start`).
    io_poller_cpu_core: usize,
}

impl UringBackend {
    /// Open shards under `shard_dir` and build the io_uring ring.
    ///
    /// Returns `Err(BackendIo(...))` immediately on non-Linux platforms.
    ///
    /// # Errors
    /// * [`FlowCastError::BackendIo`] -- not on Linux, or engine open failed.
    pub fn new(
        shard_dir: &std::path::Path,
        num_shards: u32,
        io_poller_cpu_core: usize,
    ) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            let engine = ramflow::DirectNvmeEngine::open(shard_dir, num_shards)
                .map_err(FlowCastError::RamFlow)?;
            return Ok(Self {
                engine: Arc::new(Mutex::new(engine)),
                paused: Arc::new(AtomicBool::new(false)),
                io_poller_cpu_core,
            });
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (shard_dir, num_shards, io_poller_cpu_core);
            Err(FlowCastError::BackendIo(
                "UringBackend requires Linux (io_uring is Linux-only)".to_string(),
            ))
        }
    }
}

impl IoBackend for UringBackend {
    fn start(&mut self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let engine = self.engine.lock().map_err(|p| {
                FlowCastError::BackendIo(format!("uring engine lock poisoned: {p}"))
            })?;
            return engine
                .start_cqe_poller(self.io_poller_cpu_core)
                .map_err(FlowCastError::RamFlow);
        }
        #[cfg(not(target_os = "linux"))]
        Err(FlowCastError::BackendIo(
            "UringBackend requires Linux".to_string(),
        ))
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let engine = self.engine.lock().map_err(|p| {
                FlowCastError::BackendIo(format!("uring engine lock poisoned: {p}"))
            })?;
            return engine
                .prefetch(shard_id, byte_offset, length, dst, token)
                .map_err(FlowCastError::RamFlow);
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (shard_id, byte_offset, length, dst, token);
            Err(FlowCastError::BackendIo(
                "UringBackend requires Linux".to_string(),
            ))
        }
    }

    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let engine = self.engine.lock().map_err(|p| {
                FlowCastError::BackendIo(format!("uring engine lock poisoned: {p}"))
            })?;
            return engine
                .write_async(shard_id, byte_offset, length, src, token)
                .map_err(FlowCastError::RamFlow);
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (shard_id, byte_offset, length, src, token);
            Err(FlowCastError::BackendIo(
                "UringBackend requires Linux".to_string(),
            ))
        }
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        #[cfg(target_os = "linux")]
        {
            let engine = self.engine.lock().map_err(|p| {
                FlowCastError::BackendIo(format!("uring engine lock poisoned: {p}"))
            })?;
            let mut completions = Vec::new();
            loop {
                match engine.completion_rx().try_recv() {
                    Ok(cqe) => completions.push(Completion {
                        token: cqe.token,
                        result: cqe.result,
                    }),
                    Err(_) => break,
                }
            }
            return Ok(completions);
        }
        #[cfg(not(target_os = "linux"))]
        Ok(Vec::new())
    }

    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    fn set_pause(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
        #[cfg(target_os = "linux")]
        if let Ok(engine) = self.engine.lock() {
            engine.set_pause(paused);
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_gds: false,
            supports_super_shard: true,
            supports_write_skip: true,
            supports_multi_gpu: false,
            name: "uring",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
