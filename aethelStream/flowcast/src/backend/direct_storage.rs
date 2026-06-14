//! Windows DirectStorage I/O backend for FlowCast.
//!
//! Wraps [`ramflow::nvme::direct_storage::DirectStorageQueue`] when
//! `feature = "direct-storage"` is active and the DirectStorage runtime DLL
//! is present at startup.  Falls back to synchronous Windows `ReadFile` via
//! [`FileReadBackend`] when the DLL is absent or any initialisation step fails.
//!
//! # Backend selection priority
//! ```text
//! DirectStorage (Windows + DLL present)
//!   → FileReadBackend (Windows, DLL absent — blocking ReadFile)
//!   → MockBackend (non-Windows / CI)
//! ```
//!
//! # Alignment note
//! The memory-destination path used here writes into a [`ramflow::PinnedBuffer`]
//! that is 512-byte aligned by default.  This is sufficient for the
//! `DSTORAGE_REQUEST_DESTINATION_MEMORY` path.  For the GPU-VRAM path
//! (`DSTORAGE_REQUEST_DESTINATION_BUFFER`) allocate with
//! [`ramflow::nvme::direct_storage::alloc_windows_ds_compatible`] to obtain
//! 4 096-byte alignment.
//!
//! # Thread safety
//! [`DirectStorageBackend`] is `Send + Sync`; `DirectStorageQueue` documents
//! that `IDStorageQueue::EnqueueRequest` and `Submit` are thread-safe
//! (DirectStorage SDK 1.2, §4 "Threading").

use super::file_read::FileReadBackend;
use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
#[cfg(all(target_os = "windows", feature = "direct-storage"))]
use crate::FlowCastError;
use ramflow::PinnedBuffer;
use std::path::PathBuf;
#[cfg(all(target_os = "windows", feature = "direct-storage"))]
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// FlowCast I/O backend that uses Windows DirectStorage for zero-copy SSD reads.
///
/// Constructed via [`DirectStorageBackend::new`].  On non-Windows platforms or
/// when the DirectStorage DLL is unavailable, degrades transparently to
/// synchronous `ReadFile`.
pub struct DirectStorageBackend {
    inner: DirectStorageInner,
}

// ---------------------------------------------------------------------------
// Inner enum
// ---------------------------------------------------------------------------

enum DirectStorageInner {
    /// DirectStorage COM queue — Windows + DLL present + feature enabled.
    #[cfg(all(target_os = "windows", feature = "direct-storage"))]
    Real(RealPath),
    /// Blocking ReadFile fallback — always available.
    ReadFileFallback(FileReadBackend),
}

/// Windows-side real path: wraps `DirectStorageQueue` and tracks pause state.
#[cfg(all(target_os = "windows", feature = "direct-storage"))]
struct RealPath {
    queue: ramflow::nvme::direct_storage::DirectStorageQueue,
    paused: AtomicBool,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl DirectStorageBackend {
    /// Open the DirectStorage backend for `shard_paths`.
    ///
    /// On Windows with `feature = "direct-storage"` enabled:
    /// * Probes for `dstorage.dll` via [`ramflow::nvme::direct_storage::probe_direct_storage`].
    /// * If available, opens a `DirectStorageQueue` covering all shard files.
    /// * If the DLL is absent or queue creation fails, falls back to `FileReadBackend`.
    ///
    /// On non-Windows, always uses `FileReadBackend`.
    ///
    /// # Errors
    /// Never returns `Err`; all failure modes silently degrade to the ReadFile path.
    pub fn new(shard_paths: Vec<PathBuf>) -> Self {
        #[cfg(all(target_os = "windows", feature = "direct-storage"))]
        {
            use ramflow::nvme::direct_storage::{
                probe_direct_storage, DirectStorageCapability, DirectStorageQueue,
            };

            if matches!(
                probe_direct_storage(),
                DirectStorageCapability::Available { .. }
            ) {
                let path_refs: Vec<&std::path::Path> =
                    shard_paths.iter().map(PathBuf::as_path).collect();
                if let Ok(queue) = DirectStorageQueue::open(&path_refs) {
                    return Self {
                        inner: DirectStorageInner::Real(RealPath {
                            queue,
                            paused: AtomicBool::new(false),
                        }),
                    };
                }
            }
        }

        // Fallback: synchronous ReadFile on all platforms.
        Self {
            inner: DirectStorageInner::ReadFileFallback(FileReadBackend::new(shard_paths)),
        }
    }

    /// Create a backend that always uses the ReadFile fallback.
    ///
    /// Useful in tests that don't have DirectStorage hardware available.
    pub fn new_readfile_fallback(shard_paths: Vec<PathBuf>) -> Self {
        Self {
            inner: DirectStorageInner::ReadFileFallback(FileReadBackend::new(shard_paths)),
        }
    }

    /// True when the backend is using the real DirectStorage COM path.
    pub fn is_using_direct_storage(&self) -> bool {
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(_) => true,
            DirectStorageInner::ReadFileFallback(_) => false,
        }
    }
}

// ---------------------------------------------------------------------------
// IoBackend implementation
// ---------------------------------------------------------------------------

impl IoBackend for DirectStorageBackend {
    fn start(&mut self) -> Result<()> {
        match &mut self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(_) => Ok(()), // DS queue is ready after open()
            DirectStorageInner::ReadFileFallback(fb) => fb.start(),
        }
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(rp) => {
                if rp.paused.load(Ordering::Acquire) {
                    return Err(FlowCastError::BackendIo(
                        "DirectStorage prefetch paused by memory pressure".to_string(),
                    ));
                }
                rp.queue
                    .enqueue_read(shard_id, byte_offset, length, dst, token)
                    .map_err(|ramflow_error| {
                        FlowCastError::BackendIo(format!(
                            "DirectStorage enqueue_read: {ramflow_error}"
                        ))
                    })
            }
            DirectStorageInner::ReadFileFallback(fb) => {
                fb.prefetch(shard_id, byte_offset, length, dst, token)
            }
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
        // DirectStorage 1.2 does not support a GPU-initiated write path.
        // Writes always go through the ReadFile/WriteFile sync path regardless
        // of whether DirectStorage is active for reads.
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(_) => {
                // Delegate to the write side of FileReadBackend which calls WriteFile.
                let fallback = FileReadBackend::new(Vec::new());
                fallback.write_async(shard_id, byte_offset, length, src, token)
            }
            DirectStorageInner::ReadFileFallback(fb) => {
                fb.write_async(shard_id, byte_offset, length, src, token)
            }
        }
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(rp) => {
                rp.queue
                    .poll_completions()
                    .map(|pairs| {
                        pairs
                            .into_iter()
                            .map(|(token, result)| Completion { token, result })
                            .collect()
                    })
                    .map_err(|ramflow_error| {
                        FlowCastError::BackendIo(format!(
                            "DirectStorage poll_completions: {ramflow_error}"
                        ))
                    })
            }
            DirectStorageInner::ReadFileFallback(fb) => fb.poll_completions(),
        }
    }

    fn is_paused(&self) -> bool {
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(rp) => rp.paused.load(Ordering::Acquire),
            DirectStorageInner::ReadFileFallback(fb) => fb.is_paused(),
        }
    }

    fn set_pause(&self, paused: bool) {
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(rp) => {
                rp.paused.store(paused, Ordering::Release);
                rp.queue.set_paused(paused);
            }
            DirectStorageInner::ReadFileFallback(fb) => fb.set_pause(paused),
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        match &self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(_) => BackendCapabilities {
                supports_gds: true, // DirectStorage IS a GPU-direct storage path
                supports_super_shard: false,
                supports_write_skip: false,
                supports_multi_gpu: false,
                name: "direct-storage",
            },
            DirectStorageInner::ReadFileFallback(_) => BackendCapabilities {
                supports_gds: false,
                supports_super_shard: false,
                supports_write_skip: false,
                supports_multi_gpu: false,
                name: "direct-storage-readfile",
            },
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        match &mut self.inner {
            #[cfg(all(target_os = "windows", feature = "direct-storage"))]
            DirectStorageInner::Real(_) => Ok(()), // COM objects released on drop
            DirectStorageInner::ReadFileFallback(fb) => fb.shutdown(),
        }
    }
}
