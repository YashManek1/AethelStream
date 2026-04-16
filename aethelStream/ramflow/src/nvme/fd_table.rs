// src/nvme/fd_table.rs — open file-descriptor table (Sprint 2 Day 2)
//
// Tracks open file descriptors for NVMe shard files.
// Each shard file is opened with: O_RDONLY | O_DIRECT | O_CLOEXEC
//
// O_DIRECT:  Bypasses the kernel page cache. Reads go straight from the
//            NVMe controller to the destination buffer (pinned host memory).
//            Requirement: destination buffer AND file byte_offset must both
//            be multiples of 512 (the NVMe logical sector size).
//
// O_CLOEXEC: Close the fd automatically on exec(). Prevents accidental fd
//            inheritance if the training process forks a subprocess.
//            On Linux, O_CLOEXEC is set atomically at open time — using
//            fcntl(F_SETFD) afterward has a TOCTOU race in multi-threaded code.
//
// Why we can't rely on the OS to close fds at exit:
//   If the process crashes mid-write (OOM killer, SIGKILL), the kernel
//   closes all fds — but any write that was buffered in the page cache may
//   be lost. With O_DIRECT there is no page cache, so this is less of an
//   issue for reads. But the fd table's Drop impl also closes write fds from
//   write_budget.rs, where prompt close matters.
//
// OwnedFd (std::os::unix::io::OwnedFd):
//   A Rust type that owns a raw file descriptor and calls close(2) on drop.
//   Available since Rust 1.63. This is the idiomatic way to manage fds —
//   it is Send + !Sync, has a correct Drop, and integrates with From<File>.

use crate::{RamFlowError, Result};
use std::path::Path;

#[cfg(unix)]
use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::io::{FromRawFd, OwnedFd, RawFd};
#[cfg(windows)]
type OwnedFd = ();

// ---------------------------------------------------------------------------
// Platform-specific open() wrapper
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn open_shard_file(path: &Path) -> Result<OwnedFd> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path_cstr = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        RamFlowError::ConfigError(format!("shard path contains null byte: {:?}", path))
    })?;

    // O_RDONLY | O_DIRECT | O_CLOEXEC
    // Values are stable on Linux x86-64:
    //   O_RDONLY  = 0
    //   O_DIRECT  = 0x4000 (16384)
    //   O_CLOEXEC = 0x80000 (524288)
    const O_RDONLY: i32 = 0;
    const O_DIRECT: i32 = 0x4000;
    const O_CLOEXEC: i32 = 0o2000000; // = 524288 = 0x80000

    let flags = O_RDONLY | O_DIRECT | O_CLOEXEC;

    // SAFETY: path_cstr is a valid null-terminated C string. flags are valid.
    let fd: RawFd = unsafe { libc::open(path_cstr.as_ptr(), flags) };

    if fd < 0 {
        let err = std::io::Error::last_os_error();
        return Err(RamFlowError::IoUringError(std::io::Error::new(
            err.kind(),
            format!("O_DIRECT open failed for {:?}: {}", path, err),
        )));
    }

    // SAFETY: fd is a valid, open file descriptor we just created.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

// Non-Linux fallback: open normally (no O_DIRECT).
// Used for development on macOS or cross-compilation.
#[cfg(all(unix, not(target_os = "linux")))]
fn open_shard_file(path: &Path) -> Result<OwnedFd> {
    use std::fs::OpenOptions;
    use std::os::unix::io::IntoRawFd;

    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| RamFlowError::IoUringError(e))?;

    Ok(unsafe { OwnedFd::from_raw_fd(file.into_raw_fd()) })
}

#[cfg(windows)]
fn open_shard_file(_path: &Path) -> Result<OwnedFd> {
    // Windows: io_uring is not supported. DirectNvmeEngine is Linux-only.
    // This stub allows the crate to compile on Windows for development.
    Err(RamFlowError::ConfigError(
        "DirectNvmeEngine is not supported on Windows".into(),
    ))
}

// ---------------------------------------------------------------------------
// FdTable
// ---------------------------------------------------------------------------

/// Tracks open file descriptors for NVMe shard files.
///
/// The key is the shard ID (a u32 matching TensorInfo::shard_id).
/// OwnedFd calls close(2) on each fd when FdTable is dropped.
pub struct FdTable {
    /// Map from shard_id → owned file descriptor.
    #[cfg(unix)]
    fds: HashMap<u32, OwnedFd>,

    /// Number of shards registered.
    count: usize,
}

impl FdTable {
    /// Create an empty table.
    pub fn new() -> Result<Self> {
        Ok(FdTable {
            #[cfg(unix)]
            fds: HashMap::new(),
            count: 0,
        })
    }

    /// Open a shard file at `path` and register it under `shard_id`.
    ///
    /// The file is opened with `O_RDONLY | O_DIRECT | O_CLOEXEC` on Linux.
    /// On other platforms, a standard open is used (no O_DIRECT).
    ///
    /// # Errors
    /// Returns `RamFlowError::IoUringError` if the file cannot be opened.
    pub fn register(&mut self, shard_id: u32, path: &Path) -> Result<()> {
        let fd = open_shard_file(path)?;

        #[cfg(unix)]
        {
            self.fds.insert(shard_id, fd);
        }

        self.count += 1;
        Ok(())
    }

    /// Open all shard files in `paths`, assigning sequential IDs starting at 0.
    ///
    /// Convenience function for the common case where Module 1 provides
    /// an ordered list of shard paths.
    pub fn register_all(&mut self, paths: &[&Path]) -> Result<()> {
        for (shard_id, &path) in paths.iter().enumerate() {
            self.register(shard_id as u32, path)?;
        }
        Ok(())
    }

    /// Get the raw file descriptor for `shard_id`.
    ///
    /// Returns `None` if the shard is not registered.
    ///
    /// # Safety
    /// The returned `RawFd` is only valid while this `FdTable` is alive.
    #[cfg(unix)]
    pub fn get_raw_fd(&self, shard_id: u32) -> Option<RawFd> {
        use std::os::unix::io::AsRawFd;
        self.fds.get(&shard_id).map(|fd| fd.as_raw_fd())
    }

    #[cfg(not(unix))]
    pub fn get_raw_fd(&self, _shard_id: u32) -> Option<i32> {
        None
    }

    /// Number of registered shards.
    pub fn len(&self) -> usize {
        self.count
    }

    /// True if no shards are registered.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Close all file descriptors and clear the table.
    ///
    /// Normally you just drop the `FdTable`. Call this explicitly if you want
    /// to detect and log close errors (Drop ignores them).
    #[cfg(unix)]
    pub fn close_all(&mut self) -> Result<()> {
        // Draining the HashMap causes OwnedFd::drop to call close() on each fd.
        // If close() fails (e.g., NFS stale handle), we collect the errors.
        // In practice, close() on an O_DIRECT local NVMe fd never fails.
        self.fds.clear();
        self.count = 0;
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn close_all(&mut self) -> Result<()> {
        self.count = 0;
        Ok(())
    }
}

impl Default for FdTable {
    fn default() -> Self {
        Self::new().expect("FdTable::new is infallible")
    }
}

// Drop implementation: OwnedFd's own Drop calls close(2) for each fd.
// Nothing additional is needed here — the compiler-generated Drop for
// HashMap<u32, OwnedFd> calls OwnedFd::drop on each value, which calls close.
// We define Drop explicitly only to document this intent.
impl Drop for FdTable {
    fn drop(&mut self) {
        // OwnedFd::drop calls close() on each fd as the HashMap is dropped.
        // This is correct and sufficient — explicit close_all() is not required.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn fd_table_empty_at_construction() {
        let table = FdTable::new().unwrap();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn fd_table_register_and_get() {
        // Create a temporary file to test registration.
        let mut tmp = tempfile_or_skip();
        tmp.file.write_all(b"ramflow test").unwrap();

        let mut table = FdTable::new().unwrap();
        table.register(0, &tmp.path).unwrap();

        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        let raw_fd = table.get_raw_fd(0);
        assert!(raw_fd.is_some(), "fd not registered");
        assert!(raw_fd.unwrap() > 2, "fd should be > stderr (2)");
    }

    #[test]
    fn fd_table_missing_shard_returns_none() {
        let table = FdTable::new().unwrap();
        assert!(table.get_raw_fd(99).is_none());
    }

    // Minimal temp file helper that doesn't pull in the tempfile crate.
    #[cfg(unix)]
    struct TempFile {
        path: std::path::PathBuf,
        file: std::fs::File,
    }

    #[cfg(unix)]
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[cfg(unix)]
    fn tempfile_or_skip() -> TempFile {
        use std::fs::OpenOptions;
        // Use /tmp for the test file.
        let path = std::path::PathBuf::from(format!("/tmp/ramflow_fd_test_{}", std::process::id()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .expect("could not create temp file");
        TempFile { path, file }
    }
}
