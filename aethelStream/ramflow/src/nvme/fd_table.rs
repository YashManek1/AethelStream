// src/nvme/fd_table.rs — open file-descriptor table stub

/// Tracks open file descriptors for NVMe devices registered with io_uring.
pub struct FdTable {
    _opaque: (),
}

impl FdTable {
    pub fn new() -> crate::Result<Self> {
        unimplemented!("FdTable::new — Sprint 0 stub")
    }
}

impl Default for FdTable {
    fn default() -> Self {
        Self::new().expect("FdTable default")
    }
}
