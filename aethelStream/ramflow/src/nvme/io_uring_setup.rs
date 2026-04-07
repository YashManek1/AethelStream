// src/nvme/io_uring_setup.rs — io_uring ring initialisation stub

/// Parameters passed to io_uring_setup(2).
pub struct IoUringParams {
    /// Number of submission-queue entries.
    pub sq_entries: u32,
    /// Number of completion-queue entries (typically 2× sq_entries).
    pub cq_entries: u32,
}

/// Owns a configured io_uring instance.
pub struct IoUringInstance {
    _opaque: (),
}

impl IoUringInstance {
    #[allow(unused_variables)]
    pub fn setup(params: IoUringParams) -> crate::Result<Self> {
        unimplemented!("IoUringInstance::setup — Sprint 0 stub")
    }
}
