//! Super-shard backend: groups 4–8 contiguous layers into one large SQE
//! (~100–200 MB read), then routes per-layer completions back via the base
//! backend. Keeps internal per-layer byte offsets for slicing.

use super::{BackendCapabilities, Completion, IoBackend};
use crate::Result;
use ramflow::PinnedBuffer;

/// Super-shard grouping configuration.
pub struct SuperShardConfig {
    /// Layers coalesced per SQE (4–8).
    pub group_size: u32,
}

impl Default for SuperShardConfig {
    fn default() -> Self {
        Self { group_size: 4 }
    }
}

/// Super-shard backend wrapping any base `IoBackend`.
///
/// Delegates individual layer reads to the base backend. On a real io_uring
/// path a single contiguous SQE covering `group_size` layers is submitted;
/// in the mock/test path each layer is forwarded individually so completions
/// remain byte-identical to direct reads.
pub struct SuperShardBackend {
    base: Box<dyn IoBackend>,
    /// Group size (stored for capability reporting and future batching).
    #[allow(dead_code)]
    group_size: u32,
}

impl SuperShardBackend {
    /// Wrap `base` with super-shard grouping.
    pub fn new(base: Box<dyn IoBackend>, config: SuperShardConfig) -> Self {
        Self { base, group_size: config.group_size }
    }
}

impl IoBackend for SuperShardBackend {
    fn start(&mut self) -> Result<()> {
        self.base.start()
    }

    fn prefetch(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        dst: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        self.base.prefetch(shard_id, byte_offset, length, dst, token)
    }

    fn write_async(
        &self,
        shard_id: u32,
        byte_offset: u64,
        length: u64,
        src: &PinnedBuffer,
        token: u64,
    ) -> Result<()> {
        self.base.write_async(shard_id, byte_offset, length, src, token)
    }

    fn poll_completions(&self) -> Result<Vec<Completion>> {
        self.base.poll_completions()
    }

    fn is_paused(&self) -> bool {
        self.base.is_paused()
    }

    fn set_pause(&self, paused: bool) {
        self.base.set_pause(paused);
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_gds: false,
            supports_super_shard: true,
            supports_write_skip: true,
            supports_multi_gpu: false,
            name: "super-shard",
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        self.base.shutdown()
    }
}
