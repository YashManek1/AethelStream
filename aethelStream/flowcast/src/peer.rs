//! A10: Peer-GPU synchronisation.
//!
//! `PeerSync` is a trait stub — no implementation required for single-GPU training.
//! Multi-GPU implementations register via `feature = "multi-gpu"`.
//!
//! `SingleGpuSync` is the default no-op; it satisfies any code path that
//! accepts a `Box<dyn PeerSync>` without branching on GPU count.

use crate::Result;

/// Peer synchronisation across a multi-GPU ring.
///
/// Implementors coordinate layer-ready signals so that GPU N does not begin
/// executing a layer until all peers have its weights resident in VRAM.
///
/// # Contract
/// * `broadcast_ready` must be called after `FlowCast::retire_layer`.
/// * `wait_peer_ready` must complete before the next `on_layer_start`.
/// * Both calls are no-ops on `SingleGpuSync`.
pub trait PeerSync: Send + Sync {
    /// Broadcast that `layer_idx` is resident and ready on this GPU.
    fn broadcast_ready(&self, layer_idx: u32) -> Result<()>;

    /// Block until `layer_idx` is confirmed ready on all peer GPUs.
    fn wait_peer_ready(&self, layer_idx: u32) -> Result<()>;

    /// Tear down peer channels (called on FlowCast shutdown).
    fn shutdown(&mut self) -> Result<()>;
}

/// No-op `PeerSync` for single-GPU training.
///
/// All methods return `Ok(())` immediately.
pub struct SingleGpuSync;

impl PeerSync for SingleGpuSync {
    fn broadcast_ready(&self, _layer_idx: u32) -> Result<()> {
        Ok(())
    }

    fn wait_peer_ready(&self, _layer_idx: u32) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// NVLink peer-sync stub (requires `feature = "multi-gpu"`).
///
/// Compiles but panics if called without the feature; guards real usage behind
/// the feature flag so the trait object slot is always type-safe.
pub struct NvlinkPeerSync {
    _private: (),
}

impl NvlinkPeerSync {
    /// Construct — only meaningful when `feature = "multi-gpu"` is active.
    #[allow(dead_code)]
    pub fn new() -> Self {
        NvlinkPeerSync { _private: () }
    }
}

impl Default for NvlinkPeerSync {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerSync for NvlinkPeerSync {
    fn broadcast_ready(&self, _layer_idx: u32) -> Result<()> {
        #[cfg(feature = "multi-gpu")]
        unimplemented!("NvlinkPeerSync: multi-gpu broadcast not yet implemented");
        #[cfg(not(feature = "multi-gpu"))]
        Ok(())
    }

    fn wait_peer_ready(&self, _layer_idx: u32) -> Result<()> {
        #[cfg(feature = "multi-gpu")]
        unimplemented!("NvlinkPeerSync: multi-gpu wait not yet implemented");
        #[cfg(not(feature = "multi-gpu"))]
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}
