#![allow(missing_docs)]

//! Training plan types consumed from M9 (SARP) and user configuration.

/// Number of distinct compute stages in one transformer block.
pub const NUM_OPS: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(usize)]
/// The seven compute stages of one transformer block, in execution order.
pub enum OpKind {
    /// First RMSNorm (pre-attention).
    Rms1 = 0,
    /// Q / K / V projection matmuls.
    QkvProj = 1,
    /// Attention scores, softmax, and weighted-V aggregation.
    AttnSoftmax = 2,
    /// Output projection + dropout.
    OutProj = 3,
    /// Second RMSNorm (pre-MLP).
    Rms2 = 4,
    /// Gate and Up projections + SiLU activation.
    MlpGateUp = 5,
    /// Down projection + residual add.
    MlpDown = 6,
}

impl OpKind {
    pub fn all() -> impl Iterator<Item = OpKind> {
        [
            OpKind::Rms1,
            OpKind::QkvProj,
            OpKind::AttnSoftmax,
            OpKind::OutProj,
            OpKind::Rms2,
            OpKind::MlpGateUp,
            OpKind::MlpDown,
        ]
        .into_iter()
    }

    /// Item.
    pub fn flop_fraction(self) -> f64 {
        match self {
            OpKind::Rms1 => 0.02,
            OpKind::QkvProj => 0.27,
            OpKind::AttnSoftmax => 0.08,
            OpKind::OutProj => 0.09,
            OpKind::Rms2 => 0.02,
            OpKind::MlpGateUp => 0.27,
            OpKind::MlpDown => 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Item.
pub struct SelectiveRecomputeMask {
    pub ops: [bool; NUM_OPS],
}

impl SelectiveRecomputeMask {
    /// Item.
    pub fn attn_interior_only() -> Self {
        let mut ops = [false; NUM_OPS];
        ops[OpKind::AttnSoftmax as usize] = true;
        Self { ops }
    }

    /// Item.
    pub fn full_recompute() -> Self {
        Self {
            ops: [true; NUM_OPS],
        }
    }

    /// Item.
    pub fn retain_all() -> Self {
        Self {
            ops: [false; NUM_OPS],
        }
    }

    #[inline]
    /// Item.
    pub fn should_recompute(&self, op: OpKind) -> bool {
        self.ops[op as usize]
    }

    /// Item.
    pub fn is_full_recompute(&self) -> bool {
        self.ops.iter().all(|&b| b)
    }

    /// Item.
    pub fn recompute_flop_fraction(&self) -> f64 {
        OpKind::all()
            .filter(|&op| self.should_recompute(op))
            .map(OpKind::flop_fraction)
            .sum()
    }
}

impl Default for SelectiveRecomputeMask {
    fn default() -> Self {
        Self::full_recompute()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Item.
pub enum ActivationAction {
    Recompute,
    RetainVram,
    PageCompressedRam,
    PageNvme,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Item.
pub struct SegmentPlan {
    pub segment_index: u32,
    pub action: ActivationAction,
    pub recompute_ops: Vec<bool>,
}

impl SegmentPlan {
    /// Item.
    pub fn with_full_recompute(segment_index: u32) -> Self {
        Self {
            segment_index,
            action: ActivationAction::Recompute,
            recompute_ops: vec![true; NUM_OPS],
        }
    }

    /// Item.
    pub fn with_selective_recompute(segment_index: u32) -> Self {
        let mask = SelectiveRecomputeMask::attn_interior_only();
        Self {
            segment_index,
            action: ActivationAction::Recompute,
            recompute_ops: mask.ops.to_vec(),
        }
    }

    /// Item.
    pub fn retain_vram(segment_index: u32) -> Self {
        Self {
            segment_index,
            action: ActivationAction::RetainVram,
            recompute_ops: vec![false; NUM_OPS],
        }
    }

    /// Item.
    pub fn page_compressed_ram(segment_index: u32) -> Self {
        Self {
            segment_index,
            action: ActivationAction::PageCompressedRam,
            recompute_ops: vec![false; NUM_OPS],
        }
    }

    /// Item.
    pub fn page_nvme(segment_index: u32) -> Self {
        Self {
            segment_index,
            action: ActivationAction::PageNvme,
            recompute_ops: vec![false; NUM_OPS],
        }
    }

    /// Item.
    pub fn selective_mask(&self) -> SelectiveRecomputeMask {
        let mut mask = SelectiveRecomputeMask::full_recompute();
        for (index, &flag) in self.recompute_ops.iter().enumerate() {
            if index < NUM_OPS {
                mask.ops[index] = flag;
            }
        }
        mask
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Item.
pub enum TrainingTier {
    FullGaLore,
    LoraOnly,
    TopKFreeze(u32),
    Int4Everywhere,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Item.
pub struct TrainingPlan {
    pub checkpoint_freq: u32,
    pub micro_batch: u32,
    pub grad_accum: u32,
    pub precision_schedule: Vec<crate::Precision>,
    pub optimizer_rank: u32,
    pub tier: TrainingTier,
    pub w_max_hint: u32,
    pub activation_schedule: Vec<SegmentPlan>,
    pub parity_check_interval: u64,
    pub projection_refresh_interval: u64,
    pub max_grad_norm: f32,
}

impl TrainingPlan {
    /// Item.
    pub fn has_sarp_schedule(&self) -> bool {
        !self.activation_schedule.is_empty()
    }

    /// Item.
    pub fn segment_plan(&self, segment_index: u32) -> Option<&SegmentPlan> {
        self.activation_schedule
            .iter()
            .find(|sp| sp.segment_index == segment_index)
    }
}

impl Default for TrainingPlan {
    fn default() -> Self {
        Self {
            checkpoint_freq: 4,
            micro_batch: 2048,
            grad_accum: 2,
            precision_schedule: Vec::new(),
            optimizer_rank: 64,
            tier: TrainingTier::LoraOnly,
            w_max_hint: 4,
            activation_schedule: Vec::new(),
            parity_check_interval: 500,
            projection_refresh_interval: 200,
            max_grad_norm: 1.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Item.
pub struct PlanDelta {
    pub w_max_hint: Option<u32>,
    pub checkpoint_freq: Option<u32>,
    pub precision_overrides: Vec<(u32, crate::Precision)>,
}
