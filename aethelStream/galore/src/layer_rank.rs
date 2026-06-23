//! Per-layer-type projection rank configuration.

/// Rank assignment by parameter kind.
///
/// Attention projection matrices typically use higher rank (16–64);
/// MLP layers may use lower rank (8–16).
#[derive(Debug, Clone, Copy)]
pub struct LayerRankConfig {
    /// Default rank when no specific rule matches.
    pub default_rank: usize,
    /// Rank for attention projections (`d_wq`, `d_wk`, `d_wv`, `d_wo`).
    pub attn_rank: usize,
    /// Rank for MLP projections (`d_wg`, `d_wu`, `d_wd`).
    pub mlp_rank: usize,
    /// Rank for RMSNorm scale vectors (`d_rms1_w`, `d_rms2_w`).
    pub vector_rank: usize,
}

impl Default for LayerRankConfig {
    fn default() -> Self {
        Self {
            default_rank: 16,
            attn_rank: 32,
            mlp_rank: 16,
            vector_rank: 8,
        }
    }
}

impl LayerRankConfig {
    /// Resolve the projection rank for `(param_name, m, n)`.
    pub fn rank_for_param(&self, param_name: &str, m: usize, n: usize) -> usize {
        let base = match param_name {
            "d_wq" | "d_wk" | "d_wv" | "d_wo" => self.attn_rank,
            "d_wg" | "d_wu" | "d_wd" => self.mlp_rank,
            "d_rms1_w" | "d_rms2_w" => self.vector_rank,
            _ => self.default_rank,
        };
        base.min(m.max(1)).min(n.max(1)).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::LayerRankConfig;

    #[test]
    fn per_layer_rank_attention_higher_than_mlp() {
        let cfg = LayerRankConfig::default();
        assert_eq!(cfg.rank_for_param("d_wq", 512, 512), 32);
        assert_eq!(cfg.rank_for_param("d_wk", 512, 512), 32);
        assert_eq!(cfg.rank_for_param("d_wg", 512, 512), 16);
        assert_eq!(cfg.rank_for_param("d_rms1_w", 1, 512), 1);
    }

    #[test]
    fn per_layer_rank_clamps_to_matrix_dims() {
        let cfg = LayerRankConfig {
            attn_rank: 64,
            mlp_rank: 16,
            ..LayerRankConfig::default()
        };
        assert_eq!(cfg.rank_for_param("d_wq", 32, 32), 32);
        assert_eq!(cfg.rank_for_param("d_wg", 8, 64), 8);
    }
}
