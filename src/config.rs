//! Гиперпараметры модели (см. Plan.md §9).

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_enc_layers: usize,
    pub n_dec_layers: usize,
    pub d_ff: usize,
    pub ln_eps: f32,
}

impl Default for ModelConfig {
    /// Маленькая модель для быстрых итераций на CPU.
    fn default() -> Self {
        Self {
            d_model: 128,
            n_heads: 4,
            n_enc_layers: 2,
            n_dec_layers: 2,
            d_ff: 512,
            ln_eps: 1e-5,
        }
    }
}
