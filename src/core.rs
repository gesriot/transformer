//! Ядро трансформера (Plan.md §1, §2): encoder/decoder стеки в Pre-LN схеме.
//!
//! Pre-LN: `x = x + Sublayer(LN(x))`. Норма перед подслоем — стабильнее учится
//! без lr-warmup. Маски (causal / full) — параметр задачи, передаются снаружи.

use crate::config::ModelConfig;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::ffn::FeedForward;
use crate::nn::layernorm::LayerNorm;
use crate::tensor::Tensor;
use ndarray::ArrayD;
#[cfg(any(feature = "demo", test))]
use ndarray::IxDyn;

/// Аддитивная causal-маска `[seq, seq]`: 0 на разрешённых позициях,
/// большое отрицательное на будущих (запрещённых) — добавляется к score
/// до softmax. Для авторегрессии декодера (char-LM).
#[cfg(any(feature = "demo", test))]
pub(crate) fn causal_mask(seq_len: usize) -> ArrayD<f32> {
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[seq_len, seq_len]));
    for q in 0..seq_len {
        for k in (q + 1)..seq_len {
            mask[IxDyn(&[q, k])] = -1e9;
        }
    }
    mask
}

struct EncoderLayer {
    ln_attn: LayerNorm,
    attn: MultiHeadAttention,
    ln_ffn: LayerNorm,
    ffn: FeedForward,
}

impl EncoderLayer {
    fn new(cfg: &ModelConfig) -> Self {
        Self {
            ln_attn: LayerNorm::new(cfg.d_model, cfg.ln_eps),
            attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads),
            ln_ffn: LayerNorm::new(cfg.d_model, cfg.ln_eps),
            ffn: FeedForward::new(cfg.d_model, cfg.d_ff),
        }
    }

    fn forward(&self, x: &Tensor, mask: Option<&ArrayD<f32>>) -> Tensor {
        let x = x.add(&self.attn.self_attention(&self.ln_attn.forward(x), mask));
        x.add(&self.ffn.forward(&self.ln_ffn.forward(&x)))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.ln_attn.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln_ffn.parameters());
        p.extend(self.ffn.parameters());
        p
    }
}

struct DecoderLayer {
    ln_self: LayerNorm,
    self_attn: MultiHeadAttention,
    ln_cross: LayerNorm,
    cross_attn: MultiHeadAttention,
    ln_ffn: LayerNorm,
    ffn: FeedForward,
}

impl DecoderLayer {
    fn new(cfg: &ModelConfig) -> Self {
        Self {
            ln_self: LayerNorm::new(cfg.d_model, cfg.ln_eps),
            self_attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads),
            ln_cross: LayerNorm::new(cfg.d_model, cfg.ln_eps),
            cross_attn: MultiHeadAttention::new(cfg.d_model, cfg.n_heads),
            ln_ffn: LayerNorm::new(cfg.d_model, cfg.ln_eps),
            ffn: FeedForward::new(cfg.d_model, cfg.d_ff),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        memory: &Tensor,
        self_mask: Option<&ArrayD<f32>>,
        cross_mask: Option<&ArrayD<f32>>,
    ) -> Tensor {
        let x = x.add(
            &self
                .self_attn
                .self_attention(&self.ln_self.forward(x), self_mask),
        );
        // Cross-attention: query из декодера, key/value из памяти энкодера.
        let q = self.ln_cross.forward(&x);
        let x = x.add(&self.cross_attn.forward(&q, memory, memory, cross_mask));
        x.add(&self.ffn.forward(&self.ln_ffn.forward(&x)))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.ln_self.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.ln_cross.parameters());
        p.extend(self.cross_attn.parameters());
        p.extend(self.ln_ffn.parameters());
        p.extend(self.ffn.parameters());
        p
    }
}

pub(crate) struct TransformerCore {
    enc_layers: Vec<EncoderLayer>,
    enc_norm: LayerNorm,
    dec_layers: Vec<DecoderLayer>,
    dec_norm: LayerNorm,
}

impl TransformerCore {
    pub(crate) fn new(cfg: &ModelConfig) -> Self {
        Self {
            enc_layers: (0..cfg.n_enc_layers)
                .map(|_| EncoderLayer::new(cfg))
                .collect(),
            enc_norm: LayerNorm::new(cfg.d_model, cfg.ln_eps),
            dec_layers: (0..cfg.n_dec_layers)
                .map(|_| DecoderLayer::new(cfg))
                .collect(),
            dec_norm: LayerNorm::new(cfg.d_model, cfg.ln_eps),
        }
    }

    /// Кодирует вход `[B, src_len, d_model]` в память `[B, src_len, d_model]`.
    pub(crate) fn encode(&self, src: &Tensor, src_mask: Option<&ArrayD<f32>>) -> Tensor {
        let mut x = src.clone();
        for layer in &self.enc_layers {
            x = layer.forward(&x, src_mask);
        }
        self.enc_norm.forward(&x)
    }

    /// Декодирует `[B, tgt_len, d_model]` с памятью энкодера в скрытое состояние.
    /// `self_mask` — causal для авторегрессии или None для параллельных запросов.
    pub(crate) fn decode(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        self_mask: Option<&ArrayD<f32>>,
        cross_mask: Option<&ArrayD<f32>>,
    ) -> Tensor {
        let mut x = tgt.clone();
        for layer in &self.dec_layers {
            x = layer.forward(&x, memory, self_mask, cross_mask);
        }
        self.dec_norm.forward(&x)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for layer in &self.enc_layers {
            p.extend(layer.parameters());
        }
        p.extend(self.enc_norm.parameters());
        for layer in &self.dec_layers {
            p.extend(layer.parameters());
        }
        p.extend(self.dec_norm.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::NumericInputEncoder;
    use crate::heads::RegressionHead;
    use crate::loss::cross_entropy;
    use crate::nn::embedding::{sinusoidal_positions, Embedding};
    use crate::nn::linear::Linear;
    use crate::optim::Adam;
    use ndarray::ArrayD;

    /// Главная проверка M4: модель должна выучить (overfit) один фиксированный
    /// батч до loss ≈ 0. Это доказывает, что весь граф связан и градиенты текут:
    /// embedding -> encoder -> decoder(cross-attn) -> head -> cross_entropy -> Adam.
    #[test]
    fn overfit_single_batch() {
        let cfg = ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_enc_layers: 1,
            n_dec_layers: 1,
            d_ff: 32,
            ln_eps: 1e-5,
        };
        let vocab = 7;
        let src_len = 4;
        let tgt_len = 3;

        let core = TransformerCore::new(&cfg);
        let src_emb = Embedding::new(vocab, cfg.d_model);
        let tgt_emb = Embedding::new(vocab, cfg.d_model);
        let head = Linear::new(cfg.d_model, vocab);

        // Фиксированный батч (B=1).
        let src_ids = ArrayD::from_shape_vec(IxDyn(&[1, src_len]), vec![1, 3, 2, 5]).unwrap();
        let tgt_ids = ArrayD::from_shape_vec(IxDyn(&[1, tgt_len]), vec![0, 4, 6]).unwrap();
        let labels = ArrayD::from_shape_vec(IxDyn(&[1, tgt_len]), vec![4, 6, 2]).unwrap();

        let src_pos = sinusoidal_positions(src_len, cfg.d_model);
        let tgt_pos = sinusoidal_positions(tgt_len, cfg.d_model);
        let self_mask = causal_mask(tgt_len);

        let mut params = core.parameters();
        params.extend(src_emb.parameters());
        params.extend(tgt_emb.parameters());
        params.extend(head.parameters());
        let mut opt = Adam::new(params, 0.01);

        let forward = || {
            let src = src_emb.forward(&src_ids).add(&src_pos);
            let memory = core.encode(&src, None);
            let tgt = tgt_emb.forward(&tgt_ids).add(&tgt_pos);
            let hidden = core.decode(&tgt, &memory, Some(&self_mask), None);
            let logits = head.forward(&hidden);
            cross_entropy(&logits, &labels)
        };

        let first = forward().item();
        for _ in 0..300 {
            opt.zero_grad();
            let loss = forward();
            loss.backward();
            opt.step();
        }
        let last = forward().item();

        assert!(
            last < 0.02,
            "overfit не удался: loss {first:.4} -> {last:.4} (ожидали ≈ 0)"
        );
    }

    /// Главная проверка M5 (surrogate-сценарий). Query-токены ФИКСИРОВАНЫ и
    /// одинаковы для всех примеров батча, а target зависит ТОЛЬКО от src.
    /// Значит единственный канал информации о примере — cross-attention к
    /// памяти энкодера. Если модель overfit-ит батч и градиент доходит до
    /// value_proj входа, то encoder и cross-attention реально используются.
    #[test]
    fn surrogate_uses_cross_attention() {
        let cfg = ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_enc_layers: 1,
            n_dec_layers: 1,
            d_ff: 32,
            ln_eps: 1e-5,
        };
        let batch = 8;
        let n_feat = 2;
        let n_out = 2;

        let core = TransformerCore::new(&cfg);
        let input_enc = NumericInputEncoder::new(n_feat, cfg.d_model);
        let query_emb = Embedding::new(n_out, cfg.d_model); // эмбеддинги выходных слотов
        let head = RegressionHead::new(cfg.d_model, 1);

        // Чёрный ящик: out0 = x0 + x1, out1 = x0 * x1. Зависит только от src.
        let mut vals = Vec::with_capacity(batch * n_feat);
        let mut tgts = Vec::with_capacity(batch * n_out);
        for i in 0..batch {
            let x0 = (i as f32) / batch as f32 - 0.5;
            let x1 = 0.4 - (i as f32) * 0.1;
            vals.push(x0);
            vals.push(x1);
            tgts.push(x0 + x1);
            tgts.push(x0 * x1);
        }
        let values =
            Tensor::constant(ArrayD::from_shape_vec(IxDyn(&[batch, n_feat]), vals).unwrap());
        let target =
            Tensor::constant(ArrayD::from_shape_vec(IxDyn(&[batch, n_out, 1]), tgts).unwrap());

        // Фиксированные query-id [0, 1], одинаковые для каждого примера батча.
        let mut qids = Vec::with_capacity(batch * n_out);
        for _ in 0..batch {
            for j in 0..n_out {
                qids.push(j);
            }
        }
        let query_ids = ArrayD::from_shape_vec(IxDyn(&[batch, n_out]), qids).unwrap();

        let mut params = core.parameters();
        params.extend(input_enc.parameters());
        params.extend(query_emb.parameters());
        params.extend(head.parameters());
        let mut opt = Adam::new(params, 0.01);

        let forward = || {
            let src = input_enc.forward(&values);
            let memory = core.encode(&src, None);
            let queries = query_emb.forward(&query_ids); // [B, n_out, d_model]
            let hidden = core.decode(&queries, &memory, None, None); // self_mask=None: запросы независимы
            head.mse(&hidden, &target)
        };

        let first = forward().item();
        for _ in 0..800 {
            opt.zero_grad();
            let loss = forward();
            loss.backward();
            opt.step();
        }
        let last = forward().item();
        assert!(
            last < 0.02,
            "surrogate overfit не удался: MSE {first:.4} -> {last:.4}"
        );

        // Кодировщик значений достижим ТОЛЬКО через encoder -> cross-attention.
        // Ненулевой градиент здесь доказывает, что этот путь используется.
        opt.zero_grad();
        forward().backward();
        let vp_norm: f32 = input_enc
            .value_parameters()
            .iter()
            .flat_map(|t| t.grad().into_iter().map(|g| g * g).collect::<Vec<_>>())
            .sum::<f32>()
            .sqrt();
        assert!(
            vp_norm > 1e-6,
            "градиент value-encoder нулевой — cross-attention не используется"
        );
    }
}
