//! Char-LM seq2seq модель (Plan.md §2): авторегрессионная генерация текста.
//!
//! Encoder кодирует контекст, decoder с causal-маской авторегрессионно
//! предсказывает продолжение (teacher forcing при обучении, сэмплирование при
//! генерации). Embedding общий для контекста и продолжения (один словарь).

use crate::config::ModelConfig;
use crate::core::{causal_mask, TransformerCore};
use crate::encoders::TokenInputEncoder;
use crate::heads::CharHead;
use crate::loss::cross_entropy;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

pub struct TextModel {
    embed: TokenInputEncoder,
    core: TransformerCore,
    head: CharHead,
    vocab_size: usize,
}

impl TextModel {
    pub fn new(cfg: &ModelConfig, vocab_size: usize) -> Self {
        Self {
            embed: TokenInputEncoder::new(vocab_size, cfg.d_model),
            core: TransformerCore::new(cfg),
            head: CharHead::new(cfg.d_model, vocab_size),
            vocab_size,
        }
    }

    /// `src` — `[B, ctx]`, `dec` — `[B, tgt]`. Логиты `[B, tgt, vocab]`.
    pub fn forward(&self, src: &ArrayD<usize>, dec: &ArrayD<usize>) -> Tensor {
        let dec_len = dec.shape()[1];
        let memory = self.core.encode(&self.embed.forward(src), None);
        let dec_emb = self.embed.forward(dec);
        let mask = causal_mask(dec_len);
        let hidden = self.core.decode(&dec_emb, &memory, Some(&mask), None);
        self.head.forward(&hidden)
    }

    pub fn loss(&self, src: &ArrayD<usize>, dec: &ArrayD<usize>, labels: &ArrayD<usize>) -> Tensor {
        cross_entropy(&self.forward(src, dec), labels)
    }

    /// Кодирует контекст один раз (память переиспользуется при генерации).
    pub fn encode_src(&self, src: &[usize]) -> Tensor {
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, src.len()]), src.to_vec()).unwrap();
        self.core.encode(&self.embed.forward(&arr), None)
    }

    /// Логиты следующего символа для текущей последовательности декодера.
    pub fn next_logits(&self, dec: &[usize], memory: &Tensor) -> Vec<f32> {
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, dec.len()]), dec.to_vec()).unwrap();
        let dec_emb = self.embed.forward(&arr);
        let mask = causal_mask(dec.len());
        let hidden = self.core.decode(&dec_emb, memory, Some(&mask), None);
        let logits = self.head.forward(&hidden).data(); // [1, dec_len, vocab]
        let last = dec.len() - 1;
        (0..self.vocab_size)
            .map(|v| logits[IxDyn(&[0, last, v])])
            .collect()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.embed.parameters();
        p.extend(self.core.parameters());
        p.extend(self.head.parameters());
        p
    }
}
