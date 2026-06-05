//! Surrogate-модель: единая обёртка над ядром для аппроксимации чёрного ящика
//! (Plan.md §1, §3). Вход — числовые признаки, выход — числовые величины.
//!
//! Поток: NumericInputEncoder -> encoder -> [фикс. query-токены] -> decoder
//! (cross-attention к памяти, БЕЗ causal-маски: запросы независимы) ->
//! RegressionHead. `predict(X) -> Y` за один проход, без авторегрессии.

use crate::config::ModelConfig;
use crate::core::TransformerCore;
use crate::encoders::{FeatureSpec, NumericInputEncoder, ValueEncoderConfig};
use crate::heads::RegressionHead;
use crate::nn::embedding::Embedding;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

pub struct SurrogateModel {
    input_enc: NumericInputEncoder,
    query_emb: Embedding,
    core: TransformerCore,
    head: RegressionHead,
    num_outputs: usize,
}

impl SurrogateModel {
    /// Кодировщик значений по умолчанию (Linear) — обратная совместимость.
    pub fn new(cfg: &ModelConfig, input_specs: &[FeatureSpec], num_outputs: usize) -> Self {
        Self::with_value_encoder(
            cfg,
            &ValueEncoderConfig::default(),
            input_specs,
            num_outputs,
        )
    }

    pub fn with_value_encoder(
        cfg: &ModelConfig,
        value_cfg: &ValueEncoderConfig,
        input_specs: &[FeatureSpec],
        num_outputs: usize,
    ) -> Self {
        assert!(num_outputs > 0, "нужен хотя бы один выход");
        Self {
            input_enc: NumericInputEncoder::with_specs(input_specs, cfg.d_model, value_cfg),
            query_emb: Embedding::new(num_outputs, cfg.d_model),
            core: TransformerCore::new(cfg),
            head: RegressionHead::new(cfg.d_model, 1),
            num_outputs,
        }
    }

    /// `values` — `[B, F]` нормализованных входов. Возвращает `[B, O]` предсказаний
    /// (в нормализованном пространстве выходов).
    pub fn predict(&self, values: &Tensor) -> Tensor {
        let batch = values.shape()[0];
        let src = self.input_enc.forward(values);
        let memory = self.core.encode(&src, None);

        let queries = self.query_emb.forward(&self.query_ids(batch));
        // self_mask=None: выходные запросы независимы (не авторегрессия).
        let hidden = self.core.decode(&queries, &memory, None, None);
        self.head
            .forward(&hidden)
            .reshape(&[batch, self.num_outputs]) // [B, O, 1] -> [B, O]
    }

    pub fn loss(&self, values: &Tensor, targets: &Tensor) -> Tensor {
        self.predict(values).mse_loss(targets)
    }

    /// Фиксированные id выходных слотов: каждая строка = 0..O (одинаково по батчу).
    fn query_ids(&self, batch: usize) -> ArrayD<usize> {
        let mut ids = Vec::with_capacity(batch * self.num_outputs);
        for _ in 0..batch {
            ids.extend(0..self.num_outputs);
        }
        ArrayD::from_shape_vec(IxDyn(&[batch, self.num_outputs]), ids).unwrap()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.input_enc.parameters();
        p.extend(self.query_emb.parameters());
        p.extend(self.core.parameters());
        p.extend(self.head.parameters());
        p
    }
}
