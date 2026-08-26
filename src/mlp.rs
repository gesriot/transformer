//! MLP-baseline для численной регрессии (roadmap шаг 3).
//!
//! Честный baseline к трансформеру на фиксированной схеме входов: значения
//! подаются как один вектор `[B, F]` (без токенизации), затем
//! `Linear -> GELU -> ... -> Linear`. Если MLP бьёт трансформер при меньшем
//! числе параметров — сигнал тратить бюджет на данные/кодирование значений,
//! а не на слои. Оставляется как постоянный регрессионный сторож.

use crate::nn::linear::Linear;
use crate::tensor::Tensor;

pub struct MlpBaseline {
    input: Linear,
    hidden: Vec<Linear>,
    output: Linear,
}

impl MlpBaseline {
    /// `n_layers` — число GELU-слоёв (вход + скрытые), минимум 1.
    pub fn new(n_inputs: usize, width: usize, n_layers: usize, n_outputs: usize) -> Self {
        assert!(n_layers >= 1, "MLP: нужен хотя бы один слой");
        Self {
            input: Linear::new(n_inputs, width),
            hidden: (0..n_layers - 1)
                .map(|_| Linear::new(width, width))
                .collect(),
            output: Linear::new(width, n_outputs),
        }
    }

    /// `values` — `[B, F]` (нормализованные) -> `[B, O]`.
    pub(crate) fn predict(&self, values: &Tensor) -> Tensor {
        let mut x = self.input.forward(values).gelu();
        for h in &self.hidden {
            x = h.forward(&x).gelu();
        }
        self.output.forward(&x)
    }

    pub(crate) fn loss(&self, values: &Tensor, targets: &Tensor) -> Tensor {
        self.predict(values).mse_loss(targets)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.input.parameters();
        for h in &self.hidden {
            p.extend(h.parameters());
        }
        p.extend(self.output.parameters());
        p
    }

    pub(crate) fn interface_dims(&self) -> (usize, usize) {
        (self.input.weight.shape()[0], self.output.weight.shape()[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::data::Normalizer;
    use crate::optim::Adam;

    #[test]
    fn mlp_overfits_sum() {
        let data = blackbox::sum().generate(64, 0);
        let in_norm = Normalizer::fit(&data.inputs, &Normalizer::all_continuous(2));
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let x = Tensor::constant(in_norm.transform(&data.inputs).into_dyn());
        let y = Tensor::constant(out_norm.transform(&data.outputs).into_dyn());

        let model = MlpBaseline::new(2, 32, 2, 1);
        let mut opt = Adam::new(model.parameters(), 3e-3);
        let first = model.loss(&x, &y).item();
        for _ in 0..200 {
            opt.zero_grad();
            let loss = model.loss(&x, &y);
            loss.backward();
            opt.step();
        }
        let last = model.loss(&x, &y).item();
        assert!(
            last < first * 0.1,
            "MLP не выучил sum: {first:.4} -> {last:.4}"
        );
    }
}
