//! Головы задач: из скрытого состояния декодера в Y_hat + loss (Plan.md §1).
//!
//! - `RegressionHead` — числовой выход (surrogate-модель).
//! - `CharHead` — логиты по словарю (char-LM).
//!
//! Функцию потерь считает вызывающая модель; здесь она есть только под тестом,
//! где по ней проверяются градиенты голов.

#[cfg(test)]
use crate::loss::cross_entropy;
use crate::nn::linear::Linear;
use crate::tensor::Tensor;
#[cfg(test)]
use ndarray::ArrayD;

/// Регрессионная голова: проецирует скрытое состояние каждого query-токена в
/// числовой выход. Обычно `out_dim = 1` (один скаляр на запрос).
pub(crate) struct RegressionHead {
    pub proj: Linear,
}

impl RegressionHead {
    pub(crate) fn new(d_model: usize, out_dim: usize) -> Self {
        Self {
            proj: Linear::new(d_model, out_dim),
        }
    }

    /// `hidden` — `[B, n_queries, d_model]` -> `[B, n_queries, out_dim]`.
    pub(crate) fn forward(&self, hidden: &Tensor) -> Tensor {
        self.proj.forward(hidden)
    }

    #[cfg(test)]
    pub(crate) fn mse(&self, hidden: &Tensor, target: &Tensor) -> Tensor {
        self.forward(hidden).mse_loss(target)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        self.proj.parameters()
    }
}

/// Голова char-LM: проецирует скрытое состояние в логиты по словарю.
#[cfg(any(feature = "demo", test))]
pub(crate) struct CharHead {
    pub proj: Linear,
}

#[cfg(any(feature = "demo", test))]
impl CharHead {
    pub(crate) fn new(d_model: usize, vocab_size: usize) -> Self {
        Self {
            proj: Linear::new(d_model, vocab_size),
        }
    }

    /// `hidden` -> логиты `[.., vocab]`.
    pub(crate) fn forward(&self, hidden: &Tensor) -> Tensor {
        self.proj.forward(hidden)
    }

    #[cfg(test)]
    pub(crate) fn loss(&self, hidden: &Tensor, targets: &ArrayD<usize>) -> Tensor {
        cross_entropy(&self.forward(hidden), targets)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        self.proj.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check, rand_tensor};
    use ndarray::IxDyn;

    #[test]
    fn regression_head_mse_grad() {
        let head = RegressionHead::new(6, 1);
        let hidden = rand_tensor(&[2, 3, 6]);
        let target = Tensor::constant(rand_tensor(&[2, 3, 1]).data());
        let mut inputs = vec![hidden.clone()];
        inputs.extend(head.parameters());
        grad_check(&inputs, |t| head.mse(&t[0], &target));
    }

    #[test]
    fn char_head_cross_entropy_grad() {
        let head = CharHead::new(6, 5);
        let hidden = rand_tensor(&[2, 3, 6]);
        let targets = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0, 1, 2, 3, 4, 0]).unwrap();
        let mut inputs = vec![hidden.clone()];
        inputs.extend(head.parameters());
        grad_check(&inputs, |t| head.loss(&t[0], &targets));
    }
}
