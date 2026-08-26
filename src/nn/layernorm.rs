use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

pub(crate) struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub(crate) fn new(dim: usize, eps: f32) -> Self {
        let gamma = ArrayD::from_elem(IxDyn(&[dim]), 1.0);
        let beta = ArrayD::zeros(IxDyn(&[dim]));
        Self::from_tensors(Tensor::new(gamma), Tensor::new(beta), eps)
    }

    pub(crate) fn from_tensors(gamma: Tensor, beta: Tensor, eps: f32) -> Self {
        assert_eq!(
            gamma.shape(),
            beta.shape(),
            "LayerNorm gamma/beta формы должны совпадать"
        );
        Self { gamma, beta, eps }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm_last_dim(&self.gamma, &self.beta, self.eps)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check_with_tol, rand_tensor};

    #[test]
    fn check_layernorm_layer() {
        let x = rand_tensor(&[2, 3, 4]);
        let gamma = rand_tensor(&[4]);
        let beta = rand_tensor(&[4]);
        let layer = LayerNorm::from_tensors(gamma.clone(), beta.clone(), 1e-5);
        grad_check_with_tol(&[x.clone(), gamma, beta], 1e-3, 2e-2, |_| {
            layer.forward(&x).mean()
        });
    }
}
