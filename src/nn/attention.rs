use crate::nn::linear::Linear;
use crate::ops::{merge_heads, scaled_dot_product_attention, split_heads};
use crate::tensor::Tensor;
use ndarray::ArrayD;

pub(crate) struct MultiHeadAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    n_heads: usize,
    d_model: usize,
}

impl MultiHeadAttention {
    pub(crate) fn new(d_model: usize, n_heads: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model должен делиться на n_heads");
        Self {
            w_q: Linear::new(d_model, d_model),
            w_k: Linear::new(d_model, d_model),
            w_v: Linear::new(d_model, d_model),
            w_o: Linear::new(d_model, d_model),
            n_heads,
            d_model,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_layers(
        w_q: Linear,
        w_k: Linear,
        w_v: Linear,
        w_o: Linear,
        n_heads: usize,
    ) -> Self {
        let d_model = w_q.weight.shape()[0];
        assert_eq!(d_model % n_heads, 0, "d_model должен делиться на n_heads");
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            n_heads,
            d_model,
        }
    }

    pub(crate) fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&ArrayD<f32>>,
    ) -> Tensor {
        assert_eq!(
            *query.shape().last().unwrap(),
            self.d_model,
            "query последняя ось != d_model"
        );
        let q = split_heads(&self.w_q.forward(query), self.n_heads);
        let k = split_heads(&self.w_k.forward(key), self.n_heads);
        let v = split_heads(&self.w_v.forward(value), self.n_heads);
        let context = scaled_dot_product_attention(&q, &k, &v, mask);
        self.w_o.forward(&merge_heads(&context))
    }

    pub(crate) fn self_attention(&self, x: &Tensor, mask: Option<&ArrayD<f32>>) -> Tensor {
        self.forward(x, x, x, mask)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.w_q.parameters();
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check_with_tol, rand_tensor};

    fn tiny_mha() -> MultiHeadAttention {
        MultiHeadAttention::from_layers(
            Linear::from_tensors(rand_tensor(&[4, 4]), rand_tensor(&[1, 4])),
            Linear::from_tensors(rand_tensor(&[4, 4]), rand_tensor(&[1, 4])),
            Linear::from_tensors(rand_tensor(&[4, 4]), rand_tensor(&[1, 4])),
            Linear::from_tensors(rand_tensor(&[4, 4]), rand_tensor(&[1, 4])),
            2,
        )
    }

    #[test]
    fn check_multi_head_self_attention() {
        let x = rand_tensor(&[1, 3, 4]);
        let mha = tiny_mha();
        let mut inputs = vec![x.clone()];
        inputs.extend(mha.parameters());

        grad_check_with_tol(&inputs, 1e-3, 3e-2, |_| mha.self_attention(&x, None).mean());
    }

    #[test]
    fn check_multi_head_cross_attention() {
        let query = rand_tensor(&[1, 2, 4]);
        let memory = rand_tensor(&[1, 3, 4]);
        let mha = tiny_mha();
        let mut inputs = vec![query.clone(), memory.clone()];
        inputs.extend(mha.parameters());

        grad_check_with_tol(&inputs, 1e-3, 3e-2, |_| {
            mha.forward(&query, &memory, &memory, None).mean()
        });
    }
}
