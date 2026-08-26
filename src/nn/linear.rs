use crate::init::rand_uniform;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

pub(crate) struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub(crate) fn new(in_features: usize, out_features: usize) -> Self {
        let limit = (6.0_f32 / (in_features + out_features) as f32).sqrt();
        let weight = rand_uniform(&[in_features, out_features], -limit, limit);
        let bias = ArrayD::zeros(IxDyn(&[1, out_features]));
        Self::from_tensors(Tensor::new(weight), Tensor::new(bias))
    }

    pub(crate) fn from_tensors(weight: Tensor, bias: Tensor) -> Self {
        let w_shape = weight.shape();
        let b_shape = bias.shape();
        assert_eq!(w_shape.len(), 2, "Linear weight должен быть [in, out]");
        assert_eq!(
            b_shape,
            vec![1, w_shape[1]],
            "Linear bias должен быть [1, out]"
        );
        Self {
            weight,
            bias,
            in_features: w_shape[0],
            out_features: w_shape[1],
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert!(
            !shape.is_empty(),
            "Linear input должен иметь последнюю ось in_features"
        );
        assert_eq!(
            *shape.last().unwrap(),
            self.in_features,
            "Linear input последняя ось != in_features"
        );

        if shape.len() == 2 {
            return x.matmul(&self.weight).add(&self.bias);
        }

        let rows = x.data().len() / self.in_features;
        let mut out_shape = shape;
        *out_shape.last_mut().unwrap() = self.out_features;
        x.reshape(&[rows, self.in_features])
            .matmul(&self.weight)
            .add(&self.bias)
            .reshape(&out_shape)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check, rand_tensor};

    #[test]
    fn check_linear_2d() {
        let x = rand_tensor(&[3, 4]);
        let w = rand_tensor(&[4, 5]);
        let b = rand_tensor(&[1, 5]);
        let layer = Linear::from_tensors(w.clone(), b.clone());
        let inputs = vec![x.clone(), w, b];
        grad_check(&inputs, |_| layer.forward(&x).mean());
    }

    #[test]
    fn check_linear_3d_folding() {
        let x = rand_tensor(&[2, 3, 4]);
        let w = rand_tensor(&[4, 5]);
        let b = rand_tensor(&[1, 5]);
        let layer = Linear::from_tensors(w.clone(), b.clone());
        let inputs = vec![x.clone(), w, b];
        grad_check(&inputs, |_| layer.forward(&x).mean());
    }
}
