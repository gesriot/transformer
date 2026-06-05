use crate::nn::linear::Linear;
use crate::tensor::Tensor;

pub struct FeedForward {
    pub up: Linear,
    pub down: Linear,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            up: Linear::new(d_model, d_ff),
            down: Linear::new(d_ff, d_model),
        }
    }

    pub fn from_layers(up: Linear, down: Linear) -> Self {
        Self { up, down }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.down.forward(&self.up.forward(x).gelu())
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.up.parameters();
        params.extend(self.down.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check_with_tol, rand_tensor};

    #[test]
    fn check_ffn_layer() {
        let x = rand_tensor(&[2, 3, 4]);
        let up = Linear::from_tensors(rand_tensor(&[4, 5]), rand_tensor(&[1, 5]));
        let down = Linear::from_tensors(rand_tensor(&[5, 4]), rand_tensor(&[1, 4]));
        let ffn = FeedForward::from_layers(up, down);
        let mut inputs = vec![x.clone()];
        inputs.extend(ffn.parameters());

        grad_check_with_tol(&inputs, 1e-3, 2e-2, |_| ffn.forward(&x).mean());
    }
}
