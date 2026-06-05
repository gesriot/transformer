use crate::init::rand_uniform;
use crate::tensor::{BackwardFn, Tensor};
use ndarray::{Array2, ArrayD, Ix2, IxDyn};

pub struct Embedding {
    pub weight: Tensor,
    vocab_size: usize,
    dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let limit = (3.0_f32 / dim as f32).sqrt();
        let weight = rand_uniform(&[vocab_size, dim], -limit, limit);
        Self::from_weight(Tensor::new(weight))
    }

    pub fn from_weight(weight: Tensor) -> Self {
        let shape = weight.shape();
        assert_eq!(shape.len(), 2, "Embedding weight должен быть [vocab, dim]");
        Self {
            weight,
            vocab_size: shape[0],
            dim: shape[1],
        }
    }

    pub fn forward(&self, ids: &ArrayD<usize>) -> Tensor {
        let weight = self
            .weight
            .data()
            .into_dimensionality::<Ix2>()
            .expect("Embedding weight должен быть [vocab, dim]");
        let mut out_shape = ids.shape().to_vec();
        out_shape.push(self.dim);
        let rows = ids.len();
        let mut out = Array2::<f32>::zeros((rows, self.dim));
        let flat_ids: Vec<usize> = ids.iter().copied().collect();

        for (row, &id) in flat_ids.iter().enumerate() {
            assert!(id < self.vocab_size, "Embedding id вне vocab");
            for d in 0..self.dim {
                out[[row, d]] = weight[[id, d]];
            }
        }

        let out_dyn = ArrayD::from_shape_vec(IxDyn(&out_shape), out.iter().copied().collect())
            .expect("Embedding output shape согласована");
        let weight_t = self.weight.clone();
        let vocab_size = self.vocab_size;
        let dim = self.dim;
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let g2 = Array2::from_shape_vec((rows, dim), g.iter().copied().collect())
                .expect("Embedding backward shape согласована");
            let mut dweight = Array2::<f32>::zeros((vocab_size, dim));
            for (row, &id) in flat_ids.iter().enumerate() {
                for d in 0..dim {
                    dweight[[id, d]] += g2[[row, d]];
                }
            }
            weight_t.add_grad(&dweight.into_dyn());
        });
        Tensor::from_op(out_dyn, vec![self.weight.clone()], backward)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

pub fn sinusoidal_positions(seq_len: usize, dim: usize) -> Tensor {
    let mut pe = ArrayD::<f32>::zeros(IxDyn(&[1, seq_len, dim]));
    for pos in 0..seq_len {
        for i in (0..dim).step_by(2) {
            let denom = 10000_f32.powf(i as f32 / dim as f32);
            pe[IxDyn(&[0, pos, i])] = (pos as f32 / denom).sin();
            if i + 1 < dim {
                pe[IxDyn(&[0, pos, i + 1])] = (pos as f32 / denom).cos();
            }
        }
    }
    Tensor::constant(pe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check, rand_tensor};

    #[test]
    fn check_embedding_lookup() {
        let weight = rand_tensor(&[5, 3]);
        let embedding = Embedding::from_weight(weight.clone());
        let ids = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0, 2, 2, 4, 1, 0]).unwrap();
        grad_check(&[weight], |_| embedding.forward(&ids).mean());
    }

    #[test]
    fn sinusoidal_positions_are_constant() {
        let pe = sinusoidal_positions(4, 6);
        assert_eq!(pe.shape(), vec![1, 4, 6]);
        assert!(!pe.requires_grad());
    }
}
