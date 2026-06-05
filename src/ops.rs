use crate::tensor::{BackwardFn, Tensor};
use ndarray::{Array2, Array3, Array4, ArrayD, Ix3, Ix4, IxDyn};

pub(crate) fn flatten_last_dim(a: &ArrayD<f32>) -> (Array2<f32>, Vec<usize>) {
    let shape = a.shape().to_vec();
    let cols = *shape.last().expect("тензор должен иметь хотя бы одну ось");
    assert!(cols > 0, "последняя ось не должна быть пустой");
    let rows = a.len() / cols;
    let flat = Array2::from_shape_vec((rows, cols), a.iter().copied().collect())
        .expect("flatten_last_dim: форма согласована");
    (flat, shape)
}

pub(crate) fn from_flat_last_dim(flat: &Array2<f32>, shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(shape), flat.iter().copied().collect())
        .expect("from_flat_last_dim: форма согласована")
}

impl Tensor {
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let a = self.data();
        let old_shape = a.shape().to_vec();
        let old_len = a.len();
        let new_len: usize = shape.iter().product();
        assert_eq!(old_len, new_len, "reshape должен сохранять число элементов");

        let out = a
            .into_shape_with_order(IxDyn(shape))
            .expect("reshape: форма согласована");
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let grad = g
                .clone()
                .into_shape_with_order(IxDyn(&old_shape))
                .expect("reshape backward: форма согласована");
            lhs.add_grad(&grad);
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    pub fn gelu(&self) -> Tensor {
        let a = self.data();
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        let k = 0.044_715_f32;
        let tanh_arg = a.mapv(|x| c * (x + k * x.powi(3)));
        let tanh_val = tanh_arg.mapv(f32::tanh);
        let out = &a * &tanh_val.mapv(|t| 0.5 * (1.0 + t));

        let derivative = a.mapv(|x| {
            let u = c * (x + k * x.powi(3));
            let t = u.tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * k * x * x)
        });
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&(g * &derivative));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    /// Поэлементный синус (для Fourier-фич). d/dx sin = cos.
    pub fn sin(&self) -> Tensor {
        let a = self.data();
        let out = a.mapv(f32::sin);
        let cos = a.mapv(f32::cos);
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&(g * &cos));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    /// Поэлементный косинус. d/dx cos = -sin.
    pub fn cos(&self) -> Tensor {
        let a = self.data();
        let out = a.mapv(f32::cos);
        let neg_sin = a.mapv(|x| -x.sin());
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&(g * &neg_sin));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    pub fn softmax_last_dim(&self) -> Tensor {
        let (a, shape) = flatten_last_dim(&self.data());
        let (rows, cols) = a.dim();
        let mut probs = Array2::<f32>::zeros((rows, cols));

        for r in 0..rows {
            let mut max_v = f32::NEG_INFINITY;
            for c in 0..cols {
                max_v = max_v.max(a[[r, c]]);
            }
            let mut denom = 0.0;
            for c in 0..cols {
                let e = (a[[r, c]] - max_v).exp();
                probs[[r, c]] = e;
                denom += e;
            }
            for c in 0..cols {
                probs[[r, c]] /= denom;
            }
        }

        let out = from_flat_last_dim(&probs, &shape);
        let lhs = self.clone();
        let probs_for_backward = probs;
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let (g2, _) = flatten_last_dim(g);
            let (rows, cols) = g2.dim();
            let mut dx = Array2::<f32>::zeros((rows, cols));
            for r in 0..rows {
                let mut dot = 0.0;
                for c in 0..cols {
                    dot += g2[[r, c]] * probs_for_backward[[r, c]];
                }
                for c in 0..cols {
                    dx[[r, c]] = probs_for_backward[[r, c]] * (g2[[r, c]] - dot);
                }
            }
            lhs.add_grad(&from_flat_last_dim(&dx, &shape));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    pub fn layer_norm_last_dim(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let (x, shape) = flatten_last_dim(&self.data());
        let gamma_data = gamma.data();
        let beta_data = beta.data();
        let gamma_shape = gamma_data.shape().to_vec();
        let beta_shape = beta_data.shape().to_vec();
        let gamma_vec: Vec<f32> = gamma_data.iter().copied().collect();
        let beta_vec: Vec<f32> = beta_data.iter().copied().collect();
        let (rows, cols) = x.dim();
        assert_eq!(
            gamma_vec.len(),
            cols,
            "gamma должен совпадать с последней осью"
        );
        assert_eq!(
            beta_vec.len(),
            cols,
            "beta должен совпадать с последней осью"
        );

        let mut xhat = Array2::<f32>::zeros((rows, cols));
        let mut inv_std = vec![0.0; rows];
        let mut out = Array2::<f32>::zeros((rows, cols));

        for r in 0..rows {
            let mut mean = 0.0;
            for c in 0..cols {
                mean += x[[r, c]];
            }
            mean /= cols as f32;

            let mut var = 0.0;
            for c in 0..cols {
                let centered = x[[r, c]] - mean;
                var += centered * centered;
            }
            var /= cols as f32;
            inv_std[r] = 1.0 / (var + eps).sqrt();

            for c in 0..cols {
                xhat[[r, c]] = (x[[r, c]] - mean) * inv_std[r];
                out[[r, c]] = xhat[[r, c]] * gamma_vec[c] + beta_vec[c];
            }
        }

        let out_dyn = from_flat_last_dim(&out, &shape);
        let lhs = self.clone();
        let gamma_t = gamma.clone();
        let beta_t = beta.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let (g2, _) = flatten_last_dim(g);
            let (rows, cols) = g2.dim();
            let mut dx = Array2::<f32>::zeros((rows, cols));
            let mut dgamma = vec![0.0; cols];
            let mut dbeta = vec![0.0; cols];

            for r in 0..rows {
                let mut mean_dy_gamma = 0.0;
                let mut mean_dy_gamma_xhat = 0.0;
                for c in 0..cols {
                    let dy_gamma = g2[[r, c]] * gamma_vec[c];
                    mean_dy_gamma += dy_gamma;
                    mean_dy_gamma_xhat += dy_gamma * xhat[[r, c]];
                    dgamma[c] += g2[[r, c]] * xhat[[r, c]];
                    dbeta[c] += g2[[r, c]];
                }
                mean_dy_gamma /= cols as f32;
                mean_dy_gamma_xhat /= cols as f32;

                for c in 0..cols {
                    let dy_gamma = g2[[r, c]] * gamma_vec[c];
                    dx[[r, c]] =
                        inv_std[r] * (dy_gamma - mean_dy_gamma - xhat[[r, c]] * mean_dy_gamma_xhat);
                }
            }

            lhs.add_grad(&from_flat_last_dim(&dx, &shape));
            gamma_t.add_grad(
                &ArrayD::from_shape_vec(IxDyn(&gamma_shape), dgamma)
                    .expect("layernorm dgamma: форма согласована"),
            );
            beta_t.add_grad(
                &ArrayD::from_shape_vec(IxDyn(&beta_shape), dbeta)
                    .expect("layernorm dbeta: форма согласована"),
            );
        });
        Tensor::from_op(
            out_dyn,
            vec![self.clone(), gamma.clone(), beta.clone()],
            backward,
        )
    }
}

pub fn split_heads(x: &Tensor, n_heads: usize) -> Tensor {
    let x_data = x
        .data()
        .into_dimensionality::<Ix3>()
        .expect("split_heads ожидает [batch, seq, d_model]");
    let (batch, seq, d_model) = x_data.dim();
    assert_eq!(d_model % n_heads, 0, "d_model должен делиться на n_heads");
    let head_dim = d_model / n_heads;
    let mut out = Array4::<f32>::zeros((batch, n_heads, seq, head_dim));

    for b in 0..batch {
        for t in 0..seq {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    out[[b, h, t, d]] = x_data[[b, t, h * head_dim + d]];
                }
            }
        }
    }

    let lhs = x.clone();
    let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
        let g4 = g
            .clone()
            .into_dimensionality::<Ix4>()
            .expect("split_heads backward ожидает [batch, heads, seq, head_dim]");
        let mut dx = Array3::<f32>::zeros((batch, seq, d_model));
        for b in 0..batch {
            for t in 0..seq {
                for h in 0..n_heads {
                    for d in 0..head_dim {
                        dx[[b, t, h * head_dim + d]] += g4[[b, h, t, d]];
                    }
                }
            }
        }
        lhs.add_grad(&dx.into_dyn());
    });
    Tensor::from_op(out.into_dyn(), vec![x.clone()], backward)
}

pub fn merge_heads(x: &Tensor) -> Tensor {
    let x_data = x
        .data()
        .into_dimensionality::<Ix4>()
        .expect("merge_heads ожидает [batch, heads, seq, head_dim]");
    let (batch, n_heads, seq, head_dim) = x_data.dim();
    let d_model = n_heads * head_dim;
    let mut out = Array3::<f32>::zeros((batch, seq, d_model));

    for b in 0..batch {
        for t in 0..seq {
            for h in 0..n_heads {
                for d in 0..head_dim {
                    out[[b, t, h * head_dim + d]] = x_data[[b, h, t, d]];
                }
            }
        }
    }

    let lhs = x.clone();
    let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
        let g3 = g
            .clone()
            .into_dimensionality::<Ix3>()
            .expect("merge_heads backward ожидает [batch, seq, d_model]");
        let mut dx = Array4::<f32>::zeros((batch, n_heads, seq, head_dim));
        for b in 0..batch {
            for t in 0..seq {
                for h in 0..n_heads {
                    for d in 0..head_dim {
                        dx[[b, h, t, d]] += g3[[b, t, h * head_dim + d]];
                    }
                }
            }
        }
        lhs.add_grad(&dx.into_dyn());
    });
    Tensor::from_op(out.into_dyn(), vec![x.clone()], backward)
}

fn mask_at(mask: &ArrayD<f32>, b: usize, h: usize, tq: usize, tk: usize) -> f32 {
    match mask.ndim() {
        2 => mask[IxDyn(&[tq, tk])],
        4 => mask[IxDyn(&[b, h, tq, tk])],
        _ => panic!("attention mask должен быть [tq, tk] или [batch, heads, tq, tk]"),
    }
}

pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&ArrayD<f32>>,
) -> Tensor {
    let q4 = q
        .data()
        .into_dimensionality::<Ix4>()
        .expect("attention q ожидает [batch, heads, tq, head_dim]");
    let k4 = k
        .data()
        .into_dimensionality::<Ix4>()
        .expect("attention k ожидает [batch, heads, tk, head_dim]");
    let v4 = v
        .data()
        .into_dimensionality::<Ix4>()
        .expect("attention v ожидает [batch, heads, tk, value_dim]");
    let (batch, heads, tq_len, head_dim) = q4.dim();
    let (k_batch, k_heads, tk_len, k_dim) = k4.dim();
    let (v_batch, v_heads, v_tk_len, value_dim) = v4.dim();
    assert_eq!((k_batch, k_heads, k_dim), (batch, heads, head_dim));
    assert_eq!((v_batch, v_heads, v_tk_len), (batch, heads, tk_len));

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_data = mask.cloned();
    let mut probs = Array4::<f32>::zeros((batch, heads, tq_len, tk_len));
    let mut out = Array4::<f32>::zeros((batch, heads, tq_len, value_dim));

    for b in 0..batch {
        for h in 0..heads {
            for tq_i in 0..tq_len {
                let mut max_score = f32::NEG_INFINITY;
                for tk_i in 0..tk_len {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += q4[[b, h, tq_i, d]] * k4[[b, h, tk_i, d]];
                    }
                    score *= scale;
                    if let Some(mask) = &mask_data {
                        score += mask_at(mask, b, h, tq_i, tk_i);
                    }
                    probs[[b, h, tq_i, tk_i]] = score;
                    max_score = max_score.max(score);
                }

                let mut denom = 0.0;
                for tk_i in 0..tk_len {
                    let e = (probs[[b, h, tq_i, tk_i]] - max_score).exp();
                    probs[[b, h, tq_i, tk_i]] = e;
                    denom += e;
                }
                for tk_i in 0..tk_len {
                    probs[[b, h, tq_i, tk_i]] /= denom;
                    for d in 0..value_dim {
                        out[[b, h, tq_i, d]] += probs[[b, h, tq_i, tk_i]] * v4[[b, h, tk_i, d]];
                    }
                }
            }
        }
    }

    let q_t = q.clone();
    let k_t = k.clone();
    let v_t = v.clone();
    let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
        let gout = g
            .clone()
            .into_dimensionality::<Ix4>()
            .expect("attention backward ожидает [batch, heads, tq, value_dim]");
        let mut dq = Array4::<f32>::zeros((batch, heads, tq_len, head_dim));
        let mut dk = Array4::<f32>::zeros((batch, heads, tk_len, head_dim));
        let mut dv = Array4::<f32>::zeros((batch, heads, tk_len, value_dim));

        for b in 0..batch {
            for h in 0..heads {
                for tq_i in 0..tq_len {
                    let mut dprob = vec![0.0; tk_len];
                    for tk_i in 0..tk_len {
                        for d in 0..value_dim {
                            dv[[b, h, tk_i, d]] +=
                                probs[[b, h, tq_i, tk_i]] * gout[[b, h, tq_i, d]];
                            dprob[tk_i] += gout[[b, h, tq_i, d]] * v4[[b, h, tk_i, d]];
                        }
                    }

                    let mut softmax_dot = 0.0;
                    for tk_i in 0..tk_len {
                        softmax_dot += dprob[tk_i] * probs[[b, h, tq_i, tk_i]];
                    }

                    for tk_i in 0..tk_len {
                        let dscore =
                            probs[[b, h, tq_i, tk_i]] * (dprob[tk_i] - softmax_dot) * scale;
                        for d in 0..head_dim {
                            dq[[b, h, tq_i, d]] += dscore * k4[[b, h, tk_i, d]];
                            dk[[b, h, tk_i, d]] += dscore * q4[[b, h, tq_i, d]];
                        }
                    }
                }
            }
        }

        q_t.add_grad(&dq.into_dyn());
        k_t.add_grad(&dk.into_dyn());
        v_t.add_grad(&dv.into_dyn());
    });
    Tensor::from_op(
        out.into_dyn(),
        vec![q.clone(), k.clone(), v.clone()],
        backward,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check, grad_check_with_tol, rand_tensor};
    use ndarray::IxDyn;

    #[test]
    fn check_reshape() {
        let x = rand_tensor(&[2, 3, 4]);
        grad_check(&[x], |t| t[0].reshape(&[6, 4]).mean());
    }

    #[test]
    fn check_gelu() {
        let x = rand_tensor(&[3, 4]);
        grad_check(&[x], |t| t[0].gelu().mean());
    }

    #[test]
    fn check_sin_cos() {
        let x = rand_tensor(&[3, 4]);
        grad_check(&[x.clone()], |t| t[0].sin().mean());
        grad_check(&[x], |t| t[0].cos().mean());
    }

    #[test]
    fn check_softmax_last_dim() {
        let x = rand_tensor(&[2, 3, 4]);
        grad_check(&[x], |t| t[0].softmax_last_dim().mean());
    }

    #[test]
    fn check_layer_norm_last_dim() {
        let x = rand_tensor(&[2, 3, 4]);
        let gamma = rand_tensor(&[4]);
        let beta = rand_tensor(&[4]);
        grad_check_with_tol(&[x, gamma, beta], 1e-3, 2e-2, |t| {
            t[0].layer_norm_last_dim(&t[1], &t[2], 1e-5).mean()
        });
    }

    #[test]
    fn check_split_merge_heads() {
        let x = rand_tensor(&[2, 3, 4]);
        grad_check(&[x], |t| merge_heads(&split_heads(&t[0], 2)).mean());
    }

    #[test]
    fn check_scaled_dot_product_attention() {
        let q = rand_tensor(&[1, 2, 3, 2]);
        let k = rand_tensor(&[1, 2, 3, 2]);
        let v = rand_tensor(&[1, 2, 3, 2]);
        grad_check_with_tol(&[q, k, v], 1e-3, 2e-2, |t| {
            scaled_dot_product_attention(&t[0], &t[1], &t[2], None).mean()
        });
    }

    #[test]
    fn attention_accepts_2d_mask() {
        let q = rand_tensor(&[1, 1, 2, 2]);
        let k = rand_tensor(&[1, 1, 2, 2]);
        let v = rand_tensor(&[1, 1, 2, 2]);
        let mask = ArrayD::from_elem(IxDyn(&[2, 2]), 0.0);
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
    }
}
