//! Функции потерь для разных задач (см. Plan.md §1, §2).
//!
//! - `mse_loss` / `huber_loss` — регрессия (surrogate-модель).
//! - `cross_entropy` — классификация / char-LM.

#[cfg(any(feature = "demo", test))]
use crate::ops::{flatten_last_dim, from_flat_last_dim};
#[cfg(any(feature = "demo", test))]
use crate::tensor::BackwardFn;
use crate::tensor::Tensor;
#[cfg(any(feature = "demo", test))]
use ndarray::{Array2, ArrayD};

impl Tensor {
    /// Среднеквадратичная ошибка: `mean((pred - target)^2)`.
    /// Собрана из уже проверенных примитивов — отдельный backward не нужен.
    pub(crate) fn mse_loss(&self, target: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            target.shape(),
            "mse_loss: формы pred и target должны совпадать (без неявного broadcasting)"
        );
        let diff = self.add(&target.scale(-1.0));
        diff.mul(&diff).mean()
    }

    /// Huber (smooth L1): квадратичная при |d| <= delta, линейная вне.
    /// Устойчивее MSE к выбросам. Реализована как примитив, т.к. функция
    /// кусочная и её аналитический градиент проще задать напрямую.
    /// Остаётся ради grad-check: рабочий путь считает MSE.
    #[cfg(test)]
    pub(crate) fn huber_loss(&self, target: &Tensor, delta: f32) -> Tensor {
        assert!(delta > 0.0, "huber delta должна быть > 0");
        assert_eq!(
            self.shape(),
            target.shape(),
            "huber_loss: формы pred и target должны совпадать (без неявного broadcasting)"
        );
        let pred = self.data();
        let tgt = target.data();
        let n = pred.len() as f32;

        let diff = &pred - &tgt;
        let total: f32 = diff
            .iter()
            .map(|&d| {
                if d.abs() <= delta {
                    0.5 * d * d
                } else {
                    delta * (d.abs() - 0.5 * delta)
                }
            })
            .sum();
        let grad_per_elem = diff.mapv(|d| {
            if d.abs() <= delta {
                d / n
            } else {
                delta * d.signum() / n
            }
        });
        let loss = total / n;

        let out = ArrayD::from_elem(ndarray::IxDyn(&[1]), loss);
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let scalar = g.iter().next().copied().unwrap();
            lhs.add_grad(&(&grad_per_elem * scalar));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }
}

/// Кросс-энтропия со встроенным log-softmax (fused для численной устойчивости).
///
/// `logits` — `[.., C]`, по последней оси классы. `targets` — индексы классов
/// той же формы без последней оси (плоско: по строке на каждый ряд логитов).
/// Возвращает скаляр — среднее по всем строкам.
#[cfg(any(feature = "demo", test))]
pub(crate) fn cross_entropy(logits: &Tensor, targets: &ArrayD<usize>) -> Tensor {
    let (flat, shape) = flatten_last_dim(&logits.data());
    let (n, c) = flat.dim();
    let tgt: Vec<usize> = targets.iter().copied().collect();
    assert_eq!(
        tgt.len(),
        n,
        "cross_entropy: число таргетов должно совпадать с числом строк логитов"
    );

    // Стабильный softmax + накопление loss = -mean(log p[target]).
    let mut softmax = Array2::<f32>::zeros((n, c));
    let mut loss = 0.0;
    for r in 0..n {
        let t = tgt[r];
        assert!(t < c, "cross_entropy: индекс класса вне диапазона");
        let mut max_v = f32::NEG_INFINITY;
        for j in 0..c {
            max_v = max_v.max(flat[[r, j]]);
        }
        let mut denom = 0.0;
        for j in 0..c {
            let e = (flat[[r, j]] - max_v).exp();
            softmax[[r, j]] = e;
            denom += e;
        }
        for j in 0..c {
            softmax[[r, j]] /= denom;
        }
        loss += -softmax[[r, t]].ln();
    }
    loss /= n as f32;

    let out = ArrayD::from_elem(ndarray::IxDyn(&[1]), loss);
    let lhs = logits.clone();
    let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
        let scalar = g.iter().next().copied().unwrap();
        // dlogits[r, j] = (softmax[r, j] - 1{j == target}) / n
        let mut dx = softmax.clone();
        for r in 0..n {
            dx[[r, tgt[r]]] -= 1.0;
        }
        dx *= scalar / n as f32;
        lhs.add_grad(&from_flat_last_dim(&dx, &shape));
    });
    Tensor::from_op(out, vec![logits.clone()], backward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check, grad_check_with_tol, rand_tensor};
    use ndarray::IxDyn;

    #[test]
    fn check_mse() {
        let pred = rand_tensor(&[3, 4]);
        let target = Tensor::constant(rand_tensor(&[3, 4]).data());
        grad_check(&[pred], |t| t[0].mse_loss(&target));
    }

    #[test]
    fn check_huber() {
        // delta=1.0 и значения в [-0.7,0.7] дают diff в [-1.4,1.4] — обе ветви.
        let pred = rand_tensor(&[4, 5]);
        let target = Tensor::constant(rand_tensor(&[4, 5]).data());
        grad_check_with_tol(&[pred], 1e-3, 2e-2, |t| t[0].huber_loss(&target, 1.0));
    }

    #[test]
    fn check_cross_entropy() {
        let logits = rand_tensor(&[4, 5]);
        let targets = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0, 2, 4, 1]).unwrap();
        grad_check(&[logits], |t| cross_entropy(&t[0], &targets));
    }

    #[test]
    fn check_cross_entropy_3d() {
        // [batch, seq, vocab] -> targets [batch, seq].
        let logits = rand_tensor(&[2, 3, 5]);
        let targets = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0, 1, 2, 3, 4, 0]).unwrap();
        grad_check(&[logits], |t| cross_entropy(&t[0], &targets));
    }
}
