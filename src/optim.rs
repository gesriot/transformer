//! Adam-оптимизатор (Kingma & Ba, 2014) с bias-коррекцией.
//!
//! Работает со списком параметров-`Tensor`. Каждый шаг читает `.grad`,
//! обновляет первый/второй моменты и применяет шаг к `.data` через
//! `Tensor::update_data`. Опциональный decoupled weight decay (AdamW).

use crate::tensor::Tensor;
use ndarray::ArrayD;

pub(crate) struct Adam {
    params: Vec<Tensor>,
    /// lr по умолчанию: им пользуется только `step`. Числовое обучение всегда
    /// задаёт lr расписанием, поэтому в part сборках поле никто не читает.
    #[allow(dead_code)]
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<ArrayD<f32>>,
    v: Vec<ArrayD<f32>>,
    t: i32,
}

impl Adam {
    /// Стандартные гиперпараметры: beta1=0.9, beta2=0.999, eps=1e-8, без WD.
    pub(crate) fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_config(params, lr, 0.9, 0.999, 1e-8, 0.0)
    }

    pub(crate) fn with_config(
        params: Vec<Tensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let m = params
            .iter()
            .map(|p| ArrayD::zeros(p.data().raw_dim()))
            .collect();
        let v = params
            .iter()
            .map(|p| ArrayD::zeros(p.data().raw_dim()))
            .collect();
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    pub(crate) fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// Шаг с lr по умолчанию (из конструктора). Для constant-расписания.
    #[allow(dead_code)]
    pub(crate) fn step(&mut self) {
        self.step_with_lr(self.lr);
    }

    /// Шаг с явно заданным lr. Политику lr (warmup/cosine) держит вызывающий
    /// код в train.rs — оптимизатор хранит только моменты, не расписание.
    pub(crate) fn step_with_lr(&mut self, lr: f32) {
        self.t += 1;
        let bias1 = 1.0 - self.beta1.powi(self.t);
        let bias2 = 1.0 - self.beta2.powi(self.t);
        let (wd, eps) = (self.weight_decay, self.eps);

        for i in 0..self.params.len() {
            let grad = self.params[i].grad();

            // m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
            self.m[i] = &self.m[i] * self.beta1 + &grad * (1.0 - self.beta1);
            self.v[i] = &self.v[i] * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);

            // Шаг с bias-коррекцией: lr * mhat / (sqrt(vhat) + eps).
            let mhat = &self.m[i] / bias1;
            let vhat = &self.v[i] / bias2;
            let step = &mhat * lr / &(vhat.mapv(f32::sqrt) + eps);

            self.params[i].update_data(|data, _grad| {
                if wd != 0.0 {
                    let decay = &*data * (lr * wd);
                    *data -= &decay;
                }
                *data -= &step;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    /// Adam должен минимизировать f(x) = ||x - x*||^2 до x ≈ x* без всякой сети.
    #[test]
    fn adam_minimizes_quadratic() {
        let start = Array::random(IxDyn(&[5]), Uniform::new(-3.0, 3.0));
        let x = Tensor::new(start);
        let target_vals = vec![1.0_f32, -2.0, 0.5, 3.0, -1.0];
        let target =
            Tensor::constant(ArrayD::from_shape_vec(IxDyn(&[5]), target_vals.clone()).unwrap());

        let mut opt = Adam::new(vec![x.clone()], 0.1);
        for _ in 0..1000 {
            opt.zero_grad();
            let loss = x.mse_loss(&target);
            loss.backward();
            opt.step();
        }

        let final_x = x.data();
        for (got, want) in final_x.iter().zip(target_vals.iter()) {
            assert!(
                (got - want).abs() < 1e-2,
                "Adam не сошёлся: got={got}, want={want}"
            );
        }
    }

    /// Loss должен монотонно (не строго, но в целом) падать.
    #[test]
    fn adam_decreases_loss() {
        let x = Tensor::new(Array::random(IxDyn(&[8]), Uniform::new(-2.0, 2.0)));
        let target = Tensor::constant(ArrayD::zeros(IxDyn(&[8])));
        let mut opt = Adam::new(vec![x.clone()], 0.05);

        let first = x.mse_loss(&target).item();
        for _ in 0..200 {
            opt.zero_grad();
            let loss = x.mse_loss(&target);
            loss.backward();
            opt.step();
        }
        let last = x.mse_loss(&target).item();
        assert!(
            last < first * 0.01,
            "loss упал недостаточно: {first} -> {last}"
        );
    }

    /// step_with_lr минимизирует ту же квадратичную при явном lr.
    #[test]
    fn step_with_lr_converges() {
        let x = Tensor::new(Array::random(IxDyn(&[4]), Uniform::new(-2.0, 2.0)));
        let target = Tensor::constant(ArrayD::zeros(IxDyn(&[4])));
        let mut opt = Adam::new(vec![x.clone()], 0.0); // lr из конструктора не используем
        for _ in 0..600 {
            opt.zero_grad();
            x.mse_loss(&target).backward();
            opt.step_with_lr(0.1);
        }
        assert!(x.mse_loss(&target).item() < 1e-3);
    }
}
