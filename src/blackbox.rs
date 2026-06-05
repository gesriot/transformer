//! Синтетические «чёрные ящики» для проверки surrogate-модели (Plan.md §8).
//!
//! Каждый ящик — детерминированная функция `X -> Y` с заданными диапазонами
//! входов. По порядку сложности: sum -> product -> sine -> polynomial ->
//! projectile (мини-физика с разными масштабами, проверяет нормализацию).

use crate::data::NumericDataset;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct BlackBox {
    pub name: &'static str,
    pub input_ranges: Vec<(f32, f32)>,
    pub n_outputs: usize,
    func: fn(&[f32]) -> Vec<f32>,
}

impl BlackBox {
    pub fn n_inputs(&self) -> usize {
        self.input_ranges.len()
    }

    /// Вычислить выход на одном входе (для замера чувствительности).
    pub fn eval(&self, x: &[f32]) -> Vec<f32> {
        (self.func)(x)
    }

    /// Сгенерировать `n` пар (вход равномерно из диапазонов, выход = func).
    pub fn generate(&self, n: usize, seed: u64) -> NumericDataset {
        let mut rng = StdRng::seed_from_u64(seed);
        let fin = self.n_inputs();
        let mut inputs = Array2::<f32>::zeros((n, fin));
        let mut outputs = Array2::<f32>::zeros((n, self.n_outputs));

        for i in 0..n {
            let mut row = vec![0.0; fin];
            for (j, &(lo, hi)) in self.input_ranges.iter().enumerate() {
                row[j] = rng.gen_range(lo..=hi);
            }
            let out = (self.func)(&row);
            assert_eq!(
                out.len(),
                self.n_outputs,
                "{}: неверное число выходов",
                self.name
            );
            for j in 0..fin {
                inputs[[i, j]] = row[j];
            }
            for j in 0..self.n_outputs {
                outputs[[i, j]] = out[j];
            }
        }
        NumericDataset::new(inputs, outputs)
    }
}

pub fn sum() -> BlackBox {
    BlackBox {
        name: "sum",
        input_ranges: vec![(-1.0, 1.0), (-1.0, 1.0)],
        n_outputs: 1,
        func: |x| vec![x[0] + x[1]],
    }
}

pub fn product() -> BlackBox {
    BlackBox {
        name: "product",
        input_ranges: vec![(-1.0, 1.0), (-1.0, 1.0)],
        n_outputs: 1,
        func: |x| vec![x[0] * x[1]],
    }
}

pub fn sine() -> BlackBox {
    BlackBox {
        name: "sine",
        input_ranges: vec![(-std::f32::consts::PI, std::f32::consts::PI)],
        n_outputs: 2,
        func: |x| vec![x[0].sin(), x[0].cos()],
    }
}

pub fn polynomial() -> BlackBox {
    BlackBox {
        name: "polynomial",
        input_ranges: vec![(-2.0, 2.0)],
        n_outputs: 1,
        func: |x| vec![0.5 * x[0] * x[0] - x[0] + 1.0],
    }
}

/// Дальность полёта снаряда: range = v^2 * sin(2θ) / g. Два входа разного
/// масштаба (скорость и угол) — проверяет per-feature нормализацию.
pub fn projectile() -> BlackBox {
    BlackBox {
        name: "projectile",
        input_ranges: vec![(1.0, 10.0), (0.1, 1.5)],
        n_outputs: 1,
        func: |x| {
            let g = 9.81;
            vec![x[0] * x[0] * (2.0 * x[1]).sin() / g]
        },
    }
}

pub fn by_name(name: &str) -> Option<BlackBox> {
    match name {
        "sum" => Some(sum()),
        "product" => Some(product()),
        "sine" => Some(sine()),
        "polynomial" => Some(polynomial()),
        "projectile" => Some(projectile()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_shapes_and_determinism() {
        let bb = projectile();
        let d1 = bb.generate(50, 7);
        let d2 = bb.generate(50, 7);
        assert_eq!(d1.inputs.dim(), (50, 2));
        assert_eq!(d1.outputs.dim(), (50, 1));
        assert_eq!(d1.inputs, d2.inputs); // тот же seed -> те же данные
    }

    #[test]
    fn sum_outputs_are_correct() {
        let bb = sum();
        let d = bb.generate(20, 1);
        for i in 0..20 {
            let expected = d.inputs[[i, 0]] + d.inputs[[i, 1]];
            assert!((d.outputs[[i, 0]] - expected).abs() < 1e-6);
        }
    }
}
