//! Диагностика surrogate-модели (roadmap шаг 2): инструменты понять причину
//! плохой точности ДО смены архитектуры — underfit (ёмкость/кодирование) vs
//! покрытие данных vs чувствительность карты.

use crate::blackbox::BlackBox;
use crate::data::{Normalizer, NumericDataset};
use crate::encoders::FeatureSpec;
use crate::numeric_model::NumericConfig;
use crate::train::{train_surrogate, TrainConfig};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

/// Overfit-проба ёмкости: свежая модель учится «в лоб» на маленьком подмножестве.
/// Низкий итоговый train-loss → модель СПОСОБНА подогнать данные (проблема в
/// данных/обобщении); высокий → underfit (мало ёмкости или слабое кодирование
/// значений). Возвращает нормализованный MSE последней контрольной точки.
pub fn overfit_probe(
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    n_outputs: usize,
    subset: &NumericDataset,
    epochs: usize,
) -> f32 {
    let in_norm = Normalizer::fit(&subset.inputs, specs);
    let out_norm = Normalizer::fit(&subset.outputs, &Normalizer::all_continuous(n_outputs));
    let model = nc.build(specs, n_outputs);
    let tcfg = TrainConfig {
        epochs,
        batch_size: subset.len().clamp(1, 16),
        lr: 3e-3,
        seed: 0,
        schedule: crate::train::LrSchedule::Constant,
    };
    let history = train_surrogate(&model, subset, &in_norm, &out_norm, &tcfg);
    history.last().copied().unwrap_or(f32::NAN)
}

pub struct RangeReport {
    pub rows_out: usize,
    pub total: usize,
    pub per_feature: Vec<usize>,
}

/// Сколько test-точек выходят за обученный диапазон входов (экстраполяция).
pub fn range_report(in_norm: &Normalizer, inputs: &Array2<f32>) -> RangeReport {
    let flags = in_norm.out_of_range(inputs);
    let mut per_feature = vec![0usize; inputs.ncols()];
    let mut rows = BTreeSet::new();
    for (r, c) in &flags {
        per_feature[*c] += 1;
        rows.insert(*r);
    }
    RangeReport {
        rows_out: rows.len(),
        total: inputs.nrows(),
        per_feature,
    }
}

pub struct FeatureResidual {
    /// Доля смен знака остатка вдоль признака. ~0.5 → остаток быстро
    /// осциллирует по признаку (непокрытая частота → намёк на Fourier).
    pub sign_change_rate: f32,
    /// Отношение средней |ошибки| в хвостах (внешние 20%) к середине (60%).
    /// >1.5 → ошибка растёт к краям (масштаб/нормализация/экстраполяция).
    pub tail_ratio: f32,
}

/// Анализ формы остатка `pred - target` как функции каждого входного признака.
pub fn residual_diagnostics(
    inputs: &Array2<f32>,
    pred: &Array2<f32>,
    target: &Array2<f32>,
) -> Vec<FeatureResidual> {
    let n = inputs.nrows();
    let o = target.ncols();

    // На строку: знаковый скаляр (сумма по выходам) и магнитуда остатка.
    let mut signed = vec![0.0f32; n];
    let mut mag = vec![0.0f32; n];
    for i in 0..n {
        let mut s = 0.0;
        let mut m2 = 0.0;
        for k in 0..o {
            let r = pred[[i, k]] - target[[i, k]];
            s += r;
            m2 += r * r;
        }
        signed[i] = s;
        mag[i] = m2.sqrt();
    }

    (0..inputs.ncols())
        .map(|j| {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                inputs[[a, j]]
                    .partial_cmp(&inputs[[b, j]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let changes = (1..n)
                .filter(|&w| signed[idx[w - 1]] * signed[idx[w]] < 0.0)
                .count();
            let sign_change_rate = if n > 1 {
                changes as f32 / (n - 1) as f32
            } else {
                0.0
            };
            FeatureResidual {
                sign_change_rate,
                tail_ratio: tail_ratio(&idx, &mag),
            }
        })
        .collect()
}

fn tail_ratio(idx_sorted: &[usize], mag: &[f32]) -> f32 {
    let n = idx_sorted.len();
    if n < 10 {
        return 1.0;
    }
    let k = n / 10; // 10% с каждого края
    let mean = |ix: &[usize]| -> f32 {
        if ix.is_empty() {
            0.0
        } else {
            ix.iter().map(|&i| mag[i]).sum::<f32>() / ix.len() as f32
        }
    };
    let outer: Vec<usize> = idx_sorted[..k]
        .iter()
        .chain(&idx_sorted[n - k..])
        .copied()
        .collect();
    let inner = &idx_sorted[(n / 5)..(n - n / 5)];
    let inner_mean = mean(inner);
    if inner_mean < 1e-12 {
        1.0
    } else {
        mean(&outer) / inner_mean
    }
}

/// Локальная чувствительность карты (бедный Ляпунов): близкие входы → насколько
/// расходятся выходы. Возвращает (среднее, максимум) безразмерного отношения
/// `||Δy||/||Δx||` в нормализованном (z-score) пространстве. Большой максимум →
/// области чувствительности / возможный хаос → потолок точности surrogate.
pub fn sensitivity_probe(
    bb: &BlackBox,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    n_pairs: usize,
    eps_frac: f32,
    seed: u64,
) -> (f32, f32) {
    let mut rng = StdRng::seed_from_u64(seed);
    let f = bb.n_inputs();
    let mut ratios = Vec::with_capacity(n_pairs);

    for _ in 0..n_pairs {
        let mut x = vec![0.0f32; f];
        let mut x2 = vec![0.0f32; f];
        for j in 0..f {
            let (lo, hi) = bb.input_ranges[j];
            x[j] = rng.gen_range(lo..=hi);
            let d = eps_frac * (hi - lo);
            x2[j] = x[j] + if rng.gen::<bool>() { d } else { -d };
        }
        let dx = norm_dist(&x, &x2, &in_norm.std);
        let dy = norm_dist(&bb.eval(&x), &bb.eval(&x2), &out_norm.std);
        if dx > 1e-9 {
            ratios.push(dy / dx);
        }
    }

    if ratios.is_empty() {
        return (0.0, 0.0);
    }
    let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let max = ratios.iter().copied().fold(0.0, f32::max);
    (mean, max)
}

/// Евклидово расстояние в единицах std (per-компонента).
fn norm_dist(a: &[f32], b: &[f32], std: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .zip(std)
        .map(|((&x, &y), &s)| {
            let d = (x - y) / s;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use ndarray::array;

    #[test]
    fn range_report_counts_extrapolation() {
        let train = array![[0.0], [1.0], [2.0]];
        let norm = Normalizer::fit(&train, &Normalizer::all_continuous(1));
        let probe = array![[5.0], [-1.0], [1.0]];
        let rr = range_report(&norm, &probe);
        assert_eq!(rr.rows_out, 2); // 5 и -1 вне [0,2]
        assert_eq!(rr.per_feature, vec![2]);
    }

    #[test]
    fn residual_detects_oscillation() {
        // Входы по возрастанию, остаток альтернирует знак -> высокая частота смен.
        let n = 20;
        let inputs = Array2::from_shape_fn((n, 1), |(i, _)| i as f32);
        let target = Array2::zeros((n, 1));
        let pred = Array2::from_shape_fn((n, 1), |(i, _)| if i % 2 == 0 { 1.0 } else { -1.0 });
        let d = residual_diagnostics(&inputs, &pred, &target);
        assert!(d[0].sign_change_rate > 0.8, "ожидали высокую осцилляцию");

        // Идеальное предсказание -> нет смен знака.
        let d2 = residual_diagnostics(&inputs, &target, &target);
        assert_eq!(d2[0].sign_change_rate, 0.0);
    }

    #[test]
    fn sensitivity_probe_runs() {
        let bb = blackbox::sum();
        let data = bb.generate(64, 0);
        let in_norm = Normalizer::fit(&data.inputs, &Normalizer::all_continuous(2));
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let (mean, max) = sensitivity_probe(&bb, &in_norm, &out_norm, 100, 0.01, 0);
        assert!(mean.is_finite() && max >= mean && mean > 0.0);
    }

    #[test]
    fn overfit_probe_fits_easy_data() {
        let nc = NumericConfig {
            kind: crate::numeric_model::ModelKind::Transformer,
            transformer: crate::config::ModelConfig {
                d_model: 16,
                n_heads: 2,
                n_enc_layers: 1,
                n_dec_layers: 1,
                d_ff: 32,
                ln_eps: 1e-5,
            },
            value: crate::encoders::ValueEncoderConfig::default(),
            mlp_width: 32,
            mlp_layers: 2,
            kan: Default::default(),
        };
        let subset = blackbox::sum().generate(32, 0);
        let specs = vec![FeatureSpec::Continuous; 2];
        let loss = overfit_probe(&nc, &specs, 1, &subset, 50);
        assert!(loss < 0.5, "лёгкие данные должны подгоняться, loss={loss}");
    }
}
