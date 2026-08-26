//! Диагностика surrogate-модели: инструменты понять причину
//! плохой точности ДО смены архитектуры — underfit (ёмкость/кодирование) vs
//! покрытие данных vs чувствительность карты.

use crate::data::{Normalizer, NumericDataset};
use crate::encoders::FeatureSpec;
use crate::numeric_model::NumericConfig;
use crate::train::{fit_normalizers, train_surrogate, TrainConfig};
use ndarray::Array2;
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
    let (in_norm, out_norm) = fit_normalizers(subset, specs);
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

/// Сколько точек выходят за обученный диапазон входов (экстраполяция).
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

/// Статистика чувствительности: `||Δy|| / ||Δx||` в единицах std.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensitivityStats {
    pub mean: f32,
    pub max: f32,
}

/// Чувствительность модели и — если процесс вызываем — исходной функции.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SensitivityReport {
    /// Сколько пар реальных строк удалось использовать.
    pub pairs: usize,
    /// Категориальные входы исключены из возмущения: дробно двигать код нельзя.
    pub categorical_inputs: usize,
    /// Чувствительность обученной модели — доступна всегда.
    pub model: SensitivityStats,
    /// Чувствительность исходного процесса — только у вызываемого ящика.
    pub reference: Option<SensitivityStats>,
    /// Насколько модель разошлась с процессом по средней чувствительности.
    pub divergence: Option<f32>,
}

/// Исходный процесс, с которым сравнивают модель: размерности и способ его
/// посчитать.
///
/// Ядро не знает про встроенные чёрные ящики — иначе диагностика тянула бы за
/// собой демонстрации и не собиралась бы без них. Вызывающий сам решает, что
/// считать «процессом».
pub struct Reference<'a> {
    pub n_inputs: usize,
    pub n_outputs: usize,
    /// Должен возвращать ровно `n_outputs` значений; несоответствие
    /// превращается в ошибку диагностики, а не в усечение или дополнение.
    pub eval: &'a dyn Fn(&[f32]) -> Vec<f32>,
}

/// Чувствительность по парам РЕАЛЬНЫХ строк.
///
/// Возмущать вход независимо нельзя: в данных бывают жёсткие связи (например
/// доли состава с постоянной суммой), и сдвиг одной координаты уводит точку с
/// многообразия — модель спрашивают о том, чего в задаче не существует.
/// Поэтому вторая точка берётся как ближайшая реальная строка с ТЕМИ ЖЕ
/// категориями: `x₂ = x + α(neighbor − x)`. Так аффинные ограничения данных
/// сохраняются сами собой.
///
/// Модель и процесс получают РОВНО ОДНИ И ТЕ ЖЕ пары: иначе их числа не
/// сравнимы. Расстояние считается только по непрерывным входам.
#[allow(clippy::too_many_arguments)]
pub fn sensitivity<F>(
    data: &NumericDataset,
    specs: &[FeatureSpec],
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    predict: F,
    reference: Option<&Reference>,
    alpha: f32,
    max_pairs: usize,
) -> Result<SensitivityReport, String>
where
    F: Fn(&Array2<f32>) -> Array2<f32>,
{
    if !(0.0..=1.0).contains(&alpha) || alpha <= 0.0 {
        return Err("доля шага должна быть в (0, 1]".to_string());
    }
    if max_pairs == 0 {
        return Err("число пар должно быть больше нуля".to_string());
    }
    let n_features = data.inputs.ncols();
    let n_outputs = data.outputs.ncols();
    if specs.len() != n_features {
        return Err(format!(
            "спецификации описывают {} входов, а в данных {n_features}",
            specs.len()
        ));
    }
    if in_norm.n_features() != n_features || in_norm.specs != specs {
        return Err("нормализатор входов не соответствует данным и их типам".to_string());
    }
    if out_norm.n_features() != n_outputs {
        return Err("нормализатор выходов не соответствует данным".to_string());
    }
    if data.inputs.iter().any(|v| !v.is_finite()) {
        return Err("входные данные содержат NaN или бесконечность".to_string());
    }
    if let Some(r) = reference {
        if r.n_inputs != n_features || r.n_outputs != n_outputs {
            return Err("размерность исходного процесса не соответствует данным".to_string());
        }
    }
    let continuous: Vec<usize> = specs
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s, FeatureSpec::Continuous))
        .map(|(i, _)| i)
        .collect();
    let categorical_inputs = specs.len() - continuous.len();
    if continuous.is_empty() {
        return Err("все входы категориальные: возмущать нечего".to_string());
    }
    if continuous
        .iter()
        .any(|&j| !in_norm.std[j].is_finite() || in_norm.std[j] <= 0.0)
        || out_norm.std.iter().any(|s| !s.is_finite() || *s <= 0.0)
    {
        return Err("нормализатор содержит некорректный масштаб".to_string());
    }
    if data.len() < 2 {
        return Err("нужно хотя бы две строки данных".to_string());
    }

    // Ближайший сосед с теми же категориями. Строк немного, поэтому честный
    // перебор понятнее и предсказуемее, чем индекс.
    let same_categories = |a: usize, b: usize| {
        specs.iter().enumerate().all(|(j, spec)| {
            matches!(spec, FeatureSpec::Continuous)
                || (data.inputs[[a, j]] - data.inputs[[b, j]]).abs() < 1e-6
        })
    };
    let distance = |a: usize, b: usize| {
        continuous
            .iter()
            .map(|&j| {
                let d = (data.inputs[[a, j]] - data.inputs[[b, j]]) / in_norm.std[j];
                d * d
            })
            .sum::<f32>()
            .sqrt()
    };

    let mut bases = Vec::new();
    let mut neighbours = Vec::new();
    for base in distributed_indices(data.len(), max_pairs) {
        let mut best: Option<(usize, f32)> = None;
        for other in 0..data.len() {
            if other == base || !same_categories(base, other) {
                continue;
            }
            let d = distance(base, other);
            if d > 1e-9 && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((other, d));
            }
        }
        if let Some((neighbour, _)) = best {
            bases.push(base);
            neighbours.push(neighbour);
        }
        if bases.len() >= max_pairs {
            break;
        }
    }
    if bases.is_empty() {
        return Err(
            "не нашлось пар строк с одинаковыми категориями и разными числовыми входами"
                .to_string(),
        );
    }

    // Обе точки каждой пары собираются один раз и отдаются обеим функциям.
    let mut x1 = Array2::<f32>::zeros((bases.len(), n_features));
    let mut x2 = Array2::<f32>::zeros((bases.len(), n_features));
    for (row, (&base, &neighbour)) in bases.iter().zip(neighbours.iter()).enumerate() {
        for j in 0..n_features {
            let a = data.inputs[[base, j]];
            let b = data.inputs[[neighbour, j]];
            x1[[row, j]] = a;
            // У категориальных a == b, поэтому шаг их не двигает.
            x2[[row, j]] = a + alpha * (b - a);
        }
    }

    let dx: Vec<f32> = (0..bases.len())
        .map(|r| {
            continuous
                .iter()
                .map(|&j| {
                    let d = (x1[[r, j]] - x2[[r, j]]) / in_norm.std[j];
                    d * d
                })
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    let model_y1 = predict(&x1);
    let model_y2 = predict(&x2);
    let model = ratios(&model_y1, &model_y2, &dx, out_norm, "модель")?;
    let reference = reference.map(|reference| -> Result<SensitivityStats, String> {
        let eval_all = |xs: &Array2<f32>| -> Result<Array2<f32>, String> {
            let mut out = Array2::<f32>::zeros((xs.nrows(), reference.n_outputs));
            for r in 0..xs.nrows() {
                let y = (reference.eval)(&xs.row(r).to_vec());
                if y.len() != reference.n_outputs {
                    return Err(format!(
                        "исходный процесс: ожидалось {} выходов, получено {}",
                        reference.n_outputs,
                        y.len()
                    ));
                }
                for (c, v) in y.into_iter().enumerate() {
                    out[[r, c]] = v;
                }
            }
            Ok(out)
        };
        ratios(
            &eval_all(&x1)?,
            &eval_all(&x2)?,
            &dx,
            out_norm,
            "исходный процесс",
        )
    });
    let reference = match reference {
        Some(stats) => Some(stats?),
        None => None,
    };

    Ok(SensitivityReport {
        pairs: bases.len(),
        categorical_inputs,
        divergence: reference.map(|r| (model.mean - r.mean).abs()),
        model,
        reference,
    })
}

/// Порядок базовых строк: сначала по одной из равномерных корзин по всей
/// таблице, затем вторые строки тех же корзин и т.д. Если в части страт нельзя
/// найти соседа с той же категорией, оставшиеся строки дают им замену — без
/// смещения оценки к началу отсортированного файла.
fn distributed_indices(n: usize, max_pairs: usize) -> Vec<usize> {
    let buckets = n.min(max_pairs);
    let max_bucket_len = n.div_ceil(buckets);
    let mut indices = Vec::with_capacity(n);
    for offset in 0..max_bucket_len {
        for bucket in 0..buckets {
            let start = bucket * n / buckets;
            let end = (bucket + 1) * n / buckets;
            let index = start + offset;
            if index < end {
                indices.push(index);
            }
        }
    }
    indices
}

/// Отношения `||Δy||/||Δx||` по парам. Формы и конечность проверяются здесь:
/// evaluator — внешний callback, и его ошибка не должна превращаться в panic
/// или NaN в публичном отчёте.
fn ratios(
    y1: &Array2<f32>,
    y2: &Array2<f32>,
    dx: &[f32],
    out_norm: &Normalizer,
    source: &str,
) -> Result<SensitivityStats, String> {
    if y1.dim() != y2.dim() || y1.nrows() != dx.len() || y1.ncols() != out_norm.n_features() {
        return Err(format!(
            "{source}: форма выходов не соответствует парам и нормализатору"
        ));
    }
    let mut values = Vec::with_capacity(dx.len());
    for (r, &step) in dx.iter().enumerate() {
        if !step.is_finite() {
            return Err("расстояние между входами не является конечным".to_string());
        }
        if step <= 1e-9 {
            continue;
        }
        let mut dy2 = 0.0;
        for c in 0..y1.ncols() {
            if !y1[[r, c]].is_finite() || !y2[[r, c]].is_finite() {
                return Err(format!(
                    "{source}: предсказание содержит NaN или бесконечность"
                ));
            }
            let d = (y1[[r, c]] - y2[[r, c]]) / out_norm.std[c];
            dy2 += d * d;
        }
        let ratio = dy2.sqrt() / step;
        if !ratio.is_finite() {
            return Err(format!("{source}: чувствительность не является конечной"));
        }
        values.push(ratio);
    }
    if values.is_empty() {
        return Err("пары оказались вырожденными: нулевой шаг по входам".to_string());
    }
    // Среднее копится в f64: на одинаковых отношениях сумма в f32 округляется
    // так, что среднее оказывается выше максимума, и инвариант ломается.
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64;
    Ok(SensitivityStats {
        mean: mean as f32,
        max: values.iter().copied().fold(0.0, f32::max),
    })
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

    /// Симплекс: доли состава с постоянной суммой. Независимое возмущение
    /// увело бы точку с многообразия, пара соседних строк — нет.
    fn simplex_data(n: usize) -> (NumericDataset, Vec<FeatureSpec>) {
        let mut inputs = Array2::<f32>::zeros((n, 3));
        let mut outputs = Array2::<f32>::zeros((n, 1));
        for i in 0..n {
            let x0 = 1.0 + i as f32;
            let x1 = 3.0 + (i % 7) as f32;
            inputs[[i, 0]] = x0;
            inputs[[i, 1]] = x1;
            inputs[[i, 2]] = 100.0 - x0 - x1;
            outputs[[i, 0]] = x0 * 2.0 + x1;
        }
        (
            NumericDataset::new(inputs, outputs),
            vec![FeatureSpec::Continuous; 3],
        )
    }

    #[test]
    fn sensitivity_keeps_pairs_on_the_data_manifold() {
        let (data, specs) = simplex_data(40);
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let observed_sums = std::cell::RefCell::new(Vec::new());
        // Модель, повторяющая зависимость: чувствительность конечна.
        let report = sensitivity(
            &data,
            &specs,
            &in_norm,
            &out_norm,
            |x| {
                observed_sums
                    .borrow_mut()
                    .extend(x.rows().into_iter().map(|row| row.sum()));
                Array2::from_shape_fn((x.nrows(), 1), |(r, _)| x[[r, 0]] * 2.0 + x[[r, 1]])
            },
            None,
            1.0,
            10,
        )
        .unwrap();

        assert!(report.pairs > 0);
        assert_eq!(report.categorical_inputs, 0);
        assert!(report.model.mean.is_finite() && report.model.mean > 0.0);
        assert!(report.model.max >= report.model.mean);
        assert!(observed_sums
            .borrow()
            .iter()
            .all(|sum| (sum - 100.0).abs() < 1e-4));
        // Процесс не вызываем — расхождение считать не из чего.
        assert!(report.reference.is_none() && report.divergence.is_none());
    }

    /// Модель и процесс меряются на ОДНИХ И ТЕХ ЖЕ парах, поэтому совпадающая
    /// с процессом модель даёт нулевое расхождение.
    #[test]
    fn model_and_reference_share_the_same_pairs() {
        let bb = blackbox::sum();
        let data = bb.generate(64, 0);
        let specs = vec![FeatureSpec::Continuous; 2];
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let report = sensitivity(
            &data,
            &specs,
            &in_norm,
            &out_norm,
            |x| Array2::from_shape_fn((x.nrows(), 1), |(r, _)| bb.eval(&x.row(r).to_vec())[0]),
            Some(&Reference {
                n_inputs: bb.n_inputs(),
                n_outputs: bb.n_outputs,
                eval: &|x| bb.eval(x),
            }),
            1.0,
            20,
        )
        .unwrap();
        let reference = report.reference.expect("процесс вызываем");
        assert!((report.model.mean - reference.mean).abs() < 1e-5);
        assert!(report.divergence.unwrap() < 1e-5);
    }

    #[test]
    fn categorical_inputs_are_excluded_and_counted() {
        // Категория во втором столбце: сосед ищется только среди строк с тем же
        // кодом, и шаг её не двигает.
        let mut inputs = Array2::<f32>::zeros((20, 2));
        let mut outputs = Array2::<f32>::zeros((20, 1));
        for i in 0..20 {
            inputs[[i, 0]] = i as f32;
            inputs[[i, 1]] = (i % 2) as f32;
            outputs[[i, 0]] = i as f32 * 0.5;
        }
        let data = NumericDataset::new(inputs, outputs);
        let specs = vec![
            FeatureSpec::Continuous,
            FeatureSpec::Categorical { cardinality: 2 },
        ];
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let seen_codes = std::cell::RefCell::new(Vec::new());
        let report = sensitivity(
            &data,
            &specs,
            &in_norm,
            &out_norm,
            |x| {
                for r in 0..x.nrows() {
                    seen_codes.borrow_mut().push(x[[r, 1]]);
                }
                Array2::from_shape_fn((x.nrows(), 1), |(r, _)| x[[r, 0]] * 0.5)
            },
            None,
            1.0,
            5,
        )
        .unwrap();
        assert_eq!(report.categorical_inputs, 1);
        assert!(report.pairs > 0);
        // Коды остались целыми: дробного шага по категории не было.
        assert!(seen_codes.borrow().iter().all(|c| *c == 0.0 || *c == 1.0));
    }

    /// Отказ должен объяснять причину, а не возвращать NaN.
    #[test]
    fn impossible_cases_explain_themselves() {
        let data = NumericDataset::new(
            Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap(),
        );
        let specs = vec![FeatureSpec::Categorical { cardinality: 2 }];
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let predict = |x: &Array2<f32>| Array2::<f32>::zeros((x.nrows(), 1));

        // Непрерывных входов нет вовсе.
        let err =
            sensitivity(&data, &specs, &in_norm, &out_norm, predict, None, 1.0, 5).unwrap_err();
        assert!(err.contains("категориальные"), "{err}");

        // Пары не находятся: у каждой строки свой код.
        let numeric = vec![FeatureSpec::Continuous];
        let single = NumericDataset::new(
            Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(),
            Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(),
        );
        let (in1, out1) = fit_normalizers(&single, &numeric);
        let err = sensitivity(&single, &numeric, &in1, &out1, predict, None, 1.0, 5).unwrap_err();
        assert!(err.contains("две строки"), "{err}");

        // Некорректная доля шага.
        let err =
            sensitivity(&data, &numeric, &in_norm, &out_norm, predict, None, 0.0, 5).unwrap_err();
        assert!(err.contains("доля шага"), "{err}");

        // Нулевой бюджет пар — ошибка, а не одна случайно посчитанная пара.
        let continuous = vec![FeatureSpec::Continuous];
        let (in2, out2) = fit_normalizers(&data, &continuous);
        let err = sensitivity(&data, &continuous, &in2, &out2, predict, None, 1.0, 0).unwrap_err();
        assert!(err.contains("число пар"), "{err}");
    }

    #[test]
    fn sensitivity_rejects_bad_or_non_finite_predictions() {
        let data = NumericDataset::new(
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
        );
        let specs = vec![FeatureSpec::Continuous];
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);

        let err = sensitivity(
            &data,
            &specs,
            &in_norm,
            &out_norm,
            |x| Array2::from_elem((x.nrows(), 1), f32::NAN),
            None,
            1.0,
            2,
        )
        .unwrap_err();
        assert!(err.contains("NaN"), "{err}");

        let err = sensitivity(
            &data,
            &specs,
            &in_norm,
            &out_norm,
            |x| Array2::zeros((x.nrows(), 2)),
            None,
            1.0,
            2,
        )
        .unwrap_err();
        assert!(err.contains("форма выходов"), "{err}");
    }

    #[test]
    fn sensitivity_rejects_wrong_reference_output_count() {
        let data = NumericDataset::new(
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
        );
        let specs = vec![FeatureSpec::Continuous];
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let predict = |x: &Array2<f32>| Array2::zeros((x.nrows(), 1));

        let too_few = |_: &[f32]| -> Vec<f32> { vec![] };
        let too_many = |_: &[f32]| -> Vec<f32> { vec![0.0, 1.0] };
        for eval in [
            &too_few as &dyn Fn(&[f32]) -> Vec<f32>,
            &too_many as &dyn Fn(&[f32]) -> Vec<f32>,
        ] {
            let err = sensitivity(
                &data,
                &specs,
                &in_norm,
                &out_norm,
                predict,
                Some(&Reference {
                    n_inputs: 1,
                    n_outputs: 1,
                    eval,
                }),
                1.0,
                2,
            )
            .unwrap_err();
            assert!(err.contains("ожидалось 1 выходов"), "{err}");
        }
    }

    #[test]
    fn sensitivity_base_order_covers_the_whole_table() {
        let order = distributed_indices(10, 3);
        assert_eq!(&order[..3], &[0, 3, 6]);
        let mut sorted = order;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
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
