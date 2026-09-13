//! Метрики регрессии. Считаются в денормализованных единицах.
//!
//! Нормализованные метрики (`nmae`, `nrmse`) делятся на масштаб ОБУЧАЮЩИХ
//! таргетов, а не того набора, на котором меряют. Иначе одна и та же модель на
//! узком validation выглядела бы хуже, чем на широком, хотя ошибка та же.
//!
//! Относительная ошибка необязательна: возле нуля она не имеет смысла, и
//! честнее сказать «неприменимо», чем показать сотни миллионов процентов.

use ndarray::{Array2, Axis};
use std::collections::BTreeSet;

/// Насколько target должен быть меньше масштаба, чтобы относительная ошибка
/// потеряла смысл.
const NEAR_ZERO_FRACTION: f32 = 1e-3;

/// Масштаб каждого выхода по обучающим таргетам.
///
/// `Some(0)` означает известный константный train-выход: нормализованных метрик
/// у него нет, но относительная ошибка всё ещё определена, если сам target не
/// равен нулю. `None` означает, что масштаб неизвестен вовсе.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetScale {
    per_output: Vec<Option<f32>>,
}

impl TargetScale {
    /// Посчитать масштаб по обучающим таргетам — σ каждого выхода.
    pub fn of(train_targets: &Array2<f32>) -> Self {
        let n = train_targets.nrows();
        let per_output = train_targets
            .axis_iter(Axis(1))
            .map(|col| {
                if n == 0 {
                    return None;
                }
                // f64 нужен не для дополнительной точности результата, а чтобы
                // сумма большого train-набора не переполнилась раньше среднего.
                let mean = col.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
                let var = col
                    .iter()
                    .map(|&v| {
                        let d = f64::from(v) - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / n as f64;
                let sigma = var.sqrt() as f32;
                sigma.is_finite().then_some(sigma)
            })
            .collect();
        Self { per_output }
    }

    /// Масштаб, заданный напрямую. Ноль допустим и означает известный
    /// константный выход; отрицательное или неконечное σ физического смысла не
    /// имеет.
    pub fn from_sigmas(per_output: Vec<Option<f32>>) -> Result<Self, String> {
        if let Some((output, sigma)) = per_output
            .iter()
            .enumerate()
            .find_map(|(i, value)| value.filter(|v| !v.is_finite() || *v < 0.0).map(|v| (i, v)))
        {
            return Err(format!(
                "масштаб выхода {output} должен быть конечным и неотрицательным, получено {sigma}"
            ));
        }
        Ok(Self { per_output })
    }

    /// Масштаб неизвестен: нормализованные метрики и относительная ошибка
    /// просто не определены. Для мест, где нужен только R².
    pub fn unknown(n_outputs: usize) -> Self {
        Self {
            per_output: vec![None; n_outputs],
        }
    }

    pub fn n_outputs(&self) -> usize {
        self.per_output.len()
    }

    pub fn sigma(&self, output: usize) -> Option<f32> {
        self.per_output.get(output).copied().flatten()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    pub rmse: f32,
    pub mae: f32,
    /// Средняя относительная ошибка `|pred - target| / |target|`.
    ///
    /// `None`, если хотя бы один target проходит около нуля: там она не
    /// описывает качество, а описывает близость знаменателя к нулю.
    pub rel_error: Option<f32>,
    /// Коэффициент детерминации R² (доля объяснённой дисперсии).
    pub r2: f32,
    /// `MAE / σ_train`. `None`, если масштаба нет.
    pub nmae: Option<f32>,
    /// `RMSE / σ_train`. `None`, если масштаба нет.
    ///
    /// Нормализуется тем же знаменателем, что и `nmae`: две нормализованные
    /// метрики с разными знаменателями рядом несравнимы.
    pub nrmse: Option<f32>,
}

/// Метрики по всем выходам сразу.
///
/// Нормализованные величины — среднее по выходам от их собственных
/// нормализованных значений: у выходов разный масштаб, и общий знаменатель
/// означал бы, что ошибку крупного выхода меряют мелким.
pub fn evaluate(pred: &Array2<f32>, target: &Array2<f32>, scale: &TargetScale) -> Metrics {
    assert_eq!(
        pred.dim(),
        target.dim(),
        "формы pred и target должны совпадать"
    );
    let n = pred.len() as f32;
    assert!(n > 0.0, "пустые данные для метрик");
    assert_eq!(
        scale.n_outputs(),
        target.ncols(),
        "масштаб должен покрывать все выходы"
    );

    let per_output = evaluate_per_output(pred, target, scale);
    let mut se = 0.0;
    let mut ae = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = p - t;
        se += d * d;
        ae += d.abs();
    }

    let mean = target.iter().sum::<f32>() / n;
    let ss_tot: f32 = target.iter().map(|t| (t - mean) * (t - mean)).sum();
    let r2 = if ss_tot > 1e-12 {
        1.0 - se / ss_tot
    } else {
        0.0
    };

    Metrics {
        rmse: (se / n).sqrt(),
        mae: ae / n,
        rel_error: mean_of_all(per_output.iter().map(|m| m.rel_error)),
        r2,
        nmae: mean_of_all(per_output.iter().map(|m| m.nmae)),
        nrmse: mean_of_all(per_output.iter().map(|m| m.nrmse)),
    }
}

/// Среднее, определённое только когда определены ВСЕ слагаемые: иначе «среднее
/// по части выходов» выдавалось бы за метрику всей модели.
fn mean_of_all(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value?;
        count += 1;
    }
    (count > 0).then(|| sum / count as f32)
}

/// Откуда взята метрика. Без этого поля число «R² = 0.98» неинтерпретируемо:
/// validation и test означают разное, а по validation ещё и выбирают конфиг.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvalSource {
    Validation,
    Cv { k: usize },
    Test,
}

impl EvalSource {
    /// Подпись происхождения для отчётов CLI, CSV и GUI.
    pub fn label(&self) -> String {
        match self {
            EvalSource::Validation => "validation".to_string(),
            EvalSource::Cv { k } => format!("cv-{k}"),
            EvalSource::Test => "test".to_string(),
        }
    }
}

/// Происхождение ОДНОГО прогона: номер fold (None у holdout) и seed
/// инициализации.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunOrigin {
    pub fold: Option<usize>,
    pub init_seed: u64,
}

/// Метрики одного прогона вместе с его происхождением.
#[derive(Debug, Clone)]
pub(crate) struct RunEval {
    pub metrics: Metrics,
    pub per_output: Vec<Metrics>,
    pub origin: RunOrigin,
}

/// Происхождение АГРЕГАТА по конфигурации: по каким seed и скольким folds он
/// собран.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConfigOrigin {
    pub init_seeds: Vec<u64>,
    pub folds: usize,
    pub source: EvalSource,
}

/// Агрегат по конфигурации. Порядок свёртки фиксирован: СНАЧАЛА среднее по
/// folds внутри каждого init_seed, ЗАТЕМ среднее между init_seed. Поэтому
/// `r2_std_seeds` означает устойчивость к инициализации и ничего больше;
/// разброс по данным вынесен отдельным числом `r2_std_folds`.
#[derive(Debug, Clone)]
pub(crate) struct ConfigEval {
    pub mean: Metrics,
    pub per_output_mean: Vec<Metrics>,
    /// Std R² между init_seed (0 при одном seed) — то самое `±`.
    pub r2_std_seeds: f32,
    /// Средний по seed std R² между folds (0 у holdout) — справочно.
    pub r2_std_folds: f32,
    pub origin: ConfigOrigin,
}

/// Среднее по прогонам.
///
/// Нормализованные метрики усредняются как готовые значения: выводить их из
/// усреднённых MAE или R² нельзя — у каждого fold свой масштаб train.
fn mean_metrics(items: &[Metrics]) -> Metrics {
    let n = items.len() as f32;
    Metrics {
        rmse: items.iter().map(|m| m.rmse).sum::<f32>() / n,
        mae: items.iter().map(|m| m.mae).sum::<f32>() / n,
        rel_error: mean_of_all(items.iter().map(|m| m.rel_error)),
        r2: items.iter().map(|m| m.r2).sum::<f32>() / n,
        nmae: mean_of_all(items.iter().map(|m| m.nmae)),
        nrmse: mean_of_all(items.iter().map(|m| m.nrmse)),
    }
}

/// Std по совокупности (не выборочный): при одном элементе даёт 0, а не NaN.
fn population_std(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f32;
    let m = xs.iter().sum::<f32>() / n;
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt()
}

/// Свернуть прогоны конфигурации в агрегат: folds внутри seed, затем seeds.
///
/// `init_seeds` задаёт порядок свёртки и попадает в происхождение. Прогон с
/// seed вне списка — ошибка, а не молчаливое отбрасывание: иначе агрегат
/// посчитается не по тем данным, о которых отчитывается.
pub(crate) fn aggregate_runs(
    runs: &[RunEval],
    init_seeds: &[u64],
    source: EvalSource,
) -> Result<ConfigEval, String> {
    if runs.is_empty() {
        return Err("aggregate_runs: нет прогонов".to_string());
    }
    if init_seeds.is_empty() {
        return Err("aggregate_runs: пустой список init_seeds".to_string());
    }
    let unique_seeds: BTreeSet<u64> = init_seeds.iter().copied().collect();
    if unique_seeds.len() != init_seeds.len() {
        return Err("aggregate_runs: init_seeds содержит дубликаты".to_string());
    }
    let expected_folds = match source {
        EvalSource::Validation => 1,
        EvalSource::Cv { k } if k >= 2 => k,
        EvalSource::Cv { k } => {
            return Err(format!(
                "aggregate_runs: число CV-folds должно быть >= 2, получено {k}"
            ))
        }
        EvalSource::Test => {
            return Err("aggregate_runs: test нельзя агрегировать как результат поиска".to_string())
        }
    };
    if let Some(r) = runs
        .iter()
        .find(|r| !init_seeds.contains(&r.origin.init_seed))
    {
        return Err(format!(
            "aggregate_runs: прогон с init_seed {} отсутствует в списке {init_seeds:?}",
            r.origin.init_seed
        ));
    }
    let n_outputs = runs[0].per_output.len();
    if runs.iter().any(|r| r.per_output.len() != n_outputs) {
        return Err("aggregate_runs: разное число выходов между прогонами".to_string());
    }

    let mut per_seed_mean = Vec::with_capacity(init_seeds.len());
    let mut per_seed_per_output = Vec::with_capacity(init_seeds.len());
    let mut fold_stds = Vec::with_capacity(init_seeds.len());

    for &seed in init_seeds {
        let group: Vec<&RunEval> = runs.iter().filter(|r| r.origin.init_seed == seed).collect();
        if group.is_empty() {
            return Err(format!("aggregate_runs: нет прогонов с init_seed {seed}"));
        }
        if group.len() != expected_folds {
            return Err(format!(
                "aggregate_runs: init_seed {seed} даёт {} прогонов вместо {expected_folds}",
                group.len()
            ));
        }
        match source {
            EvalSource::Validation => {
                if group[0].origin.fold.is_some() {
                    return Err(
                        "aggregate_runs: holdout validation должна иметь fold=None".to_string()
                    );
                }
            }
            EvalSource::Cv { k } => {
                let actual: BTreeSet<usize> = group.iter().filter_map(|r| r.origin.fold).collect();
                let expected: BTreeSet<usize> = (0..k).collect();
                if actual != expected || group.iter().any(|r| r.origin.fold.is_none()) {
                    return Err(format!(
                        "aggregate_runs: init_seed {seed} должен содержать folds 0..{k} ровно по одному"
                    ));
                }
            }
            EvalSource::Test => unreachable!("test отвергнут выше"),
        }

        let metrics: Vec<Metrics> = group.iter().map(|r| r.metrics.clone()).collect();
        fold_stds.push(population_std(
            &metrics.iter().map(|m| m.r2).collect::<Vec<_>>(),
        ));
        per_seed_mean.push(mean_metrics(&metrics));

        let per_output: Vec<Metrics> = (0..n_outputs)
            .map(|j| {
                mean_metrics(
                    &group
                        .iter()
                        .map(|r| r.per_output[j].clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        per_seed_per_output.push(per_output);
    }

    let per_output_mean = (0..n_outputs)
        .map(|j| {
            mean_metrics(
                &per_seed_per_output
                    .iter()
                    .map(|p| p[j].clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    Ok(ConfigEval {
        mean: mean_metrics(&per_seed_mean),
        per_output_mean,
        r2_std_seeds: population_std(&per_seed_mean.iter().map(|m| m.r2).collect::<Vec<_>>()),
        r2_std_folds: fold_stds.iter().sum::<f32>() / fold_stds.len() as f32,
        origin: ConfigOrigin {
            init_seeds: init_seeds.to_vec(),
            folds: expected_folds,
            source,
        },
    })
}

/// Метрики отдельно для каждого выхода (столбца). Агрегатный `evaluate`
/// считает R² по всем выходам сразу, что у мультимасштабных целей скрывает
/// слабый выход — per-output это вскрывает.
/// Метрики каждого выхода отдельно.
///
/// Здесь и считается вся «поштучная» арифметика: нормализация масштабом своего
/// выхода и решение, определена ли относительная ошибка.
pub(crate) fn evaluate_per_output(
    pred: &Array2<f32>,
    target: &Array2<f32>,
    scale: &TargetScale,
) -> Vec<Metrics> {
    assert_eq!(
        pred.dim(),
        target.dim(),
        "формы pred и target должны совпадать"
    );
    assert_eq!(
        scale.n_outputs(),
        target.ncols(),
        "масштаб должен покрывать все выходы"
    );
    (0..pred.ncols())
        .map(|j| {
            let p = pred.column(j);
            let t = target.column(j);
            let n = p.len() as f32;
            let mut se = 0.0;
            let mut ae = 0.0;
            let mut rel = 0.0;
            for (p, t) in p.iter().zip(t.iter()) {
                let d = p - t;
                se += d * d;
                ae += d.abs();
                rel += d.abs() / t.abs();
            }
            let mean = t.sum() / n;
            let ss_tot: f32 = t.iter().map(|v| (v - mean) * (v - mean)).sum();
            let r2 = if ss_tot > 1e-12 {
                1.0 - se / ss_tot
            } else {
                0.0
            };
            let rmse = (se / n).sqrt();
            let mae = ae / n;
            let known_sigma = scale.sigma(j);
            let normalization_sigma = known_sigma.filter(|&sigma| sigma > 0.0);
            // Возле нуля относительная ошибка описывает знаменатель, а не
            // качество: тогда её нет вовсе. «Около нуля» — относительно
            // масштаба обучения, а не абсолютной константы.
            let rel_error = known_sigma.and_then(|sigma| {
                let near_zero = t.iter().any(|v| v.abs() <= NEAR_ZERO_FRACTION * sigma);
                (!near_zero && rel.is_finite()).then(|| rel / n)
            });
            Metrics {
                rmse,
                mae,
                rel_error,
                r2,
                nmae: normalization_sigma.map(|sigma| mae / sigma),
                nrmse: normalization_sigma.map(|sigma| rmse / sigma),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Масштаб берётся у ОБУЧАЮЩИХ таргетов, поэтому nMAE не зависит от того,
    /// на каком наборе меряют: узкий validation не должен портить метрику.
    #[test]
    fn normalized_error_uses_the_train_scale_not_the_evaluation_one() {
        let train = array![[0.0], [10.0], [20.0], [30.0]];
        let scale = TargetScale::of(&train);

        // Ошибка одна и та же, наборы разного разброса.
        let wide_target = array![[0.0], [30.0]];
        let wide_pred = array![[1.0], [31.0]];
        let narrow_target = array![[14.0], [16.0]];
        let narrow_pred = array![[15.0], [17.0]];

        let wide = evaluate(&wide_pred, &wide_target, &scale);
        let narrow = evaluate(&narrow_pred, &narrow_target, &scale);
        assert!((wide.nmae.unwrap() - narrow.nmae.unwrap()).abs() < 1e-6);
        // R² при этом честно разный: он как раз нормирован своим набором.
        assert!(wide.r2 > narrow.r2);
    }

    /// Константный обучающий выход не имеет масштаба: нормализованных метрик у
    /// него нет, а не «деление на зажатый эпсилон».
    #[test]
    fn a_constant_train_output_has_no_scale() {
        let train = array![[5.0], [5.0], [5.0]];
        let scale = TargetScale::of(&train);
        assert_eq!(scale.sigma(0), Some(0.0));

        let m = evaluate(&array![[5.0], [6.0]], &array![[5.0], [5.0]], &scale);
        assert!(m.nmae.is_none());
        assert!(m.nrmse.is_none());
        assert!(
            m.rel_error.is_some(),
            "для ненулевой константы MAPE определена"
        );
        // Ненормализованные метрики при этом считаются как обычно.
        assert!(m.mae > 0.0);

        let zero = evaluate(&array![[1.0], [0.0]], &array![[0.0], [0.0]], &scale);
        assert!(
            zero.rel_error.is_none(),
            "при нулевом target MAPE не определена"
        );
    }

    /// Возле нуля относительная ошибка описывает знаменатель, а не качество:
    /// тогда её нет. Молча выбрасывать такие строки нельзя — это меняло бы
    /// смысл усреднения.
    #[test]
    fn relative_error_is_undefined_near_zero() {
        let train = array![[-10.0], [0.0], [10.0]];
        let scale = TargetScale::of(&train);

        let ok = evaluate(&array![[9.0], [11.0]], &array![[10.0], [10.0]], &scale);
        assert!(ok.rel_error.is_some());

        // Один target у нуля — относительной ошибки нет у всего выхода.
        let near_zero = evaluate(&array![[0.1], [11.0]], &array![[0.0], [10.0]], &scale);
        assert!(near_zero.rel_error.is_none());
        // Остальные метрики от этого не исчезают.
        assert!(near_zero.nmae.is_some());
    }

    #[test]
    fn explicit_scale_rejects_impossible_sigmas_and_must_match_outputs() {
        assert!(TargetScale::from_sigmas(vec![Some(-1.0)]).is_err());
        assert!(TargetScale::from_sigmas(vec![Some(f32::NAN)]).is_err());
        let wrong = TargetScale::unknown(2);
        let result = std::panic::catch_unwind(|| {
            evaluate(&array![[1.0]], &array![[1.0]], &wrong);
        });
        assert!(result.is_err(), "чужой размер масштаба принят");
    }

    /// Агрегат по выходам определён, только если определены все слагаемые:
    /// «среднее по части выходов» выдавалось бы за метрику всей модели.
    #[test]
    fn aggregate_needs_every_output() {
        let train = array![[1.0, 5.0], [3.0, 5.0]];
        let scale = TargetScale::of(&train);
        assert!(scale.sigma(0).is_some());
        assert_eq!(scale.sigma(1), Some(0.0), "второй выход константен");

        let m = evaluate(
            &array![[1.0, 5.0], [3.0, 6.0]],
            &array![[1.0, 5.0], [3.0, 5.0]],
            &scale,
        );
        assert!(m.nmae.is_none(), "у одного из выходов масштаба нет");
        let per = evaluate_per_output(
            &array![[1.0, 5.0], [3.0, 6.0]],
            &array![[1.0, 5.0], [3.0, 5.0]],
            &scale,
        );
        assert!(per[0].nmae.is_some(), "у первого выхода метрика есть");
        assert!(per[1].nmae.is_none());
    }

    #[test]
    fn per_output_separates_columns() {
        // Выход 0 предсказан идеально, выход 1 — с ошибкой.
        let pred = array![[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]];
        let target = array![[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]];
        let scale = TargetScale::of(&target);
        let per = evaluate_per_output(&pred, &target, &scale);
        assert_eq!(per.len(), 2);
        assert!(per[0].rmse < 1e-6); // выход 0 идеален
        assert!(per[1].rmse > 0.1); // выход 1 хуже
    }

    #[test]
    fn perfect_prediction() {
        let y = array![[1.0], [2.0], [3.0]];
        let m = evaluate(&y, &y, &TargetScale::of(&y));
        assert!(m.rmse < 1e-6);
        assert!(m.rel_error.unwrap() < 1e-6);
        assert!((m.r2 - 1.0).abs() < 1e-6);
        assert!(m.nmae.unwrap() < 1e-6);
    }

    fn run(fold: Option<usize>, init_seed: u64, r2: f32) -> RunEval {
        let m = Metrics {
            rmse: 1.0 - r2,
            mae: 1.0 - r2,
            rel_error: Some(1.0 - r2),
            r2,
            nmae: Some(1.0 - r2),
            nrmse: Some(1.0 - r2),
        };
        RunEval {
            per_output: vec![m.clone()],
            metrics: m,
            origin: RunOrigin { fold, init_seed },
        }
    }

    #[test]
    fn aggregate_folds_then_seeds() {
        // seed 0: folds 0.8 и 1.0 -> 0.9; seed 1: 0.6 и 0.8 -> 0.7.
        // Среднее между seed = 0.8, std между seed = 0.1.
        // Разброс по folds (0.1 у обоих) в `±` не попадает.
        let runs = vec![
            run(Some(0), 0, 0.8),
            run(Some(1), 0, 1.0),
            run(Some(0), 1, 0.6),
            run(Some(1), 1, 0.8),
        ];
        let agg = aggregate_runs(&runs, &[0, 1], EvalSource::Cv { k: 2 }).unwrap();
        assert!((agg.mean.r2 - 0.8).abs() < 1e-6);
        assert!((agg.r2_std_seeds - 0.1).abs() < 1e-6);
        assert!((agg.r2_std_folds - 0.1).abs() < 1e-6);
        assert_eq!(agg.origin.folds, 2);
        assert_eq!(agg.origin.source, EvalSource::Cv { k: 2 });
    }

    #[test]
    fn single_seed_single_fold_has_zero_spread() {
        let agg = aggregate_runs(&[run(None, 0, 0.9)], &[0], EvalSource::Validation).unwrap();
        assert_eq!(agg.r2_std_seeds, 0.0);
        assert_eq!(agg.r2_std_folds, 0.0);
        assert!((agg.mean.r2 - 0.9).abs() < 1e-6);
    }

    #[test]
    fn aggregate_rejects_unlisted_or_uneven_seeds() {
        // Прогон с seed вне списка — ошибка, а не тихое отбрасывание.
        let runs = vec![run(None, 0, 0.9), run(None, 7, 0.1)];
        assert!(aggregate_runs(&runs, &[0], EvalSource::Validation).is_err());
        // Разное число folds между seed делает `±` бессмысленным.
        let uneven = vec![
            run(Some(0), 0, 0.9),
            run(Some(1), 0, 0.8),
            run(Some(0), 1, 0.7),
        ];
        assert!(aggregate_runs(&uneven, &[0, 1], EvalSource::Cv { k: 2 }).is_err());
    }

    #[test]
    fn aggregate_rejects_duplicate_seeds_and_folds() {
        let one = vec![run(None, 0, 0.9)];
        assert!(aggregate_runs(&one, &[0, 0], EvalSource::Validation).is_err());

        // Число прогонов совпадает, но fold 0 продублирован, а fold 1 потерян.
        let duplicate_fold = vec![run(Some(0), 0, 0.9), run(Some(0), 0, 0.8)];
        assert!(aggregate_runs(&duplicate_fold, &[0], EvalSource::Cv { k: 2 }).is_err());
    }

    #[test]
    fn aggregate_rejects_wrong_origin_shape() {
        assert!(aggregate_runs(&[run(Some(0), 0, 0.9)], &[0], EvalSource::Validation).is_err());
        assert!(aggregate_runs(&[run(None, 0, 0.9)], &[0], EvalSource::Test).is_err());
    }

    #[test]
    fn aggregate_rejects_different_output_counts_between_seeds() {
        let mut a = run(None, 0, 0.9);
        let b = run(None, 1, 0.8);
        a.per_output.push(a.metrics.clone());
        assert!(aggregate_runs(&[a, b], &[0, 1], EvalSource::Validation).is_err());
    }

    #[test]
    fn known_error() {
        let pred = array![[2.0], [2.0]];
        let target = array![[1.0], [3.0]];
        let m = evaluate(&pred, &target, &TargetScale::of(&target));
        assert!((m.rmse - 1.0).abs() < 1e-6); // обе ошибки по 1
        assert!((m.mae - 1.0).abs() < 1e-6);
    }
}
