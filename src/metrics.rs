//! Метрики регрессии. Считаются в денормализованных единицах.
//! Относительная ошибка — основная для расчётов; MSE недостаточно.

use ndarray::{Array2, Axis};
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Metrics {
    pub rmse: f32,
    pub mae: f32,
    /// Средняя относительная ошибка `|pred - target| / (|target| + eps)`.
    pub rel_error: f32,
    /// Коэффициент детерминации R² (доля объяснённой дисперсии).
    pub r2: f32,
}

pub fn evaluate(pred: &Array2<f32>, target: &Array2<f32>) -> Metrics {
    assert_eq!(
        pred.dim(),
        target.dim(),
        "формы pred и target должны совпадать"
    );
    let n = pred.len() as f32;
    assert!(n > 0.0, "пустые данные для метрик");

    let mut se = 0.0;
    let mut ae = 0.0;
    let mut rel = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = p - t;
        se += d * d;
        ae += d.abs();
        rel += d.abs() / (t.abs() + 1e-8);
    }

    let mean = target.iter().sum::<f32>() / n;
    let ss_tot: f32 = target.iter().map(|t| (t - mean) * (t - mean)).sum();
    let ss_res = se;
    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Metrics {
        rmse: (se / n).sqrt(),
        mae: ae / n,
        rel_error: rel / n,
        r2,
    }
}

/// Откуда взята метрика. Без этого поля число «R² = 0.98» неинтерпретируемо:
/// validation и test означают разное, а по validation ещё и выбирают конфиг.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub struct RunOrigin {
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

fn mean_metrics(items: &[Metrics]) -> Metrics {
    let n = items.len() as f32;
    Metrics {
        rmse: items.iter().map(|m| m.rmse).sum::<f32>() / n,
        mae: items.iter().map(|m| m.mae).sum::<f32>() / n,
        rel_error: items.iter().map(|m| m.rel_error).sum::<f32>() / n,
        r2: items.iter().map(|m| m.r2).sum::<f32>() / n,
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
pub(crate) fn evaluate_per_output(pred: &Array2<f32>, target: &Array2<f32>) -> Vec<Metrics> {
    assert_eq!(
        pred.dim(),
        target.dim(),
        "формы pred и target должны совпадать"
    );
    (0..pred.ncols())
        .map(|j| {
            let p = pred.column(j).to_owned().insert_axis(Axis(1));
            let t = target.column(j).to_owned().insert_axis(Axis(1));
            evaluate(&p, &t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn per_output_separates_columns() {
        // Выход 0 предсказан идеально, выход 1 — с ошибкой.
        let pred = array![[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]];
        let target = array![[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]];
        let per = evaluate_per_output(&pred, &target);
        assert_eq!(per.len(), 2);
        assert!(per[0].rmse < 1e-6); // выход 0 идеален
        assert!(per[1].rmse > 0.1); // выход 1 хуже
    }

    #[test]
    fn perfect_prediction() {
        let y = array![[1.0], [2.0], [3.0]];
        let m = evaluate(&y, &y);
        assert!(m.rmse < 1e-6);
        assert!(m.rel_error < 1e-6);
        assert!((m.r2 - 1.0).abs() < 1e-6);
    }

    fn run(fold: Option<usize>, init_seed: u64, r2: f32) -> RunEval {
        let m = Metrics {
            rmse: 1.0 - r2,
            mae: 1.0 - r2,
            rel_error: 1.0 - r2,
            r2,
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
        let m = evaluate(&pred, &target);
        assert!((m.rmse - 1.0).abs() < 1e-6); // обе ошибки по 1
        assert!((m.mae - 1.0).abs() < 1e-6);
    }
}
