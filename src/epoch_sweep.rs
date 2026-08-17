//! Кривая обучения по числу эпох — нативный порт `epoch_sweep.py`.
//!
//! В отличие от Python (тренировал независимые модели на каждый счётчик эпох,
//! что добавляло шум инициализации), здесь обучается ОДНА модель на fold, а
//! метрики снимаются на чекпойнтах эпох — это даёт честную кривую обучения без
//! межточечного шума.
//!
//! Метрики снимаются на VALIDATION, а не на test: точка остановки — такой же
//! подбираемый гиперпараметр, как ширина слоя, и выбирать его по отложенным
//! данным значит их потратить.

use crate::metrics::EvalSource;
use crate::numeric_model::NumericConfig;
use crate::split::SearchPool;
use crate::train::TrainConfig;
use crate::training::{train_candidate, Dataset, EvalSchedule, TrainingSetup};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct EpochRow {
    pub epochs: usize,
    pub train_loss: f32,
    pub rmse: f32,
    pub mae: f32,
    pub rel_error: f32,
    pub r2: f32,
    /// Происхождение: validation у holdout, CV у K-fold. Никогда не test.
    pub source: EvalSource,
}

/// Кривая обучения на подготовленном pool: метрики на validation в точках
/// `milestones`.
pub fn run_epoch_sweep(
    dataset: &Dataset,
    pool: &SearchPool,
    nc: &NumericConfig,
    base_tcfg: &TrainConfig,
    milestones: &[usize],
) -> Result<Vec<EpochRow>, String> {
    let never = AtomicBool::new(false);
    run_epoch_sweep_cb(
        dataset,
        pool,
        nc,
        base_tcfg,
        milestones,
        &never,
        &mut |_| {},
    )
}

/// Как `run_epoch_sweep`, но отдаёт каждую строку через callback и уважает
/// cooperative cancel между minibatch-ами.
///
/// При K-fold точка кривой — среднее по folds: строка отдаётся в callback
/// только когда посчитаны все folds этой эпохи, иначе UI показывал бы кривую
/// одного fold как общую.
#[allow(clippy::too_many_arguments)]
pub fn run_epoch_sweep_cb(
    dataset: &Dataset,
    pool: &SearchPool,
    nc: &NumericConfig,
    base_tcfg: &TrainConfig,
    milestones: &[usize],
    cancel: &AtomicBool,
    on_row: &mut dyn FnMut(EpochRow),
) -> Result<Vec<EpochRow>, String> {
    let mut points: Vec<usize> = milestones.iter().copied().filter(|&e| e > 0).collect();
    points.sort_unstable();
    points.dedup();
    if points.is_empty() {
        return Err("нужен хотя бы один счётчик эпох > 0".to_string());
    }
    let max_epoch = *points.last().unwrap();

    // [точка эпохи][fold] -> метрики; свёртка по folds по мере готовности.
    let mut per_point: Vec<Vec<EpochRow>> = vec![Vec::new(); points.len()];
    let point_index: HashMap<usize, usize> =
        points.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let n_folds = pool.n_folds();
    let mut rows = Vec::new();

    // Кривая — это обычное обучение с замерами в контрольных точках, поэтому
    // она идёт через общее ядро, а не через собственную копию цикла.
    let mut setup = TrainingSetup::new(nc.clone(), base_tcfg.clone());
    setup.train.epochs = max_epoch;
    setup.eval = EvalSchedule::At(points.clone());

    for f in 0..n_folds {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        train_candidate(
            dataset,
            pool,
            f,
            &setup,
            base_tcfg.seed,
            cancel,
            &mut |point| {
                let Some(m) = &point.val else { return };
                let bucket = &mut per_point[point_index[&point.epoch]];
                bucket.push(EpochRow {
                    epochs: point.epoch,
                    train_loss: point.train_loss,
                    rmse: m.rmse,
                    mae: m.mae,
                    rel_error: m.rel_error,
                    r2: m.r2,
                    source: pool.source(),
                });
                // На последнем fold строка уже полна — отдаём её сразу.
                // У holdout это сохраняет live-обновление на каждом milestone.
                if bucket.len() == n_folds {
                    let n = bucket.len() as f32;
                    let row = EpochRow {
                        epochs: bucket[0].epochs,
                        train_loss: bucket.iter().map(|r| r.train_loss).sum::<f32>() / n,
                        rmse: bucket.iter().map(|r| r.rmse).sum::<f32>() / n,
                        mae: bucket.iter().map(|r| r.mae).sum::<f32>() / n,
                        rel_error: bucket.iter().map(|r| r.rel_error).sum::<f32>() / n,
                        r2: bucket.iter().map(|r| r.r2).sum::<f32>() / n,
                        source: bucket[0].source,
                    };
                    on_row(row.clone());
                    rows.push(row);
                }
            },
            &mut |_| {},
        )?;
    }
    Ok(rows)
}

/// Рекомендованная остановка: первый чекпойнт с R² ≥ target; иначе плато (после
/// `plateau_min` прирост ΔR² < `min_gain`); иначе последний.
pub fn recommended_stop(
    rows: &[EpochRow],
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
) -> Option<(usize, String)> {
    if rows.is_empty() {
        return None;
    }
    for row in rows {
        if row.r2 >= target_r2 {
            return Some((row.epochs, format!("target R²≥{target_r2}")));
        }
    }
    for w in rows.windows(2) {
        let (prev, cur) = (&w[0], &w[1]);
        if prev.r2 >= plateau_min && cur.r2 - prev.r2 < min_gain {
            return Some((prev.epochs, format!("плато ΔR²<{min_gain}")));
        }
    }
    Some((
        rows.last().unwrap().epochs,
        "лучшее из имеющегося".to_string(),
    ))
}

/// CSV с колонкой происхождения: без неё через месяц не отличить кривую по
/// validation от старой кривой по test.
pub fn rows_to_csv(rows: &[EpochRow]) -> String {
    let mut s = String::from("epochs,train_loss,rmse,mae,rel_error,r2,source\n");
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            r.epochs,
            r.train_loss,
            r.rmse,
            r.mae,
            r.rel_error,
            r.r2,
            source_label(r.source)
        ));
    }
    s
}

/// Подпись происхождения для отчётов CLI, CSV и GUI.
pub fn source_label(source: EvalSource) -> String {
    match source {
        EvalSource::Validation => "validation".to_string(),
        EvalSource::Cv { k } => format!("cv-{k}"),
        EvalSource::Test => "test".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(epochs: usize, r2: f32) -> EpochRow {
        EpochRow {
            epochs,
            train_loss: 0.0,
            rmse: 0.0,
            mae: 0.0,
            rel_error: 0.0,
            r2,
            source: EvalSource::Validation,
        }
    }

    #[test]
    fn recommends_target_first() {
        let rows = vec![row(1, 0.5), row(2, 0.96), row(5, 0.99)];
        assert_eq!(recommended_stop(&rows, 0.95, 0.02, 0.8).unwrap().0, 2);
    }

    #[test]
    fn recommends_plateau() {
        // target недостижим; после R²≥0.8 прирост 2->5 уже < 0.02 -> отмечаем prev (2).
        let rows = vec![row(1, 0.5), row(2, 0.82), row(5, 0.83), row(10, 0.835)];
        let (e, why) = recommended_stop(&rows, 0.99, 0.02, 0.8).unwrap();
        assert_eq!(e, 2);
        assert!(why.contains("плато"));
    }

    #[test]
    fn falls_back_to_last() {
        let rows = vec![row(1, 0.2), row(2, 0.5)];
        assert_eq!(recommended_stop(&rows, 0.99, 0.02, 0.8).unwrap().0, 2);
    }

    #[test]
    fn sweep_produces_rows_at_milestones() {
        let bb = crate::blackbox::sum();
        let data = bb.generate(128, 0);
        let schema = crate::schema::ModelSchema::synthetic(2, 1).unwrap();
        let dataset = Dataset::new(data, schema).unwrap();
        let prepared = crate::split::SplitPlan::default()
            .prepare(dataset.data())
            .unwrap();
        let nc = NumericConfig {
            kind: crate::numeric_model::ModelKind::Mlp,
            transformer: crate::config::ModelConfig {
                d_model: 16,
                n_heads: 2,
                n_enc_layers: 1,
                n_dec_layers: 1,
                d_ff: 32,
                ln_eps: 1e-5,
            },
            value: crate::encoders::ValueEncoderConfig::default(),
            mlp_width: 16,
            mlp_layers: 2,
            kan: Default::default(),
        };
        let base = TrainConfig {
            epochs: 0,
            batch_size: 32,
            lr: 3e-3,
            seed: 0,
            schedule: crate::train::LrSchedule::Constant,
        };
        let rows = run_epoch_sweep(&dataset, &prepared.search, &nc, &base, &[1, 2, 4]).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.epochs).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        // Кривая снимается на validation, а не на test.
        assert!(rows.iter().all(|r| r.source == EvalSource::Validation));
        // На простом sum за 4 эпохи R² должен заметно подрасти.
        assert!(rows.last().unwrap().r2 > rows[0].r2);
    }

    #[test]
    fn kfold_curve_averages_folds() {
        let bb = crate::blackbox::sum();
        let data = bb.generate(128, 0);
        let plan = crate::split::SplitPlan::KFold {
            k: 3,
            folds_seed: 1,
            test_frac: 0.2,
            test_seed: 1,
        };
        let schema = crate::schema::ModelSchema::synthetic(2, 1).unwrap();
        let dataset = Dataset::new(data, schema).unwrap();
        let prepared = plan.prepare(dataset.data()).unwrap();
        let nc = NumericConfig {
            kind: crate::numeric_model::ModelKind::Mlp,
            transformer: crate::config::ModelConfig::default(),
            value: crate::encoders::ValueEncoderConfig::default(),
            mlp_width: 16,
            mlp_layers: 2,
            kan: Default::default(),
        };
        let base = TrainConfig {
            epochs: 0,
            batch_size: 32,
            lr: 3e-3,
            seed: 0,
            schedule: crate::train::LrSchedule::Constant,
        };
        let rows = run_epoch_sweep(&dataset, &prepared.search, &nc, &base, &[1, 2]).unwrap();
        // Одна строка на точку, а не по строке на каждый fold.
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.source == EvalSource::Cv { k: 3 }));
    }
}
