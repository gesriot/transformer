//! Свип по числу эпох (PlanUI.md §1.2) — нативный порт `epoch_sweep.py`.
//!
//! В отличие от Python (тренировал независимые модели на каждый счётчик эпох,
//! что добавляло шум инициализации), здесь обучается ОДНА модель, а метрики
//! снимаются на чекпойнтах эпох — это даёт честную кривую обучения без
//! межточечного шума. Запускается в процессе (без subprocess/regex). CLI-only.

use crate::data::{Normalizer, NumericDataset};
use crate::encoders::FeatureSpec;
use crate::metrics::evaluate;
use crate::numeric_model::NumericConfig;
use crate::train::{predict_dataset, train_surrogate_cb, TrainConfig};
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Debug)]
pub struct EpochRow {
    pub epochs: usize,
    pub train_loss: f32,
    pub rmse: f32,
    pub mae: f32,
    pub rel_error: f32,
    pub r2: f32,
}

/// Обучает одну модель до максимального чекпойнта и снимает метрики на тесте в
/// заданных точках `milestones`.
#[allow(clippy::too_many_arguments)]
pub fn run_epoch_sweep(
    train: &NumericDataset,
    test: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    n_outputs: usize,
    base_tcfg: &TrainConfig,
    milestones: &[usize],
) -> Vec<EpochRow> {
    let never = AtomicBool::new(false);
    run_epoch_sweep_cb(
        train,
        test,
        in_norm,
        out_norm,
        nc,
        specs,
        n_outputs,
        base_tcfg,
        milestones,
        &never,
        &mut |_| {},
    )
}

/// Как `run_epoch_sweep`, но отдаёт каждую строку через callback и уважает
/// cooperative cancel между minibatch-ами.
#[allow(clippy::too_many_arguments)]
pub fn run_epoch_sweep_cb(
    train: &NumericDataset,
    test: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    n_outputs: usize,
    base_tcfg: &TrainConfig,
    milestones: &[usize],
    cancel: &AtomicBool,
    on_row: &mut dyn FnMut(EpochRow),
) -> Vec<EpochRow> {
    let mut points: Vec<usize> = milestones.iter().copied().filter(|&e| e > 0).collect();
    points.sort_unstable();
    points.dedup();
    assert!(!points.is_empty(), "нужен хотя бы один счётчик эпох > 0");
    let max_epoch = *points.last().unwrap();
    let want: HashSet<usize> = points.iter().copied().collect();

    // Воспроизводимая инициализация: одна модель, фиксированный seed.
    crate::init::set_init_seed(base_tcfg.seed);
    let model = nc.build(specs, n_outputs);
    let mut tcfg = base_tcfg.clone();
    tcfg.epochs = max_epoch;

    let mut rows = Vec::new();
    train_surrogate_cb(
        &model,
        train,
        in_norm,
        out_norm,
        &tcfg,
        &mut |epoch, loss| {
            let e = epoch + 1; // 1-based
            if want.contains(&e) {
                let pred = predict_dataset(&model, test, in_norm, out_norm);
                let m = evaluate(&pred, &test.outputs);
                let row = EpochRow {
                    epochs: e,
                    train_loss: loss,
                    rmse: m.rmse,
                    mae: m.mae,
                    rel_error: m.rel_error,
                    r2: m.r2,
                };
                on_row(row.clone());
                rows.push(row);
            }
        },
        cancel,
    );
    rows
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

pub fn rows_to_csv(rows: &[EpochRow]) -> String {
    let mut s = String::from("epochs,train_loss,rmse,mae,rel_error,r2\n");
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{}\n",
            r.epochs, r.train_loss, r.rmse, r.mae, r.rel_error, r.r2
        ));
    }
    s
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
        let (tr, te) = data.split(0.8, 1);
        let specs = vec![FeatureSpec::Continuous; 2];
        let in_norm = Normalizer::fit(&tr.inputs, &specs);
        let out_norm = Normalizer::fit(&tr.outputs, &Normalizer::all_continuous(1));
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
        };
        let base = TrainConfig {
            epochs: 0,
            batch_size: 32,
            lr: 3e-3,
            seed: 0,
            schedule: crate::train::LrSchedule::Constant,
        };
        let rows = run_epoch_sweep(
            &tr,
            &te,
            &in_norm,
            &out_norm,
            &nc,
            &specs,
            1,
            &base,
            &[1, 2, 4],
        );
        assert_eq!(
            rows.iter().map(|r| r.epochs).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        // На простом sum за 4 эпохи R² должен заметно подрасти.
        assert!(rows.last().unwrap().r2 > rows[0].r2);
    }
}
