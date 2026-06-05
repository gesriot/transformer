//! Sweep по конфигам численной surrogate-модели.
//!
//! Общий движок для CLI `sweep` и GUI Sweep-панели: строит декартову сетку
//! конфигов, валидирует их, обучает на синтетическом blackbox и ранжирует строки
//! по среднему R2. UI получает строки через callback без парсинга stdout.

use crate::blackbox;
use crate::config::ModelConfig;
use crate::data::Normalizer;
use crate::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use crate::init::set_init_seed;
use crate::numeric_model::{validate_numeric, ModelKind, NumericConfig};
use crate::train::{
    evaluate_surrogate, train_surrogate_cb, validate_train, LrSchedule, TrainConfig,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
pub struct SweepAxes {
    pub seeds: Vec<u64>,
    pub d_models: Vec<usize>,
    pub layers: Vec<usize>,
    pub d_ffs: Vec<usize>,
    pub lrs: Vec<f32>,
    pub value_encoders: Vec<ValueEncoderKind>,
    pub fourier_scales: Vec<f32>,
    pub fourier_bands: usize,
    pub schedules: Vec<LrSchedule>,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Default for SweepAxes {
    fn default() -> Self {
        Self {
            seeds: vec![0],
            d_models: vec![32],
            layers: vec![2],
            d_ffs: vec![64],
            lrs: vec![1e-3],
            value_encoders: vec![ValueEncoderKind::Linear],
            fourier_scales: vec![2.0],
            fourier_bands: 6,
            schedules: vec![LrSchedule::Constant],
            epochs: 30,
            batch_size: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SweepRow {
    pub label: String,
    pub r2_mean: f32,
    pub r2_std: f32,
    pub nrmse_mean: f32,
    pub rel_mean: f32,
}

#[derive(Clone, Debug)]
pub struct SweepResult {
    pub rows: Vec<SweepRow>,
    pub total_configs: usize,
    pub total_runs: usize,
    pub cancelled: bool,
}

struct Candidate {
    label: String,
    nc: NumericConfig,
    schedule: LrSchedule,
    lr: f32,
}

/// n_heads, делящее d_model (предпочитаем 4).
pub fn pick_heads(d_model: usize) -> usize {
    [4, 2, 1]
        .into_iter()
        .find(|&h| d_model.is_multiple_of(h))
        .unwrap_or(1)
}

pub fn value_encoder_label(v: ValueEncoderKind) -> &'static str {
    match v {
        ValueEncoderKind::Linear => "linear",
        ValueEncoderKind::Mlp => "mlp",
        ValueEncoderKind::Fourier => "fourier",
    }
}

pub fn schedule_label(s: LrSchedule) -> &'static str {
    match s {
        LrSchedule::Constant => "constant",
        LrSchedule::WarmupCosine { .. } => "warmup-cosine",
    }
}

pub fn validate_axes(axes: &SweepAxes) -> Result<(), String> {
    if axes.seeds.is_empty() {
        return Err("seeds: пустой список".to_string());
    }
    if axes.d_models.is_empty() {
        return Err("d-models: пустой список".to_string());
    }
    if axes.layers.is_empty() {
        return Err("layers-list: пустой список".to_string());
    }
    if axes.d_ffs.is_empty() {
        return Err("d-ffs: пустой список".to_string());
    }
    if axes.lrs.is_empty() {
        return Err("lrs: пустой список".to_string());
    }
    if axes.value_encoders.is_empty() {
        return Err("value-encoders: пустой список".to_string());
    }
    if axes.fourier_scales.is_empty() {
        return Err("fourier-scales: пустой список".to_string());
    }
    if axes.schedules.is_empty() {
        return Err("schedulers: пустой список".to_string());
    }
    if axes.epochs == 0 {
        return Err("epochs должен быть >= 1".to_string());
    }
    if axes.fourier_bands == 0 {
        return Err("fourier_bands должен быть >= 1".to_string());
    }
    validate_train(axes.lrs[0], axes.batch_size)?;
    for &lr in &axes.lrs {
        validate_train(lr, axes.batch_size)?;
    }
    for &scale in &axes.fourier_scales {
        if !scale.is_finite() || scale <= 0.0 {
            return Err("fourier_scale должен быть конечным и > 0".to_string());
        }
    }
    for &schedule in &axes.schedules {
        if let LrSchedule::WarmupCosine {
            warmup_frac,
            min_lr_ratio,
        } = schedule
        {
            if !(0.0..1.0).contains(&warmup_frac) {
                return Err("warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&min_lr_ratio) {
                return Err("min-lr-ratio должен быть в [0, 1]".to_string());
            }
        }
    }
    Ok(())
}

fn build_candidates(axes: &SweepAxes) -> Result<Vec<Candidate>, String> {
    validate_axes(axes)?;
    let mut configs = Vec::new();
    for &dm in &axes.d_models {
        let heads = pick_heads(dm);
        for &nl in &axes.layers {
            for &dff in &axes.d_ffs {
                for &lr in &axes.lrs {
                    for &vkind in &axes.value_encoders {
                        for &fs in &axes.fourier_scales {
                            if vkind != ValueEncoderKind::Fourier && fs != axes.fourier_scales[0] {
                                continue;
                            }
                            for &schedule in &axes.schedules {
                                let nc = NumericConfig {
                                    kind: ModelKind::Transformer,
                                    transformer: ModelConfig {
                                        d_model: dm,
                                        n_heads: heads,
                                        n_enc_layers: nl,
                                        n_dec_layers: nl,
                                        d_ff: dff,
                                        ln_eps: 1e-5,
                                    },
                                    value: ValueEncoderConfig {
                                        kind: vkind,
                                        fourier_bands: axes.fourier_bands,
                                        fourier_scale: fs,
                                    },
                                    mlp_width: 128,
                                    mlp_layers: 3,
                                };
                                validate_numeric(&nc)?;
                                validate_train(lr, axes.batch_size)?;
                                let vlabel = if vkind == ValueEncoderKind::Fourier {
                                    format!("fourier@{fs}")
                                } else {
                                    value_encoder_label(vkind).to_string()
                                };
                                let label = format!(
                                    "dm={dm} L={nl} dff={dff} lr={lr} v={vlabel} s={}",
                                    schedule_label(schedule)
                                );
                                configs.push(Candidate {
                                    label,
                                    nc,
                                    schedule,
                                    lr,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    if configs.is_empty() {
        return Err("sweep не содержит ни одного валидного конфига".to_string());
    }
    Ok(configs)
}

pub fn sweep_size(axes: &SweepAxes) -> Result<(usize, usize), String> {
    let configs = build_candidates(axes)?;
    Ok((configs.len(), configs.len() * axes.seeds.len()))
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len().max(1) as f32;
    (m, var.sqrt())
}

fn sort_rows(rows: &mut [SweepRow]) {
    rows.sort_by(|a, b| {
        b.r2_mean
            .partial_cmp(&a.r2_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub fn run_blackbox_sweep<F>(
    blackbox_name: &str,
    axes: &SweepAxes,
    cancel: &AtomicBool,
    mut on_row: F,
) -> Result<SweepResult, String>
where
    F: FnMut(&SweepRow),
{
    let bb = blackbox::by_name(blackbox_name)
        .ok_or_else(|| format!("неизвестный чёрный ящик: {blackbox_name}"))?;
    let configs = build_candidates(axes)?;
    let total_configs = configs.len();
    let total_runs = total_configs * axes.seeds.len();
    let in_specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
    let n_out = bb.n_outputs;
    let mut rows = Vec::new();

    for candidate in configs {
        if cancel.load(Ordering::Relaxed) {
            sort_rows(&mut rows);
            return Ok(SweepResult {
                rows,
                total_configs,
                total_runs,
                cancelled: true,
            });
        }

        let mut r2s = Vec::new();
        let mut rels = Vec::new();
        for &seed in &axes.seeds {
            if cancel.load(Ordering::Relaxed) {
                sort_rows(&mut rows);
                return Ok(SweepResult {
                    rows,
                    total_configs,
                    total_runs,
                    cancelled: true,
                });
            }

            let data = bb.generate(2000, seed);
            let (train, test) = data.split(0.8, 1);
            let in_norm = Normalizer::fit(&train.inputs, &in_specs);
            let out_norm = Normalizer::fit(&train.outputs, &Normalizer::all_continuous(n_out));
            set_init_seed(seed);
            let model = candidate.nc.build(&in_specs, n_out);
            let tcfg = TrainConfig {
                epochs: axes.epochs,
                batch_size: axes.batch_size,
                lr: candidate.lr,
                seed,
                schedule: candidate.schedule,
            };
            train_surrogate_cb(
                &model,
                &train,
                &in_norm,
                &out_norm,
                &tcfg,
                &mut |_, _| {},
                cancel,
            );
            if cancel.load(Ordering::Relaxed) {
                sort_rows(&mut rows);
                return Ok(SweepResult {
                    rows,
                    total_configs,
                    total_runs,
                    cancelled: true,
                });
            }

            let m = evaluate_surrogate(&model, &test, &in_norm, &out_norm);
            r2s.push(m.r2);
            rels.push(m.rel_error);
        }

        let (r2_mean, r2_std) = mean_std(&r2s);
        let nrmse: Vec<f32> = r2s.iter().map(|r| (1.0 - r).max(0.0).sqrt()).collect();
        let row = SweepRow {
            label: candidate.label,
            r2_mean,
            r2_std,
            nrmse_mean: mean(&nrmse),
            rel_mean: mean(&rels),
        };
        on_row(&row);
        rows.push(row);
    }

    sort_rows(&mut rows);
    Ok(SweepResult {
        rows,
        total_configs,
        total_runs,
        cancelled: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn validate_rejects_empty_axes() {
        let mut axes = SweepAxes::default();
        axes.d_models.clear();
        assert!(validate_axes(&axes).is_err());
    }

    #[test]
    fn blackbox_sweep_runs_and_ranks() {
        let axes = SweepAxes {
            epochs: 1,
            batch_size: 64,
            d_models: vec![16],
            layers: vec![1],
            d_ffs: vec![32],
            lrs: vec![3e-3],
            value_encoders: vec![ValueEncoderKind::Linear],
            ..SweepAxes::default()
        };
        let cancel = AtomicBool::new(false);
        let mut seen = 0;
        let result = run_blackbox_sweep("sum", &axes, &cancel, |_| seen += 1).unwrap();
        assert_eq!(result.total_configs, 1);
        assert_eq!(seen, 1);
        assert_eq!(result.rows.len(), 1);
        assert!(result.rows[0].r2_mean.is_finite());
    }
}
