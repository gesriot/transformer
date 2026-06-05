//! Sweep по конфигам численной surrogate-модели.
//!
//! Общий движок для CLI `sweep` и GUI Sweep-панели: строит декартову сетку
//! конфигов, валидирует их, обучает на синтетическом blackbox и ранжирует строки
//! по среднему R2. UI получает строки через callback без парсинга stdout.

use crate::blackbox;
use crate::config::ModelConfig;
use crate::data::{Normalizer, NumericDataset};
use crate::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use crate::init::set_init_seed;
use crate::metrics::{evaluate, evaluate_per_output};
use crate::numeric_model::{validate_numeric, ModelKind, NumericConfig};
use crate::train::{predict_dataset, train_surrogate_cb, validate_train, LrSchedule, TrainConfig};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
pub struct SweepAxes {
    pub model_kinds: Vec<ModelKind>,
    pub seeds: Vec<u64>,
    pub d_models: Vec<usize>,
    pub layers: Vec<usize>,
    pub d_ffs: Vec<usize>,
    pub lrs: Vec<f32>,
    pub value_encoders: Vec<ValueEncoderKind>,
    pub fourier_scales: Vec<f32>,
    pub fourier_bands: usize,
    pub mlp_widths: Vec<usize>,
    pub mlp_layers: Vec<usize>,
    pub schedules: Vec<LrSchedule>,
    /// Эпохи search-фазы (обучение каждого кандидата при ранжировании).
    pub epochs: usize,
    /// Рекомендуемые эпохи финального обучения (переносятся в Train при Apply).
    pub final_epochs: usize,
    pub batch_size: usize,
}

impl Default for SweepAxes {
    fn default() -> Self {
        Self {
            model_kinds: vec![ModelKind::Transformer],
            seeds: vec![0],
            d_models: vec![32],
            layers: vec![2],
            d_ffs: vec![64],
            lrs: vec![1e-3],
            value_encoders: vec![ValueEncoderKind::Linear],
            fourier_scales: vec![2.0],
            fourier_bands: 6,
            mlp_widths: vec![128],
            mlp_layers: vec![3],
            schedules: vec![LrSchedule::Constant],
            epochs: 30,
            final_epochs: 30,
            batch_size: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SweepRow {
    pub label: String,
    pub choice: SweepChoice,
    pub r2_mean: f32,
    pub r2_std: f32,
    pub worst_output_r2_mean: f32,
    pub mean_output_r2_mean: f32,
    pub nrmse_mean: f32,
    pub rel_mean: f32,
}

#[derive(Clone, Debug)]
pub struct SweepChoice {
    pub kind: ModelKind,
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub d_ff: usize,
    pub value: ValueEncoderConfig,
    pub mlp_width: usize,
    pub mlp_layers: usize,
    pub lr: f32,
    pub schedule: LrSchedule,
    /// Эпохи, на которых кандидат обучался в search-фазе.
    pub epochs: usize,
    /// Рекомендуемые эпохи финального обучения (Apply → Train).
    pub final_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
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
    choice: SweepChoice,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SweepObjective {
    #[default]
    AggregateR2,
    WorstOutputR2,
    MeanOutputR2,
    Nrmse,
}

impl SweepObjective {
    pub fn score(self, row: &SweepRow) -> f32 {
        match self {
            SweepObjective::AggregateR2 => row.r2_mean,
            SweepObjective::WorstOutputR2 => row.worst_output_r2_mean,
            SweepObjective::MeanOutputR2 => row.mean_output_r2_mean,
            SweepObjective::Nrmse => -row.nrmse_mean,
        }
    }
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
    if axes.model_kinds.is_empty() {
        return Err("model-kinds: пустой список".to_string());
    }
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
    if axes.final_epochs == 0 {
        return Err("final_epochs должен быть >= 1".to_string());
    }
    if axes.fourier_bands == 0 {
        return Err("fourier_bands должен быть >= 1".to_string());
    }
    if axes.mlp_widths.is_empty() {
        return Err("mlp-widths: пустой список".to_string());
    }
    if axes.mlp_layers.is_empty() {
        return Err("mlp-layers-list: пустой список".to_string());
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
    for &width in &axes.mlp_widths {
        if width == 0 {
            return Err("mlp_width должен быть > 0".to_string());
        }
    }
    for &layers in &axes.mlp_layers {
        if layers == 0 {
            return Err("mlp_layers должен быть >= 1".to_string());
        }
    }
    Ok(())
}

fn build_candidates(axes: &SweepAxes) -> Result<Vec<Candidate>, String> {
    validate_axes(axes)?;
    let mut configs = Vec::new();
    if axes.model_kinds.contains(&ModelKind::Transformer) {
        for &dm in &axes.d_models {
            let heads = pick_heads(dm);
            for &nl in &axes.layers {
                for &dff in &axes.d_ffs {
                    for &lr in &axes.lrs {
                        for &vkind in &axes.value_encoders {
                            for &fs in &axes.fourier_scales {
                                if vkind != ValueEncoderKind::Fourier
                                    && fs != axes.fourier_scales[0]
                                {
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
                                        mlp_width: axes.mlp_widths[0],
                                        mlp_layers: axes.mlp_layers[0],
                                    };
                                    validate_numeric(&nc)?;
                                    validate_train(lr, axes.batch_size)?;
                                    let vlabel = if vkind == ValueEncoderKind::Fourier {
                                        format!("fourier@{fs}")
                                    } else {
                                        value_encoder_label(vkind).to_string()
                                    };
                                    let label = format!(
                                        "tf dm={dm} L={nl} dff={dff} lr={lr} v={vlabel} s={}",
                                        schedule_label(schedule)
                                    );
                                    configs.push(Candidate {
                                        label,
                                        nc,
                                        schedule,
                                        lr,
                                        choice: SweepChoice {
                                            kind: ModelKind::Transformer,
                                            d_model: dm,
                                            heads,
                                            layers: nl,
                                            d_ff: dff,
                                            value: ValueEncoderConfig {
                                                kind: vkind,
                                                fourier_bands: axes.fourier_bands,
                                                fourier_scale: fs,
                                            },
                                            mlp_width: axes.mlp_widths[0],
                                            mlp_layers: axes.mlp_layers[0],
                                            lr,
                                            schedule,
                                            epochs: axes.epochs,
                                            final_epochs: axes.final_epochs,
                                            batch_size: axes.batch_size,
                                            seed: axes.seeds[0],
                                        },
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if axes.model_kinds.contains(&ModelKind::Mlp) {
        let transformer = ModelConfig {
            d_model: axes.d_models[0],
            n_heads: pick_heads(axes.d_models[0]),
            n_enc_layers: axes.layers[0],
            n_dec_layers: axes.layers[0],
            d_ff: axes.d_ffs[0],
            ln_eps: 1e-5,
        };
        for &width in &axes.mlp_widths {
            for &layers in &axes.mlp_layers {
                for &lr in &axes.lrs {
                    for &schedule in &axes.schedules {
                        let nc = NumericConfig {
                            kind: ModelKind::Mlp,
                            transformer: transformer.clone(),
                            value: ValueEncoderConfig {
                                kind: ValueEncoderKind::Linear,
                                fourier_bands: axes.fourier_bands,
                                fourier_scale: axes.fourier_scales[0],
                            },
                            mlp_width: width,
                            mlp_layers: layers,
                        };
                        validate_numeric(&nc)?;
                        validate_train(lr, axes.batch_size)?;
                        let label = format!(
                            "mlp width={width} L={layers} lr={lr} s={}",
                            schedule_label(schedule)
                        );
                        configs.push(Candidate {
                            label,
                            nc,
                            schedule,
                            lr,
                            choice: SweepChoice {
                                kind: ModelKind::Mlp,
                                d_model: transformer.d_model,
                                heads: transformer.n_heads,
                                layers: transformer.n_enc_layers,
                                d_ff: transformer.d_ff,
                                value: ValueEncoderConfig {
                                    kind: ValueEncoderKind::Linear,
                                    fourier_bands: axes.fourier_bands,
                                    fourier_scale: axes.fourier_scales[0],
                                },
                                mlp_width: width,
                                mlp_layers: layers,
                                lr,
                                schedule,
                                epochs: axes.epochs,
                                final_epochs: axes.final_epochs,
                                batch_size: axes.batch_size,
                                seed: axes.seeds[0],
                            },
                        });
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

fn sort_rows(rows: &mut [SweepRow], objective: SweepObjective) {
    rows.sort_by(|a, b| {
        objective
            .score(b)
            .partial_cmp(&objective.score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

struct RunEval {
    r2: f32,
    rel: f32,
    nrmse: f32,
    worst_output_r2: f32,
    mean_output_r2: f32,
}

fn evaluate_run(
    model: &crate::numeric_model::NumericModel,
    test: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
) -> RunEval {
    let pred = predict_dataset(model, test, in_norm, out_norm);
    let aggregate = evaluate(&pred, &test.outputs);
    let per = evaluate_per_output(&pred, &test.outputs);
    let worst_output_r2 = per
        .iter()
        .map(|m| m.r2)
        .fold(f32::INFINITY, |a, b| a.min(b));
    let mean_output_r2 = mean(&per.iter().map(|m| m.r2).collect::<Vec<_>>());
    RunEval {
        r2: aggregate.r2,
        rel: aggregate.rel_error,
        nrmse: (1.0 - aggregate.r2).max(0.0).sqrt(),
        worst_output_r2,
        mean_output_r2,
    }
}

fn row_from_runs(label: String, choice: SweepChoice, runs: &[RunEval]) -> SweepRow {
    let r2s: Vec<f32> = runs.iter().map(|m| m.r2).collect();
    let rels: Vec<f32> = runs.iter().map(|m| m.rel).collect();
    let nrmses: Vec<f32> = runs.iter().map(|m| m.nrmse).collect();
    let worsts: Vec<f32> = runs.iter().map(|m| m.worst_output_r2).collect();
    let means: Vec<f32> = runs.iter().map(|m| m.mean_output_r2).collect();
    let (r2_mean, r2_std) = mean_std(&r2s);
    SweepRow {
        label,
        choice,
        r2_mean,
        r2_std,
        worst_output_r2_mean: mean(&worsts),
        mean_output_r2_mean: mean(&means),
        nrmse_mean: mean(&nrmses),
        rel_mean: mean(&rels),
    }
}

pub fn run_blackbox_sweep<F>(
    blackbox_name: &str,
    axes: &SweepAxes,
    cancel: &AtomicBool,
    on_row: F,
) -> Result<SweepResult, String>
where
    F: FnMut(&SweepRow),
{
    run_blackbox_sweep_with_objective(
        blackbox_name,
        axes,
        SweepObjective::AggregateR2,
        cancel,
        on_row,
    )
}

pub fn run_blackbox_sweep_with_objective<F>(
    blackbox_name: &str,
    axes: &SweepAxes,
    objective: SweepObjective,
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
            sort_rows(&mut rows, objective);
            return Ok(SweepResult {
                rows,
                total_configs,
                total_runs,
                cancelled: true,
            });
        }

        let mut runs = Vec::new();
        for &seed in &axes.seeds {
            if cancel.load(Ordering::Relaxed) {
                sort_rows(&mut rows, objective);
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
                sort_rows(&mut rows, objective);
                return Ok(SweepResult {
                    rows,
                    total_configs,
                    total_runs,
                    cancelled: true,
                });
            }

            runs.push(evaluate_run(&model, &test, &in_norm, &out_norm));
        }

        let row = row_from_runs(candidate.label, candidate.choice, &runs);
        on_row(&row);
        rows.push(row);
    }

    sort_rows(&mut rows, objective);
    Ok(SweepResult {
        rows,
        total_configs,
        total_runs,
        cancelled: false,
    })
}

pub fn run_file_sweep<F>(
    data: &NumericDataset,
    in_specs: &[FeatureSpec],
    axes: &SweepAxes,
    objective: SweepObjective,
    cancel: &AtomicBool,
    mut on_row: F,
) -> Result<SweepResult, String>
where
    F: FnMut(&SweepRow),
{
    let configs = build_candidates(axes)?;
    let total_configs = configs.len();
    let total_runs = total_configs * axes.seeds.len();
    let n_out = data.outputs.ncols();
    let (train, test) = data.split(0.8, 1);
    let in_norm = Normalizer::fit(&train.inputs, in_specs);
    let out_norm = Normalizer::fit(&train.outputs, &Normalizer::all_continuous(n_out));
    let mut rows = Vec::new();

    for candidate in configs {
        if cancel.load(Ordering::Relaxed) {
            sort_rows(&mut rows, objective);
            return Ok(SweepResult {
                rows,
                total_configs,
                total_runs,
                cancelled: true,
            });
        }

        let mut runs = Vec::new();
        for &seed in &axes.seeds {
            if cancel.load(Ordering::Relaxed) {
                sort_rows(&mut rows, objective);
                return Ok(SweepResult {
                    rows,
                    total_configs,
                    total_runs,
                    cancelled: true,
                });
            }

            set_init_seed(seed);
            let model = candidate.nc.build(in_specs, n_out);
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
                sort_rows(&mut rows, objective);
                return Ok(SweepResult {
                    rows,
                    total_configs,
                    total_runs,
                    cancelled: true,
                });
            }

            runs.push(evaluate_run(&model, &test, &in_norm, &out_norm));
        }

        let row = row_from_runs(candidate.label, candidate.choice, &runs);
        on_row(&row);
        rows.push(row);
    }

    sort_rows(&mut rows, objective);
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

    #[test]
    fn file_sweep_runs_mlp_and_transformer() {
        let bb = blackbox::by_name("sum").unwrap();
        let data = bb.generate(96, 0);
        let specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
        let axes = SweepAxes {
            model_kinds: vec![ModelKind::Transformer, ModelKind::Mlp],
            epochs: 1,
            batch_size: 32,
            d_models: vec![16],
            layers: vec![1],
            d_ffs: vec![32],
            lrs: vec![3e-3],
            value_encoders: vec![ValueEncoderKind::Linear],
            mlp_widths: vec![16],
            mlp_layers: vec![1],
            ..SweepAxes::default()
        };
        let cancel = AtomicBool::new(false);
        let result = run_file_sweep(
            &data,
            &specs,
            &axes,
            SweepObjective::WorstOutputR2,
            &cancel,
            |_| {},
        )
        .unwrap();
        assert_eq!(result.total_configs, 2);
        assert_eq!(result.rows.len(), 2);
        assert!(result.rows[0].worst_output_r2_mean.is_finite());
    }
}
