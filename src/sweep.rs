//! Sweep по конфигам численной surrogate-модели.
//!
//! Общий движок для CLI `sweep` и GUI Sweep-панели: строит декартову сетку
//! конфигов и превращает результат общего поиска в прежние `SweepRow`. По
//! умолчанию ранжирование идёт по худшему выходу; UI получает строки через
//! callback без парсинга stdout.

use crate::blackbox;
use crate::config::ModelConfig;
use crate::encoders::{ValueEncoderConfig, ValueEncoderKind};
use crate::metrics::{ConfigEval, EvalSource, RunEval};
use crate::numeric_model::{validate_numeric, KanConfig, ModelKind, NumericConfig};
use crate::schema::ModelSchema;
use crate::split::{SearchPool, SplitPlan, DEFAULT_DATA_SEED};
use crate::train::{validate_train, LrSchedule, TrainConfig};
use crate::training::{
    compare_scores_desc, search, search_cost, Dataset, SearchCandidate, SearchCost, SearchPlan,
    SearchRow, TrainingSetup,
};
use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

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
    pub kan_widths: Vec<usize>,
    pub kan_layers: Vec<usize>,
    pub kan_grids: Vec<usize>,
    pub schedules: Vec<LrSchedule>,
    /// Эпохи search-фазы (обучение каждого кандидата при ранжировании).
    pub epochs: usize,
    /// Рекомендуемые эпохи финального обучения (переносятся в Train при Apply).
    pub final_epochs: usize,
    pub batch_size: usize,
}

/// Бюджет поиска: сколько мы готовы потратить.
///
/// Это именно бюджет, а не «уровень качества»: рядом всегда показывается
/// [`SearchCost`], потому что понимать цену операции важнее, чем помнить, что
/// означает название пресета.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchBudget {
    /// Быстро сравнить архитектуры между собой.
    #[default]
    Quick,
    /// Средняя сетка, один seed.
    Balanced,
    /// Широкая сетка и два seed — устойчивость выбора.
    Thorough,
}

impl SearchBudget {
    pub fn label(self) -> &'static str {
        match self {
            SearchBudget::Quick => "Быстро",
            SearchBudget::Balanced => "Сбалансированно",
            SearchBudget::Thorough => "Тщательно",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            SearchBudget::Quick => "короткий поиск: быстро сравнить выбранные архитектуры",
            SearchBudget::Balanced => "средняя сетка, один seed",
            SearchBudget::Thorough => "широкая сетка, два seed — устойчивость выбора",
        }
    }
}

impl SweepAxes {
    /// Сетка по бюджету. Раньше эти три набора жили внутри GUI, поэтому CLI не
    /// мог запустить тот же поиск, что и кнопка в интерфейсе.
    pub fn for_budget(budget: SearchBudget, model_kinds: Vec<ModelKind>) -> Self {
        let schedules = vec![LrSchedule::WarmupCosine {
            warmup_frac: 0.1,
            min_lr_ratio: 0.1,
        }];
        match budget {
            SearchBudget::Quick => Self {
                model_kinds,
                seeds: vec![0],
                d_models: vec![64],
                layers: vec![2],
                d_ffs: vec![128],
                lrs: vec![1e-3],
                value_encoders: vec![ValueEncoderKind::Linear, ValueEncoderKind::Mlp],
                fourier_scales: vec![2.0],
                fourier_bands: 6,
                mlp_widths: vec![128, 256],
                mlp_layers: vec![3],
                kan_widths: vec![16],
                kan_layers: vec![2],
                kan_grids: vec![8, 16],
                schedules,
                epochs: 25,
                final_epochs: 60,
                batch_size: 64,
            },
            SearchBudget::Balanced => Self {
                model_kinds,
                seeds: vec![0],
                d_models: vec![64, 96],
                layers: vec![2, 3],
                d_ffs: vec![128, 384],
                lrs: vec![1e-3],
                value_encoders: vec![ValueEncoderKind::Linear, ValueEncoderKind::Mlp],
                fourier_scales: vec![2.0],
                fourier_bands: 6,
                mlp_widths: vec![128, 256],
                mlp_layers: vec![3, 4],
                kan_widths: vec![16, 32],
                kan_layers: vec![2],
                kan_grids: vec![8, 16],
                schedules,
                epochs: 40,
                final_epochs: 80,
                batch_size: 64,
            },
            SearchBudget::Thorough => Self {
                model_kinds,
                seeds: vec![0, 1],
                d_models: vec![64, 96, 128],
                layers: vec![2, 3],
                d_ffs: vec![128, 256, 384],
                lrs: vec![1e-3],
                value_encoders: vec![ValueEncoderKind::Linear, ValueEncoderKind::Mlp],
                fourier_scales: vec![2.0],
                fourier_bands: 6,
                mlp_widths: vec![128, 256, 512],
                mlp_layers: vec![3, 4],
                kan_widths: vec![16, 32],
                kan_layers: vec![2, 3],
                kan_grids: vec![8, 16, 32],
                schedules,
                epochs: 40,
                final_epochs: 80,
                batch_size: 64,
            },
        }
    }
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
            kan_widths: vec![16],
            kan_layers: vec![2],
            kan_grids: vec![8],
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
    /// Разброс R² по init_seed (folds уже свёрнуты внутри seed).
    pub r2_std: f32,
    /// Средний по seed разброс R² между folds (0 у holdout).
    pub r2_std_folds: f32,
    pub worst_output_r2_mean: f32,
    pub mean_output_r2_mean: f32,
    pub nrmse_mean: f32,
    pub rel_mean: f32,
    /// Откуда метрики: validation или CV. Ранжирование по test невозможно —
    /// поиск его не видит.
    pub source: EvalSource,
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
    pub kan: KanConfig,
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

/// Цель ранжирования живёт в ядре: поиск и отображение должны означать одно и
/// то же.
pub use crate::training::SearchObjective as SweepObjective;

/// Счёт уже построенной строки — для сортировки таблицы в интерфейсе.
pub fn row_score(objective: SweepObjective, row: &SweepRow) -> f32 {
    match objective {
        SweepObjective::AggregateR2 => row.r2_mean,
        SweepObjective::WorstOutputR2 => row.worst_output_r2_mean,
        SweepObjective::MeanOutputR2 => row.mean_output_r2_mean,
        SweepObjective::Nrmse => -row.nrmse_mean,
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
    if axes.seeds.iter().copied().collect::<BTreeSet<_>>().len() != axes.seeds.len() {
        return Err("seeds содержит дубликаты".to_string());
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
    if axes.kan_widths.is_empty() {
        return Err("kan-widths: пустой список".to_string());
    }
    if axes.kan_layers.is_empty() {
        return Err("kan-layers-list: пустой список".to_string());
    }
    if axes.kan_grids.is_empty() {
        return Err("kan-grids: пустой список".to_string());
    }
    for &width in &axes.kan_widths {
        if width == 0 {
            return Err("kan_width должен быть > 0".to_string());
        }
    }
    for &layers in &axes.kan_layers {
        if layers == 0 {
            return Err("kan_layers должен быть >= 1".to_string());
        }
    }
    for &grid in &axes.kan_grids {
        if grid < 2 {
            return Err("kan_grid должен быть >= 2".to_string());
        }
    }
    Ok(())
}

/// KAN-конфиг из первых значений осей (для choices не-KAN кандидатов).
fn default_kan(axes: &SweepAxes) -> KanConfig {
    KanConfig {
        width: axes.kan_widths[0],
        layers: axes.kan_layers[0],
        grid: axes.kan_grids[0],
    }
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
                                        kan: default_kan(axes),
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
                                            kan: default_kan(axes),
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
                            kan: default_kan(axes),
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
                                kan: default_kan(axes),
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
    if axes.model_kinds.contains(&ModelKind::Kan) {
        let transformer = ModelConfig {
            d_model: axes.d_models[0],
            n_heads: pick_heads(axes.d_models[0]),
            n_enc_layers: axes.layers[0],
            n_dec_layers: axes.layers[0],
            d_ff: axes.d_ffs[0],
            ln_eps: 1e-5,
        };
        for &width in &axes.kan_widths {
            for &layers in &axes.kan_layers {
                for &grid in &axes.kan_grids {
                    for &lr in &axes.lrs {
                        for &schedule in &axes.schedules {
                            let kan = KanConfig {
                                width,
                                layers,
                                grid,
                            };
                            let nc = NumericConfig {
                                kind: ModelKind::Kan,
                                transformer: transformer.clone(),
                                value: ValueEncoderConfig {
                                    kind: ValueEncoderKind::Linear,
                                    fourier_bands: axes.fourier_bands,
                                    fourier_scale: axes.fourier_scales[0],
                                },
                                mlp_width: axes.mlp_widths[0],
                                mlp_layers: axes.mlp_layers[0],
                                kan,
                            };
                            validate_numeric(&nc)?;
                            validate_train(lr, axes.batch_size)?;
                            let label = format!(
                                "kan width={width} L={layers} grid={grid} lr={lr} s={}",
                                schedule_label(schedule)
                            );
                            configs.push(Candidate {
                                label,
                                nc,
                                schedule,
                                lr,
                                choice: SweepChoice {
                                    kind: ModelKind::Kan,
                                    d_model: transformer.d_model,
                                    heads: transformer.n_heads,
                                    layers: transformer.n_enc_layers,
                                    d_ff: transformer.d_ff,
                                    value: ValueEncoderConfig {
                                        kind: ValueEncoderKind::Linear,
                                        fourier_bands: axes.fourier_bands,
                                        fourier_scale: axes.fourier_scales[0],
                                    },
                                    mlp_width: axes.mlp_widths[0],
                                    mlp_layers: axes.mlp_layers[0],
                                    kan,
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
    if configs.is_empty() {
        return Err("sweep не содержит ни одного валидного конфига".to_string());
    }
    Ok(configs)
}

/// Совместимость: пара (конфигураций, прогонов) для старых адаптеров.
pub fn sweep_size(axes: &SweepAxes) -> Result<(usize, usize), String> {
    let cost = sweep_cost(axes, 1)?;
    Ok((cost.configs, cost.runs))
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

/// Пересортировать готовые строки — ядро ранжирует сам, это для интерфейса и
/// тестов ранжирования.
pub fn sort_rows(rows: &mut [SweepRow], objective: SweepObjective) {
    rows.sort_by(|a, b| compare_scores_desc(row_score(objective, a), row_score(objective, b)));
}

/// Строка ранжирования из агрегата. `r2_std` — разброс по init_seed (свёртка
/// folds уже произошла внутри seed), поэтому `±` означает устойчивость к
/// инициализации, а не к разбиению.
fn row_from_config_eval(
    label: String,
    choice: SweepChoice,
    agg: &ConfigEval,
    runs: &[RunEval],
) -> SweepRow {
    let per_output_r2: Vec<f32> = agg.per_output_mean.iter().map(|m| m.r2).collect();
    // nRMSE — нелинейное преобразование R², поэтому его нужно считать для
    // каждого seed × fold до усреднения, а не из уже среднего R².
    let nrmse_mean = runs
        .iter()
        .map(|run| (1.0 - run.metrics.r2).max(0.0).sqrt())
        .sum::<f32>()
        / runs.len().max(1) as f32;
    SweepRow {
        label,
        choice,
        r2_mean: agg.mean.r2,
        r2_std: agg.r2_std_seeds,
        r2_std_folds: agg.r2_std_folds,
        worst_output_r2_mean: per_output_r2.iter().copied().fold(f32::INFINITY, f32::min),
        mean_output_r2_mean: mean(&per_output_r2),
        nrmse_mean,
        rel_mean: agg.mean.rel_error,
        source: agg.origin.source,
    }
}

/// Поиск по сетке на подготовленном pool. Test сюда не попадает физически:
/// [`SearchPool`] его не содержит, поэтому отбор конфигурации не может
/// подсмотреть отложенные данные.
///
/// Для каждого кандидата: все init_seed × все folds; нормализаторы строятся по
/// train КАЖДОГО fold, метрики снимаются на его validation. Свёртка — через
/// [`aggregate_runs`] (folds внутри seed, затем seeds).
/// Кандидаты сетки как список для ядра поиска.
fn search_candidates(axes: &SweepAxes) -> Result<(Vec<SearchCandidate>, Vec<Candidate>), String> {
    let configs = build_candidates(axes)?;
    let candidates = configs
        .iter()
        .map(|c| SearchCandidate {
            label: c.label.clone(),
            setup: TrainingSetup::new(
                c.nc.clone(),
                TrainConfig {
                    epochs: axes.epochs,
                    batch_size: axes.batch_size,
                    lr: c.lr,
                    seed: 0, // seed прогона задаёт план поиска
                    schedule: c.schedule,
                },
            ),
        })
        .collect();
    Ok((candidates, configs))
}

/// Стоимость поиска до запуска: сколько конфигураций, прогонов и эпох.
///
/// Число folds передаётся отдельно, потому что оценку показывают ДО чтения
/// файла и разбиения: у holdout это 1, у K-fold — k.
pub fn sweep_cost(axes: &SweepAxes, folds: usize) -> Result<SearchCost, String> {
    let (candidates, _) = search_candidates(axes)?;
    Ok(search_cost(
        &candidates,
        &plan_from(axes, SweepObjective::default()),
        folds,
    ))
}

fn plan_from(axes: &SweepAxes, objective: SweepObjective) -> SearchPlan {
    SearchPlan {
        seeds: axes.seeds.clone(),
        objective,
    }
}

/// Перебор сетки — адаптер над общей операцией поиска: сетку и подписи строит
/// он, а обучение, свёртку и ранжирование выполняет ядро.
pub fn run_sweep<F>(
    dataset: &Dataset,
    pool: &SearchPool,
    axes: &SweepAxes,
    objective: SweepObjective,
    cancel: &AtomicBool,
    mut on_row: F,
) -> Result<SweepResult, String>
where
    F: FnMut(&SweepRow),
{
    let (candidates, configs) = search_candidates(axes)?;
    let plan = plan_from(axes, objective);
    let results = search(dataset, pool, &candidates, &plan, cancel, &mut |row| {
        on_row(&row_from_search(row, &configs));
    })?;

    let rows: Vec<SweepRow> = results
        .rows
        .iter()
        .map(|row| row_from_search(row, &configs))
        .collect();
    Ok(SweepResult {
        rows,
        total_configs: results.cost.configs,
        total_runs: results.cost.runs,
        cancelled: results.cancelled,
    })
}

fn row_from_search(row: &SearchRow, configs: &[Candidate]) -> SweepRow {
    row_from_config_eval(
        row.label.clone(),
        configs[row.candidate].choice.clone(),
        &row.eval,
        &row.runs,
    )
}

/// Поиск на встроенном чёрном ящике (демо). Данные генерируются ОДИН РАЗ с
/// фиксированным `data_seed`: ось `seeds` меняет только инициализацию модели,
/// иначе `±` смешивал бы разброс инициализации с разбросом выборки.
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
        SweepObjective::default(),
        cancel,
        on_row,
    )
}

pub fn run_blackbox_sweep_with_objective<F>(
    blackbox_name: &str,
    axes: &SweepAxes,
    objective: SweepObjective,
    cancel: &AtomicBool,
    on_row: F,
) -> Result<SweepResult, String>
where
    F: FnMut(&SweepRow),
{
    let bb = blackbox::by_name(blackbox_name)
        .ok_or_else(|| format!("неизвестный чёрный ящик: {blackbox_name}"))?;
    let data = bb.generate(2000, DEFAULT_DATA_SEED);
    let dataset = Dataset::new(data, ModelSchema::synthetic(bb.n_inputs(), bb.n_outputs)?)?;
    let prepared = SplitPlan::default().prepare(dataset.data())?;
    run_sweep(&dataset, &prepared.search, axes, objective, cancel, on_row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::NumericDataset;
    use crate::encoders::FeatureSpec;
    use crate::metrics::{aggregate_runs, RunOrigin};
    use crate::train::fit_normalizers;
    use ndarray::Array2;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn budgets_grow_and_cost_is_known_before_launch() {
        let kinds = vec![ModelKind::Mlp, ModelKind::Kan];
        let quick = sweep_cost(
            &SweepAxes::for_budget(SearchBudget::Quick, kinds.clone()),
            1,
        )
        .unwrap();
        let balanced = sweep_cost(
            &SweepAxes::for_budget(SearchBudget::Balanced, kinds.clone()),
            1,
        )
        .unwrap();
        let thorough =
            sweep_cost(&SweepAxes::for_budget(SearchBudget::Thorough, kinds), 1).unwrap();

        assert!(quick.configs < balanced.configs);
        assert!(balanced.configs < thorough.configs);
        // «Тщательно» — два seed, поэтому прогонов вдвое больше конфигураций.
        assert_eq!(thorough.seeds, 2);
        assert_eq!(thorough.runs, thorough.configs * 2);
        assert!(quick.epochs_upper_bound() < thorough.epochs_upper_bound());

        // K-fold умножает стоимость на число folds.
        let five = sweep_cost(
            &SweepAxes::for_budget(SearchBudget::Quick, vec![ModelKind::Mlp]),
            5,
        )
        .unwrap();
        let one = sweep_cost(
            &SweepAxes::for_budget(SearchBudget::Quick, vec![ModelKind::Mlp]),
            1,
        )
        .unwrap();
        assert_eq!(five.runs, one.runs * 5);
    }

    #[test]
    fn budget_grid_covers_only_requested_architectures() {
        let axes = SweepAxes::for_budget(SearchBudget::Quick, vec![ModelKind::Kan]);
        assert!(validate_axes(&axes).is_ok());
        assert_eq!(axes.model_kinds, vec![ModelKind::Kan]);
        let cancel = AtomicBool::new(false);
        let bb = blackbox::by_name("sum").unwrap();
        let data = bb.generate(64, 0);
        let specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
        let dataset = dataset_of(&data, &specs);
        let prepared = SplitPlan::default().prepare(dataset.data()).unwrap();
        let mut small = axes.clone();
        small.epochs = 1;
        let result = run_sweep(
            &dataset,
            &prepared.search,
            &small,
            SweepObjective::default(),
            &cancel,
            |_| {},
        )
        .unwrap();
        assert!(result.rows.iter().all(|r| r.choice.kind == ModelKind::Kan));
    }

    #[test]
    fn validate_rejects_empty_axes() {
        let mut axes = SweepAxes::default();
        axes.d_models.clear();
        assert!(validate_axes(&axes).is_err());
    }

    #[test]
    fn validate_rejects_duplicate_seeds_before_training() {
        let axes = SweepAxes {
            seeds: vec![7, 7],
            ..SweepAxes::default()
        };
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

    /// Копия данных со схемой: тестам удобнее строить `Dataset` из готовой пары.
    fn dataset_of(data: &NumericDataset, specs: &[FeatureSpec]) -> Dataset {
        let copy = data.gather(&(0..data.len()).collect::<Vec<_>>());
        let schema = ModelSchema::synthetic_from_specs(specs, data.outputs.ncols()).unwrap();
        Dataset::new(copy, schema).unwrap()
    }

    fn tiny_axes() -> SweepAxes {
        SweepAxes {
            model_kinds: vec![ModelKind::Transformer, ModelKind::Mlp, ModelKind::Kan],
            epochs: 1,
            batch_size: 32,
            d_models: vec![16],
            layers: vec![1],
            d_ffs: vec![32],
            lrs: vec![3e-3],
            value_encoders: vec![ValueEncoderKind::Linear],
            mlp_widths: vec![16],
            mlp_layers: vec![1],
            kan_widths: vec![8],
            kan_layers: vec![2],
            kan_grids: vec![5],
            ..SweepAxes::default()
        }
    }

    fn sweep_r2s(data: &NumericDataset, specs: &[FeatureSpec]) -> Vec<f32> {
        let dataset = dataset_of(data, specs);
        let prepared = SplitPlan::default().prepare(dataset.data()).unwrap();
        let cancel = AtomicBool::new(false);
        let result = run_sweep(
            &dataset,
            &prepared.search,
            &tiny_axes(),
            SweepObjective::WorstOutputR2,
            &cancel,
            |_| {},
        )
        .unwrap();
        result.rows.iter().map(|r| r.r2_mean).collect()
    }

    #[test]
    fn file_sweep_runs_all_model_kinds() {
        let bb = blackbox::by_name("sum").unwrap();
        let data = bb.generate(96, 0);
        let specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
        let dataset = dataset_of(&data, &specs);
        let prepared = SplitPlan::default().prepare(dataset.data()).unwrap();
        let cancel = AtomicBool::new(false);
        let result = run_sweep(
            &dataset,
            &prepared.search,
            &tiny_axes(),
            SweepObjective::WorstOutputR2,
            &cancel,
            |_| {},
        )
        .unwrap();
        assert_eq!(result.total_configs, 3);
        assert_eq!(result.rows.len(), 3);
        assert!(result.rows.iter().all(|r| r.r2_mean.is_finite()));
        // Ранжирование по validation — поиск не видит test даже по типу.
        assert!(result
            .rows
            .iter()
            .all(|r| r.source == EvalSource::Validation));
        let kan_row = result
            .rows
            .iter()
            .find(|r| r.choice.kind == ModelKind::Kan)
            .expect("kan-кандидат должен попасть в результаты");
        assert_eq!(kan_row.choice.kan.width, 8);
        assert_eq!(kan_row.choice.kan.grid, 5);
    }

    /// Ключевой тест Э1: порча целевых значений, попавших в test, не меняет
    /// результат поиска НИ НА БИТ. Разбиение детерминировано, поэтому строки
    /// test одни и те же в обоих прогонах.
    #[test]
    fn search_ignores_test_targets() {
        let bb = blackbox::by_name("sum").unwrap();
        let data = bb.generate(96, 0);
        let specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
        let baseline = sweep_r2s(&data, &specs);

        // Выясняем, какие строки ушли в test, и портим ИМЕННО их.
        let prepared = SplitPlan::default().prepare(&data).unwrap();
        let test_rows = prepared.test.len();
        let mut test_inputs = Vec::new();
        prepared
            .test
            .evaluate(
                |inputs| {
                    test_inputs = inputs.rows().into_iter().map(|r| r.to_vec()).collect();
                    Array2::zeros((inputs.nrows(), data.outputs.ncols()))
                },
                0,
            )
            .unwrap();

        let mut poisoned = NumericDataset::new(data.inputs.clone(), data.outputs.clone());
        let mut poisoned_rows = 0;
        for i in 0..poisoned.len() {
            let row: Vec<f32> = poisoned.inputs.row(i).to_vec();
            if test_inputs.contains(&row) {
                poisoned_rows += 1;
                for j in 0..poisoned.outputs.ncols() {
                    poisoned.outputs[[i, j]] = 1e6;
                }
            }
        }
        assert_eq!(poisoned_rows, test_rows, "испортили не те строки");

        assert_eq!(
            baseline,
            sweep_r2s(&poisoned, &specs),
            "порча test изменила результат поиска — это утечка"
        );
    }

    /// Обратная сторона: поиск обязан РЕАГИРОВАТЬ на validation. Проверяем не
    /// перестановку кандидатов (на реальных моделях она не гарантирована и тест
    /// был бы flaky), а сам факт зависимости метрик от validation-таргетов.
    #[test]
    fn search_depends_on_validation_targets() {
        let bb = blackbox::by_name("sum").unwrap();
        let data = bb.generate(96, 0);
        let specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
        let baseline = sweep_r2s(&data, &specs);

        let prepared = SplitPlan::default().prepare(&data).unwrap();
        let (_, val) = prepared.search.fold(0).unwrap();
        let val_inputs: Vec<Vec<f32>> = val.inputs.rows().into_iter().map(|r| r.to_vec()).collect();

        let mut poisoned = NumericDataset::new(data.inputs.clone(), data.outputs.clone());
        for i in 0..poisoned.len() {
            let row: Vec<f32> = poisoned.inputs.row(i).to_vec();
            if val_inputs.contains(&row) {
                for j in 0..poisoned.outputs.ncols() {
                    poisoned.outputs[[i, j]] = 1e6;
                }
            }
        }

        assert_ne!(
            baseline,
            sweep_r2s(&poisoned, &specs),
            "поиск не отреагировал на validation — значит меряет что-то другое"
        );
    }

    /// Ранжирование обязано переставляться, когда меняются validation-данные.
    /// На настоящих моделях это не гарантировано, поэтому проверяем на
    /// детерминированном фиктивном scorer-е: он ранжирует кандидатов по сумме
    /// validation-таргетов, и порядок меняется по построению.
    #[test]
    fn ranking_follows_validation_scores() {
        let inputs = Array2::from_shape_fn((40, 1), |(i, _)| i as f32);
        let outputs = Array2::from_shape_fn((40, 1), |(i, _)| (i % 4) as f32);
        let data = NumericDataset::new(inputs, outputs);
        let prepared = SplitPlan::default().prepare(&data).unwrap();
        let (_, val) = prepared.search.fold(0).unwrap();

        // «Кандидат» = множитель; счёт = корреляция с validation-таргетами.
        let score = |k: f32, val: &NumericDataset| -> f32 {
            val.outputs.iter().map(|&y| k * y).sum::<f32>()
        };
        let choice = build_candidates(&tiny_axes()).unwrap().remove(0).choice;
        let row = |label: &str, k: f32, validation: &NumericDataset| SweepRow {
            label: label.to_string(),
            choice: choice.clone(),
            r2_mean: score(k, validation),
            r2_std: 0.0,
            r2_std_folds: 0.0,
            worst_output_r2_mean: 0.0,
            mean_output_r2_mean: 0.0,
            nrmse_mean: 0.0,
            rel_mean: 0.0,
            source: EvalSource::Validation,
        };
        let mut ranked = vec![row("negative", -1.0, &val), row("positive", 2.0, &val)];
        sort_rows(&mut ranked, SweepObjective::AggregateR2);
        assert_eq!(ranked[0].label, "positive");

        // Меняем знак validation-таргетов -> порядок обязан перевернуться.
        let flipped = NumericDataset::new(val.inputs.clone(), -val.outputs.clone());
        let mut ranked = vec![
            row("negative", -1.0, &flipped),
            row("positive", 2.0, &flipped),
        ];
        sort_rows(&mut ranked, SweepObjective::AggregateR2);
        assert_eq!(ranked[0].label, "negative");
    }

    #[test]
    fn nrmse_is_averaged_per_run_not_derived_from_mean_r2() {
        let metric = |r2| crate::metrics::Metrics {
            rmse: 0.0,
            mae: 0.0,
            rel_error: 0.0,
            r2,
        };
        let runs = vec![
            RunEval {
                metrics: metric(0.0),
                per_output: vec![metric(0.0)],
                origin: RunOrigin {
                    fold: None,
                    init_seed: 0,
                },
            },
            RunEval {
                metrics: metric(1.0),
                per_output: vec![metric(1.0)],
                origin: RunOrigin {
                    fold: None,
                    init_seed: 1,
                },
            },
        ];
        let agg = aggregate_runs(&runs, &[0, 1], EvalSource::Validation).unwrap();
        let choice = build_candidates(&tiny_axes()).unwrap().remove(0).choice;
        let row = row_from_config_eval("synthetic".to_string(), choice, &agg, &runs);

        assert!((row.nrmse_mean - 0.5).abs() < 1e-6);
        assert!((row.nrmse_mean - (1.0_f32 - agg.mean.r2).sqrt()).abs() > 0.1);
    }

    /// Нормализаторы каждого fold строятся ТОЛЬКО по его train: статистики
    /// обязаны отличаться от статистик всего pool.
    #[test]
    fn fold_normalizers_use_only_fold_train() {
        let inputs = Array2::from_shape_fn((60, 1), |(i, _)| i as f32);
        let outputs = Array2::from_shape_fn((60, 1), |(i, _)| i as f32 * 2.0);
        let data = NumericDataset::new(inputs, outputs);
        let plan = SplitPlan::KFold {
            k: 3,
            folds_seed: 1,
            test_frac: 0.2,
            test_seed: 1,
        };
        let prepared = plan.prepare(&data).unwrap();
        let specs = vec![FeatureSpec::Continuous; 1];

        let pool = prepared.search.all();
        let (pool_norm, _) = fit_normalizers(&pool, &specs);
        let mut differ = 0;
        for i in 0..prepared.search.n_folds() {
            let (train, val) = prepared.search.fold(i).unwrap();
            let (fold_norm, _) = fit_normalizers(&train, &specs);
            // Статистики fold считаются по его train, а не по pool.
            let expected = train.inputs.iter().sum::<f32>() / train.len() as f32;
            assert!((fold_norm.mean[0] - expected).abs() < 1e-3);
            if (fold_norm.mean[0] - pool_norm.mean[0]).abs() > 1e-6 {
                differ += 1;
            }
            assert!(!val.is_empty());
        }
        assert!(
            differ > 0,
            "хотя бы один fold обязан иметь статистики, отличные от pool"
        );
    }
}
