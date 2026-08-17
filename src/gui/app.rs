//! egui-приложение: формы обучения, поиска, прогноза и диагностики.
//! UI только рендерит и общается с worker каналами; ML-состояние — в worker.

use super::messages::{
    Command, DataSource, DiagnosticsResult, Event, KanModelInfo, KanSymbolicInfo,
};
use super::worker::Worker;
use crate::config::ModelConfig;
use crate::data::{NumericDataset, OutOfRange};
use crate::encoders::{ValueEncoderConfig, ValueEncoderKind};
use crate::epoch_sweep::{self, EpochRow};
use crate::markup::{analyze_roles, DraftType, RoleReport, SchemaDraft, Severity, TableProfile};
use crate::metrics::Metrics;
use crate::numeric_model::{validate_numeric, KanConfig, ModelKind, NumericConfig};
use crate::schema::{ColumnRole, ColumnType, ModelSchema};
use crate::sweep::{self, SweepAxes, SweepChoice, SweepObjective, SweepRow};
use crate::table::Table;
use crate::tnum::{infer_prepare_spec_from_path, parse_categorical, Delimiter, PrepareSpec};
use crate::train::{validate_train, LrSchedule, TextTrainConfig, TrainConfig};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Train,
    Predict,
    KanCurves,
    KanFormulas,
    Diagnose,
    Optimize,
    Sweep,
    Text,
    Prepare,
    EpochSweep,
}

const BLACKBOXES: &[&str] = &["sum", "product", "sine", "polynomial", "projectile"];
const KAN_CURVE_SAMPLES: usize = 201;

/// Состояние формы Train. Числа редактируются `DragValue` (без строкового
/// парсинга); валидность проверяется теми же `validate_*`, что и в CLI.
/// Откуда берутся данные для обучения.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Blackbox,
    /// `.tnum` — схема уже подтверждена, разметка не нужна.
    Tnum,
    /// XLSX/CSV/TSV — обязательно через диалог разметки.
    Table,
}

struct TrainForm {
    source_kind: SourceKind,
    blackbox: String,
    file_path: String,
    /// Результат диалога разметки: данные и схема, а не путь.
    prepared: Option<PreparedTable>,
    kind: ModelKind,
    d_model: usize,
    heads: usize,
    layers: usize,
    d_ff: usize,
    venc: usize, // 0 linear, 1 mlp, 2 fourier
    fourier_bands: usize,
    fourier_scale: f32,
    mlp_width: usize,
    mlp_layers: usize,
    kan_width: usize,
    kan_layers: usize,
    kan_grid: usize,
    lr: f32,
    batch: usize,
    epochs: usize,
    seed: u64,
    warmup_cosine: bool,
    warmup: f32,
    min_lr_ratio: f32,
}

impl Default for TrainForm {
    fn default() -> Self {
        Self {
            source_kind: SourceKind::Blackbox,
            blackbox: "sum".to_string(),
            file_path: String::new(),
            prepared: None,
            kind: ModelKind::Transformer,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 64,
            venc: 0,
            fourier_bands: 6,
            fourier_scale: 8.0,
            mlp_width: 128,
            mlp_layers: 3,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
            lr: 1e-3,
            batch: 64,
            epochs: 40,
            seed: 0,
            warmup_cosine: false,
            warmup: 0.1,
            min_lr_ratio: 0.1,
        }
    }
}

impl TrainForm {
    fn build(&self) -> Result<(DataSource, NumericConfig, TrainConfig), String> {
        let source = match self.source_kind {
            SourceKind::Blackbox => DataSource::Blackbox(self.blackbox.clone()),
            SourceKind::Tnum => {
                if self.file_path.is_empty() {
                    return Err("укажите .tnum файл".to_string());
                }
                DataSource::File(self.file_path.clone())
            }
            SourceKind::Table => {
                let prepared = self
                    .prepared
                    .as_ref()
                    .ok_or_else(|| "откройте таблицу и подтвердите разметку".to_string())?;
                DataSource::Prepared {
                    name: prepared.name.clone(),
                    data: Arc::clone(&prepared.data),
                    schema: prepared.schema.clone(),
                }
            }
        };

        let value = ValueEncoderConfig {
            kind: match self.kind {
                ModelKind::Transformer => match self.venc {
                    0 => ValueEncoderKind::Linear,
                    1 => ValueEncoderKind::Mlp,
                    _ => ValueEncoderKind::Fourier,
                },
                ModelKind::Mlp | ModelKind::Kan => ValueEncoderKind::Linear,
            },
            fourier_bands: self.fourier_bands,
            fourier_scale: self.fourier_scale,
        };
        let nc = NumericConfig {
            kind: self.kind,
            transformer: ModelConfig {
                d_model: self.d_model,
                n_heads: self.heads,
                n_enc_layers: self.layers,
                n_dec_layers: self.layers,
                d_ff: self.d_ff,
                ln_eps: 1e-5,
            },
            value,
            mlp_width: self.mlp_width,
            mlp_layers: self.mlp_layers,
            kan: KanConfig {
                width: self.kan_width,
                layers: self.kan_layers,
                grid: self.kan_grid,
            },
        };
        validate_numeric(&nc)?;

        let schedule = if self.warmup_cosine {
            if !(0.0..1.0).contains(&self.warmup) {
                return Err("warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&self.min_lr_ratio) {
                return Err("min-lr-ratio должен быть в [0, 1]".to_string());
            }
            LrSchedule::WarmupCosine {
                warmup_frac: self.warmup,
                min_lr_ratio: self.min_lr_ratio,
            }
        } else {
            LrSchedule::Constant
        };
        let tcfg = TrainConfig {
            epochs: self.epochs,
            batch_size: self.batch,
            lr: self.lr,
            seed: self.seed,
            schedule,
        };
        validate_train(tcfg.lr, tcfg.batch_size)?;
        Ok((source, nc, tcfg))
    }
}

struct SweepForm {
    blackbox: String,
    seeds: String,
    d_models: String,
    layers: String,
    d_ffs: String,
    lrs: String,
    value_encoders: String,
    fourier_scales: String,
    fourier_bands: usize,
    schedulers: String,
    epochs: usize,
    batch: usize,
}

impl Default for SweepForm {
    fn default() -> Self {
        Self {
            blackbox: "sum".to_string(),
            seeds: "0".to_string(),
            d_models: "32".to_string(),
            layers: "2".to_string(),
            d_ffs: "64".to_string(),
            lrs: "0.001".to_string(),
            value_encoders: "linear".to_string(),
            fourier_scales: "2".to_string(),
            fourier_bands: 6,
            schedulers: "constant".to_string(),
            epochs: 30,
            batch: 64,
        }
    }
}

impl SweepForm {
    fn build(&self) -> Result<(String, SweepAxes), String> {
        let axes = SweepAxes {
            model_kinds: vec![ModelKind::Transformer],
            seeds: parse_csv_u64(&self.seeds, "seeds")?,
            d_models: parse_csv_usize(&self.d_models, "d-models")?,
            layers: parse_csv_usize(&self.layers, "layers-list")?,
            d_ffs: parse_csv_usize(&self.d_ffs, "d-ffs")?,
            lrs: parse_csv_f32(&self.lrs, "lrs")?,
            value_encoders: parse_value_encoders(&self.value_encoders)?,
            fourier_scales: parse_csv_f32(&self.fourier_scales, "fourier-scales")?,
            fourier_bands: self.fourier_bands,
            mlp_widths: vec![128],
            mlp_layers: vec![3],
            kan_widths: vec![16],
            kan_layers: vec![2],
            kan_grids: vec![8],
            schedules: parse_schedules(&self.schedulers)?,
            epochs: self.epochs,
            final_epochs: self.epochs,
            batch_size: self.batch,
        };
        sweep::validate_axes(&axes)?;
        Ok((self.blackbox.clone(), axes))
    }
}

struct OptimizeForm {
    file_path: String,
    preset: usize,    // 0 quick, 1 balanced, 2 deep
    objective: usize, // 0 worst, 1 aggregate, 2 mean, 3 nrmse
    include_mlp: bool,
    include_transformer: bool,
    include_kan: bool,
}

impl Default for OptimizeForm {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            preset: 0,
            objective: 0,
            include_mlp: true,
            include_transformer: true,
            include_kan: true,
        }
    }
}

impl OptimizeForm {
    fn objective(&self) -> SweepObjective {
        match self.objective {
            1 => SweepObjective::AggregateR2,
            2 => SweepObjective::MeanOutputR2,
            3 => SweepObjective::Nrmse,
            _ => SweepObjective::WorstOutputR2,
        }
    }

    /// Оси сетки из пресета (без требования файла) — для оценки размера в UI.
    fn axes(&self) -> Result<SweepAxes, String> {
        let mut model_kinds = Vec::new();
        if self.include_transformer {
            model_kinds.push(ModelKind::Transformer);
        }
        if self.include_mlp {
            model_kinds.push(ModelKind::Mlp);
        }
        if self.include_kan {
            model_kinds.push(ModelKind::Kan);
        }
        if model_kinds.is_empty() {
            return Err("выберите хотя бы одну архитектуру (transformer/mlp/kan)".to_string());
        }

        let axes = match self.preset {
            1 => SweepAxes {
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
                schedules: vec![LrSchedule::WarmupCosine {
                    warmup_frac: 0.1,
                    min_lr_ratio: 0.1,
                }],
                epochs: 40,
                final_epochs: 80,
                batch_size: 64,
            },
            2 => SweepAxes {
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
                schedules: vec![LrSchedule::WarmupCosine {
                    warmup_frac: 0.1,
                    min_lr_ratio: 0.1,
                }],
                epochs: 40,
                final_epochs: 80,
                batch_size: 64,
            },
            _ => SweepAxes {
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
                schedules: vec![LrSchedule::WarmupCosine {
                    warmup_frac: 0.1,
                    min_lr_ratio: 0.1,
                }],
                epochs: 25,
                final_epochs: 60,
                batch_size: 64,
            },
        };
        sweep::validate_axes(&axes)?;
        Ok(axes)
    }

    fn build(&self) -> Result<(String, SweepAxes, SweepObjective), String> {
        if self.file_path.is_empty() {
            return Err("выберите .tnum файл".to_string());
        }
        let axes = self.axes()?;
        Ok((self.file_path.clone(), axes, self.objective()))
    }
}

struct TextForm {
    file_path: String,
    d_model: usize,
    heads: usize,
    layers: usize,
    d_ff: usize,
    steps: usize,
    batch: usize,
    ctx_len: usize,
    tgt_len: usize,
    lr: f32,
    seed: u64,
    seed_text: String,
    total_new: usize,
    temperature: f32,
    top_k: usize,
    gen_seed: u64,
}

impl Default for TextForm {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            d_model: 64,
            heads: 4,
            layers: 2,
            d_ff: 128,
            steps: 500,
            batch: 32,
            ctx_len: 32,
            tgt_len: 32,
            lr: 1e-3,
            seed: 0,
            seed_text: String::new(),
            total_new: 400,
            temperature: 0.8,
            top_k: 10,
            gen_seed: 42,
        }
    }
}

impl TextForm {
    fn build(&self) -> Result<(String, ModelConfig, TextTrainConfig), String> {
        if self.file_path.is_empty() {
            return Err("выберите текстовый файл".to_string());
        }
        let model_cfg = ModelConfig {
            d_model: self.d_model,
            n_heads: self.heads,
            n_enc_layers: self.layers,
            n_dec_layers: self.layers,
            d_ff: self.d_ff,
            ln_eps: 1e-5,
        };
        if model_cfg.d_model == 0
            || model_cfg.d_ff == 0
            || model_cfg.n_heads == 0
            || !model_cfg.d_model.is_multiple_of(model_cfg.n_heads)
            || model_cfg.n_enc_layers == 0
            || model_cfg.n_dec_layers == 0
        {
            return Err("некорректный text model config".to_string());
        }
        let train_cfg = TextTrainConfig {
            steps: self.steps,
            batch_size: self.batch,
            ctx_len: self.ctx_len,
            tgt_len: self.tgt_len,
            lr: self.lr,
            seed: self.seed,
        };
        if train_cfg.steps == 0
            || train_cfg.batch_size == 0
            || train_cfg.ctx_len == 0
            || train_cfg.tgt_len == 0
            || !train_cfg.lr.is_finite()
            || train_cfg.lr <= 0.0
        {
            return Err("некорректный text train config".to_string());
        }
        Ok((self.file_path.clone(), model_cfg, train_cfg))
    }
}

struct PrepareForm {
    input_path: String,
    output_path: String,
    inputs: usize,
    outputs: usize,
    delimiter: usize,
    has_header: bool,
    categorical: String,
}

impl Default for PrepareForm {
    fn default() -> Self {
        Self {
            input_path: String::new(),
            output_path: String::new(),
            inputs: 2,
            outputs: 1,
            delimiter: 0,
            has_header: false,
            categorical: String::new(),
        }
    }
}

impl PrepareForm {
    fn build(&self) -> Result<(String, String, PrepareSpec), String> {
        if self.input_path.is_empty() {
            return Err("выберите входную таблицу".to_string());
        }
        if self.output_path.is_empty() {
            return Err("укажите выходной .tnum".to_string());
        }
        let categorical = parse_categorical(&self.categorical, self.inputs)?;
        let delimiter = match self.delimiter {
            1 => Delimiter::Comma,
            2 => Delimiter::Tab,
            3 => Delimiter::Space,
            _ => Delimiter::Auto,
        };
        Ok((
            self.input_path.clone(),
            self.output_path.clone(),
            PrepareSpec {
                n_inputs: self.inputs,
                n_outputs: self.outputs,
                delimiter,
                has_header: self.has_header,
                categorical,
            },
        ))
    }
}

struct EpochSweepForm {
    file_path: String,
    epochs: String,
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
    kind: ModelKind,
    d_model: usize,
    heads: usize,
    layers: usize,
    d_ff: usize,
    venc: usize,
    fourier_bands: usize,
    fourier_scale: f32,
    mlp_width: usize,
    mlp_layers: usize,
    kan_width: usize,
    kan_layers: usize,
    kan_grid: usize,
    lr: f32,
    batch: usize,
    seed: u64,
    warmup_cosine: bool,
    warmup: f32,
    min_lr_ratio: f32,
}

struct EpochSweepRequest {
    path: String,
    nc: NumericConfig,
    base_tcfg: TrainConfig,
    milestones: Vec<usize>,
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
}

impl Default for EpochSweepForm {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            epochs: "1,2,5,10,20,40".to_string(),
            target_r2: 0.95,
            min_gain: 0.02,
            plateau_min: 0.80,
            kind: ModelKind::Transformer,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 64,
            venc: 0,
            fourier_bands: 6,
            fourier_scale: 8.0,
            mlp_width: 128,
            mlp_layers: 3,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
            lr: 1e-3,
            batch: 64,
            seed: 0,
            warmup_cosine: false,
            warmup: 0.1,
            min_lr_ratio: 0.1,
        }
    }
}

impl EpochSweepForm {
    fn build(&self) -> Result<EpochSweepRequest, String> {
        if self.file_path.is_empty() {
            return Err("выберите .tnum файл".to_string());
        }
        let milestones = parse_csv_usize(&self.epochs, "epochs")?;
        if milestones.is_empty() || milestones.contains(&0) {
            return Err("epochs: список должен быть непустым и > 0".to_string());
        }
        if !self.target_r2.is_finite() {
            return Err("target-r2 должен быть конечным".to_string());
        }
        if !self.min_gain.is_finite() || self.min_gain < 0.0 {
            return Err("min-r2-gain должен быть конечным и >= 0".to_string());
        }
        if !self.plateau_min.is_finite() {
            return Err("plateau-min-r2 должен быть конечным".to_string());
        }

        let value = ValueEncoderConfig {
            kind: match self.kind {
                ModelKind::Transformer => match self.venc {
                    0 => ValueEncoderKind::Linear,
                    1 => ValueEncoderKind::Mlp,
                    _ => ValueEncoderKind::Fourier,
                },
                ModelKind::Mlp | ModelKind::Kan => ValueEncoderKind::Linear,
            },
            fourier_bands: self.fourier_bands,
            fourier_scale: self.fourier_scale,
        };
        let nc = NumericConfig {
            kind: self.kind,
            transformer: ModelConfig {
                d_model: self.d_model,
                n_heads: self.heads,
                n_enc_layers: self.layers,
                n_dec_layers: self.layers,
                d_ff: self.d_ff,
                ln_eps: 1e-5,
            },
            value,
            mlp_width: self.mlp_width,
            mlp_layers: self.mlp_layers,
            kan: KanConfig {
                width: self.kan_width,
                layers: self.kan_layers,
                grid: self.kan_grid,
            },
        };
        validate_numeric(&nc)?;

        let schedule = if self.warmup_cosine {
            if !(0.0..1.0).contains(&self.warmup) {
                return Err("warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&self.min_lr_ratio) {
                return Err("min-lr-ratio должен быть в [0, 1]".to_string());
            }
            LrSchedule::WarmupCosine {
                warmup_frac: self.warmup,
                min_lr_ratio: self.min_lr_ratio,
            }
        } else {
            LrSchedule::Constant
        };
        let max_epoch = milestones.iter().copied().max().unwrap_or(1);
        let tcfg = TrainConfig {
            epochs: max_epoch,
            batch_size: self.batch,
            lr: self.lr,
            seed: self.seed,
            schedule,
        };
        validate_train(tcfg.lr, tcfg.batch_size)?;
        Ok(EpochSweepRequest {
            path: self.file_path.clone(),
            nc,
            base_tcfg: tcfg,
            milestones,
            target_r2: self.target_r2,
            min_gain: self.min_gain,
            plateau_min: self.plateau_min,
        })
    }
}

fn split_csv(raw: &str, name: &str) -> Result<Vec<String>, String> {
    let parts: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if parts.is_empty() {
        return Err(format!("{name}: пустой список"));
    }
    Ok(parts)
}

fn parse_csv_usize(raw: &str, name: &str) -> Result<Vec<usize>, String> {
    split_csv(raw, name)?
        .into_iter()
        .map(|p| {
            p.parse()
                .map_err(|_| format!("{name}: '{p}' не целое число"))
        })
        .collect()
}

fn parse_csv_u64(raw: &str, name: &str) -> Result<Vec<u64>, String> {
    split_csv(raw, name)?
        .into_iter()
        .map(|p| {
            p.parse()
                .map_err(|_| format!("{name}: '{p}' не целое число"))
        })
        .collect()
}

fn parse_csv_f32(raw: &str, name: &str) -> Result<Vec<f32>, String> {
    split_csv(raw, name)?
        .into_iter()
        .map(|p| p.parse().map_err(|_| format!("{name}: '{p}' не число")))
        .collect()
}

fn parse_value_encoders(raw: &str) -> Result<Vec<ValueEncoderKind>, String> {
    split_csv(raw, "value-encoders")?
        .into_iter()
        .map(|p| match p.as_str() {
            "linear" => Ok(ValueEncoderKind::Linear),
            "mlp" => Ok(ValueEncoderKind::Mlp),
            "fourier" => Ok(ValueEncoderKind::Fourier),
            _ => Err(format!("value-encoders: '{p}' не linear|mlp|fourier")),
        })
        .collect()
}

fn parse_schedules(raw: &str) -> Result<Vec<LrSchedule>, String> {
    split_csv(raw, "schedulers")?
        .into_iter()
        .map(|p| match p.as_str() {
            "constant" => Ok(LrSchedule::Constant),
            "warmup-cosine" => Ok(LrSchedule::WarmupCosine {
                warmup_frac: 0.1,
                min_lr_ratio: 0.1,
            }),
            _ => Err(format!("schedulers: '{p}' не constant|warmup-cosine")),
        })
        .collect()
}

fn objective_label(objective: SweepObjective) -> &'static str {
    match objective {
        SweepObjective::WorstOutputR2 => "worst-output R²",
        SweepObjective::AggregateR2 => "aggregate R²",
        SweepObjective::MeanOutputR2 => "mean-output R²",
        SweepObjective::Nrmse => "aggregate nRMSE",
    }
}

fn objective_display_score(objective: SweepObjective, row: &SweepRow) -> f32 {
    match objective {
        SweepObjective::Nrmse => row.nrmse_mean,
        _ => objective.score(row),
    }
}

/// Подтверждённая разметка таблицы: готовые данные и схема.
#[derive(Clone)]
struct PreparedTable {
    name: String,
    path: String,
    has_header: bool,
    data: Arc<NumericDataset>,
    schema: ModelSchema,
}

/// Состояние диалога разметки. Профиль считается один раз при открытии файла;
/// пересчёт отчёта по ролям — только при смене ролей и типов.
struct MarkupState {
    path: String,
    has_header: bool,
    table: Table,
    profile: TableProfile,
    draft: SchemaDraft,
    report: RoleReport,
    issues: Vec<String>,
    /// Ошибка подтверждения: разметка валидна, но данные ей не соответствуют.
    apply_error: Option<String>,
}

impl MarkupState {
    fn new(
        path: String,
        has_header: bool,
        table: Table,
        profile: TableProfile,
        suggested_inputs: Option<usize>,
        suggested_categories: &[usize],
    ) -> Self {
        let mut draft = SchemaDraft::from_profile(&profile);
        // Автоопределение только заполняет начальные роли — решение за
        // пользователем, поэтому неудача здесь не является ошибкой.
        if let Some(n_inputs) = suggested_inputs {
            let _ = draft.set_output_split(n_inputs);
        }
        for &index in suggested_categories {
            let _ = draft.set_type(index, DraftType::Categorical);
        }
        let report = analyze_roles(&table, &draft);
        let issues = draft.issues();
        Self {
            path,
            has_header,
            table,
            profile,
            draft,
            report,
            issues,
            apply_error: None,
        }
    }

    /// Роли и типы меняют связи между колонками — отчёт пересчитывается.
    fn on_roles_changed(&mut self) {
        self.report = analyze_roles(&self.table, &self.draft);
        self.on_any_change();
    }

    /// Имя и единица на анализ не влияют: отчёт хранит индексы и подставляет
    /// актуальные имена при выводе.
    fn on_any_change(&mut self) {
        self.issues = self.draft.issues();
        self.apply_error = None;
    }

    /// Подтвердить разметку: данные превращаются в датасет ЗДЕСЬ, чтобы
    /// обучение получило готовую пару, а не путь к файлу.
    fn apply(&self) -> Result<PreparedTable, String> {
        if let Some(message) = self
            .profile
            .messages()
            .into_iter()
            .find(|message| message.severity == Severity::Blocking)
        {
            return Err(message.text);
        }
        let schema = self.draft.finish()?;
        let data = self.table.to_dataset(&schema)?;
        Ok(PreparedTable {
            name: std::path::Path::new(&self.path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| self.path.clone()),
            path: self.path.clone(),
            has_header: self.has_header,
            data: Arc::new(data),
            schema: schema.to_model_schema()?,
        })
    }

    fn can_apply(&self) -> bool {
        self.issues.is_empty()
            && self.apply_error.is_none()
            && !self
                .profile
                .messages()
                .iter()
                .any(|message| message.severity == Severity::Blocking)
    }
}

/// Активная модель глазами UI. Схема — источник истины про число входов и
/// выходов и про их имена, поэтому отдельных счётчиков рядом нет.
#[derive(Clone)]
struct ModelInfo {
    schema: ModelSchema,
    kind: ModelKind,
    source: String,
    parameter_count: usize,
}

impl ModelInfo {
    /// MLP и KAN получают код категории как обычное число и воспринимают
    /// порядок кодов как расстояние. Embedding категорий есть только у
    /// transformer, поэтому здесь предупреждение, а не молчание.
    fn categorical_warning(&self) -> Option<String> {
        if self.kind == ModelKind::Transformer {
            return None;
        }
        let names: Vec<&str> = self
            .schema
            .inputs()
            .iter()
            .filter(|c| c.cardinality().is_some())
            .map(|c| c.name())
            .collect();
        if names.is_empty() {
            return None;
        }
        Some(format!(
            "⚠ категориальные входы ({}) кодируются числами: порядок кодов будет \
             воспринят как расстояние. Embedding категорий есть только у transformer.",
            names.join(", ")
        ))
    }
}

pub struct App {
    worker: Worker,
    tab: Tab,
    status: String,
    form: TrainForm,
    training: bool,
    sweeping: bool,
    loss_curve: Vec<[f64; 2]>,
    metrics: Option<Metrics>,
    train_parameter_count: Option<usize>,
    // Predict (UI-M5)
    model_info: Option<ModelInfo>,
    /// Worker читает и профилирует таблицу. Пока ответ не пришёл, нельзя
    /// запустить действие со старым `prepared`.
    table_opening: bool,
    /// Открытый диалог разметки таблицы.
    markup: Option<MarkupState>,
    predict_inputs: Vec<f32>,
    predict_outputs: Option<Vec<f32>>,
    extrapolation: Vec<OutOfRange>,
    batch_predicting: bool,
    // KAN curves (данные графика приходят из worker, не тензоры)
    kan_info: Option<KanModelInfo>,
    kan_layer: usize,
    kan_input: usize,
    kan_output: usize,
    kan_curve: Vec<[f64; 2]>,
    // KAN symbolic formulas (worker возвращает только текст и метрики)
    kan_symbolic: Option<KanSymbolicInfo>,
    kan_symbolic_pending: bool,
    // Diagnose (UI-M6)
    diagnostics: Option<DiagnosticsResult>,
    // Optimize (file-based sweep)
    optimize_form: OptimizeForm,
    optimizing: bool,
    optimize_rows: Vec<SweepRow>,
    optimize_total: Option<(usize, usize)>,
    optimize_cancelled: bool,
    // Sweep (UI-M6)
    sweep_form: SweepForm,
    sweep_rows: Vec<SweepRow>,
    sweep_total: Option<(usize, usize)>,
    sweep_cancelled: bool,
    // Text (UI-M7)
    text_form: TextForm,
    text_training: bool,
    text_curve: Vec<[f64; 2]>,
    text_ready: bool,
    text_vocab_size: Option<usize>,
    generated_text: String,
    // Prepare / Epoch-sweep (UI-M8)
    prepare_form: PrepareForm,
    epoch_form: EpochSweepForm,
    epoch_sweeping: bool,
    epoch_rows: Vec<EpochRow>,
    epoch_total: Option<usize>,
    epoch_recommendation: Option<(usize, String)>,
    epoch_cancelled: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            worker: Worker::spawn(cc.egui_ctx.clone()),
            tab: Tab::Train,
            status: "–".to_string(),
            form: TrainForm::default(),
            training: false,
            sweeping: false,
            loss_curve: Vec::new(),
            metrics: None,
            train_parameter_count: None,
            model_info: None,
            table_opening: false,
            markup: None,
            predict_inputs: Vec::new(),
            predict_outputs: None,
            extrapolation: Vec::new(),
            batch_predicting: false,
            kan_info: None,
            kan_layer: 0,
            kan_input: 0,
            kan_output: 0,
            kan_curve: Vec::new(),
            kan_symbolic: None,
            kan_symbolic_pending: false,
            diagnostics: None,
            optimize_form: OptimizeForm::default(),
            optimizing: false,
            optimize_rows: Vec::new(),
            optimize_total: None,
            optimize_cancelled: false,
            sweep_form: SweepForm::default(),
            sweep_rows: Vec::new(),
            sweep_total: None,
            sweep_cancelled: false,
            text_form: TextForm::default(),
            text_training: false,
            text_curve: Vec::new(),
            text_ready: false,
            text_vocab_size: None,
            generated_text: String::new(),
            prepare_form: PrepareForm::default(),
            epoch_form: EpochSweepForm::default(),
            epoch_sweeping: false,
            epoch_rows: Vec::new(),
            epoch_total: None,
            epoch_recommendation: None,
            epoch_cancelled: false,
        }
    }

    fn drain_events(&mut self) {
        while let Some(ev) = self.worker.try_recv() {
            match ev {
                Event::Status(s) => self.status = s,
                Event::Error(e) => {
                    self.training = false;
                    self.sweeping = false;
                    self.optimizing = false;
                    self.text_training = false;
                    self.epoch_sweeping = false;
                    self.batch_predicting = false;
                    self.kan_symbolic_pending = false;
                    self.table_opening = false;
                    self.status = format!("Ошибка: {e}");
                }
                Event::TrainStarted {
                    total_epochs,
                    parameter_count,
                } => {
                    self.training = true;
                    self.loss_curve.clear();
                    self.metrics = None;
                    self.train_parameter_count = Some(parameter_count);
                    self.status =
                        format!("обучение: 0/{total_epochs} эпох, {parameter_count} параметров");
                }
                Event::Epoch { epoch, loss } => {
                    self.loss_curve.push([epoch as f64, loss as f64]);
                    self.status = format!("эпоха {epoch}: loss {loss:.5}");
                }
                Event::TrainDone { metrics } => {
                    self.training = false;
                    match metrics {
                        Some(m) => {
                            self.metrics = Some(m);
                            self.status = "обучение завершено".to_string();
                        }
                        None => self.status = "обучение отменено".to_string(),
                    }
                }
                Event::TableOpened {
                    path,
                    has_header,
                    table,
                    profile,
                    suggested_inputs,
                    suggested_categories,
                } => {
                    self.table_opening = false;
                    self.status = format!("таблица открыта: {path}");
                    self.markup = Some(MarkupState::new(
                        path,
                        has_header,
                        *table,
                        *profile,
                        suggested_inputs,
                        &suggested_categories,
                    ));
                }
                Event::ModelReady {
                    schema,
                    kind,
                    source,
                    parameter_count,
                    kan,
                } => {
                    let n_inputs = schema.n_inputs();
                    self.model_info = Some(ModelInfo {
                        schema,
                        kind,
                        source,
                        parameter_count,
                    });
                    self.predict_inputs = vec![0.0; n_inputs];
                    self.predict_outputs = None;
                    self.extrapolation.clear();
                    self.kan_info = kan;
                    self.kan_layer = 0;
                    self.kan_input = 0;
                    self.kan_output = 0;
                    self.kan_curve.clear();
                    self.kan_symbolic = None;
                    self.kan_symbolic_pending = false;
                    self.request_kan_curve();
                    self.status = "модель готова к предсказанию".to_string();
                }
                Event::PredictResult {
                    outputs,
                    extrapolation,
                } => {
                    self.predict_outputs = Some(outputs);
                    self.extrapolation = extrapolation;
                }
                Event::PredictFileDone {
                    output,
                    rows,
                    extrapolation_rows,
                } => {
                    self.batch_predicting = false;
                    self.status = if extrapolation_rows == 0 {
                        format!("Excel заполнен: {output} ({rows} строк)")
                    } else {
                        format!(
                            "Excel заполнен: {output} ({rows} строк, {extrapolation_rows} вне train-диапазона)"
                        )
                    };
                }
                Event::KanEdgeCurve {
                    layer,
                    input,
                    output,
                    points,
                } => {
                    if (layer, input, output) == (self.kan_layer, self.kan_input, self.kan_output) {
                        self.kan_curve = points
                            .into_iter()
                            .map(|(x, y)| [x as f64, y as f64])
                            .collect();
                    }
                }
                Event::KanSymbolic { result } => {
                    self.kan_symbolic = Some(result);
                    self.kan_symbolic_pending = false;
                    self.status = "символьные формулы готовы".to_string();
                }
                Event::Diagnostics { result } => {
                    self.diagnostics = Some(result);
                    self.status = "диагностика готова".to_string();
                }
                Event::SweepStarted {
                    total_configs,
                    total_runs,
                } => {
                    self.sweeping = true;
                    self.sweep_rows.clear();
                    self.sweep_total = Some((total_configs, total_runs));
                    self.sweep_cancelled = false;
                    self.status =
                        format!("sweep: 0/{total_configs} конфигов ({total_runs} прогонов)");
                }
                Event::SweepRow { row } => {
                    self.sweep_rows.push(row);
                    self.sweep_rows.sort_by(|a, b| {
                        b.r2_mean
                            .partial_cmp(&a.r2_mean)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if let Some((total_configs, _)) = self.sweep_total {
                        self.status =
                            format!("sweep: {}/{total_configs} конфигов", self.sweep_rows.len());
                    }
                }
                Event::SweepDone { rows, cancelled } => {
                    self.sweeping = false;
                    self.sweep_rows = rows;
                    self.sweep_cancelled = cancelled;
                    self.status = if cancelled {
                        "sweep отменён".to_string()
                    } else {
                        "sweep завершён".to_string()
                    };
                }
                Event::OptimizeStarted {
                    total_configs,
                    total_runs,
                } => {
                    self.optimizing = true;
                    self.optimize_rows.clear();
                    self.optimize_total = Some((total_configs, total_runs));
                    self.optimize_cancelled = false;
                    self.status =
                        format!("optimize: 0/{total_configs} конфигов ({total_runs} прогонов)");
                }
                Event::OptimizeRow { row } => {
                    self.optimize_rows.push(row);
                    self.sort_optimize_rows();
                    if let Some((total_configs, _)) = self.optimize_total {
                        self.status = format!(
                            "optimize: {}/{total_configs} конфигов",
                            self.optimize_rows.len()
                        );
                    }
                }
                Event::OptimizeDone { rows, cancelled } => {
                    self.optimizing = false;
                    self.optimize_rows = rows;
                    self.sort_optimize_rows();
                    self.optimize_cancelled = cancelled;
                    self.status = if cancelled {
                        "optimize отменён".to_string()
                    } else {
                        "optimize завершён".to_string()
                    };
                }
                Event::TextStarted { total_steps } => {
                    self.text_training = true;
                    self.text_ready = false;
                    self.text_curve.clear();
                    self.generated_text.clear();
                    self.text_vocab_size = None;
                    self.status = format!("text: 0/{total_steps} шагов");
                }
                Event::TextProgress { step, loss } => {
                    self.text_curve.push([step as f64, loss.exp() as f64]);
                    self.status = format!("text шаг {step}: loss {loss:.4}, ppl {:.2}", loss.exp());
                }
                Event::TextDone {
                    final_loss,
                    cancelled,
                    vocab_size,
                    seed_hint,
                } => {
                    self.text_training = false;
                    self.text_vocab_size = Some(vocab_size);
                    if !cancelled {
                        self.text_ready = true;
                        if self.text_form.seed_text.is_empty() {
                            self.text_form.seed_text = seed_hint;
                        }
                    }
                    self.status = match (cancelled, final_loss) {
                        (true, _) => "text обучение отменено".to_string(),
                        (false, Some(loss)) => {
                            format!("text готов: loss {loss:.4}, ppl {:.2}", loss.exp())
                        }
                        (false, None) => "text готов".to_string(),
                    };
                }
                Event::GeneratedText { text } => {
                    self.generated_text = text;
                    self.status = "генерация готова".to_string();
                }
                Event::PrepareDone {
                    output,
                    rows,
                    n_inputs,
                    n_outputs,
                } => {
                    self.status = format!(
                        "записано {output}: {rows} строк, {n_inputs} вход -> {n_outputs} выход"
                    );
                }
                Event::EpochSweepStarted { total_points } => {
                    self.epoch_sweeping = true;
                    self.epoch_rows.clear();
                    self.epoch_total = Some(total_points);
                    self.epoch_recommendation = None;
                    self.epoch_cancelled = false;
                    self.status = format!("epoch-sweep: 0/{total_points}");
                }
                Event::EpochSweepRow { row } => {
                    self.epoch_rows.push(row);
                    if let Some(total) = self.epoch_total {
                        self.status = format!("epoch-sweep: {}/{total}", self.epoch_rows.len());
                    }
                }
                Event::EpochSweepDone {
                    rows,
                    recommendation,
                    cancelled,
                } => {
                    self.epoch_sweeping = false;
                    self.epoch_rows = rows;
                    self.epoch_recommendation = recommendation;
                    self.epoch_cancelled = cancelled;
                    self.status = if cancelled {
                        "epoch-sweep отменён".to_string()
                    } else {
                        "epoch-sweep завершён".to_string()
                    };
                }
            }
        }
    }

    fn busy(&self) -> bool {
        self.training
            || self.sweeping
            || self.optimizing
            || self.text_training
            || self.epoch_sweeping
            || self.batch_predicting
            || self.kan_symbolic_pending
            || self.table_opening
            || self.markup.is_some()
    }

    fn request_kan_curve(&mut self) {
        let Some((n_inputs, n_outputs)) = self
            .kan_info
            .as_ref()
            .and_then(|info| info.layer_dims.get(self.kan_layer).copied())
        else {
            return;
        };
        self.kan_input = self.kan_input.min(n_inputs - 1);
        self.kan_output = self.kan_output.min(n_outputs - 1);
        self.kan_curve.clear();
        self.worker.send(Command::SampleKanEdge {
            layer: self.kan_layer,
            input: self.kan_input,
            output: self.kan_output,
            samples: KAN_CURVE_SAMPLES,
        });
    }

    fn sort_optimize_rows(&mut self) {
        let objective = self.optimize_form.objective();
        self.optimize_rows.sort_by(|a, b| {
            objective
                .score(b)
                .partial_cmp(&objective.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn open_table(&mut self, path: String, has_header: bool) {
        self.table_opening = true;
        self.status = format!("чтение {path}…");
        self.worker.send(Command::OpenTable { path, has_header });
    }

    fn apply_choice_to_train(&mut self, choice: &SweepChoice) {
        self.form.source_kind = SourceKind::Tnum;
        self.form.file_path = self.optimize_form.file_path.clone();
        self.form.kind = choice.kind;
        self.form.kan_width = choice.kan.width;
        self.form.kan_layers = choice.kan.layers;
        self.form.kan_grid = choice.kan.grid;
        self.form.d_model = choice.d_model;
        self.form.heads = choice.heads;
        self.form.layers = choice.layers;
        self.form.d_ff = choice.d_ff;
        self.form.venc = match choice.value.kind {
            ValueEncoderKind::Linear => 0,
            ValueEncoderKind::Mlp => 1,
            ValueEncoderKind::Fourier => 2,
        };
        self.form.fourier_bands = choice.value.fourier_bands;
        self.form.fourier_scale = choice.value.fourier_scale;
        self.form.mlp_width = choice.mlp_width;
        self.form.mlp_layers = choice.mlp_layers;
        self.form.lr = choice.lr;
        self.form.batch = choice.batch_size;
        // Двухфазность: Optimize ранжировал на search-эпохах, а Train делает
        // финальное обучение на полном бюджете.
        self.form.epochs = choice.final_epochs;
        self.form.seed = choice.seed;
        match choice.schedule {
            LrSchedule::Constant => {
                self.form.warmup_cosine = false;
            }
            LrSchedule::WarmupCosine {
                warmup_frac,
                min_lr_ratio,
            } => {
                self.form.warmup_cosine = true;
                self.form.warmup = warmup_frac;
                self.form.min_lr_ratio = min_lr_ratio;
            }
        }
        self.tab = Tab::Train;
        self.status = "лучший конфиг применён во вкладке Train".to_string();
    }

    /// Переносит лучший конфиг Optimize в Epoch-sweep, чтобы подобрать число
    /// эпох. Конфиг тот же; эпохи Optimize не переносим — задаём список для
    /// прохода.
    fn apply_choice_to_epoch_sweep(&mut self, choice: &SweepChoice) {
        self.epoch_form.file_path = self.optimize_form.file_path.clone();
        self.epoch_form.kind = choice.kind;
        self.epoch_form.kan_width = choice.kan.width;
        self.epoch_form.kan_layers = choice.kan.layers;
        self.epoch_form.kan_grid = choice.kan.grid;
        self.epoch_form.d_model = choice.d_model;
        self.epoch_form.heads = choice.heads;
        self.epoch_form.layers = choice.layers;
        self.epoch_form.d_ff = choice.d_ff;
        self.epoch_form.venc = match choice.value.kind {
            ValueEncoderKind::Linear => 0,
            ValueEncoderKind::Mlp => 1,
            ValueEncoderKind::Fourier => 2,
        };
        self.epoch_form.fourier_bands = choice.value.fourier_bands;
        self.epoch_form.fourier_scale = choice.value.fourier_scale;
        self.epoch_form.mlp_width = choice.mlp_width;
        self.epoch_form.mlp_layers = choice.mlp_layers;
        self.epoch_form.lr = choice.lr;
        self.epoch_form.batch = choice.batch_size;
        self.epoch_form.seed = choice.seed;
        match choice.schedule {
            LrSchedule::Constant => {
                self.epoch_form.warmup_cosine = false;
            }
            LrSchedule::WarmupCosine {
                warmup_frac,
                min_lr_ratio,
            } => {
                self.epoch_form.warmup_cosine = true;
                self.epoch_form.warmup = warmup_frac;
                self.epoch_form.min_lr_ratio = min_lr_ratio;
            }
        }
        self.epoch_form.epochs = "20,40,60,80,120".to_string();
        self.epoch_rows.clear();
        self.epoch_total = None;
        self.epoch_recommendation = None;
        self.epoch_cancelled = false;
        self.tab = Tab::EpochSweep;
        self.status = "конфиг перенесён в Epoch-sweep – запусти подбор эпох".to_string();
    }

    /// Переносит текущий конфиг Epoch-sweep и рекомендованное число эпох в
    /// Train. Замыкает поток Optimize → Check epochs → Train.
    fn apply_epoch_form_to_train(&mut self, epochs: usize) {
        let f = &self.epoch_form;
        self.form.source_kind = SourceKind::Tnum;
        self.form.file_path = f.file_path.clone();
        self.form.kind = f.kind;
        self.form.d_model = f.d_model;
        self.form.heads = f.heads;
        self.form.layers = f.layers;
        self.form.d_ff = f.d_ff;
        self.form.venc = f.venc;
        self.form.fourier_bands = f.fourier_bands;
        self.form.fourier_scale = f.fourier_scale;
        self.form.mlp_width = f.mlp_width;
        self.form.mlp_layers = f.mlp_layers;
        self.form.kan_width = f.kan_width;
        self.form.kan_layers = f.kan_layers;
        self.form.kan_grid = f.kan_grid;
        self.form.lr = f.lr;
        self.form.batch = f.batch;
        self.form.seed = f.seed;
        self.form.warmup_cosine = f.warmup_cosine;
        self.form.warmup = f.warmup;
        self.form.min_lr_ratio = f.min_lr_ratio;
        self.form.epochs = epochs;
        self.tab = Tab::Train;
        self.status = format!("конфиг и {epochs} эпох применены во вкладке Train");
    }

    fn save_model_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("bin", &["bin"])
            .save_file()
        {
            self.worker
                .send(Command::SaveModel(p.display().to_string()));
            self.status = "сохранение модели…".to_string();
        }
    }

    fn batch_predict_dialog(&mut self) {
        let Some(input) = rfd::FileDialog::new()
            .add_filter("Excel", &["xlsx"])
            .pick_file()
        else {
            return;
        };
        let default_name = input
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}_predicted.xlsx"))
            .unwrap_or_else(|| "predicted.xlsx".to_string());
        let Some(output) = rfd::FileDialog::new()
            .add_filter("Excel", &["xlsx"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };
        self.batch_predicting = true;
        self.status = "заполнение Excel…".to_string();
        self.worker.send(Command::PredictFile {
            input: input.display().to_string(),
            output: output.display().to_string(),
        });
    }

    fn apply_prepare_inference(&mut self) {
        if self.prepare_form.input_path.is_empty() {
            return;
        }
        match infer_prepare_spec_from_path(&self.prepare_form.input_path, Delimiter::Auto) {
            Ok(inferred) => {
                self.prepare_form.inputs = inferred.n_inputs;
                self.prepare_form.outputs = inferred.n_outputs;
                self.prepare_form.delimiter = match inferred.delimiter {
                    Delimiter::Auto => 0,
                    Delimiter::Comma => 1,
                    Delimiter::Tab => 2,
                    Delimiter::Space => 3,
                };
                self.prepare_form.has_header = inferred.has_header;
                self.prepare_form.categorical = inferred
                    .categorical
                    .iter()
                    .map(|(i, c)| format!("{i}:{c}"))
                    .collect::<Vec<_>>()
                    .join(",");
                self.status = format!(
                    "prepare auto: {} вход -> {} выход",
                    inferred.n_inputs, inferred.n_outputs
                );
            }
            Err(e) => {
                self.status = format!("prepare auto не сработал: {e}");
            }
        }
    }

    /// Диалог разметки. Для XLSX/CSV он показывается ВСЕГДА: автоопределение
    /// только заполняет начальное состояние, подтверждает роли пользователь.
    fn ui_markup(&mut self, ctx: &egui::Context) {
        let Some(state) = &mut self.markup else {
            return;
        };
        let mut open = true;
        let mut close_after_apply = false;
        let mut reopen: Option<(String, bool)> = None;

        egui::Window::new("Разметка таблицы")
            .open(&mut open)
            .resizable(true)
            .default_width(760.0)
            .show(ctx, |ui| {
                ui.label(format!(
                    "{} — {} строк, {} колонок",
                    state.path,
                    state.profile.rows,
                    state.draft.len()
                ));
                let mut has_header = state.has_header;
                if ui
                    .checkbox(&mut has_header, "первая строка — заголовок")
                    .changed()
                {
                    // Заголовок меняет разбор файла, поэтому таблица читается
                    // заново — иначе имена и данные разъедутся.
                    reopen = Some((state.path.clone(), has_header));
                }

                ui.separator();
                let mut roles_changed = false;
                let mut other_changed = false;
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        egui::Grid::new("markup_grid")
                            .num_columns(5)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("колонка");
                                ui.label("роль");
                                ui.label("тип");
                                ui.label("единица");
                                ui.label("данные");
                                ui.end_row();

                                for i in 0..state.draft.len() {
                                    let column = state.draft.columns()[i].clone();
                                    let mut name = column.name.clone();
                                    if ui.text_edit_singleline(&mut name).changed() {
                                        let _ = state.draft.set_name(i, name);
                                        other_changed = true;
                                    }

                                    let mut role = column.role;
                                    egui::ComboBox::from_id_salt(format!("role_{i}"))
                                        .selected_text(role.label())
                                        .width(120.0)
                                        .show_ui(ui, |ui| {
                                            for candidate in [
                                                ColumnRole::Input,
                                                ColumnRole::Output,
                                                ColumnRole::Ignore,
                                            ] {
                                                ui.selectable_value(
                                                    &mut role,
                                                    candidate,
                                                    candidate.label(),
                                                );
                                            }
                                        });
                                    if role != column.role {
                                        let _ = state.draft.set_role(i, role);
                                        roles_changed = true;
                                    }

                                    let mut ty = column.ty;
                                    egui::ComboBox::from_id_salt(format!("type_{i}"))
                                        .selected_text(match ty {
                                            DraftType::Numeric => "число",
                                            DraftType::Categorical => "категория",
                                        })
                                        .width(110.0)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut ty,
                                                DraftType::Numeric,
                                                "число",
                                            );
                                            ui.selectable_value(
                                                &mut ty,
                                                DraftType::Categorical,
                                                "категория",
                                            );
                                        });
                                    if ty != column.ty {
                                        match state.draft.set_type(i, ty) {
                                            Ok(()) => roles_changed = true,
                                            Err(e) => state.apply_error = Some(e),
                                        }
                                    }

                                    let mut unit = column.unit.clone().unwrap_or_default();
                                    if ui.text_edit_singleline(&mut unit).changed() {
                                        let trimmed = unit.trim().to_string();
                                        let _ = state
                                            .draft
                                            .set_unit(i, (!trimmed.is_empty()).then_some(trimmed));
                                        other_changed = true;
                                    }

                                    let p = &state.profile.columns[i];
                                    let distinct = p
                                        .n_distinct()
                                        .map(|n| n.to_string())
                                        .unwrap_or_else(|| "много".to_string());
                                    ui.label(format!(
                                        "различных {distinct}, пропусков {}, текст {}",
                                        p.missing, p.non_numeric
                                    ));
                                    ui.end_row();
                                }
                            });
                    });

                if roles_changed {
                    state.on_roles_changed();
                } else if other_changed {
                    state.on_any_change();
                }

                ui.separator();
                let blocking = egui::Color32::from_rgb(200, 60, 60);
                let warn = egui::Color32::from_rgb(200, 120, 0);
                for issue in &state.issues {
                    ui.colored_label(blocking, format!("✖ {issue}"));
                }
                if let Some(e) = &state.apply_error {
                    ui.colored_label(blocking, format!("✖ {e}"));
                }
                for message in state
                    .profile
                    .messages()
                    .into_iter()
                    .chain(state.report.messages(&state.draft))
                {
                    match message.severity {
                        Severity::Blocking => {
                            ui.colored_label(blocking, format!("✖ {}", message.text))
                        }
                        Severity::Warning => ui.colored_label(warn, format!("⚠ {}", message.text)),
                        Severity::Note => ui.label(format!("• {}", message.text)),
                    };
                }

                ui.separator();
                // При переключении заголовка текущий Table уже устарел, а
                // blocking-сообщения должны не только окрашиваться красным.
                let ready = state.can_apply() && reopen.is_none();
                if ui
                    .add_enabled(ready, egui::Button::new("Применить разметку"))
                    .clicked()
                {
                    match state.apply() {
                        Ok(prepared) => {
                            self.form.prepared = Some(prepared);
                            self.form.source_kind = SourceKind::Table;
                            self.status = "разметка применена".to_string();
                            close_after_apply = true;
                        }
                        Err(e) => state.apply_error = Some(e),
                    }
                }
            });

        if let Some((path, has_header)) = reopen {
            // Старую интерпретацию больше нельзя применить, пока worker читает
            // файл заново с другой семантикой первой строки.
            self.markup = None;
            self.open_table(path, has_header);
        } else if !open || close_after_apply {
            self.markup = None;
        }
    }

    fn ui_train(&mut self, ui: &mut egui::Ui) {
        ui.heading("Train (numeric)");

        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.form.source_kind,
                SourceKind::Blackbox,
                "Чёрный ящик",
            );
            ui.selectable_value(&mut self.form.source_kind, SourceKind::Tnum, ".tnum файл");
            ui.selectable_value(&mut self.form.source_kind, SourceKind::Table, "Таблица");
        });
        match self.form.source_kind {
            SourceKind::Tnum => {
                ui.horizontal(|ui| {
                    if ui.button("Выбрать .tnum…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("tnum", &["tnum"])
                            .pick_file()
                        {
                            self.form.file_path = p.display().to_string();
                        }
                    }
                    ui.label(if self.form.file_path.is_empty() {
                        "(файл не выбран)"
                    } else {
                        &self.form.file_path
                    });
                });
                ui.label(".tnum уже содержит подтверждённую схему — разметка не нужна.");
            }
            SourceKind::Table => {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!self.busy(), egui::Button::new("Открыть таблицу…"))
                        .clicked()
                    {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter(
                                "таблицы",
                                &["xlsx", "xlsm", "xlsb", "xls", "ods", "csv", "tsv", "txt"],
                            )
                            .pick_file()
                        {
                            self.open_table(p.display().to_string(), true);
                        }
                    }
                    match &self.form.prepared {
                        Some(p) => {
                            ui.label(format!(
                                "{}: {} строк, {} вход → {} выход",
                                p.name,
                                p.data.len(),
                                p.schema.n_inputs(),
                                p.schema.n_outputs()
                            ));
                        }
                        None => {
                            ui.label("(таблица не размечена)");
                        }
                    }
                });
                if self.form.prepared.is_some()
                    && ui
                        .add_enabled(
                            self.markup.is_none() && !self.busy(),
                            egui::Button::new("Разметить заново…"),
                        )
                        .clicked()
                {
                    if let Some(p) = self.form.prepared.clone() {
                        self.open_table(p.path.clone(), p.has_header);
                    }
                }
            }
            SourceKind::Blackbox => {
                egui::ComboBox::from_label("чёрный ящик")
                    .selected_text(&self.form.blackbox)
                    .show_ui(ui, |ui| {
                        for &name in BLACKBOXES {
                            ui.selectable_value(&mut self.form.blackbox, name.to_string(), name);
                        }
                    });
            }
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Модель:");
            ui.selectable_value(&mut self.form.kind, ModelKind::Transformer, "transformer");
            ui.selectable_value(&mut self.form.kind, ModelKind::Mlp, "mlp");
            ui.selectable_value(&mut self.form.kind, ModelKind::Kan, "kan");
        });

        egui::Grid::new("cfg_grid").num_columns(2).show(ui, |ui| {
            match self.form.kind {
                ModelKind::Mlp => {
                    ui.label("mlp width");
                    ui.add(egui::DragValue::new(&mut self.form.mlp_width).range(1..=2048));
                    ui.end_row();
                    ui.label("mlp layers");
                    ui.add(egui::DragValue::new(&mut self.form.mlp_layers).range(1..=12));
                    ui.end_row();
                }
                ModelKind::Kan => {
                    ui.label("kan width");
                    ui.add(egui::DragValue::new(&mut self.form.kan_width).range(1..=2048));
                    ui.end_row();
                    ui.label("kan layers");
                    ui.add(egui::DragValue::new(&mut self.form.kan_layers).range(1..=12));
                    ui.end_row();
                    ui.label("kan grid");
                    ui.add(egui::DragValue::new(&mut self.form.kan_grid).range(2..=128));
                    ui.end_row();
                }
                ModelKind::Transformer => {
                    ui.label("d_model");
                    ui.add(egui::DragValue::new(&mut self.form.d_model).range(1..=1024));
                    ui.end_row();
                    ui.label("heads");
                    ui.add(egui::DragValue::new(&mut self.form.heads).range(1..=32));
                    ui.end_row();
                    ui.label("layers");
                    ui.add(egui::DragValue::new(&mut self.form.layers).range(1..=12));
                    ui.end_row();
                    ui.label("d_ff");
                    ui.add(egui::DragValue::new(&mut self.form.d_ff).range(1..=4096));
                    ui.end_row();
                    ui.label("value-encoder");
                    egui::ComboBox::from_id_salt("venc")
                        .selected_text(["linear", "mlp", "fourier"][self.form.venc])
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.form.venc, 0, "linear");
                            ui.selectable_value(&mut self.form.venc, 1, "mlp");
                            ui.selectable_value(&mut self.form.venc, 2, "fourier");
                        });
                    ui.end_row();
                    if self.form.venc == 2 {
                        ui.label("fourier bands");
                        ui.add(egui::DragValue::new(&mut self.form.fourier_bands).range(1..=64));
                        ui.end_row();
                        ui.label("fourier scale");
                        ui.add(
                            egui::DragValue::new(&mut self.form.fourier_scale)
                                .range(0.1..=128.0)
                                .speed(0.1),
                        );
                        ui.end_row();
                    }
                }
            }
            ui.label("lr");
            ui.add(
                egui::DragValue::new(&mut self.form.lr)
                    .range(1e-6..=1.0)
                    .speed(1e-4),
            );
            ui.end_row();
            ui.label("batch");
            ui.add(egui::DragValue::new(&mut self.form.batch).range(1..=8192));
            ui.end_row();
            ui.label("epochs");
            ui.add(egui::DragValue::new(&mut self.form.epochs).range(1..=100000));
            ui.end_row();
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.form.seed));
            ui.end_row();
        });

        ui.checkbox(&mut self.form.warmup_cosine, "scheduler: warmup-cosine");
        if self.form.warmup_cosine {
            ui.horizontal(|ui| {
                ui.label("warmup");
                ui.add(
                    egui::DragValue::new(&mut self.form.warmup)
                        .range(0.0..=0.99)
                        .speed(0.01),
                );
                ui.label("min-lr-ratio");
                ui.add(
                    egui::DragValue::new(&mut self.form.min_lr_ratio)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
            });
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Train"))
                .clicked()
            {
                match self.form.build() {
                    Ok((source, nc, tcfg)) => {
                        self.train_parameter_count = None;
                        self.worker.reset_cancel();
                        self.worker.send(Command::TrainNumeric { source, nc, tcfg });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.training, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена…".to_string();
            }
            if ui
                .add_enabled(
                    self.model_info.is_some() && !self.busy(),
                    egui::Button::new("Save model…"),
                )
                .clicked()
            {
                self.save_model_dialog();
            }
        });

        if let Some(count) = self.train_parameter_count {
            ui.label(format!("Параметров: {count}"));
        }

        if !self.loss_curve.is_empty() {
            let points = PlotPoints::from(self.loss_curve.clone());
            Plot::new("loss_plot")
                .height(220.0)
                .show(ui, |pui| pui.line(Line::new(points).name("train loss")));
        }
        if let Some(m) = &self.metrics {
            ui.separator();
            ui.label(format!(
                "validation: RMSE={:.5}   MAE={:.5}   rel.error={:.2}%   R²={:.5}",
                m.rmse,
                m.mae,
                m.rel_error * 100.0,
                m.r2
            ));
            ui.label("Test отложен и в этой сессии не открывается.");
        }
        if let Some(warning) = self
            .model_info
            .as_ref()
            .and_then(ModelInfo::categorical_warning)
        {
            ui.colored_label(egui::Color32::from_rgb(200, 120, 0), warning);
        }
    }

    fn ui_predict(&mut self, ui: &mut egui::Ui) {
        ui.heading("Predict");
        if ui.button("Загрузить модель (.bin)…").clicked() {
            if let Some(p) = rfd::FileDialog::new()
                .add_filter("bin", &["bin"])
                .pick_file()
            {
                self.worker
                    .send(Command::LoadModel(p.display().to_string()));
            }
        }

        let info = self.model_info.clone();
        match info {
            None => {
                ui.label("Обучите модель (вкладка Train) или загрузите .bin.");
            }
            Some(info) => {
                let (n_in, n_out) = (info.schema.n_inputs(), info.schema.n_outputs());
                let (source, parameter_count) = (&info.source, info.parameter_count);
                ui.label(format!(
                    "Модель: {source} ({n_in} вход → {n_out} выход, {parameter_count} параметров)"
                ));
                if let Some(warning) = info.categorical_warning() {
                    ui.colored_label(egui::Color32::from_rgb(200, 120, 0), warning);
                }
                ui.separator();
                egui::Grid::new("predict_inputs")
                    .num_columns(2)
                    .show(ui, |ui| {
                        for (i, v) in self.predict_inputs.iter_mut().enumerate() {
                            let column = &info.schema.inputs()[i];
                            ui.label(column.display_name());
                            match column.ty() {
                                // Категория выбирается подписью: набрать код
                                // вручную нельзя, поэтому неизвестный уровень
                                // не может попасть в модель.
                                ColumnType::Categorical { levels } => {
                                    let code = (*v as usize).min(levels.len().saturating_sub(1));
                                    egui::ComboBox::from_id_salt(format!("cat_{i}"))
                                        .selected_text(&levels[code])
                                        .show_ui(ui, |ui| {
                                            for (c, level) in levels.iter().enumerate() {
                                                if ui.selectable_label(c == code, level).clicked() {
                                                    *v = c as f32;
                                                }
                                            }
                                        });
                                }
                                ColumnType::Numeric => {
                                    ui.add(egui::DragValue::new(v).speed(0.05));
                                }
                            }
                            ui.end_row();
                        }
                    });
                if ui
                    .add_enabled(!self.busy(), egui::Button::new("Predict"))
                    .clicked()
                {
                    self.worker
                        .send(Command::Predict(self.predict_inputs.clone()));
                }

                if let Some(out) = &self.predict_outputs {
                    ui.separator();
                    for (i, v) in out.iter().enumerate() {
                        ui.label(format!(
                            "{} = {v:.6}",
                            info.schema.outputs()[i].display_name()
                        ));
                    }
                }
                if !self.extrapolation.is_empty() {
                    ui.separator();
                    let warn = egui::Color32::from_rgb(200, 120, 0);
                    ui.colored_label(warn, "⚠ экстраполяция – модель ненадёжна вне диапазона:");
                    for e in &self.extrapolation {
                        ui.colored_label(
                            warn,
                            format!(
                                "{} = {} вне [{}, {}]",
                                info.schema.inputs()[e.feature].display_name(),
                                e.value,
                                e.min,
                                e.max
                            ),
                        );
                    }
                }

                ui.separator();
                ui.label("Пакетный Predict из Excel");
                ui.label(format!(
                    "Ожидается первый лист с колонками x0..x{} и y0..y{}; y-колонки будут перезаписаны.",
                    n_in.saturating_sub(1),
                    n_out.saturating_sub(1)
                ));
                if ui
                    .add_enabled(
                        !self.busy() && self.model_info.is_some(),
                        egui::Button::new("Заполнить Excel (.xlsx)…"),
                    )
                    .clicked()
                {
                    self.batch_predict_dialog();
                }
            }
        }
    }

    fn ui_kan_curves(&mut self, ui: &mut egui::Ui) {
        ui.heading("KAN: функции рёбер");
        let Some((layer_dims, domain)) = self
            .kan_info
            .as_ref()
            .map(|info| (info.layer_dims.clone(), info.domain))
        else {
            ui.label("Обучите или загрузите KAN-модель, чтобы увидеть функции её рёбер.");
            return;
        };
        if layer_dims.is_empty() {
            ui.label("В модели нет KAN-слоёв.");
            return;
        }

        self.kan_layer = self.kan_layer.min(layer_dims.len() - 1);
        let previous_layer = self.kan_layer;
        egui::ComboBox::from_label("слой")
            .selected_text(format!(
                "{} ({} → {})",
                self.kan_layer, layer_dims[self.kan_layer].0, layer_dims[self.kan_layer].1
            ))
            .show_ui(ui, |ui| {
                for (layer, &(n_inputs, n_outputs)) in layer_dims.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.kan_layer,
                        layer,
                        format!("{layer} ({n_inputs} → {n_outputs})"),
                    );
                }
            });
        let mut changed = self.kan_layer != previous_layer;
        if changed {
            self.kan_input = 0;
            self.kan_output = 0;
        }

        let (n_inputs, n_outputs) = layer_dims[self.kan_layer];
        self.kan_input = self.kan_input.min(n_inputs - 1);
        self.kan_output = self.kan_output.min(n_outputs - 1);
        ui.horizontal(|ui| {
            ui.label("вход");
            changed |= ui
                .add(egui::DragValue::new(&mut self.kan_input).range(0..=n_inputs - 1))
                .changed();
            ui.label("выход");
            changed |= ui
                .add(egui::DragValue::new(&mut self.kan_output).range(0..=n_outputs - 1))
                .changed();
        });
        if changed {
            self.request_kan_curve();
        }

        let x_label = if self.kan_layer == 0 {
            "нормализованный исходный вход"
        } else {
            "активация предыдущего KAN-слоя"
        };
        ui.label(format!(
            "φ{}→{}(x), слой {}; x – {}, сетка [{:.1}, {:.1}]",
            self.kan_input, self.kan_output, self.kan_layer, x_label, domain.0, domain.1
        ));
        if self.kan_curve.is_empty() {
            ui.label("Выборка кривой…");
            return;
        }
        let points = PlotPoints::from(self.kan_curve.clone());
        Plot::new("kan_edge_curve")
            .height(320.0)
            .include_x(domain.0 as f64)
            .include_x(domain.1 as f64)
            .show(ui, |pui| {
                pui.line(Line::new(points).name("φ(x)"));
            });
    }

    fn ui_kan_formulas(&mut self, ui: &mut egui::Ui) {
        ui.heading("KAN: символьные формулы");
        let Some(symbolic_available) = self.kan_info.as_ref().map(|info| info.symbolic_available)
        else {
            ui.label("Обучите KAN-модель, чтобы извлечь формулы.");
            return;
        };
        if !symbolic_available {
            ui.label(
                "Checkpoint без калибровочной секции (сохранён старой версией): обучите KAN в этой сессии или пересохраните модель – новые .bin несут выборку train-активаций.",
            );
            return;
        }

        ui.label(
            "Фит строится по train-активациям, а ниже показаны формулы в исходных единицах данных.",
        );
        let action = if self.kan_symbolic.is_some() {
            "Обновить формулы"
        } else {
            "Извлечь формулы"
        };
        if ui
            .add_enabled(!self.busy(), egui::Button::new(action))
            .clicked()
        {
            self.kan_symbolic = None;
            self.kan_symbolic_pending = true;
            self.status = "символьная экстракция…".to_string();
            self.worker.send(Command::ExtractKanSymbolic);
        }
        if self.kan_symbolic_pending {
            ui.label("Подбор примитивов по рёбрам…");
            return;
        }

        let Some(result) = &self.kan_symbolic else {
            return;
        };
        ui.separator();
        egui::Grid::new("kan_symbolic_metrics")
            .num_columns(2)
            .show(ui, |ui| {
                match (&result.formula_metrics, result.kan_r2) {
                    (Some(metrics), Some(kan_r2)) => {
                        ui.label("R² формул на validation");
                        ui.label(format!("{:.5} (KAN: {kan_r2:.5})", metrics.r2));
                        ui.end_row();
                        ui.label("Ошибка формул");
                        ui.label(format!(
                            "RMSE {:.5}, rel. {:.2}%",
                            metrics.rmse,
                            metrics.rel_error * 100.0
                        ));
                        ui.end_row();
                    }
                    _ => {
                        ui.label("Test-метрики");
                        ui.label("недоступны: модель из checkpoint-а (нет validation-набора)");
                        ui.end_row();
                    }
                }
                ui.label("Подгонка активных рёбер");
                ui.label(format!(
                    "min R² {:.4}, среднее R² {:.4}",
                    result.min_edge_r2, result.mean_edge_r2
                ));
                ui.end_row();
            });

        if ui.button("Скопировать формулы").clicked() {
            ui.ctx().copy_text(result.formulas.clone());
        }

        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("kan_symbolic_formulas")
            .max_height(300.0)
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&result.formulas).monospace())
                        .selectable(true),
                );
            });

        if result.weak_edges.is_empty() {
            ui.label("Все активные рёбра подогнаны с R² ≥ 0.99.");
            return;
        }
        ui.separator();
        let warn = egui::Color32::from_rgb(200, 120, 0);
        ui.colored_label(
            warn,
            format!(
                "{} приближённых рёбер (R² < 0.99): формулы для них требуют проверки.",
                result.weak_edges.len()
            ),
        );
        egui::Grid::new("kan_symbolic_weak_edges")
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.label("слой");
                ui.label("вход");
                ui.label("выход");
                ui.label("примитив");
                ui.label("R²");
                ui.end_row();
                for edge in &result.weak_edges {
                    ui.label(edge.layer.to_string());
                    ui.label(&edge.input);
                    ui.label(&edge.output);
                    ui.label(&edge.primitive);
                    ui.label(format!("{:.4}", edge.r2));
                    ui.end_row();
                }
            });
    }

    fn ui_diagnose(&mut self, ui: &mut egui::Ui) {
        ui.heading("Diagnose");
        if self.model_info.is_none() {
            ui.label("Обучите модель (вкладка Train) – диагностика по её данным.");
            return;
        }
        if ui.button("Запустить диагностику").clicked() {
            self.diagnostics = None;
            self.status = "диагностика…".to_string();
            self.worker.send(Command::Diagnose);
        }
        if let Some(d) = &self.diagnostics {
            ui.separator();
            ui.label(format!(
                "Overfit-проба: норм. train MSE = {:.5}",
                d.overfit_loss
            ));
            ui.label(if d.overfit_loss < 0.02 {
                "  → ёмкости хватает (проблема в данных/обобщении)"
            } else {
                "  → underfit: ёмкость или кодирование значений (value encoder / Fourier)"
            });
            ui.label(format!(
                "Экстраполяция: {} из {} validation-строк вне обученного диапазона",
                d.extrapolation_rows, d.extrapolation_total
            ));

            ui.separator();
            ui.label("Остаток по входным признакам:");
            egui::Grid::new("resid_grid").num_columns(3).show(ui, |ui| {
                ui.label("признак");
                ui.label("смена знака");
                ui.label("tail/inner");
                ui.end_row();
                for (i, (sc, tr)) in d.residuals.iter().enumerate() {
                    ui.label(format!("{i}"));
                    ui.label(format!("{:.0}%", sc * 100.0));
                    ui.label(format!("{tr:.2}"));
                    ui.end_row();
                }
            });
            ui.label("(высокая смена знака → частота/Fourier; tail/inner>1.5 → масштаб/хвосты)");

            ui.separator();
            match d.sensitivity {
                Some((mean, max)) => {
                    ui.label(format!(
                        "Чувствительность ‖Δy‖/‖Δx‖ (норм.): среднее {mean:.2}, макс {max:.2}"
                    ));
                    ui.label(if max < 10.0 {
                        "  → карта гладкая, surrogate надёжен"
                    } else {
                        "  → высокая чувствительность/хаос: потолок точности"
                    });
                }
                None => {
                    ui.label("Чувствительность: только для blackbox (для .tnum пропущена)");
                }
            }
        }
    }

    fn ui_optimize(&mut self, ui: &mut egui::Ui) {
        ui.heading("Optimize");

        ui.horizontal(|ui| {
            if ui.button("Выбрать .tnum…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("tnum", &["tnum"])
                    .pick_file()
                {
                    self.optimize_form.file_path = p.display().to_string();
                }
            }
            ui.label(if self.optimize_form.file_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.optimize_form.file_path
            });
        });

        ui.horizontal(|ui| {
            ui.label("preset");
            egui::ComboBox::from_id_salt("optimize_preset")
                .selected_text(["Quick", "Balanced", "Deep"][self.optimize_form.preset])
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.optimize_form.preset, 0, "Quick");
                    ui.selectable_value(&mut self.optimize_form.preset, 1, "Balanced");
                    ui.selectable_value(&mut self.optimize_form.preset, 2, "Deep");
                });
            ui.label("objective");
            egui::ComboBox::from_id_salt("optimize_objective")
                .selected_text(objective_label(self.optimize_form.objective()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.optimize_form.objective, 0, "worst-output R²");
                    ui.selectable_value(&mut self.optimize_form.objective, 1, "aggregate R²");
                    ui.selectable_value(&mut self.optimize_form.objective, 2, "mean-output R²");
                    ui.selectable_value(&mut self.optimize_form.objective, 3, "aggregate nRMSE");
                });
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.optimize_form.include_transformer, "transformer");
            ui.checkbox(&mut self.optimize_form.include_mlp, "mlp");
            ui.checkbox(&mut self.optimize_form.include_kan, "kan");
        });
        ui.label(match self.optimize_form.preset {
            1 => "Balanced: средняя сетка, поиск на 40 эпохах, один seed.",
            2 => "Deep: широкая сетка, поиск на 40 эпохах, два seed (устойчивость выбора).",
            _ => "Quick: короткий поиск на 25 эпохах – быстро сравнить transformer/MLP/KAN.",
        });
        ui.label(
            "Optimize только ищет конфиг. Финальное обучение делает Train \
             после «Apply best» (на полном бюджете эпох).",
        );
        // Оценка размера до запуска — чтобы случайно не словить долгий прогон.
        if let Ok((cfgs, runs)) = self
            .optimize_form
            .axes()
            .and_then(|a| sweep::sweep_size(&a))
        {
            ui.label(format!(
                "Оценка: {cfgs} конфигов, {runs} прогонов (на реальных данных трансформер ~минуту/прогон)"
            ));
        }
        self.sort_optimize_rows();

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Run optimize"))
                .clicked()
            {
                match self.optimize_form.build() {
                    Ok((path, axes, objective)) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::OptimizeFile {
                            path,
                            axes,
                            objective,
                        });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.optimizing, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена optimize…".to_string();
            }
            let best_choice = self.optimize_rows.first().map(|r| r.choice.clone());
            let has_best = best_choice.is_some() && !self.optimizing;
            if ui
                .add_enabled(has_best, egui::Button::new("Apply to Train"))
                .clicked()
            {
                if let Some(choice) = &best_choice {
                    self.apply_choice_to_train(choice);
                }
            }
            if ui
                .add_enabled(has_best, egui::Button::new("Check epochs"))
                .on_hover_text("Перенести конфиг в Epoch-sweep и подобрать число эпох")
                .clicked()
            {
                if let Some(choice) = &best_choice {
                    self.apply_choice_to_epoch_sweep(choice);
                }
            }
        });

        if let Some((cfgs, runs)) = self.optimize_total {
            ui.label(format!(
                "Конфигов: {cfgs}; прогонов: {runs}; готово: {}",
                self.optimize_rows.len()
            ));
        }
        if self.optimize_cancelled {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Optimize отменён; показаны завершённые конфиги.",
            );
        }

        if !self.optimize_rows.is_empty() {
            ui.separator();
            let (search_epochs, final_epochs) = self
                .optimize_rows
                .first()
                .map(|r| (r.choice.epochs, r.choice.final_epochs))
                .unwrap_or((0, 0));
            let source = epoch_sweep::source_label(self.optimize_rows[0].source);
            ui.label(format!(
                "Ранжирование: {}; метрики {source} (поиск на {search_epochs} эпох; Apply -> {final_epochs})",
                objective_label(self.optimize_form.objective())
            ));
            egui::ScrollArea::vertical()
                .max_height(360.0)
                .show(ui, |ui| {
                    egui::Grid::new("optimize_rows")
                        .num_columns(8)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("#");
                            ui.label("score");
                            ui.label("R²");
                            ui.label("worst y");
                            ui.label("mean y");
                            ui.label("aggr nRMSE");
                            ui.label("rel");
                            ui.label("config");
                            ui.end_row();
                            let objective = self.optimize_form.objective();
                            for (i, row) in self.optimize_rows.iter().enumerate() {
                                let mark = if i == 0 { "*" } else { "" };
                                ui.label(format!("{mark}{}", i + 1));
                                ui.label(format!("{:.5}", objective_display_score(objective, row)));
                                ui.label(format!("{:.5}", row.r2_mean));
                                ui.label(format!("{:.5}", row.worst_output_r2_mean));
                                ui.label(format!("{:.5}", row.mean_output_r2_mean));
                                ui.label(format!("{:.5}", row.nrmse_mean));
                                ui.label(format!("{:.1}%", row.rel_mean * 100.0));
                                ui.label(&row.label);
                                ui.end_row();
                            }
                        });
                });
        }
    }

    fn ui_sweep(&mut self, ui: &mut egui::Ui) {
        ui.heading("Sweep");

        egui::ComboBox::from_label("чёрный ящик")
            .selected_text(&self.sweep_form.blackbox)
            .show_ui(ui, |ui| {
                for &name in BLACKBOXES {
                    ui.selectable_value(&mut self.sweep_form.blackbox, name.to_string(), name);
                }
            });

        ui.separator();
        egui::Grid::new("sweep_axes")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("seeds");
                ui.text_edit_singleline(&mut self.sweep_form.seeds);
                ui.end_row();
                ui.label("d-models");
                ui.text_edit_singleline(&mut self.sweep_form.d_models);
                ui.end_row();
                ui.label("layers-list");
                ui.text_edit_singleline(&mut self.sweep_form.layers);
                ui.end_row();
                ui.label("d-ffs");
                ui.text_edit_singleline(&mut self.sweep_form.d_ffs);
                ui.end_row();
                ui.label("lrs");
                ui.text_edit_singleline(&mut self.sweep_form.lrs);
                ui.end_row();
                ui.label("value-encoders");
                ui.text_edit_singleline(&mut self.sweep_form.value_encoders);
                ui.end_row();
                ui.label("fourier-scales");
                ui.text_edit_singleline(&mut self.sweep_form.fourier_scales);
                ui.end_row();
                ui.label("fourier-bands");
                ui.add(egui::DragValue::new(&mut self.sweep_form.fourier_bands).range(1..=64));
                ui.end_row();
                ui.label("schedulers");
                ui.text_edit_singleline(&mut self.sweep_form.schedulers);
                ui.end_row();
                ui.label("epochs");
                ui.add(egui::DragValue::new(&mut self.sweep_form.epochs).range(1..=100000));
                ui.end_row();
                ui.label("batch-size");
                ui.add(egui::DragValue::new(&mut self.sweep_form.batch).range(1..=8192));
                ui.end_row();
            });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.training && !self.sweeping,
                    egui::Button::new("Run sweep"),
                )
                .clicked()
            {
                match self.sweep_form.build() {
                    Ok((blackbox, axes)) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::Sweep { blackbox, axes });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.sweeping, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена sweep…".to_string();
            }
        });

        if let Some((cfgs, runs)) = self.sweep_total {
            ui.label(format!(
                "Конфигов: {cfgs}; прогонов: {runs}; готово: {}",
                self.sweep_rows.len()
            ));
        }
        if self.sweep_cancelled {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Sweep отменён; показаны завершённые конфиги.",
            );
        }

        if !self.sweep_rows.is_empty() {
            ui.separator();
            ui.label(format!(
                "Источник метрик: {}",
                epoch_sweep::source_label(self.sweep_rows[0].source)
            ));
            egui::ScrollArea::vertical()
                .max_height(320.0)
                .show(ui, |ui| {
                    egui::Grid::new("sweep_rows")
                        .num_columns(6)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("#");
                            ui.label("R²");
                            ui.label("std");
                            ui.label("aggr nRMSE");
                            ui.label("rel");
                            ui.label("config");
                            ui.end_row();
                            for (i, row) in self.sweep_rows.iter().enumerate() {
                                let mark = if i == 0 { "*" } else { "" };
                                ui.label(format!("{mark}{}", i + 1));
                                ui.label(format!("{:.5}", row.r2_mean));
                                ui.label(format!("{:.5}", row.r2_std));
                                ui.label(format!("{:.5}", row.nrmse_mean));
                                ui.label(format!("{:.1}%", row.rel_mean * 100.0));
                                ui.label(&row.label);
                                ui.end_row();
                            }
                        });
                });
        }
    }

    fn ui_text(&mut self, ui: &mut egui::Ui) {
        ui.heading("Text");

        ui.horizontal(|ui| {
            if ui.button("Выбрать .txt…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("text", &["txt"])
                    .pick_file()
                {
                    self.text_form.file_path = p.display().to_string();
                }
            }
            ui.label(if self.text_form.file_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.text_form.file_path
            });
        });

        ui.separator();
        egui::Grid::new("text_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("d_model");
                ui.add(egui::DragValue::new(&mut self.text_form.d_model).range(1..=1024));
                ui.end_row();
                ui.label("heads");
                ui.add(egui::DragValue::new(&mut self.text_form.heads).range(1..=32));
                ui.end_row();
                ui.label("layers");
                ui.add(egui::DragValue::new(&mut self.text_form.layers).range(1..=12));
                ui.end_row();
                ui.label("d_ff");
                ui.add(egui::DragValue::new(&mut self.text_form.d_ff).range(1..=4096));
                ui.end_row();
                ui.label("steps");
                ui.add(egui::DragValue::new(&mut self.text_form.steps).range(1..=200000));
                ui.end_row();
                ui.label("batch");
                ui.add(egui::DragValue::new(&mut self.text_form.batch).range(1..=1024));
                ui.end_row();
                ui.label("ctx_len");
                ui.add(egui::DragValue::new(&mut self.text_form.ctx_len).range(1..=512));
                ui.end_row();
                ui.label("tgt_len");
                ui.add(egui::DragValue::new(&mut self.text_form.tgt_len).range(1..=512));
                ui.end_row();
                ui.label("lr");
                ui.add(
                    egui::DragValue::new(&mut self.text_form.lr)
                        .range(1e-6..=1.0)
                        .speed(1e-4),
                );
                ui.end_row();
                ui.label("seed");
                ui.add(egui::DragValue::new(&mut self.text_form.seed));
                ui.end_row();
            });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Train text"))
                .clicked()
            {
                match self.text_form.build() {
                    Ok((path, model_cfg, train_cfg)) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::TrainText {
                            path,
                            model_cfg,
                            train_cfg,
                        });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.text_training, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена text…".to_string();
            }
        });

        if !self.text_curve.is_empty() {
            let points = PlotPoints::from(self.text_curve.clone());
            Plot::new("text_ppl_plot")
                .height(220.0)
                .show(ui, |pui| pui.line(Line::new(points).name("perplexity")));
        }
        if let Some(vocab) = self.text_vocab_size {
            ui.label(format!("vocab: {vocab}"));
        }

        ui.separator();
        egui::Grid::new("text_gen")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("new chars");
                ui.add(egui::DragValue::new(&mut self.text_form.total_new).range(1..=5000));
                ui.end_row();
                ui.label("temperature");
                ui.add(
                    egui::DragValue::new(&mut self.text_form.temperature)
                        .range(0.0..=5.0)
                        .speed(0.05),
                );
                ui.end_row();
                ui.label("top_k");
                ui.add(egui::DragValue::new(&mut self.text_form.top_k).range(0..=512));
                ui.end_row();
                ui.label("rng seed");
                ui.add(egui::DragValue::new(&mut self.text_form.gen_seed));
                ui.end_row();
            });
        ui.label("seed text");
        ui.text_edit_multiline(&mut self.text_form.seed_text);
        if ui
            .add_enabled(
                self.text_ready && !self.text_training,
                egui::Button::new("Generate"),
            )
            .clicked()
        {
            self.worker.send(Command::GenerateText {
                seed: self.text_form.seed_text.clone(),
                total_new: self.text_form.total_new,
                temperature: self.text_form.temperature,
                top_k: self.text_form.top_k,
                rng_seed: self.text_form.gen_seed,
            });
        }

        if !self.generated_text.is_empty() {
            ui.separator();
            ui.label("generated");
            egui::ScrollArea::vertical()
                .max_height(240.0)
                .show(ui, |ui| {
                    ui.label(&self.generated_text);
                });
        }
    }
    fn ui_prepare(&mut self, ui: &mut egui::Ui) {
        ui.heading("Prepare");
        ui.horizontal(|ui| {
            if ui.button("Вход…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter(
                        "tables",
                        &["csv", "tsv", "txt", "xlsx", "xlsm", "xlsb", "xls", "ods"],
                    )
                    .pick_file()
                {
                    self.prepare_form.input_path = p.display().to_string();
                    self.apply_prepare_inference();
                }
            }
            ui.label(if self.prepare_form.input_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.prepare_form.input_path
            });
        });
        ui.horizontal(|ui| {
            if ui.button("Выход .tnum…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("tnum", &["tnum"])
                    .save_file()
                {
                    self.prepare_form.output_path = p.display().to_string();
                }
            }
            ui.label(if self.prepare_form.output_path.is_empty() {
                "(путь не выбран)"
            } else {
                &self.prepare_form.output_path
            });
        });
        egui::Grid::new("prepare_grid")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("inputs");
                ui.add(egui::DragValue::new(&mut self.prepare_form.inputs).range(1..=256));
                ui.end_row();
                ui.label("outputs");
                ui.add(egui::DragValue::new(&mut self.prepare_form.outputs).range(1..=256));
                ui.end_row();
                ui.label("delimiter");
                egui::ComboBox::from_id_salt("prepare_delim")
                    .selected_text(["auto", "comma", "tab", "space"][self.prepare_form.delimiter])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.prepare_form.delimiter, 0, "auto");
                        ui.selectable_value(&mut self.prepare_form.delimiter, 1, "comma");
                        ui.selectable_value(&mut self.prepare_form.delimiter, 2, "tab");
                        ui.selectable_value(&mut self.prepare_form.delimiter, 3, "space");
                    });
                ui.end_row();
                ui.label("categorical");
                ui.text_edit_singleline(&mut self.prepare_form.categorical);
                ui.end_row();
            });
        ui.checkbox(&mut self.prepare_form.has_header, "has header");
        if ui
            .add_enabled(!self.busy(), egui::Button::new("Convert"))
            .clicked()
        {
            match self.prepare_form.build() {
                Ok((input, output, spec)) => {
                    self.worker.send(Command::Prepare {
                        input,
                        output,
                        spec,
                    });
                }
                Err(e) => self.status = format!("Ошибка: {e}"),
            }
        }
    }

    fn ui_epoch_sweep(&mut self, ui: &mut egui::Ui) {
        ui.heading("Epoch-sweep");
        ui.horizontal(|ui| {
            if ui.button("Выбрать .tnum…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("tnum", &["tnum"])
                    .pick_file()
                {
                    self.epoch_form.file_path = p.display().to_string();
                }
            }
            ui.label(if self.epoch_form.file_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.epoch_form.file_path
            });
        });
        ui.horizontal(|ui| {
            ui.label("Модель:");
            ui.selectable_value(
                &mut self.epoch_form.kind,
                ModelKind::Transformer,
                "transformer",
            );
            ui.selectable_value(&mut self.epoch_form.kind, ModelKind::Mlp, "mlp");
            ui.selectable_value(&mut self.epoch_form.kind, ModelKind::Kan, "kan");
        });
        egui::Grid::new("epoch_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                match self.epoch_form.kind {
                    ModelKind::Mlp => {
                        ui.label("mlp width");
                        ui.add(
                            egui::DragValue::new(&mut self.epoch_form.mlp_width).range(1..=2048),
                        );
                        ui.end_row();
                        ui.label("mlp layers");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.mlp_layers).range(1..=12));
                        ui.end_row();
                    }
                    ModelKind::Kan => {
                        ui.label("kan width");
                        ui.add(
                            egui::DragValue::new(&mut self.epoch_form.kan_width).range(1..=2048),
                        );
                        ui.end_row();
                        ui.label("kan layers");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.kan_layers).range(1..=12));
                        ui.end_row();
                        ui.label("kan grid");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.kan_grid).range(2..=128));
                        ui.end_row();
                    }
                    ModelKind::Transformer => {
                        ui.label("d_model");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.d_model).range(1..=1024));
                        ui.end_row();
                        ui.label("heads");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.heads).range(1..=32));
                        ui.end_row();
                        ui.label("layers");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.layers).range(1..=12));
                        ui.end_row();
                        ui.label("d_ff");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.d_ff).range(1..=4096));
                        ui.end_row();
                        ui.label("value-encoder");
                        egui::ComboBox::from_id_salt("epoch_venc")
                            .selected_text(["linear", "mlp", "fourier"][self.epoch_form.venc])
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.epoch_form.venc, 0, "linear");
                                ui.selectable_value(&mut self.epoch_form.venc, 1, "mlp");
                                ui.selectable_value(&mut self.epoch_form.venc, 2, "fourier");
                            });
                        ui.end_row();
                        if self.epoch_form.venc == 2 {
                            ui.label("fourier bands");
                            ui.add(
                                egui::DragValue::new(&mut self.epoch_form.fourier_bands)
                                    .range(1..=64),
                            );
                            ui.end_row();
                            ui.label("fourier scale");
                            ui.add(
                                egui::DragValue::new(&mut self.epoch_form.fourier_scale)
                                    .range(0.1..=128.0)
                                    .speed(0.1),
                            );
                            ui.end_row();
                        }
                    }
                }
                ui.label("epochs");
                ui.text_edit_singleline(&mut self.epoch_form.epochs);
                ui.end_row();
                ui.label("lr");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.lr)
                        .range(1e-6..=1.0)
                        .speed(1e-4),
                );
                ui.end_row();
                ui.label("batch");
                ui.add(egui::DragValue::new(&mut self.epoch_form.batch).range(1..=8192));
                ui.end_row();
                ui.label("seed");
                ui.add(egui::DragValue::new(&mut self.epoch_form.seed));
                ui.end_row();
                ui.label("target-r2");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.target_r2)
                        .range(-10.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label("min-r2-gain");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.min_gain)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label("plateau-min-r2");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.plateau_min)
                        .range(-10.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
            });
        ui.checkbox(
            &mut self.epoch_form.warmup_cosine,
            "scheduler: warmup-cosine",
        );
        if self.epoch_form.warmup_cosine {
            ui.horizontal(|ui| {
                ui.label("warmup");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.warmup)
                        .range(0.0..=0.99)
                        .speed(0.01),
                );
                ui.label("min-lr-ratio");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.min_lr_ratio)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
            });
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Run"))
                .clicked()
            {
                match self.epoch_form.build() {
                    Ok(req) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::EpochSweep {
                            path: req.path,
                            nc: req.nc,
                            base_tcfg: req.base_tcfg,
                            milestones: req.milestones,
                            target_r2: req.target_r2,
                            min_gain: req.min_gain,
                            plateau_min: req.plateau_min,
                        });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.epoch_sweeping, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена epoch-sweep…".to_string();
            }
            if ui
                .add_enabled(!self.epoch_rows.is_empty(), egui::Button::new("Save CSV"))
                .clicked()
            {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("csv", &["csv"])
                    .save_file()
                {
                    match std::fs::write(&p, epoch_sweep::rows_to_csv(&self.epoch_rows)) {
                        Ok(()) => self.status = format!("CSV: {}", p.display()),
                        Err(e) => self.status = format!("Ошибка: запись CSV: {e}"),
                    }
                }
            }
            let recommended = self.epoch_recommendation.as_ref().map(|(e, _)| *e);
            if ui
                .add_enabled(
                    recommended.is_some() && !self.epoch_sweeping,
                    egui::Button::new("Apply recommended to Train"),
                )
                .on_hover_text("Перенести конфиг и рекомендованное число эпох в Train")
                .clicked()
            {
                if let Some(epochs) = recommended {
                    self.apply_epoch_form_to_train(epochs);
                }
            }
        });
        if let Some(total) = self.epoch_total {
            ui.label(format!("точек: {total}; готово: {}", self.epoch_rows.len()));
        }
        if let Some((epoch, why)) = &self.epoch_recommendation {
            ui.label(format!("Рекомендованная остановка: {epoch} эпох ({why})"));
        }
        if self.epoch_cancelled {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Epoch-sweep отменён; показаны завершённые точки.",
            );
        }
        if !self.epoch_rows.is_empty() {
            let r2_points = PlotPoints::from(
                self.epoch_rows
                    .iter()
                    .map(|r| [r.epochs as f64, r.r2 as f64])
                    .collect::<Vec<_>>(),
            );
            let loss_points = PlotPoints::from(
                self.epoch_rows
                    .iter()
                    .map(|r| [r.epochs as f64, r.train_loss as f64])
                    .collect::<Vec<_>>(),
            );
            Plot::new("epoch_sweep_plot")
                .height(260.0)
                .include_y(0.0)
                .include_y(1.0)
                .show(ui, |pui| {
                    pui.line(Line::new(r2_points).name("R²"));
                    pui.line(Line::new(loss_points).name("train loss"));
                });
            egui::ScrollArea::vertical()
                .max_height(260.0)
                .show(ui, |ui| {
                    egui::Grid::new("epoch_rows")
                        .num_columns(6)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("epochs");
                            ui.label("loss");
                            ui.label("RMSE");
                            ui.label("MAE");
                            ui.label("rel");
                            ui.label("R²");
                            ui.end_row();
                            for row in &self.epoch_rows {
                                ui.label(format!("{}", row.epochs));
                                ui.label(format!("{:.5}", row.train_loss));
                                ui.label(format!("{:.5}", row.rmse));
                                ui.label(format!("{:.5}", row.mae));
                                ui.label(format!("{:.1}%", row.rel_error * 100.0));
                                ui.label(format!("{:.5}", row.r2));
                                ui.end_row();
                            }
                        });
                });
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.ui_markup(ctx);

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Train, "Train");
                ui.selectable_value(&mut self.tab, Tab::Predict, "Predict");
                ui.selectable_value(&mut self.tab, Tab::KanCurves, "KAN curves");
                ui.selectable_value(&mut self.tab, Tab::KanFormulas, "KAN formulas");
                ui.selectable_value(&mut self.tab, Tab::Diagnose, "Diagnose");
                ui.selectable_value(&mut self.tab, Tab::Optimize, "Optimize");
                ui.selectable_value(&mut self.tab, Tab::Sweep, "Sweep");
                ui.selectable_value(&mut self.tab, Tab::Text, "Text");
                ui.selectable_value(&mut self.tab, Tab::Prepare, "Prepare");
                ui.selectable_value(&mut self.tab, Tab::EpochSweep, "Epoch-sweep");
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Статус:");
                ui.label(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match self.tab {
                    Tab::Train => self.ui_train(ui),
                    Tab::Predict => self.ui_predict(ui),
                    Tab::KanCurves => self.ui_kan_curves(ui),
                    Tab::KanFormulas => self.ui_kan_formulas(ui),
                    Tab::Diagnose => self.ui_diagnose(ui),
                    Tab::Optimize => self.ui_optimize(ui),
                    Tab::Sweep => self.ui_sweep(ui),
                    Tab::Text => self.ui_text(ui),
                    Tab::Prepare => self.ui_prepare(ui),
                    Tab::EpochSweep => self.ui_epoch_sweep(ui),
                });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Delimiter;

    fn markup(text: &str, suggested: Option<usize>) -> MarkupState {
        let table = Table::parse_text(text, Delimiter::Auto, true).unwrap();
        let profile = TableProfile::of(&table);
        MarkupState::new("t.csv".to_string(), true, table, profile, suggested, &[])
    }

    #[test]
    fn markup_applies_suggested_split_but_user_decides() {
        let mut state = markup("a,b,c\n1,2,3\n4,5,6\n", Some(2));
        let roles: Vec<ColumnRole> = state.draft.columns().iter().map(|c| c.role).collect();
        assert_eq!(
            roles,
            vec![ColumnRole::Input, ColumnRole::Input, ColumnRole::Output]
        );

        // Пользователь переопределяет подсказку.
        state.draft.set_role(1, ColumnRole::Ignore).unwrap();
        state.on_roles_changed();
        let prepared = state.apply().unwrap();
        assert_eq!(prepared.schema.input_names(), vec!["a"]);
        assert_eq!(prepared.schema.output_names(), vec!["c"]);
        assert_eq!(prepared.data.inputs.dim(), (2, 1));
    }

    /// Результат диалога — готовые данные и схема, а не путь: иначе worker
    /// открыл бы файл заново и потерял ручную разметку.
    #[test]
    fn prepared_source_carries_data_and_schema() {
        let mut state = markup("t,mat,y\n80,песок,1\n60,глина,2\n", Some(2));
        state.on_roles_changed();
        let prepared = state.apply().unwrap();

        let form = TrainForm {
            source_kind: SourceKind::Table,
            prepared: Some(prepared),
            ..Default::default()
        };
        let (source, _, _) = form.build().unwrap();
        match source {
            DataSource::Prepared { data, schema, .. } => {
                assert_eq!(data.inputs.dim(), (2, 2));
                assert_eq!(schema.input_names(), vec!["t", "mat"]);
                // Категория распознана по подписям, коды воспроизводимы.
                assert_eq!(schema.inputs()[1].cardinality(), Some(2));
                assert_eq!(data.inputs[[0, 1]], 1.0); // песок — второй по алфавиту
            }
            _ => panic!("ожидался Prepared"),
        }
    }

    #[test]
    fn table_source_without_markup_refuses_to_train() {
        let form = TrainForm {
            source_kind: SourceKind::Table,
            ..Default::default()
        };
        let err = match form.build() {
            Err(e) => e,
            Ok(_) => panic!("обучение без разметки должно быть отклонено"),
        };
        assert!(err.contains("подтвердите разметку"), "{err}");
    }

    #[test]
    fn blocking_issues_prevent_applying() {
        // Текстовая колонка назначена выходом — это блокирующая проблема.
        let mut state = markup("a,b\n1,x\n2,y\n", Some(1));
        assert!(!state.issues.is_empty());
        assert!(state.apply().is_err());

        state.draft.set_role(1, ColumnRole::Ignore).unwrap();
        state.draft.set_role(0, ColumnRole::Output).unwrap();
        state.on_roles_changed();
        // Без входов тоже нельзя.
        assert!(!state.issues.is_empty());
    }

    #[test]
    fn blocking_profile_message_disables_and_rejects_apply() {
        let table = Table::parse_text("a,b\n1,2\n3\n", Delimiter::Auto, true).unwrap();
        let profile = TableProfile::of(&table);
        let state = MarkupState::new("ragged.csv".to_string(), true, table, profile, Some(1), &[]);

        assert!(!state.can_apply());
        let error = match state.apply() {
            Err(error) => error,
            Ok(_) => panic!("таблица с рваными строками не должна применяться"),
        };
        assert!(error.contains("другим числом колонок"), "{error}");
    }

    #[test]
    fn categorical_suggestion_only_prefills_type() {
        let table =
            Table::parse_text("x,material_id,y\n1,0,2\n3,1,4\n", Delimiter::Auto, true).unwrap();
        let profile = TableProfile::of(&table);
        let mut state =
            MarkupState::new("coded.csv".to_string(), true, table, profile, Some(2), &[1]);
        assert_eq!(state.draft.columns()[1].ty, DraftType::Categorical);

        // Это подсказка, не запрет: пользователь может вернуть числовой тип.
        state.draft.set_type(1, DraftType::Numeric).unwrap();
        state.on_roles_changed();
        assert_eq!(state.draft.columns()[1].ty, DraftType::Numeric);
        assert!(state.can_apply());
    }

    /// Отчёт по ролям пересчитывается при смене ролей и не зависит от имён.
    #[test]
    fn report_tracks_roles_and_ignores_renames() {
        let mut text = String::from("x0,x1,x2,y\n");
        for i in 0..30 {
            let x0 = 2.0 + i as f64;
            let x1 = 5.0 + (i % 4) as f64;
            text.push_str(&format!("{x0},{x1},{},{}\n", 100.0 - x0 - x1, i));
        }
        let mut state = markup(&text, Some(3));
        assert_eq!(state.report.dependencies.len(), 1);

        // Переименование не меняет отчёт, но имя в сообщении обновляется.
        state.draft.set_name(0, "доля A").unwrap();
        state.on_any_change();
        assert_eq!(state.report.dependencies.len(), 1);
        let text = state
            .report
            .messages(&state.draft)
            .into_iter()
            .map(|m| m.text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("доля A"), "{text}");

        // Исключение колонки убирает связь.
        state.draft.set_role(2, ColumnRole::Ignore).unwrap();
        state.on_roles_changed();
        assert!(state.report.dependencies.is_empty());
    }

    #[test]
    fn train_form_builds_kan_config() {
        let form = TrainForm {
            kind: ModelKind::Kan,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
            venc: 2,
            ..Default::default()
        };

        let (_, config, _) = form.build().unwrap();
        assert_eq!(config.kind, ModelKind::Kan);
        assert_eq!(config.value.kind, ValueEncoderKind::Linear);
        assert_eq!(config.kan.width, 16);
        assert_eq!(config.kan.layers, 2);
        assert_eq!(config.kan.grid, 8);
    }

    #[test]
    fn epoch_sweep_form_builds_kan_config() {
        let form = EpochSweepForm {
            file_path: "test.tnum".to_string(),
            kind: ModelKind::Kan,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
            ..Default::default()
        };

        let request = form.build().unwrap();
        assert_eq!(request.nc.kind, ModelKind::Kan);
        assert_eq!(request.nc.kan.width, 16);
        assert_eq!(request.nc.kan.layers, 2);
        assert_eq!(request.nc.kan.grid, 8);
    }
}
