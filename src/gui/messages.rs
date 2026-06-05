//! Сообщения UI <-> worker (PlanUI §2.2). Только `Send`-данные; Rc-модели не
//! пересекают границу потока.

use crate::config::ModelConfig;
use crate::data::OutOfRange;
use crate::epoch_sweep::EpochRow;
use crate::metrics::Metrics;
use crate::numeric_model::NumericConfig;
use crate::sweep::{SweepAxes, SweepObjective, SweepRow};
use crate::tnum::PrepareSpec;
use crate::train::{TextTrainConfig, TrainConfig};

/// Источник данных для обучения.
#[derive(Clone)]
pub enum DataSource {
    Blackbox(String),
    File(String),
}

/// Результат диагностики (числа для UI; PlanUI шаг 2).
pub struct DiagnosticsResult {
    pub overfit_loss: f32,
    pub extrapolation_rows: usize,
    pub extrapolation_total: usize,
    /// На признак: (доля смен знака остатка, tail/inner).
    pub residuals: Vec<(f32, f32)>,
    /// (среднее, макс) чувствительности — только для blackbox.
    pub sensitivity: Option<(f32, f32)>,
}

/// Команды UI -> worker.
pub enum Command {
    TrainNumeric {
        source: DataSource,
        nc: NumericConfig,
        tcfg: TrainConfig,
    },
    LoadModel(String),
    SaveModel(String),
    Predict(Vec<f32>),
    PredictFile {
        input: String,
        output: String,
    },
    Diagnose,
    Sweep {
        blackbox: String,
        axes: SweepAxes,
    },
    OptimizeFile {
        path: String,
        axes: SweepAxes,
        objective: SweepObjective,
    },
    TrainText {
        path: String,
        model_cfg: ModelConfig,
        train_cfg: TextTrainConfig,
    },
    GenerateText {
        seed: String,
        total_new: usize,
        temperature: f32,
        top_k: usize,
        rng_seed: u64,
    },
    Prepare {
        input: String,
        output: String,
        spec: PrepareSpec,
    },
    EpochSweep {
        path: String,
        nc: NumericConfig,
        base_tcfg: TrainConfig,
        milestones: Vec<usize>,
        target_r2: f32,
        min_gain: f32,
        plateau_min: f32,
    },
    Shutdown,
}

/// События worker -> UI.
pub enum Event {
    Status(String),
    Error(String),
    TrainStarted {
        total_epochs: usize,
    },
    Epoch {
        epoch: usize,
        loss: f32,
    },
    /// Завершение обучения: `Some` — метрики на тесте, `None` — отменено.
    TrainDone {
        metrics: Option<Metrics>,
    },
    /// Модель готова к предсказанию (после обучения или загрузки `.bin`).
    ModelReady {
        n_inputs: usize,
        n_outputs: usize,
        source: String,
    },
    PredictResult {
        outputs: Vec<f32>,
        extrapolation: Vec<OutOfRange>,
    },
    PredictFileDone {
        output: String,
        rows: usize,
        extrapolation_rows: usize,
    },
    Diagnostics {
        result: DiagnosticsResult,
    },
    SweepStarted {
        total_configs: usize,
        total_runs: usize,
    },
    SweepRow {
        row: SweepRow,
    },
    SweepDone {
        rows: Vec<SweepRow>,
        cancelled: bool,
    },
    OptimizeStarted {
        total_configs: usize,
        total_runs: usize,
    },
    OptimizeRow {
        row: SweepRow,
    },
    OptimizeDone {
        rows: Vec<SweepRow>,
        cancelled: bool,
    },
    TextStarted {
        total_steps: usize,
    },
    TextProgress {
        step: usize,
        loss: f32,
    },
    TextDone {
        final_loss: Option<f32>,
        cancelled: bool,
        vocab_size: usize,
        seed_hint: String,
    },
    GeneratedText {
        text: String,
    },
    PrepareDone {
        output: String,
        rows: usize,
        n_inputs: usize,
        n_outputs: usize,
    },
    EpochSweepStarted {
        total_points: usize,
    },
    EpochSweepRow {
        row: EpochRow,
    },
    EpochSweepDone {
        rows: Vec<EpochRow>,
        recommendation: Option<(usize, String)>,
        cancelled: bool,
    },
}
