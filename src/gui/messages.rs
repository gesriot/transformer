//! Сообщения UI <-> worker. Только `Send`-данные; Rc-модели не
//! пересекают границу потока.

use crate::config::ModelConfig;
use crate::data::OutOfRange;
use crate::epoch_sweep::EpochRow;
use crate::metrics::Metrics;
use crate::numeric_model::{ModelKind, NumericConfig};
use crate::schema::ModelSchema;
use crate::sweep::{SweepAxes, SweepObjective, SweepRow};
use crate::tnum::PrepareSpec;
use crate::train::{TextTrainConfig, TrainConfig};

/// Источник данных для обучения.
#[derive(Clone)]
pub enum DataSource {
    Blackbox(String),
    File(String),
}

/// Результат диагностики (числа для UI).
pub struct DiagnosticsResult {
    pub overfit_loss: f32,
    pub extrapolation_rows: usize,
    pub extrapolation_total: usize,
    /// На признак: (доля смен знака остатка, tail/inner).
    pub residuals: Vec<(f32, f32)>,
    /// (среднее, макс) чувствительности — только для blackbox.
    pub sensitivity: Option<(f32, f32)>,
}

/// Метаданные KAN, безопасные для передачи из worker в UI. Сами тензоры и
/// модель остаются в worker-потоке; UI получает только размеры и выборки.
pub struct KanModelInfo {
    pub layer_dims: Vec<(usize, usize)>,
    pub domain: (f32, f32),
    /// Символьный фит требует исходные train-активации. Они есть только у
    /// модели, обученной в текущей сессии, а не у загруженного checkpoint-а.
    pub symbolic_available: bool,
}

/// Слабое символьное ребро, передаваемое в UI без тензоров.
pub struct KanWeakEdge {
    pub layer: usize,
    pub input: String,
    pub output: String,
    pub primitive: String,
    pub r2: f32,
}

/// Готовый результат symbolic extraction в исходных единицах данных.
pub struct KanSymbolicInfo {
    pub formulas: String,
    pub min_edge_r2: f32,
    pub mean_edge_r2: f32,
    /// Метрики формул на validation: `None` у модели из checkpoint-а.
    pub formula_metrics: Option<Metrics>,
    pub kan_r2: Option<f32>,
    pub weak_edges: Vec<KanWeakEdge>,
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
    SampleKanEdge {
        layer: usize,
        input: usize,
        output: usize,
        samples: usize,
    },
    ExtractKanSymbolic,
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
        parameter_count: usize,
    },
    Epoch {
        epoch: usize,
        loss: f32,
    },
    /// Завершение обучения: `Some` — метрики на validation, `None` — отменено.
    TrainDone {
        metrics: Option<Metrics>,
    },
    /// Модель готова к предсказанию (после обучения или загрузки `.bin`).
    /// Число входов и выходов берётся из схемы, отдельных полей для них нет.
    ModelReady {
        schema: ModelSchema,
        /// Нужен UI, чтобы предупредить о категориях без embedding.
        kind: ModelKind,
        source: String,
        parameter_count: usize,
        kan: Option<KanModelInfo>,
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
    KanEdgeCurve {
        layer: usize,
        input: usize,
        output: usize,
        points: Vec<(f32, f32)>,
    },
    KanSymbolic {
        result: KanSymbolicInfo,
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
