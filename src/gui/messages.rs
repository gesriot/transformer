//! Сообщения UI <-> worker. Только `Send`-данные; Rc-модели не
//! пересекают границу потока.

use crate::config::ModelConfig;
use crate::data::{NumericDataset, OutOfRange};
use crate::interpret::{InterpretProfile, InterpretReport};
use crate::markup::TableProfile;
use crate::metrics::Metrics;
use crate::numeric_model::{ModelKind, NumericConfig};
use crate::schema::ModelSchema;
use crate::split::{FinalEval, SplitPlan};
use crate::sweep::{SweepAxes, SweepObjective, SweepRow};
use crate::table::Table;
use crate::tnum::PrepareSpec;
use crate::train::{TextTrainConfig, TrainConfig};
use crate::training::EvalSchedule;
use crate::training::Phase;
use std::sync::Arc;

/// Откуда взялся активный набор данных.
///
/// Нужен не для чтения — данные уже прочитаны, — а для подписи в интерфейсе и
/// для диагностики: чувствительность исходного процесса считается только у
/// вызываемого чёрного ящика.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DatasetOrigin {
    Blackbox(String),
    /// `.tnum` со своей схемой.
    File(String),
    /// Таблица, размеченная пользователем в диалоге.
    Table(String),
}

impl DatasetOrigin {
    pub fn blackbox(&self) -> Option<&str> {
        match self {
            DatasetOrigin::Blackbox(name) => Some(name),
            _ => None,
        }
    }

    /// Короткая подпись для шапки: путь целиком там не помещается.
    pub fn short_name(&self) -> String {
        match self {
            DatasetOrigin::Blackbox(name) => format!("чёрный ящик: {name}"),
            DatasetOrigin::File(path) | DatasetOrigin::Table(path) => std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone()),
        }
    }
}

/// Отчёты конвейера интерпретации по фазам.
///
/// Их два, потому что смысл разный: у модели разработки видно, как прунинг
/// повлиял на validation, у финальной — какой стала структура.
#[derive(Clone, Debug)]
pub struct InterpretReports {
    pub development: InterpretReport,
    pub final_model: Option<InterpretReport>,
}

/// Готовый набор данных: значения, схема и происхождение.
///
/// В worker передаются именно данные, а не путь: иначе он открыл бы файл
/// заново через автоопределение и ручная разметка потерялась бы.
#[derive(Clone)]
pub struct PreparedData {
    pub origin: DatasetOrigin,
    pub data: Arc<NumericDataset>,
    pub schema: ModelSchema,
}

/// Результат диагностики (числа для UI).
pub struct DiagnosticsResult {
    pub overfit_loss: f32,
    pub extrapolation_rows: usize,
    pub extrapolation_total: usize,
    /// Набор, на котором считались остатки и экстраполяция.
    pub evaluation_label: String,
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
    /// Метрики формул на доступном наборе: `None` у модели из checkpoint-а.
    pub formula_metrics: Option<Metrics>,
    pub kan_r2: Option<f32>,
    /// Набор для сравнения формулы с KAN. У финальной модели это
    /// train+validation, поэтому такая метрика описывает fidelity, не обобщение.
    pub evaluation_label: Option<String>,
    pub weak_edges: Vec<KanWeakEdge>,
}

/// Происхождение итоговой validation-метрики development-модели.
#[derive(Clone, Copy)]
pub struct ValidationOrigin {
    pub plan: SplitPlan,
    pub init_seed: u64,
}

/// Команды UI -> worker.
pub enum Command {
    /// Открыть набор данных: сгенерировать чёрный ящик или прочитать `.tnum`.
    /// Чтение — в worker-е, дальше сессия работает с готовыми данными.
    OpenDataset {
        origin: DatasetOrigin,
    },
    TrainNumeric {
        data: PreparedData,
        split: SplitPlan,
        nc: NumericConfig,
        tcfg: TrainConfig,
        /// Когда снимать метрики на validation по ходу обучения. Кривая по
        /// эпохам — настройка обычного обучения, а не отдельный сценарий.
        eval: EvalSchedule,
        /// Конвейер интерпретации: применяется в обеих фазах или нигде.
        interpret: Option<InterpretProfile>,
        /// Переобучить выбранную конфигурацию на train+validation и один раз
        /// открыть test. Запрашивается только для финального обучения.
        final_phase: bool,
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
    /// Поиск конфигурации по активному набору данных. Единственная команда
    /// поиска: отдельный blackbox-перебор обходил активный датасет стороной.
    Search {
        data: PreparedData,
        split: SplitPlan,
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
    /// Открыть таблицу для разметки: чтение и профиль считаются в worker-е,
    /// дальше диалог работает с ними локально.
    OpenTable {
        path: String,
        has_header: bool,
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
        phase: Phase,
        epoch: usize,
        loss: f32,
        /// R² на validation — только в точках, заданных расписанием.
        val_r2: Option<f32>,
    },
    /// Завершение обучения. У development есть validation-метрики, у refit —
    /// финальный test; отмена помечается отдельно, чтобы не смешивать случаи.
    TrainDone {
        metrics: Option<Metrics>,
        /// Отчёты конвейера по фазам: развитие и финальная модель. В боксе,
        /// потому что иначе один этот вариант раздувает всё перечисление.
        interpret: Option<Box<InterpretReports>>,
        /// Поколоночные validation-метрики development-модели.
        per_output: Option<Vec<Metrics>>,
        validation_origin: Option<ValidationOrigin>,
        /// Единственный замер на test: есть только у финального обучения.
        final_eval: Option<FinalEval>,
        cancelled: bool,
    },
    /// Набор данных открыт и готов к работе.
    DatasetOpened {
        data: PreparedData,
    },
    /// Таблица прочитана и профилирована. Подсказки только заполняют начальное
    /// состояние диалога, решение остаётся за пользователем.
    TableOpened {
        path: String,
        has_header: bool,
        table: Box<Table>,
        profile: Box<TableProfile>,
        suggested_inputs: Option<usize>,
        suggested_categories: Vec<usize>,
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
        /// После обучения `TrainDone` уже установил метрики этой модели. При
        /// загрузке checkpoint-а метрик в файле нет, и старые надо очистить.
        keep_evaluation: bool,
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
    SearchStarted {
        total_configs: usize,
        total_runs: usize,
    },
    SearchRow {
        row: SweepRow,
    },
    SearchDone {
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
}
