//! Сообщения UI <-> worker. Только `Send`-данные; Rc-модели не
//! пересекают границу потока.

use crate::batch_predict::ExportSummary;
#[cfg(feature = "demo")]
use crate::config::ModelConfig;
use crate::data::{NumericDataset, OutOfRange};
use crate::diagnostics::SensitivityReport;
use crate::interpret::InterpretReport;
use crate::lifecycle::RunStamp;
use crate::markup::TableProfile;
use crate::metrics::{EvalSource, Metrics};
use crate::numeric_model::ModelKind;
use crate::schema::ModelSchema;
use crate::split::{FinalEval, SplitPlan};
use crate::sweep::{SweepAxes, SweepObjective, SweepRow};
use crate::table::Table;
use crate::tnum::PrepareSpec;
#[cfg(feature = "demo")]
use crate::train::TextTrainConfig;
use crate::training::EvalSchedule;
use crate::training::Phase;
use std::sync::Arc;

/// Откуда взялся активный набор данных.
///
/// Нужен не для чтения — данные уже прочитаны, — а для подписи в интерфейсе и
/// для диагностики: чувствительность исходного процесса считается только у
/// вызываемого чёрного ящика.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DatasetOrigin {
    #[cfg(any(feature = "demo", test))]
    Blackbox(String),
    /// `.tnum` со своей схемой.
    File(String),
    /// Таблица, размеченная пользователем в диалоге.
    Table(String),
}

impl DatasetOrigin {
    #[cfg(feature = "demo")]
    pub(crate) fn blackbox(&self) -> Option<&str> {
        match self {
            DatasetOrigin::Blackbox(name) => Some(name),
            _ => None,
        }
    }

    /// Короткая подпись для шапки: путь целиком там не помещается.
    pub(crate) fn short_name(&self) -> String {
        match self {
            #[cfg(any(feature = "demo", test))]
            DatasetOrigin::Blackbox(name) => format!("чёрный ящик: {name}"),
            DatasetOrigin::File(path) | DatasetOrigin::Table(path) => std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone()),
        }
    }
}

/// Готовый набор данных: значения, схема и происхождение.
///
/// В worker передаются именно данные, а не путь: иначе он открыл бы файл
/// заново через автоопределение и ручная разметка потерялась бы.
#[derive(Clone)]
pub(crate) struct PreparedData {
    pub origin: DatasetOrigin,
    pub data: Arc<NumericDataset>,
    pub schema: ModelSchema,
}

/// Результат диагностики (числа для UI).
pub(crate) struct DiagnosticsResult {
    pub overfit_loss: f32,
    pub extrapolation_rows: usize,
    pub extrapolation_total: usize,
    /// Набор, на котором считались остатки и экстраполяция.
    pub evaluation_label: String,
    /// На признак: (доля смен знака остатка, tail/inner).
    pub residuals: Vec<(f32, f32)>,
    /// (среднее, макс) чувствительности — только для blackbox.
    /// Чувствительность модели и — у демо-ящика — исходного процесса.
    /// `Err` — понятная причина, по которой замер невозможен.
    pub sensitivity: Result<SensitivityReport, String>,
}

/// Метаданные KAN, безопасные для передачи из worker в UI. Сами тензоры и
/// модель остаются в worker-потоке; UI получает только размеры и выборки.
pub(crate) struct KanModelInfo {
    pub layer_dims: Vec<(usize, usize)>,
    pub domain: (f32, f32),
    /// Символьный фит требует исходные train-активации. Они есть только у
    /// модели, обученной в текущей сессии, а не у загруженного checkpoint-а.
    pub symbolic_available: bool,
}

/// Слабое символьное ребро, передаваемое в UI без тензоров.
pub(crate) struct KanWeakEdge {
    pub layer: usize,
    pub input: String,
    pub output: String,
    pub primitive: String,
    pub r2: f32,
}

/// Готовый результат symbolic extraction в исходных единицах данных.
pub(crate) struct KanSymbolicInfo {
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

/// Точка кривой обучения, пригодная для передачи в UI.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CurvePoint {
    pub epoch: usize,
    pub train_loss: f32,
    /// R² на validation этого fold; есть только там, где расписание попросило
    /// замер.
    pub val_r2: Option<f32>,
}

/// Происхождение активной модели.
///
/// Разница не косметическая: отладочная модель обучена только на train, и
/// сохранять её как результат работы нельзя без явной подписи.
#[derive(Clone, Debug)]
pub(crate) enum ModelOrigin {
    /// Проверка кандидата: обучена на train, доступные данные использованы не
    /// полностью.
    Development(Box<RunStamp>),
    /// Финальная: train + validation, test открыт ровно один раз.
    Final(Box<RunStamp>),
    /// Загружена из checkpoint: происхождение известно только из файла.
    Checkpoint,
}

impl ModelOrigin {
    /// Подпись для интерфейса и для кнопки сохранения.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            ModelOrigin::Development(_) => "отладочная (обучена на train, без validation)",
            ModelOrigin::Final(_) => "финальная (train + validation)",
            ModelOrigin::Checkpoint => "загружена из файла",
        }
    }

    pub(crate) fn is_final(&self) -> bool {
        matches!(self, ModelOrigin::Final(_))
    }
}

/// Команды UI -> worker.
pub(crate) enum Command {
    /// Открыть набор данных: сгенерировать чёрный ящик или прочитать `.tnum`.
    /// Чтение — в worker-е, дальше сессия работает с готовыми данными.
    OpenDataset {
        origin: DatasetOrigin,
    },
    /// Проверить кандидата: фаза разработки и конвейер интерпретации, test не
    /// трогаем. У K-fold проверяются все folds, поэтому проверка означает
    /// CV-оценку, а не модель одного произвольного fold.
    CheckCandidate {
        data: PreparedData,
        stamp: Box<RunStamp>,
        /// Когда снимать метрики по ходу обучения. Настройка наблюдения, не
        /// личность кандидата: без ранней остановки она не меняет модель.
        eval: EvalSchedule,
    },
    /// Зафиксировать проверенного кандидата: refit на train+validation и
    /// единственный замер на test.
    ///
    /// Отпечаток приходит из проверки как есть, а не собирается заново по
    /// форме: иначе «зафиксировать» могло бы обучить не то, что проверяли.
    FinalizeCandidate {
        data: PreparedData,
        stamp: Box<RunStamp>,
    },
    LoadModel(String),
    SaveModel(String),
    Predict(Vec<f32>),
    /// Экспорт таблицы с прогнозами: результат — новая книга, исходная не
    /// сохраняется.
    ExportPredictions {
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
        /// В боксе: оси перебора — самый крупный вариант команды.
        axes: Box<SweepAxes>,
        objective: SweepObjective,
    },
    #[cfg(feature = "demo")]
    TrainText {
        path: String,
        model_cfg: ModelConfig,
        train_cfg: TextTrainConfig,
    },
    #[cfg(feature = "demo")]
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
pub(crate) enum Event {
    Status(String),
    Error(String),
    TrainStarted {
        total_epochs: usize,
        parameter_count: usize,
    },
    Epoch {
        phase: Phase,
        /// Номер fold: у holdout всегда 0, у K-fold растёт. Живая кривая по
        /// нескольким folds невозможна — номера эпох повторяются.
        fold: usize,
        epoch: usize,
        loss: f32,
        /// R² на validation — только в точках, заданных расписанием.
        val_r2: Option<f32>,
    },
    /// Завершение обучения. У development есть validation-метрики, у refit —
    /// финальный test; отмена помечается отдельно, чтобы не смешивать случаи.
    TrainDone {
        /// Отпечаток запуска, к которому относится результат. Форма могла
        /// измениться, пока шло обучение, — тогда ответ относится не к ней.
        /// В боксе: иначе конфигурация кандидата раздувает всё перечисление.
        stamp: Box<RunStamp>,
        metrics: Option<Metrics>,
        /// Отчёты конвейера ПРОВЕРКИ — по одному на fold. У финализации их
        /// нет: её отчёт принадлежит модели и едет вместе с ней.
        check_interpret: Vec<InterpretReport>,
        /// Поколоночные validation-метрики development-модели.
        per_output: Option<Vec<Metrics>>,
        /// Чем оценка ЯВЛЯЕТСЯ по факту прогона: источник берётся у пула,
        /// который реально считал, и сверяется с тем, что обещал отпечаток.
        check_source: Option<EvalSource>,
        /// Разброс R² между folds у CV-проверки; `None` у holdout и финала.
        r2_std_folds: Option<f32>,
        /// Кривые обучения ПО FOLD-ам, по одной на каждый. Склеивать их в одну
        /// ломаную нельзя: номера эпох повторяются, и результат не описывает
        /// ни один прогон.
        curves: Vec<Vec<CurvePoint>>,
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
        /// Чем является активная модель: отладочной, финальной или загруженной.
        /// Без этого «Сохранить модель» одинаково выглядит для модели, обученной
        /// на части данных, и для финальной.
        model_origin: ModelOrigin,
        parameter_count: usize,
        kan: Option<KanModelInfo>,
        /// Отчёт конвейера ЭТОЙ модели. Едет вместе с ней: иначе рядом с
        /// моделью оказывался бы отчёт последней проверки.
        interpret: Option<Box<InterpretReport>>,
    },
    PredictResult {
        outputs: Vec<f32>,
        extrapolation: Vec<OutOfRange>,
    },
    ExportDone {
        output: String,
        summary: ExportSummary,
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
    #[cfg(feature = "demo")]
    TextStarted {
        total_steps: usize,
    },
    #[cfg(feature = "demo")]
    TextProgress {
        step: usize,
        loss: f32,
    },
    #[cfg(feature = "demo")]
    TextDone {
        final_loss: Option<f32>,
        cancelled: bool,
        vocab_size: usize,
        seed_hint: String,
    },
    #[cfg(feature = "demo")]
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
