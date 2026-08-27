//! Surrogate-модели по числовым таблицам: подготовка данных, обучение, поиск
//! конфигурации, интерпретация KAN и прогноз.
//!
//! Публичный API — корневые `pub use` в конце файла плюс модули `diagnostics`,
//! `interpret` и `predict`, оставленные пространствами имён. При включённой
//! фиче `gui` публичен также модуль `gui`. Всё остальное — устройство расчёта,
//! и наружу не обещано.
//!
//! Возвращаемые отчёты и растущие enum помечены `#[non_exhaustive]`: они
//! пополняются по мере развития расчёта, и внешний `match` не должен ломаться
//! от нового поля или варианта. Конфигурационные структуры (`TrainConfig`,
//! `NumericConfig`, `ModelConfig`, `SweepAxes`, `KanConfig`, `TrainingSetup`)
//! атрибута не несут: их создают литералом, и запрет на это сделал бы API
//! неудобным без конструкторов.
//!
//! `unreachable_pub` включён намеренно: он ловит `pub`, который на самом деле
//! никуда не ведёт, — такой элемент либо часть контракта и должен попасть в
//! фасад, либо внутренний и должен стать `pub(crate)`.
#![warn(unreachable_pub)]

mod atomic_write;
mod batch_predict;
// Встроенные ящики — демонстрация, но ими же порождаются данные для тестов
// ядра: под `test` они доступны и без фичи, иначе каждому модулю пришлось бы
// заводить свой генератор.
#[cfg(any(feature = "demo", test))]
mod blackbox;
mod config;
mod core;
mod data;
pub mod diagnostics;
mod encoders;
#[cfg(feature = "demo")]
mod generate;
#[cfg(feature = "gui")]
pub mod gui;
mod heads;
mod init;
pub mod interpret;
mod kan;
mod lifecycle;
mod loss;
mod markup;
mod metrics;
mod mlp;
mod nn;
mod numeric_model;
mod ops;
mod optim;
pub mod predict;
mod schema;
mod serialize;
mod split;
mod surrogate;
mod sweep;
mod symbolic;
mod table;
mod tensor;
#[cfg(feature = "demo")]
mod textmodel;
mod tnum;
mod train;
mod training;

// --- Публичный API ---
//
// Всё, что крейт обещает наружу, перечислено здесь: bin-цель обращается к
// библиотеке как внешний потребитель, поэтому список ниже — это и есть
// поддерживаемая поверхность. Модули `diagnostics`, `interpret` и `predict`
// остаются пространствами имён: их короткие глаголы (`resolve`, `sensitivity`,
// `parse_row`) без префикса читаются неоднозначно.

// Данные и схема.
pub use data::{Normalizer, NumericDataset, OutOfRange};
pub use encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
pub use schema::{Column, ColumnRole, ColumnType, ModelSchema, TableSchema};
// Разметка таблицы: профиль -> черновик схемы -> отчёт по ролям. Ядро
// независимо от интерфейса, поэтому идёт наружу целиком, а не через GUI.
pub use markup::{
    analyze_roles, ColumnProfile, DraftColumn, DraftType, LinearDependency, Message, RoleReport,
    SchemaDraft, Severity, TableProfile,
};
pub use table::{Delimiter, Table};
pub use tnum::{
    infer_prepare_spec_from_path, infer_prepare_spec_from_text, parse_categorical,
    prepare_tnum_file, read_numeric_source, table_path_to_tnum, table_to_tnum, InferredPrepareSpec,
    PrepareSpec, PrepareStats,
};

// Модель и её конфигурация. Полезная нагрузка `NumericModel` называется явно:
// она и так видна через варианты enum, а безымянный тип нельзя ни принять, ни
// вернуть. Тензорные методы у этих типов внутренние — снаружи модель работает
// через `predict`, `save_numeric` и отчёты.
pub use config::ModelConfig;
pub use kan::{CompactReport, KanNet, PruneReport};
pub use mlp::MlpBaseline;
pub use numeric_model::{validate_numeric, KanConfig, ModelKind, NumericConfig, NumericModel};
pub use surrogate::SurrogateModel;

// Обучение: активный набор, один сценарий и его протокол оценки.
pub use metrics::{evaluate, EvalSource, Metrics};
pub use split::{
    FinalEval, FinalOrigin, HoldoutTest, PreparedSplit, SearchPool, SplitPlan, DEFAULT_DATA_SEED,
    DEFAULT_FINAL_INIT_SEED, DEFAULT_K, DEFAULT_SPLIT_SEED, DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
};
pub use train::{evaluate_surrogate, predict_dataset, validate_train, LrSchedule, TrainConfig};
pub use training::{
    evaluate_on, recommended_epoch, refit, run_training, ConfigureModel, Dataset, EarlyStopping,
    EpochPoint, EvalSchedule, Phase, PostTrain, RefitOutcome, TrainedModel, TrainingHistory,
    TrainingOutcome, TrainingSetup,
};

// Жизненный цикл: что проверено и не потрачен ли test на этих данных.
// Дисциплина «test открывают один раз» — часть протокола оценки, а не деталь
// интерфейса, поэтому контракт состояния идёт наружу вместе с ним.
pub use lifecycle::{
    CandidateSpec, CheckEval, CheckedRun, FinalizeRefusal, Lifecycle, RunStamp, TestDisclosure,
};

// Поиск конфигурации: сетка поверх того же сценария обучения.
pub use sweep::{
    row_score, run_sweep, sort_rows, sweep_cost, sweep_size, validate_axes, SearchBudget,
    SweepAxes, SweepChoice, SweepResult, SweepRow,
};
// Цель и цена поиска общие с ядром: `sweep` — только сетка поверх него,
// поэтому наружу идёт одно имя, а не два синонима.
pub use training::{SearchCost, SearchObjective};

// Готовая модель: checkpoint, формулы, прогноз по таблице.
pub use batch_predict::{export_predictions, ExportSummary};
pub use init::set_init_seed;
pub use serialize::{
    calibration_sample, load_numeric, load_numeric_full, save_numeric, NumericCheckpoint,
};
pub use symbolic::{symbolize, EdgeFit, SymbolicKan, SymbolicLayer};

// Интерпретация KAN: профиль версионируется и едет в checkpoint.
pub use interpret::{InterpretOverrides, InterpretProfile, InterpretReport};

// Демонстрации.
// Ящики компилируются и под `test` (см. объявление модуля), поэтому и их
// реэкспорт: иначе в тестовой сборке без `demo` их `pub` вёл бы в никуда.
#[cfg(any(feature = "demo", test))]
pub use blackbox::{by_name as blackbox_by_name, BlackBox};
#[cfg(feature = "demo")]
pub use data::{TextDataset, Vocab};
#[cfg(feature = "demo")]
pub use generate::generate;
#[cfg(feature = "demo")]
pub use serialize::{load_text, save_text};
#[cfg(feature = "demo")]
pub use sweep::{run_blackbox_sweep, run_blackbox_sweep_with_objective};
#[cfg(feature = "demo")]
pub use textmodel::TextModel;
#[cfg(feature = "demo")]
pub use train::{train_text, train_text_cb, TextTrainConfig};

#[cfg(test)]
pub(crate) mod gradcheck;
