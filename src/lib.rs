//! Surrogate-модели по числовым таблицам: подготовка данных, обучение, поиск
//! конфигурации, интерпретация KAN и прогноз.
//!
//! Публичный API — корневые `pub use` в конце файла плюс модули `diagnostics`,
//! `interpret` и `predict`, оставленные пространствами имён. Всё остальное —
//! устройство расчёта, и наружу не обещано.

pub mod batch_predict;
// Встроенные ящики — демонстрация, но ими же порождаются данные для тестов
// ядра: под `test` они доступны и без фичи, иначе каждому модулю пришлось бы
// заводить свой генератор.
#[cfg(any(feature = "demo", test))]
pub mod blackbox;
pub mod config;
pub mod core;
pub mod data;
pub mod diagnostics;
pub mod encoders;
#[cfg(feature = "demo")]
pub mod generate;
#[cfg(feature = "gui")]
pub mod gui;
pub mod heads;
pub mod init;
pub mod interpret;
pub mod kan;
pub mod loss;
pub mod markup;
pub mod metrics;
pub mod mlp;
pub mod nn;
pub mod numeric_model;
pub mod ops;
pub mod optim;
pub mod predict;
pub mod schema;
pub mod serialize;
pub mod split;
pub mod surrogate;
pub mod sweep;
pub mod symbolic;
pub mod table;
pub mod tensor;
#[cfg(feature = "demo")]
pub mod textmodel;
pub mod tnum;
pub mod train;
pub mod training;

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
pub use table::Delimiter;
pub use tnum::{
    infer_prepare_spec_from_path, infer_prepare_spec_from_text, parse_categorical,
    read_numeric_source, table_path_to_tnum, table_to_tnum, InferredPrepareSpec, PrepareSpec,
};

// Модель и её конфигурация.
pub use config::ModelConfig;
pub use numeric_model::{validate_numeric, KanConfig, ModelKind, NumericConfig, NumericModel};

// Обучение: активный набор, один сценарий и его протокол оценки.
pub use metrics::{evaluate, EvalSource, Metrics};
pub use split::{
    FinalEval, FinalOrigin, HoldoutTest, PreparedSplit, SearchPool, SplitPlan, DEFAULT_DATA_SEED,
    DEFAULT_FINAL_INIT_SEED, DEFAULT_K, DEFAULT_SPLIT_SEED, DEFAULT_TEST_FRAC, DEFAULT_TRAIN_FRAC,
    DEFAULT_VAL_FRAC,
};
pub use train::{evaluate_surrogate, predict_dataset, validate_train, LrSchedule, TrainConfig};
pub use training::{
    evaluate_on, recommended_epoch, refit, run_training, Dataset, EarlyStopping, EpochPoint,
    EvalSchedule, Phase, RefitOutcome, TrainedModel, TrainingHistory, TrainingOutcome,
    TrainingSetup,
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
pub use symbolic::{symbolize, SymbolicKan, SymbolicLayer};

// Интерпретация KAN: профиль версионируется и едет в checkpoint.
pub use interpret::{InterpretOverrides, InterpretProfile, InterpretReport};

// Демонстрации.
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
