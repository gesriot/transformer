//! Происхождение модели: что за данные, как выбиралась конфигурация, чем её
//! проверяли и чем измерили.
//!
//! Без этого загруженный `.bin` — только инференс: метрики и история остаются в
//! сессии, где обучали, и через неделю о модели нельзя сказать ничего, кроме
//! её формы. Отчёт делает checkpoint самодостаточным.
//!
//! Три происхождения подписаны раздельно и намеренно не сливаются в одно
//! «качество модели»:
//!
//! - [`Selection`] — как выбрали конфигурацию: вручную или поиском (и тогда с
//!   какой целью и по каким seed);
//! - [`CheckRecord`] — проверка кандидата: validation или CV по всем folds;
//! - [`FinalRecord`] — переобучение на train + validation и единственный замер
//!   на test.
//!
//! Оба seed берутся из [`RunStamp`]: `candidate.train.seed` — инициализация
//! проверки, `final_init_seed` — финального переобучения. Дублировать их
//! рядом нельзя: две копии одного числа рано или поздно разойдутся.

use crate::fingerprint::DatasetFingerprint;
use crate::interpret::InterpretReport;
use crate::lifecycle::RunStamp;
use crate::metrics::{EvalSource, Metrics};
use crate::schema::ModelSchema;
use crate::split::FinalEval;
use crate::training::{SearchObjective, TrainingHistory};

/// Версия отчёта. Секция необязательна, поэтому старый checkpoint даёт `None`,
/// а не «test точно не открывался»: отсутствие записи означает неизвестность.
pub const TRAINING_REPORT_VERSION: u32 = 1;

/// Как была выбрана конфигурация.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Selection {
    /// Гиперпараметры заданы человеком.
    Manual,
    /// Конфигурация выбрана перебором.
    Search {
        objective: SearchObjective,
        /// Seed инициализации, по которым усреднялся поиск.
        seeds: Vec<u64>,
        /// Значение цели у выбранной строки — по нему её и выбрали.
        objective_value: f32,
        /// Подпись строки ранжирования: по ней видно, что именно выбрано.
        label: String,
    },
}

/// Проверка кандидата: то, по чему принимали решение.
#[derive(Clone, Debug)]
pub struct CheckRecord {
    /// Чем является оценка: validation у holdout, CV у K-fold.
    pub source: EvalSource,
    pub metrics: Metrics,
    pub per_output: Vec<Metrics>,
    /// Разброс R² между folds; 0 у holdout.
    pub r2_std_folds: f32,
    /// История обучения каждого fold — целиком, без прореживания.
    pub histories: Vec<TrainingHistory>,
    /// Отчёт конвейера каждого fold; пусто, если конвейера не просили.
    pub interpret: Vec<InterpretReport>,
}

/// Финальное переобучение и единственный замер на test.
#[derive(Clone, Debug)]
pub struct FinalRecord {
    /// История refit целиком. Validation в ней нет и быть не может: модель на
    /// ней училась.
    pub history: TrainingHistory,
    pub eval: FinalEval,
    /// Отчёт конвейера ИМЕННО сохраняемой модели.
    pub interpret: Option<InterpretReport>,
}

/// Полное происхождение модели.
///
/// Атрибута `non_exhaustive` у записей отчёта нет намеренно: их создаёт тот,
/// кто обучает, — включая бинарь, для которого библиотека внешняя.
#[derive(Clone, Debug)]
pub struct TrainingReport {
    /// Данные, на которых всё происходило.
    pub dataset: DatasetFingerprint,
    /// Схема целиком: имена и единицы в отпечаток не входят, но без них отчёт
    /// нечитаем.
    pub schema: ModelSchema,
    /// Разбиение, конфигурация кандидата и оба seed.
    pub stamp: RunStamp,
    pub selection: Selection,
    /// Проверка: `None`, если модель получена без неё (например, загружена и
    /// пересохранена).
    pub check: Option<CheckRecord>,
    /// Финальный замер: `None` у отладочной модели.
    pub final_run: Option<FinalRecord>,
}

impl TrainingReport {
    /// Открывался ли test для этих данных. Именно этот факт восстанавливает
    /// бюджет после перезапуска.
    pub fn test_disclosed(&self) -> bool {
        self.final_run.is_some()
    }

    /// Короткая подпись для интерфейса.
    pub fn describe(&self) -> String {
        let selection = match &self.selection {
            Selection::Manual => "ручной выбор".to_string(),
            Selection::Search { objective, .. } => format!("поиск по {}", objective.label()),
        };
        let evaluation = match (&self.check, &self.final_run) {
            (_, Some(final_run)) => format!(
                "test {} строк, R² {:.5}",
                final_run.eval.origin.test_rows, final_run.eval.metrics.r2
            ),
            (Some(check), None) => format!("{}, R² {:.5}", check.source.label(), check.metrics.r2),
            (None, None) => "без оценок".to_string(),
        };
        format!("{selection}; {evaluation}; данные {}", self.dataset.short())
    }
}
