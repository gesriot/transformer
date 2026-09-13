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
//! Оба seed берутся из [`RunIdentity`]: `candidate.train.seed` — инициализация
//! проверки, `final_init_seed` — финального переобучения. Дублировать их
//! рядом нельзя: две копии одного числа рано или поздно разойдутся.

use crate::fingerprint::{DatasetFingerprint, ModelFingerprint};
use crate::interpret::InterpretProfile;
use crate::interpret::InterpretReport;
use crate::lifecycle::RunIdentity;
use crate::metrics::{EvalSource, Metrics};
use crate::numeric_model::NumericConfig;
use crate::schema::ModelSchema;
use crate::split::FinalEval;
use crate::training::{SearchObjective, TrainingHistory};
use std::collections::BTreeSet;

/// Версия отчёта. Секция необязательна, поэтому старый checkpoint даёт `None`,
/// а не «test точно не открывался»: отсутствие записи означает неизвестность.
///
/// v2 добавила отпечаток модели, v3 — нормализованные метрики и необязательную
/// относительную ошибку, v4 — разброс между повторами CV. Старые версии
/// читаются как есть: повысить их молча значило бы утверждать то, чего та
/// версия не наблюдала — ни связи с весами, ни масштаба train, по которому
/// только и считаются nMAE и nRMSE, ни разброса между разбиениями.
pub const TRAINING_REPORT_VERSION: u32 = TRAINING_REPORT_VERSION_V4;

/// Первая версия отчёта — без отпечатка модели.
pub const TRAINING_REPORT_VERSION_V1: u32 = 1;

/// Вторая версия — с отпечатком модели, но со старыми метриками.
pub const TRAINING_REPORT_VERSION_V2: u32 = 2;

/// Третья версия — нормализованные метрики, но без разброса между повторами.
pub const TRAINING_REPORT_VERSION_V3: u32 = 3;

/// Четвёртая версия — повторённая CV и отдельный разброс между повторами.
///
/// Именованная граница нужна reader-у и writer-у: сравнение с подвижной
/// [`TRAINING_REPORT_VERSION`] сломало бы чтение v4 при появлении v5.
pub const TRAINING_REPORT_VERSION_V4: u32 = 4;

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
    /// Разброс R² между folds внутри одного повтора; 0 у holdout.
    pub r2_std_folds: f32,
    /// Разброс R² между повторами; 0 везде, кроме повторённой CV.
    pub r2_std_repeats: f32,
    /// История обучения каждого разбиения — целиком, без прореживания.
    pub histories: Vec<TrainingHistory>,
    /// Отчёт конвейера каждого разбиения; пусто, если конвейера не просили.
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
    /// Отпечаток самой модели. `None` у отчёта v1: связь с весами там не
    /// проверялась, и делать вид, что проверялась, нельзя.
    pub model: Option<ModelFingerprint>,
    /// Схема целиком: имена и единицы в отпечаток не входят, но без них отчёт
    /// нечитаем.
    pub schema: ModelSchema,
    /// Разбиение, конфигурация кандидата и оба seed.
    pub stamp: RunIdentity,
    pub selection: Selection,
    /// Проверка: `None`, если модель получена без неё (например, загружена и
    /// пересохранена).
    pub check: Option<CheckRecord>,
    /// Финальный замер: `None` у отладочной модели.
    pub final_run: Option<FinalRecord>,
}

impl TrainingReport {
    /// Проверить отчёт против самого checkpoint.
    ///
    /// Вызывается и перед записью, и после чтения: отчёт, противоречащий
    /// модели, рядом с которой лежит, хуже отсутствующего — он выглядит как
    /// достоверное происхождение.
    pub fn validate_against(
        &self,
        model: ModelFingerprint,
        config: &NumericConfig,
        schema: &ModelSchema,
        interpret: Option<&InterpretProfile>,
    ) -> Result<(), String> {
        // Отчёт v1 отпечатка не несёт: он связан с моделью только конфигурацией
        // и схемой, и это видно в интерфейсе.
        if let Some(expected) = self.model {
            if expected != model {
                return Err(format!(
                    "отчёт описывает другую модель: ожидался отпечаток {}, у модели {}",
                    expected.short(),
                    model.short()
                ));
            }
        }
        if self.dataset != self.stamp.dataset {
            return Err("отчёт и его личность запуска описывают разные данные".to_string());
        }
        self.stamp
            .split
            .validate_parameters()
            .map_err(|error| format!("некорректное разбиение отчёта: {error}"))?;
        if self.schema != *schema {
            return Err("схема отчёта не совпадает со схемой checkpoint".to_string());
        }
        if self.stamp.candidate.config != *config {
            return Err("конфигурация кандидата не совпадает с сохраняемой моделью".to_string());
        }
        if self.stamp.candidate.interpret.as_ref() != interpret {
            return Err(
                "профиль интерпретации в отчёте не совпадает с профилем модели".to_string(),
            );
        }
        if let Selection::Search { seeds, .. } = &self.selection {
            if seeds.is_empty() {
                return Err("поиск записан без seeds: усреднять было не по чему".to_string());
            }
            let unique: BTreeSet<u64> = seeds.iter().copied().collect();
            if unique.len() != seeds.len() {
                return Err("seeds поиска содержат повторы".to_string());
            }
        }
        // Разбиений столько, сколько обучений требует проверка: у повторённой
        // CV это k × repeats, а не k.
        let folds = self.stamp.split.n_splits();
        if let Some(check) = &self.check {
            if check.source != self.stamp.eval_source() {
                return Err(format!(
                    "проверка подписана как {}, а разбиение даёт {}",
                    check.source.label(),
                    self.stamp.eval_source().label()
                ));
            }
            if check.histories.len() != folds {
                return Err(format!(
                    "историй проверки {}, а разбиений {folds}",
                    check.histories.len()
                ));
            }
            // Разброс между повторами при единственном разбиении описывать
            // нечему: такое число означает, что запись собрана не из того
            // прогона, которым подписана.
            if check.source.repeats() < 2 && check.r2_std_repeats != 0.0 {
                return Err(format!(
                    "проверка {} не имеет повторов, но записан разброс между ними",
                    check.source.label()
                ));
            }
            // Правило точное: с профилем — отчёт на каждый fold, без профиля
            // — ни одного. «Пусто при заданном профиле» означало бы, что
            // конвейер просили, но что он сделал — неизвестно.
            let expected = if self.stamp.candidate.interpret.is_some() {
                folds
            } else {
                0
            };
            if check.interpret.len() != expected {
                return Err(format!(
                    "отчётов конвейера у проверки {}, а ожидается {expected}",
                    check.interpret.len()
                ));
            }
            for history in &check.histories {
                if history.source != check.source {
                    return Err(format!(
                        "история проверки подписана как {}, а сама проверка — как {}",
                        history.source.label(),
                        check.source.label()
                    ));
                }
            }
            if check.per_output.len() != schema.n_outputs() {
                return Err(format!(
                    "поколоночных метрик проверки {}, а выходов {}",
                    check.per_output.len(),
                    schema.n_outputs()
                ));
            }
            validate_reports(&check.interpret, self.stamp.candidate.interpret.as_ref())?;
        }
        if let Some(final_run) = &self.final_run {
            // Финальная модель существует только после проверки: test
            // открывают по разрешённому кандидату, а не по любому.
            if self.check.is_none() {
                return Err("финальный замер без проверки кандидата".to_string());
            }
            if final_run.eval.origin.plan != self.stamp.split {
                return Err("test и личность запуска описывают разные разбиения".to_string());
            }
            if final_run.eval.origin.final_init_seed != self.stamp.final_init_seed {
                return Err("final seed в замере не совпадает с личностью запуска".to_string());
            }
            if final_run.history.source != self.stamp.eval_source() {
                return Err(format!(
                    "история финального переобучения подписана как {}, а разбиение даёт {}",
                    final_run.history.source.label(),
                    self.stamp.eval_source().label()
                ));
            }
            if final_run.interpret.is_some() != self.stamp.candidate.interpret.is_some() {
                return Err(
                    "отчёт конвейера финальной модели не соответствует запрошенному профилю"
                        .to_string(),
                );
            }
            // Refit учится на train + validation: validation-метрик в его
            // истории быть не может — они означали бы замер по своим же
            // обучающим данным.
            if final_run.history.points.iter().any(|p| p.val.is_some()) {
                return Err(
                    "история финального переобучения содержит validation-метрики".to_string(),
                );
            }
            if final_run.eval.per_output.len() != schema.n_outputs() {
                return Err(format!(
                    "поколоночных метрик test {}, а выходов {}",
                    final_run.eval.per_output.len(),
                    schema.n_outputs()
                ));
            }
            validate_reports(
                final_run.interpret.as_slice(),
                self.stamp.candidate.interpret.as_ref(),
            )?;
        }
        Ok(())
    }

    /// Проверена ли связь отчёта с весами модели. У отчётов v1 — нет.
    pub fn weights_verified(&self) -> bool {
        self.model.is_some()
    }

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

/// Сохранённый отчёт конвейера обязан быть завершённым и относиться к тому же
/// профилю: прерванный конвейер оставляет модель в промежуточном состоянии, и
/// описывать её как результат нельзя.
fn validate_reports(
    reports: &[InterpretReport],
    profile: Option<&InterpretProfile>,
) -> Result<(), String> {
    for report in reports {
        if report.cancelled {
            return Err("в отчёте сохранён прерванный конвейер интерпретации".to_string());
        }
        if Some(&report.profile) != profile {
            return Err("отчёт конвейера описывает другой профиль".to_string());
        }
    }
    Ok(())
}
