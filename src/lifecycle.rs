//! Жизненный цикл модели: что проверено и что уже стоило раскрытия test.
//!
//! Протокол оценки (см. [`crate::split`]) держится на том, что test открывают
//! один раз. Форму же можно править сколько угодно, поэтому нужен ответ на два
//! вопроса: относится ли показанный validation-результат к тому, что сейчас
//! введено, и не потрачен ли test на этих данных.
//!
//! Оба вопроса решаются сравнением отпечатков, а не флагами: флаг «результат
//! актуален» пришлось бы сбрасывать в каждом месте, где меняется поле.
//!
//! Ревизия набора защищает только текущую сессию: повторное открытие того же
//! файла и перезапуск приложения дают новую ревизию и снимают запрет. Честная
//! защита требует отпечатка самих данных в checkpoint — это отдельный шаг.

use crate::interpret::InterpretProfile;
use crate::metrics::{EvalSource, Metrics};
use crate::numeric_model::NumericConfig;
use crate::split::{FinalEval, SplitPlan};
use crate::train::TrainConfig;
use crate::training::EvalSchedule;

/// Кандидат целиком: разрешённая конфигурация модели, параметры обучения и
/// профиль интерпретации.
///
/// Профиль входит в кандидата, а не идёт сбоку: прунинг меняет саму модель, и
/// результат «с профилем» относится к другому кандидату, чем без него.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSpec {
    pub config: NumericConfig,
    pub train: TrainConfig,
    pub eval: EvalSchedule,
    pub interpret: Option<InterpretProfile>,
}

/// Отпечаток запуска: на каких данных, с каким разбиением и какого кандидата.
///
/// Разбиение входит сюда наравне с кандидатом: те же гиперпараметры при другом
/// split — другой результат, и выдавать один за другой нельзя.
#[derive(Clone, Debug, PartialEq)]
pub struct RunStamp {
    pub dataset_revision: u64,
    pub split: SplitPlan,
    pub candidate: CandidateSpec,
}

/// Результат фазы разработки: то, по чему принимают решения.
#[derive(Clone, Debug)]
pub struct CheckEval {
    pub metrics: Metrics,
    pub per_output: Vec<Metrics>,
    /// `Validation` у holdout, `Cv { k }` у K-fold. `Test` сюда не попадает.
    pub source: EvalSource,
}

/// Проверенный кандидат: отпечаток и то, что показал validation.
#[derive(Clone, Debug)]
pub struct CheckedRun {
    pub stamp: RunStamp,
    pub eval: CheckEval,
}

/// Состоявшееся раскрытие test: на какой ревизии данных, каким кандидатом и с
/// каким результатом.
#[derive(Clone, Debug)]
pub struct TestDisclosure {
    pub dataset_revision: u64,
    pub stamp: RunStamp,
    pub eval: FinalEval,
}

/// Почему финальное обучение сейчас запрещено.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalizeRefusal {
    /// Кандидат ещё не проверен на validation.
    NotChecked,
    /// Проверка относится к другой конфигурации, профилю или разбиению.
    Stale,
    /// Ровно этот кандидат уже финализирован — есть готовый результат.
    AlreadyFinalized,
    /// Test на этой ревизии данных уже открыт другим кандидатом.
    TestDisclosed,
}

impl FinalizeRefusal {
    pub fn message(self) -> &'static str {
        match self {
            FinalizeRefusal::NotChecked => {
                "сначала «Проверить конфигурацию»: финальное обучение идёт только по \
                 проверенному кандидату"
            }
            FinalizeRefusal::Stale => {
                "конфигурация, профиль интерпретации или разбиение изменились после проверки — \
                 проверьте заново"
            }
            FinalizeRefusal::AlreadyFinalized => {
                "этот кандидат уже обучен финально: test открыт, результат показан ниже"
            }
            FinalizeRefusal::TestDisclosed => {
                "test на этих данных уже открыт другой конфигурацией. Выбор по нему больше не \
                 является честной оценкой: возьмите новые данные"
            }
        }
    }
}

/// Что известно про текущего кандидата и про потраченный test.
///
/// Проверка живёт ровно до изменения отпечатка, раскрытие test — до смены
/// данных: правка формы не возвращает потраченный замер.
#[derive(Debug, Default)]
pub struct Lifecycle {
    checked: Option<CheckedRun>,
    disclosed: Option<TestDisclosure>,
}

impl Lifecycle {
    /// Запомнить результат проверки. Прежняя проверка вытесняется: актуальна
    /// всегда последняя.
    pub fn record_check(&mut self, run: CheckedRun) {
        self.checked = Some(run);
    }

    /// Проверка ровно этого отпечатка, если она есть.
    pub fn checked_for(&self, stamp: &RunStamp) -> Option<&CheckedRun> {
        self.checked.as_ref().filter(|run| run.stamp == *stamp)
    }

    /// Есть ли проверка, и относится ли она к текущему отпечатку.
    pub fn check_is_stale(&self, stamp: &RunStamp) -> bool {
        matches!(&self.checked, Some(run) if run.stamp != *stamp)
    }

    pub fn disclosure(&self) -> Option<&TestDisclosure> {
        self.disclosed.as_ref()
    }

    /// Результат финального обучения ровно этого кандидата.
    pub fn disclosure_for(&self, stamp: &RunStamp) -> Option<&TestDisclosure> {
        self.disclosed.as_ref().filter(|d| d.stamp == *stamp)
    }

    /// Можно ли открывать test под этот отпечаток.
    pub fn can_finalize(&self, stamp: &RunStamp) -> Result<(), FinalizeRefusal> {
        if let Some(disclosed) = &self.disclosed {
            if disclosed.dataset_revision == stamp.dataset_revision {
                return Err(if disclosed.stamp == *stamp {
                    FinalizeRefusal::AlreadyFinalized
                } else {
                    FinalizeRefusal::TestDisclosed
                });
            }
        }
        match &self.checked {
            Some(run) if run.stamp == *stamp => Ok(()),
            Some(_) => Err(FinalizeRefusal::Stale),
            None => Err(FinalizeRefusal::NotChecked),
        }
    }

    /// Записать состоявшееся раскрытие test.
    ///
    /// Отказ здесь означал бы, что замер уже сделан, а мы делаем вид, что нет:
    /// раскрытие фиксируется всегда, а решение «пускать ли» принимает
    /// [`Lifecycle::can_finalize`] ДО запуска.
    pub fn record_disclosure(&mut self, disclosure: TestDisclosure) {
        self.disclosed = Some(disclosure);
    }

    /// Данные сменились: проверка относится к прежнему набору, а раскрытие
    /// остаётся историей — оно привязано к своей ревизии и на новой не мешает.
    pub fn on_dataset_changed(&mut self) {
        self.checked = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::encoders::ValueEncoderConfig;
    use crate::numeric_model::{KanConfig, ModelKind};
    use crate::split::FinalOrigin;
    use crate::train::LrSchedule;

    fn metrics() -> Metrics {
        Metrics {
            rmse: 1.0,
            mae: 0.5,
            rel_error: 0.1,
            r2: 0.9,
        }
    }

    fn candidate(width: usize) -> CandidateSpec {
        CandidateSpec {
            config: NumericConfig {
                kind: ModelKind::Mlp,
                transformer: ModelConfig::default(),
                value: ValueEncoderConfig::default(),
                mlp_width: width,
                mlp_layers: 2,
                kan: KanConfig::default(),
            },
            train: TrainConfig {
                epochs: 10,
                batch_size: 32,
                lr: 1e-3,
                seed: 0,
                schedule: LrSchedule::Constant,
            },
            eval: EvalSchedule::Never,
            interpret: None,
        }
    }

    fn stamp(revision: u64, candidate: CandidateSpec) -> RunStamp {
        RunStamp {
            dataset_revision: revision,
            split: SplitPlan::default(),
            candidate,
        }
    }

    fn checked(stamp: RunStamp) -> CheckedRun {
        CheckedRun {
            stamp,
            eval: CheckEval {
                metrics: metrics(),
                per_output: vec![metrics()],
                source: EvalSource::Validation,
            },
        }
    }

    fn disclosure(stamp: RunStamp) -> TestDisclosure {
        TestDisclosure {
            dataset_revision: stamp.dataset_revision,
            stamp,
            eval: FinalEval {
                metrics: metrics(),
                per_output: vec![metrics()],
                origin: FinalOrigin {
                    test_rows: 18,
                    final_init_seed: 0,
                    plan: SplitPlan::default(),
                },
            },
        }
    }

    #[test]
    fn finalize_requires_a_check_of_exactly_this_candidate() {
        let mut life = Lifecycle::default();
        let current = stamp(1, candidate(16));
        assert_eq!(
            life.can_finalize(&current),
            Err(FinalizeRefusal::NotChecked)
        );

        life.record_check(checked(current.clone()));
        assert!(life.can_finalize(&current).is_ok());
        assert!(life.checked_for(&current).is_some());
        assert!(!life.check_is_stale(&current));
    }

    #[test]
    fn changing_the_candidate_split_or_profile_makes_the_check_stale() {
        let base = stamp(1, candidate(16));
        let mut life = Lifecycle::default();
        life.record_check(checked(base.clone()));

        // Другая ширина модели.
        let other_model = stamp(1, candidate(32));
        assert_eq!(life.can_finalize(&other_model), Err(FinalizeRefusal::Stale));
        assert!(life.check_is_stale(&other_model));
        assert!(life.checked_for(&other_model).is_none());

        // Другой профиль интерпретации при той же модели.
        let mut with_profile = base.clone();
        with_profile.candidate.interpret = Some(InterpretProfile::v1());
        assert_eq!(
            life.can_finalize(&with_profile),
            Err(FinalizeRefusal::Stale)
        );

        // Другое разбиение.
        let mut other_split = base.clone();
        other_split.split = SplitPlan::KFold {
            k: 5,
            folds_seed: 1,
            test_frac: 0.15,
            test_seed: 1,
        };
        assert_eq!(life.can_finalize(&other_split), Err(FinalizeRefusal::Stale));

        // Исходный кандидат по-прежнему проверен.
        assert!(life.can_finalize(&base).is_ok());
    }

    #[test]
    fn test_stays_disclosed_for_the_whole_dataset_revision() {
        let first = stamp(1, candidate(16));
        let mut life = Lifecycle::default();
        life.record_check(checked(first.clone()));
        life.record_disclosure(disclosure(first.clone()));

        // Тот же кандидат: повторно не учим, показываем готовый результат.
        assert_eq!(
            life.can_finalize(&first),
            Err(FinalizeRefusal::AlreadyFinalized)
        );
        assert!(life.disclosure_for(&first).is_some());

        // Другой кандидат на тех же данных: test уже потрачен.
        let second = stamp(1, candidate(32));
        life.record_check(checked(second.clone()));
        assert_eq!(
            life.can_finalize(&second),
            Err(FinalizeRefusal::TestDisclosed)
        );
        assert!(life.disclosure_for(&second).is_none());
        // Проверка кандидата при этом остаётся законной.
        assert!(life.checked_for(&second).is_some());
    }

    #[test]
    fn a_new_dataset_revision_clears_the_check_but_keeps_the_history() {
        let old = stamp(1, candidate(16));
        let mut life = Lifecycle::default();
        life.record_check(checked(old.clone()));
        life.record_disclosure(disclosure(old.clone()));

        life.on_dataset_changed();
        assert!(life.checked_for(&old).is_none());
        // Раскрытие никуда не делось: оно относится к прежней ревизии.
        assert!(life.disclosure().is_some());

        // На новых данных запрет не действует, но проверка нужна заново.
        let fresh = stamp(2, candidate(16));
        assert_eq!(life.can_finalize(&fresh), Err(FinalizeRefusal::NotChecked));
        life.record_check(checked(fresh.clone()));
        assert!(life.can_finalize(&fresh).is_ok());
    }

    #[test]
    fn editing_the_form_does_not_return_a_spent_test() {
        let first = stamp(1, candidate(16));
        let mut life = Lifecycle::default();
        life.record_check(checked(first.clone()));
        life.record_disclosure(disclosure(first));

        // Любая правка формы на той же ревизии упирается в потраченный test,
        // а не в «нужна проверка».
        let edited = stamp(1, candidate(64));
        assert_eq!(
            life.can_finalize(&edited),
            Err(FinalizeRefusal::TestDisclosed)
        );
        life.record_check(checked(edited.clone()));
        assert_eq!(
            life.can_finalize(&edited),
            Err(FinalizeRefusal::TestDisclosed)
        );
    }
}
