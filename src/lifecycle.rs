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
//! Идентичность данных — их отпечаток ([`DatasetFingerprint`]), а не номер
//! набора в сессии: повторное открытие того же файла потраченный test не
//! возвращает. Раскрытия помнятся по каждому набору отдельно, поэтому возврат
//! к прежним данным возвращает и запрет, а не только историю.
//!
//! Память при этом живёт в процессе: перезапуск приложения бюджет сбрасывает.
//! Восстановить его можно только из checkpoint — это отдельный шаг.

use crate::fingerprint::DatasetFingerprint;
use crate::interpret::{InterpretProfile, InterpretReport};
use crate::metrics::{EvalSource, Metrics};
use crate::numeric_model::NumericConfig;
use crate::split::{FinalEval, SplitPlan};
use crate::train::TrainConfig;

/// Кандидат целиком: разрешённая конфигурация модели, параметры обучения и
/// профиль интерпретации.
///
/// Профиль входит в кандидата, а не идёт сбоку: прунинг меняет саму модель, и
/// результат «с профилем» относится к другому кандидату, чем без него.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSpec {
    pub config: NumericConfig,
    /// Параметры обучения проверки. `train.seed` — seed инициализации ИМЕННО
    /// проверки; финальный переобучается со своим, см. [`RunStamp`].
    pub train: TrainConfig,
    pub interpret: Option<InterpretProfile>,
}

/// Отпечаток запуска: на каких данных, с каким разбиением и какого кандидата.
///
/// Разбиение входит сюда наравне с кандидатом: те же гиперпараметры при другом
/// split — другой результат, и выдавать один за другой нельзя.
#[derive(Clone, Debug, PartialEq)]
pub struct RunStamp {
    /// Что за данные. Именно по нему считается «те же данные»: ревизия — лишь
    /// номер внутри сессии, а отпечаток переживает и повторное открытие файла,
    /// и перезапуск приложения.
    pub dataset: DatasetFingerprint,
    /// Номер набора в текущей сессии. Нужен интерфейсу, чтобы отличать
    /// устаревшие ответы worker-а, но идентичностью данных не является.
    pub dataset_revision: u64,
    pub split: SplitPlan,
    pub candidate: CandidateSpec,
    /// Seed инициализации финального переобучения. Отличается от seed
    /// проверки: выбирать его по результату значит подбирать по тем же данным,
    /// поэтому он фиксирован заранее — и в отчёте это две разные величины.
    pub final_init_seed: u64,
}

impl RunStamp {
    /// Чем является development-метрика этого запуска. Источник выводится из
    /// разбиения, а не хранится вторым полем, которое могло бы с ним разойтись.
    pub fn eval_source(&self) -> EvalSource {
        match self.split {
            SplitPlan::Holdout { .. } => EvalSource::Validation,
            SplitPlan::KFold { k, .. } => EvalSource::Cv { k },
        }
    }
}

/// Результат фазы разработки: то, по чему принимают решения.
///
/// У K-fold среднего мало: одинаковое среднее при разном разбросе между folds
/// означает разную надёжность вывода, поэтому разброс едет рядом.
///
/// Атрибута `non_exhaustive` здесь нет намеренно: структуру создаёт тот, кто
/// записывает проверку, а не только тот, кто её читает.
#[derive(Clone, Debug)]
pub struct CheckEval {
    pub metrics: Metrics,
    pub per_output: Vec<Metrics>,
    /// Разброс R² между folds; 0 у holdout, где fold один.
    pub r2_std_folds: f32,
}

/// Проверенный кандидат: отпечаток, оценка и отчёты конвейера.
#[derive(Clone, Debug)]
pub struct CheckedRun {
    pub stamp: RunStamp,
    pub eval: CheckEval,
    /// Отчёт конвейера по каждому fold, по порядку; пусто, если конвейера не
    /// просили. Отчёт одного fold не описывает CV-проверку, поэтому их
    /// столько же, сколько folds, а не один.
    pub interpret: Vec<InterpretReport>,
}

impl CheckedRun {
    /// `Validation` у holdout, `Cv { k }` у K-fold. `Test` сюда не попадает.
    pub fn source(&self) -> EvalSource {
        self.stamp.eval_source()
    }
}

/// Состоявшееся раскрытие test: на какой ревизии данных, каким кандидатом и с
/// каким результатом.
#[derive(Clone, Debug)]
pub struct TestDisclosure {
    pub stamp: RunStamp,
    pub eval: FinalEval,
}

impl TestDisclosure {
    /// На каких данных был потрачен test.
    pub fn dataset(&self) -> DatasetFingerprint {
        self.stamp.dataset
    }
}

/// Почему финальное обучение сейчас запрещено.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
    /// По одному раскрытию на набор данных.
    ///
    /// Не «последнее»: с одним слотом цепочка A → B → A снимала бы запрет с A,
    /// хотя его test давно потрачен.
    disclosed: Vec<TestDisclosure>,
}

impl Lifecycle {
    /// Последняя проверка, какому бы кандидату она ни принадлежала.
    ///
    /// Нужна интерфейсу: результат показывается там, где его получили, даже
    /// если форму успели изменить, — с подписью, к какому кандидату он
    /// относится.
    pub fn checked(&self) -> Option<&CheckedRun> {
        self.checked.as_ref()
    }

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

    /// Последнее по времени раскрытие — какому бы набору оно ни принадлежало.
    pub fn disclosure(&self) -> Option<&TestDisclosure> {
        self.disclosed.last()
    }

    /// Раскрытие для конкретного набора данных.
    pub fn disclosure_on(&self, dataset: DatasetFingerprint) -> Option<&TestDisclosure> {
        self.disclosed.iter().find(|d| d.dataset() == dataset)
    }

    /// Результат финального обучения ровно этого кандидата.
    pub fn disclosure_for(&self, stamp: &RunStamp) -> Option<&TestDisclosure> {
        self.disclosed.iter().find(|d| d.stamp == *stamp)
    }

    /// Можно ли открывать test под этот отпечаток.
    pub fn can_finalize(&self, stamp: &RunStamp) -> Result<(), FinalizeRefusal> {
        // Сравниваются сами данные, а не номер набора в сессии: повторно
        // открыв тот же файл, потраченный test не вернуть.
        if let Some(disclosed) = self.disclosure_on(stamp.dataset) {
            return Err(if disclosed.stamp == *stamp {
                FinalizeRefusal::AlreadyFinalized
            } else {
                FinalizeRefusal::TestDisclosed
            });
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
        // На набор приходится одно раскрытие: повтор того же кандидата
        // обновляет запись, а не заводит вторую.
        self.disclosed
            .retain(|d| d.dataset() != disclosure.dataset());
        self.disclosed.push(disclosure);
    }

    /// Данные сменились: проверка относится к прежнему набору. Раскрытия
    /// остаются все — вернувшись к прежним данным, пользователь обязан снова
    /// упереться в потраченный там test.
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
            interpret: None,
        }
    }

    /// Отпечаток тестовых данных: `seed` меняет числа, а значит и данные.
    fn fingerprint(seed: f32) -> DatasetFingerprint {
        let data = crate::data::NumericDataset::new(
            ndarray::Array2::from_shape_vec((2, 2), vec![seed, 2.0, 3.0, 4.0]).unwrap(),
            ndarray::Array2::from_shape_vec((2, 1), vec![5.0, 6.0]).unwrap(),
        );
        let schema = crate::schema::ModelSchema::synthetic(2, 1).unwrap();
        DatasetFingerprint::of(&data, &schema).unwrap()
    }

    fn stamp(revision: u64, candidate: CandidateSpec) -> RunStamp {
        stamp_on(fingerprint(1.0), revision, candidate)
    }

    fn stamp_on(dataset: DatasetFingerprint, revision: u64, candidate: CandidateSpec) -> RunStamp {
        RunStamp {
            dataset,
            dataset_revision: revision,
            split: SplitPlan::default(),
            candidate,
            final_init_seed: 0,
        }
    }

    fn checked(stamp: RunStamp) -> CheckedRun {
        CheckedRun {
            stamp,
            eval: CheckEval {
                metrics: metrics(),
                per_output: vec![metrics()],
                r2_std_folds: 0.0,
            },
            interpret: Vec::new(),
        }
    }

    fn disclosure(stamp: RunStamp) -> TestDisclosure {
        TestDisclosure {
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
        assert_eq!(
            life.checked_for(&current).unwrap().source(),
            EvalSource::Validation
        );
    }

    #[test]
    fn check_source_is_derived_from_the_split() {
        let mut cv = stamp(1, candidate(16));
        cv.split = SplitPlan::KFold {
            k: 4,
            folds_seed: 1,
            test_frac: 0.15,
            test_seed: 1,
        };
        let run = checked(cv);
        assert_eq!(run.source(), EvalSource::Cv { k: 4 });
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
    fn test_stays_disclosed_for_the_whole_dataset() {
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

    /// Повторное открытие ТЕХ ЖЕ данных даёт новую ревизию, но не возвращает
    /// потраченный test: идентичность данных — это их отпечаток.
    #[test]
    fn reopening_the_same_data_does_not_return_the_spent_test() {
        let first = stamp(1, candidate(16));
        let mut life = Lifecycle::default();
        life.record_check(checked(first.clone()));
        life.record_disclosure(disclosure(first.clone()));

        life.on_dataset_changed();
        assert!(
            life.checked_for(&first).is_none(),
            "проверка не переносится"
        );

        // Тот же файл открыт заново: ревизия другая, данные те же.
        let reopened = stamp(2, candidate(32));
        life.record_check(checked(reopened.clone()));
        assert_eq!(
            life.can_finalize(&reopened),
            Err(FinalizeRefusal::TestDisclosed)
        );
    }

    /// Возврат к прежним данным возвращает и запрет: их test уже потрачен.
    /// С одним слотом раскрытия цепочка A → B → A разрешала бы второй замер на
    /// A, хотя первый никуда не делся.
    #[test]
    fn returning_to_earlier_data_finds_its_test_already_spent() {
        let a = fingerprint(1.0);
        let b = fingerprint(2.0);
        let mut life = Lifecycle::default();

        let on_a = stamp_on(a, 1, candidate(16));
        life.record_check(checked(on_a.clone()));
        life.record_disclosure(disclosure(on_a.clone()));

        // Переходим к другому набору и тратим test и там.
        life.on_dataset_changed();
        let on_b = stamp_on(b, 2, candidate(16));
        life.record_check(checked(on_b.clone()));
        assert!(life.can_finalize(&on_b).is_ok(), "у B свой бюджет");
        life.record_disclosure(disclosure(on_b.clone()));

        // Возвращаемся к A: даже другой кандидат упирается в потраченный test.
        life.on_dataset_changed();
        let back_to_a = stamp_on(a, 3, candidate(32));
        life.record_check(checked(back_to_a.clone()));
        assert_eq!(
            life.can_finalize(&back_to_a),
            Err(FinalizeRefusal::TestDisclosed)
        );
        // И оба раскрытия остались доступны каждый на своём наборе.
        assert!(life.disclosure_on(a).is_some());
        assert!(life.disclosure_on(b).is_some());
        assert_eq!(life.disclosure_for(&on_a).map(|d| d.dataset()), Some(a));
    }

    /// Другие данные начинают жизненный цикл заново.
    #[test]
    fn other_data_starts_a_new_lifecycle() {
        let old = stamp(1, candidate(16));
        let mut life = Lifecycle::default();
        life.record_check(checked(old.clone()));
        life.record_disclosure(disclosure(old));
        life.on_dataset_changed();

        let fresh = stamp_on(fingerprint(9.0), 2, candidate(16));
        assert_eq!(life.can_finalize(&fresh), Err(FinalizeRefusal::NotChecked));
        life.record_check(checked(fresh.clone()));
        assert!(life.can_finalize(&fresh).is_ok());
        // История прежнего набора никуда не делась.
        assert!(life.disclosure().is_some());
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
