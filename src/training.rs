//! Единый сценарий обучения: активный набор данных, одна конфигурация запуска,
//! одна функция, которая доводит дело от разбиения до финального замера.
//!
//! До этого модуля каждая поверхность собирала обучение сама: CLI, GUI, sweep и
//! кривая по эпохам повторяли одну и ту же последовательность «разбить —
//! нормализовать — обучить — измерить», расходясь в мелочах. Здесь она одна.
//!
//! Протокол оценки (см. [`crate::split`]) соблюдается по построению: фаза
//! разработки видит только train и validation, а test открывается ровно один
//! раз в финальной фазе — и только если её попросили.

use crate::data::{Normalizer, NumericDataset};
use crate::init::set_init_seed;
use crate::metrics::{evaluate, evaluate_per_output, EvalSource, Metrics};
use crate::numeric_model::{validate_numeric, NumericConfig, NumericModel};
use crate::schema::ModelSchema;
use crate::split::{FinalEval, SearchPool, SplitPlan};
use crate::train::{
    fit_normalizers, predict_dataset, train_surrogate_cb, validate_train, LrSchedule, TrainConfig,
};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

/// Активный набор данных: значения и схема, которая объясняет, что они значат.
///
/// Пара, а не два аргумента: расходившиеся данные и схема — источник ошибок,
/// которые проявляются далеко от места, где их допустили.
pub struct Dataset {
    data: NumericDataset,
    schema: ModelSchema,
}

impl Dataset {
    pub fn new(data: NumericDataset, schema: ModelSchema) -> Result<Self, String> {
        schema.check_dims(data.inputs.ncols(), data.outputs.ncols())?;
        Ok(Self { data, schema })
    }

    pub fn data(&self) -> &NumericDataset {
        &self.data
    }
    pub fn schema(&self) -> &ModelSchema {
        &self.schema
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Когда снимать метрики на validation во время обучения.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalSchedule {
    /// Только итог: как обучение работало до появления кривых.
    Never,
    /// Каждые N эпох (N >= 1).
    Every(usize),
    /// В заданных точках — кривая обучения по контрольным эпохам.
    At(Vec<usize>),
}

impl EvalSchedule {
    fn wants(&self, epoch: usize) -> bool {
        match self {
            EvalSchedule::Never => false,
            EvalSchedule::Every(n) => *n > 0 && epoch.is_multiple_of(*n),
            EvalSchedule::At(points) => points.contains(&epoch),
        }
    }

    fn validate(&self, epochs: usize) -> Result<(), String> {
        match self {
            EvalSchedule::Never => Ok(()),
            EvalSchedule::Every(0) => Err("eval_every должен быть >= 1".to_string()),
            EvalSchedule::Every(n) if *n > epochs => Err(format!(
                "eval_every={n} больше числа эпох {epochs}: validation не будет измерен"
            )),
            EvalSchedule::Every(_) => Ok(()),
            EvalSchedule::At(points) if points.is_empty() => {
                Err("eval points: список не должен быть пустым".to_string())
            }
            EvalSchedule::At(points) => {
                if points.contains(&0) {
                    return Err("eval points должны быть >= 1".to_string());
                }
                if let Some(point) = points.iter().find(|&&point| point > epochs) {
                    return Err(format!("eval point {point} больше числа эпох {epochs}"));
                }
                if points.iter().copied().collect::<BTreeSet<_>>().len() != points.len() {
                    return Err("eval points содержат дубликаты".to_string());
                }
                Ok(())
            }
        }
    }

    fn measures_validation(&self) -> bool {
        !matches!(self, EvalSchedule::Never)
    }
}

/// Ранняя остановка по validation. Порог `min_delta` отсекает шум: без него
/// любое случайное колебание считалось бы улучшением. `patience` измеряется в
/// последовательных validation-замерах, а не обязательно в эпохах.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f32,
}

/// Точка кривой обучения. `val` есть только там, где расписание попросило
/// замер.
#[derive(Clone, Debug)]
pub struct EpochPoint {
    pub epoch: usize,
    pub train_loss: f32,
    pub val: Option<Metrics>,
}

/// История одного обучения.
#[derive(Clone, Debug)]
pub struct TrainingHistory {
    pub points: Vec<EpochPoint>,
    /// Чем являются `val`-метрики: validation у holdout, CV у K-fold.
    pub source: EvalSource,
    /// Эпоха с лучшим R² на validation (если замеры вообще были).
    pub best_epoch: Option<usize>,
    /// Обучение прервано ранней остановкой, а не дошло до последней эпохи.
    pub stopped_early: bool,
}

impl TrainingHistory {
    pub fn last_val(&self) -> Option<&Metrics> {
        self.points.iter().rev().find_map(|p| p.val.as_ref())
    }

    pub fn best_val_r2(&self) -> Option<f32> {
        self.points
            .iter()
            .filter_map(|p| p.val.as_ref())
            .map(|m| m.r2)
            .fold(None, |best: Option<f32>, r2| {
                Some(best.map_or(r2, |b| b.max(r2)))
            })
    }
}

/// Конфигурация запуска: что за модель, как её учить и когда мерить.
#[derive(Clone)]
pub struct TrainingSetup {
    pub config: NumericConfig,
    pub train: TrainConfig,
    pub eval: EvalSchedule,
    pub early_stopping: Option<EarlyStopping>,
}

impl TrainingSetup {
    pub fn new(config: NumericConfig, train: TrainConfig) -> Self {
        Self {
            config,
            train,
            eval: EvalSchedule::Never,
            early_stopping: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_numeric(&self.config)?;
        validate_train(self.train.lr, self.train.batch_size)?;
        if self.train.epochs == 0 {
            return Err("epochs должен быть >= 1".to_string());
        }
        if let LrSchedule::WarmupCosine {
            warmup_frac,
            min_lr_ratio,
        } = self.train.schedule
        {
            if !(0.0..1.0).contains(&warmup_frac) {
                return Err("warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&min_lr_ratio) {
                return Err("min-lr-ratio должен быть в [0, 1]".to_string());
            }
        }
        self.eval.validate(self.train.epochs)?;
        if let Some(early) = self.early_stopping {
            if early.patience == 0 {
                return Err("early-stopping patience должен быть >= 1".to_string());
            }
            if !early.min_delta.is_finite() || early.min_delta < 0.0 {
                return Err("early-stopping min_delta должен быть конечным и >= 0".to_string());
            }
            if !self.eval.measures_validation() {
                return Err(
                    "early stopping требует validation-расписание, отличное от Never".to_string(),
                );
            }
        }
        Ok(())
    }
}

/// Хук после обучения фазы: меняет модель (у KAN — прунинг и сжатие) и,
/// если есть на чём, отчитывается о влиянии.
///
/// Аргументы: фаза, обученная модель, данные обучения этой фазы и набор для
/// отчёта (validation в разработке, `None` в финале).
pub type PostTrain<'a> =
    &'a mut dyn FnMut(Phase, &mut TrainedModel, &NumericDataset, Option<&NumericDataset>);

/// Настройка только что построенной модели до первой эпохи. Отделена от
/// [`PostTrain`]: регуляризатор должен участвовать в обучении, а прунинг — нет.
pub type ConfigureModel<'a> = &'a mut dyn FnMut(Phase, &NumericModel);

/// Фаза обучения — нужна хукам, которые ведут себя в них по-разному.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Обучение на train: по его validation принимаются все решения.
    Development,
    /// Переобучение на train + validation перед единственным замером test.
    Final,
}

/// Обученный кандидат вместе со своими нормализаторами: врозь они бессмысленны.
pub struct TrainedModel {
    pub model: NumericModel,
    pub in_norm: Normalizer,
    pub out_norm: Normalizer,
    pub history: TrainingHistory,
}

/// Результат полного сценария.
pub struct TrainingOutcome {
    pub development: TrainedModel,
    /// Финальная модель: `None`, если фазу не запрашивали либо запуск отменён.
    pub final_model: Option<TrainedModel>,
    /// Единственный замер на test: `None`, если фазу не запрашивали либо запуск
    /// отменён до оценки.
    pub final_eval: Option<FinalEval>,
}

/// Обучить одного кандидата на fold `fold` и снять историю.
///
/// Единственное место, где создаётся и учится модель: нормализаторы строятся по
/// train ЭТОГО fold, метрики снимаются на его validation.
#[allow(clippy::too_many_arguments)]
pub fn train_candidate(
    dataset: &Dataset,
    pool: &SearchPool,
    fold: usize,
    setup: &TrainingSetup,
    init_seed: u64,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&EpochPoint),
    configure_model: &mut dyn FnMut(&NumericModel),
) -> Result<TrainedModel, String> {
    setup.validate()?;
    let (train, val) = pool.fold(fold)?;
    let specs = dataset.schema().feature_specs();
    let (in_norm, out_norm) = fit_normalizers(&train, &specs);

    set_init_seed(init_seed);
    let model = setup.config.build(&specs, dataset.schema().n_outputs());
    configure_model(&model);

    let mut points: Vec<EpochPoint> = Vec::new();
    let mut best_observed: Option<(usize, f32)> = None;
    let mut stopping_best: Option<f32> = None;
    let mut since_improvement = 0usize;
    let mut stopped_early = false;

    train_surrogate_cb(
        &model,
        &train,
        &in_norm,
        &out_norm,
        &setup.train,
        &mut |epoch, loss| {
            let epoch = epoch + 1; // 1-based: «после первой эпохи»
            let val_metrics = setup.eval.wants(epoch).then(|| {
                let pred = predict_dataset(&model, &val, &in_norm, &out_norm);
                evaluate(&pred, &val.outputs)
            });
            if let Some(m) = &val_metrics {
                if best_observed.is_none_or(|(_, r2)| m.r2 > r2) {
                    best_observed = Some((epoch, m.r2));
                }
                if let Some(es) = setup.early_stopping {
                    match stopping_best {
                        Some(best_r2) if m.r2 <= best_r2 + es.min_delta => {
                            since_improvement += 1;
                            stopped_early = since_improvement >= es.patience;
                        }
                        _ => {
                            stopping_best = Some(m.r2);
                            since_improvement = 0;
                        }
                    }
                }
            }
            let point = EpochPoint {
                epoch,
                train_loss: loss,
                val: val_metrics,
            };
            on_point(&point);
            points.push(point);
            !stopped_early
        },
        cancel,
    );
    Ok(TrainedModel {
        model,
        in_norm,
        out_norm,
        history: TrainingHistory {
            source: pool.source(),
            best_epoch: best_observed.map(|(e, _)| e),
            stopped_early,
            points,
        },
    })
}

/// Полный сценарий: разбиение, фаза разработки и — по запросу — финальное
/// переобучение с единственным замером test.
///
/// `post_train` вызывается после каждого полностью завершённого обучения фазы:
/// там живут операции, меняющие саму модель (у KAN — прунинг и структурное
/// сжатие). Так один и тот же конвейер применяется и к модели разработки, и к
/// финальной. У отменённой фазы хук не запускается.
///
/// Последний аргумент хука — набор для отчёта о влиянии этих операций: в фазе
/// разработки это validation, в финальной его нет (test тратить нельзя, а
/// train+validation уже внутри обучения).
///
/// Пока полный сценарий поддерживает holdout. Для K-fold нужно сначала
/// агрегировать кандидата по всем folds через поиск; выдавать модель fold 0 за
/// CV-оценку здесь запрещено явно.
#[allow(clippy::too_many_arguments)]
pub fn run_training(
    dataset: &Dataset,
    split: SplitPlan,
    setup: &TrainingSetup,
    final_phase: bool,
    final_init_seed: u64,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(Phase, &EpochPoint),
    configure_model: ConfigureModel<'_>,
    post_train: PostTrain<'_>,
) -> Result<TrainingOutcome, String> {
    setup.validate()?;
    let prepared = split.prepare(dataset.data())?;
    if prepared.search.n_folds() != 1 {
        return Err(
            "run_training пока принимает только holdout; K-fold выполняется через search"
                .to_string(),
        );
    }
    let (train, _) = prepared.search.fold(0)?;

    let (_, val) = prepared.search.fold(0)?;
    let mut configure_development = |model: &NumericModel| {
        configure_model(Phase::Development, model);
    };
    let mut development = train_candidate(
        dataset,
        &prepared.search,
        0,
        setup,
        setup.train.seed,
        cancel,
        &mut |p| on_point(Phase::Development, p),
        &mut configure_development,
    )?;
    if cancel.load(Ordering::Relaxed) {
        return Ok(TrainingOutcome {
            development,
            final_model: None,
            final_eval: None,
        });
    }
    post_train(Phase::Development, &mut development, &train, Some(&val));

    if !final_phase || cancel.load(Ordering::Relaxed) {
        return Ok(TrainingOutcome {
            development,
            final_model: None,
            final_eval: None,
        });
    }

    // Финальная модель учится на train + validation с заранее заданным seed:
    // выбирать seed по результату — та же форма подбора. Если development
    // выбирал число эпох early stopping-ом, refit обязан использовать именно
    // эту выбранную по validation точку, а не исходный максимум.
    let final_epochs = if setup.early_stopping.is_some() {
        development
            .history
            .best_epoch
            .ok_or_else(|| "early stopping не получил ни одного validation-замера".to_string())?
    } else {
        setup.train.epochs
    };
    let pool = prepared.search.all();
    let specs = dataset.schema().feature_specs();
    let (in_norm, out_norm) = fit_normalizers(&pool, &specs);
    set_init_seed(final_init_seed);
    let model = setup.config.build(&specs, dataset.schema().n_outputs());
    configure_model(Phase::Final, &model);
    let mut points = Vec::new();
    train_surrogate_cb(
        &model,
        &pool,
        &in_norm,
        &out_norm,
        &TrainConfig {
            epochs: final_epochs,
            seed: final_init_seed,
            ..setup.train.clone()
        },
        &mut |epoch, loss| {
            let point = EpochPoint {
                epoch: epoch + 1,
                train_loss: loss,
                val: None,
            };
            on_point(Phase::Final, &point);
            points.push(point);
            true
        },
        cancel,
    );
    if cancel.load(Ordering::Relaxed) {
        return Ok(TrainingOutcome {
            development,
            final_model: None,
            final_eval: None,
        });
    }
    let mut final_model = TrainedModel {
        model,
        in_norm,
        out_norm,
        history: TrainingHistory {
            points,
            source: prepared.search.source(),
            best_epoch: None,
            stopped_early: false,
        },
    };
    post_train(Phase::Final, &mut final_model, &pool, None);
    if cancel.load(Ordering::Relaxed) {
        return Ok(TrainingOutcome {
            development,
            final_model: None,
            final_eval: None,
        });
    }

    let n_outputs = dataset.schema().n_outputs();
    let final_eval = prepared.test.evaluate(
        |inputs| {
            let ds = NumericDataset::new(
                inputs.clone(),
                ndarray::Array2::zeros((inputs.nrows(), n_outputs)),
            );
            predict_dataset(
                &final_model.model,
                &ds,
                &final_model.in_norm,
                &final_model.out_norm,
            )
        },
        final_init_seed,
    )?;

    Ok(TrainingOutcome {
        development,
        final_model: Some(final_model),
        final_eval: Some(final_eval),
    })
}

/// Метрики модели на произвольном наборе — для отчётов вызывающего.
pub fn evaluate_on(model: &TrainedModel, data: &NumericDataset) -> (Metrics, Vec<Metrics>) {
    let pred = predict_dataset(&model.model, data, &model.in_norm, &model.out_norm);
    (
        evaluate(&pred, &data.outputs),
        evaluate_per_output(&pred, &data.outputs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::config::ModelConfig;
    use crate::encoders::ValueEncoderConfig;
    use crate::numeric_model::ModelKind;
    use crate::split::DEFAULT_FINAL_INIT_SEED;
    use crate::train::LrSchedule;

    fn dataset(n: usize) -> Dataset {
        let data = blackbox::sum().generate(n, 0);
        let schema = ModelSchema::synthetic(data.inputs.ncols(), data.outputs.ncols()).unwrap();
        Dataset::new(data, schema).unwrap()
    }

    fn setup(epochs: usize) -> TrainingSetup {
        TrainingSetup::new(
            NumericConfig {
                kind: ModelKind::Mlp,
                transformer: ModelConfig::default(),
                value: ValueEncoderConfig::default(),
                mlp_width: 16,
                mlp_layers: 2,
                kan: Default::default(),
            },
            TrainConfig {
                epochs,
                batch_size: 32,
                lr: 3e-3,
                seed: 0,
                schedule: LrSchedule::Constant,
            },
        )
    }

    #[test]
    fn dataset_rejects_schema_of_wrong_shape() {
        let data = blackbox::sum().generate(8, 0);
        assert!(Dataset::new(data, ModelSchema::synthetic(5, 1).unwrap()).is_err());
    }

    #[test]
    fn history_records_train_loss_every_epoch() {
        let ds = dataset(96);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        let trained = train_candidate(
            &ds,
            &prepared.search,
            0,
            &setup(4),
            0,
            &never,
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(trained.history.points.len(), 4);
        assert_eq!(trained.history.points[0].epoch, 1);
        // Без расписания validation не считается вовсе.
        assert!(trained.history.points.iter().all(|p| p.val.is_none()));
        assert_eq!(trained.history.source, EvalSource::Validation);
    }

    #[test]
    fn eval_schedule_controls_validation_points() {
        let ds = dataset(96);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();

        let mut every = setup(6);
        every.eval = EvalSchedule::Every(2);
        let trained = train_candidate(
            &ds,
            &prepared.search,
            0,
            &every,
            0,
            &never,
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        let measured: Vec<usize> = trained
            .history
            .points
            .iter()
            .filter(|p| p.val.is_some())
            .map(|p| p.epoch)
            .collect();
        assert_eq!(measured, vec![2, 4, 6]);
        assert!(trained.history.best_epoch.is_some());

        let mut at = setup(6);
        at.eval = EvalSchedule::At(vec![1, 5]);
        let trained = train_candidate(
            &ds,
            &prepared.search,
            0,
            &at,
            0,
            &never,
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        let measured: Vec<usize> = trained
            .history
            .points
            .iter()
            .filter(|p| p.val.is_some())
            .map(|p| p.epoch)
            .collect();
        assert_eq!(measured, vec![1, 5]);
    }

    #[test]
    fn callback_sees_the_same_points_as_history() {
        let ds = dataset(96);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        let mut seen = Vec::new();
        let trained = train_candidate(
            &ds,
            &prepared.search,
            0,
            &setup(3),
            0,
            &never,
            &mut |p| seen.push(p.epoch),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            seen,
            trained
                .history
                .points
                .iter()
                .map(|p| p.epoch)
                .collect::<Vec<_>>()
        );
    }

    /// Фаза разработки не открывает test; финальная — открывает ровно один раз.
    #[test]
    fn final_phase_is_optional_and_opens_test_once() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);

        let without = run_training(
            &ds,
            SplitPlan::default(),
            &setup(2),
            false,
            DEFAULT_FINAL_INIT_SEED,
            &never,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |_, _, _, _| {},
        )
        .unwrap();
        assert!(without.final_model.is_none());
        assert!(without.final_eval.is_none());

        let with = run_training(
            &ds,
            SplitPlan::default(),
            &setup(2),
            true,
            DEFAULT_FINAL_INIT_SEED,
            &never,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |_, _, _, _| {},
        )
        .unwrap();
        let eval = with.final_eval.expect("финальный замер");
        assert_eq!(eval.origin.final_init_seed, DEFAULT_FINAL_INIT_SEED);
        assert!(eval.metrics.r2.is_finite());
        assert!(with.final_model.is_some());
    }

    /// Хук вызывается в обеих фазах — иначе сохранённая модель отличалась бы от
    /// той, по которой принимали решения.
    #[test]
    fn post_train_hook_runs_in_both_phases() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let mut phases = Vec::new();
        run_training(
            &ds,
            SplitPlan::default(),
            &setup(1),
            true,
            DEFAULT_FINAL_INIT_SEED,
            &never,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |phase, _, data, eval| phases.push((phase, data.len(), eval.is_some())),
        )
        .unwrap();
        assert_eq!(phases.len(), 2);
        assert_eq!(phases[0].0, Phase::Development);
        assert_eq!(phases[1].0, Phase::Final);
        // Финальная фаза учится на train + validation, поэтому строк больше.
        assert!(phases[1].1 > phases[0].1);
        // Отчитываться о влиянии конвейера можно только в фазе разработки.
        assert!(phases[0].2, "в разработке есть validation для отчёта");
        assert!(!phases[1].2, "в финале отчитываться не на чем");
    }

    #[test]
    fn early_stopping_halts_on_plateau() {
        let ds = dataset(96);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        let mut s = setup(30);
        s.eval = EvalSchedule::Every(1);
        // Любое отсутствие улучшения останавливает сразу.
        s.early_stopping = Some(EarlyStopping {
            patience: 1,
            min_delta: 1.0,
        });
        let trained = train_candidate(
            &ds,
            &prepared.search,
            0,
            &s,
            0,
            &never,
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert!(trained.history.stopped_early);
        assert!(
            trained.history.points.len() < 30,
            "обучение должно прерваться: {} эпох",
            trained.history.points.len()
        );
    }

    #[test]
    fn cancel_is_not_reported_as_early_stopping() {
        let ds = dataset(96);
        let cancel = AtomicBool::new(true);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        let trained = train_candidate(
            &ds,
            &prepared.search,
            0,
            &setup(5),
            0,
            &cancel,
            &mut |_| {},
            &mut |_| {},
        )
        .unwrap();
        assert!(!trained.history.stopped_early);
        assert!(trained.history.points.is_empty());
    }

    #[test]
    fn setup_rejects_invalid_eval_and_early_stopping() {
        let mut invalid_schedule = setup(3);
        invalid_schedule.eval = EvalSchedule::Every(0);
        assert!(invalid_schedule
            .validate()
            .unwrap_err()
            .contains("eval_every"));

        let mut without_validation = setup(3);
        without_validation.early_stopping = Some(EarlyStopping {
            patience: 1,
            min_delta: 0.0,
        });
        assert!(without_validation
            .validate()
            .unwrap_err()
            .contains("validation"));

        let mut invalid_delta = setup(3);
        invalid_delta.eval = EvalSchedule::Every(1);
        invalid_delta.early_stopping = Some(EarlyStopping {
            patience: 1,
            min_delta: f32::NAN,
        });
        assert!(invalid_delta.validate().unwrap_err().contains("min_delta"));
    }

    #[test]
    fn model_is_configured_before_first_epoch() {
        let ds = dataset(96);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        let configured = std::cell::Cell::new(false);
        train_candidate(
            &ds,
            &prepared.search,
            0,
            &setup(1),
            0,
            &never,
            &mut |_| assert!(configured.get()),
            &mut |_| configured.set(true),
        )
        .unwrap();
        assert!(configured.get());
    }

    #[test]
    fn cancellation_during_final_phase_does_not_open_test() {
        let ds = dataset(120);
        let cancel = AtomicBool::new(false);
        let mut post_phases = Vec::new();
        let outcome = run_training(
            &ds,
            SplitPlan::default(),
            &setup(3),
            true,
            DEFAULT_FINAL_INIT_SEED,
            &cancel,
            &mut |phase, point| {
                if phase == Phase::Final && point.epoch == 1 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
            &mut |_, _| {},
            &mut |phase, _, _, _| post_phases.push(phase),
        )
        .unwrap();

        assert_eq!(post_phases, vec![Phase::Development]);
        assert!(outcome.final_model.is_none());
        assert!(outcome.final_eval.is_none());
    }

    #[test]
    fn full_training_rejects_unaggregated_kfold() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let error = run_training(
            &ds,
            SplitPlan::kfold_default(),
            &setup(1),
            false,
            DEFAULT_FINAL_INIT_SEED,
            &never,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |_, _, _, _| {},
        )
        .err()
        .expect("K-fold нельзя выдавать за один development fold");
        assert!(error.contains("K-fold"), "{error}");
    }

    #[test]
    fn final_refit_uses_epoch_selected_by_early_stopping() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let mut s = setup(12);
        s.eval = EvalSchedule::Every(1);
        s.early_stopping = Some(EarlyStopping {
            patience: 1,
            min_delta: 1.0,
        });

        let outcome = run_training(
            &ds,
            SplitPlan::default(),
            &s,
            true,
            DEFAULT_FINAL_INIT_SEED,
            &never,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |_, _, _, _| {},
        )
        .unwrap();
        let selected = outcome.development.history.best_epoch.unwrap();
        assert!(outcome.development.history.stopped_early);
        assert_eq!(outcome.final_model.unwrap().history.points.len(), selected);
    }
}
