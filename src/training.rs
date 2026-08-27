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
use crate::metrics::{
    aggregate_runs, evaluate, evaluate_per_output, ConfigEval, EvalSource, Metrics, RunEval,
};
use crate::numeric_model::{validate_numeric, NumericConfig, NumericModel};
use crate::schema::ModelSchema;
use crate::split::{FinalEval, PreparedSplit, SearchPool, SplitPlan};
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

/// Рекомендованная точка остановки по последовательности замеров
/// `(эпоха, R²(validation))`: сначала достижение цели, затем плато, иначе
/// последний доступный замер.
///
/// Функция принимает только те данные, от которых действительно зависит
/// решение. Поэтому GUI не приходится конструировать фиктивные строки отчёта
/// с нулевыми RMSE/MAE, а CLI-адаптер может передать свою таблицу как пары.
pub fn recommended_epoch(
    points: impl IntoIterator<Item = (usize, f32)>,
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
) -> Option<(usize, String)> {
    let points: Vec<(usize, f32)> = points.into_iter().collect();
    if points.is_empty() {
        return None;
    }
    for &(epoch, r2) in &points {
        if r2 >= target_r2 {
            return Some((epoch, format!("target R²≥{target_r2}")));
        }
    }
    for pair in points.windows(2) {
        let (prev_epoch, prev_r2) = pair[0];
        let (_, current_r2) = pair[1];
        if prev_r2 >= plateau_min && current_r2 - prev_r2 < min_gain {
            return Some((prev_epoch, format!("плато ΔR²<{min_gain}")));
        }
    }
    Some((
        points.last().expect("проверено выше").0,
        "лучшее из имеющегося".to_string(),
    ))
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
#[non_exhaustive]
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

/// Тот же хук для проверки кандидата: фаза здесь всегда «разработка», зато
/// важен номер fold — конвейер применяется к каждому из них.
pub type PostCheck<'a> =
    &'a mut dyn FnMut(usize, &mut TrainedModel, &NumericDataset, Option<&NumericDataset>);

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
#[non_exhaustive]
pub struct TrainingOutcome {
    pub development: TrainedModel,
    /// Финальная модель: `None`, если фазу не запрашивали либо запуск отменён.
    pub final_model: Option<TrainedModel>,
    /// Единственный замер на test: `None`, если фазу не запрашивали либо запуск
    /// отменён до оценки.
    pub final_eval: Option<FinalEval>,
}

/// Результат финального refit после уже выполненного выбора конфигурации.
/// `None` означает отмену до открытия test.
#[non_exhaustive]
pub struct RefitOutcome {
    pub model: Option<TrainedModel>,
    pub eval: Option<FinalEval>,
}

// --- поиск конфигурации ---

/// Что оптимизирует поиск.
///
/// По умолчанию worst-output R²: aggregate умеет скрыть полностью проваленный
/// выход, а у задачи с несколькими выходами это ровно то, что важно заметить.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchObjective {
    #[default]
    WorstOutputR2,
    AggregateR2,
    MeanOutputR2,
    Nrmse,
}

impl SearchObjective {
    pub fn label(self) -> &'static str {
        match self {
            SearchObjective::WorstOutputR2 => "worst-output R²",
            SearchObjective::AggregateR2 => "aggregate R²",
            SearchObjective::MeanOutputR2 => "mean-output R²",
            SearchObjective::Nrmse => "aggregate nRMSE",
        }
    }

    /// Чем больше, тем лучше — включая nRMSE, который для этого меняет знак.
    ///
    /// Внутренняя: считает по агрегату прогонов, а низкоуровневый поиск наружу
    /// не обещан. Снаружи цель видна как выбор ранжирования, не как формула.
    pub(crate) fn score(self, eval: &ConfigEval, runs: &[RunEval]) -> f32 {
        let per_output_r2 = || eval.per_output_mean.iter().map(|m| m.r2);
        match self {
            SearchObjective::AggregateR2 => eval.mean.r2,
            SearchObjective::WorstOutputR2 => per_output_r2().fold(f32::INFINITY, f32::min),
            SearchObjective::MeanOutputR2 => {
                let values: Vec<f32> = per_output_r2().collect();
                values.iter().sum::<f32>() / values.len().max(1) as f32
            }
            // nRMSE — нелинейное преобразование R², поэтому считается по каждому
            // прогону до усреднения, а не из уже среднего R².
            SearchObjective::Nrmse => {
                -runs
                    .iter()
                    .map(|run| (1.0 - run.metrics.r2).max(0.0).sqrt())
                    .sum::<f32>()
                    / runs.len().max(1) as f32
            }
        }
    }
}

/// План поиска: по каким seed повторять каждую конфигурацию и что оптимизировать.
///
/// Данные и разбиение сюда не входят: они приходят готовыми, а `seeds` меняет
/// только инициализацию (см. [`crate::split`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SearchPlan {
    pub seeds: Vec<u64>,
    pub objective: SearchObjective,
}

impl Default for SearchPlan {
    fn default() -> Self {
        Self {
            seeds: vec![0],
            objective: SearchObjective::default(),
        }
    }
}

impl SearchPlan {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.seeds.is_empty() {
            return Err("seeds: пустой список".to_string());
        }
        let unique: std::collections::BTreeSet<u64> = self.seeds.iter().copied().collect();
        if unique.len() != self.seeds.len() {
            return Err("seeds: повторяющиеся значения".to_string());
        }
        Ok(())
    }
}

/// Стоимость поиска — считается ДО запуска, чтобы её можно было показать.
///
/// Пользователю важнее понимать цену операции, чем название пресета: «600
/// прогонов по 40 эпох» останавливает вовремя, «Тщательно» — нет.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchCost {
    pub configs: usize,
    pub seeds: usize,
    pub folds: usize,
    pub runs: usize,
    /// Верхняя граница эпох на прогон: ранняя остановка может её сократить.
    pub max_epochs: usize,
    /// Верхняя граница суммы эпох с учётом длины каждого кандидата. Это не
    /// обязательно `runs × max_epochs`: кандидаты могут иметь разный бюджет.
    epochs_upper_bound: usize,
}

impl SearchCost {
    pub fn new(configs: usize, seeds: usize, folds: usize, max_epochs: usize) -> Self {
        let runs = configs.saturating_mul(seeds).saturating_mul(folds);
        Self {
            configs,
            seeds,
            folds,
            runs,
            max_epochs,
            epochs_upper_bound: runs.saturating_mul(max_epochs),
        }
    }

    fn with_epochs_upper_bound(
        configs: usize,
        seeds: usize,
        folds: usize,
        max_epochs: usize,
        epochs_upper_bound: usize,
    ) -> Self {
        let mut cost = Self::new(configs, seeds, folds, max_epochs);
        cost.epochs_upper_bound = epochs_upper_bound;
        cost
    }

    /// Оценка сверху по числу обучающих эпох во всём поиске.
    pub fn epochs_upper_bound(&self) -> usize {
        self.epochs_upper_bound
    }

    pub fn describe(&self) -> String {
        let folds = if self.folds > 1 {
            format!(" × {} folds", self.folds)
        } else {
            String::new()
        };
        let configs = russian_count(self.configs, "конфигурация", "конфигурации", "конфигураций");
        let runs = russian_count(self.runs, "прогон", "прогона", "прогонов");
        let per_run_epochs = russian_epochs(self.max_epochs);
        let total_epochs = russian_epochs(self.epochs_upper_bound());
        format!(
            "{configs} × {} seed{folds} = {runs}, до {per_run_epochs} на прогон (не более {total_epochs} всего)",
            self.seeds
        )
    }
}

fn russian_count(n: usize, one: &str, few: &str, many: &str) -> String {
    let form = if (11..=14).contains(&(n % 100)) {
        many
    } else {
        match n % 10 {
            1 => one,
            2..=4 => few,
            _ => many,
        }
    };
    format!("{n} {form}")
}

fn russian_epochs(n: usize) -> String {
    let form = if n % 10 == 1 && n % 100 != 11 {
        "эпохи"
    } else {
        "эпох"
    };
    format!("{n} {form}")
}

/// Кандидат поиска: подпись для отчёта и конфигурация запуска.
pub(crate) struct SearchCandidate {
    pub label: String,
    pub setup: TrainingSetup,
}

/// Результат по одной конфигурации.
pub(crate) struct SearchRow {
    /// Позиция кандидата в исходном списке: ранжирование меняет порядок, а
    /// вызывающему нужно вернуться к своим данным о конфигурации.
    pub candidate: usize,
    pub label: String,
    pub eval: ConfigEval,
    pub runs: Vec<RunEval>,
    pub score: f32,
}

pub(crate) struct SearchResults {
    pub rows: Vec<SearchRow>,
    pub cost: SearchCost,
    pub cancelled: bool,
}

/// Перебрать кандидатов на подготовленном pool и отранжировать их.
///
/// Test сюда не попадает физически: [`SearchPool`] его не содержит. Каждый
/// кандидат обучается на всех seed и всех folds, свёртка — через
/// [`aggregate_runs`] (folds внутри seed, затем seeds).
pub(crate) fn search(
    dataset: &Dataset,
    pool: &SearchPool,
    candidates: &[SearchCandidate],
    plan: &SearchPlan,
    cancel: &AtomicBool,
    on_row: &mut dyn FnMut(&SearchRow),
) -> Result<SearchResults, String> {
    plan.validate()?;
    if candidates.is_empty() {
        return Err("поиск без кандидатов".to_string());
    }
    // Проверяем всю работу до первого дорогостоящего запуска: ошибка во втором
    // кандидате не должна обнаруживаться после обучения первого.
    for (index, candidate) in candidates.iter().enumerate() {
        candidate
            .setup
            .validate()
            .map_err(|error| format!("кандидат {index} ('{}'): {error}", candidate.label))?;
    }
    let cost = search_cost(candidates, plan, pool.n_folds());

    let mut rows: Vec<SearchRow> = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(finish(rows, cost, true));
        }

        let mut runs = Vec::new();
        for &seed in &plan.seeds {
            for fold in 0..pool.n_folds() {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(finish(rows, cost, true));
                }
                let trained = train_candidate(
                    dataset,
                    pool,
                    fold,
                    &candidate.setup,
                    seed,
                    cancel,
                    &mut |_| {},
                    &mut |_| {},
                )?;
                if cancel.load(Ordering::Relaxed) {
                    return Ok(finish(rows, cost, true));
                }
                let (_, val) = pool.fold(fold)?;
                let (metrics, per_output) = evaluate_on(&trained, &val);
                runs.push(RunEval {
                    metrics,
                    per_output,
                    origin: pool.run_origin(fold, seed),
                });
            }
        }

        let eval = aggregate_runs(&runs, &plan.seeds, pool.source())?;
        let row = SearchRow {
            candidate: index,
            label: candidate.label.clone(),
            score: plan.objective.score(&eval, &runs),
            eval,
            runs,
        };
        on_row(&row);
        rows.push(row);
    }

    Ok(finish(rows, cost, false))
}

/// Стоимость поиска по кандидатам и плану.
pub(crate) fn search_cost(
    candidates: &[SearchCandidate],
    plan: &SearchPlan,
    folds: usize,
) -> SearchCost {
    let epochs_per_seed_fold = candidates.iter().fold(0usize, |total, candidate| {
        total.saturating_add(candidate.setup.train.epochs)
    });
    SearchCost::with_epochs_upper_bound(
        candidates.len(),
        plan.seeds.len(),
        folds,
        candidates
            .iter()
            .map(|c| c.setup.train.epochs)
            .max()
            .unwrap_or(0),
        epochs_per_seed_fold
            .saturating_mul(plan.seeds.len())
            .saturating_mul(folds),
    )
}

pub(crate) fn compare_scores_desc(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => b.total_cmp(&a),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => std::cmp::Ordering::Equal,
    }
}

fn finish(mut rows: Vec<SearchRow>, cost: SearchCost, cancelled: bool) -> SearchResults {
    // Разошедшийся кандидат остаётся виден в отчёте, но NaN/inf не может
    // случайно оказаться рекомендацией из-за `partial_cmp(None)`.
    rows.sort_by(|a, b| compare_scores_desc(a.score, b.score));
    SearchResults {
        rows,
        cost,
        cancelled,
    }
}

/// Обучить одного кандидата на fold `fold` и снять историю.
///
/// Единственное место, где создаётся и учится модель: нормализаторы строятся по
/// train ЭТОГО fold, метрики снимаются на его validation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_candidate(
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

/// Результат проверки кандидата: оценка и — у holdout — сама модель.
#[non_exhaustive]
pub struct CheckOutcome {
    /// Средние метрики по всем folds (у holdout — просто validation).
    pub metrics: Metrics,
    pub per_output: Vec<Metrics>,
    /// Разброс R² между folds; 0 у holdout, где fold один.
    pub r2_std_folds: f32,
    /// Чем является оценка: validation у holdout, CV у K-fold.
    pub source: EvalSource,
    /// Модель фазы разработки. У K-fold её нет: моделей столько же, сколько
    /// folds, и выдавать одну из них за общую оценку нельзя.
    pub model: Option<TrainedModel>,
    /// Проверка прервана: оценки нет, решение принимать не по чему.
    pub cancelled: bool,
}

/// Проверить кандидата: обучить его на каждом fold и снять оценку, не трогая
/// test.
///
/// Единственный путь «проверки» для всех режимов интерфейса. У holdout это одно
/// обучение на train с метриками на validation; у K-fold — обучение на каждом
/// fold и свёртка, поэтому «Проверить» при K-fold означает CV-оценку, а не
/// произвольную модель одного fold.
///
/// `post_train` вызывается ДО оценки каждого fold: конвейер интерпретации
/// меняет саму модель, и мерить нужно ту модель, которая получится в итоге.
#[allow(clippy::too_many_arguments)]
pub fn check_candidate(
    dataset: &Dataset,
    split: SplitPlan,
    setup: &TrainingSetup,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(usize, &EpochPoint),
    configure_model: &mut dyn FnMut(&NumericModel),
    post_train: PostCheck<'_>,
) -> Result<CheckOutcome, String> {
    setup.validate()?;
    let prepared = split.prepare(dataset.data())?;
    let pool = prepared.search;
    let folds = pool.n_folds();
    let init_seed = setup.train.seed;

    let mut runs: Vec<RunEval> = Vec::with_capacity(folds);
    let mut model = None;
    for fold in 0..folds {
        if cancel.load(Ordering::Relaxed) {
            return Ok(cancelled_check(pool.source()));
        }
        let mut trained = train_candidate(
            dataset,
            &pool,
            fold,
            setup,
            init_seed,
            cancel,
            &mut |point| on_point(fold, point),
            configure_model,
        )?;
        if cancel.load(Ordering::Relaxed) {
            return Ok(cancelled_check(pool.source()));
        }
        let (train, val) = pool.fold(fold)?;
        post_train(fold, &mut trained, &train, Some(&val));
        let (metrics, per_output) = evaluate_on(&trained, &val);
        runs.push(RunEval {
            metrics,
            per_output,
            origin: pool.run_origin(fold, init_seed),
        });
        // Модель отдаём только когда fold один: тогда она и есть оценка.
        if folds == 1 {
            model = Some(trained);
        }
    }

    let eval = aggregate_runs(&runs, &[init_seed], pool.source())?;
    Ok(CheckOutcome {
        metrics: eval.mean,
        per_output: eval.per_output_mean,
        r2_std_folds: eval.r2_std_folds,
        source: eval.origin.source,
        model,
        cancelled: false,
    })
}

/// Отменённая проверка: метрик нет, и подставлять нули нельзя — по ним начали
/// бы принимать решения.
fn cancelled_check(source: EvalSource) -> CheckOutcome {
    let empty = Metrics {
        rmse: f32::NAN,
        mae: f32::NAN,
        rel_error: f32::NAN,
        r2: f32::NAN,
    };
    CheckOutcome {
        metrics: empty,
        per_output: Vec::new(),
        r2_std_folds: f32::NAN,
        source,
        model: None,
        cancelled: true,
    }
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
    let refit = refit_prepared(
        dataset,
        prepared,
        setup,
        final_epochs,
        final_init_seed,
        cancel,
        on_point,
        configure_model,
        post_train,
    )?;
    Ok(TrainingOutcome {
        development,
        final_model: refit.model,
        final_eval: refit.eval,
    })
}

/// Переобучить уже выбранную конфигурацию на всём search-pool и один раз
/// измерить её на отложенном test.
///
/// В отличие от [`run_training`], здесь нет development-фазы: выбор уже был
/// сделан поиском, включая агрегацию K-fold. Поэтому функция одинаково
/// работает для holdout и K-fold и не выдаёт один fold за общую оценку.
#[allow(clippy::too_many_arguments)]
pub fn refit(
    dataset: &Dataset,
    split: SplitPlan,
    setup: &TrainingSetup,
    final_init_seed: u64,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(Phase, &EpochPoint),
    configure_model: ConfigureModel<'_>,
    post_train: PostTrain<'_>,
) -> Result<RefitOutcome, String> {
    setup.validate()?;
    if setup.early_stopping.is_some() {
        return Err(
            "refit требует заранее выбранного числа эпох; early stopping без validation невозможен"
                .to_string(),
        );
    }
    let prepared = split.prepare(dataset.data())?;
    refit_prepared(
        dataset,
        prepared,
        setup,
        setup.train.epochs,
        final_init_seed,
        cancel,
        on_point,
        configure_model,
        post_train,
    )
}

#[allow(clippy::too_many_arguments)]
fn refit_prepared(
    dataset: &Dataset,
    prepared: PreparedSplit,
    setup: &TrainingSetup,
    final_epochs: usize,
    final_init_seed: u64,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(Phase, &EpochPoint),
    configure_model: ConfigureModel<'_>,
    post_train: PostTrain<'_>,
) -> Result<RefitOutcome, String> {
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
        return Ok(RefitOutcome {
            model: None,
            eval: None,
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
        return Ok(RefitOutcome {
            model: None,
            eval: None,
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

    Ok(RefitOutcome {
        model: Some(final_model),
        eval: Some(final_eval),
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
    use crate::metrics::{ConfigOrigin, RunOrigin};
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

    fn candidate(label: &str, epochs: usize) -> SearchCandidate {
        SearchCandidate {
            label: label.to_string(),
            setup: setup(epochs),
        }
    }

    #[test]
    fn recommendation_depends_only_on_validation_curve() {
        let points = [(1, 0.50), (5, 0.82), (10, 0.83), (20, 0.96)];
        let (epoch, reason) = recommended_epoch(points, 0.99, 0.02, 0.80).unwrap();
        assert_eq!(epoch, 5);
        assert!(reason.contains("плато"), "{reason}");
    }

    /// Holdout: одна модель — она же и оценка, поэтому её отдают наружу.
    /// K-fold: моделей столько же, сколько folds, и ни одна не представляет
    /// оценку — наружу идёт только свёртка по всем folds.
    #[test]
    fn check_covers_every_fold_and_returns_a_model_only_for_holdout() {
        let data = dataset(96);
        let s = setup(2);
        let never = AtomicBool::new(false);

        let mut folds_seen = Vec::new();
        let holdout = check_candidate(
            &data,
            SplitPlan::default(),
            &s,
            &never,
            &mut |_, _| {},
            &mut |_| {},
            &mut |fold, _, _, _| folds_seen.push(fold),
        )
        .unwrap();
        assert_eq!(folds_seen, vec![0]);
        assert_eq!(holdout.source, EvalSource::Validation);
        assert_eq!(holdout.r2_std_folds, 0.0, "у holdout разброса по folds нет");
        assert!(
            holdout.model.is_some(),
            "у holdout модель — это и есть оценка"
        );
        assert!(holdout.metrics.r2.is_finite());
        assert!(!holdout.cancelled);

        let mut folds_seen = Vec::new();
        let kfold = check_candidate(
            &data,
            SplitPlan::KFold {
                k: 3,
                folds_seed: 1,
                test_frac: 0.2,
                test_seed: 1,
            },
            &s,
            &never,
            &mut |_, _| {},
            &mut |_| {},
            &mut |fold, _, _, _| folds_seen.push(fold),
        )
        .unwrap();
        assert_eq!(folds_seen, vec![0, 1, 2], "конвейер идёт по каждому fold");
        assert_eq!(kfold.source, EvalSource::Cv { k: 3 });
        assert!(kfold.model.is_none(), "ни один fold не представляет CV");
        assert!(kfold.r2_std_folds.is_finite());
    }

    /// Конвейер интерпретации меняет саму модель, поэтому метрики снимаются
    /// ПОСЛЕ него: иначе проверка описывала бы не ту модель, которая поедет
    /// дальше.
    #[test]
    fn check_evaluates_the_model_the_pipeline_produced() {
        let data = dataset(64);
        let s = setup(2);
        let never = AtomicBool::new(false);

        let honest = check_candidate(
            &data,
            SplitPlan::default(),
            &s,
            &never,
            &mut |_, _| {},
            &mut |_| {},
            &mut |_, _, _, _| {},
        )
        .unwrap();

        // Хук портит модель: обнуляет её выход. Если бы оценка снималась до
        // хука, метрики совпали бы с честными.
        let broken = check_candidate(
            &data,
            SplitPlan::default(),
            &s,
            &never,
            &mut |_, _| {},
            &mut |_| {},
            &mut |_, trained, _, _| {
                for p in trained.model.parameters() {
                    p.update_data(|d, _| d.fill(0.0));
                }
            },
        )
        .unwrap();

        assert!(
            broken.metrics.r2 < honest.metrics.r2,
            "оценка обязана относиться к модели после конвейера: {} vs {}",
            broken.metrics.r2,
            honest.metrics.r2
        );
    }

    /// Отмена не даёт оценки: подставленные нули выглядели бы как результат.
    #[test]
    fn cancelled_check_reports_no_usable_evaluation() {
        let data = dataset(48);
        let cancelled = AtomicBool::new(true);
        let outcome = check_candidate(
            &data,
            SplitPlan::default(),
            &setup(2),
            &cancelled,
            &mut |_, _| {},
            &mut |_| {},
            &mut |_, _, _, _| {},
        )
        .unwrap();
        assert!(outcome.cancelled);
        assert!(outcome.model.is_none());
        assert!(outcome.metrics.r2.is_nan());
    }

    #[test]
    fn recommendation_prefers_target_then_plateau_then_last() {
        // Достигнутый target важнее плато: остановка на первой такой точке.
        let target = [(1, 0.5), (2, 0.96), (5, 0.99)];
        assert_eq!(recommended_epoch(target, 0.95, 0.02, 0.8).unwrap().0, 2);
        // Плато засчитывается только после plateau_min, иначе ранний шум
        // остановил бы обучение на первой же паре близких точек.
        let plateau = [(1, 0.5), (2, 0.82), (5, 0.83), (10, 0.835)];
        assert_eq!(recommended_epoch(plateau, 0.99, 0.02, 0.8).unwrap().0, 2);
        // Ни target, ни плато — последняя точка, а не отсутствие ответа.
        let growing = [(1, 0.2), (2, 0.5)];
        assert_eq!(recommended_epoch(growing, 0.99, 0.02, 0.8).unwrap().0, 2);
        assert!(recommended_epoch([], 0.9, 0.02, 0.8).is_none());
    }

    #[test]
    fn cost_counts_runs_and_epochs_before_launch() {
        let cost = SearchCost::new(6, 2, 5, 40);
        assert_eq!(cost.runs, 60);
        assert_eq!(cost.epochs_upper_bound(), 2400);
        let text = cost.describe();
        assert!(text.contains("6 конфигураций"), "{text}");
        assert!(text.contains("× 5 folds"), "{text}");
        assert!(text.contains("60 прогонов"), "{text}");

        // У holdout про folds не пишем — это шум.
        let holdout = SearchCost::new(3, 1, 1, 10).describe();
        assert!(!holdout.contains("folds"));
        assert!(SearchCost::new(1, 1, 1, 1)
            .describe()
            .contains("до 1 эпохи на прогон"));
    }

    #[test]
    fn cost_comes_from_candidates_and_plan() {
        let candidates = vec![candidate("a", 10), candidate("b", 25)];
        let plan = SearchPlan {
            seeds: vec![0, 1, 2],
            objective: SearchObjective::default(),
        };
        let cost = search_cost(&candidates, &plan, 4);
        assert_eq!(cost.configs, 2);
        assert_eq!(cost.seeds, 3);
        assert_eq!(cost.folds, 4);
        assert_eq!(cost.runs, 24);
        // Верхняя граница — по самому длинному кандидату.
        assert_eq!(cost.max_epochs, 25);
        // Но сумма учитывает реальную длину обоих кандидатов, а не считает
        // короткий как 25 эпох: (10 + 25) × 3 seeds × 4 folds.
        assert_eq!(cost.epochs_upper_bound(), 420);
        assert!(cost.describe().starts_with("2 конфигурации"));
    }

    #[test]
    fn default_objective_is_worst_output_r2() {
        assert_eq!(
            SearchPlan::default().objective,
            SearchObjective::WorstOutputR2
        );
        assert_eq!(SearchPlan::default().seeds, vec![0]);
    }

    #[test]
    fn plan_rejects_empty_or_duplicate_seeds() {
        let bad = SearchPlan {
            seeds: vec![],
            objective: SearchObjective::default(),
        };
        assert!(bad.validate().is_err());
        let dup = SearchPlan {
            seeds: vec![0, 0],
            objective: SearchObjective::default(),
        };
        assert!(dup.validate().unwrap_err().contains("повторя"));
    }

    /// Худший выход не должен теряться в среднем: это и есть причина, по
    /// которой worst-output R² выбран целью по умолчанию.
    #[test]
    fn objective_scores_differ_on_a_failed_output() {
        let good = Metrics {
            rmse: 0.0,
            mae: 0.0,
            rel_error: 0.0,
            r2: 1.0,
        };
        let bad = Metrics {
            r2: -1.0,
            ..good.clone()
        };
        let eval = ConfigEval {
            mean: good.clone(),
            per_output_mean: vec![good.clone(), bad],
            r2_std_seeds: 0.0,
            r2_std_folds: 0.0,
            origin: ConfigOrigin {
                init_seeds: vec![0],
                folds: 1,
                source: EvalSource::Validation,
            },
        };
        let runs = [RunEval {
            metrics: good.clone(),
            per_output: vec![good],
            origin: RunOrigin {
                fold: None,
                init_seed: 0,
            },
        }];
        assert_eq!(SearchObjective::AggregateR2.score(&eval, &runs), 1.0);
        assert_eq!(SearchObjective::WorstOutputR2.score(&eval, &runs), -1.0);
        assert_eq!(SearchObjective::MeanOutputR2.score(&eval, &runs), 0.0);
    }

    #[test]
    fn search_ranks_candidates_and_reports_cost() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        // Один кандидат учится дольше — на простом sum он должен выиграть.
        let candidates = vec![candidate("короткий", 1), candidate("длинный", 12)];
        let mut seen = Vec::new();
        let results = search(
            &ds,
            &prepared.search,
            &candidates,
            &SearchPlan::default(),
            &never,
            &mut |row| seen.push(row.label.clone()),
        )
        .unwrap();

        assert_eq!(results.cost.configs, 2);
        assert_eq!(results.cost.runs, 2);
        assert_eq!(results.cost.max_epochs, 12);
        assert!(!results.cancelled);
        // Callback видит кандидатов в исходном порядке, результат — по счёту.
        assert_eq!(seen, vec!["короткий", "длинный"]);
        assert_eq!(results.rows[0].label, "длинный");
        assert!(results.rows[0].score >= results.rows[1].score);
        // Индекс кандидата переживает ранжирование.
        assert_eq!(results.rows[0].candidate, 1);
    }

    #[test]
    fn search_rejects_empty_input_and_respects_cancel() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        assert!(search(
            &ds,
            &prepared.search,
            &[],
            &SearchPlan::default(),
            &never,
            &mut |_| {}
        )
        .is_err());

        let cancelled = AtomicBool::new(true);
        let results = search(
            &ds,
            &prepared.search,
            &[candidate("a", 2)],
            &SearchPlan::default(),
            &cancelled,
            &mut |_| {},
        )
        .unwrap();
        assert!(results.cancelled);
        assert!(results.rows.is_empty());
    }

    #[test]
    fn search_validates_every_candidate_before_training() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let prepared = SplitPlan::default().prepare(ds.data()).unwrap();
        let mut invalid = candidate("сломанный", 1);
        invalid.setup.train.epochs = 0;
        let mut completed = 0;
        let error = search(
            &ds,
            &prepared.search,
            &[candidate("дорогой", 10), invalid],
            &SearchPlan::default(),
            &never,
            &mut |_| completed += 1,
        )
        .err()
        .expect("невалидный кандидат должен остановить поиск");

        assert_eq!(completed, 0, "ни один кандидат не должен обучаться");
        assert!(error.contains("сломанный"), "{error}");
    }

    #[test]
    fn non_finite_search_scores_are_ranked_last() {
        use std::cmp::Ordering;

        assert_eq!(compare_scores_desc(0.5, f32::NAN), Ordering::Less);
        assert_eq!(compare_scores_desc(f32::INFINITY, 0.5), Ordering::Greater);
        assert_eq!(
            compare_scores_desc(f32::NAN, f32::NEG_INFINITY),
            Ordering::Equal
        );
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

    /// Хук финальной обработки выполняется после обучения, но до test. Он
    /// может обнаружить ошибку или отмену и обязан иметь возможность закрыть
    /// последнюю границу без единого замера test.
    #[test]
    fn cancellation_from_final_post_train_does_not_open_test() {
        let ds = dataset(120);
        let cancel = AtomicBool::new(false);
        let outcome = refit(
            &ds,
            SplitPlan::default(),
            &setup(1),
            DEFAULT_FINAL_INIT_SEED,
            &cancel,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |phase, _, _, _| {
                assert_eq!(phase, Phase::Final);
                cancel.store(true, Ordering::Relaxed);
            },
        )
        .unwrap();

        assert!(outcome.model.is_none());
        assert!(outcome.eval.is_none());
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
    fn refit_supports_kfold_after_search() {
        let ds = dataset(120);
        let never = AtomicBool::new(false);
        let plan = SplitPlan::kfold_default();
        let outcome = refit(
            &ds,
            plan,
            &setup(1),
            DEFAULT_FINAL_INIT_SEED,
            &never,
            &mut |_, _| {},
            &mut |_, _| {},
            &mut |_, _, _, _| {},
        )
        .expect("K-fold refit должен обучиться на всём pool");

        assert!(outcome.model.is_some());
        let eval = outcome.eval.expect("test открывается после refit");
        assert_eq!(eval.origin.plan, plan);
        assert_eq!(eval.origin.final_init_seed, DEFAULT_FINAL_INIT_SEED);
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
