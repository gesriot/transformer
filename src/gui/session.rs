//! Состояние сессии GUI: активная модель, активные данные и их общий доступ.
//!
//! Экраны живут в соседних модулях и работают с этим состоянием: здесь только
//! то, что общее для всех — поля [`App`], разбор событий worker-а и переносы
//! конфигурации между экранами.

use super::data::{MarkupState, PrepareForm};
#[cfg(feature = "demo")]
use super::demo::TextForm;
use super::messages::{
    Command, CurvePoint, DatasetOrigin, DiagnosticsResult, Event, KanModelInfo, KanSymbolicInfo,
    ModelOrigin, PreparedData,
};
use super::model::{ModelInfo, ModelView};
use super::train::{CustomSearchForm, SearchForm, TrainForm, TrainingMode};
use super::worker::Worker;
use crate::data::OutOfRange;
use crate::encoders::ValueEncoderKind;
use crate::fingerprint::DatasetFingerprint;
use crate::interpret::InterpretOverrides;
use crate::lifecycle::{CheckEval, CheckedRun, Lifecycle, TestDisclosure};
use crate::markup::{Message, TableProfile};
use crate::split::SplitPlan;
use crate::sweep::{self, SweepChoice, SweepRow};
use crate::train::LrSchedule;
use crate::training::Phase;
use eframe::egui;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq)]
/// Разделы окна. Их пять, и они соответствуют шагам работы, а не командам:
/// открыть данные, обучить, разобраться с моделью, посчитать прогноз.
///
/// «Демо» стоит особняком: встроенные задачи и char-LM к рабочему сценарию не
/// относятся и держатся отдельно, чтобы не выглядеть его частью.
pub(super) enum Section {
    Data,
    Training,
    Model,
    Predict,
    #[cfg(feature = "demo")]
    Demo,
}

impl Section {
    pub(super) fn label(self) -> &'static str {
        match self {
            Section::Data => "Данные",
            Section::Training => "Обучение",
            Section::Model => "Модель",
            Section::Predict => "Прогноз",
            #[cfg(feature = "demo")]
            Section::Demo => "Демо",
        }
    }
}

/// Средняя кривая по folds.
///
/// Точка берётся только там, где замер есть у КАЖДОГО fold: иначе «среднее»
/// менялось бы вместе с числом слагаемых, и провал одного fold выглядел бы как
/// улучшение.
fn mean_curve(
    curves: &[Vec<CurvePoint>],
    value: impl Fn(&CurvePoint) -> Option<f32>,
) -> Vec<[f64; 2]> {
    let mut by_epoch: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    for curve in curves {
        for point in curve {
            if let Some(v) = value(point) {
                by_epoch.entry(point.epoch).or_default().push(v);
            }
        }
    }
    by_epoch
        .into_iter()
        .filter(|(_, values)| values.len() == curves.len())
        .map(|(epoch, values)| {
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            [epoch as f64, mean as f64]
        })
        .collect()
}

/// Одно сообщение на все экраны: данные общие, значит и текст общий.
pub(super) const NO_DATASET: &str = "сначала откройте данные";

#[cfg(feature = "demo")]
pub(super) const BLACKBOXES: &[&str] = &["sum", "product", "sine", "polynomial", "projectile"];
pub(super) const KAN_CURVE_SAMPLES: usize = 201;

/// Активный набор данных сессии.
///
/// Один на всё окно: экраны обучения, поиска и кривой по эпохам берут данные
/// отсюда, а не выбирают файл каждый сам. Иначе один и тот же файл открывался
/// бы по-разному в разных формах, а ручная разметка терялась бы.
pub(super) struct ActiveDataset {
    pub(super) prepared: PreparedData,
    /// Профиль есть только у таблиц: `.tnum` и чёрный ящик уже размечены.
    pub(super) profile: Option<TableProfile>,
    /// Замечания, зависящие от подтверждённых ролей: например, аффинные
    /// зависимости между входами. `TableProfile` их не содержит.
    pub(super) role_messages: Vec<Message>,
    /// План разбиения — свойство набора данных, а не отдельного запуска.
    pub(super) split: SplitPlan,
    /// Как читалась таблица: «Разметить заново» должно открыть её так же.
    pub(super) table_has_header: bool,
    /// Номер набора в сессии: по нему видно, что данные сменились.
    pub(super) revision: u64,
    /// Отпечаток данных. Считается один раз при открытии: по нему решается,
    /// те же это данные или другие, — в том числе после перезапуска.
    pub(super) fingerprint: DatasetFingerprint,
}

pub(super) fn model_kind_label(kind: crate::numeric_model::ModelKind) -> &'static str {
    match kind {
        crate::numeric_model::ModelKind::Transformer => "transformer",
        crate::numeric_model::ModelKind::Mlp => "mlp",
        crate::numeric_model::ModelKind::Kan => "kan",
    }
}

pub(super) fn split_plan_label(plan: SplitPlan) -> String {
    match plan {
        SplitPlan::Holdout {
            train_frac,
            val_frac,
            split_seed,
        } => format!(
            "holdout {:.0}/{:.0}/{:.0}, split seed {split_seed}",
            train_frac * 100.0,
            val_frac * 100.0,
            (1.0 - train_frac - val_frac) * 100.0
        ),
        SplitPlan::KFold {
            k,
            folds_seed,
            test_frac,
            test_seed,
        } => format!(
            "{k}-fold, test {:.0}%, folds seed {folds_seed}, test seed {test_seed}",
            test_frac * 100.0
        ),
    }
}

impl ActiveDataset {
    pub(super) fn new(
        prepared: PreparedData,
        profile: Option<TableProfile>,
        table_has_header: bool,
        revision: u64,
    ) -> Self {
        // Схема и данные согласованы ещё в worker-е, поэтому расхождение здесь
        // означало бы ошибку в программе, а не во входных данных.
        let fingerprint = DatasetFingerprint::of(&prepared.data, &prepared.schema)
            .expect("данные и схема согласованы при открытии набора");
        Self {
            fingerprint,
            prepared,
            profile,
            role_messages: Vec::new(),
            split: SplitPlan::default(),
            table_has_header,
            revision,
        }
    }

    pub(super) fn with_role_messages(mut self, messages: Vec<Message>) -> Self {
        self.role_messages = messages;
        self
    }

    /// Строка для шапки: что открыто и какой оно формы.
    pub(super) fn summary(&self) -> String {
        let schema = &self.prepared.schema;
        format!(
            "{} · {} строк · {} вход → {} выход · {}",
            self.prepared.origin.short_name(),
            self.prepared.data.len(),
            schema.n_inputs(),
            schema.n_outputs(),
            self.split_summary()
        )
    }

    /// Короткое, но проверяемое описание протокола для постоянной шапки.
    pub(super) fn split_summary(&self) -> String {
        let rows = self.prepared.data.len();
        match self.split {
            SplitPlan::Holdout {
                train_frac,
                val_frac,
                ..
            } => {
                let train = (rows as f32 * train_frac).round() as usize;
                let validation = (rows as f32 * val_frac).round() as usize;
                let test = rows.saturating_sub(train + validation);
                format!("holdout {train}/{validation}/{test}")
            }
            SplitPlan::KFold { k, test_frac, .. } => {
                let test = (rows as f32 * test_frac).round() as usize;
                let pool = rows.saturating_sub(test);
                format!("{k}-fold по {pool} строкам + test {test}")
            }
        }
    }

    /// Сколько замечаний к качеству данных нашёл профиль. У `.tnum` и чёрного
    /// ящика профиля нет — там и предупреждать не о чем.
    pub(super) fn data_notes(&self) -> usize {
        self.profile.as_ref().map_or(0, |p| p.messages().len()) + self.role_messages.len()
    }
}

pub(crate) struct App {
    pub(super) worker: Worker,
    pub(super) section: Section,
    pub(super) status: String,
    pub(super) form: TrainForm,
    /// Активный набор данных сессии; `None` — данные ещё не открыты.
    pub(super) dataset: Option<ActiveDataset>,
    /// Идёт чтение набора данных: запускать что-либо на старых данных нельзя.
    pub(super) dataset_opening: bool,
    pub(super) dataset_revision: u64,
    pub(super) training: bool,
    pub(super) searching: bool,
    /// Loss development-фазы и финального refit хранятся отдельно: обе фазы
    /// начинают нумерацию эпох с единицы.
    pub(super) loss_curve: Vec<[f64; 2]>,
    /// R² на validation по эпохам: кривая обучения вместо отдельного сценария.
    pub(super) val_curve: Vec<[f64; 2]>,
    /// Как часто снимать validation во время обучения (0 — не снимать).
    pub(super) eval_every: usize,
    pub(super) final_loss_curve: Vec<[f64; 2]>,
    /// Сколько folds стоит за нарисованной кривой: 1 — обычная кривая одного
    /// обучения, больше — среднее по folds.
    pub(super) curve_folds: usize,
    pub(super) train_parameter_count: Option<usize>,
    // Predict (UI-M5)
    pub(super) model_info: Option<ModelInfo>,
    pub(super) model_view: ModelView,
    /// Worker читает и профилирует таблицу. Пока ответ не пришёл, нельзя
    /// запустить действие со старым активным набором.
    pub(super) table_opening: bool,
    /// Открытый диалог разметки таблицы.
    pub(super) markup: Option<MarkupState>,
    pub(super) predict_inputs: Vec<f32>,
    pub(super) predict_outputs: Option<Vec<f32>>,
    pub(super) extrapolation: Vec<OutOfRange>,
    pub(super) batch_predicting: bool,
    // KAN curves (данные графика приходят из worker, не тензоры)
    pub(super) kan_info: Option<KanModelInfo>,
    pub(super) kan_layer: usize,
    pub(super) kan_input: usize,
    pub(super) kan_output: usize,
    pub(super) kan_curve: Vec<[f64; 2]>,
    // KAN symbolic formulas (worker возвращает только текст и метрики)
    pub(super) kan_symbolic: Option<KanSymbolicInfo>,
    pub(super) kan_symbolic_pending: bool,
    // Diagnose (UI-M6)
    pub(super) diagnostics: Option<DiagnosticsResult>,
    // Общие настройки и результаты поиска
    pub(super) search_form: SearchForm,
    pub(super) search_rows: Vec<SweepRow>,
    pub(super) search_total: Option<(usize, usize)>,
    pub(super) search_cancelled: bool,
    /// Данные и разбиение, на которых получен результат поиска. После смены
    /// данных запускать по нему финальное обучение нельзя: строки описывают
    /// уже другой набор.
    pub(super) search_stamp: Option<(u64, SplitPlan)>,
    /// Что проверено и не потрачен ли test на этих данных.
    pub(super) lifecycle: Lifecycle,
    /// Отчёты конвейера интерпретации по фазам.
    /// Запускать ли конвейер интерпретации и с какими переопределениями.
    pub(super) interpret_enabled: bool,
    pub(super) interpret_overrides: InterpretOverrides,
    // Режим и ручная сетка поиска
    pub(super) custom_form: CustomSearchForm,
    pub(super) mode: TrainingMode,
    /// Строка поиска, выбранная для финального обучения.
    pub(super) search_selected: Option<usize>,
    // Text (UI-M7)
    #[cfg(feature = "demo")]
    pub(super) text_form: TextForm,
    #[cfg(feature = "demo")]
    pub(super) text_training: bool,
    #[cfg(feature = "demo")]
    pub(super) text_curve: Vec<[f64; 2]>,
    #[cfg(feature = "demo")]
    pub(super) text_ready: bool,
    #[cfg(feature = "demo")]
    pub(super) text_vocab_size: Option<usize>,
    #[cfg(feature = "demo")]
    pub(super) generated_text: String,
    // Совместимая явная конвертация в .tnum внутри раздела «Данные».
    pub(super) prepare_form: PrepareForm,
}

impl App {
    pub(crate) fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            worker: Worker::spawn(cc.egui_ctx.clone()),
            section: Section::Data,
            status: "–".to_string(),
            form: TrainForm::default(),
            dataset: None,
            dataset_opening: false,
            dataset_revision: 0,
            training: false,
            searching: false,
            loss_curve: Vec::new(),
            val_curve: Vec::new(),
            eval_every: 5,
            final_loss_curve: Vec::new(),
            curve_folds: 1,
            train_parameter_count: None,
            model_info: None,
            model_view: ModelView::Summary,
            table_opening: false,
            markup: None,
            predict_inputs: Vec::new(),
            predict_outputs: None,
            extrapolation: Vec::new(),
            batch_predicting: false,
            kan_info: None,
            kan_layer: 0,
            kan_input: 0,
            kan_output: 0,
            kan_curve: Vec::new(),
            kan_symbolic: None,
            kan_symbolic_pending: false,
            diagnostics: None,
            search_form: SearchForm::default(),
            search_rows: Vec::new(),
            search_total: None,
            search_cancelled: false,
            search_stamp: None,
            lifecycle: Lifecycle::default(),
            interpret_enabled: false,
            interpret_overrides: InterpretOverrides::default(),
            custom_form: CustomSearchForm::default(),
            mode: TrainingMode::Single,
            search_selected: None,
            #[cfg(feature = "demo")]
            text_form: TextForm::default(),
            #[cfg(feature = "demo")]
            text_training: false,
            #[cfg(feature = "demo")]
            text_curve: Vec::new(),
            #[cfg(feature = "demo")]
            text_ready: false,
            #[cfg(feature = "demo")]
            text_vocab_size: None,
            #[cfg(feature = "demo")]
            generated_text: String::new(),
            prepare_form: PrepareForm::default(),
        }
    }

    pub(super) fn drain_events(&mut self) {
        while let Some(ev) = self.worker.try_recv() {
            match ev {
                Event::Status(s) => self.status = s,
                Event::Error(e) => {
                    self.training = false;
                    self.searching = false;
                    #[cfg(feature = "demo")]
                    {
                        self.text_training = false;
                    }
                    self.batch_predicting = false;
                    self.kan_symbolic_pending = false;
                    self.table_opening = false;
                    self.dataset_opening = false;
                    self.status = format!("Ошибка: {e}");
                }
                Event::TrainStarted {
                    total_epochs,
                    parameter_count,
                } => {
                    self.training = true;
                    self.loss_curve.clear();
                    self.val_curve.clear();
                    self.final_loss_curve.clear();
                    self.train_parameter_count = Some(parameter_count);
                    self.status =
                        format!("обучение: 0/{total_epochs} эпох, {parameter_count} параметров");
                }
                Event::Epoch {
                    phase,
                    fold,
                    epoch,
                    loss,
                    val_r2,
                } => {
                    let label = match phase {
                        Phase::Development => "development",
                        Phase::Final => "финальное обучение",
                    };
                    // Живая кривая рисуется только для первого fold: у K-fold
                    // номера эпох повторяются, и общая ломаная не описывает ни
                    // один прогон. Средняя кривая приходит по завершении.
                    if fold == 0 {
                        let curve = match phase {
                            Phase::Development => &mut self.loss_curve,
                            Phase::Final => &mut self.final_loss_curve,
                        };
                        curve.push([epoch as f64, loss as f64]);
                        // Validation снимается только в точках расписания и
                        // только в фазе разработки: финальная на validation
                        // училась.
                        if let (Phase::Development, Some(r2)) = (phase, val_r2) {
                            self.val_curve.push([epoch as f64, r2 as f64]);
                        }
                    }
                    self.status = match (fold, val_r2) {
                        (0, Some(r2)) => {
                            format!("{label}: эпоха {epoch}, loss {loss:.5}, validation R² {r2:.5}")
                        }
                        (0, None) => format!("{label}: эпоха {epoch}, loss {loss:.5}"),
                        (f, _) => format!("{label}: fold {}, эпоха {epoch}, loss {loss:.5}", f + 1),
                    };
                }
                Event::TrainDone {
                    stamp,
                    metrics,
                    per_output,
                    check_source,
                    r2_std_folds,
                    curves,
                    final_eval,
                    check_interpret,
                    cancelled,
                } => {
                    self.training = false;
                    let mut protocol_error = None;
                    let mut disclosed = false;
                    if !cancelled {
                        // Результат подписывается СВОИМ отпечатком, а не
                        // текущей формой: если её успели изменить, проверка
                        // относится к прежнему кандидату и к нему же вернётся,
                        // если поля вернуть обратно.
                        if let (Some(m), Some(per), Some(source)) =
                            (&metrics, &per_output, check_source)
                        {
                            // Источник считает пул, который реально работал.
                            // При расхождении с отпечатком безопаснее не
                            // разблокировать final, чем приписать проверку
                            // другому протоколу.
                            if source != stamp.eval_source() {
                                protocol_error = Some(
                                    "внутренняя ошибка: оценка и stamp описывают разные протоколы"
                                        .to_string(),
                                );
                            } else {
                                self.lifecycle.record_check(CheckedRun {
                                    stamp: (*stamp).clone(),
                                    eval: CheckEval {
                                        metrics: m.clone(),
                                        per_output: per.clone(),
                                        r2_std_folds: r2_std_folds.unwrap_or(0.0),
                                    },
                                    interpret: check_interpret,
                                });
                            }
                        }
                        // Раскрытие test фиксируется всегда: замер уже сделан,
                        // и правка формы его не возвращает.
                        if let Some(eval) = &final_eval {
                            if eval.origin.plan != stamp.split {
                                protocol_error = Some(
                                    "внутренняя ошибка: test и stamp описывают разные split"
                                        .to_string(),
                                );
                            }
                            disclosed = true;
                            self.lifecycle.record_disclosure(TestDisclosure {
                                stamp: (*stamp).clone(),
                                eval: eval.clone(),
                            });
                        }
                        // Кривая по нескольким folds — это среднее по эпохам,
                        // а не склейка: у каждого fold свои номера эпох.
                        if curves.len() > 1 {
                            self.loss_curve = mean_curve(&curves, |p| Some(p.train_loss));
                            self.val_curve = mean_curve(&curves, |p| p.val_r2);
                        }
                        self.curve_folds = curves.len().max(1);
                    }
                    self.status = if let Some(error) = protocol_error {
                        error
                    } else if cancelled {
                        "обучение отменено".to_string()
                    } else if disclosed {
                        "финальное обучение завершено, test открыт".to_string()
                    } else {
                        "проверка завершена".to_string()
                    };
                }
                Event::DatasetOpened { data } => {
                    self.dataset_opening = false;
                    self.status = format!("данные открыты: {}", data.origin.short_name());
                    let revision = self.next_revision();
                    self.set_dataset(ActiveDataset::new(data, None, true, revision));
                }
                Event::TableOpened {
                    path,
                    has_header,
                    table,
                    profile,
                    suggested_inputs,
                    suggested_categories,
                } => {
                    self.table_opening = false;
                    self.status = format!("таблица открыта: {path}");
                    self.markup = Some(MarkupState::new(
                        path,
                        has_header,
                        *table,
                        *profile,
                        suggested_inputs,
                        &suggested_categories,
                    ));
                }
                Event::ModelReady {
                    schema,
                    kind,
                    source,
                    model_origin,
                    parameter_count,
                    kan,
                    interpret,
                    report,
                } => {
                    let n_inputs = schema.n_inputs();
                    // Метрики не хранятся отдельно от отпечатка: раздел
                    // «Модель» показывает только те, что относятся к этой
                    // самой модели. Чистить остаётся лишь кривые чужого
                    // запуска.
                    if matches!(model_origin, ModelOrigin::Checkpoint) {
                        self.loss_curve.clear();
                        self.val_curve.clear();
                        self.final_loss_curve.clear();
                        self.train_parameter_count = None;
                    }
                    // Загруженный checkpoint возвращает потраченный test:
                    // отпечаток данных в отчёте говорит, на чём он был открыт.
                    if let Some(report) = &report {
                        if let Some(final_run) = &report.final_run {
                            self.lifecycle.record_disclosure(TestDisclosure {
                                stamp: report.stamp.clone(),
                                eval: final_run.eval.clone(),
                            });
                        }
                    }
                    self.model_info = Some(ModelInfo {
                        schema,
                        kind,
                        source,
                        origin: model_origin,
                        interpret: interpret.map(|report| *report),
                        report: report.map(|report| *report),
                        parameter_count,
                    });
                    self.predict_inputs = vec![0.0; n_inputs];
                    self.predict_outputs = None;
                    self.extrapolation.clear();
                    self.kan_info = kan;
                    self.kan_layer = 0;
                    self.kan_input = 0;
                    self.kan_output = 0;
                    self.kan_curve.clear();
                    self.kan_symbolic = None;
                    self.kan_symbolic_pending = false;
                    self.diagnostics = None;
                    self.request_kan_curve();
                    self.status = "модель готова к предсказанию".to_string();
                }
                Event::PredictResult {
                    outputs,
                    extrapolation,
                } => {
                    self.predict_outputs = Some(outputs);
                    self.extrapolation = extrapolation;
                }
                Event::ExportDone { output, summary } => {
                    self.batch_predicting = false;
                    let mut text =
                        format!("Таблица с прогнозами: {output} ({} строк", summary.rows);
                    if summary.extrapolated_rows > 0 {
                        text.push_str(&format!(
                            ", {} вне обученного диапазона",
                            summary.extrapolated_rows
                        ));
                    }
                    if !summary.replaced.is_empty() {
                        text.push_str(&format!("; заменены: {}", summary.replaced.join(", ")));
                    }
                    if !summary.added.is_empty() {
                        text.push_str(&format!("; добавлены: {}", summary.added.join(", ")));
                    }
                    text.push(')');
                    self.status = text;
                }
                Event::KanEdgeCurve {
                    layer,
                    input,
                    output,
                    points,
                } => {
                    if (layer, input, output) == (self.kan_layer, self.kan_input, self.kan_output) {
                        self.kan_curve = points
                            .into_iter()
                            .map(|(x, y)| [x as f64, y as f64])
                            .collect();
                    }
                }
                Event::KanSymbolic { result } => {
                    self.kan_symbolic = Some(result);
                    self.kan_symbolic_pending = false;
                    self.status = "символьные формулы готовы".to_string();
                }
                Event::Diagnostics { result } => {
                    self.diagnostics = Some(result);
                    self.status = "диагностика готова".to_string();
                }
                Event::SearchStarted {
                    total_configs,
                    total_runs,
                } => {
                    self.searching = true;
                    self.search_rows.clear();
                    self.search_total = Some((total_configs, total_runs));
                    self.search_cancelled = false;
                    self.status =
                        format!("поиск: 0/{total_configs} конфигураций ({total_runs} прогонов)");
                }
                Event::SearchRow { row } => {
                    self.search_rows.push(row);
                    self.sort_search_rows();
                    if let Some((total_configs, _)) = self.search_total {
                        self.status = format!(
                            "поиск: {}/{total_configs} конфигураций",
                            self.search_rows.len()
                        );
                    }
                }
                Event::SearchDone { rows, cancelled } => {
                    self.searching = false;
                    self.search_rows = rows;
                    self.sort_search_rows();
                    self.search_cancelled = cancelled;
                    self.status = if cancelled {
                        "поиск отменён".to_string()
                    } else {
                        "поиск завершён".to_string()
                    };
                }
                #[cfg(feature = "demo")]
                Event::TextStarted { total_steps } => {
                    self.text_training = true;
                    self.text_ready = false;
                    self.text_curve.clear();
                    self.generated_text.clear();
                    self.text_vocab_size = None;
                    self.status = format!("text: 0/{total_steps} шагов");
                }
                #[cfg(feature = "demo")]
                Event::TextProgress { step, loss } => {
                    self.text_curve.push([step as f64, loss.exp() as f64]);
                    self.status = format!("text шаг {step}: loss {loss:.4}, ppl {:.2}", loss.exp());
                }
                #[cfg(feature = "demo")]
                Event::TextDone {
                    final_loss,
                    cancelled,
                    vocab_size,
                    seed_hint,
                } => {
                    self.text_training = false;
                    self.text_vocab_size = Some(vocab_size);
                    if !cancelled {
                        self.text_ready = true;
                        if self.text_form.seed_text.is_empty() {
                            self.text_form.seed_text = seed_hint;
                        }
                    }
                    self.status = match (cancelled, final_loss) {
                        (true, _) => "text обучение отменено".to_string(),
                        (false, Some(loss)) => {
                            format!("text готов: loss {loss:.4}, ppl {:.2}", loss.exp())
                        }
                        (false, None) => "text готов".to_string(),
                    };
                }
                #[cfg(feature = "demo")]
                Event::GeneratedText { text } => {
                    self.generated_text = text;
                    self.status = "генерация готова".to_string();
                }
                Event::PrepareDone {
                    output,
                    rows,
                    n_inputs,
                    n_outputs,
                } => {
                    self.status = format!(
                        "записано {output}: {rows} строк, {n_inputs} вход -> {n_outputs} выход"
                    );
                }
            }
        }
    }

    pub(super) fn busy(&self) -> bool {
        #[cfg(feature = "demo")]
        if self.text_training {
            return true;
        }
        self.training
            || self.searching
            || self.batch_predicting
            || self.kan_symbolic_pending
            || self.table_opening
            || self.dataset_opening
            || self.markup.is_some()
    }

    /// Постоянная шапка: что открыто и что обучено. Видна из любого раздела,
    /// потому что оба объекта — общие для всей сессии.
    pub(super) fn ui_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Данные:");
            match &self.dataset {
                Some(active) => {
                    ui.label(active.summary());
                    let notes = active.data_notes();
                    if notes > 0 {
                        ui.label(format!("· замечаний: {notes}"));
                    }
                }
                None => {
                    ui.label("не открыты");
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Модель:");
            match &self.model_info {
                Some(info) => {
                    ui.label(format!(
                        "{} · {} · {}",
                        model_kind_label(info.kind),
                        info.source,
                        info.origin.label()
                    ));
                    // В шапке — только та метрика, что относится к этой самой
                    // модели: чужая рядом с ней читалась бы как её собственная.
                    let stamp = match &info.origin {
                        ModelOrigin::Development(stamp) | ModelOrigin::Final(stamp) => {
                            Some(stamp.as_ref())
                        }
                        ModelOrigin::Checkpoint => None,
                    };
                    if let Some(eval) = stamp.and_then(|s| self.lifecycle.disclosure_for(s)) {
                        ui.label(format!("· test R² {:.3}", eval.eval.metrics.r2));
                    } else if let Some(run) = stamp.and_then(|s| self.lifecycle.checked_for(s)) {
                        ui.label(format!("· validation R² {:.3}", run.eval.metrics.r2));
                    }
                    // Модель и данные могли разойтись: причина важнее самого
                    // факта несовместимости.
                    if let Some(reason) = self.schema_mismatch() {
                        ui.colored_label(
                            egui::Color32::from_rgb(200, 120, 0),
                            format!("⚠ не подходит к данным: {reason}"),
                        );
                    }
                }
                None => {
                    ui.label("не обучена");
                }
            }
        });
    }

    /// Почему активная модель не подходит к активным данным. `None` — подходит
    /// либо сравнивать нечего.
    pub(super) fn schema_mismatch(&self) -> Option<String> {
        let model = self.model_info.as_ref()?;
        let data = self.dataset.as_ref()?;
        model.schema.compatibility_with(&data.prepared.schema).err()
    }

    /// Действия выбора активного набора. Они живут только в разделе «Данные»;
    /// остальные разделы видят результат в постоянной шапке.
    pub(super) fn ui_dataset_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let idle = !self.busy();
            if ui
                .add_enabled(idle, egui::Button::new("Открыть .tnum…"))
                .clicked()
            {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("tnum", &["tnum"])
                    .pick_file()
                {
                    self.open_dataset(DatasetOrigin::File(p.display().to_string()));
                }
            }
            if ui
                .add_enabled(idle, egui::Button::new("Открыть таблицу…"))
                .clicked()
            {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter(
                        "таблицы",
                        &["xlsx", "xlsm", "xlsb", "xls", "ods", "csv", "tsv", "txt"],
                    )
                    .pick_file()
                {
                    self.open_table(p.display().to_string(), true);
                }
            }
            if let Some((path, has_header)) = self.markup_source() {
                if ui
                    .add_enabled(idle, egui::Button::new("Разметить заново…"))
                    .clicked()
                {
                    self.open_table(path, has_header);
                }
            }
        });
    }

    /// Путь таблицы, если активные данные пришли из разметки.
    fn markup_source(&self) -> Option<(String, bool)> {
        let active = self.dataset.as_ref()?;
        match &active.prepared.origin {
            DatasetOrigin::Table(path) => Some((path.clone(), active.table_has_header)),
            _ => None,
        }
    }

    /// Активные данные и их план разбиения для команды worker-у.
    ///
    /// Ошибка — данные не открыты либо выбранный план нельзя применить к их
    /// числу строк.
    pub(super) fn active_data(&self) -> Result<(PreparedData, SplitPlan), String> {
        let active = self
            .dataset
            .as_ref()
            .ok_or_else(|| NO_DATASET.to_string())?;
        active.split.validate(active.prepared.data.len())?;
        Ok((active.prepared.clone(), active.split))
    }

    /// Номер для следующего набора данных: ревизии не переиспользуются, иначе
    /// устаревший результат поиска мог бы совпасть с новым набором.
    pub(super) fn next_revision(&mut self) -> u64 {
        self.dataset_revision += 1;
        self.dataset_revision
    }

    /// Сменить активный набор. Результаты поиска относятся к старым данным и
    /// очищаются. Сама модель и её метрики остаются доступны: шапка отдельно
    /// покажет, совместима ли она с новым набором.
    pub(super) fn set_dataset(&mut self, dataset: ActiveDataset) {
        self.dataset = Some(dataset);
        // Проверка относилась к прежним данным. Раскрытие test остаётся: оно
        // привязано к своей ревизии и новую не блокирует.
        self.lifecycle.on_dataset_changed();
        self.search_rows.clear();
        self.search_total = None;
        self.search_stamp = None;
        self.search_selected = None;
    }

    pub(super) fn open_dataset(&mut self, origin: DatasetOrigin) {
        self.dataset_opening = true;
        self.status = format!("открываю {}…", origin.short_name());
        self.worker.send(Command::OpenDataset { origin });
    }

    pub(super) fn sort_search_rows(&mut self) {
        let objective = self.search_form.objective();
        sweep::sort_rows(&mut self.search_rows, objective);
    }

    /// Ревизия активного набора данных.
    pub(super) fn dataset_revision(&self) -> Option<u64> {
        self.dataset.as_ref().map(|active| active.revision)
    }

    /// Отпечаток активного набора данных.
    pub(super) fn dataset_fingerprint(&self) -> Option<DatasetFingerprint> {
        self.dataset.as_ref().map(|active| active.fingerprint)
    }

    /// Отпечаток активных данных: ревизия набора и план разбиения.
    pub(super) fn dataset_stamp(&self) -> Option<(u64, SplitPlan)> {
        self.dataset
            .as_ref()
            .map(|active| (active.revision, active.split))
    }

    /// Результат поиска годен, только если данные и разбиение не менялись.
    pub(super) fn search_matches_dataset(&self) -> bool {
        self.search_stamp.is_some() && self.search_stamp == self.dataset_stamp()
    }

    pub(super) fn open_table(&mut self, path: String, has_header: bool) {
        self.table_opening = true;
        self.status = format!("чтение {path}…");
        self.worker.send(Command::OpenTable { path, has_header });
    }

    pub(super) fn apply_choice_to_train(&mut self, choice: &SweepChoice) {
        self.form.kind = choice.kind;
        self.form.kan_width = choice.kan.width;
        self.form.kan_layers = choice.kan.layers;
        self.form.kan_grid = choice.kan.grid;
        self.form.d_model = choice.d_model;
        self.form.heads = choice.heads;
        self.form.layers = choice.layers;
        self.form.d_ff = choice.d_ff;
        self.form.venc = match choice.value.kind {
            ValueEncoderKind::Linear => 0,
            ValueEncoderKind::Mlp => 1,
            ValueEncoderKind::Fourier => 2,
        };
        self.form.fourier_bands = choice.value.fourier_bands;
        self.form.fourier_scale = choice.value.fourier_scale;
        self.form.mlp_width = choice.mlp_width;
        self.form.mlp_layers = choice.mlp_layers;
        self.form.lr = choice.lr;
        self.form.batch = choice.batch_size;
        // Поиск ранжировал на search-эпохах, а ручной режим получает полный
        // бюджет для возможной ручной правки.
        self.form.epochs = choice.final_epochs;
        self.form.seed = choice.seed;
        match choice.schedule {
            LrSchedule::Constant => {
                self.form.warmup_cosine = false;
            }
            LrSchedule::WarmupCosine {
                warmup_frac,
                min_lr_ratio,
            } => {
                self.form.warmup_cosine = true;
                self.form.warmup = warmup_frac;
                self.form.min_lr_ratio = min_lr_ratio;
            }
        }
        self.section = Section::Training;
        self.status = "выбранная конфигурация перенесена в ручной режим".to_string();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.ui_markup(ctx);

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                for section in [
                    Section::Data,
                    Section::Training,
                    Section::Model,
                    Section::Predict,
                    #[cfg(feature = "demo")]
                    Section::Demo,
                ] {
                    ui.selectable_value(&mut self.section, section, section.label());
                }
            });
        });

        egui::TopBottomPanel::top("session_header").show(ctx, |ui| {
            self.ui_header(ui);
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Статус:");
                ui.label(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match self.section {
                    Section::Data => self.ui_data(ui),
                    Section::Training => self.ui_train(ui),
                    Section::Model => self.ui_model(ui),
                    Section::Predict => self.ui_predict(ui),
                    #[cfg(feature = "demo")]
                    Section::Demo => self.ui_demo(ui),
                });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::schema::ModelSchema;
    use std::sync::Arc;

    fn active(revision: u64) -> ActiveDataset {
        let data = blackbox::sum().generate(16, 0);
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        ActiveDataset::new(
            PreparedData {
                origin: DatasetOrigin::Blackbox("sum".to_string()),
                data: Arc::new(data),
                schema,
            },
            None,
            true,
            revision,
        )
    }

    fn point(epoch: usize, loss: f32, val: Option<f32>) -> CurvePoint {
        CurvePoint {
            epoch,
            train_loss: loss,
            val_r2: val,
        }
    }

    /// Кривая по нескольким folds — среднее по одинаковым эпохам, а не
    /// склейка. Эпоха, замеренная не у всех folds, не берётся: иначе «среднее»
    /// менялось бы вместе с числом слагаемых.
    #[test]
    fn cv_curve_averages_only_epochs_measured_in_every_fold() {
        let curves = vec![
            vec![point(1, 1.0, Some(0.1)), point(2, 0.5, Some(0.3))],
            vec![point(1, 3.0, Some(0.3)), point(2, 1.5, None)],
        ];

        assert_eq!(
            mean_curve(&curves, |p| Some(p.train_loss)),
            vec![[1.0, 2.0], [2.0, 1.0]]
        );
        // Второй fold не мерил validation на второй эпохе — точки нет вовсе.
        let val = mean_curve(&curves, |p| p.val_r2);
        assert_eq!(val.len(), 1);
        assert_eq!(val[0][0], 1.0);
        assert!((val[0][1] - 0.2).abs() < 1e-6, "{:?}", val[0]);

        // Одна кривая усредняется сама с собой без изменений.
        let single = vec![vec![point(1, 2.0, Some(0.5))]];
        assert_eq!(mean_curve(&single, |p| p.val_r2), vec![[1.0, 0.5]]);
    }

    /// Отпечаток результата поиска — данные и разбиение. По нему видно, что
    /// финальное обучение запускать уже нельзя.
    #[test]
    fn search_stamp_follows_dataset_and_split() {
        let mut first = active(1);
        let stamp = (first.revision, first.split);

        // Тот же набор и то же разбиение — результат актуален.
        assert_eq!(stamp, (first.revision, first.split));

        // Сменилось разбиение — отпечаток другой.
        first.split = SplitPlan::kfold_default();
        assert_ne!(stamp, (first.revision, first.split));

        // Сменились данные — ревизия другая, даже если разбиение прежнее.
        let second = active(2);
        assert_ne!(stamp, (second.revision, second.split));
    }

    #[test]
    fn summary_describes_the_active_dataset() {
        let mut dataset = active(1);
        let text = dataset.summary();
        assert!(text.contains("чёрный ящик: sum"), "{text}");
        assert!(text.contains("16 строк"), "{text}");
        assert!(text.contains("2 вход → 1 выход"), "{text}");
        assert!(text.contains("holdout 11/2/3"), "{text}");

        dataset.split = SplitPlan::KFold {
            k: 5,
            folds_seed: 1,
            test_frac: 0.25,
            test_seed: 2,
        };
        assert_eq!(dataset.split_summary(), "5-fold по 12 строкам + test 4");
    }
}
