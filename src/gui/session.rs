//! Состояние сессии GUI: активная модель, активные данные и их общий доступ.
//!
//! Экраны живут в соседних модулях и работают с этим состоянием: здесь только
//! то, что общее для всех — поля [`App`], разбор событий worker-а и переносы
//! конфигурации между экранами.

use super::data::{MarkupState, PrepareForm};
#[cfg(feature = "demo")]
use super::demo::TextForm;
use super::messages::{
    Command, DatasetOrigin, DiagnosticsResult, Event, InterpretReports, KanModelInfo,
    KanSymbolicInfo, PreparedData, ValidationOrigin,
};
use super::model::{ModelInfo, ModelView};
use super::train::{CustomSearchForm, SearchForm, TrainForm, TrainingMode};
use super::worker::Worker;
use crate::data::OutOfRange;
use crate::encoders::ValueEncoderKind;
use crate::interpret::InterpretOverrides;
use crate::markup::{Message, TableProfile};
use crate::metrics::Metrics;
use crate::split::{FinalEval, SplitPlan};
use crate::sweep::{self, SweepChoice, SweepRow};
use crate::train::LrSchedule;
use crate::training::Phase;
use eframe::egui;

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
        Self {
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

pub struct App {
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
    pub(super) metrics: Option<Metrics>,
    pub(super) metrics_per_output: Option<Vec<Metrics>>,
    pub(super) validation_origin: Option<ValidationOrigin>,
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
    /// Отчёты конвейера интерпретации по фазам.
    pub(super) interpret_reports: Option<Box<InterpretReports>>,
    /// Запускать ли конвейер интерпретации и с какими переопределениями.
    pub(super) interpret_enabled: bool,
    pub(super) interpret_overrides: InterpretOverrides,
    /// Единственный замер на test — только у финального обучения.
    pub(super) final_eval: Option<FinalEval>,
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
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
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
            metrics: None,
            metrics_per_output: None,
            validation_origin: None,
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
            interpret_reports: None,
            interpret_enabled: false,
            interpret_overrides: InterpretOverrides::default(),
            final_eval: None,
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
                    epoch,
                    loss,
                    val_r2,
                } => {
                    let (curve, label) = match phase {
                        Phase::Development => (&mut self.loss_curve, "development"),
                        Phase::Final => (&mut self.final_loss_curve, "финальное обучение"),
                    };
                    curve.push([epoch as f64, loss as f64]);
                    // Кривая validation снимается только в точках расписания и
                    // только в фазе разработки: финальная фаза validation не
                    // измеряет — она на ней училась.
                    if let (Phase::Development, Some(r2)) = (phase, val_r2) {
                        self.val_curve.push([epoch as f64, r2 as f64]);
                    }
                    self.status = match val_r2 {
                        Some(r2) => {
                            format!("{label}: эпоха {epoch}, loss {loss:.5}, validation R² {r2:.5}")
                        }
                        None => format!("{label}: эпоха {epoch}, loss {loss:.5}"),
                    };
                }
                Event::TrainDone {
                    metrics,
                    per_output,
                    validation_origin,
                    final_eval,
                    interpret,
                    cancelled,
                } => {
                    self.training = false;
                    // До успешного завершения активной остаётся прежняя
                    // модель worker-а, поэтому отмена не должна стирать её
                    // метрики из шапки.
                    if !cancelled {
                        self.metrics = metrics;
                        self.metrics_per_output = per_output;
                        self.validation_origin = validation_origin;
                        self.final_eval = final_eval;
                        self.interpret_reports = interpret;
                    }
                    self.status = if cancelled {
                        "обучение отменено".to_string()
                    } else if self.final_eval.is_some() {
                        "финальное обучение завершено, test открыт".to_string()
                    } else {
                        "обучение завершено".to_string()
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
                    parameter_count,
                    kan,
                    keep_evaluation,
                } => {
                    let n_inputs = schema.n_inputs();
                    if !keep_evaluation {
                        // Checkpoint не хранит протокол оценки. Оставить здесь
                        // числа предыдущей модели означало бы приписать их
                        // только что загруженной.
                        self.metrics = None;
                        self.metrics_per_output = None;
                        self.validation_origin = None;
                        self.final_eval = None;
                        self.loss_curve.clear();
                        self.val_curve.clear();
                        self.final_loss_curve.clear();
                        self.train_parameter_count = None;
                    }
                    self.model_info = Some(ModelInfo {
                        schema,
                        kind,
                        source,
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
                    ui.label(format!("{} · {}", model_kind_label(info.kind), info.source));
                    if let Some(final_eval) = &self.final_eval {
                        ui.label(format!("· test R² {:.3}", final_eval.metrics.r2));
                    } else if let Some(metrics) = &self.metrics {
                        ui.label(format!("· validation R² {:.3}", metrics.r2));
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
