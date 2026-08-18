//! Состояние сессии GUI: активная модель, активные данные и их общий доступ.
//!
//! Экраны живут в соседних модулях и работают с этим состоянием: здесь только
//! то, что общее для всех — поля [`App`], разбор событий worker-а и переносы
//! конфигурации между экранами.

use super::data::{MarkupState, PrepareForm};
use super::demo::TextForm;
use super::messages::{
    Command, DatasetOrigin, DiagnosticsResult, Event, KanModelInfo, KanSymbolicInfo, PreparedData,
};
use super::model::ModelInfo;
use super::train::{EpochSweepForm, OptimizeForm, SweepForm, TrainForm};
use super::worker::Worker;
use crate::data::OutOfRange;
use crate::encoders::ValueEncoderKind;
use crate::epoch_sweep::EpochRow;
use crate::markup::TableProfile;
use crate::metrics::Metrics;
use crate::split::SplitPlan;
use crate::sweep::{self, SweepChoice, SweepObjective, SweepRow};
use crate::train::LrSchedule;
use eframe::egui;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Tab {
    Train,
    Predict,
    KanCurves,
    KanFormulas,
    Diagnose,
    Optimize,
    Sweep,
    Text,
    Prepare,
    EpochSweep,
}

/// Одно сообщение на все экраны: данные общие, значит и текст общий.
pub(super) const NO_DATASET: &str = "сначала откройте данные";

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
    /// План разбиения — свойство набора данных, а не отдельного запуска.
    pub(super) split: SplitPlan,
    /// Как читалась таблица: «Разметить заново» должно открыть её так же.
    pub(super) table_has_header: bool,
}

impl ActiveDataset {
    pub(super) fn new(
        prepared: PreparedData,
        profile: Option<TableProfile>,
        table_has_header: bool,
    ) -> Self {
        Self {
            prepared,
            profile,
            split: SplitPlan::default(),
            table_has_header,
        }
    }

    /// Строка для шапки: что открыто и какой оно формы.
    pub(super) fn summary(&self) -> String {
        let schema = &self.prepared.schema;
        format!(
            "{} · {} строк · {} вход → {} выход",
            self.prepared.origin.short_name(),
            self.prepared.data.len(),
            schema.n_inputs(),
            schema.n_outputs()
        )
    }

    /// Сколько замечаний к качеству данных нашёл профиль. У `.tnum` и чёрного
    /// ящика профиля нет — там и предупреждать не о чем.
    pub(super) fn data_notes(&self) -> usize {
        self.profile.as_ref().map_or(0, |p| p.messages().len())
    }
}

pub struct App {
    pub(super) worker: Worker,
    pub(super) tab: Tab,
    pub(super) status: String,
    pub(super) form: TrainForm,
    /// Активный набор данных сессии; `None` — данные ещё не открыты.
    pub(super) dataset: Option<ActiveDataset>,
    /// Идёт чтение набора данных: запускать что-либо на старых данных нельзя.
    pub(super) dataset_opening: bool,
    pub(super) training: bool,
    pub(super) sweeping: bool,
    pub(super) loss_curve: Vec<[f64; 2]>,
    pub(super) metrics: Option<Metrics>,
    pub(super) train_parameter_count: Option<usize>,
    // Predict (UI-M5)
    pub(super) model_info: Option<ModelInfo>,
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
    // Optimize (file-based sweep)
    pub(super) optimize_form: OptimizeForm,
    pub(super) optimizing: bool,
    pub(super) optimize_rows: Vec<SweepRow>,
    pub(super) optimize_total: Option<(usize, usize)>,
    pub(super) optimize_cancelled: bool,
    // Sweep (UI-M6)
    pub(super) sweep_form: SweepForm,
    pub(super) sweep_rows: Vec<SweepRow>,
    pub(super) sweep_total: Option<(usize, usize)>,
    pub(super) sweep_cancelled: bool,
    // Text (UI-M7)
    pub(super) text_form: TextForm,
    pub(super) text_training: bool,
    pub(super) text_curve: Vec<[f64; 2]>,
    pub(super) text_ready: bool,
    pub(super) text_vocab_size: Option<usize>,
    pub(super) generated_text: String,
    // Prepare / Epoch-sweep (UI-M8)
    pub(super) prepare_form: PrepareForm,
    pub(super) epoch_form: EpochSweepForm,
    pub(super) epoch_sweeping: bool,
    pub(super) epoch_rows: Vec<EpochRow>,
    pub(super) epoch_total: Option<usize>,
    pub(super) epoch_recommendation: Option<(usize, String)>,
    pub(super) epoch_cancelled: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            worker: Worker::spawn(cc.egui_ctx.clone()),
            tab: Tab::Train,
            status: "–".to_string(),
            form: TrainForm::default(),
            dataset: None,
            dataset_opening: false,
            training: false,
            sweeping: false,
            loss_curve: Vec::new(),
            metrics: None,
            train_parameter_count: None,
            model_info: None,
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
            optimize_form: OptimizeForm::default(),
            optimizing: false,
            optimize_rows: Vec::new(),
            optimize_total: None,
            optimize_cancelled: false,
            sweep_form: SweepForm::default(),
            sweep_rows: Vec::new(),
            sweep_total: None,
            sweep_cancelled: false,
            text_form: TextForm::default(),
            text_training: false,
            text_curve: Vec::new(),
            text_ready: false,
            text_vocab_size: None,
            generated_text: String::new(),
            prepare_form: PrepareForm::default(),
            epoch_form: EpochSweepForm::default(),
            epoch_sweeping: false,
            epoch_rows: Vec::new(),
            epoch_total: None,
            epoch_recommendation: None,
            epoch_cancelled: false,
        }
    }

    pub(super) fn drain_events(&mut self) {
        while let Some(ev) = self.worker.try_recv() {
            match ev {
                Event::Status(s) => self.status = s,
                Event::Error(e) => {
                    self.training = false;
                    self.sweeping = false;
                    self.optimizing = false;
                    self.text_training = false;
                    self.epoch_sweeping = false;
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
                    self.metrics = None;
                    self.train_parameter_count = Some(parameter_count);
                    self.status =
                        format!("обучение: 0/{total_epochs} эпох, {parameter_count} параметров");
                }
                Event::Epoch { epoch, loss } => {
                    self.loss_curve.push([epoch as f64, loss as f64]);
                    self.status = format!("эпоха {epoch}: loss {loss:.5}");
                }
                Event::TrainDone { metrics } => {
                    self.training = false;
                    match metrics {
                        Some(m) => {
                            self.metrics = Some(m);
                            self.status = "обучение завершено".to_string();
                        }
                        None => self.status = "обучение отменено".to_string(),
                    }
                }
                Event::DatasetOpened { data } => {
                    self.dataset_opening = false;
                    self.status = format!("данные открыты: {}", data.origin.short_name());
                    self.dataset = Some(ActiveDataset::new(data, None, true));
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
                } => {
                    let n_inputs = schema.n_inputs();
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
                Event::PredictFileDone {
                    output,
                    rows,
                    extrapolation_rows,
                } => {
                    self.batch_predicting = false;
                    self.status = if extrapolation_rows == 0 {
                        format!("Excel заполнен: {output} ({rows} строк)")
                    } else {
                        format!(
                            "Excel заполнен: {output} ({rows} строк, {extrapolation_rows} вне train-диапазона)"
                        )
                    };
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
                Event::SweepStarted {
                    total_configs,
                    total_runs,
                } => {
                    self.sweeping = true;
                    self.sweep_rows.clear();
                    self.sweep_total = Some((total_configs, total_runs));
                    self.sweep_cancelled = false;
                    self.status =
                        format!("sweep: 0/{total_configs} конфигов ({total_runs} прогонов)");
                }
                Event::SweepRow { row } => {
                    self.sweep_rows.push(row);
                    sweep::sort_rows(&mut self.sweep_rows, SweepObjective::default());
                    if let Some((total_configs, _)) = self.sweep_total {
                        self.status =
                            format!("sweep: {}/{total_configs} конфигов", self.sweep_rows.len());
                    }
                }
                Event::SweepDone { rows, cancelled } => {
                    self.sweeping = false;
                    self.sweep_rows = rows;
                    self.sweep_cancelled = cancelled;
                    self.status = if cancelled {
                        "sweep отменён".to_string()
                    } else {
                        "sweep завершён".to_string()
                    };
                }
                Event::OptimizeStarted {
                    total_configs,
                    total_runs,
                } => {
                    self.optimizing = true;
                    self.optimize_rows.clear();
                    self.optimize_total = Some((total_configs, total_runs));
                    self.optimize_cancelled = false;
                    self.status =
                        format!("optimize: 0/{total_configs} конфигов ({total_runs} прогонов)");
                }
                Event::OptimizeRow { row } => {
                    self.optimize_rows.push(row);
                    self.sort_optimize_rows();
                    if let Some((total_configs, _)) = self.optimize_total {
                        self.status = format!(
                            "optimize: {}/{total_configs} конфигов",
                            self.optimize_rows.len()
                        );
                    }
                }
                Event::OptimizeDone { rows, cancelled } => {
                    self.optimizing = false;
                    self.optimize_rows = rows;
                    self.sort_optimize_rows();
                    self.optimize_cancelled = cancelled;
                    self.status = if cancelled {
                        "optimize отменён".to_string()
                    } else {
                        "optimize завершён".to_string()
                    };
                }
                Event::TextStarted { total_steps } => {
                    self.text_training = true;
                    self.text_ready = false;
                    self.text_curve.clear();
                    self.generated_text.clear();
                    self.text_vocab_size = None;
                    self.status = format!("text: 0/{total_steps} шагов");
                }
                Event::TextProgress { step, loss } => {
                    self.text_curve.push([step as f64, loss.exp() as f64]);
                    self.status = format!("text шаг {step}: loss {loss:.4}, ppl {:.2}", loss.exp());
                }
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
                Event::EpochSweepStarted { total_points } => {
                    self.epoch_sweeping = true;
                    self.epoch_rows.clear();
                    self.epoch_total = Some(total_points);
                    self.epoch_recommendation = None;
                    self.epoch_cancelled = false;
                    self.status = format!("epoch-sweep: 0/{total_points}");
                }
                Event::EpochSweepRow { row } => {
                    self.epoch_rows.push(row);
                    if let Some(total) = self.epoch_total {
                        self.status = format!("epoch-sweep: {}/{total}", self.epoch_rows.len());
                    }
                }
                Event::EpochSweepDone {
                    rows,
                    recommendation,
                    cancelled,
                } => {
                    self.epoch_sweeping = false;
                    self.epoch_rows = rows;
                    self.epoch_recommendation = recommendation;
                    self.epoch_cancelled = cancelled;
                    self.status = if cancelled {
                        "epoch-sweep отменён".to_string()
                    } else {
                        "epoch-sweep завершён".to_string()
                    };
                }
            }
        }
    }

    pub(super) fn busy(&self) -> bool {
        self.training
            || self.sweeping
            || self.optimizing
            || self.text_training
            || self.epoch_sweeping
            || self.batch_predicting
            || self.kan_symbolic_pending
            || self.table_opening
            || self.dataset_opening
            || self.markup.is_some()
    }

    /// Полоса активного набора данных. Её рисует каждый экран обучения: пока
    /// разделы не объединены, это единственное место выбора данных.
    pub(super) fn ui_dataset_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Данные:");
            match &self.dataset {
                Some(active) => {
                    ui.label(active.summary());
                    let notes = active.data_notes();
                    if notes > 0 {
                        ui.label(format!("· замечаний к данным: {notes}"));
                    }
                }
                None => {
                    ui.label("не открыты");
                }
            }
        });
        ui.horizontal(|ui| {
            let idle = !self.busy();
            ui.add_enabled_ui(idle, |ui| {
                egui::ComboBox::from_id_salt("dataset_blackbox")
                    .selected_text("Чёрный ящик…")
                    .show_ui(ui, |ui| {
                        for &name in BLACKBOXES {
                            if ui.selectable_label(false, name).clicked() {
                                self.open_dataset(DatasetOrigin::Blackbox(name.to_string()));
                            }
                        }
                    });
            });
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
    /// `None` — данные не открыты: экран сам решает, что показать.
    pub(super) fn active_data(&self) -> Option<(PreparedData, SplitPlan)> {
        self.dataset
            .as_ref()
            .map(|active| (active.prepared.clone(), active.split))
    }

    pub(super) fn open_dataset(&mut self, origin: DatasetOrigin) {
        self.dataset_opening = true;
        self.status = format!("открываю {}…", origin.short_name());
        self.worker.send(Command::OpenDataset { origin });
    }

    pub(super) fn sort_optimize_rows(&mut self) {
        let objective = self.optimize_form.objective();
        sweep::sort_rows(&mut self.optimize_rows, objective);
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
        // Двухфазность: Optimize ранжировал на search-эпохах, а Train делает
        // финальное обучение на полном бюджете.
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
        self.tab = Tab::Train;
        self.status = "лучший конфиг применён во вкладке Train".to_string();
    }

    /// Переносит лучший конфиг Optimize в Epoch-sweep, чтобы подобрать число
    /// эпох. Конфиг тот же; эпохи Optimize не переносим — задаём список для
    /// прохода.
    pub(super) fn apply_choice_to_epoch_sweep(&mut self, choice: &SweepChoice) {
        self.epoch_form.kind = choice.kind;
        self.epoch_form.kan_width = choice.kan.width;
        self.epoch_form.kan_layers = choice.kan.layers;
        self.epoch_form.kan_grid = choice.kan.grid;
        self.epoch_form.d_model = choice.d_model;
        self.epoch_form.heads = choice.heads;
        self.epoch_form.layers = choice.layers;
        self.epoch_form.d_ff = choice.d_ff;
        self.epoch_form.venc = match choice.value.kind {
            ValueEncoderKind::Linear => 0,
            ValueEncoderKind::Mlp => 1,
            ValueEncoderKind::Fourier => 2,
        };
        self.epoch_form.fourier_bands = choice.value.fourier_bands;
        self.epoch_form.fourier_scale = choice.value.fourier_scale;
        self.epoch_form.mlp_width = choice.mlp_width;
        self.epoch_form.mlp_layers = choice.mlp_layers;
        self.epoch_form.lr = choice.lr;
        self.epoch_form.batch = choice.batch_size;
        self.epoch_form.seed = choice.seed;
        match choice.schedule {
            LrSchedule::Constant => {
                self.epoch_form.warmup_cosine = false;
            }
            LrSchedule::WarmupCosine {
                warmup_frac,
                min_lr_ratio,
            } => {
                self.epoch_form.warmup_cosine = true;
                self.epoch_form.warmup = warmup_frac;
                self.epoch_form.min_lr_ratio = min_lr_ratio;
            }
        }
        self.epoch_form.epochs = "20,40,60,80,120".to_string();
        self.epoch_rows.clear();
        self.epoch_total = None;
        self.epoch_recommendation = None;
        self.epoch_cancelled = false;
        self.tab = Tab::EpochSweep;
        self.status = "конфиг перенесён в Epoch-sweep – запусти подбор эпох".to_string();
    }

    /// Переносит текущий конфиг Epoch-sweep и рекомендованное число эпох в
    /// Train. Замыкает поток Optimize → Check epochs → Train.
    pub(super) fn apply_epoch_form_to_train(&mut self, epochs: usize) {
        let f = &self.epoch_form;
        self.form.kind = f.kind;
        self.form.d_model = f.d_model;
        self.form.heads = f.heads;
        self.form.layers = f.layers;
        self.form.d_ff = f.d_ff;
        self.form.venc = f.venc;
        self.form.fourier_bands = f.fourier_bands;
        self.form.fourier_scale = f.fourier_scale;
        self.form.mlp_width = f.mlp_width;
        self.form.mlp_layers = f.mlp_layers;
        self.form.kan_width = f.kan_width;
        self.form.kan_layers = f.kan_layers;
        self.form.kan_grid = f.kan_grid;
        self.form.lr = f.lr;
        self.form.batch = f.batch;
        self.form.seed = f.seed;
        self.form.warmup_cosine = f.warmup_cosine;
        self.form.warmup = f.warmup;
        self.form.min_lr_ratio = f.min_lr_ratio;
        self.form.epochs = epochs;
        self.tab = Tab::Train;
        self.status = format!("конфиг и {epochs} эпох применены во вкладке Train");
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.ui_markup(ctx);

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Train, "Train");
                ui.selectable_value(&mut self.tab, Tab::Predict, "Predict");
                ui.selectable_value(&mut self.tab, Tab::KanCurves, "KAN curves");
                ui.selectable_value(&mut self.tab, Tab::KanFormulas, "KAN formulas");
                ui.selectable_value(&mut self.tab, Tab::Diagnose, "Diagnose");
                ui.selectable_value(&mut self.tab, Tab::Optimize, "Optimize");
                ui.selectable_value(&mut self.tab, Tab::Sweep, "Sweep");
                ui.selectable_value(&mut self.tab, Tab::Text, "Text");
                ui.selectable_value(&mut self.tab, Tab::Prepare, "Prepare");
                ui.selectable_value(&mut self.tab, Tab::EpochSweep, "Epoch-sweep");
            });
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
                .show(ui, |ui| match self.tab {
                    Tab::Train => self.ui_train(ui),
                    Tab::Predict => self.ui_predict(ui),
                    Tab::KanCurves => self.ui_kan_curves(ui),
                    Tab::KanFormulas => self.ui_kan_formulas(ui),
                    Tab::Diagnose => self.ui_diagnose(ui),
                    Tab::Optimize => self.ui_optimize(ui),
                    Tab::Sweep => self.ui_sweep(ui),
                    Tab::Text => self.ui_text(ui),
                    Tab::Prepare => self.ui_prepare(ui),
                    Tab::EpochSweep => self.ui_epoch_sweep(ui),
                });
        });
    }
}
