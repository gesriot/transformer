//! egui-приложение (PlanUI §3). UI-M4: Train-панель с конфигом, живой кривой
//! loss и кооперативной отменой. Прочие вкладки — заглушки (M5+).
//! UI только рендерит и общается с worker каналами; ML-состояние — в worker.

use super::messages::{Command, DataSource, DiagnosticsResult, Event};
use super::worker::Worker;
use crate::config::ModelConfig;
use crate::data::OutOfRange;
use crate::encoders::{ValueEncoderConfig, ValueEncoderKind};
use crate::epoch_sweep::{self, EpochRow};
use crate::metrics::Metrics;
use crate::numeric_model::{validate_numeric, ModelKind, NumericConfig};
use crate::sweep::{self, SweepAxes, SweepRow};
use crate::tnum::{parse_categorical, Delimiter, PrepareSpec};
use crate::train::{validate_train, LrSchedule, TextTrainConfig, TrainConfig};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Train,
    Predict,
    Diagnose,
    Sweep,
    Text,
    Prepare,
    EpochSweep,
}

const BLACKBOXES: &[&str] = &["sum", "product", "sine", "polynomial", "projectile"];

/// Состояние формы Train. Числа редактируются `DragValue` (без строкового
/// парсинга); валидность проверяется теми же `validate_*`, что и в CLI.
struct TrainForm {
    use_file: bool,
    blackbox: String,
    file_path: String,
    mlp: bool,
    d_model: usize,
    heads: usize,
    layers: usize,
    d_ff: usize,
    venc: usize, // 0 linear, 1 mlp, 2 fourier
    fourier_bands: usize,
    fourier_scale: f32,
    mlp_width: usize,
    mlp_layers: usize,
    lr: f32,
    batch: usize,
    epochs: usize,
    seed: u64,
    warmup_cosine: bool,
    warmup: f32,
    min_lr_ratio: f32,
}

impl Default for TrainForm {
    fn default() -> Self {
        Self {
            use_file: false,
            blackbox: "sum".to_string(),
            file_path: String::new(),
            mlp: false,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 64,
            venc: 0,
            fourier_bands: 6,
            fourier_scale: 8.0,
            mlp_width: 128,
            mlp_layers: 3,
            lr: 1e-3,
            batch: 64,
            epochs: 40,
            seed: 0,
            warmup_cosine: false,
            warmup: 0.1,
            min_lr_ratio: 0.1,
        }
    }
}

impl TrainForm {
    fn build(&self) -> Result<(DataSource, NumericConfig, TrainConfig), String> {
        let source = if self.use_file {
            if self.file_path.is_empty() {
                return Err("укажите .tnum файл".to_string());
            }
            DataSource::File(self.file_path.clone())
        } else {
            DataSource::Blackbox(self.blackbox.clone())
        };

        let value = ValueEncoderConfig {
            kind: match self.venc {
                0 => ValueEncoderKind::Linear,
                1 => ValueEncoderKind::Mlp,
                _ => ValueEncoderKind::Fourier,
            },
            fourier_bands: self.fourier_bands,
            fourier_scale: self.fourier_scale,
        };
        let nc = NumericConfig {
            kind: if self.mlp {
                ModelKind::Mlp
            } else {
                ModelKind::Transformer
            },
            transformer: ModelConfig {
                d_model: self.d_model,
                n_heads: self.heads,
                n_enc_layers: self.layers,
                n_dec_layers: self.layers,
                d_ff: self.d_ff,
                ln_eps: 1e-5,
            },
            value,
            mlp_width: self.mlp_width,
            mlp_layers: self.mlp_layers,
        };
        validate_numeric(&nc)?;

        let schedule = if self.warmup_cosine {
            if !(0.0..1.0).contains(&self.warmup) {
                return Err("warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&self.min_lr_ratio) {
                return Err("min-lr-ratio должен быть в [0, 1]".to_string());
            }
            LrSchedule::WarmupCosine {
                warmup_frac: self.warmup,
                min_lr_ratio: self.min_lr_ratio,
            }
        } else {
            LrSchedule::Constant
        };
        let tcfg = TrainConfig {
            epochs: self.epochs,
            batch_size: self.batch,
            lr: self.lr,
            seed: self.seed,
            schedule,
        };
        validate_train(tcfg.lr, tcfg.batch_size)?;
        Ok((source, nc, tcfg))
    }
}

struct SweepForm {
    blackbox: String,
    seeds: String,
    d_models: String,
    layers: String,
    d_ffs: String,
    lrs: String,
    value_encoders: String,
    fourier_scales: String,
    fourier_bands: usize,
    schedulers: String,
    epochs: usize,
    batch: usize,
}

impl Default for SweepForm {
    fn default() -> Self {
        Self {
            blackbox: "sum".to_string(),
            seeds: "0".to_string(),
            d_models: "32".to_string(),
            layers: "2".to_string(),
            d_ffs: "64".to_string(),
            lrs: "0.001".to_string(),
            value_encoders: "linear".to_string(),
            fourier_scales: "2".to_string(),
            fourier_bands: 6,
            schedulers: "constant".to_string(),
            epochs: 30,
            batch: 64,
        }
    }
}

impl SweepForm {
    fn build(&self) -> Result<(String, SweepAxes), String> {
        let axes = SweepAxes {
            seeds: parse_csv_u64(&self.seeds, "seeds")?,
            d_models: parse_csv_usize(&self.d_models, "d-models")?,
            layers: parse_csv_usize(&self.layers, "layers-list")?,
            d_ffs: parse_csv_usize(&self.d_ffs, "d-ffs")?,
            lrs: parse_csv_f32(&self.lrs, "lrs")?,
            value_encoders: parse_value_encoders(&self.value_encoders)?,
            fourier_scales: parse_csv_f32(&self.fourier_scales, "fourier-scales")?,
            fourier_bands: self.fourier_bands,
            schedules: parse_schedules(&self.schedulers)?,
            epochs: self.epochs,
            batch_size: self.batch,
        };
        sweep::validate_axes(&axes)?;
        Ok((self.blackbox.clone(), axes))
    }
}

struct TextForm {
    file_path: String,
    d_model: usize,
    heads: usize,
    layers: usize,
    d_ff: usize,
    steps: usize,
    batch: usize,
    ctx_len: usize,
    tgt_len: usize,
    lr: f32,
    seed: u64,
    seed_text: String,
    total_new: usize,
    temperature: f32,
    top_k: usize,
    gen_seed: u64,
}

impl Default for TextForm {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            d_model: 64,
            heads: 4,
            layers: 2,
            d_ff: 128,
            steps: 500,
            batch: 32,
            ctx_len: 32,
            tgt_len: 32,
            lr: 1e-3,
            seed: 0,
            seed_text: String::new(),
            total_new: 400,
            temperature: 0.8,
            top_k: 10,
            gen_seed: 42,
        }
    }
}

impl TextForm {
    fn build(&self) -> Result<(String, ModelConfig, TextTrainConfig), String> {
        if self.file_path.is_empty() {
            return Err("выберите текстовый файл".to_string());
        }
        let model_cfg = ModelConfig {
            d_model: self.d_model,
            n_heads: self.heads,
            n_enc_layers: self.layers,
            n_dec_layers: self.layers,
            d_ff: self.d_ff,
            ln_eps: 1e-5,
        };
        if model_cfg.d_model == 0
            || model_cfg.d_ff == 0
            || model_cfg.n_heads == 0
            || !model_cfg.d_model.is_multiple_of(model_cfg.n_heads)
            || model_cfg.n_enc_layers == 0
            || model_cfg.n_dec_layers == 0
        {
            return Err("некорректный text model config".to_string());
        }
        let train_cfg = TextTrainConfig {
            steps: self.steps,
            batch_size: self.batch,
            ctx_len: self.ctx_len,
            tgt_len: self.tgt_len,
            lr: self.lr,
            seed: self.seed,
        };
        if train_cfg.steps == 0
            || train_cfg.batch_size == 0
            || train_cfg.ctx_len == 0
            || train_cfg.tgt_len == 0
            || !train_cfg.lr.is_finite()
            || train_cfg.lr <= 0.0
        {
            return Err("некорректный text train config".to_string());
        }
        Ok((self.file_path.clone(), model_cfg, train_cfg))
    }
}

struct PrepareForm {
    input_path: String,
    output_path: String,
    inputs: usize,
    outputs: usize,
    delimiter: usize,
    has_header: bool,
    categorical: String,
}

impl Default for PrepareForm {
    fn default() -> Self {
        Self {
            input_path: String::new(),
            output_path: String::new(),
            inputs: 2,
            outputs: 1,
            delimiter: 0,
            has_header: false,
            categorical: String::new(),
        }
    }
}

impl PrepareForm {
    fn build(&self) -> Result<(String, String, PrepareSpec), String> {
        if self.input_path.is_empty() {
            return Err("выберите входную таблицу".to_string());
        }
        if self.output_path.is_empty() {
            return Err("укажите выходной .tnum".to_string());
        }
        let categorical = parse_categorical(&self.categorical, self.inputs)?;
        let delimiter = match self.delimiter {
            1 => Delimiter::Comma,
            2 => Delimiter::Tab,
            3 => Delimiter::Space,
            _ => Delimiter::Auto,
        };
        Ok((
            self.input_path.clone(),
            self.output_path.clone(),
            PrepareSpec {
                n_inputs: self.inputs,
                n_outputs: self.outputs,
                delimiter,
                has_header: self.has_header,
                categorical,
            },
        ))
    }
}

struct EpochSweepForm {
    file_path: String,
    epochs: String,
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
    mlp: bool,
    d_model: usize,
    heads: usize,
    layers: usize,
    d_ff: usize,
    venc: usize,
    fourier_bands: usize,
    fourier_scale: f32,
    mlp_width: usize,
    mlp_layers: usize,
    lr: f32,
    batch: usize,
    seed: u64,
    warmup_cosine: bool,
    warmup: f32,
    min_lr_ratio: f32,
}

struct EpochSweepRequest {
    path: String,
    nc: NumericConfig,
    base_tcfg: TrainConfig,
    milestones: Vec<usize>,
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
}

impl Default for EpochSweepForm {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            epochs: "1,2,5,10,20,40".to_string(),
            target_r2: 0.95,
            min_gain: 0.02,
            plateau_min: 0.80,
            mlp: false,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 64,
            venc: 0,
            fourier_bands: 6,
            fourier_scale: 8.0,
            mlp_width: 128,
            mlp_layers: 3,
            lr: 1e-3,
            batch: 64,
            seed: 0,
            warmup_cosine: false,
            warmup: 0.1,
            min_lr_ratio: 0.1,
        }
    }
}

impl EpochSweepForm {
    fn build(&self) -> Result<EpochSweepRequest, String> {
        if self.file_path.is_empty() {
            return Err("выберите .tnum файл".to_string());
        }
        let milestones = parse_csv_usize(&self.epochs, "epochs")?;
        if milestones.is_empty() || milestones.contains(&0) {
            return Err("epochs: список должен быть непустым и > 0".to_string());
        }
        if !self.target_r2.is_finite() {
            return Err("target-r2 должен быть конечным".to_string());
        }
        if !self.min_gain.is_finite() || self.min_gain < 0.0 {
            return Err("min-r2-gain должен быть конечным и >= 0".to_string());
        }
        if !self.plateau_min.is_finite() {
            return Err("plateau-min-r2 должен быть конечным".to_string());
        }

        let value = ValueEncoderConfig {
            kind: match self.venc {
                0 => ValueEncoderKind::Linear,
                1 => ValueEncoderKind::Mlp,
                _ => ValueEncoderKind::Fourier,
            },
            fourier_bands: self.fourier_bands,
            fourier_scale: self.fourier_scale,
        };
        let nc = NumericConfig {
            kind: if self.mlp {
                ModelKind::Mlp
            } else {
                ModelKind::Transformer
            },
            transformer: ModelConfig {
                d_model: self.d_model,
                n_heads: self.heads,
                n_enc_layers: self.layers,
                n_dec_layers: self.layers,
                d_ff: self.d_ff,
                ln_eps: 1e-5,
            },
            value,
            mlp_width: self.mlp_width,
            mlp_layers: self.mlp_layers,
        };
        validate_numeric(&nc)?;

        let schedule = if self.warmup_cosine {
            if !(0.0..1.0).contains(&self.warmup) {
                return Err("warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&self.min_lr_ratio) {
                return Err("min-lr-ratio должен быть в [0, 1]".to_string());
            }
            LrSchedule::WarmupCosine {
                warmup_frac: self.warmup,
                min_lr_ratio: self.min_lr_ratio,
            }
        } else {
            LrSchedule::Constant
        };
        let max_epoch = milestones.iter().copied().max().unwrap_or(1);
        let tcfg = TrainConfig {
            epochs: max_epoch,
            batch_size: self.batch,
            lr: self.lr,
            seed: self.seed,
            schedule,
        };
        validate_train(tcfg.lr, tcfg.batch_size)?;
        Ok(EpochSweepRequest {
            path: self.file_path.clone(),
            nc,
            base_tcfg: tcfg,
            milestones,
            target_r2: self.target_r2,
            min_gain: self.min_gain,
            plateau_min: self.plateau_min,
        })
    }
}

fn split_csv(raw: &str, name: &str) -> Result<Vec<String>, String> {
    let parts: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if parts.is_empty() {
        return Err(format!("{name}: пустой список"));
    }
    Ok(parts)
}

fn parse_csv_usize(raw: &str, name: &str) -> Result<Vec<usize>, String> {
    split_csv(raw, name)?
        .into_iter()
        .map(|p| {
            p.parse()
                .map_err(|_| format!("{name}: '{p}' не целое число"))
        })
        .collect()
}

fn parse_csv_u64(raw: &str, name: &str) -> Result<Vec<u64>, String> {
    split_csv(raw, name)?
        .into_iter()
        .map(|p| {
            p.parse()
                .map_err(|_| format!("{name}: '{p}' не целое число"))
        })
        .collect()
}

fn parse_csv_f32(raw: &str, name: &str) -> Result<Vec<f32>, String> {
    split_csv(raw, name)?
        .into_iter()
        .map(|p| p.parse().map_err(|_| format!("{name}: '{p}' не число")))
        .collect()
}

fn parse_value_encoders(raw: &str) -> Result<Vec<ValueEncoderKind>, String> {
    split_csv(raw, "value-encoders")?
        .into_iter()
        .map(|p| match p.as_str() {
            "linear" => Ok(ValueEncoderKind::Linear),
            "mlp" => Ok(ValueEncoderKind::Mlp),
            "fourier" => Ok(ValueEncoderKind::Fourier),
            _ => Err(format!("value-encoders: '{p}' не linear|mlp|fourier")),
        })
        .collect()
}

fn parse_schedules(raw: &str) -> Result<Vec<LrSchedule>, String> {
    split_csv(raw, "schedulers")?
        .into_iter()
        .map(|p| match p.as_str() {
            "constant" => Ok(LrSchedule::Constant),
            "warmup-cosine" => Ok(LrSchedule::WarmupCosine {
                warmup_frac: 0.1,
                min_lr_ratio: 0.1,
            }),
            _ => Err(format!("schedulers: '{p}' не constant|warmup-cosine")),
        })
        .collect()
}

pub struct App {
    worker: Worker,
    tab: Tab,
    status: String,
    form: TrainForm,
    training: bool,
    sweeping: bool,
    loss_curve: Vec<[f64; 2]>,
    metrics: Option<Metrics>,
    // Predict (UI-M5)
    model_info: Option<(usize, usize, String)>, // n_inputs, n_outputs, source
    predict_inputs: Vec<f32>,
    predict_outputs: Option<Vec<f32>>,
    extrapolation: Vec<OutOfRange>,
    // Diagnose (UI-M6)
    diagnostics: Option<DiagnosticsResult>,
    // Sweep (UI-M6)
    sweep_form: SweepForm,
    sweep_rows: Vec<SweepRow>,
    sweep_total: Option<(usize, usize)>,
    sweep_cancelled: bool,
    // Text (UI-M7)
    text_form: TextForm,
    text_training: bool,
    text_curve: Vec<[f64; 2]>,
    text_ready: bool,
    text_vocab_size: Option<usize>,
    generated_text: String,
    // Prepare / Epoch-sweep (UI-M8)
    prepare_form: PrepareForm,
    epoch_form: EpochSweepForm,
    epoch_sweeping: bool,
    epoch_rows: Vec<EpochRow>,
    epoch_total: Option<usize>,
    epoch_recommendation: Option<(usize, String)>,
    epoch_cancelled: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            worker: Worker::spawn(cc.egui_ctx.clone()),
            tab: Tab::Train,
            status: "—".to_string(),
            form: TrainForm::default(),
            training: false,
            sweeping: false,
            loss_curve: Vec::new(),
            metrics: None,
            model_info: None,
            predict_inputs: Vec::new(),
            predict_outputs: None,
            extrapolation: Vec::new(),
            diagnostics: None,
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

    fn drain_events(&mut self) {
        while let Some(ev) = self.worker.try_recv() {
            match ev {
                Event::Status(s) => self.status = s,
                Event::Error(e) => {
                    self.training = false;
                    self.sweeping = false;
                    self.text_training = false;
                    self.epoch_sweeping = false;
                    self.status = format!("Ошибка: {e}");
                }
                Event::TrainStarted { total_epochs } => {
                    self.training = true;
                    self.loss_curve.clear();
                    self.metrics = None;
                    self.status = format!("обучение: 0/{total_epochs} эпох");
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
                Event::ModelReady {
                    n_inputs,
                    n_outputs,
                    source,
                } => {
                    self.model_info = Some((n_inputs, n_outputs, source));
                    self.predict_inputs = vec![0.0; n_inputs];
                    self.predict_outputs = None;
                    self.extrapolation.clear();
                    self.status = "модель готова к предсказанию".to_string();
                }
                Event::PredictResult {
                    outputs,
                    extrapolation,
                } => {
                    self.predict_outputs = Some(outputs);
                    self.extrapolation = extrapolation;
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
                    self.sweep_rows.sort_by(|a, b| {
                        b.r2_mean
                            .partial_cmp(&a.r2_mean)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
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

    fn busy(&self) -> bool {
        self.training || self.sweeping || self.text_training || self.epoch_sweeping
    }

    fn save_model_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("bin", &["bin"])
            .save_file()
        {
            self.worker
                .send(Command::SaveModel(p.display().to_string()));
            self.status = "сохранение модели…".to_string();
        }
    }

    fn ui_train(&mut self, ui: &mut egui::Ui) {
        ui.heading("Train (numeric)");

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.form.use_file, false, "Чёрный ящик");
            ui.selectable_value(&mut self.form.use_file, true, ".tnum файл");
        });
        if self.form.use_file {
            ui.horizontal(|ui| {
                if ui.button("Выбрать .tnum…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("tnum", &["tnum"])
                        .pick_file()
                    {
                        self.form.file_path = p.display().to_string();
                    }
                }
                ui.label(if self.form.file_path.is_empty() {
                    "(файл не выбран)"
                } else {
                    &self.form.file_path
                });
            });
        } else {
            egui::ComboBox::from_label("чёрный ящик")
                .selected_text(&self.form.blackbox)
                .show_ui(ui, |ui| {
                    for &name in BLACKBOXES {
                        ui.selectable_value(&mut self.form.blackbox, name.to_string(), name);
                    }
                });
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Модель:");
            ui.selectable_value(&mut self.form.mlp, false, "transformer");
            ui.selectable_value(&mut self.form.mlp, true, "mlp");
        });

        egui::Grid::new("cfg_grid").num_columns(2).show(ui, |ui| {
            if self.form.mlp {
                ui.label("mlp width");
                ui.add(egui::DragValue::new(&mut self.form.mlp_width).range(1..=2048));
                ui.end_row();
                ui.label("mlp layers");
                ui.add(egui::DragValue::new(&mut self.form.mlp_layers).range(1..=12));
                ui.end_row();
            } else {
                ui.label("d_model");
                ui.add(egui::DragValue::new(&mut self.form.d_model).range(1..=1024));
                ui.end_row();
                ui.label("heads");
                ui.add(egui::DragValue::new(&mut self.form.heads).range(1..=32));
                ui.end_row();
                ui.label("layers");
                ui.add(egui::DragValue::new(&mut self.form.layers).range(1..=12));
                ui.end_row();
                ui.label("d_ff");
                ui.add(egui::DragValue::new(&mut self.form.d_ff).range(1..=4096));
                ui.end_row();
                ui.label("value-encoder");
                egui::ComboBox::from_id_salt("venc")
                    .selected_text(["linear", "mlp", "fourier"][self.form.venc])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.form.venc, 0, "linear");
                        ui.selectable_value(&mut self.form.venc, 1, "mlp");
                        ui.selectable_value(&mut self.form.venc, 2, "fourier");
                    });
                ui.end_row();
                if self.form.venc == 2 {
                    ui.label("fourier bands");
                    ui.add(egui::DragValue::new(&mut self.form.fourier_bands).range(1..=64));
                    ui.end_row();
                    ui.label("fourier scale");
                    ui.add(
                        egui::DragValue::new(&mut self.form.fourier_scale)
                            .range(0.1..=128.0)
                            .speed(0.1),
                    );
                    ui.end_row();
                }
            }
            ui.label("lr");
            ui.add(
                egui::DragValue::new(&mut self.form.lr)
                    .range(1e-6..=1.0)
                    .speed(1e-4),
            );
            ui.end_row();
            ui.label("batch");
            ui.add(egui::DragValue::new(&mut self.form.batch).range(1..=8192));
            ui.end_row();
            ui.label("epochs");
            ui.add(egui::DragValue::new(&mut self.form.epochs).range(1..=100000));
            ui.end_row();
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.form.seed));
            ui.end_row();
        });

        ui.checkbox(&mut self.form.warmup_cosine, "scheduler: warmup-cosine");
        if self.form.warmup_cosine {
            ui.horizontal(|ui| {
                ui.label("warmup");
                ui.add(
                    egui::DragValue::new(&mut self.form.warmup)
                        .range(0.0..=0.99)
                        .speed(0.01),
                );
                ui.label("min-lr-ratio");
                ui.add(
                    egui::DragValue::new(&mut self.form.min_lr_ratio)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
            });
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Train"))
                .clicked()
            {
                match self.form.build() {
                    Ok((source, nc, tcfg)) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::TrainNumeric { source, nc, tcfg });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.training, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена…".to_string();
            }
            if ui
                .add_enabled(
                    self.model_info.is_some() && !self.busy(),
                    egui::Button::new("Save model…"),
                )
                .clicked()
            {
                self.save_model_dialog();
            }
        });

        if !self.loss_curve.is_empty() {
            let points = PlotPoints::from(self.loss_curve.clone());
            Plot::new("loss_plot")
                .height(220.0)
                .show(ui, |pui| pui.line(Line::new(points).name("train loss")));
        }
        if let Some(m) = &self.metrics {
            ui.separator();
            ui.label(format!(
                "RMSE={:.5}   MAE={:.5}   rel.error={:.2}%   R²={:.5}",
                m.rmse,
                m.mae,
                m.rel_error * 100.0,
                m.r2
            ));
        }
    }

    fn ui_predict(&mut self, ui: &mut egui::Ui) {
        ui.heading("Predict");
        if ui.button("Загрузить модель (.bin)…").clicked() {
            if let Some(p) = rfd::FileDialog::new()
                .add_filter("bin", &["bin"])
                .pick_file()
            {
                self.worker
                    .send(Command::LoadModel(p.display().to_string()));
            }
        }

        let info = self.model_info.clone();
        match info {
            None => {
                ui.label("Обучите модель (вкладка Train) или загрузите .bin.");
            }
            Some((n_in, n_out, source)) => {
                ui.label(format!("Модель: {source} ({n_in} вход → {n_out} выход)"));
                ui.separator();
                egui::Grid::new("predict_inputs")
                    .num_columns(2)
                    .show(ui, |ui| {
                        for (i, v) in self.predict_inputs.iter_mut().enumerate() {
                            ui.label(format!("x{i}"));
                            ui.add(egui::DragValue::new(v).speed(0.05));
                            ui.end_row();
                        }
                    });
                if ui.button("Predict").clicked() {
                    self.worker
                        .send(Command::Predict(self.predict_inputs.clone()));
                }

                if let Some(out) = &self.predict_outputs {
                    ui.separator();
                    for (i, v) in out.iter().enumerate() {
                        ui.label(format!("y{i} = {v:.6}"));
                    }
                }
                if !self.extrapolation.is_empty() {
                    ui.separator();
                    let warn = egui::Color32::from_rgb(200, 120, 0);
                    ui.colored_label(warn, "⚠ экстраполяция — модель ненадёжна вне диапазона:");
                    for e in &self.extrapolation {
                        ui.colored_label(
                            warn,
                            format!(
                                "признак {} = {} вне [{}, {}]",
                                e.feature, e.value, e.min, e.max
                            ),
                        );
                    }
                }
            }
        }
    }

    fn ui_diagnose(&mut self, ui: &mut egui::Ui) {
        ui.heading("Diagnose");
        if self.model_info.is_none() {
            ui.label("Обучите модель (вкладка Train) — диагностика по её данным.");
            return;
        }
        if ui.button("Запустить диагностику").clicked() {
            self.diagnostics = None;
            self.status = "диагностика…".to_string();
            self.worker.send(Command::Diagnose);
        }
        if let Some(d) = &self.diagnostics {
            ui.separator();
            ui.label(format!(
                "Overfit-проба: норм. train MSE = {:.5}",
                d.overfit_loss
            ));
            ui.label(if d.overfit_loss < 0.02 {
                "  → ёмкости хватает (проблема в данных/обобщении)"
            } else {
                "  → underfit: ёмкость или кодирование значений (value encoder / Fourier)"
            });
            ui.label(format!(
                "Экстраполяция: {} из {} test-строк вне обученного диапазона",
                d.extrapolation_rows, d.extrapolation_total
            ));

            ui.separator();
            ui.label("Остаток по входным признакам:");
            egui::Grid::new("resid_grid").num_columns(3).show(ui, |ui| {
                ui.label("признак");
                ui.label("смена знака");
                ui.label("tail/inner");
                ui.end_row();
                for (i, (sc, tr)) in d.residuals.iter().enumerate() {
                    ui.label(format!("{i}"));
                    ui.label(format!("{:.0}%", sc * 100.0));
                    ui.label(format!("{tr:.2}"));
                    ui.end_row();
                }
            });
            ui.label("(высокая смена знака → частота/Fourier; tail/inner>1.5 → масштаб/хвосты)");

            ui.separator();
            match d.sensitivity {
                Some((mean, max)) => {
                    ui.label(format!(
                        "Чувствительность ‖Δy‖/‖Δx‖ (норм.): среднее {mean:.2}, макс {max:.2}"
                    ));
                    ui.label(if max < 10.0 {
                        "  → карта гладкая, surrogate надёжен"
                    } else {
                        "  → высокая чувствительность/хаос: потолок точности"
                    });
                }
                None => {
                    ui.label("Чувствительность: только для blackbox (для .tnum пропущена)");
                }
            }
        }
    }

    fn ui_sweep(&mut self, ui: &mut egui::Ui) {
        ui.heading("Sweep");

        egui::ComboBox::from_label("чёрный ящик")
            .selected_text(&self.sweep_form.blackbox)
            .show_ui(ui, |ui| {
                for &name in BLACKBOXES {
                    ui.selectable_value(&mut self.sweep_form.blackbox, name.to_string(), name);
                }
            });

        ui.separator();
        egui::Grid::new("sweep_axes")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("seeds");
                ui.text_edit_singleline(&mut self.sweep_form.seeds);
                ui.end_row();
                ui.label("d-models");
                ui.text_edit_singleline(&mut self.sweep_form.d_models);
                ui.end_row();
                ui.label("layers-list");
                ui.text_edit_singleline(&mut self.sweep_form.layers);
                ui.end_row();
                ui.label("d-ffs");
                ui.text_edit_singleline(&mut self.sweep_form.d_ffs);
                ui.end_row();
                ui.label("lrs");
                ui.text_edit_singleline(&mut self.sweep_form.lrs);
                ui.end_row();
                ui.label("value-encoders");
                ui.text_edit_singleline(&mut self.sweep_form.value_encoders);
                ui.end_row();
                ui.label("fourier-scales");
                ui.text_edit_singleline(&mut self.sweep_form.fourier_scales);
                ui.end_row();
                ui.label("fourier-bands");
                ui.add(egui::DragValue::new(&mut self.sweep_form.fourier_bands).range(1..=64));
                ui.end_row();
                ui.label("schedulers");
                ui.text_edit_singleline(&mut self.sweep_form.schedulers);
                ui.end_row();
                ui.label("epochs");
                ui.add(egui::DragValue::new(&mut self.sweep_form.epochs).range(1..=100000));
                ui.end_row();
                ui.label("batch-size");
                ui.add(egui::DragValue::new(&mut self.sweep_form.batch).range(1..=8192));
                ui.end_row();
            });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    !self.training && !self.sweeping,
                    egui::Button::new("Run sweep"),
                )
                .clicked()
            {
                match self.sweep_form.build() {
                    Ok((blackbox, axes)) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::Sweep { blackbox, axes });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.sweeping, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена sweep…".to_string();
            }
        });

        if let Some((cfgs, runs)) = self.sweep_total {
            ui.label(format!(
                "Конфигов: {cfgs}; прогонов: {runs}; готово: {}",
                self.sweep_rows.len()
            ));
        }
        if self.sweep_cancelled {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Sweep отменён; показаны завершённые конфиги.",
            );
        }

        if !self.sweep_rows.is_empty() {
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(320.0)
                .show(ui, |ui| {
                    egui::Grid::new("sweep_rows")
                        .num_columns(6)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("#");
                            ui.label("R²");
                            ui.label("std");
                            ui.label("nRMSE");
                            ui.label("rel");
                            ui.label("config");
                            ui.end_row();
                            for (i, row) in self.sweep_rows.iter().enumerate() {
                                let mark = if i == 0 { "*" } else { "" };
                                ui.label(format!("{mark}{}", i + 1));
                                ui.label(format!("{:.5}", row.r2_mean));
                                ui.label(format!("{:.5}", row.r2_std));
                                ui.label(format!("{:.5}", row.nrmse_mean));
                                ui.label(format!("{:.1}%", row.rel_mean * 100.0));
                                ui.label(&row.label);
                                ui.end_row();
                            }
                        });
                });
        }
    }

    fn ui_text(&mut self, ui: &mut egui::Ui) {
        ui.heading("Text");

        ui.horizontal(|ui| {
            if ui.button("Выбрать .txt…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("text", &["txt"])
                    .pick_file()
                {
                    self.text_form.file_path = p.display().to_string();
                }
            }
            ui.label(if self.text_form.file_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.text_form.file_path
            });
        });

        ui.separator();
        egui::Grid::new("text_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("d_model");
                ui.add(egui::DragValue::new(&mut self.text_form.d_model).range(1..=1024));
                ui.end_row();
                ui.label("heads");
                ui.add(egui::DragValue::new(&mut self.text_form.heads).range(1..=32));
                ui.end_row();
                ui.label("layers");
                ui.add(egui::DragValue::new(&mut self.text_form.layers).range(1..=12));
                ui.end_row();
                ui.label("d_ff");
                ui.add(egui::DragValue::new(&mut self.text_form.d_ff).range(1..=4096));
                ui.end_row();
                ui.label("steps");
                ui.add(egui::DragValue::new(&mut self.text_form.steps).range(1..=200000));
                ui.end_row();
                ui.label("batch");
                ui.add(egui::DragValue::new(&mut self.text_form.batch).range(1..=1024));
                ui.end_row();
                ui.label("ctx_len");
                ui.add(egui::DragValue::new(&mut self.text_form.ctx_len).range(1..=512));
                ui.end_row();
                ui.label("tgt_len");
                ui.add(egui::DragValue::new(&mut self.text_form.tgt_len).range(1..=512));
                ui.end_row();
                ui.label("lr");
                ui.add(
                    egui::DragValue::new(&mut self.text_form.lr)
                        .range(1e-6..=1.0)
                        .speed(1e-4),
                );
                ui.end_row();
                ui.label("seed");
                ui.add(egui::DragValue::new(&mut self.text_form.seed));
                ui.end_row();
            });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Train text"))
                .clicked()
            {
                match self.text_form.build() {
                    Ok((path, model_cfg, train_cfg)) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::TrainText {
                            path,
                            model_cfg,
                            train_cfg,
                        });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.text_training, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена text…".to_string();
            }
        });

        if !self.text_curve.is_empty() {
            let points = PlotPoints::from(self.text_curve.clone());
            Plot::new("text_ppl_plot")
                .height(220.0)
                .show(ui, |pui| pui.line(Line::new(points).name("perplexity")));
        }
        if let Some(vocab) = self.text_vocab_size {
            ui.label(format!("vocab: {vocab}"));
        }

        ui.separator();
        egui::Grid::new("text_gen")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("new chars");
                ui.add(egui::DragValue::new(&mut self.text_form.total_new).range(1..=5000));
                ui.end_row();
                ui.label("temperature");
                ui.add(
                    egui::DragValue::new(&mut self.text_form.temperature)
                        .range(0.0..=5.0)
                        .speed(0.05),
                );
                ui.end_row();
                ui.label("top_k");
                ui.add(egui::DragValue::new(&mut self.text_form.top_k).range(0..=512));
                ui.end_row();
                ui.label("rng seed");
                ui.add(egui::DragValue::new(&mut self.text_form.gen_seed));
                ui.end_row();
            });
        ui.label("seed text");
        ui.text_edit_multiline(&mut self.text_form.seed_text);
        if ui
            .add_enabled(
                self.text_ready && !self.text_training,
                egui::Button::new("Generate"),
            )
            .clicked()
        {
            self.worker.send(Command::GenerateText {
                seed: self.text_form.seed_text.clone(),
                total_new: self.text_form.total_new,
                temperature: self.text_form.temperature,
                top_k: self.text_form.top_k,
                rng_seed: self.text_form.gen_seed,
            });
        }

        if !self.generated_text.is_empty() {
            ui.separator();
            ui.label("generated");
            egui::ScrollArea::vertical()
                .max_height(240.0)
                .show(ui, |ui| {
                    ui.label(&self.generated_text);
                });
        }
    }
    fn ui_prepare(&mut self, ui: &mut egui::Ui) {
        ui.heading("Prepare");
        ui.horizontal(|ui| {
            if ui.button("Вход…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_file() {
                    self.prepare_form.input_path = p.display().to_string();
                }
            }
            ui.label(if self.prepare_form.input_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.prepare_form.input_path
            });
        });
        ui.horizontal(|ui| {
            if ui.button("Выход .tnum…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("tnum", &["tnum"])
                    .save_file()
                {
                    self.prepare_form.output_path = p.display().to_string();
                }
            }
            ui.label(if self.prepare_form.output_path.is_empty() {
                "(путь не выбран)"
            } else {
                &self.prepare_form.output_path
            });
        });
        egui::Grid::new("prepare_grid")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("inputs");
                ui.add(egui::DragValue::new(&mut self.prepare_form.inputs).range(1..=256));
                ui.end_row();
                ui.label("outputs");
                ui.add(egui::DragValue::new(&mut self.prepare_form.outputs).range(1..=256));
                ui.end_row();
                ui.label("delimiter");
                egui::ComboBox::from_id_salt("prepare_delim")
                    .selected_text(["auto", "comma", "tab", "space"][self.prepare_form.delimiter])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.prepare_form.delimiter, 0, "auto");
                        ui.selectable_value(&mut self.prepare_form.delimiter, 1, "comma");
                        ui.selectable_value(&mut self.prepare_form.delimiter, 2, "tab");
                        ui.selectable_value(&mut self.prepare_form.delimiter, 3, "space");
                    });
                ui.end_row();
                ui.label("categorical");
                ui.text_edit_singleline(&mut self.prepare_form.categorical);
                ui.end_row();
            });
        ui.checkbox(&mut self.prepare_form.has_header, "has header");
        if ui
            .add_enabled(!self.busy(), egui::Button::new("Convert"))
            .clicked()
        {
            match self.prepare_form.build() {
                Ok((input, output, spec)) => {
                    self.worker.send(Command::Prepare {
                        input,
                        output,
                        spec,
                    });
                }
                Err(e) => self.status = format!("Ошибка: {e}"),
            }
        }
    }

    fn ui_epoch_sweep(&mut self, ui: &mut egui::Ui) {
        ui.heading("Epoch-sweep");
        ui.horizontal(|ui| {
            if ui.button("Выбрать .tnum…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("tnum", &["tnum"])
                    .pick_file()
                {
                    self.epoch_form.file_path = p.display().to_string();
                }
            }
            ui.label(if self.epoch_form.file_path.is_empty() {
                "(файл не выбран)"
            } else {
                &self.epoch_form.file_path
            });
        });
        ui.horizontal(|ui| {
            ui.label("Модель:");
            ui.selectable_value(&mut self.epoch_form.mlp, false, "transformer");
            ui.selectable_value(&mut self.epoch_form.mlp, true, "mlp");
        });
        egui::Grid::new("epoch_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                if self.epoch_form.mlp {
                    ui.label("mlp width");
                    ui.add(egui::DragValue::new(&mut self.epoch_form.mlp_width).range(1..=2048));
                    ui.end_row();
                    ui.label("mlp layers");
                    ui.add(egui::DragValue::new(&mut self.epoch_form.mlp_layers).range(1..=12));
                    ui.end_row();
                } else {
                    ui.label("d_model");
                    ui.add(egui::DragValue::new(&mut self.epoch_form.d_model).range(1..=1024));
                    ui.end_row();
                    ui.label("heads");
                    ui.add(egui::DragValue::new(&mut self.epoch_form.heads).range(1..=32));
                    ui.end_row();
                    ui.label("layers");
                    ui.add(egui::DragValue::new(&mut self.epoch_form.layers).range(1..=12));
                    ui.end_row();
                    ui.label("d_ff");
                    ui.add(egui::DragValue::new(&mut self.epoch_form.d_ff).range(1..=4096));
                    ui.end_row();
                    ui.label("value-encoder");
                    egui::ComboBox::from_id_salt("epoch_venc")
                        .selected_text(["linear", "mlp", "fourier"][self.epoch_form.venc])
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.epoch_form.venc, 0, "linear");
                            ui.selectable_value(&mut self.epoch_form.venc, 1, "mlp");
                            ui.selectable_value(&mut self.epoch_form.venc, 2, "fourier");
                        });
                    ui.end_row();
                    if self.epoch_form.venc == 2 {
                        ui.label("fourier bands");
                        ui.add(
                            egui::DragValue::new(&mut self.epoch_form.fourier_bands).range(1..=64),
                        );
                        ui.end_row();
                        ui.label("fourier scale");
                        ui.add(
                            egui::DragValue::new(&mut self.epoch_form.fourier_scale)
                                .range(0.1..=128.0)
                                .speed(0.1),
                        );
                        ui.end_row();
                    }
                }
                ui.label("epochs");
                ui.text_edit_singleline(&mut self.epoch_form.epochs);
                ui.end_row();
                ui.label("lr");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.lr)
                        .range(1e-6..=1.0)
                        .speed(1e-4),
                );
                ui.end_row();
                ui.label("batch");
                ui.add(egui::DragValue::new(&mut self.epoch_form.batch).range(1..=8192));
                ui.end_row();
                ui.label("seed");
                ui.add(egui::DragValue::new(&mut self.epoch_form.seed));
                ui.end_row();
                ui.label("target-r2");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.target_r2)
                        .range(-10.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label("min-r2-gain");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.min_gain)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
                ui.label("plateau-min-r2");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.plateau_min)
                        .range(-10.0..=1.0)
                        .speed(0.01),
                );
                ui.end_row();
            });
        ui.checkbox(
            &mut self.epoch_form.warmup_cosine,
            "scheduler: warmup-cosine",
        );
        if self.epoch_form.warmup_cosine {
            ui.horizontal(|ui| {
                ui.label("warmup");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.warmup)
                        .range(0.0..=0.99)
                        .speed(0.01),
                );
                ui.label("min-lr-ratio");
                ui.add(
                    egui::DragValue::new(&mut self.epoch_form.min_lr_ratio)
                        .range(0.0..=1.0)
                        .speed(0.01),
                );
            });
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Run"))
                .clicked()
            {
                match self.epoch_form.build() {
                    Ok(req) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::EpochSweep {
                            path: req.path,
                            nc: req.nc,
                            base_tcfg: req.base_tcfg,
                            milestones: req.milestones,
                            target_r2: req.target_r2,
                            min_gain: req.min_gain,
                            plateau_min: req.plateau_min,
                        });
                    }
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.epoch_sweeping, egui::Button::new("Cancel"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена epoch-sweep…".to_string();
            }
            if ui
                .add_enabled(!self.epoch_rows.is_empty(), egui::Button::new("Save CSV"))
                .clicked()
            {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("csv", &["csv"])
                    .save_file()
                {
                    match std::fs::write(&p, epoch_sweep::rows_to_csv(&self.epoch_rows)) {
                        Ok(()) => self.status = format!("CSV: {}", p.display()),
                        Err(e) => self.status = format!("Ошибка: запись CSV: {e}"),
                    }
                }
            }
        });
        if let Some(total) = self.epoch_total {
            ui.label(format!("точек: {total}; готово: {}", self.epoch_rows.len()));
        }
        if let Some((epoch, why)) = &self.epoch_recommendation {
            ui.label(format!("Рекомендованная остановка: {epoch} эпох ({why})"));
        }
        if self.epoch_cancelled {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Epoch-sweep отменён; показаны завершённые точки.",
            );
        }
        if !self.epoch_rows.is_empty() {
            let r2_points = PlotPoints::from(
                self.epoch_rows
                    .iter()
                    .map(|r| [r.epochs as f64, r.r2 as f64])
                    .collect::<Vec<_>>(),
            );
            let loss_points = PlotPoints::from(
                self.epoch_rows
                    .iter()
                    .map(|r| [r.epochs as f64, r.train_loss as f64])
                    .collect::<Vec<_>>(),
            );
            Plot::new("epoch_sweep_plot")
                .height(260.0)
                .include_y(0.0)
                .include_y(1.0)
                .show(ui, |pui| {
                    pui.line(Line::new(r2_points).name("R²"));
                    pui.line(Line::new(loss_points).name("train loss"));
                });
            egui::ScrollArea::vertical()
                .max_height(260.0)
                .show(ui, |ui| {
                    egui::Grid::new("epoch_rows")
                        .num_columns(6)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("epochs");
                            ui.label("loss");
                            ui.label("RMSE");
                            ui.label("MAE");
                            ui.label("rel");
                            ui.label("R²");
                            ui.end_row();
                            for row in &self.epoch_rows {
                                ui.label(format!("{}", row.epochs));
                                ui.label(format!("{:.5}", row.train_loss));
                                ui.label(format!("{:.5}", row.rmse));
                                ui.label(format!("{:.5}", row.mae));
                                ui.label(format!("{:.1}%", row.rel_error * 100.0));
                                ui.label(format!("{:.5}", row.r2));
                                ui.end_row();
                            }
                        });
                });
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Train, "Train");
                ui.selectable_value(&mut self.tab, Tab::Predict, "Predict");
                ui.selectable_value(&mut self.tab, Tab::Diagnose, "Diagnose");
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
                    Tab::Diagnose => self.ui_diagnose(ui),
                    Tab::Sweep => self.ui_sweep(ui),
                    Tab::Text => self.ui_text(ui),
                    Tab::Prepare => self.ui_prepare(ui),
                    Tab::EpochSweep => self.ui_epoch_sweep(ui),
                });
        });
    }
}
