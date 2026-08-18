//! Экран обучения: одна конфигурация, поиск по сетке и кривая по эпохам.

use super::messages::Command;
use super::model::ModelInfo;
use super::session::{App, NO_DATASET};
use crate::config::ModelConfig;
use crate::encoders::{ValueEncoderConfig, ValueEncoderKind};
use crate::epoch_sweep::{self};
use crate::numeric_model::{validate_numeric, KanConfig, ModelKind, NumericConfig};
use crate::split::DEFAULT_FINAL_INIT_SEED;
use crate::sweep::{self, SearchBudget, SweepAxes, SweepChoice, SweepObjective, SweepRow};
use crate::train::{validate_train, LrSchedule, TrainConfig};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

/// Что делает экран обучения.
///
/// Поиск и одиночное обучение — разные операции, но одно состояние экрана:
/// данные, разбиение и результат у них общие.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TrainingMode {
    /// Одна конфигурация: ручные гиперпараметры.
    Single,
    /// Готовый бюджет поиска.
    Auto(SearchBudget),
    /// Своя сетка из [`CustomSearchForm`].
    Custom,
}

impl TrainingMode {
    pub(super) fn label(self) -> &'static str {
        match self {
            TrainingMode::Single => "Одна конфигурация",
            TrainingMode::Auto(_) => "Авто",
            TrainingMode::Custom => "Своя сетка",
        }
    }
}

/// Состояние формы Train. Числа редактируются `DragValue` (без строкового
/// парсинга); валидность проверяется теми же `validate_*`, что и в CLI.
pub(super) struct TrainForm {
    pub(super) kind: ModelKind,
    pub(super) d_model: usize,
    pub(super) heads: usize,
    pub(super) layers: usize,
    pub(super) d_ff: usize,
    pub(super) venc: usize, // 0 linear, 1 mlp, 2 fourier
    pub(super) fourier_bands: usize,
    pub(super) fourier_scale: f32,
    pub(super) mlp_width: usize,
    pub(super) mlp_layers: usize,
    pub(super) kan_width: usize,
    pub(super) kan_layers: usize,
    pub(super) kan_grid: usize,
    pub(super) lr: f32,
    pub(super) batch: usize,
    pub(super) epochs: usize,
    pub(super) seed: u64,
    pub(super) warmup_cosine: bool,
    pub(super) warmup: f32,
    pub(super) min_lr_ratio: f32,
}

impl Default for TrainForm {
    fn default() -> Self {
        Self {
            kind: ModelKind::Transformer,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 64,
            venc: 0,
            fourier_bands: 6,
            fourier_scale: 8.0,
            mlp_width: 128,
            mlp_layers: 3,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
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
    pub(super) fn build(&self) -> Result<(NumericConfig, TrainConfig), String> {
        let value = ValueEncoderConfig {
            kind: match self.kind {
                ModelKind::Transformer => match self.venc {
                    0 => ValueEncoderKind::Linear,
                    1 => ValueEncoderKind::Mlp,
                    _ => ValueEncoderKind::Fourier,
                },
                ModelKind::Mlp | ModelKind::Kan => ValueEncoderKind::Linear,
            },
            fourier_bands: self.fourier_bands,
            fourier_scale: self.fourier_scale,
        };
        let nc = NumericConfig {
            kind: self.kind,
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
            kan: KanConfig {
                width: self.kan_width,
                layers: self.kan_layers,
                grid: self.kan_grid,
            },
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
        Ok((nc, tcfg))
    }
}

/// Редактируемая своя сетка. Хранится именно форма, а не готовые [`SweepAxes`]:
/// промежуточные значения бывают невалидными, и представлять их проще строками.
pub(super) struct CustomSearchForm {
    pub(super) seeds: String,
    pub(super) d_models: String,
    pub(super) layers: String,
    pub(super) d_ffs: String,
    pub(super) lrs: String,
    pub(super) value_encoders: String,
    pub(super) fourier_scales: String,
    pub(super) fourier_bands: usize,
    pub(super) schedulers: String,
    pub(super) epochs: usize,
    pub(super) batch: usize,
}

impl Default for CustomSearchForm {
    fn default() -> Self {
        Self {
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

impl CustomSearchForm {
    /// Собрать и проверить сетку. Невалидная форма не доходит до запуска.
    pub(super) fn build(&self, model_kinds: Vec<ModelKind>) -> Result<SweepAxes, String> {
        let axes = SweepAxes {
            model_kinds,
            seeds: parse_csv_u64(&self.seeds, "seeds")?,
            d_models: parse_csv_usize(&self.d_models, "d-models")?,
            layers: parse_csv_usize(&self.layers, "layers-list")?,
            d_ffs: parse_csv_usize(&self.d_ffs, "d-ffs")?,
            lrs: parse_csv_f32(&self.lrs, "lrs")?,
            value_encoders: parse_value_encoders(&self.value_encoders)?,
            fourier_scales: parse_csv_f32(&self.fourier_scales, "fourier-scales")?,
            fourier_bands: self.fourier_bands,
            mlp_widths: vec![128],
            mlp_layers: vec![3],
            kan_widths: vec![16],
            kan_layers: vec![2],
            kan_grids: vec![8],
            schedules: parse_schedules(&self.schedulers)?,
            epochs: self.epochs,
            final_epochs: self.epochs,
            batch_size: self.batch,
        };
        sweep::validate_axes(&axes)?;
        Ok(axes)
    }
}

pub(super) struct OptimizeForm {
    pub(super) objective: usize, // 0 worst, 1 aggregate, 2 mean, 3 nrmse
    pub(super) include_mlp: bool,
    pub(super) include_transformer: bool,
    pub(super) include_kan: bool,
}

impl Default for OptimizeForm {
    fn default() -> Self {
        Self {
            objective: 0,
            include_mlp: true,
            include_transformer: true,
            include_kan: true,
        }
    }
}

impl OptimizeForm {
    pub(super) fn objective(&self) -> SweepObjective {
        match self.objective {
            1 => SweepObjective::AggregateR2,
            2 => SweepObjective::MeanOutputR2,
            3 => SweepObjective::Nrmse,
            _ => SweepObjective::WorstOutputR2,
        }
    }

    /// Выбранные архитектуры. Пустой список — ошибка: искать не по чему.
    pub(super) fn model_kinds(&self) -> Result<Vec<ModelKind>, String> {
        let mut kinds = Vec::new();
        if self.include_transformer {
            kinds.push(ModelKind::Transformer);
        }
        if self.include_mlp {
            kinds.push(ModelKind::Mlp);
        }
        if self.include_kan {
            kinds.push(ModelKind::Kan);
        }
        if kinds.is_empty() {
            return Err("выберите хотя бы одну архитектуру (transformer/mlp/kan)".to_string());
        }
        Ok(kinds)
    }

    /// Оси сетки из выбранного бюджета — для оценки размера и запуска.
    pub(super) fn axes(&self, budget: SearchBudget) -> Result<SweepAxes, String> {
        let model_kinds = self.model_kinds()?;
        // Сетки бюджетов живут в ядре: раньше они были только здесь, и CLI не
        // мог запустить тот же поиск, что и кнопка в интерфейсе.
        let axes = SweepAxes::for_budget(budget, model_kinds);
        sweep::validate_axes(&axes)?;
        Ok(axes)
    }
}

pub(super) struct EpochSweepForm {
    pub(super) epochs: String,
    pub(super) target_r2: f32,
    pub(super) min_gain: f32,
    pub(super) plateau_min: f32,
    pub(super) kind: ModelKind,
    pub(super) d_model: usize,
    pub(super) heads: usize,
    pub(super) layers: usize,
    pub(super) d_ff: usize,
    pub(super) venc: usize,
    pub(super) fourier_bands: usize,
    pub(super) fourier_scale: f32,
    pub(super) mlp_width: usize,
    pub(super) mlp_layers: usize,
    pub(super) kan_width: usize,
    pub(super) kan_layers: usize,
    pub(super) kan_grid: usize,
    pub(super) lr: f32,
    pub(super) batch: usize,
    pub(super) seed: u64,
    pub(super) warmup_cosine: bool,
    pub(super) warmup: f32,
    pub(super) min_lr_ratio: f32,
}

pub(super) struct EpochSweepRequest {
    pub(super) nc: NumericConfig,
    pub(super) base_tcfg: TrainConfig,
    pub(super) milestones: Vec<usize>,
    pub(super) target_r2: f32,
    pub(super) min_gain: f32,
    pub(super) plateau_min: f32,
}

impl Default for EpochSweepForm {
    fn default() -> Self {
        Self {
            epochs: "1,2,5,10,20,40".to_string(),
            target_r2: 0.95,
            min_gain: 0.02,
            plateau_min: 0.80,
            kind: ModelKind::Transformer,
            d_model: 32,
            heads: 4,
            layers: 2,
            d_ff: 64,
            venc: 0,
            fourier_bands: 6,
            fourier_scale: 8.0,
            mlp_width: 128,
            mlp_layers: 3,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
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
    pub(super) fn build(&self) -> Result<EpochSweepRequest, String> {
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
            kind: match self.kind {
                ModelKind::Transformer => match self.venc {
                    0 => ValueEncoderKind::Linear,
                    1 => ValueEncoderKind::Mlp,
                    _ => ValueEncoderKind::Fourier,
                },
                ModelKind::Mlp | ModelKind::Kan => ValueEncoderKind::Linear,
            },
            fourier_bands: self.fourier_bands,
            fourier_scale: self.fourier_scale,
        };
        let nc = NumericConfig {
            kind: self.kind,
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
            kan: KanConfig {
                width: self.kan_width,
                layers: self.kan_layers,
                grid: self.kan_grid,
            },
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

fn objective_label(objective: SweepObjective) -> &'static str {
    match objective {
        SweepObjective::WorstOutputR2 => "worst-output R²",
        SweepObjective::AggregateR2 => "aggregate R²",
        SweepObjective::MeanOutputR2 => "mean-output R²",
        SweepObjective::Nrmse => "aggregate nRMSE",
    }
}

fn objective_display_score(objective: SweepObjective, row: &SweepRow) -> f32 {
    match objective {
        SweepObjective::Nrmse => row.nrmse_mean,
        _ => sweep::row_score(objective, row),
    }
}

impl App {
    pub(super) fn ui_train(&mut self, ui: &mut egui::Ui) {
        ui.heading("Обучение");

        self.ui_dataset_bar(ui);

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Режим:");
            ui.selectable_value(
                &mut self.mode,
                TrainingMode::Single,
                TrainingMode::Single.label(),
            );
            let auto = matches!(self.mode, TrainingMode::Auto(_));
            if ui
                .selectable_label(auto, TrainingMode::Auto(SearchBudget::default()).label())
                .clicked()
                && !auto
            {
                self.mode = TrainingMode::Auto(SearchBudget::default());
            }
            ui.selectable_value(
                &mut self.mode,
                TrainingMode::Custom,
                TrainingMode::Custom.label(),
            );
        });

        match self.mode {
            TrainingMode::Single => self.ui_single_config(ui),
            TrainingMode::Auto(budget) => self.ui_auto_search(ui, budget),
            TrainingMode::Custom => self.ui_custom_search(ui),
        }

        self.ui_search_results(ui);
        self.ui_training_output(ui);
    }

    /// Ручная конфигурация: та же сетка гиперпараметров, что и раньше.
    fn ui_single_config(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Модель:");
            ui.selectable_value(&mut self.form.kind, ModelKind::Transformer, "transformer");
            ui.selectable_value(&mut self.form.kind, ModelKind::Mlp, "mlp");
            ui.selectable_value(&mut self.form.kind, ModelKind::Kan, "kan");
        });

        egui::Grid::new("cfg_grid").num_columns(2).show(ui, |ui| {
            match self.form.kind {
                ModelKind::Mlp => {
                    ui.label("mlp width");
                    ui.add(egui::DragValue::new(&mut self.form.mlp_width).range(1..=2048));
                    ui.end_row();
                    ui.label("mlp layers");
                    ui.add(egui::DragValue::new(&mut self.form.mlp_layers).range(1..=12));
                    ui.end_row();
                }
                ModelKind::Kan => {
                    ui.label("kan width");
                    ui.add(egui::DragValue::new(&mut self.form.kan_width).range(1..=2048));
                    ui.end_row();
                    ui.label("kan layers");
                    ui.add(egui::DragValue::new(&mut self.form.kan_layers).range(1..=12));
                    ui.end_row();
                    ui.label("kan grid");
                    ui.add(egui::DragValue::new(&mut self.form.kan_grid).range(2..=128));
                    ui.end_row();
                }
                ModelKind::Transformer => {
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
                .add_enabled(!self.busy(), egui::Button::new("Обучить"))
                .clicked()
            {
                match (self.form.build(), self.active_data()) {
                    (Ok((nc, tcfg)), Some((data, split))) => {
                        self.train_parameter_count = None;
                        self.worker.reset_cancel();
                        self.worker.send(Command::TrainNumeric {
                            data,
                            split,
                            nc,
                            tcfg,
                            // Ручной запуск — фаза разработки: test не трогаем.
                            final_phase: false,
                        });
                    }
                    (Ok(_), None) => self.status = NO_DATASET.to_string(),
                    (Err(e), _) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.training, egui::Button::new("Отмена"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена…".to_string();
            }
            if ui
                .add_enabled(
                    self.model_info.is_some() && !self.busy(),
                    egui::Button::new("Сохранить модель…"),
                )
                .clicked()
            {
                self.save_model_dialog();
            }
        });
    }

    /// Готовый бюджет: архитектуры, цель и цена операции.
    fn ui_auto_search(&mut self, ui: &mut egui::Ui, budget: SearchBudget) {
        ui.horizontal(|ui| {
            ui.label("Бюджет:");
            for candidate in [
                SearchBudget::Quick,
                SearchBudget::Balanced,
                SearchBudget::Thorough,
            ] {
                if ui
                    .selectable_label(budget == candidate, candidate.label())
                    .clicked()
                {
                    self.mode = TrainingMode::Auto(candidate);
                }
            }
        });
        ui.label(format!("{}: {}", budget.label(), budget.hint()));
        self.ui_search_common(ui, self.optimize_form.axes(budget));
    }

    /// Своя сетка: форма редактируется свободно, оси собираются перед запуском.
    fn ui_custom_search(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("custom_grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("seeds");
                ui.text_edit_singleline(&mut self.custom_form.seeds);
                ui.end_row();
                ui.label("d-models");
                ui.text_edit_singleline(&mut self.custom_form.d_models);
                ui.end_row();
                ui.label("layers");
                ui.text_edit_singleline(&mut self.custom_form.layers);
                ui.end_row();
                ui.label("d-ffs");
                ui.text_edit_singleline(&mut self.custom_form.d_ffs);
                ui.end_row();
                ui.label("lrs");
                ui.text_edit_singleline(&mut self.custom_form.lrs);
                ui.end_row();
                ui.label("value-encoders");
                ui.text_edit_singleline(&mut self.custom_form.value_encoders);
                ui.end_row();
                ui.label("fourier-scales");
                ui.text_edit_singleline(&mut self.custom_form.fourier_scales);
                ui.end_row();
                ui.label("schedulers");
                ui.text_edit_singleline(&mut self.custom_form.schedulers);
                ui.end_row();
                ui.label("epochs");
                ui.add(egui::DragValue::new(&mut self.custom_form.epochs).range(1..=100000));
                ui.end_row();
                ui.label("batch");
                ui.add(egui::DragValue::new(&mut self.custom_form.batch).range(1..=8192));
                ui.end_row();
            });
        let axes = self
            .optimize_form
            .model_kinds()
            .and_then(|kinds| self.custom_form.build(kinds));
        self.ui_search_common(ui, axes);
    }

    /// Общая часть обоих режимов поиска: архитектуры, цель, цена и запуск.
    fn ui_search_common(&mut self, ui: &mut egui::Ui, axes: Result<SweepAxes, String>) {
        let objective_before = self.optimize_form.objective;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.optimize_form.include_transformer, "transformer");
            ui.checkbox(&mut self.optimize_form.include_mlp, "mlp");
            ui.checkbox(&mut self.optimize_form.include_kan, "kan");
            ui.label("цель:");
            egui::ComboBox::from_id_salt("search_objective")
                .selected_text(objective_label(self.optimize_form.objective()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.optimize_form.objective, 0, "worst-output R²");
                    ui.selectable_value(&mut self.optimize_form.objective, 1, "aggregate R²");
                    ui.selectable_value(&mut self.optimize_form.objective, 2, "mean-output R²");
                    ui.selectable_value(&mut self.optimize_form.objective, 3, "aggregate nRMSE");
                });
        });
        if self.optimize_form.objective != objective_before {
            self.sort_search_rows();
            self.search_selected = None;
        }

        // Цена операции — до запуска: она понятнее названия бюджета.
        let folds = self.dataset.as_ref().map_or(1, |d| match d.split {
            crate::split::SplitPlan::KFold { k, .. } => k,
            crate::split::SplitPlan::Holdout { .. } => 1,
        });
        match &axes {
            Ok(a) => match sweep::sweep_cost(a, folds) {
                Ok(cost) => {
                    ui.label(format!("Оценка: {}", cost.describe()));
                }
                Err(e) => {
                    ui.colored_label(egui::Color32::from_rgb(200, 60, 60), e);
                }
            },
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(200, 60, 60), e.clone());
            }
        }

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Искать"))
                .clicked()
            {
                match (axes, self.active_data()) {
                    (Ok(axes), Some((data, split))) => {
                        self.worker.reset_cancel();
                        // Отпечаток фиксируется в момент отправки команды, а не
                        // когда worker успеет ответить SearchStarted.
                        self.search_stamp = self.dataset_stamp();
                        self.searching = true;
                        self.search_selected = None;
                        self.search_rows.clear();
                        self.search_total = None;
                        self.search_cancelled = false;
                        self.worker.send(Command::Search {
                            data,
                            split,
                            axes,
                            objective: self.optimize_form.objective(),
                        });
                    }
                    (Ok(_), None) => self.status = NO_DATASET.to_string(),
                    (Err(e), _) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.searching, egui::Button::new("Отмена"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена поиска…".to_string();
            }
        });
    }

    /// Таблица результатов и запуск финального обучения по выбранной строке.
    fn ui_search_results(&mut self, ui: &mut egui::Ui) {
        if self.search_rows.is_empty() {
            return;
        }
        ui.separator();
        if let Some((cfgs, runs)) = self.search_total {
            ui.label(format!(
                "Конфигураций: {cfgs}; прогонов: {runs}; готово: {}",
                self.search_rows.len()
            ));
        }
        if self.search_cancelled {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Поиск отменён; показаны завершённые конфигурации.",
            );
        }
        let stale = !self.search_matches_dataset();
        if stale {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Данные или разбиение изменились: результат относится к другому \
                 набору, финальное обучение по нему запрещено.",
            );
        }

        let source = epoch_sweep::source_label(self.search_rows[0].source);
        ui.label(format!(
            "Ранжирование: {}; метрики {source}",
            objective_label(self.optimize_form.objective())
        ));

        let objective = self.optimize_form.objective();
        let mut selected = self.search_selected;
        egui::ScrollArea::vertical()
            .max_height(300.0)
            .show(ui, |ui| {
                egui::Grid::new("search_rows")
                    .num_columns(8)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("#");
                        ui.label("score");
                        ui.label("R²");
                        ui.label("worst y");
                        ui.label("mean y");
                        ui.label("aggr nRMSE");
                        ui.label("rel");
                        ui.label("конфигурация");
                        ui.end_row();
                        for (i, row) in self.search_rows.iter().enumerate() {
                            let mark = if i == 0 { "*" } else { "" };
                            if ui
                                .add_enabled_ui(!self.searching, |ui| {
                                    ui.selectable_label(
                                        selected == Some(i),
                                        format!("{mark}{}", i + 1),
                                    )
                                })
                                .inner
                                .clicked()
                            {
                                selected = Some(i);
                            }
                            ui.label(format!("{:.5}", objective_display_score(objective, row)));
                            ui.label(format!("{:.5}", row.r2_mean));
                            ui.label(format!("{:.5}", row.worst_output_r2_mean));
                            ui.label(format!("{:.5}", row.mean_output_r2_mean));
                            ui.label(format!("{:.5}", row.nrmse_mean));
                            ui.label(format!("{:.1}%", row.rel_mean * 100.0));
                            ui.label(&row.label);
                            ui.end_row();
                        }
                    });
            });
        self.search_selected = selected;

        // По умолчанию берётся лучшая строка: она же первая в ранжировании.
        let chosen = self
            .search_selected
            .or(if self.search_rows.is_empty() {
                None
            } else {
                Some(0)
            })
            .and_then(|i| self.search_rows.get(i).map(|r| r.choice.clone()));
        ui.horizontal(|ui| {
            let ready = chosen.is_some() && !self.busy() && !stale;
            if ui
                .add_enabled(ready, egui::Button::new("Обучить финально"))
                .on_hover_text(
                    "Переобучить выбранную конфигурацию на train+validation и один раз \
                     открыть test",
                )
                .clicked()
            {
                if let Some(choice) = &chosen {
                    self.train_final_from_choice(choice);
                }
            }
            if ui
                .add_enabled(
                    chosen.is_some() && !self.busy(),
                    egui::Button::new("В ручной режим"),
                )
                .clicked()
            {
                if let Some(choice) = &chosen {
                    self.apply_choice_to_train(choice);
                    self.mode = TrainingMode::Single;
                }
            }
            if ui
                .add_enabled(
                    chosen.is_some() && !self.busy(),
                    egui::Button::new("Подобрать эпохи"),
                )
                .clicked()
            {
                if let Some(choice) = &chosen {
                    self.apply_choice_to_epoch_sweep(choice);
                }
            }
        });
    }

    /// Запустить финальное обучение по выбранной строке поиска.
    ///
    /// Конфигурация берётся из строки напрямую, а seed — заранее заданный
    /// `final_init_seed`: выбирать seed по результату поиска значит подбирать
    /// его по тем же данным.
    fn train_final_from_choice(&mut self, choice: &SweepChoice) {
        let Some((data, split)) = self.active_data() else {
            self.status = NO_DATASET.to_string();
            return;
        };
        let nc = NumericConfig {
            kind: choice.kind,
            transformer: ModelConfig {
                d_model: choice.d_model,
                n_heads: choice.heads,
                n_enc_layers: choice.layers,
                n_dec_layers: choice.layers,
                d_ff: choice.d_ff,
                ln_eps: 1e-5,
            },
            value: choice.value,
            mlp_width: choice.mlp_width,
            mlp_layers: choice.mlp_layers,
            kan: choice.kan,
        };
        let tcfg = TrainConfig {
            epochs: choice.final_epochs,
            batch_size: choice.batch_size,
            lr: choice.lr,
            seed: DEFAULT_FINAL_INIT_SEED,
            schedule: choice.schedule,
        };
        if let Err(e) =
            validate_numeric(&nc).and_then(|()| validate_train(tcfg.lr, tcfg.batch_size))
        {
            self.status = format!("Ошибка: {e}");
            return;
        }
        self.train_parameter_count = None;
        self.worker.reset_cancel();
        self.worker.send(Command::TrainNumeric {
            data,
            split,
            nc,
            tcfg,
            final_phase: true,
        });
    }

    /// Результат обучения: кривая, метрики и единственный замер на test.
    fn ui_training_output(&mut self, ui: &mut egui::Ui) {
        if let Some(count) = self.train_parameter_count {
            ui.label(format!("Параметров: {count}"));
        }

        if !self.loss_curve.is_empty() || !self.final_loss_curve.is_empty() {
            let development = PlotPoints::from(self.loss_curve.clone());
            let final_refit = PlotPoints::from(self.final_loss_curve.clone());
            Plot::new("loss_plot").height(220.0).show(ui, |pui| {
                if !self.loss_curve.is_empty() {
                    pui.line(Line::new(development).name("development train loss"));
                }
                if !self.final_loss_curve.is_empty() {
                    pui.line(Line::new(final_refit).name("final train+validation loss"));
                }
            });
        }
        if let Some(m) = &self.metrics {
            ui.separator();
            ui.label(format!(
                "validation: RMSE={:.5}   MAE={:.5}   rel.error={:.2}%   R²={:.5}",
                m.rmse,
                m.mae,
                m.rel_error * 100.0,
                m.r2
            ));
            if self.final_eval.is_none() {
                ui.label("Test отложен: его открывает только финальное обучение.");
            }
        }
        if let Some(f) = &self.final_eval {
            ui.separator();
            ui.label(format!(
                "test ({} строк, единственный замер): RMSE={:.5}   MAE={:.5}   \
                 rel.error={:.2}%   R²={:.5}",
                f.origin.test_rows,
                f.metrics.rmse,
                f.metrics.mae,
                f.metrics.rel_error * 100.0,
                f.metrics.r2
            ));
        }
        if let Some(warning) = self
            .model_info
            .as_ref()
            .and_then(ModelInfo::categorical_warning)
        {
            ui.colored_label(egui::Color32::from_rgb(200, 120, 0), warning);
        }
    }

    pub(super) fn ui_epoch_sweep(&mut self, ui: &mut egui::Ui) {
        ui.heading("Epoch-sweep");
        self.ui_dataset_bar(ui);
        ui.horizontal(|ui| {
            ui.label("Модель:");
            ui.selectable_value(
                &mut self.epoch_form.kind,
                ModelKind::Transformer,
                "transformer",
            );
            ui.selectable_value(&mut self.epoch_form.kind, ModelKind::Mlp, "mlp");
            ui.selectable_value(&mut self.epoch_form.kind, ModelKind::Kan, "kan");
        });
        egui::Grid::new("epoch_cfg")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                match self.epoch_form.kind {
                    ModelKind::Mlp => {
                        ui.label("mlp width");
                        ui.add(
                            egui::DragValue::new(&mut self.epoch_form.mlp_width).range(1..=2048),
                        );
                        ui.end_row();
                        ui.label("mlp layers");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.mlp_layers).range(1..=12));
                        ui.end_row();
                    }
                    ModelKind::Kan => {
                        ui.label("kan width");
                        ui.add(
                            egui::DragValue::new(&mut self.epoch_form.kan_width).range(1..=2048),
                        );
                        ui.end_row();
                        ui.label("kan layers");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.kan_layers).range(1..=12));
                        ui.end_row();
                        ui.label("kan grid");
                        ui.add(egui::DragValue::new(&mut self.epoch_form.kan_grid).range(2..=128));
                        ui.end_row();
                    }
                    ModelKind::Transformer => {
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
                                egui::DragValue::new(&mut self.epoch_form.fourier_bands)
                                    .range(1..=64),
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
                match (self.epoch_form.build(), self.active_data()) {
                    (Ok(req), Some((data, split))) => {
                        self.worker.reset_cancel();
                        self.worker.send(Command::EpochSweep {
                            data,
                            split,
                            nc: req.nc,
                            base_tcfg: req.base_tcfg,
                            milestones: req.milestones,
                            target_r2: req.target_r2,
                            min_gain: req.min_gain,
                            plateau_min: req.plateau_min,
                        });
                    }
                    (Ok(_), None) => self.status = NO_DATASET.to_string(),
                    (Err(e), _) => self.status = format!("Ошибка: {e}"),
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
            let recommended = self.epoch_recommendation.as_ref().map(|(e, _)| *e);
            if ui
                .add_enabled(
                    recommended.is_some() && !self.epoch_sweeping,
                    egui::Button::new("Apply recommended to Train"),
                )
                .on_hover_text("Перенести конфиг и рекомендованное число эпох в Train")
                .clicked()
            {
                if let Some(epochs) = recommended {
                    self.apply_epoch_form_to_train(epochs);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_form_builds_kan_config() {
        let form = TrainForm {
            kind: ModelKind::Kan,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
            venc: 2,
            ..Default::default()
        };

        let (config, _) = form.build().unwrap();
        assert_eq!(config.kind, ModelKind::Kan);
        assert_eq!(config.value.kind, ValueEncoderKind::Linear);
        assert_eq!(config.kan.width, 16);
        assert_eq!(config.kan.layers, 2);
        assert_eq!(config.kan.grid, 8);
    }

    #[test]
    fn epoch_sweep_form_builds_kan_config() {
        let form = EpochSweepForm {
            kind: ModelKind::Kan,
            kan_width: 16,
            kan_layers: 2,
            kan_grid: 8,
            ..Default::default()
        };

        let request = form.build().unwrap();
        assert_eq!(request.nc.kind, ModelKind::Kan);
        assert_eq!(request.nc.kan.width, 16);
        assert_eq!(request.nc.kan.layers, 2);
        assert_eq!(request.nc.kan.grid, 8);
    }
}
