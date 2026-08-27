//! Экран обучения: одна конфигурация, поиск по сетке и кривая по эпохам.

use super::messages::{Command, ModelOrigin, PreparedData};
use super::model::ModelInfo;
use super::session::{App, NO_DATASET};
use crate::config::ModelConfig;
use crate::encoders::{ValueEncoderConfig, ValueEncoderKind};
use crate::interpret::{self, InterpretOverrides, InterpretProfile};
use crate::lifecycle::{CandidateSpec, RunStamp};
use crate::metrics::EvalSource;
use crate::numeric_model::{validate_numeric, KanConfig, ModelKind, NumericConfig};
use crate::split::DEFAULT_FINAL_INIT_SEED;
use crate::sweep::{self, SearchBudget, SweepAxes, SweepChoice, SweepObjective, SweepRow};
use crate::train::{validate_train, LrSchedule, TrainConfig};
use crate::training::{recommended_epoch, EvalSchedule};
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

pub(super) struct SearchForm {
    pub(super) objective: usize, // 0 worst, 1 aggregate, 2 mean, 3 nrmse
    pub(super) include_mlp: bool,
    pub(super) include_transformer: bool,
    pub(super) include_kan: bool,
}

impl Default for SearchForm {
    fn default() -> Self {
        Self {
            objective: 0,
            include_mlp: true,
            include_transformer: true,
            include_kan: true,
        }
    }
}

impl SearchForm {
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

        if self.dataset.is_none() {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Сначала откройте набор в разделе «Данные».",
            );
        }

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
        // Второй шаг общий для всех трёх режимов: кандидат приходит из формы
        // или из выбранной строки, а правило одно.
        self.ui_finalize(ui);
        self.ui_training_output(ui);
    }

    /// Рекомендованная точка остановки по собранной кривой validation.
    fn recommended_epochs(&self) -> Option<(usize, String)> {
        recommended_epoch(
            self.val_curve
                .iter()
                .map(|point| (point[0] as usize, point[1] as f32)),
            0.95,
            0.02,
            0.80,
        )
    }

    /// Профиль интерпретации для запуска. Только у KAN: у остальных моделей
    /// конвейера нет, и молча применять его нельзя.
    ///
    /// Ошибка разрешения профиля возвращается наверх, а не гасится: иначе
    /// обучение молча пошло бы БЕЗ конвейера, хотя его просили.
    fn interpret_profile(&self, kind: ModelKind) -> Result<Option<InterpretProfile>, String> {
        if !self.interpret_enabled || kind != ModelKind::Kan {
            return Ok(None);
        }
        interpret::resolve(true, &self.interpret_overrides)
            .map_err(|e| format!("конвейер интерпретации: {e}"))
    }

    /// Переключатель конвейера рядом с параметрами KAN; переопределения — под
    /// «Дополнительно», потому что нужны редко.
    fn ui_interpret_controls(&mut self, ui: &mut egui::Ui) {
        self.ui_interpret_controls_for(ui, self.form.kind, "single");
    }

    /// То же для модели, выбранной поиском: её вид берётся из строки, а не из
    /// формы ручного режима.
    fn ui_interpret_controls_for(
        &mut self,
        ui: &mut egui::Ui,
        kind: ModelKind,
        context: &'static str,
    ) {
        if kind != ModelKind::Kan {
            return;
        }
        ui.checkbox(
            &mut self.interpret_enabled,
            "Интерпретируемая KAN, профиль v1",
        );
        if !self.interpret_enabled {
            return;
        }
        match interpret::resolve(true, &self.interpret_overrides) {
            Ok(Some(profile)) => {
                ui.label(format!("Конвейер {}", profile.describe()));
            }
            Ok(None) => {}
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(200, 60, 60), e);
            }
        }
        egui::CollapsingHeader::new("Дополнительно")
            // Результаты поиска не очищаются при переходе в ручной режим, и
            // оба блока могут оказаться в одном кадре. Их состояние раскрытия
            // не должно делить один egui-id.
            .id_salt(("interpret_overrides", context))
            .show(ui, |ui| {
                let o = &mut self.interpret_overrides;
                let default = InterpretProfile::v1();

                let mut l1 = o.l1.unwrap_or(default.l1);
                if ui
                    .add(
                        egui::DragValue::new(&mut l1)
                            .range(0.0..=1.0)
                            .speed(1e-4)
                            .prefix("L1 "),
                    )
                    .changed()
                {
                    o.l1 = Some(l1);
                }

                // Текущее значение: переопределение, иначе профильное.
                let effective_prune = o.prune.unwrap_or(default.prune);
                let mut prune_enabled = effective_prune.is_some();
                if ui.checkbox(&mut prune_enabled, "прунинг").changed() {
                    // Выключение записывается ЯВНО: без этого следующий кадр
                    // снова взял бы порог из профиля и флажок бы «отскочил».
                    o.prune = Some(prune_enabled.then(|| default.prune.unwrap_or(0.05)));
                    if !prune_enabled {
                        o.finetune_epochs = None;
                    }
                }
                if prune_enabled {
                    let mut prune = effective_prune.unwrap_or(0.05);
                    if ui
                        .add(
                            egui::DragValue::new(&mut prune)
                                .range(0.0..=0.99)
                                .speed(0.01)
                                .prefix("порог "),
                        )
                        .changed()
                    {
                        o.prune = Some(Some(prune));
                    }
                    let mut epochs = o.finetune_epochs.unwrap_or(default.finetune_epochs);
                    if ui
                        .add(
                            egui::DragValue::new(&mut epochs)
                                .range(1..=1000)
                                .prefix("fine-tune "),
                        )
                        .changed()
                    {
                        o.finetune_epochs = Some(epochs);
                    }
                }

                let mut compact = o.compact.unwrap_or(default.compact);
                if ui.checkbox(&mut compact, "структурное сжатие").changed() {
                    o.compact = Some(compact);
                }
                if ui.button("Сбросить к профилю").clicked() {
                    *o = InterpretOverrides::default();
                }
            });
    }

    /// Расписание замеров validation по эпохам: кривая обучения — настройка
    /// обычного обучения, а не отдельный сценарий.
    fn eval_schedule(&self) -> EvalSchedule {
        if self.eval_every == 0 {
            EvalSchedule::Never
        } else {
            EvalSchedule::Every(self.eval_every)
        }
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
                    ui.label("ширина MLP");
                    ui.add(egui::DragValue::new(&mut self.form.mlp_width).range(1..=2048));
                    ui.end_row();
                    ui.label("слоёв MLP");
                    ui.add(egui::DragValue::new(&mut self.form.mlp_layers).range(1..=12));
                    ui.end_row();
                }
                ModelKind::Kan => {
                    ui.label("ширина KAN");
                    ui.add(egui::DragValue::new(&mut self.form.kan_width).range(1..=2048));
                    ui.end_row();
                    ui.label("слоёв KAN");
                    ui.add(egui::DragValue::new(&mut self.form.kan_layers).range(1..=12));
                    ui.end_row();
                    ui.label("сетка KAN");
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
                    ui.label("слоёв");
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
            ui.label("размер пакета");
            ui.add(egui::DragValue::new(&mut self.form.batch).range(1..=8192));
            ui.end_row();
            ui.label("эпохи");
            ui.add(egui::DragValue::new(&mut self.form.epochs).range(1..=100000));
            ui.end_row();
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.form.seed));
            ui.end_row();
        });

        self.eval_every = self.eval_every.min(self.form.epochs);
        self.ui_interpret_controls(ui);
        ui.horizontal(|ui| {
            ui.label("validation каждые");
            ui.add(egui::DragValue::new(&mut self.eval_every).range(0..=self.form.epochs));
            ui.label(if self.eval_every == 0 {
                "эпох (0 — не измерять по ходу)"
            } else {
                "эпох"
            });
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
            // Проверка доступна всегда: она не трогает test. При K-fold она
            // означает оценку по всем folds, а не одну произвольную модель.
            if ui
                .add_enabled(!self.busy(), egui::Button::new("Проверить конфигурацию"))
                .clicked()
            {
                match self.current_stamp() {
                    Ok((data, stamp)) => self.send_check(data, stamp),
                    Err(e) => self.status = format!("Ошибка: {e}"),
                }
            }
            if ui
                .add_enabled(self.training, egui::Button::new("Отмена"))
                .clicked()
            {
                self.worker.request_cancel();
                self.status = "отмена…".to_string();
            }
            // Кнопка называет то, что сохранит: отладочная модель обучена
            // только на train, и путать её с результатом работы нельзя.
            let save_label = match self.model_info.as_ref().map(|info| &info.origin) {
                Some(ModelOrigin::Final(_)) => "Сохранить финальную модель…",
                Some(ModelOrigin::Development(_)) => "Сохранить отладочную модель…",
                _ => "Сохранить модель…",
            };
            if ui
                .add_enabled(
                    self.model_info.is_some() && !self.busy(),
                    egui::Button::new(save_label),
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
        self.ui_search_common(ui, self.search_form.axes(budget));
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
            .search_form
            .model_kinds()
            .and_then(|kinds| self.custom_form.build(kinds));
        self.ui_search_common(ui, axes);
    }

    /// Общая часть обоих режимов поиска: архитектуры, цель, цена и запуск.
    fn ui_search_common(&mut self, ui: &mut egui::Ui, axes: Result<SweepAxes, String>) {
        let objective_before = self.search_form.objective;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.search_form.include_transformer, "transformer");
            ui.checkbox(&mut self.search_form.include_mlp, "mlp");
            ui.checkbox(&mut self.search_form.include_kan, "kan");
            ui.label("цель:");
            egui::ComboBox::from_id_salt("search_objective")
                .selected_text(objective_label(self.search_form.objective()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.search_form.objective, 0, "worst-output R²");
                    ui.selectable_value(&mut self.search_form.objective, 1, "aggregate R²");
                    ui.selectable_value(&mut self.search_form.objective, 2, "mean-output R²");
                    ui.selectable_value(&mut self.search_form.objective, 3, "aggregate nRMSE");
                });
        });
        if self.search_form.objective != objective_before {
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
                    (Ok(axes), Ok((data, split))) => {
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
                            axes: Box::new(axes),
                            objective: self.search_form.objective(),
                        });
                    }
                    (Err(e), _) | (_, Err(e)) => self.status = format!("Ошибка: {e}"),
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

        let source = self.search_rows[0].source.label();
        ui.label(format!(
            "Ранжирование: {}; метрики {source}",
            objective_label(self.search_form.objective())
        ));

        let objective = self.search_form.objective();
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

        // Конвейер применяется уже к выбранной модели, поэтому его настройки
        // стоят здесь же, а не только в ручном режиме.
        if chosen.as_ref().is_some_and(|c| c.kind == ModelKind::Kan) {
            ui.separator();
            ui.label("Ранжирование выполнено до конвейера интерпретации.");
            self.ui_interpret_controls_for(ui, ModelKind::Kan, "search");
        }

        ui.horizontal(|ui| {
            let ready = chosen.is_some() && !self.busy() && !stale;
            if ui
                .add_enabled(ready, egui::Button::new("Проверить выбранную"))
                .on_hover_text(
                    "Обучить выбранную конфигурацию на полном числе эпох и снять \
                     validation/CV-оценку. Test не открывается",
                )
                .clicked()
            {
                match self.current_stamp() {
                    Ok((data, stamp)) => self.send_check(data, stamp),
                    Err(e) => self.status = format!("Ошибка: {e}"),
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
        });
    }

    /// Кандидат из выбранной строки поиска.
    ///
    /// Число эпох — `final_epochs`, а не короткий бюджет самого перебора:
    /// проверять надо ровно ту конфигурацию, которая поедет в финал. Seed —
    /// заранее заданный `final_init_seed`: выбирать его по результату поиска
    /// значит подбирать его по тем же данным.
    fn candidate_from_choice(&self, choice: &SweepChoice) -> Result<CandidateSpec, String> {
        let config = NumericConfig {
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
        let train = TrainConfig {
            epochs: choice.final_epochs,
            batch_size: choice.batch_size,
            lr: choice.lr,
            seed: DEFAULT_FINAL_INIT_SEED,
            schedule: choice.schedule,
        };
        validate_numeric(&config)?;
        validate_train(train.lr, train.batch_size)?;
        Ok(CandidateSpec {
            config,
            train,
            interpret: self.interpret_profile(choice.kind)?,
        })
    }

    /// Кандидат, о котором сейчас идёт речь: из формы в ручном режиме и из
    /// выбранной строки после поиска.
    fn current_candidate(&self) -> Result<CandidateSpec, String> {
        match self.mode {
            TrainingMode::Single => {
                let (config, train) = self.form.build()?;
                let interpret = self.interpret_profile(config.kind)?;
                Ok(CandidateSpec {
                    config,
                    train,
                    interpret,
                })
            }
            TrainingMode::Auto(_) | TrainingMode::Custom => {
                if !self.search_matches_dataset() {
                    return Err(
                        "результат поиска относится к другим данным или разбиению".to_string()
                    );
                }
                let choice = self
                    .search_selected
                    .or((!self.search_rows.is_empty()).then_some(0))
                    .and_then(|i| self.search_rows.get(i))
                    .map(|row| row.choice.clone())
                    .ok_or_else(|| "сначала выполните поиск и выберите строку".to_string())?;
                self.candidate_from_choice(&choice)
            }
        }
    }

    /// Активные данные и отпечаток текущего кандидата.
    pub(super) fn current_stamp(&self) -> Result<(PreparedData, RunStamp), String> {
        let (data, split) = self.active_data()?;
        let dataset_revision = self
            .dataset_revision()
            .ok_or_else(|| NO_DATASET.to_string())?;
        let dataset = self
            .dataset_fingerprint()
            .ok_or_else(|| NO_DATASET.to_string())?;
        Ok((
            data,
            RunStamp {
                dataset,
                dataset_revision,
                split,
                candidate: self.current_candidate()?,
                // Финальный seed задан заранее: подбирать его по результату
                // проверки значит подбирать по тем же данным.
                final_init_seed: DEFAULT_FINAL_INIT_SEED,
            },
        ))
    }

    /// Расписание замеров — настройка запуска, а не личность кандидата: без
    /// ранней остановки оно не меняет модель, только частоту наблюдений.
    /// Поэтому оно едет в команде, а не в отпечатке.
    fn send_check(&mut self, data: PreparedData, stamp: RunStamp) {
        // После поиска число эпох уже выбрано, и отдельная кривая по эпохам
        // означала бы CV-ломаную по одному fold.
        let eval = match self.mode {
            TrainingMode::Single => self.eval_schedule(),
            TrainingMode::Auto(_) | TrainingMode::Custom => EvalSchedule::Never,
        };
        self.train_parameter_count = None;
        self.worker.reset_cancel();
        self.worker.send(Command::CheckCandidate {
            data,
            stamp: Box::new(stamp),
            eval,
        });
    }

    /// Зафиксировать проверенного кандидата. Отпечаток берётся из проверки как
    /// есть: пересобранный по форме мог бы отличаться от проверенного.
    fn send_finalize(&mut self, data: PreparedData, stamp: RunStamp) {
        self.train_parameter_count = None;
        self.worker.reset_cancel();
        self.worker.send(Command::FinalizeCandidate {
            data,
            stamp: Box::new(stamp),
        });
    }

    /// Второй шаг сценария: фиксация кандидата и единственное открытие test.
    ///
    /// Кнопка включается только при точном совпадении с проверенным
    /// кандидатом; во всех остальных случаях причина написана рядом, а не
    /// угадывается по недоступной кнопке.
    pub(super) fn ui_finalize(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let Ok((data, stamp)) = self.current_stamp() else {
            return;
        };
        if let Some(disclosed) = self.lifecycle.disclosure_for(&stamp) {
            ui.label(format!(
                "Этот кандидат уже обучен финально: test открыт ({} строк), результат ниже.",
                disclosed.eval.origin.test_rows
            ));
            return;
        }
        match self.lifecycle.can_finalize(&stamp) {
            Ok(()) => {
                if ui
                    .add_enabled(
                        !self.busy(),
                        egui::Button::new("Зафиксировать и обучить финально"),
                    )
                    .on_hover_text(
                        "Переобучить проверенного кандидата на train+validation и один раз \
                         открыть test",
                    )
                    .clicked()
                {
                    let checked = self
                        .lifecycle
                        .checked_for(&stamp)
                        .map(|run| run.stamp.clone());
                    match checked {
                        Some(stamp) => self.send_finalize(data, stamp),
                        None => self.status = "Ошибка: проверка кандидата не найдена".to_string(),
                    }
                }
            }
            Err(refusal) => {
                ui.add_enabled(false, egui::Button::new("Зафиксировать и обучить финально"));
                ui.colored_label(egui::Color32::from_rgb(200, 120, 0), refusal.message());
            }
        }
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
                    let name = if self.curve_folds > 1 {
                        format!("train loss, среднее по {} folds", self.curve_folds)
                    } else {
                        "development train loss".to_string()
                    };
                    pui.line(Line::new(development).name(name));
                }
                if !self.final_loss_curve.is_empty() {
                    pui.line(Line::new(final_refit).name("final train+validation loss"));
                }
            });
        }
        if !self.val_curve.is_empty() {
            // Кривая по эпохам — то, ради чего раньше был отдельный сценарий.
            let points = PlotPoints::from(self.val_curve.clone());
            let recommendation = self.recommended_epochs();
            // Название говорит, что нарисовано: у K-fold это среднее по
            // folds, а не кривая одного обучения.
            let name = if self.curve_folds > 1 {
                format!("validation R², среднее по {} folds", self.curve_folds)
            } else {
                "validation R²".to_string()
            };
            Plot::new("val_plot")
                .height(200.0)
                .show(ui, |pui| pui.line(Line::new(points).name(name)));
            if let Some((epochs, why)) = recommendation {
                ui.label(format!("Рекомендованная остановка: {epochs} эпох ({why})"));
            }
        }
        // Результат проверки живёт здесь, а не в разделе «Модель»: он
        // относится к кандидату, а не к активной модели.
        if let Some(run) = self.lifecycle.checked() {
            ui.separator();
            let current = self
                .current_stamp()
                .is_ok_and(|(_, stamp)| stamp == run.stamp);
            let source = match run.stamp.eval_source() {
                EvalSource::Cv { k } => format!("cv-{k}"),
                _ => "validation".to_string(),
            };
            let m = &run.eval.metrics;
            ui.label(format!(
                "{source}: RMSE={:.5}   MAE={:.5}   rel.error={:.2}%   R²={:.5}",
                m.rmse,
                m.mae,
                m.rel_error * 100.0,
                m.r2
            ));
            if run.eval.r2_std_folds > 0.0 {
                ui.label(format!(
                    "Разброс R² между folds: ±{:.5}",
                    run.eval.r2_std_folds
                ));
            }
            if !current {
                ui.colored_label(
                    egui::Color32::from_rgb(200, 120, 0),
                    "Проверка относится к другому кандидату: форма изменилась после неё.",
                );
            }
            // Отчёт конвейера этой же проверки: у активной модели он может быть
            // своим, и путать их нельзя.
            let stamp = run.stamp.clone();
            if let Some((_, reports)) = self
                .interpret_reports
                .as_ref()
                .filter(|(reported, _)| *reported == stamp)
            {
                if let Some(profile) = reports.profile() {
                    ui.label(format!("Конвейер интерпретации {}", profile.describe()));
                }
                if let Some(d) = &reports.development {
                    ui.label(format!(
                        "Активных рёбер после прунинга: {}/{}",
                        d.active_edges.0, d.active_edges.1
                    ));
                }
            }
        }
        if let Some(disclosed) = self.lifecycle.disclosure() {
            // Раскрытие переживает смену набора намеренно, но выдавать его за
            // результат текущих данных нельзя.
            let historical = self.dataset_fingerprint() != Some(disclosed.dataset());
            ui.separator();
            let f = &disclosed.eval;
            let prefix = if historical {
                "test прежнего набора данных"
            } else {
                "test"
            };
            ui.label(format!(
                "{prefix} ({} строк, единственный замер): RMSE={:.5}   MAE={:.5}   \
                 rel.error={:.2}%   R²={:.5}",
                f.origin.test_rows,
                f.metrics.rmse,
                f.metrics.mae,
                f.metrics.rel_error * 100.0,
                f.metrics.r2
            ));
            if historical {
                ui.colored_label(
                    egui::Color32::from_rgb(200, 120, 0),
                    "Этот замер относится к прежнему набору данных и к активному отношения не \
                     имеет.",
                );
            }
        }
        if let Some(warning) = self
            .model_info
            .as_ref()
            .and_then(ModelInfo::categorical_warning)
        {
            ui.colored_label(egui::Color32::from_rgb(200, 120, 0), warning);
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
}
