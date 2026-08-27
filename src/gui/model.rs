//! Экран модели: кривые рёбер KAN, формулы и диагностика.

use super::messages::Command;
use super::messages::ModelOrigin;
use super::session::{split_plan_label, App, KAN_CURVE_SAMPLES};
use crate::metrics::Metrics;
use crate::numeric_model::ModelKind;
use crate::schema::ModelSchema;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

/// Активная модель глазами UI. Схема — источник истины про число входов и
/// выходов и про их имена, поэтому отдельных счётчиков рядом нет.
#[derive(Clone)]
pub(super) struct ModelInfo {
    pub(super) schema: ModelSchema,
    pub(super) kind: ModelKind,
    pub(super) source: String,
    /// Чем является активная модель: отладочной, финальной или загруженной.
    pub(super) origin: ModelOrigin,
    pub(super) parameter_count: usize,
}

impl ModelInfo {
    /// MLP и KAN получают код категории как обычное число и воспринимают
    /// порядок кодов как расстояние. Embedding категорий есть только у
    /// transformer, поэтому здесь предупреждение, а не молчание.
    pub(super) fn categorical_warning(&self) -> Option<String> {
        if self.kind == ModelKind::Transformer {
            return None;
        }
        let names: Vec<&str> = self
            .schema
            .inputs()
            .iter()
            .filter(|c| c.cardinality().is_some())
            .map(|c| c.name())
            .collect();
        if names.is_empty() {
            return None;
        }
        Some(format!(
            "⚠ категориальные входы ({}) кодируются числами: порядок кодов будет \
             воспринят как расстояние. Embedding категорий есть только у transformer.",
            names.join(", ")
        ))
    }
}

/// Подразделы раздела «Модель».
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelView {
    Summary,
    KanCurves,
    KanFormulas,
    Diagnose,
}

impl ModelView {
    pub(super) fn label(self) -> &'static str {
        match self {
            ModelView::Summary => "Итоги",
            ModelView::KanCurves => "Кривые KAN",
            ModelView::KanFormulas => "Формулы",
            ModelView::Diagnose => "Диагностика",
        }
    }
}

impl App {
    /// Раздел «Модель»: всё, что можно узнать про обученную модель.
    pub(super) fn ui_model(&mut self, ui: &mut egui::Ui) {
        ui.heading("Модель");
        ui.horizontal(|ui| {
            for view in [
                ModelView::Summary,
                ModelView::KanCurves,
                ModelView::KanFormulas,
                ModelView::Diagnose,
            ] {
                ui.selectable_value(&mut self.model_view, view, view.label());
            }
        });
        ui.separator();
        match self.model_view {
            ModelView::Summary => self.ui_model_summary(ui),
            ModelView::KanCurves => self.ui_kan_curves(ui),
            ModelView::KanFormulas => self.ui_kan_formulas(ui),
            ModelView::Diagnose => self.ui_diagnose(ui),
        }
    }

    /// Итоги: происхождение модели, метрики и единственный замер на test.
    fn ui_model_summary(&mut self, ui: &mut egui::Ui) {
        let Some(info) = &self.model_info else {
            ui.label("Модель ещё не обучена и не загружена.");
            return;
        };
        ui.label(format!("Источник: {}", info.source));
        // Происхождение самой модели, а не данных: по нему видно, на чём она
        // обучена и что означают показанные ниже метрики.
        ui.label(format!("Модель: {}", info.origin.label()));
        if let ModelOrigin::Development(stamp) | ModelOrigin::Final(stamp) = &info.origin {
            ui.label(format!(
                "Протокол: {}; конфигурация проверена на этих данных",
                split_plan_label(stamp.split)
            ));
        }
        if !info.origin.is_final() {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                "Доступные данные использованы не полностью: validation осталась вне обучения. \
                 Для результата работы зафиксируйте кандидата и обучите финально.",
            );
        }
        ui.label(format!("Параметров: {}", info.parameter_count));
        ui.label(format!("Входы: {}", info.schema.input_names().join(", ")));
        ui.label(format!("Выходы: {}", info.schema.output_names().join(", ")));
        if let Some(warning) = info.categorical_warning() {
            ui.colored_label(egui::Color32::from_rgb(200, 120, 0), warning);
        }
        if let Some(reason) = self.schema_mismatch() {
            ui.colored_label(
                egui::Color32::from_rgb(200, 120, 0),
                format!("Не подходит к активным данным: {reason}"),
            );
        }
        ui.separator();
        if let Some(metrics) = &self.metrics {
            show_metrics(
                ui,
                "validation",
                metrics,
                self.metrics_per_output.as_deref(),
                info.schema.outputs(),
            );
            if let Some(origin) = self.validation_origin {
                ui.label(format!(
                    "Протокол: {}; init seed {}",
                    split_plan_label(origin.plan),
                    origin.init_seed
                ));
            }
            // У CV одного среднего мало: одинаковое среднее при разном разбросе
            // между folds означает разную надёжность вывода.
            if let Some(std) = self.r2_std_folds {
                ui.label(format!("Разброс R² между folds: ±{std:.5}"));
            }
        }
        if let Some(final_eval) = &self.final_eval {
            show_metrics(
                ui,
                &format!(
                    "test ({} строк, единственный замер)",
                    final_eval.origin.test_rows
                ),
                &final_eval.metrics,
                Some(&final_eval.per_output),
                info.schema.outputs(),
            );
            ui.label(format!(
                "Протокол: {}; final init seed {}",
                split_plan_label(final_eval.origin.plan),
                final_eval.origin.final_init_seed
            ));
        } else if self.metrics.is_some() {
            ui.label("Test отложен: его открывает только финальное обучение.");
        } else {
            ui.label("Checkpoint не хранит метрики: они появятся после обучения в этой сессии.");
        }
        if let Some(reports) = &self.interpret_reports {
            ui.separator();
            if let Some(profile) = reports.profile() {
                ui.label(format!("Конвейер интерпретации {}", profile.describe()));
            }
            // У модели разработки виден эффект прунинга на validation…
            if let Some(d) = &reports.development {
                if let (Some(before), Some(after), Some(ft)) =
                    (d.r2_before, d.r2_after_prune, d.r2_after_finetune)
                {
                    ui.label(format!(
                        "R² на validation: до прунинга {before:.5} → после {after:.5} → \
                         после fine-tune {ft:.5}"
                    ));
                }
                ui.label(format!(
                    "Активных рёбер после прунинга: {}/{}",
                    d.active_edges.0, d.active_edges.1
                ));
            }
            // …а у финальной — какой стала структура сохранённой модели.
            if let Some(f) = &reports.final_model {
                if let Some(c) = f.compaction {
                    ui.label(format!(
                        "Финальная модель: скрытых узлов {} → {}, параметров {} → {}",
                        c.nodes_before, c.nodes_after, c.params_before, c.params_after
                    ));
                }
                ui.label(format!(
                    "Активных рёбер финальной модели: {}/{}",
                    f.active_edges.0, f.active_edges.1
                ));
            }
        }

        if ui
            .add_enabled(!self.busy(), egui::Button::new("Сохранить модель…"))
            .clicked()
        {
            self.save_model_dialog();
        }
    }

    pub(super) fn request_kan_curve(&mut self) {
        let Some((n_inputs, n_outputs)) = self
            .kan_info
            .as_ref()
            .and_then(|info| info.layer_dims.get(self.kan_layer).copied())
        else {
            return;
        };
        self.kan_input = self.kan_input.min(n_inputs - 1);
        self.kan_output = self.kan_output.min(n_outputs - 1);
        self.kan_curve.clear();
        self.worker.send(Command::SampleKanEdge {
            layer: self.kan_layer,
            input: self.kan_input,
            output: self.kan_output,
            samples: KAN_CURVE_SAMPLES,
        });
    }

    pub(super) fn save_model_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("bin", &["bin"])
            .save_file()
        {
            self.worker
                .send(Command::SaveModel(p.display().to_string()));
            self.status = "сохранение модели…".to_string();
        }
    }

    pub(super) fn ui_kan_curves(&mut self, ui: &mut egui::Ui) {
        ui.heading("KAN: функции рёбер");
        let Some((layer_dims, domain)) = self
            .kan_info
            .as_ref()
            .map(|info| (info.layer_dims.clone(), info.domain))
        else {
            ui.label("Обучите или загрузите KAN-модель, чтобы увидеть функции её рёбер.");
            return;
        };
        if layer_dims.is_empty() {
            ui.label("В модели нет KAN-слоёв.");
            return;
        }

        self.kan_layer = self.kan_layer.min(layer_dims.len() - 1);
        let previous_layer = self.kan_layer;
        egui::ComboBox::from_label("слой")
            .selected_text(format!(
                "{} ({} → {})",
                self.kan_layer, layer_dims[self.kan_layer].0, layer_dims[self.kan_layer].1
            ))
            .show_ui(ui, |ui| {
                for (layer, &(n_inputs, n_outputs)) in layer_dims.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.kan_layer,
                        layer,
                        format!("{layer} ({n_inputs} → {n_outputs})"),
                    );
                }
            });
        let mut changed = self.kan_layer != previous_layer;
        if changed {
            self.kan_input = 0;
            self.kan_output = 0;
        }

        let (n_inputs, n_outputs) = layer_dims[self.kan_layer];
        self.kan_input = self.kan_input.min(n_inputs - 1);
        self.kan_output = self.kan_output.min(n_outputs - 1);
        ui.horizontal(|ui| {
            ui.label("вход");
            changed |= ui
                .add(egui::DragValue::new(&mut self.kan_input).range(0..=n_inputs - 1))
                .changed();
            ui.label("выход");
            changed |= ui
                .add(egui::DragValue::new(&mut self.kan_output).range(0..=n_outputs - 1))
                .changed();
        });
        if changed {
            self.request_kan_curve();
        }

        let x_label = if self.kan_layer == 0 {
            "нормализованный исходный вход"
        } else {
            "активация предыдущего KAN-слоя"
        };
        ui.label(format!(
            "φ{}→{}(x), слой {}; x – {}, сетка [{:.1}, {:.1}]",
            self.kan_input, self.kan_output, self.kan_layer, x_label, domain.0, domain.1
        ));
        if self.kan_curve.is_empty() {
            ui.label("Выборка кривой…");
            return;
        }
        let points = PlotPoints::from(self.kan_curve.clone());
        Plot::new("kan_edge_curve")
            .height(320.0)
            .include_x(domain.0 as f64)
            .include_x(domain.1 as f64)
            .show(ui, |pui| {
                pui.line(Line::new(points).name("φ(x)"));
            });
    }

    pub(super) fn ui_kan_formulas(&mut self, ui: &mut egui::Ui) {
        ui.heading("KAN: символьные формулы");
        let Some(symbolic_available) = self.kan_info.as_ref().map(|info| info.symbolic_available)
        else {
            ui.label("Обучите KAN-модель, чтобы извлечь формулы.");
            return;
        };
        if !symbolic_available {
            ui.label(
                "Checkpoint без калибровочной секции (сохранён старой версией): обучите KAN в этой сессии или пересохраните модель – новые .bin несут выборку train-активаций.",
            );
            return;
        }

        ui.label(
            "Фит строится по train-активациям, а ниже показаны формулы в исходных единицах данных.",
        );
        let action = if self.kan_symbolic.is_some() {
            "Обновить формулы"
        } else {
            "Извлечь формулы"
        };
        if ui
            .add_enabled(!self.busy(), egui::Button::new(action))
            .clicked()
        {
            self.kan_symbolic = None;
            self.kan_symbolic_pending = true;
            self.status = "символьная экстракция…".to_string();
            self.worker.send(Command::ExtractKanSymbolic);
        }
        if self.kan_symbolic_pending {
            ui.label("Подбор примитивов по рёбрам…");
            return;
        }

        let Some(result) = &self.kan_symbolic else {
            return;
        };
        ui.separator();
        egui::Grid::new("kan_symbolic_metrics")
            .num_columns(2)
            .show(ui, |ui| {
                match (
                    &result.formula_metrics,
                    result.kan_r2,
                    &result.evaluation_label,
                ) {
                    (Some(metrics), Some(kan_r2), Some(label)) => {
                        ui.label(format!("R² формул на {label}"));
                        ui.label(format!("{:.5} (KAN: {kan_r2:.5})", metrics.r2));
                        ui.end_row();
                        ui.label("Ошибка формул");
                        ui.label(format!(
                            "RMSE {:.5}, rel. {:.2}%",
                            metrics.rmse,
                            metrics.rel_error * 100.0
                        ));
                        ui.end_row();
                        if label == "train+validation" {
                            ui.label("Интерпретация");
                            ui.label("верность формул финальной модели, не оценка обобщения");
                            ui.end_row();
                        }
                    }
                    _ => {
                        ui.label("Сравнение формул с KAN");
                        ui.label("недоступно: модель из checkpoint-а без набора оценки");
                        ui.end_row();
                    }
                }
                ui.label("Подгонка активных рёбер");
                ui.label(format!(
                    "min R² {:.4}, среднее R² {:.4}",
                    result.min_edge_r2, result.mean_edge_r2
                ));
                ui.end_row();
            });

        if ui.button("Скопировать формулы").clicked() {
            ui.ctx().copy_text(result.formulas.clone());
        }

        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("kan_symbolic_formulas")
            .max_height(300.0)
            .show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(&result.formulas).monospace())
                        .selectable(true),
                );
            });

        if result.weak_edges.is_empty() {
            ui.label("Все активные рёбра подогнаны с R² ≥ 0.99.");
            return;
        }
        ui.separator();
        let warn = egui::Color32::from_rgb(200, 120, 0);
        ui.colored_label(
            warn,
            format!(
                "{} приближённых рёбер (R² < 0.99): формулы для них требуют проверки.",
                result.weak_edges.len()
            ),
        );
        egui::Grid::new("kan_symbolic_weak_edges")
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.label("слой");
                ui.label("вход");
                ui.label("выход");
                ui.label("примитив");
                ui.label("R²");
                ui.end_row();
                for edge in &result.weak_edges {
                    ui.label(edge.layer.to_string());
                    ui.label(&edge.input);
                    ui.label(&edge.output);
                    ui.label(&edge.primitive);
                    ui.label(format!("{:.4}", edge.r2));
                    ui.end_row();
                }
            });
    }

    pub(super) fn ui_diagnose(&mut self, ui: &mut egui::Ui) {
        ui.heading("Диагностика");
        if self.model_info.is_none() {
            ui.label("Обучите модель в разделе «Обучение» — диагностика использует её данные.");
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
                "Экстраполяция: {} из {} строк ({}) вне обученного диапазона",
                d.extrapolation_rows, d.extrapolation_total, d.evaluation_label
            ));
            if d.evaluation_label == "train+validation" {
                ui.label(
                    "Финальная модель обучена на этом наборе: диагностика описывает fit, а не обобщение.",
                );
            }

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
            match &d.sensitivity {
                Ok(r) => {
                    ui.label(format!(
                        "Чувствительность ‖Δy‖/‖Δx‖ (норм., {} пар соседних строк):",
                        r.pairs
                    ));
                    ui.label(format!(
                        "  модель: среднее {:.2}, макс {:.2}",
                        r.model.mean, r.model.max
                    ));
                    match (r.reference, r.divergence) {
                        (Some(reference), Some(divergence)) => {
                            ui.label(format!(
                                "  процесс: среднее {:.2}, макс {:.2}",
                                reference.mean, reference.max
                            ));
                            // Диагностика — именно расхождение: чувствительность
                            // модели сама по себе точности не доказывает.
                            ui.label(format!(
                                "  расхождение средних: {divergence:.2} (надёжность \
                                 видна по нему вместе с метриками на validation)"
                            ));
                        }
                        _ => {
                            ui.label("  процесс: недоступен (известен только у встроенной задачи)");
                        }
                    }
                    if r.categorical_inputs > 0 {
                        ui.label(format!(
                            "  категориальные входы ({}) не возмущались",
                            r.categorical_inputs
                        ));
                    }
                }
                Err(e) => {
                    ui.label(format!("Чувствительность: не посчитана — {e}"));
                }
            }
        }
    }
}

fn show_metrics(
    ui: &mut egui::Ui,
    label: &str,
    aggregate: &Metrics,
    per_output: Option<&[Metrics]>,
    outputs: &[crate::schema::Column],
) {
    ui.label(format!(
        "{label}: RMSE={:.5}   MAE={:.5}   rel.error={:.2}%   R²={:.5}",
        aggregate.rmse,
        aggregate.mae,
        aggregate.rel_error * 100.0,
        aggregate.r2
    ));
    let Some(per_output) = per_output else {
        return;
    };
    egui::Grid::new(format!("{label}_per_output_metrics"))
        .num_columns(5)
        .striped(true)
        .show(ui, |ui| {
            ui.label("выход");
            ui.label("RMSE");
            ui.label("MAE");
            ui.label("rel.error");
            ui.label("R²");
            ui.end_row();
            for (column, metrics) in outputs.iter().zip(per_output) {
                ui.label(column.display_name());
                ui.label(format!("{:.5}", metrics.rmse));
                ui.label(format!("{:.5}", metrics.mae));
                ui.label(format!("{:.2}%", metrics.rel_error * 100.0));
                ui.label(format!("{:.5}", metrics.r2));
                ui.end_row();
            }
        });
}
