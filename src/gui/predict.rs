//! Экран прогноза: единичный прогноз и пакетный расчёт из Excel.

use super::messages::Command;
use super::session::App;
use crate::schema::ColumnType;
use eframe::egui;

impl App {
    /// Экспорт таблицы с прогнозами. Именно экспорт: исходная книга не
    /// сохраняется, результат — новая минимальная книга со значениями.
    pub(super) fn export_predictions_dialog(&mut self) {
        let Some(input) = rfd::FileDialog::new()
            .add_filter(
                "таблицы",
                &["xlsx", "xlsm", "xlsb", "xls", "ods", "csv", "tsv"],
            )
            .pick_file()
        else {
            return;
        };
        let default_name = input
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}_прогноз.xlsx"))
            .unwrap_or_else(|| "predicted.xlsx".to_string());
        let Some(output) = rfd::FileDialog::new()
            .add_filter("Excel", &["xlsx"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };
        self.batch_predicting = true;
        self.status = "экспорт таблицы с прогнозами…".to_string();
        self.worker.send(Command::ExportPredictions {
            input: input.display().to_string(),
            output: output.display().to_string(),
        });
    }

    pub(super) fn ui_predict(&mut self, ui: &mut egui::Ui) {
        ui.heading("Прогноз");
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
                ui.label("Обучите модель в разделе «Обучение» или загрузите .bin.");
            }
            Some(info) => {
                let (n_in, n_out) = (info.schema.n_inputs(), info.schema.n_outputs());
                let (source, parameter_count) = (&info.source, info.parameter_count);
                ui.label(format!(
                    "Модель: {source} ({n_in} вход → {n_out} выход, {parameter_count} параметров)"
                ));
                if let Some(warning) = info.categorical_warning() {
                    ui.colored_label(egui::Color32::from_rgb(200, 120, 0), warning);
                }
                ui.separator();
                egui::Grid::new("predict_inputs")
                    .num_columns(2)
                    .show(ui, |ui| {
                        for (i, v) in self.predict_inputs.iter_mut().enumerate() {
                            let column = &info.schema.inputs()[i];
                            ui.label(column.display_name());
                            match column.ty() {
                                // Категория выбирается подписью: набрать код
                                // вручную нельзя, поэтому неизвестный уровень
                                // не может попасть в модель.
                                ColumnType::Categorical { levels } => {
                                    let code = (*v as usize).min(levels.len().saturating_sub(1));
                                    egui::ComboBox::from_id_salt(format!("cat_{i}"))
                                        .selected_text(&levels[code])
                                        .show_ui(ui, |ui| {
                                            for (c, level) in levels.iter().enumerate() {
                                                if ui.selectable_label(c == code, level).clicked() {
                                                    *v = c as f32;
                                                }
                                            }
                                        });
                                }
                                ColumnType::Numeric => {
                                    ui.add(egui::DragValue::new(v).speed(0.05));
                                }
                            }
                            ui.end_row();
                        }
                    });
                if ui
                    .add_enabled(!self.busy(), egui::Button::new("Рассчитать"))
                    .clicked()
                {
                    self.worker
                        .send(Command::Predict(self.predict_inputs.clone()));
                }

                if let Some(out) = &self.predict_outputs {
                    ui.separator();
                    for (i, v) in out.iter().enumerate() {
                        ui.label(format!(
                            "{} = {v:.6}",
                            info.schema.outputs()[i].display_name()
                        ));
                    }
                }
                if !self.extrapolation.is_empty() {
                    ui.separator();
                    let warn = egui::Color32::from_rgb(200, 120, 0);
                    ui.colored_label(warn, "⚠ экстраполяция – модель ненадёжна вне диапазона:");
                    for e in &self.extrapolation {
                        ui.colored_label(
                            warn,
                            format!(
                                "{} = {} вне [{}, {}]",
                                info.schema.inputs()[e.feature].display_name(),
                                e.value,
                                e.min,
                                e.max
                            ),
                        );
                    }
                }

                ui.separator();
                ui.label("Прогноз по таблице");
                ui.label(format!(
                    "Колонки входов ищутся по именам: {}. Колонки выходов ({}) \
                     заменяются, если есть, и добавляются, если нет; посторонние \
                     колонки переносятся как значения.",
                    info.schema.input_names().join(", "),
                    info.schema.output_names().join(", ")
                ));
                // Про потерю оформления говорим прямо: это экспорт, а не
                // правка исходной книги.
                ui.label(
                    "Результат — НОВАЯ книга только со значениями: стили, формулы, \
                     другие листы и структура исходного файла не сохраняются.",
                );
                if ui
                    .add_enabled(
                        !self.busy() && self.model_info.is_some(),
                        egui::Button::new("Экспортировать таблицу с прогнозами…"),
                    )
                    .clicked()
                {
                    self.export_predictions_dialog();
                }
            }
        }
    }
}
