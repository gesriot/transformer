//! Демонстрация char-level языковой модели. С численным сценарием не связана;
//! в основной навигации ей не место — она остаётся примером архитектуры.

use super::messages::Command;
use super::messages::DatasetOrigin;
use super::session::{App, Section, BLACKBOXES};
use crate::config::ModelConfig;
use crate::train::TextTrainConfig;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub(super) struct TextForm {
    pub(super) file_path: String,
    pub(super) d_model: usize,
    pub(super) heads: usize,
    pub(super) layers: usize,
    pub(super) d_ff: usize,
    pub(super) steps: usize,
    pub(super) batch: usize,
    pub(super) ctx_len: usize,
    pub(super) tgt_len: usize,
    pub(super) lr: f32,
    pub(super) seed: u64,
    pub(super) seed_text: String,
    pub(super) total_new: usize,
    pub(super) temperature: f32,
    pub(super) top_k: usize,
    pub(super) gen_seed: u64,
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
    pub(super) fn build(&self) -> Result<(String, ModelConfig, TextTrainConfig), String> {
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

impl App {
    /// Раздел «Демо»: встроенные задачи и char-LM. К рабочему сценарию не
    /// относятся и живут отдельно, чтобы не выглядеть его частью.
    pub(super) fn ui_demo(&mut self, ui: &mut egui::Ui) {
        ui.heading("Демо");
        ui.label(
            "Встроенные задачи — источник данных для проверки самого приложения, \
             а не рабочий сценарий.",
        );
        ui.horizontal(|ui| {
            ui.label("Открыть как данные:");
            for &name in BLACKBOXES {
                if ui
                    .add_enabled(!self.busy(), egui::Button::new(name))
                    .clicked()
                {
                    self.open_dataset(DatasetOrigin::Blackbox(name.to_string()));
                    self.section = Section::Data;
                }
            }
        });
        ui.separator();
        self.ui_text(ui);
    }

    fn ui_text(&mut self, ui: &mut egui::Ui) {
        ui.heading("Char-level LM");

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
                .add_enabled(!self.busy(), egui::Button::new("Обучить text-модель"))
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
                .add_enabled(self.text_training, egui::Button::new("Отмена"))
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
}
