//! Экран данных: разметка таблицы и подготовка `.tnum`.

use super::messages::Command;
use super::messages::{DatasetOrigin, PreparedData};
use super::session::{ActiveDataset, App};
use crate::data::NumericDataset;
use crate::markup::{analyze_roles, DraftType, RoleReport, SchemaDraft, Severity, TableProfile};
use crate::schema::{ColumnRole, ModelSchema};
use crate::table::Table;
use crate::tnum::{infer_prepare_spec_from_path, parse_categorical, Delimiter, PrepareSpec};
use eframe::egui;
use std::sync::Arc;

/// Подтверждённая разметка таблицы: готовые данные и схема.
#[derive(Clone)]
pub(super) struct PreparedTable {
    pub(super) path: String,
    pub(super) has_header: bool,
    pub(super) data: Arc<NumericDataset>,
    pub(super) schema: ModelSchema,
}

/// Состояние диалога разметки. Профиль считается один раз при открытии файла;
/// пересчёт отчёта по ролям — только при смене ролей и типов.
pub(super) struct MarkupState {
    pub(super) path: String,
    pub(super) has_header: bool,
    pub(super) table: Table,
    pub(super) profile: TableProfile,
    pub(super) draft: SchemaDraft,
    pub(super) report: RoleReport,
    pub(super) issues: Vec<String>,
    /// Ошибка подтверждения: разметка валидна, но данные ей не соответствуют.
    pub(super) apply_error: Option<String>,
}

impl MarkupState {
    pub(super) fn new(
        path: String,
        has_header: bool,
        table: Table,
        profile: TableProfile,
        suggested_inputs: Option<usize>,
        suggested_categories: &[usize],
    ) -> Self {
        let mut draft = SchemaDraft::from_profile(&profile);
        // Автоопределение только заполняет начальные роли — решение за
        // пользователем, поэтому неудача здесь не является ошибкой.
        if let Some(n_inputs) = suggested_inputs {
            let _ = draft.set_output_split(n_inputs);
        }
        for &index in suggested_categories {
            let _ = draft.set_type(index, DraftType::Categorical);
        }
        let report = analyze_roles(&table, &draft);
        let issues = draft.issues();
        Self {
            path,
            has_header,
            table,
            profile,
            draft,
            report,
            issues,
            apply_error: None,
        }
    }

    /// Роли и типы меняют связи между колонками — отчёт пересчитывается.
    pub(super) fn on_roles_changed(&mut self) {
        self.report = analyze_roles(&self.table, &self.draft);
        self.on_any_change();
    }

    /// Имя и единица на анализ не влияют: отчёт хранит индексы и подставляет
    /// актуальные имена при выводе.
    pub(super) fn on_any_change(&mut self) {
        self.issues = self.draft.issues();
        self.apply_error = None;
    }

    /// Подтвердить разметку: данные превращаются в датасет ЗДЕСЬ, чтобы
    /// обучение получило готовую пару, а не путь к файлу.
    pub(super) fn apply(&self) -> Result<PreparedTable, String> {
        if let Some(message) = self
            .profile
            .messages()
            .into_iter()
            .find(|message| message.severity == Severity::Blocking)
        {
            return Err(message.text);
        }
        let schema = self.draft.finish()?;
        let data = self.table.to_dataset(&schema)?;
        Ok(PreparedTable {
            path: self.path.clone(),
            has_header: self.has_header,
            data: Arc::new(data),
            schema: schema.to_model_schema()?,
        })
    }

    pub(super) fn can_apply(&self) -> bool {
        self.issues.is_empty()
            && self.apply_error.is_none()
            && !self
                .profile
                .messages()
                .iter()
                .any(|message| message.severity == Severity::Blocking)
    }
}

pub(super) struct PrepareForm {
    pub(super) input_path: String,
    pub(super) output_path: String,
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    pub(super) delimiter: usize,
    pub(super) has_header: bool,
    pub(super) categorical: String,
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
    pub(super) fn build(&self) -> Result<(String, String, PrepareSpec), String> {
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

impl App {
    pub(super) fn apply_prepare_inference(&mut self) {
        if self.prepare_form.input_path.is_empty() {
            return;
        }
        match infer_prepare_spec_from_path(&self.prepare_form.input_path, Delimiter::Auto) {
            Ok(inferred) => {
                self.prepare_form.inputs = inferred.n_inputs;
                self.prepare_form.outputs = inferred.n_outputs;
                self.prepare_form.delimiter = match inferred.delimiter {
                    Delimiter::Auto => 0,
                    Delimiter::Comma => 1,
                    Delimiter::Tab => 2,
                    Delimiter::Space => 3,
                };
                self.prepare_form.has_header = inferred.has_header;
                self.prepare_form.categorical = inferred
                    .categorical
                    .iter()
                    .map(|(i, c)| format!("{i}:{c}"))
                    .collect::<Vec<_>>()
                    .join(",");
                self.status = format!(
                    "prepare auto: {} вход -> {} выход",
                    inferred.n_inputs, inferred.n_outputs
                );
            }
            Err(e) => {
                self.status = format!("prepare auto не сработал: {e}");
            }
        }
    }

    /// Диалог разметки. Для XLSX/CSV он показывается ВСЕГДА: автоопределение
    /// только заполняет начальное состояние, подтверждает роли пользователь.
    pub(super) fn ui_markup(&mut self, ctx: &egui::Context) {
        let Some(state) = &mut self.markup else {
            return;
        };
        let mut open = true;
        let mut close_after_apply = false;
        let mut reopen: Option<(String, bool)> = None;

        egui::Window::new("Разметка таблицы")
            .open(&mut open)
            .resizable(true)
            .default_width(760.0)
            .show(ctx, |ui| {
                ui.label(format!(
                    "{} — {} строк, {} колонок",
                    state.path,
                    state.profile.rows,
                    state.draft.len()
                ));
                let mut has_header = state.has_header;
                if ui
                    .checkbox(&mut has_header, "первая строка — заголовок")
                    .changed()
                {
                    // Заголовок меняет разбор файла, поэтому таблица читается
                    // заново — иначе имена и данные разъедутся.
                    reopen = Some((state.path.clone(), has_header));
                }

                ui.separator();
                let mut roles_changed = false;
                let mut other_changed = false;
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        egui::Grid::new("markup_grid")
                            .num_columns(5)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("колонка");
                                ui.label("роль");
                                ui.label("тип");
                                ui.label("единица");
                                ui.label("данные");
                                ui.end_row();

                                for i in 0..state.draft.len() {
                                    let column = state.draft.columns()[i].clone();
                                    let mut name = column.name.clone();
                                    if ui.text_edit_singleline(&mut name).changed() {
                                        let _ = state.draft.set_name(i, name);
                                        other_changed = true;
                                    }

                                    let mut role = column.role;
                                    egui::ComboBox::from_id_salt(format!("role_{i}"))
                                        .selected_text(role.label())
                                        .width(120.0)
                                        .show_ui(ui, |ui| {
                                            for candidate in [
                                                ColumnRole::Input,
                                                ColumnRole::Output,
                                                ColumnRole::Ignore,
                                            ] {
                                                ui.selectable_value(
                                                    &mut role,
                                                    candidate,
                                                    candidate.label(),
                                                );
                                            }
                                        });
                                    if role != column.role {
                                        let _ = state.draft.set_role(i, role);
                                        roles_changed = true;
                                    }

                                    let mut ty = column.ty;
                                    egui::ComboBox::from_id_salt(format!("type_{i}"))
                                        .selected_text(match ty {
                                            DraftType::Numeric => "число",
                                            DraftType::Categorical => "категория",
                                        })
                                        .width(110.0)
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut ty,
                                                DraftType::Numeric,
                                                "число",
                                            );
                                            ui.selectable_value(
                                                &mut ty,
                                                DraftType::Categorical,
                                                "категория",
                                            );
                                        });
                                    if ty != column.ty {
                                        match state.draft.set_type(i, ty) {
                                            Ok(()) => roles_changed = true,
                                            Err(e) => state.apply_error = Some(e),
                                        }
                                    }

                                    let mut unit = column.unit.clone().unwrap_or_default();
                                    if ui.text_edit_singleline(&mut unit).changed() {
                                        let trimmed = unit.trim().to_string();
                                        let _ = state
                                            .draft
                                            .set_unit(i, (!trimmed.is_empty()).then_some(trimmed));
                                        other_changed = true;
                                    }

                                    let p = &state.profile.columns[i];
                                    let distinct = p
                                        .n_distinct()
                                        .map(|n| n.to_string())
                                        .unwrap_or_else(|| "много".to_string());
                                    ui.label(format!(
                                        "различных {distinct}, пропусков {}, текст {}",
                                        p.missing, p.non_numeric
                                    ));
                                    ui.end_row();
                                }
                            });
                    });

                if roles_changed {
                    state.on_roles_changed();
                } else if other_changed {
                    state.on_any_change();
                }

                ui.separator();
                let blocking = egui::Color32::from_rgb(200, 60, 60);
                let warn = egui::Color32::from_rgb(200, 120, 0);
                for issue in &state.issues {
                    ui.colored_label(blocking, format!("✖ {issue}"));
                }
                if let Some(e) = &state.apply_error {
                    ui.colored_label(blocking, format!("✖ {e}"));
                }
                for message in state
                    .profile
                    .messages()
                    .into_iter()
                    .chain(state.report.messages(&state.draft))
                {
                    match message.severity {
                        Severity::Blocking => {
                            ui.colored_label(blocking, format!("✖ {}", message.text))
                        }
                        Severity::Warning => ui.colored_label(warn, format!("⚠ {}", message.text)),
                        Severity::Note => ui.label(format!("• {}", message.text)),
                    };
                }

                ui.separator();
                // При переключении заголовка текущий Table уже устарел, а
                // blocking-сообщения должны не только окрашиваться красным.
                let ready = state.can_apply() && reopen.is_none();
                if ui
                    .add_enabled(ready, egui::Button::new("Применить разметку"))
                    .clicked()
                {
                    match state.apply() {
                        Ok(prepared) => {
                            self.dataset = Some(ActiveDataset::new(
                                PreparedData {
                                    origin: DatasetOrigin::Table(prepared.path.clone()),
                                    data: Arc::clone(&prepared.data),
                                    schema: prepared.schema.clone(),
                                },
                                Some(state.profile.clone()),
                                prepared.has_header,
                            ));
                            self.status = "разметка применена".to_string();
                            close_after_apply = true;
                        }
                        Err(e) => state.apply_error = Some(e),
                    }
                }
            });

        if let Some((path, has_header)) = reopen {
            // Старую интерпретацию больше нельзя применить, пока worker читает
            // файл заново с другой семантикой первой строки.
            self.markup = None;
            self.open_table(path, has_header);
        } else if !open || close_after_apply {
            self.markup = None;
        }
    }

    pub(super) fn ui_prepare(&mut self, ui: &mut egui::Ui) {
        ui.heading("Prepare");
        ui.horizontal(|ui| {
            if ui.button("Вход…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter(
                        "tables",
                        &["csv", "tsv", "txt", "xlsx", "xlsm", "xlsb", "xls", "ods"],
                    )
                    .pick_file()
                {
                    self.prepare_form.input_path = p.display().to_string();
                    self.apply_prepare_inference();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::split::SplitPlan;
    use crate::table::Delimiter;

    fn markup(text: &str, suggested: Option<usize>) -> MarkupState {
        let table = Table::parse_text(text, Delimiter::Auto, true).unwrap();
        let profile = TableProfile::of(&table);
        MarkupState::new("t.csv".to_string(), true, table, profile, suggested, &[])
    }

    #[test]
    fn markup_applies_suggested_split_but_user_decides() {
        let mut state = markup("a,b,c\n1,2,3\n4,5,6\n", Some(2));
        let roles: Vec<ColumnRole> = state.draft.columns().iter().map(|c| c.role).collect();
        assert_eq!(
            roles,
            vec![ColumnRole::Input, ColumnRole::Input, ColumnRole::Output]
        );

        // Пользователь переопределяет подсказку.
        state.draft.set_role(1, ColumnRole::Ignore).unwrap();
        state.on_roles_changed();
        let prepared = state.apply().unwrap();
        assert_eq!(prepared.schema.input_names(), vec!["a"]);
        assert_eq!(prepared.schema.output_names(), vec!["c"]);
        assert_eq!(prepared.data.inputs.dim(), (2, 1));
    }

    /// Разметка становится активным набором данных сессии: в worker уходят
    /// данные и схема, а не путь, иначе он открыл бы файл заново и потерял
    /// ручную разметку.
    #[test]
    fn markup_result_becomes_the_active_dataset() {
        let mut state = markup("t,mat,y\n80,песок,1\n60,глина,2\n", Some(2));
        state.on_roles_changed();
        let prepared = state.apply().unwrap();

        let active = ActiveDataset::new(
            PreparedData {
                origin: DatasetOrigin::Table(prepared.path.clone()),
                data: Arc::clone(&prepared.data),
                schema: prepared.schema.clone(),
            },
            Some(state.profile.clone()),
            prepared.has_header,
        );

        assert_eq!(active.prepared.data.inputs.dim(), (2, 2));
        assert_eq!(active.prepared.schema.input_names(), vec!["t", "mat"]);
        // Категория распознана по подписям, коды воспроизводимы.
        assert_eq!(active.prepared.schema.inputs()[1].cardinality(), Some(2));
        assert_eq!(active.prepared.data.inputs[[0, 1]], 1.0); // песок — второй по алфавиту
                                                              // Разбиение — свойство набора данных.
        assert_eq!(active.split, SplitPlan::default());
        assert!(active.summary().contains("2 строк"));
    }

    #[test]
    fn blocking_issues_prevent_applying() {
        // Текстовая колонка назначена выходом — это блокирующая проблема.
        let mut state = markup("a,b\n1,x\n2,y\n", Some(1));
        assert!(!state.issues.is_empty());
        assert!(state.apply().is_err());

        state.draft.set_role(1, ColumnRole::Ignore).unwrap();
        state.draft.set_role(0, ColumnRole::Output).unwrap();
        state.on_roles_changed();
        // Без входов тоже нельзя.
        assert!(!state.issues.is_empty());
    }

    #[test]
    fn blocking_profile_message_disables_and_rejects_apply() {
        let table = Table::parse_text("a,b\n1,2\n3\n", Delimiter::Auto, true).unwrap();
        let profile = TableProfile::of(&table);
        let state = MarkupState::new("ragged.csv".to_string(), true, table, profile, Some(1), &[]);

        assert!(!state.can_apply());
        let error = match state.apply() {
            Err(error) => error,
            Ok(_) => panic!("таблица с рваными строками не должна применяться"),
        };
        assert!(error.contains("другим числом колонок"), "{error}");
    }

    #[test]
    fn categorical_suggestion_only_prefills_type() {
        let table =
            Table::parse_text("x,material_id,y\n1,0,2\n3,1,4\n", Delimiter::Auto, true).unwrap();
        let profile = TableProfile::of(&table);
        let mut state =
            MarkupState::new("coded.csv".to_string(), true, table, profile, Some(2), &[1]);
        assert_eq!(state.draft.columns()[1].ty, DraftType::Categorical);

        // Это подсказка, не запрет: пользователь может вернуть числовой тип.
        state.draft.set_type(1, DraftType::Numeric).unwrap();
        state.on_roles_changed();
        assert_eq!(state.draft.columns()[1].ty, DraftType::Numeric);
        assert!(state.can_apply());
    }

    /// Отчёт по ролям пересчитывается при смене ролей и не зависит от имён.
    #[test]
    fn report_tracks_roles_and_ignores_renames() {
        let mut text = String::from("x0,x1,x2,y\n");
        for i in 0..30 {
            let x0 = 2.0 + i as f64;
            let x1 = 5.0 + (i % 4) as f64;
            text.push_str(&format!("{x0},{x1},{},{}\n", 100.0 - x0 - x1, i));
        }
        let mut state = markup(&text, Some(3));
        assert_eq!(state.report.dependencies.len(), 1);

        // Переименование не меняет отчёт, но имя в сообщении обновляется.
        state.draft.set_name(0, "доля A").unwrap();
        state.on_any_change();
        assert_eq!(state.report.dependencies.len(), 1);
        let text = state
            .report
            .messages(&state.draft)
            .into_iter()
            .map(|m| m.text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("доля A"), "{text}");

        // Исключение колонки убирает связь.
        state.draft.set_role(2, ColumnRole::Ignore).unwrap();
        state.on_roles_changed();
        assert!(state.report.dependencies.is_empty());
    }
}
