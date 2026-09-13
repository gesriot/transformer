//! Выбор листа книги в интерфейсе.
//!
//! Книга с несколькими листами не читается молча: worker возвращает список
//! имён, а выбирает человек. Титульный лист, прошлогодние данные и рабочая
//! таблица выглядят в файле одинаково, и ошибка «взяли не тот лист» не видна
//! потом ни в одной метрике.
//!
//! Состояние своё у каждого файлового поля: книга, выбранная для разметки,
//! ничего не говорит о книге, выбранной для прогноза.

use eframe::egui;

/// Предложенный выбор листа для одного файлового поля.
#[derive(Default)]
pub(super) struct SheetChoice {
    /// Книга, к которой относится список. Пусто — выбирать нечего.
    path: String,
    sheets: Vec<String>,
    selected: Option<String>,
}

impl SheetChoice {
    /// Worker вернул список листов: дальше решает человек.
    pub(super) fn offer(&mut self, path: String, sheets: Vec<String>) {
        self.path = path;
        self.sheets = sheets;
        self.selected = None;
    }

    /// Выбор больше не относится к делу: выбран другой файл.
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Выбранный лист — только если он относится именно к этому файлу.
    /// Выбор от прошлой книги подставлять нельзя: имена листов совпадают
    /// случайно.
    pub(super) fn sheet_for(&self, path: &str) -> Option<&str> {
        if self.path != path {
            return None;
        }
        self.selected.as_deref()
    }

    /// Ждёт ли этот файл выбора листа.
    pub(super) fn awaiting(&self, path: &str) -> bool {
        self.path == path && !self.sheets.is_empty() && self.selected.is_none()
    }

    /// Список листов; рисуется только когда он относится к `path`.
    pub(super) fn ui(&mut self, ui: &mut egui::Ui, id: &'static str, path: &str) -> bool {
        if self.path != path || self.sheets.is_empty() {
            return false;
        }
        let before = self.selected.clone();
        ui.horizontal(|ui| {
            ui.label(format!("листов в книге: {}", self.sheets.len()));
            let mut picked = None;
            egui::ComboBox::from_id_salt(id)
                .selected_text(
                    self.selected
                        .clone()
                        .unwrap_or_else(|| "(выберите лист)".to_string()),
                )
                .show_ui(ui, |ui| {
                    for sheet in &self.sheets {
                        let chosen = self.selected.as_deref() == Some(sheet.as_str());
                        if ui.selectable_label(chosen, sheet).clicked() {
                            picked = Some(sheet.clone());
                        }
                    }
                });
            if picked.is_some() {
                self.selected = picked;
            }
        });
        self.selected != before
    }
}

/// Состояние выбора листа по всем файловым полям интерфейса.
///
/// Здесь же лежит то, что нужно повторить после выбора: команда уже уходила в
/// worker и вернулась вопросом, а её аргументы жили только в диалоге выбора
/// файла.
#[derive(Default)]
pub(super) struct SheetState {
    /// Таблица для разметки.
    pub(super) markup: SheetChoice,
    /// Вход конвертации в `.tnum`.
    pub(super) prepare: SheetChoice,
    /// Вход экспорта прогнозов.
    pub(super) export: SheetChoice,
    /// Запрошенная разметка: путь и наличие заголовка.
    pub(super) markup_request: Option<(String, bool)>,
    /// Запрошенный экспорт: вход и выход.
    pub(super) export_request: Option<(String, String)>,
}
