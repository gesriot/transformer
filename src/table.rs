//! Чтение исходных таблиц: CSV, TSV, пробельные и Excel.
//!
//! Слой намеренно НИЧЕГО не интерпретирует: он не решает, где входы, а где
//! выходы, не превращает текст в числа и не отбрасывает пустые ячейки. Всё это
//! делает разметка ([`TableSchema`]) на следующем шаге.
//!
//! Причина в том, что потерянную ячейку потом не восстановить: без пустых
//! ячеек невозможен отчёт о пропусках, а без исходного текста — категории по
//! строковым подписям. Поэтому единственный путь к данным такой:
//!
//! ```text
//! файл -> Table -> (Table + TableSchema) -> NumericDataset
//! ```
//!
//! `.tnum` в эту цепочку не входит: он уже содержит и данные, и схему, поэтому
//! остаётся отдельным готовым источником.

use crate::data::NumericDataset;
use crate::schema::{ColumnRole, ColumnType, TableSchema};
use calamine::{open_workbook_auto, Data, Reader, SheetType};
use ndarray::Array2;
use std::path::Path;

type LocatedRows = Vec<(usize, Vec<String>)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delimiter {
    Auto,
    Comma,
    Tab,
    Space,
}

/// Прочитанная таблица: заголовок (если есть) и ячейки как текст.
///
/// Пустая ячейка — пустая строка, а не пропуск столбца: `1,,3` остаётся тремя
/// колонками, иначе пропуск молча сдвинул бы данные соседних колонок.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    source: String,
    /// Выбранный worksheet книги; `None` у текстового источника.
    sheet: Option<String>,
    header: Option<Vec<String>>,
    rows: Vec<Vec<String>>,
    // Номер каждой строки в исходном файле. Простого `index + 1` недостаточно:
    // комментарии и пустые строки между записями не попадают в `rows`.
    row_numbers: Vec<usize>,
}

/// Книга ли это: решается по расширению, как и выбор читателя.
pub fn is_workbook(path: impl AsRef<Path>) -> bool {
    matches!(
        path.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("xlsx" | "xlsm" | "xlsb" | "xls" | "ods")
    )
}

/// Имена табличных листов книги в порядке самой книги.
///
/// Отдельная операция, потому что выбор листа делается ДО чтения данных: иначе
/// пришлось бы угадывать лист. Chart/dialog/macro sheets не возвращаются:
/// считать из них таблицу значений этот слой всё равно не может.
pub fn workbook_sheets(path: impl AsRef<Path>) -> Result<Vec<String>, String> {
    let path = path.as_ref();
    if !is_workbook(path) {
        return Err(format!(
            "{}: это не книга Excel/ODS, листов у неё нет",
            path.display()
        ));
    }
    let workbook =
        open_workbook_auto(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
    Ok(worksheet_names(workbook.sheets_metadata()))
}

fn worksheet_names(sheets: &[calamine::Sheet]) -> Vec<String> {
    sheets
        .iter()
        .filter(|sheet| sheet.typ == SheetType::WorkSheet)
        .map(|sheet| sheet.name.clone())
        .collect()
}

impl Table {
    /// Прочитать таблицу, не выбирая лист.
    ///
    /// У книги с единственным листом выбирать нечего; книга с несколькими
    /// листами — ошибка, см. [`Table::read_sheet`].
    pub fn read_path(
        path: impl AsRef<Path>,
        delimiter: Delimiter,
        has_header: bool,
    ) -> Result<Self, String> {
        Self::read_sheet(path, None, delimiter, has_header)
    }

    /// Прочитать конкретный лист книги.
    ///
    /// `sheet: None` означает «лист не выбран»: у книги с одним листом это тот
    /// самый лист, у книги с несколькими — ошибка со списком имён. Молча брать
    /// первый нельзя: в реальных книгах первым часто лежит титульный лист или
    /// прошлогодние данные, и такая ошибка не видна ни в одной метрике.
    ///
    /// У текстовой таблицы листов нет, поэтому имя для неё — тоже ошибка, а не
    /// игнорируемый аргумент.
    pub fn read_sheet(
        path: impl AsRef<Path>,
        sheet: Option<&str>,
        delimiter: Delimiter,
        has_header: bool,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let source = path.display().to_string();
        let (sheet, rows) = if is_workbook(path) {
            let (sheet, rows) = read_workbook(path, sheet)?;
            (Some(sheet), rows)
        } else {
            if let Some(sheet) = sheet {
                return Err(format!(
                    "{source}: лист '{sheet}' указан для текстовой таблицы, а листы есть только у книг"
                ));
            }
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("чтение {source}: {e}"))?;
            (
                None,
                split_text(&text, delimiter).map_err(|e| format!("{source}: {e}"))?,
            )
        };
        Self::from_rows(source, sheet, rows, has_header)
    }

    pub fn parse_text(text: &str, delimiter: Delimiter, has_header: bool) -> Result<Self, String> {
        Self::from_rows(
            "<текст>".to_string(),
            None,
            split_text(text, delimiter).map_err(|e| format!("<текст>: {e}"))?,
            has_header,
        )
    }

    fn from_rows(
        source: String,
        sheet: Option<String>,
        mut located_rows: LocatedRows,
        has_header: bool,
    ) -> Result<Self, String> {
        let source_label = source_label(&source, sheet.as_deref());
        if located_rows.is_empty() {
            return Err(format!("{source_label}: нет строк данных"));
        }
        let header = if has_header {
            let (_, header) = located_rows.remove(0);
            if located_rows.is_empty() {
                return Err(format!("{source_label}: нет строк данных после заголовка"));
            }
            Some(header)
        } else {
            None
        };
        let (row_numbers, rows) = located_rows.into_iter().unzip();
        Ok(Self {
            source,
            sheet,
            header,
            rows,
            row_numbers,
        })
    }

    /// Считать первую строку уже прочитанной таблицы заголовком.
    ///
    /// Авторазметка сначала смотрит на первую строку как на данные. После
    /// распознавания не нужно читать изменяемый файл второй раз.
    pub(crate) fn promote_first_row_to_header(mut self) -> Result<Self, String> {
        if self.header.is_some() {
            return Ok(self);
        }
        if self.rows.len() < 2 {
            return Err(format!(
                "{}: нет строк данных после заголовка",
                self.source_label()
            ));
        }
        self.header = Some(self.rows.remove(0));
        self.row_numbers.remove(0);
        Ok(self)
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Выбранный worksheet книги; у текстовой таблицы листа нет.
    pub fn sheet(&self) -> Option<&str> {
        self.sheet.as_deref()
    }

    /// Источник для сообщений человеку: имя листа нельзя терять рядом с путём.
    pub fn source_label(&self) -> String {
        source_label(&self.source, self.sheet())
    }

    pub fn header(&self) -> Option<&[String]> {
        self.header.as_deref()
    }

    pub fn rows(&self) -> &[Vec<String>] {
        &self.rows
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Число колонок: по заголовку, иначе по первой строке данных. Строки
    /// другой ширины не отбраковываются здесь — это работа разметки, которая
    /// знает, сколько колонок ожидается.
    pub fn n_columns(&self) -> usize {
        self.header
            .as_ref()
            .map(Vec::len)
            .or_else(|| self.rows.first().map(Vec::len))
            .unwrap_or(0)
    }

    /// Единственный путь от таблицы к данным модели.
    ///
    /// Игнорируемые колонки пропускаются; категориальные значения переводятся в
    /// коды строго по ПОДПИСЯМ уровней. Старые таблицы с числовыми кодами
    /// подключаются отдельным явным адаптером.
    pub fn to_dataset(&self, schema: &TableSchema) -> Result<NumericDataset, String> {
        self.to_dataset_impl(schema, &[])
    }

    /// Совместимость с `PrepareSpec`, где категории задавались числовыми
    /// кодами, а подписи уровней отсутствовали. Разрешение передаётся явно:
    /// по самим подписям `"0"…"n-1"` нельзя понять, настоящие это названия
    /// категорий или синтетические коды старого формата.
    pub(crate) fn to_dataset_with_category_codes(
        &self,
        schema: &TableSchema,
        code_columns: &[usize],
    ) -> Result<NumericDataset, String> {
        self.to_dataset_impl(schema, code_columns)
    }

    fn to_dataset_impl(
        &self,
        schema: &TableSchema,
        code_columns: &[usize],
    ) -> Result<NumericDataset, String> {
        let columns = schema.columns();
        if columns.len() != self.n_columns() {
            return Err(format!(
                "{}: схема описывает {} колонок, в таблице {}",
                self.source_label(),
                columns.len(),
                self.n_columns()
            ));
        }
        let input_idx = schema.indices(ColumnRole::Input);
        let output_idx = schema.indices(ColumnRole::Output);

        let mut inputs = Array2::<f32>::zeros((self.rows.len(), input_idx.len()));
        let mut outputs = Array2::<f32>::zeros((self.rows.len(), output_idx.len()));
        for (r, row) in self.rows.iter().enumerate() {
            if row.len() != columns.len() {
                return Err(format!(
                    "{}: строка {}: ожидалось {} колонок, получено {}",
                    self.source_label(),
                    self.file_row(r),
                    columns.len(),
                    row.len()
                ));
            }
            for (slot, &c) in input_idx.iter().enumerate() {
                inputs[[r, slot]] =
                    self.parse_cell(schema, r, c, &row[c], code_columns.contains(&c))?;
            }
            for (slot, &c) in output_idx.iter().enumerate() {
                outputs[[r, slot]] = self.parse_cell(schema, r, c, &row[c], false)?;
            }
        }
        Ok(NumericDataset::new(inputs, outputs))
    }

    /// Номер строки данных `r` так, как она пронумерована в файле (1-based, с
    /// учётом заголовка, комментариев и пустых строк) — иначе пользователь ищет
    /// ошибку не там.
    ///
    /// # Panics
    ///
    /// Если `r >= self.n_rows()`.
    pub fn file_row(&self, r: usize) -> usize {
        self.row_numbers[r]
    }

    fn parse_cell(
        &self,
        schema: &TableSchema,
        r: usize,
        c: usize,
        text: &str,
        allow_category_code: bool,
    ) -> Result<f32, String> {
        let column = &schema.columns()[c];
        let at = format!("{}: строка {}", self.source_label(), self.file_row(r));
        let where_ = format!("{at}, колонка '{}'", column.name());
        if text.trim().is_empty() {
            return Err(format!("{where_}: пустая ячейка"));
        }
        if let ColumnType::Categorical { levels } = column.ty() {
            if let Ok(code) = column.category_code(text) {
                return Ok(code as f32);
            }
            if allow_category_code {
                if let Ok(raw) = text.trim().parse::<f32>() {
                    let rounded = raw.round();
                    if raw.is_finite() && (raw - rounded).abs() < 1e-4 && rounded >= 0.0 {
                        let code = rounded as usize;
                        if code < levels.len() {
                            return Ok(code as f32);
                        }
                        return Err(format!(
                            "{where_}: категория {code} вне [0, {})",
                            levels.len()
                        ));
                    }
                    return Err(format!(
                        "{where_}: код категории должен быть целым, получено {raw}"
                    ));
                }
            }
            // Ошибка схемы уже называет колонку — не дублируем её в префиксе.
            return column
                .category_code(text)
                .map(|code| code as f32)
                .map_err(|e| format!("{at}: {e}"));
        }
        let value: f32 = text
            .trim()
            .parse()
            .map_err(|_| format!("{where_}: не число: '{text}'"))?;
        if !value.is_finite() {
            return Err(format!("{where_}: значение не конечно: '{text}'"));
        }
        Ok(value)
    }
}

fn source_label(source: &str, sheet: Option<&str>) -> String {
    match sheet {
        Some(sheet) => format!("{source}, лист '{sheet}'"),
        None => source.to_string(),
    }
}

fn detect_delim(line: &str, mode: Delimiter) -> Option<char> {
    match mode {
        Delimiter::Comma => Some(','),
        Delimiter::Tab => Some('\t'),
        Delimiter::Space => None,
        Delimiter::Auto => {
            if line.contains(',') {
                Some(',')
            } else if line.contains('\t') {
                Some('\t')
            } else {
                None
            }
        }
    }
}

/// Текстовая таблица в ячейки. Комментарии после `#` и пустые строки
/// отбрасываются, но пустые ЯЧЕЙКИ сохраняются.
fn split_text(input: &str, mode: Delimiter) -> Result<Vec<(usize, Vec<String>)>, String> {
    let lines: Vec<(usize, &str)> = input
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let before_comment = line.split('#').next().unwrap_or("");
            (i + 1, before_comment.trim_end_matches('\r'))
        })
        .filter(|(_, line)| !line.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err("нет строк данных".to_string());
    }
    let delim = detect_delim(lines[0].1, mode);
    Ok(lines
        .into_iter()
        .map(|(number, line)| {
            let cells = match delim {
                // Всю строку не тримим: у TSV начальный/конечный tab означает
                // пустую ячейку и должен сохраниться.
                Some(d) => line.split(d).map(|t| t.trim().to_string()).collect(),
                // У пробельного разделителя пустая ячейка непредставима.
                None => line.split_whitespace().map(str::to_string).collect(),
            };
            (number, cells)
        })
        .collect())
}

fn cell_to_text(cell: &Data, row: usize, col: usize) -> Result<String, String> {
    match cell {
        Data::Empty => Ok(String::new()),
        Data::String(s) => Ok(s.trim().to_string()),
        Data::Float(v) => {
            if v.is_finite() {
                Ok(format!("{v}"))
            } else {
                Err(format!("строка {row}, колонка {col}: значение не конечно"))
            }
        }
        Data::Int(v) => Ok(v.to_string()),
        Data::Bool(v) => Ok(if *v { "1".to_string() } else { "0".to_string() }),
        Data::DateTime(_) | Data::DateTimeIso(_) | Data::DurationIso(_) => Err(format!(
            "строка {row}, колонка {col}: даты/время в .xlsx не поддерживаются как числовые данные"
        )),
        Data::Error(e) => Err(format!("строка {row}, колонка {col}: ошибка Excel {e}")),
    }
}

/// Какой лист читать: выбранный по имени либо единственный.
fn choose_sheet(path: &Path, names: &[String], wanted: Option<&str>) -> Result<String, String> {
    let listed = || names.join(", ");
    match wanted {
        Some(name) => names
            .iter()
            .find(|sheet| sheet.as_str() == name)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "{}: листа '{name}' в книге нет. Есть: {}",
                    path.display(),
                    listed()
                )
            }),
        None => match names {
            [] => Err(format!("{}: книга без табличных листов", path.display())),
            [only] => Ok(only.clone()),
            _ => Err(format!(
                "{}: в книге несколько листов: {}. Укажите лист явно: брать первый молча нельзя",
                path.display(),
                listed()
            )),
        },
    }
}

fn read_workbook(path: &Path, wanted: Option<&str>) -> Result<(String, LocatedRows), String> {
    let mut workbook =
        open_workbook_auto(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
    let sheet = choose_sheet(path, &worksheet_names(workbook.sheets_metadata()), wanted)?;
    let range = workbook
        .worksheet_range(&sheet)
        .map_err(|e| format!("чтение {} листа '{sheet}': {e}", path.display()))?;

    let (start_row, start_col) = range.start().unwrap_or((0, 0));
    let mut rows = Vec::new();
    for (r, row) in range.rows().enumerate() {
        let source_row = start_row as usize + r + 1;
        let cells = row
            .iter()
            .enumerate()
            .map(|(c, cell)| cell_to_text(cell, source_row, start_col as usize + c + 1))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}, лист '{sheet}': {e}", path.display()))?;
        if cells.iter().any(|t| !t.trim().is_empty()) {
            // Внутренние и хвостовые пустые ячейки сохраняются. `Range` уже
            // ограничен используемой областью листа; удалив хвост, мы бы
            // превратили пропуск последнего признака в «рваную строку».
            rows.push((source_row, cells));
        }
    }
    if rows.is_empty() {
        return Err(format!(
            "{}, лист '{sheet}': нет строк данных",
            path.display()
        ));
    }
    Ok((sheet, rows))
}

/// Тестовая книга с несколькими листами.
///
/// Живёт рядом с чтением, а не в тестах одного модуля: такой файл нужен и
/// конвертации, и экспорту, и worker-у, а три копии одного XML разойдутся.
#[cfg(test)]
pub(crate) fn write_test_workbook(path: &Path, sheets: &[(&str, &[&[&str]])]) {
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    let file = std::fs::File::create(path).unwrap();
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let part = |zip: &mut ZipWriter<std::fs::File>, name: &str, xml: &str| {
        zip.start_file(name, options).unwrap();
        zip.write_all(xml.as_bytes()).unwrap();
    };

    let mut types = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>"#,
    );
    let mut book = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>"#,
    );
    let mut rels = String::from(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
    );
    for (i, (name, rows)) in sheets.iter().enumerate() {
        let n = i + 1;
        types.push_str(&format!(
            r#"<Override PartName="/xl/worksheets/sheet{n}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>"#
        ));
        book.push_str(&format!(
            r#"<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>"#
        ));
        rels.push_str(&format!(
            r#"<Relationship Id="rId{n}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{n}.xml"/>"#
        ));

        let mut sheet = String::from(
            r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>"#,
        );
        for (r, row) in rows.iter().enumerate() {
            sheet.push_str(&format!(r#"<row r="{}">"#, r + 1));
            for (c, cell) in row.iter().enumerate() {
                let col = (b'A' + c as u8) as char;
                sheet.push_str(&format!(
                    r#"<c r="{col}{}" t="inlineStr"><is><t>{cell}</t></is></c>"#,
                    r + 1
                ));
            }
            sheet.push_str("</row>");
        }
        sheet.push_str("</sheetData></worksheet>");
        part(&mut zip, &format!("xl/worksheets/sheet{n}.xml"), &sheet);
    }
    types.push_str("</Types>");
    book.push_str("</sheets></workbook>");
    rels.push_str("</Relationships>");

    part(&mut zip, "[Content_Types].xml", &types);
    part(
        &mut zip,
        "_rels/.rels",
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#,
    );
    part(&mut zip, "xl/workbook.xml", &book);
    part(&mut zip, "xl/_rels/workbook.xml.rels", &rels);
    zip.finish().unwrap();
}

#[cfg(test)]
mod tests {
    use super::write_test_workbook as write_workbook;
    use super::*;
    use crate::schema::{Column, ColumnRole, TableSchema};
    use calamine::{Sheet, SheetVisible};

    fn tmp_book(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("transformer_sheets_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Единственный лист — выбирать нечего, и спрашивать не о чем.
    #[test]
    fn a_single_sheet_workbook_reads_without_a_choice() {
        let path = tmp_book("one.xlsx");
        write_workbook(&path, &[("Данные", &[&["x0", "y0"], &["1", "2"]])]);

        let table = Table::read_path(&path, Delimiter::Auto, true).unwrap();
        assert_eq!(table.sheet(), Some("Данные"));
        assert_eq!(
            table.header().unwrap(),
            &["x0".to_string(), "y0".to_string()]
        );
        assert_eq!(table.rows()[0], vec!["1", "2"]);
        assert_eq!(workbook_sheets(&path).unwrap(), vec!["Данные".to_string()]);
        std::fs::remove_file(&path).ok();
    }

    /// Несколько листов без выбора — ошибка со списком, а не первый лист.
    /// Титульный лист и рабочая таблица в файле выглядят одинаково.
    #[test]
    fn a_multi_sheet_workbook_refuses_to_guess() {
        let path = tmp_book("many.xlsx");
        write_workbook(
            &path,
            &[
                ("Титульный", &[&["отчёт за год"]]),
                ("Опыты", &[&["x0", "y0"], &["3", "4"]]),
            ],
        );

        let err = Table::read_path(&path, Delimiter::Auto, true).unwrap_err();
        assert!(err.contains("Титульный"), "{err}");
        assert!(err.contains("Опыты"), "{err}");
        assert!(err.contains("молча"), "{err}");

        assert_eq!(
            workbook_sheets(&path).unwrap(),
            vec!["Титульный".to_string(), "Опыты".to_string()],
            "порядок — как в книге"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_named_sheet_is_read_and_an_unknown_one_lists_the_others() {
        let path = tmp_book("named.xlsx");
        write_workbook(
            &path,
            &[
                ("Титульный", &[&["отчёт за год"]]),
                ("Опыты", &[&["x0", "y0"], &["3", "4"]]),
            ],
        );

        let table = Table::read_sheet(&path, Some("Опыты"), Delimiter::Auto, true).unwrap();
        assert_eq!(table.sheet(), Some("Опыты"));
        assert!(table.source_label().contains("лист 'Опыты'"));
        assert_eq!(table.rows()[0], vec!["3", "4"]);

        let err = Table::read_sheet(&path, Some("опыты"), Delimiter::Auto, true).unwrap_err();
        assert!(err.contains("Титульный, Опыты"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    /// У текста листов нет: имя листа для него — ошибка, а не игнорируемый
    /// аргумент.
    #[test]
    fn a_text_table_has_no_sheets() {
        let path = tmp_book("plain.csv");
        std::fs::write(&path, "x0,y0\n1,2\n").unwrap();

        assert!(Table::read_path(&path, Delimiter::Auto, true).is_ok());
        let err = Table::read_sheet(&path, Some("Лист1"), Delimiter::Auto, true).unwrap_err();
        assert!(err.contains("листы есть только у книг"), "{err}");
        assert!(workbook_sheets(&path).is_err());
        assert!(!is_workbook(&path));
        assert_eq!(
            Table::read_path(&path, Delimiter::Auto, true)
                .unwrap()
                .sheet(),
            None
        );
        std::fs::remove_file(&path).ok();
    }

    /// Chart/dialog sheets Excel называет листами, но таблицу данных из них
    /// прочитать нельзя. Они не должны превращать одну таблицу в ложную
    /// неоднозначность и не должны появляться среди вариантов GUI/CLI.
    #[test]
    fn only_worksheets_are_offered_as_data_sources() {
        let sheets = vec![
            Sheet {
                name: "Данные".to_string(),
                typ: SheetType::WorkSheet,
                visible: SheetVisible::Visible,
            },
            Sheet {
                name: "Диаграмма".to_string(),
                typ: SheetType::ChartSheet,
                visible: SheetVisible::Visible,
            },
            Sheet {
                name: "Скрытые данные".to_string(),
                typ: SheetType::WorkSheet,
                visible: SheetVisible::Hidden,
            },
        ];
        assert_eq!(
            worksheet_names(&sheets),
            vec!["Данные".to_string(), "Скрытые данные".to_string()]
        );
    }

    fn schema(cols: Vec<Column>) -> TableSchema {
        TableSchema::new(cols).unwrap()
    }

    fn numeric(name: &str, role: ColumnRole) -> Column {
        Column::numeric(name, role).unwrap()
    }

    #[test]
    fn keeps_empty_cells_instead_of_shifting_columns() {
        let t = Table::parse_text("a,b,c\n1,,3\n", Delimiter::Auto, true).unwrap();
        assert_eq!(t.n_columns(), 3);
        assert_eq!(t.rows()[0], vec!["1", "", "3"]);

        // Пропуск виден как пустая ячейка, а не как «строка короче».
        let s = schema(vec![
            numeric("a", ColumnRole::Input),
            numeric("b", ColumnRole::Input),
            numeric("c", ColumnRole::Output),
        ]);
        let err = t.to_dataset(&s).unwrap_err();
        assert!(err.contains("пустая ячейка"), "{err}");
        assert!(err.contains("колонка 'b'"), "{err}");
        assert!(err.contains("строка 2"), "номер строки в файле: {err}");
    }

    #[test]
    fn header_is_optional() {
        let with = Table::parse_text("a,b\n1,2\n", Delimiter::Auto, true).unwrap();
        assert_eq!(
            with.header(),
            Some(["a".to_string(), "b".to_string()].as_slice())
        );
        assert_eq!(with.n_rows(), 1);

        let without = Table::parse_text("1,2\n3,4\n", Delimiter::Auto, false).unwrap();
        assert!(without.header().is_none());
        assert_eq!(without.n_rows(), 2);
        assert_eq!(without.n_columns(), 2);
    }

    #[test]
    fn ignored_columns_are_dropped_and_order_is_kept() {
        let t = Table::parse_text(
            "temp,note,mat,moisture\n80,ok,глина,12.5\n60,-,песок,18\n",
            Delimiter::Auto,
            true,
        )
        .unwrap();
        let s = schema(vec![
            numeric("temp", ColumnRole::Input),
            numeric("note", ColumnRole::Ignore),
            Column::categorical(
                "mat",
                ColumnRole::Input,
                vec!["песок".into(), "глина".into()],
            )
            .unwrap(),
            numeric("moisture", ColumnRole::Output),
        ]);
        let ds = t.to_dataset(&s).unwrap();
        assert_eq!(ds.inputs.dim(), (2, 2));
        assert_eq!(ds.outputs.dim(), (2, 1));
        // Игнорируемая колонка с нечисловым текстом не мешает.
        assert_eq!(ds.inputs[[0, 0]], 80.0);
        assert_eq!(ds.outputs[[1, 0]], 18.0);
    }

    #[test]
    fn categories_accept_labels_and_reject_unknown() {
        let t = Table::parse_text("mat,y\nглина,1\nпесок,2\n", Delimiter::Auto, true).unwrap();
        let s = schema(vec![
            Column::categorical(
                "mat",
                ColumnRole::Input,
                vec!["песок".into(), "глина".into()],
            )
            .unwrap(),
            numeric("y", ColumnRole::Output),
        ]);
        let ds = t.to_dataset(&s).unwrap();
        assert_eq!(ds.inputs[[0, 0]], 1.0); // глина -> код 1
        assert_eq!(ds.inputs[[1, 0]], 0.0); // песок -> код 0

        let bad = Table::parse_text("mat,y\nгранит,1\n", Delimiter::Auto, true).unwrap();
        let err = bad.to_dataset(&s).unwrap_err();
        assert!(err.contains("гранит"), "{err}");
        assert!(err.contains("песок, глина"), "{err}");
    }

    /// Совместимость включается явно адаптером PrepareSpec: по подписям
    /// `"0", "1", ...` невозможно определить, являются ли они кодами.
    #[test]
    fn explicitly_enabled_numeric_codes_are_accepted() {
        let s = schema(vec![
            Column::categorical(
                "mat",
                ColumnRole::Input,
                vec!["0".into(), "1".into(), "2".into()],
            )
            .unwrap(),
            numeric("y", ColumnRole::Output),
        ]);
        let t = Table::parse_text("mat,y\n1.0,5\n2,6\n", Delimiter::Auto, true).unwrap();
        let ds = t.to_dataset_with_category_codes(&s, &[0]).unwrap();
        assert_eq!(ds.inputs[[0, 0]], 1.0);
        assert_eq!(ds.inputs[[1, 0]], 2.0);

        let fractional = Table::parse_text("mat,y\n0.5,5\n", Delimiter::Auto, true).unwrap();
        assert!(fractional
            .to_dataset_with_category_codes(&s, &[0])
            .unwrap_err()
            .contains("целым"));
        let out_of_range = Table::parse_text("mat,y\n7,5\n", Delimiter::Auto, true).unwrap();
        assert!(out_of_range
            .to_dataset_with_category_codes(&s, &[0])
            .unwrap_err()
            .contains("вне [0, 3)"));
    }

    #[test]
    fn numeric_level_names_do_not_implicitly_enable_codes() {
        let s = schema(vec![
            Column::categorical(
                "rating",
                ColumnRole::Input,
                vec!["0".into(), "1".into(), "2".into()],
            )
            .unwrap(),
            numeric("y", ColumnRole::Output),
        ]);
        // Точная подпись по-прежнему допустима.
        let label = Table::parse_text("rating,y\n1,5\n", Delimiter::Auto, true).unwrap();
        assert_eq!(label.to_dataset(&s).unwrap().inputs[[0, 0]], 1.0);

        // Но `1.0` не становится кодом только из-за вида списка уровней.
        let code = Table::parse_text("rating,y\n1.0,5\n", Delimiter::Auto, true).unwrap();
        assert!(code
            .to_dataset(&s)
            .unwrap_err()
            .contains("неизвестный уровень '1.0'"));
    }

    /// При настоящих подписях числовой код НЕ принимается: это защита от
    /// перепутанных колонок.
    #[test]
    fn numeric_codes_are_rejected_when_levels_have_labels() {
        let s = schema(vec![
            Column::categorical(
                "mat",
                ColumnRole::Input,
                vec!["песок".into(), "глина".into()],
            )
            .unwrap(),
            numeric("y", ColumnRole::Output),
        ]);
        let t = Table::parse_text("mat,y\n1,5\n", Delimiter::Auto, true).unwrap();
        let err = t.to_dataset(&s).unwrap_err();
        assert!(err.contains("неизвестный уровень '1'"), "{err}");
    }

    #[test]
    fn schema_width_and_row_width_are_checked() {
        let t = Table::parse_text("a,b\n1,2\n", Delimiter::Auto, true).unwrap();
        let narrow = schema(vec![
            numeric("a", ColumnRole::Input),
            numeric("b", ColumnRole::Output),
        ]);
        assert!(t.to_dataset(&narrow).is_ok());

        let wide = schema(vec![
            numeric("a", ColumnRole::Input),
            numeric("b", ColumnRole::Input),
            numeric("c", ColumnRole::Output),
        ]);
        let err = t.to_dataset(&wide).unwrap_err();
        assert!(err.contains("схема описывает 3"), "{err}");

        // Рваная строка ловится с номером строки файла.
        let ragged = Table::parse_text("a,b\n1,2\n3\n", Delimiter::Auto, true).unwrap();
        let err = ragged.to_dataset(&narrow).unwrap_err();
        assert!(err.contains("строка 3"), "{err}");
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let t = Table::parse_text(
            "a,b\n# комментарий\n1,2\n\n3,4  # хвост\n",
            Delimiter::Auto,
            true,
        )
        .unwrap();
        assert_eq!(t.n_rows(), 2);
        assert_eq!(t.rows()[1], vec!["3", "4"]);

        let s = schema(vec![
            numeric("a", ColumnRole::Input),
            numeric("b", ColumnRole::Output),
        ]);
        let bad = Table::parse_text(
            "a,b\n# комментарий\n1,2\n\nnot-a-number,4\n",
            Delimiter::Auto,
            true,
        )
        .unwrap();
        let err = bad.to_dataset(&s).unwrap_err();
        assert!(
            err.contains("строка 5"),
            "номер строки исходного файла: {err}"
        );
    }

    #[test]
    fn tsv_keeps_empty_edge_cells() {
        let t = Table::parse_text("a\tb\tc\n\t2\t\n", Delimiter::Tab, true).unwrap();
        assert_eq!(t.rows()[0], vec!["", "2", ""]);
    }

    #[test]
    fn delimiters_are_detected_and_forced() {
        let tab = Table::parse_text("a\tb\n1\t2\n", Delimiter::Auto, true).unwrap();
        assert_eq!(tab.n_columns(), 2);
        let space = Table::parse_text("a b\n1 2\n", Delimiter::Auto, true).unwrap();
        assert_eq!(space.n_columns(), 2);
        // Принудительная запятая: табуляция остаётся частью ячейки.
        let forced = Table::parse_text("a\tb\n1\t2\n", Delimiter::Comma, true).unwrap();
        assert_eq!(forced.n_columns(), 1);
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(Table::parse_text("", Delimiter::Auto, false).is_err());
        assert!(Table::parse_text("# только комментарий\n", Delimiter::Auto, false).is_err());
        // Заголовок без данных.
        assert!(Table::parse_text("a,b\n", Delimiter::Auto, true).is_err());
    }
}
