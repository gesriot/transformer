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
use calamine::{open_workbook_auto, Data, Reader};
use ndarray::Array2;
use std::path::Path;

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
    header: Option<Vec<String>>,
    rows: Vec<Vec<String>>,
    // Номер каждой строки в исходном файле. Простого `index + 1` недостаточно:
    // комментарии и пустые строки между записями не попадают в `rows`.
    row_numbers: Vec<usize>,
}

impl Table {
    pub fn read_path(
        path: impl AsRef<Path>,
        delimiter: Delimiter,
        has_header: bool,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let source = path.display().to_string();
        let rows = match path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("xlsx" | "xlsm" | "xlsb" | "xls" | "ods") => read_workbook(path)?,
            _ => {
                let text =
                    std::fs::read_to_string(path).map_err(|e| format!("чтение {source}: {e}"))?;
                split_text(&text, delimiter).map_err(|e| format!("{source}: {e}"))?
            }
        };
        Self::from_rows(source, rows, has_header)
    }

    pub fn parse_text(text: &str, delimiter: Delimiter, has_header: bool) -> Result<Self, String> {
        Self::from_rows(
            "<текст>".to_string(),
            split_text(text, delimiter).map_err(|e| format!("<текст>: {e}"))?,
            has_header,
        )
    }

    fn from_rows(
        source: String,
        mut located_rows: Vec<(usize, Vec<String>)>,
        has_header: bool,
    ) -> Result<Self, String> {
        if located_rows.is_empty() {
            return Err(format!("{source}: нет строк данных"));
        }
        let header = if has_header {
            let (_, header) = located_rows.remove(0);
            if located_rows.is_empty() {
                return Err(format!("{source}: нет строк данных после заголовка"));
            }
            Some(header)
        } else {
            None
        };
        let (row_numbers, rows) = located_rows.into_iter().unzip();
        Ok(Self {
            source,
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
            return Err(format!("{}: нет строк данных после заголовка", self.source));
        }
        self.header = Some(self.rows.remove(0));
        self.row_numbers.remove(0);
        Ok(self)
    }

    pub fn source(&self) -> &str {
        &self.source
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
                self.source,
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
                    self.source,
                    self.row_label(r),
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

    /// Номер строки как в файле (1-based, с учётом заголовка) — иначе
    /// пользователь ищет ошибку не там.
    fn row_label(&self, r: usize) -> usize {
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
        let at = format!("{}: строка {}", self.source, self.row_label(r));
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

fn read_workbook(path: &Path) -> Result<Vec<(usize, Vec<String>)>, String> {
    let mut workbook =
        open_workbook_auto(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
    let sheet = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| format!("{}: workbook без листов", path.display()))?;
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
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if cells.iter().any(|t| !t.trim().is_empty()) {
            // Внутренние и хвостовые пустые ячейки сохраняются. `Range` уже
            // ограничен используемой областью листа; удалив хвост, мы бы
            // превратили пропуск последнего признака в «рваную строку».
            rows.push((source_row, cells));
        }
    }
    if rows.is_empty() {
        return Err(format!("{}: нет строк данных", path.display()));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Column, ColumnRole, TableSchema};

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
