//! Пакетный Predict для Excel: первый лист `.xlsx`, заголовки `x0..xN` и
//! `y0..yM`, заполнение выходных колонок предсказаниями.
//!
//! Writer намеренно минимальный: сохраняет значения первого листа без стилей,
//! формул и дополнительных листов. Это достаточно для сценария "вставить
//! предсказанные y-колонки в расчетную таблицу".

use crate::predict::{parse_rows, Predictions};
use crate::schema::ModelSchema;
use crate::table::{Delimiter, Table};
use ndarray::Array2;
use std::fs::File;
use std::io::{Seek, Write};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

#[derive(Clone, Debug, PartialEq)]
pub enum SheetCell {
    Blank,
    Text(String),
    Number(f64),
    Bool(bool),
}

impl SheetCell {
    fn is_blank(&self) -> bool {
        match self {
            SheetCell::Blank => true,
            SheetCell::Text(s) => s.trim().is_empty(),
            SheetCell::Number(_) | SheetCell::Bool(_) => false,
        }
    }

    fn as_f32(&self, row: usize, col: usize) -> Result<f32, String> {
        match self {
            SheetCell::Number(v) => {
                if v.is_finite() {
                    Ok(*v as f32)
                } else {
                    Err(format!("строка {row}, колонка {col}: число не конечно"))
                }
            }
            SheetCell::Text(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return Err(format!("строка {row}, колонка {col}: пустой вход"));
                }
                let v: f32 = trimmed
                    .parse()
                    .map_err(|_| format!("строка {row}, колонка {col}: '{s}' не число"))?;
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(format!("строка {row}, колонка {col}: число не конечно"))
                }
            }
            SheetCell::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            SheetCell::Blank => Err(format!("строка {row}, колонка {col}: пустой вход")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PredictionSheet {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<SheetCell>>,
    input_cols: Vec<usize>,
    output_cols: Vec<usize>,
}

impl PredictionSheet {
    pub fn input_rows(&self) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::new();
        for (r, row) in self.rows.iter().enumerate() {
            if row.iter().all(SheetCell::is_blank) {
                continue;
            }
            let excel_row = r + 2;
            let mut values = Vec::with_capacity(self.input_cols.len());
            for &col in &self.input_cols {
                let cell = row.get(col).unwrap_or(&SheetCell::Blank);
                values.push(cell.as_f32(excel_row, col + 1)?);
            }
            out.push(values);
        }
        Ok(out)
    }

    pub fn fill_outputs(&self, predictions: &[Vec<f32>]) -> Result<PredictionSheet, String> {
        let rows_to_fill = self
            .rows
            .iter()
            .filter(|row| !row.iter().all(SheetCell::is_blank))
            .count();
        if predictions.len() != rows_to_fill {
            return Err(format!(
                "ожидалось {rows_to_fill} строк предсказаний, получено {}",
                predictions.len()
            ));
        }

        let mut out = self.clone();
        let width = out.headers.len();
        let mut pred_idx = 0;
        for row in &mut out.rows {
            if row.iter().all(SheetCell::is_blank) {
                continue;
            }
            if predictions[pred_idx].len() != out.output_cols.len() {
                return Err(format!(
                    "строка предсказания {}: ожидалось {} выходов, получено {}",
                    pred_idx + 1,
                    out.output_cols.len(),
                    predictions[pred_idx].len()
                ));
            }
            if row.len() < width {
                row.resize(width, SheetCell::Blank);
            }
            for (j, &col) in out.output_cols.iter().enumerate() {
                row[col] = SheetCell::Number(predictions[pred_idx][j] as f64);
            }
            pred_idx += 1;
        }
        Ok(out)
    }
}

/// Что получилось при экспорте — для отчёта пользователю.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSummary {
    pub rows: usize,
    pub extrapolated_rows: usize,
    /// Колонки выходов, которые уже были в таблице и перезаписаны.
    pub replaced: Vec<String>,
    /// Колонки выходов, которых не было и которые добавлены справа.
    pub added: Vec<String>,
}

/// Прочитать таблицу, посчитать прогноз и записать НОВУЮ книгу.
///
/// Входные колонки связываются по настоящим именам из [`ModelSchema`]: у
/// модели без имён это по-прежнему `x0…xN`. Выходные колонки называются по
/// схеме — существующие заменяются, отсутствующие добавляются; посторонние
/// колонки сохраняются как значения.
///
/// Исходная книга НЕ сохраняется: стили, формулы, дополнительные листы и
/// структура теряются, результат — минимальная новая книга.
pub fn export_predictions<F>(
    input: &str,
    output: &str,
    schema: &ModelSchema,
    predict: F,
) -> Result<ExportSummary, String>
where
    F: Fn(&Array2<f32>) -> Result<Predictions, String>,
{
    let table = Table::read_path(input, Delimiter::Auto, true)?;
    let header = table
        .header()
        .ok_or_else(|| format!("{input}: нужна строка заголовков с именами колонок"))?
        .to_vec();

    // Связывание по именам: без него таблица и модель молча разошлись бы.
    let column_of = |name: &str| header.iter().position(|h| h.trim() == name);
    let input_cols = schema
        .input_names()
        .iter()
        .map(|name| {
            column_of(name).ok_or_else(|| {
                format!(
                    "{input}: нет колонки '{name}'. Модель ждёт входы: {}. В таблице: {}",
                    schema.input_names().join(", "),
                    header.join(", ")
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Разбор — слоем схемы: подписи категорий и адресные ошибки.
    let mut cells = Vec::with_capacity(table.n_rows());
    let mut labels = Vec::with_capacity(table.n_rows());
    for (r, row) in table.rows().iter().enumerate() {
        let picked = input_cols
            .iter()
            .map(|&c| row.get(c).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        cells.push(picked);
        labels.push(table.file_row(r));
    }
    let inputs = parse_rows(schema, &cells, &labels).map_err(|e| format!("{input}: {e}"))?;
    let predictions = predict(&inputs)?;

    // Выходные колонки: существующие заменяем, недостающие добавляем справа.
    let mut headers = header.clone();
    let mut replaced = Vec::new();
    let mut added = Vec::new();
    let mut output_cols = Vec::with_capacity(schema.n_outputs());
    for name in schema.output_names() {
        match column_of(name) {
            Some(col) => {
                replaced.push(name.to_string());
                output_cols.push(col);
            }
            None => {
                added.push(name.to_string());
                output_cols.push(headers.len());
                headers.push(name.to_string());
            }
        }
    }

    let width = headers.len();
    let mut rows = Vec::with_capacity(table.n_rows());
    for (r, row) in table.rows().iter().enumerate() {
        let mut cells: Vec<SheetCell> = (0..width)
            .map(|c| match row.get(c) {
                // Посторонние колонки переносятся как значения — но числом
                // становится только то, что записывается обратно теми же
                // символами. Иначе артикул «007» уехал бы в таблицу как 7.
                Some(text) if !text.trim().is_empty() => match text.trim().parse::<f64>() {
                    Ok(v) if v.is_finite() && format!("{v}") == text.trim() => SheetCell::Number(v),
                    _ => SheetCell::Text(text.clone()),
                },
                _ => SheetCell::Blank,
            })
            .collect();
        for (slot, &col) in output_cols.iter().enumerate() {
            cells[col] = SheetCell::Number(predictions.outputs[[r, slot]] as f64);
        }
        rows.push(cells);
    }

    write_sheet(output, &headers, &rows)?;
    Ok(ExportSummary {
        rows: predictions.rows(),
        extrapolated_rows: predictions.extrapolated_rows(),
        replaced,
        added,
    })
}

/// Записать минимальную книгу: один лист, только значения.
fn write_sheet(path: &str, headers: &[String], rows: &[Vec<SheetCell>]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("создание {path}: {e}"))?;
    let mut zip = ZipWriter::new(file);
    add_zip_file(&mut zip, "[Content_Types].xml", CONTENT_TYPES)?;
    add_zip_file(&mut zip, "_rels/.rels", ROOT_RELS)?;
    add_zip_file(&mut zip, "xl/workbook.xml", WORKBOOK)?;
    add_zip_file(&mut zip, "xl/_rels/workbook.xml.rels", WORKBOOK_RELS)?;
    add_zip_file(
        &mut zip,
        "xl/worksheets/sheet1.xml",
        &worksheet_xml(headers, rows),
    )?;
    zip.finish()
        .map_err(|e| format!("завершение xlsx {path}: {e}"))?;
    Ok(())
}

fn worksheet_xml(headers: &[String], rows: &[Vec<SheetCell>]) -> String {
    let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>"#,
    );
    write_row(&mut xml, 1, &header_cells(headers));
    for (i, row) in rows.iter().enumerate() {
        write_row(&mut xml, i + 2, row);
    }
    xml.push_str("</sheetData></worksheet>");
    xml
}

fn header_cells(headers: &[String]) -> Vec<SheetCell> {
    headers.iter().map(|h| SheetCell::Text(h.clone())).collect()
}

fn write_row(xml: &mut String, row_num: usize, cells: &[SheetCell]) {
    if cells.iter().all(SheetCell::is_blank) {
        return;
    }
    xml.push_str(&format!(r#"<row r="{row_num}">"#));
    for (col, cell) in cells.iter().enumerate() {
        if cell.is_blank() {
            continue;
        }
        let r = cell_ref(col, row_num);
        match cell {
            SheetCell::Blank => {}
            SheetCell::Text(s) => {
                xml.push_str(&format!(r#"<c r="{r}" t="inlineStr"><is>"#));
                write_text(xml, s);
                xml.push_str("</is></c>");
            }
            SheetCell::Number(v) => {
                if v.is_finite() {
                    xml.push_str(&format!(r#"<c r="{r}"><v>{v}</v></c>"#));
                }
            }
            SheetCell::Bool(v) => {
                xml.push_str(&format!(
                    r#"<c r="{r}" t="b"><v>{}</v></c>"#,
                    if *v { 1 } else { 0 }
                ));
            }
        }
    }
    xml.push_str("</row>");
}

fn write_text(xml: &mut String, text: &str) {
    let preserve = text.starts_with(char::is_whitespace) || text.ends_with(char::is_whitespace);
    if preserve {
        xml.push_str(r#"<t xml:space="preserve">"#);
    } else {
        xml.push_str("<t>");
    }
    xml.push_str(&escape_xml(text));
    xml.push_str("</t>");
}

fn escape_xml(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

fn cell_ref(mut col: usize, row: usize) -> String {
    let mut letters = Vec::new();
    loop {
        let rem = col % 26;
        letters.push((b'A' + rem as u8) as char);
        if col < 26 {
            break;
        }
        col = col / 26 - 1;
    }
    letters.iter().rev().collect::<String>() + &row.to_string()
}

fn add_zip_file<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    contents: &str,
) -> Result<(), String> {
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file(name, options)
        .map_err(|e| format!("xlsx: запись {name}: {e}"))?;
    zip.write_all(contents.as_bytes())
        .map_err(|e| format!("xlsx: запись {name}: {e}"))?;
    Ok(())
}

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#;

const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#;

const WORKBOOK: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Predictions" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"#;

const WORKBOOK_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Column, ColumnRole};

    #[test]
    fn cell_refs_cover_excel_columns() {
        assert_eq!(cell_ref(0, 1), "A1");
        assert_eq!(cell_ref(25, 7), "Z7");
        assert_eq!(cell_ref(26, 3), "AA3");
        assert_eq!(cell_ref(27, 3), "AB3");
    }

    fn schema() -> ModelSchema {
        ModelSchema::new(
            vec![
                Column::numeric("температура", ColumnRole::Input).unwrap(),
                Column::categorical(
                    "материал",
                    ColumnRole::Input,
                    vec!["песок".into(), "глина".into()],
                )
                .unwrap(),
            ],
            vec![
                Column::numeric("влажность", ColumnRole::Output).unwrap(),
                Column::numeric("плотность", ColumnRole::Output).unwrap(),
            ],
        )
        .unwrap()
    }

    /// Прогноз по каждой строке: сумма входов и их разность — так по значению
    /// в ячейке видно, что предсказание попало в свою строку и колонку.
    fn double(inputs: &Array2<f32>) -> Result<Predictions, String> {
        let outputs = Array2::from_shape_fn((inputs.nrows(), 2), |(r, c)| {
            if c == 0 {
                inputs[[r, 0]] + inputs[[r, 1]]
            } else {
                inputs[[r, 0]] - inputs[[r, 1]]
            }
        });
        Ok(Predictions {
            outputs,
            warnings: Vec::new(),
        })
    }

    fn read_back(path: &std::path::Path) -> (Vec<String>, Vec<Vec<String>>) {
        let table = Table::read_path(path, Delimiter::Auto, true).unwrap();
        (table.header().unwrap().to_vec(), table.rows().to_vec())
    }

    /// Колонки связываются по именам, а не по позиции: порядок в таблице свой,
    /// посторонняя колонка сохраняется, отсутствующий выход добавляется.
    #[test]
    fn export_binds_columns_by_name() {
        let input = tmp_path("input.csv");
        let output = tmp_path("output.xlsx");
        std::fs::write(
            &input,
            "заметка,материал,температура,влажность
хорошо,глина,70,0
плохо,песок,60,0
",
        )
        .unwrap();

        let summary = export_predictions(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap();

        assert_eq!(summary.rows, 2);
        assert_eq!(summary.replaced, vec!["влажность"]);
        assert_eq!(summary.added, vec!["плотность"]);

        let (headers, rows) = read_back(&output);
        assert_eq!(
            headers,
            vec![
                "заметка",
                "материал",
                "температура",
                "влажность",
                "плотность"
            ]
        );
        // Посторонняя колонка переехала как значение.
        assert_eq!(rows[0][0], "хорошо");
        // Категория осталась подписью, а не превратилась в код.
        assert_eq!(rows[0][1], "глина");
        // 70 + 1 (код «глина») и 70 − 1.
        assert_eq!(rows[0][3], "71");
        assert_eq!(rows[0][4], "69");
        // Вторая строка: 60 + 0 и 60 − 0.
        assert_eq!(rows[1][3], "60");
        assert_eq!(rows[1][4], "60");

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    /// Посторонняя колонка переносится как есть: число остаётся числом, а
    /// «числоподобный» текст — текстом, иначе артикул или код теряет форму.
    #[test]
    fn foreign_columns_keep_their_form() {
        let input = tmp_path("foreign.csv");
        let output = tmp_path("foreign_out.xlsx");
        std::fs::write(
            &input,
            "артикул,счёт,температура,материал\n007,1.5,70,глина\n0123,2,60,песок\n",
        )
        .unwrap();

        export_predictions(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap();

        let (_, rows) = read_back(&output);
        // Ведущие нули сохранены: это идентификатор, а не число.
        assert_eq!(rows[0][0], "007");
        assert_eq!(rows[1][0], "0123");
        // Настоящие числа остались числами.
        assert_eq!(rows[0][1], "1.5");
        assert_eq!(rows[1][1], "2");

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn export_explains_a_missing_input_column() {
        let input = tmp_path("missing.csv");
        std::fs::write(
            &input,
            "материал,влажность
глина,1
",
        )
        .unwrap();
        let err = export_predictions(
            input.to_str().unwrap(),
            tmp_path("unused.xlsx").to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap_err();
        assert!(err.contains("нет колонки 'температура'"), "{err}");
        // Ошибка показывает и что нужно, и что есть.
        assert!(err.contains("В таблице: материал, влажность"), "{err}");
        let _ = std::fs::remove_file(input);
    }

    /// Ошибка ячейки адресуется номером строки ФАЙЛА, а не порядком в выборке.
    #[test]
    fn export_addresses_a_bad_cell() {
        let input = tmp_path("bad.csv");
        std::fs::write(
            &input,
            "температура,материал
70,глина
# комментарий
60,мрамор
",
        )
        .unwrap();
        let err = export_predictions(
            input.to_str().unwrap(),
            tmp_path("unused2.xlsx").to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap_err();
        assert!(err.contains("строка 4"), "{err}");
        assert!(err.contains("мрамор"), "{err}");
        let _ = std::fs::remove_file(input);
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "transformer_batch_predict_{}_{}",
            std::process::id(),
            name
        ))
    }
}
