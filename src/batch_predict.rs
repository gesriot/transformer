//! Пакетный Predict для Excel: первый лист `.xlsx`, заголовки `x0..xN` и
//! `y0..yM`, заполнение выходных колонок предсказаниями.
//!
//! Writer намеренно минимальный: сохраняет значения первого листа без стилей,
//! формул и дополнительных листов. Это достаточно для сценария "вставить
//! предсказанные y-колонки в расчетную таблицу".

use calamine::{open_workbook_auto, Data, Reader};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::Path;
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

    fn header_text(&self) -> Option<String> {
        match self {
            SheetCell::Text(s) => Some(s.trim().to_string()),
            SheetCell::Number(v) => Some(v.to_string()),
            SheetCell::Bool(v) => Some(if *v { "true" } else { "false" }.to_string()),
            SheetCell::Blank => None,
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

pub fn read_prediction_xlsx(
    path: &str,
    n_inputs: usize,
    n_outputs: usize,
) -> Result<PredictionSheet, String> {
    let path_ref = Path::new(path);
    let mut workbook =
        open_workbook_auto(path_ref).map_err(|e| format!("чтение {}: {e}", path_ref.display()))?;
    let sheet_name = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| format!("{}: workbook без листов", path_ref.display()))?;
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| format!("чтение листа '{sheet_name}': {e}"))?;

    let mut rows: Vec<Vec<SheetCell>> = range
        .rows()
        .map(|row| row.iter().map(data_to_cell).collect())
        .collect();
    trim_empty_edges(&mut rows);
    if rows.is_empty() {
        return Err(format!("{}: лист пуст", path_ref.display()));
    }

    let raw_headers = rows.remove(0);
    let mut headers = Vec::new();
    for (i, cell) in raw_headers.iter().enumerate() {
        let header = cell
            .header_text()
            .ok_or_else(|| format!("заголовок в колонке {} пуст", i + 1))?;
        headers.push(header);
    }
    while headers.last().is_some_and(|h| h.trim().is_empty()) {
        headers.pop();
    }
    if headers.is_empty() {
        return Err("строка заголовков пуста".to_string());
    }

    let index = header_index(&headers)?;
    let input_cols = collect_cols(&index, 'x', n_inputs)?;
    let output_cols = collect_cols(&index, 'y', n_outputs)?;
    let max_col = input_cols
        .iter()
        .chain(output_cols.iter())
        .copied()
        .max()
        .unwrap_or(0);
    if headers.len() <= max_col {
        headers.resize_with(max_col + 1, String::new);
    }

    Ok(PredictionSheet {
        headers,
        rows,
        input_cols,
        output_cols,
    })
}

pub fn write_prediction_xlsx(path: &str, sheet: &PredictionSheet) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("создание {path}: {e}"))?;
    let mut zip = ZipWriter::new(file);
    add_zip_file(&mut zip, "[Content_Types].xml", CONTENT_TYPES)?;
    add_zip_file(&mut zip, "_rels/.rels", ROOT_RELS)?;
    add_zip_file(&mut zip, "xl/workbook.xml", WORKBOOK)?;
    add_zip_file(&mut zip, "xl/_rels/workbook.xml.rels", WORKBOOK_RELS)?;
    let worksheet = worksheet_xml(sheet);
    add_zip_file(&mut zip, "xl/worksheets/sheet1.xml", &worksheet)?;
    zip.finish()
        .map_err(|e| format!("завершение xlsx {path}: {e}"))?;
    Ok(())
}

fn data_to_cell(cell: &Data) -> SheetCell {
    match cell {
        Data::Empty => SheetCell::Blank,
        Data::String(s) => {
            if s.trim().is_empty() {
                SheetCell::Blank
            } else {
                SheetCell::Text(s.clone())
            }
        }
        Data::Float(v) => SheetCell::Number(*v),
        Data::Int(v) => SheetCell::Number(*v as f64),
        Data::Bool(v) => SheetCell::Bool(*v),
        Data::DateTime(v) => SheetCell::Text(v.to_string()),
        Data::DateTimeIso(s) | Data::DurationIso(s) => SheetCell::Text(s.clone()),
        Data::Error(e) => SheetCell::Text(format!("{e}")),
    }
}

fn trim_empty_edges(rows: &mut Vec<Vec<SheetCell>>) {
    for row in rows.iter_mut() {
        while row.last().is_some_and(SheetCell::is_blank) {
            row.pop();
        }
    }
    while rows
        .last()
        .is_some_and(|row| row.iter().all(SheetCell::is_blank))
    {
        rows.pop();
    }
}

fn header_index(headers: &[String]) -> Result<HashMap<String, usize>, String> {
    let mut out = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        let key = header.trim().to_ascii_lowercase();
        if key.is_empty() {
            continue;
        }
        if out.insert(key.clone(), i).is_some() {
            return Err(format!("дублирующийся заголовок '{key}'"));
        }
    }
    Ok(out)
}

fn collect_cols(
    index: &HashMap<String, usize>,
    prefix: char,
    count: usize,
) -> Result<Vec<usize>, String> {
    let mut cols = Vec::with_capacity(count);
    for i in 0..count {
        let name = format!("{prefix}{i}");
        let col = index
            .get(&name)
            .copied()
            .ok_or_else(|| format!("в Excel-файле нет колонки '{name}'"))?;
        cols.push(col);
    }
    Ok(cols)
}

fn worksheet_xml(sheet: &PredictionSheet) -> String {
    let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>"#,
    );
    write_row(&mut xml, 1, &header_cells(&sheet.headers));
    for (i, row) in sheet.rows.iter().enumerate() {
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

    #[test]
    fn cell_refs_cover_excel_columns() {
        assert_eq!(cell_ref(0, 1), "A1");
        assert_eq!(cell_ref(25, 7), "Z7");
        assert_eq!(cell_ref(26, 3), "AA3");
        assert_eq!(cell_ref(27, 3), "AB3");
    }

    #[test]
    fn xlsx_roundtrip_fills_outputs() {
        let input = tmp_path("input.xlsx");
        let output = tmp_path("output.xlsx");
        let sheet = PredictionSheet {
            headers: vec![
                "x0".to_string(),
                "x1".to_string(),
                "note".to_string(),
                "y0".to_string(),
                "y1".to_string(),
            ],
            rows: vec![
                vec![
                    SheetCell::Number(1.0),
                    SheetCell::Number(2.0),
                    SheetCell::Text("a".to_string()),
                    SheetCell::Blank,
                    SheetCell::Blank,
                ],
                vec![
                    SheetCell::Text("3.5".to_string()),
                    SheetCell::Number(4.0),
                    SheetCell::Text("b".to_string()),
                    SheetCell::Blank,
                    SheetCell::Blank,
                ],
            ],
            input_cols: vec![0, 1],
            output_cols: vec![3, 4],
        };
        write_prediction_xlsx(input.to_str().unwrap(), &sheet).unwrap();

        let read = read_prediction_xlsx(input.to_str().unwrap(), 2, 2).unwrap();
        assert_eq!(
            read.input_rows().unwrap(),
            vec![vec![1.0, 2.0], vec![3.5, 4.0]]
        );

        let filled = read
            .fill_outputs(&[vec![10.0, 20.0], vec![30.0, 40.0]])
            .unwrap();
        write_prediction_xlsx(output.to_str().unwrap(), &filled).unwrap();

        let reread = read_prediction_xlsx(output.to_str().unwrap(), 2, 2).unwrap();
        assert_eq!(reread.rows[0][3], SheetCell::Number(10.0));
        assert_eq!(reread.rows[0][4], SheetCell::Number(20.0));
        assert_eq!(reread.rows[1][3], SheetCell::Number(30.0));
        assert_eq!(reread.rows[1][4], SheetCell::Number(40.0));

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "transformer_batch_predict_{}_{}",
            std::process::id(),
            name
        ))
    }
}
