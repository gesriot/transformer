//! Конвертер числовой таблицы (CSV/TSV/space) в формат `.tnum` (TRNUM1).
//! Нативный порт `tools/prepare_numeric_dataset.py` (PlanUI.md §1.1).
//!
//! Логика и валидация совпадают с Python-версией; формат уже читается
//! `data::read_numeric_tnum`. Числа пишутся в кратчайшем round-trip
//! представлении f32 (не байт-в-байт с `.9g` Python, но f32-эквивалентно).

use calamine::{open_workbook_auto, Data, Reader};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delimiter {
    Auto,
    Comma,
    Tab,
    Space,
}

pub struct PrepareSpec {
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub delimiter: Delimiter,
    pub has_header: bool,
    /// (индекс входа, cardinality) для категориальных признаков.
    pub categorical: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct InferredPrepareSpec {
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub delimiter: Delimiter,
    pub has_header: bool,
    pub categorical: Vec<(usize, usize)>,
}

/// Разбирает спецификацию `2:5,7:3` в пары (индекс, cardinality) с валидацией.
pub fn parse_categorical(spec: &str, n_inputs: usize) -> Result<Vec<(usize, usize)>, String> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (i, c) = part
            .split_once(':')
            .ok_or_else(|| format!("плохой --categorical '{part}': ожидался index:cardinality"))?;
        let idx: usize = i
            .trim()
            .parse()
            .map_err(|_| format!("--categorical: индекс '{i}' не целое"))?;
        let card: usize = c
            .trim()
            .parse()
            .map_err(|_| format!("--categorical: cardinality '{c}' не целое"))?;
        if idx >= n_inputs {
            return Err(format!(
                "категориальный индекс {idx} вне диапазона входов 0..{n_inputs}"
            ));
        }
        if card == 0 {
            return Err(format!("cardinality должна быть > 0 для входа {idx}"));
        }
        out.push((idx, card));
    }
    Ok(out)
}

fn clean_lines(input: &str) -> Vec<&str> {
    input
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .collect()
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

fn split_line(line: &str, delim: Option<char>) -> Vec<&str> {
    match delim {
        Some(d) => line
            .split(d)
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect(),
        None => line.split_whitespace().collect(),
    }
}

fn split_line_owned(line: &str, delim: Option<char>) -> Vec<String> {
    split_line(line, delim)
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn text_to_rows(input: &str, mode: Delimiter) -> Result<Vec<Vec<String>>, String> {
    let lines = clean_lines(input);
    if lines.is_empty() {
        return Err("нет строк данных".to_string());
    }
    let delim = detect_delim(lines[0], mode);
    Ok(lines
        .into_iter()
        .map(|line| split_line_owned(line, delim))
        .collect())
}

fn cell_to_token(cell: &Data, row: usize, col: usize) -> Result<String, String> {
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

fn xlsx_to_rows(path: &Path) -> Result<Vec<Vec<String>>, String> {
    let mut workbook =
        open_workbook_auto(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
    let sheet = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| format!("{}: workbook без листов", path.display()))?;
    let range = workbook
        .worksheet_range(&sheet)
        .map_err(|e| format!("чтение листа '{sheet}': {e}"))?;

    let mut rows = Vec::new();
    for (r, row) in range.rows().enumerate() {
        let mut toks = row
            .iter()
            .enumerate()
            .map(|(c, cell)| cell_to_token(cell, r, c))
            .collect::<Result<Vec<_>, _>>()?;
        while toks.last().is_some_and(|t| t.trim().is_empty()) {
            toks.pop();
        }
        if toks.iter().any(|t| !t.trim().is_empty()) {
            rows.push(toks);
        }
    }
    if rows.is_empty() {
        return Err(format!("{}: нет строк данных", path.display()));
    }
    Ok(rows)
}

fn rows_from_path(path: &Path, delimiter: Delimiter) -> Result<Vec<Vec<String>>, String> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("xlsx") | Some("xlsm") | Some("xlsb") | Some("xls") | Some("ods") => {
            xlsx_to_rows(path)
        }
        _ => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("чтение {}: {e}", path.display()))?;
            text_to_rows(&text, delimiter)
        }
    }
}

fn is_finite_number(token: &str) -> bool {
    token.parse::<f32>().map(|v| v.is_finite()).unwrap_or(false)
}

fn is_output_header(raw: &str) -> bool {
    let lower = raw.trim().to_ascii_lowercase();
    lower == "y"
        || lower
            .strip_prefix('y')
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
        || lower.starts_with("out")
}

fn header_split(headers: &[String]) -> Option<(usize, usize)> {
    let first_output = headers.iter().position(|h| is_output_header(h))?;
    if first_output == 0 {
        return None;
    }
    Some((first_output, headers.len().saturating_sub(first_output)))
}

fn infer_categorical(rows: &[Vec<String>], n_inputs: usize) -> Vec<(usize, usize)> {
    let Some(headers) = rows.first() else {
        return Vec::new();
    };
    let data = &rows[1..];
    let mut out = Vec::new();
    for i in 0..n_inputs {
        let name = headers
            .get(i)
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let name_suggests_category = name == "id"
            || name.ends_with("_id")
            || name.contains("material")
            || name.contains("category")
            || name.contains("class");
        if !name_suggests_category {
            continue;
        }

        let mut max_code = None::<usize>;
        let mut valid = true;
        for row in data {
            let Some(tok) = row.get(i) else {
                valid = false;
                break;
            };
            let Ok(raw) = tok.parse::<f32>() else {
                valid = false;
                break;
            };
            let rounded = raw.round();
            if !raw.is_finite() || (raw - rounded).abs() >= 1e-4 || rounded < 0.0 {
                valid = false;
                break;
            }
            max_code = Some(max_code.map_or(rounded as usize, |m| m.max(rounded as usize)));
        }
        if valid {
            if let Some(max_code) = max_code {
                out.push((i, max_code + 1));
            }
        }
    }
    out
}

fn infer_prepare_spec_from_rows(
    rows: &[Vec<String>],
    delimiter: Delimiter,
) -> Result<InferredPrepareSpec, String> {
    if rows.is_empty() {
        return Err("нет строк данных".to_string());
    }
    let first = &rows[0];
    if first.is_empty() {
        return Err("первая строка пустая".to_string());
    }
    let has_header = first.iter().any(|t| !is_finite_number(t));
    if !has_header {
        return Err("не удалось вывести схему: нет заголовка с колонками x.../y...".to_string());
    }
    let (n_inputs, n_outputs) = header_split(first).ok_or_else(|| {
        "не удалось вывести inputs/outputs: ожидаются входные колонки перед y0/y1/...".to_string()
    })?;
    Ok(InferredPrepareSpec {
        n_inputs,
        n_outputs,
        delimiter,
        has_header,
        categorical: infer_categorical(rows, n_inputs),
    })
}

pub fn infer_prepare_spec_from_text(
    input: &str,
    delimiter: Delimiter,
) -> Result<InferredPrepareSpec, String> {
    let rows = text_to_rows(input, delimiter)?;
    infer_prepare_spec_from_rows(&rows, delimiter)
}

pub fn infer_prepare_spec_from_path(
    path: impl AsRef<Path>,
    delimiter: Delimiter,
) -> Result<InferredPrepareSpec, String> {
    let path = path.as_ref();
    let rows = rows_from_path(path, delimiter)?;
    infer_prepare_spec_from_rows(&rows, delimiter)
}

/// Конвертирует таблицу в строку формата `.tnum`. Валидирует число колонок,
/// конечность значений и категориальные коды (целочисленность + диапазон).
pub fn table_to_tnum(input: &str, spec: &PrepareSpec) -> Result<String, String> {
    rows_to_tnum(text_to_rows(input, spec.delimiter)?, spec)
}

pub fn table_path_to_tnum(path: impl AsRef<Path>, spec: &PrepareSpec) -> Result<String, String> {
    rows_to_tnum(rows_from_path(path.as_ref(), spec.delimiter)?, spec)
}

fn rows_to_tnum(mut lines: Vec<Vec<String>>, spec: &PrepareSpec) -> Result<String, String> {
    if spec.n_inputs == 0 || spec.n_outputs == 0 {
        return Err("inputs и outputs должны быть > 0".to_string());
    }
    for &(idx, card) in &spec.categorical {
        if idx >= spec.n_inputs {
            return Err(format!(
                "категориальный индекс {idx} вне диапазона 0..{}",
                spec.n_inputs
            ));
        }
        if card == 0 {
            return Err("cardinality должна быть > 0".to_string());
        }
    }

    if lines.is_empty() {
        return Err("нет строк данных".to_string());
    }
    if spec.has_header {
        lines.remove(0);
    }
    if lines.is_empty() {
        return Err("нет строк данных после заголовка".to_string());
    }

    let expected = spec.n_inputs + spec.n_outputs;

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(lines.len());
    for (r, line) in lines.iter().enumerate() {
        if line.len() != expected {
            return Err(format!(
                "строка {r}: ожидалось {expected} колонок ({} вход + {} выход), получено {}",
                spec.n_inputs,
                spec.n_outputs,
                line.len()
            ));
        }
        let mut row = Vec::with_capacity(expected);
        for (c, t) in line.iter().enumerate() {
            let v: f32 = t
                .parse()
                .map_err(|_| format!("строка {r}, колонка {c}: не число: '{t}'"))?;
            if !v.is_finite() {
                return Err(format!(
                    "строка {r}, колонка {c}: значение не конечно: '{t}'"
                ));
            }
            row.push(v);
        }
        rows.push(row);
    }

    for (r, row) in rows.iter().enumerate() {
        for &(idx, card) in &spec.categorical {
            let raw = row[idx];
            let rounded = raw.round();
            if (raw - rounded).abs() >= 1e-4 {
                return Err(format!(
                    "строка {r}, вход {idx}: код категории должен быть целым, получено {raw}"
                ));
            }
            if rounded < 0.0 || (rounded as usize) >= card {
                return Err(format!(
                    "строка {r}, вход {idx}: категория {rounded} вне [0, {card})"
                ));
            }
        }
    }

    let cat_map: HashMap<usize, usize> = spec.categorical.iter().copied().collect();
    let specs: Vec<String> = (0..spec.n_inputs)
        .map(|i| match cat_map.get(&i) {
            Some(&card) => format!("K:{card}"),
            None => "C".to_string(),
        })
        .collect();

    let mut out = String::new();
    out.push_str("TRNUM1\n");
    out.push_str(&format!("inputs {}\n", spec.n_inputs));
    out.push_str(&format!("outputs {}\n", spec.n_outputs));
    out.push_str(&format!("specs {}\n", specs.join(" ")));
    out.push_str(&format!("rows {}\n", rows.len()));
    out.push_str("data\n");
    for row in &rows {
        let line: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        out.push_str(&line.join(" "));
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(cat: Vec<(usize, usize)>) -> PrepareSpec {
        PrepareSpec {
            n_inputs: 3,
            n_outputs: 1,
            delimiter: Delimiter::Auto,
            has_header: true,
            categorical: cat,
        }
    }

    #[test]
    fn converts_csv_with_categorical() {
        let csv = "x0,x1,mat,y\n0.5,-0.2,1,2.0\n# комментарий\n1.5,0.3,2,3.0\n";
        let out = table_to_tnum(csv, &spec(vec![(2, 3)])).unwrap();
        assert!(out.starts_with("TRNUM1\n"));
        assert!(out.contains("inputs 3\n"));
        assert!(out.contains("outputs 1\n"));
        assert!(out.contains("specs C C K:3\n"));
        assert!(out.contains("rows 2\n"));
        assert!(out.contains("0.5 -0.2 1 2\n")); // 1.0 пишется как "1", 2.0 как "2"
    }

    #[test]
    fn infers_example_complex_style_schema() {
        let csv = "x0,x1,x2,material_id,y0,y1\n0.5,-0.2,1.0,2,3.0,4.0\n1.5,0.3,2.0,0,5.0,6.0\n";
        let inferred = infer_prepare_spec_from_text(csv, Delimiter::Auto).unwrap();
        assert_eq!(inferred.n_inputs, 4);
        assert_eq!(inferred.n_outputs, 2);
        assert!(inferred.has_header);
        assert_eq!(inferred.categorical, vec![(3, 3)]);

        let spec = PrepareSpec {
            n_inputs: inferred.n_inputs,
            n_outputs: inferred.n_outputs,
            delimiter: inferred.delimiter,
            has_header: inferred.has_header,
            categorical: inferred.categorical,
        };
        let out = table_to_tnum(csv, &spec).unwrap();
        assert!(out.contains("specs C C C K:3\n"));
        assert!(out.contains("rows 2\n"));
    }

    #[test]
    fn round_trip_via_read_numeric_tnum() {
        let csv = "x0,x1,mat,y\n0.5,-0.2,1,2.0\n1.5,0.3,2,3.0\n";
        let out = table_to_tnum(csv, &spec(vec![(2, 3)])).unwrap();
        let path = std::env::temp_dir().join("tnum_roundtrip.tnum");
        std::fs::write(&path, &out).unwrap();
        let (ds, specs) = crate::data::read_numeric_tnum(path.to_str().unwrap()).unwrap();
        assert_eq!(ds.inputs.dim(), (2, 3));
        assert_eq!(ds.outputs.dim(), (2, 1));
        assert_eq!(
            specs[2],
            crate::encoders::FeatureSpec::Categorical { cardinality: 3 }
        );
        assert!((ds.inputs[[0, 0]] - 0.5).abs() < 1e-6);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_wrong_column_count() {
        let csv = "0.5,-0.2,1\n"; // 3 колонки вместо 4
        let s = PrepareSpec {
            has_header: false,
            ..spec(vec![])
        };
        assert!(table_to_tnum(csv, &s).unwrap_err().contains("ожидалось 4"));
    }

    #[test]
    fn rejects_fractional_and_out_of_range_category() {
        let frac = "0.5,-0.2,1.7,2.0\n";
        let oor = "0.5,-0.2,5,2.0\n";
        let s = PrepareSpec {
            has_header: false,
            ..spec(vec![(2, 3)])
        };
        assert!(table_to_tnum(frac, &s).unwrap_err().contains("целым"));
        assert!(table_to_tnum(oor, &s).unwrap_err().contains("вне [0, 3)"));
    }

    #[test]
    fn rejects_non_numeric_and_non_finite() {
        let s = PrepareSpec {
            has_header: false,
            ..spec(vec![])
        };
        assert!(table_to_tnum("a,b,c,d\n", &s)
            .unwrap_err()
            .contains("не число"));
        assert!(table_to_tnum("0.1,0.2,0.3,inf\n", &s)
            .unwrap_err()
            .contains("не конечно"));
    }

    #[test]
    fn parse_categorical_validates() {
        assert_eq!(
            parse_categorical("2:5,7:3", 10).unwrap(),
            vec![(2, 5), (7, 3)]
        );
        assert!(parse_categorical("12:5", 10).is_err()); // индекс вне диапазона
        assert!(parse_categorical("2:0", 10).is_err()); // cardinality 0
        assert!(parse_categorical("2", 10).is_err()); // нет двоеточия
        assert_eq!(parse_categorical("", 10).unwrap(), vec![]);
    }
}
