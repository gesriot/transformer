//! Конвертер числовой таблицы (CSV/TSV/space) в формат `.tnum` (TRNUM1).
//! Нативный порт `tools/prepare_numeric_dataset.py` (PlanUI.md §1.1).
//!
//! Логика и валидация совпадают с Python-версией; формат уже читается
//! `data::read_numeric_tnum`. Числа пишутся в кратчайшем round-trip
//! представлении f32 (не байт-в-байт с `.9g` Python, но f32-эквивалентно).

use std::collections::HashMap;

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

/// Конвертирует таблицу в строку формата `.tnum`. Валидирует число колонок,
/// конечность значений и категориальные коды (целочисленность + диапазон).
pub fn table_to_tnum(input: &str, spec: &PrepareSpec) -> Result<String, String> {
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

    let mut lines = clean_lines(input);
    if lines.is_empty() {
        return Err("нет строк данных".to_string());
    }
    if spec.has_header {
        lines.remove(0);
    }
    if lines.is_empty() {
        return Err("нет строк данных после заголовка".to_string());
    }

    let delim = detect_delim(lines[0], spec.delimiter);
    let expected = spec.n_inputs + spec.n_outputs;

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(lines.len());
    for (r, line) in lines.iter().enumerate() {
        let toks = split_line(line, delim);
        if toks.len() != expected {
            return Err(format!(
                "строка {r}: ожидалось {expected} колонок ({} вход + {} выход), получено {}",
                spec.n_inputs,
                spec.n_outputs,
                toks.len()
            ));
        }
        let mut row = Vec::with_capacity(expected);
        for (c, t) in toks.iter().enumerate() {
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
