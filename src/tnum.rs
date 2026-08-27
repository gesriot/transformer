//! Конвертер числовой таблицы в формат `.tnum` (TRNUM2) и эвристика разметки.
//!
//! Чтение файлов живёт в [`crate::table`]; здесь остаётся только то, что
//! ИНТЕРПРЕТИРУЕТ таблицу: где входы, где выходы, какие колонки категориальные.
//! Сама конвертация — это `Table + TableSchema -> NumericDataset -> TRNUM2`,
//! то есть тот же путь, которым таблицу открывает обучение.

use crate::atomic_write::{same_file, write_atomically};
use crate::data::{read_numeric_tnum, write_numeric_tnum, NumericDataset};
use crate::schema::{Column, ColumnRole, ModelSchema, TableSchema};
use crate::table::Table;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

pub(crate) use crate::table::Delimiter;

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

/// Эвристика старого `prepare` над уже прочитанной таблицей без выделенного
/// заголовка: первая строка рассматривается как возможный header. Нужна GUI,
/// чтобы подсказка не заставляла читать тот же файл второй раз.
pub(crate) fn infer_prepare_spec_from_table(
    table: &Table,
    delimiter: Delimiter,
) -> Result<InferredPrepareSpec, String> {
    infer_prepare_spec_from_rows(table.rows(), delimiter)
}

pub fn infer_prepare_spec_from_text(
    input: &str,
    delimiter: Delimiter,
) -> Result<InferredPrepareSpec, String> {
    // Заголовок ищет сама эвристика, поэтому таблица читается как «без
    // заголовка»: иначе первая строка ушла бы из выборки до анализа.
    let table = Table::parse_text(input, delimiter, false)?;
    infer_prepare_spec_from_table(&table, delimiter)
}

pub fn infer_prepare_spec_from_path(
    path: impl AsRef<Path>,
    delimiter: Delimiter,
) -> Result<InferredPrepareSpec, String> {
    let table = Table::read_path(path, delimiter, false)?;
    infer_prepare_spec_from_table(&table, delimiter)
}

/// Разметка колонок по [`PrepareSpec`]: первые `n_inputs` колонок — входы,
/// остальные — выходы. Имена берутся из заголовка таблицы, а у категорий
/// подписями служат сами коды: настоящих подписей `PrepareSpec` не знает.
pub(crate) fn table_schema_from_prepare_spec(
    table: &Table,
    spec: &PrepareSpec,
) -> Result<TableSchema, String> {
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
    let expected = spec.n_inputs + spec.n_outputs;
    if table.n_columns() != expected {
        return Err(format!(
            "{}: ожидалось {expected} колонок ({} вход + {} выход), получено {}",
            table.source(),
            spec.n_inputs,
            spec.n_outputs,
            table.n_columns()
        ));
    }

    let cat_map: HashMap<usize, usize> = spec.categorical.iter().copied().collect();
    let mut columns = Vec::with_capacity(expected);
    for i in 0..expected {
        let name = match table.header() {
            Some(header) => header[i].clone(),
            None if i < spec.n_inputs => format!("x{i}"),
            None => format!("y{}", i - spec.n_inputs),
        };
        let role = if i < spec.n_inputs {
            ColumnRole::Input
        } else {
            ColumnRole::Output
        };
        columns.push(match cat_map.get(&i) {
            Some(&cardinality) => Column::categorical(
                name,
                role,
                (0..cardinality).map(|code| code.to_string()).collect(),
            )?,
            None => Column::numeric(name, role)?,
        });
    }
    TableSchema::new(columns)
}

/// Открыть источник данных: `.tnum` читается как есть, любая другая таблица
/// размечается эвристикой и превращается в датасет напрямую — без промежуточной
/// строки TRNUM2.
///
/// Эвристика та же, что у `prepare`: заголовок вида `x…/y…`. Если её не хватает,
/// ошибка отправляет к `prepare`, где разметка задаётся флагами.
pub fn read_numeric_source(path: &str) -> Result<(NumericDataset, ModelSchema), String> {
    let source_path = Path::new(path);
    if is_tnum_source(source_path)? {
        return read_numeric_tnum(path).map_err(|e| format!("чтение {path}: {e}"));
    }
    // Читаем таблицу один раз: иначе файл мог измениться между распознаванием
    // заголовка и конвертацией, а Excel пришлось бы разбирать дважды.
    let table = Table::read_path(source_path, Delimiter::Auto, false)?;
    let inferred = infer_prepare_spec_from_table(&table, Delimiter::Auto).map_err(|e| {
        format!(
            "{path}: {e}\n\
             Подготовьте файл явно: transformer prepare {path} data.tnum --inputs N --outputs M"
        )
    })?;
    let spec = PrepareSpec {
        n_inputs: inferred.n_inputs,
        n_outputs: inferred.n_outputs,
        delimiter: inferred.delimiter,
        has_header: inferred.has_header,
        categorical: inferred.categorical,
    };
    let table = if spec.has_header {
        table.promote_first_row_to_header()?
    } else {
        table
    };
    let schema = table_schema_from_prepare_spec(&table, &spec)?;
    let code_columns = category_columns(&spec);
    let dataset = table.to_dataset_with_category_codes(&schema, &code_columns)?;
    Ok((dataset, schema.to_model_schema()?))
}

/// `.tnum` обычно узнаётся по расширению, но сериализованный датасет остаётся
/// таким же источником и после переименования: план Э3 требует учитывать и
/// содержимое. Для таблиц читается только короткий префикс, не весь файл.
fn is_tnum_source(path: &Path) -> Result<bool, String> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tnum"))
    {
        return Ok(true);
    }
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("чтение {}: {e}", path.display()))?;
    let mut prefix = [0_u8; 6];
    let bytes_read = file
        .read(&mut prefix)
        .map_err(|e| format!("чтение {}: {e}", path.display()))?;
    Ok(bytes_read == prefix.len() && matches!(&prefix, b"TRNUM1" | b"TRNUM2"))
}

fn category_columns(spec: &PrepareSpec) -> Vec<usize> {
    spec.categorical.iter().map(|&(index, _)| index).collect()
}

/// Конвертирует таблицу в строку формата `.tnum`.
pub fn table_to_tnum(input: &str, spec: &PrepareSpec) -> Result<String, String> {
    to_tnum(
        &Table::parse_text(input, spec.delimiter, spec.has_header)?,
        spec,
    )
}

pub fn table_path_to_tnum(path: impl AsRef<Path>, spec: &PrepareSpec) -> Result<String, String> {
    to_tnum(
        &Table::read_path(path.as_ref(), spec.delimiter, spec.has_header)?,
        spec,
    )
}

/// Сколько получилось в записанном `.tnum`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PrepareStats {
    pub rows: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
}

/// Конвертировать таблицу в `.tnum` и записать его атомарно.
///
/// Единственный путь записи `.tnum` для всех поверхностей: раньше CLI и GUI
/// делали это порознь и по-разному считали строки — GUI вычитал из числа строк
/// фиксированные шесть заголовочных, хотя в TRNUM2 их больше при наличии
/// `units`/`levels`. Входной файл нельзя указать как выходной: успешная
/// конвертация не должна уничтожать исходную таблицу.
pub fn prepare_tnum_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    spec: &PrepareSpec,
) -> Result<PrepareStats, String> {
    let input = input.as_ref();
    let output = output.as_ref();
    if same_file(input, output) {
        return Err(
            "входной и выходной путь совпадают: prepare не должен перезаписывать исходную таблицу"
                .to_string(),
        );
    }
    let text = table_path_to_tnum(input, spec)?;
    // Число строк берём из заголовка: он единственный, кто знает его точно.
    let rows = text
        .lines()
        .find_map(|line| line.strip_prefix("rows "))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .ok_or_else(|| "в записанном .tnum нет строки rows".to_string())?;

    write_atomically(output, |file| file.write_all(text.as_bytes()))
        .map_err(|e| format!("запись {}: {e}", output.display()))?;
    Ok(PrepareStats {
        rows,
        n_inputs: spec.n_inputs,
        n_outputs: spec.n_outputs,
    })
}

fn to_tnum(table: &Table, spec: &PrepareSpec) -> Result<String, String> {
    let schema = table_schema_from_prepare_spec(table, spec)?;
    let dataset = table.to_dataset_with_category_codes(&schema, &category_columns(spec))?;
    write_numeric_tnum(&schema.to_model_schema()?, &dataset)
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
        assert!(out.starts_with("TRNUM2\n"));
        assert!(out.contains("inputs 3\n"));
        assert!(out.contains("outputs 1\n"));
        assert!(out.contains("specs C C K:3\n"));
        assert!(out.contains("names \"x0\" \"x1\" \"mat\" \"y\"\n"));
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
        let (ds, schema) = crate::data::read_numeric_tnum(path.to_str().unwrap()).unwrap();
        assert_eq!(ds.inputs.dim(), (2, 3));
        assert_eq!(ds.outputs.dim(), (2, 1));
        assert_eq!(
            schema.feature_specs()[2],
            crate::encoders::FeatureSpec::Categorical { cardinality: 3 }
        );
        // PrepareSpec подписей категорий не знает: уровни — честные коды.
        assert_eq!(schema.inputs()[2].category_level(1).unwrap(), "1");
        assert_eq!(schema.input_names(), vec!["x0", "x1", "mat"]);
        assert_eq!(schema.output_names(), vec!["y"]);
        assert!((ds.inputs[[0, 0]] - 0.5).abs() < 1e-6);
        std::fs::remove_file(&path).ok();
    }

    /// Единственный путь записи `.tnum`: заменяет существующий файл целиком,
    /// читается обратно и считает строки по заголовку, а не по числу строк
    /// файла — в TRNUM2 их больше при наличии `units`/`levels`.
    #[test]
    fn prepare_tnum_file_replaces_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("transformer_prepare_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.csv");
        let output = dir.join("out.tnum");
        std::fs::write(&input, "x0,x1,mat,y\n0.5,-0.2,1,2.0\n1.5,0.3,2,3.0\n").unwrap();
        std::fs::write(&output, "прошлый результат").unwrap();

        let stats = prepare_tnum_file(&input, &output, &spec(vec![(2, 3)])).unwrap();
        assert_eq!(
            (stats.rows, stats.n_inputs, stats.n_outputs),
            (2, 3, 1),
            "строки считаются по заголовку rows"
        );

        let (ds, schema) = crate::data::read_numeric_tnum(output.to_str().unwrap()).unwrap();
        assert_eq!(ds.inputs.dim(), (2, 3));
        assert_eq!(schema.output_names(), vec!["y"]);

        // Ни временных файлов, ни следов прежнего содержимого.
        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, vec!["in.csv".to_string(), "out.tnum".to_string()]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prepare_tnum_file_rejects_overwriting_its_input() {
        let path = std::env::temp_dir().join(format!(
            "transformer_prepare_same_path_{}.csv",
            std::process::id()
        ));
        let contents = "x0,y0\n1,2\n";
        std::fs::write(&path, contents).unwrap();

        let err = prepare_tnum_file(
            &path,
            &path,
            &PrepareSpec {
                n_inputs: 1,
                n_outputs: 1,
                delimiter: Delimiter::Auto,
                has_header: true,
                categorical: Vec::new(),
            },
        )
        .unwrap_err();

        assert!(err.contains("не должен перезаписывать"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), contents);
        std::fs::remove_file(path).ok();
    }

    /// Одна и та же числовая таблица, пришедшая из CSV и из TRNUM2, — те же
    /// данные: отпечаток берётся из чисел и типов, а не из формата файла.
    #[test]
    fn the_same_table_has_one_fingerprint_in_any_format() {
        let dir = std::env::temp_dir().join(format!(
            "transformer_fingerprint_formats_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("in.csv");
        let tnum = dir.join("out.tnum");
        // Разметка одна и та же: сравниваются форматы, а не трактовки колонок.
        // Другая трактовка — это уже другие данные, и отпечаток обязан их
        // различать (см. тесты fingerprint).
        std::fs::write(&csv, "x0,x1,x2,y0\n0.5,-0.2,7.0,2.0\n1.5,0.3,8.0,3.0\n").unwrap();

        prepare_tnum_file(&csv, &tnum, &spec(vec![])).unwrap();
        let (from_csv, csv_schema) = read_numeric_source(csv.to_str().unwrap()).unwrap();
        let (from_tnum, tnum_schema) = read_numeric_source(tnum.to_str().unwrap()).unwrap();

        assert_eq!(
            csv_schema.feature_specs(),
            tnum_schema.feature_specs(),
            "типы входов"
        );
        assert_eq!(from_csv.inputs, from_tnum.inputs, "входы");
        assert_eq!(from_csv.outputs, from_tnum.outputs, "выходы");
        assert_eq!(
            crate::fingerprint::DatasetFingerprint::of(&from_csv, &csv_schema).unwrap(),
            crate::fingerprint::DatasetFingerprint::of(&from_tnum, &tnum_schema).unwrap(),
            "формат файла не является частью данных"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn numeric_source_recognizes_tnum_by_contents() {
        let path = std::env::temp_dir().join("transformer_tnum_magic_without_extension.data");
        std::fs::write(
            &path,
            "TRNUM1\ninputs 1\noutputs 1\nspecs C\nrows 1\ndata\n2 3\n",
        )
        .unwrap();
        let (dataset, schema) = read_numeric_source(path.to_str().unwrap()).unwrap();
        assert_eq!(dataset.inputs[[0, 0]], 2.0);
        assert_eq!(dataset.outputs[[0, 0]], 3.0);
        assert_eq!(schema.n_inputs(), 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn direct_table_source_matches_prepare_path() {
        let csv = "x0,material_id,y0\n0.5,1.0,2\n1.5,2,3\n";
        let path = std::env::temp_dir().join("transformer_direct_numeric_source.csv");
        std::fs::write(&path, csv).unwrap();

        let (direct_data, direct_schema) = read_numeric_source(path.to_str().unwrap()).unwrap();
        let prepared = table_to_tnum(
            csv,
            &PrepareSpec {
                n_inputs: 2,
                n_outputs: 1,
                delimiter: Delimiter::Auto,
                has_header: true,
                categorical: vec![(1, 3)],
            },
        )
        .unwrap();
        let (prepared_data, prepared_schema) = crate::data::parse_numeric_tnum(&prepared).unwrap();

        assert_eq!(direct_data.inputs, prepared_data.inputs);
        assert_eq!(direct_data.outputs, prepared_data.outputs);
        assert_eq!(direct_schema, prepared_schema);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn prepare_preserves_header_names_and_falls_back_without_header() {
        let named = PrepareSpec {
            n_inputs: 2,
            n_outputs: 1,
            delimiter: Delimiter::Comma,
            has_header: true,
            categorical: vec![],
        };
        let text =
            table_to_tnum("температура,скорость потока,влажность\n80,1.5,12\n", &named).unwrap();
        let (_, schema) = crate::data::parse_numeric_tnum(&text).unwrap();
        assert_eq!(schema.input_names(), vec!["температура", "скорость потока"]);
        assert_eq!(schema.output_names(), vec!["влажность"]);

        let unnamed = PrepareSpec {
            has_header: false,
            ..named
        };
        let text = table_to_tnum("80,1.5,12\n", &unnamed).unwrap();
        let (_, schema) = crate::data::parse_numeric_tnum(&text).unwrap();
        assert_eq!(schema.input_names(), vec!["x0", "x1"]);
        assert_eq!(schema.output_names(), vec!["y0"]);
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
