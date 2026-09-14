//! Экспорт таблицы с прогнозами по именам колонок из схемы модели.
//!
//! Книга `.xlsx` сохраняется: результат — копия исходного пакета, в которой
//! изменён только выбранный лист (см. [`crate::xlsx`]). Для остальных
//! источников — текстовых таблиц, `.xlsm`, `.ods` — сохранять нечего или пока
//! не обещано, и writer создаёт новую минимальную книгу: значения выбранного
//! листа переносятся, стили и остальные листы — нет. Что именно получилось,
//! видно в [`ExportSummary::preserved`].

use crate::atomic_write::{same_file, write_atomically};
use crate::predict::{parse_rows, Predictions};
use crate::schema::ModelSchema;
use crate::table::{Delimiter, Table};
use crate::xlsx::{CellEdit, CellValue};
use ndarray::Array2;
use std::collections::HashMap;
use std::io::{self, Seek, Write};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

#[derive(Clone, Debug, PartialEq)]
enum SheetCell {
    Blank,
    Text(String),
    Number(f64),
}

impl SheetCell {
    fn is_blank(&self) -> bool {
        match self {
            SheetCell::Blank => true,
            SheetCell::Text(s) => s.trim().is_empty(),
            SheetCell::Number(_) => false,
        }
    }
}

/// Что получилось при экспорте — для отчёта пользователю.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ExportSummary {
    pub rows: usize,
    pub extrapolated_rows: usize,
    /// Сохранена ли исходная книга. `false` — результат собран заново, и
    /// оформление исходного файла в него не попало.
    pub preserved: bool,
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
/// Читается ОДИН лист: `sheet` выбирает его по имени, `None` подходит книге с
/// единственным листом. Молча брать первый нельзя — см. [`Table::read_sheet`].
///
/// Исходная книга НЕ сохраняется: стили, формулы, дополнительные листы и
/// структура теряются, результат — минимальная новая книга.
pub fn export_predictions<F>(
    input: &str,
    sheet: Option<&str>,
    output: &str,
    schema: &ModelSchema,
    predict: F,
) -> Result<ExportSummary, String>
where
    F: Fn(&Array2<f32>) -> Result<Predictions, String>,
{
    if same_file(Path::new(input), Path::new(output)) {
        return Err(
            "входной и выходной путь совпадают: экспорт не должен перезаписывать исходную книгу"
                .to_string(),
        );
    }
    let table = Table::read_sheet(input, sheet, Delimiter::Auto, true)?;
    let header = table
        .header()
        .ok_or_else(|| format!("{input}: нужна строка заголовков с именами колонок"))?
        .to_vec();

    // Связывание по именам: без него таблица и модель молча разошлись бы.
    // Дубликат — ошибка, иначе `position()` молча выбрал бы первую колонку.
    let index = header_index(&header).map_err(|e| format!("{input}: {e}"))?;
    let input_cols = schema
        .input_names()
        .iter()
        .map(|name| {
            index.get(name).copied().ok_or_else(|| {
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
    if predictions.outputs.dim() != (table.n_rows(), schema.n_outputs()) {
        return Err(format!(
            "прогноз имеет форму {}×{}, ожидалось {}×{}",
            predictions.outputs.nrows(),
            predictions.outputs.ncols(),
            table.n_rows(),
            schema.n_outputs()
        ));
    }

    // Выходные колонки: существующие заменяем, недостающие добавляем справа.
    let mut headers = header.clone();
    let mut replaced = Vec::new();
    let mut added = Vec::new();
    let mut output_cols = Vec::with_capacity(schema.n_outputs());
    for name in schema.output_names() {
        match index.get(name).copied() {
            Some(col) => {
                replaced.push(name.to_string());
                output_cols.push(col);
                // Имя в новой книге каноническое, даже если в исходной вокруг
                // него были пробелы.
                headers[col] = name.to_string();
            }
            None => {
                added.push(name.to_string());
                output_cols.push(headers.len());
                headers.push(name.to_string());
            }
        }
    }

    // Исходную книгу правим точечно; всё остальное собирается заново, и
    // тогда нужна полная таблица значений.
    let preserved = preserves_workbook(input, output);
    let width = headers.len();
    let mut rows = Vec::with_capacity(if preserved { 0 } else { table.n_rows() });
    for (r, row) in table.rows().iter().enumerate().filter(|_| !preserved) {
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

    if preserved {
        // Заголовки существующих колонок не трогаем: их писал человек, и
        // «канонизировать» чужой текст — правка, которой никто не просил.
        let sheet = table
            .sheet()
            .ok_or_else(|| format!("{input}: книга без выбранного листа"))?;
        let edits = sheet_edits(&table, &headers, &output_cols, &predictions)
            .map_err(|e| format!("{input}: {e}"))?;
        crate::xlsx::patch_sheet(Path::new(input), Path::new(output), sheet, &edits)?;
    } else {
        write_sheet(output, &headers, &rows)?;
    }
    Ok(ExportSummary {
        rows: predictions.rows(),
        extrapolated_rows: predictions.extrapolated_rows(),
        preserved,
        replaced,
        added,
    })
}

/// Сохраняется ли исходная книга при таком экспорте.
///
/// Пока только `.xlsx` → `.xlsx`: у текстовой таблицы сохранять нечего, а
/// `.xlsm` и `.ods` — отдельная работа со своими проверками. Решение принимается
/// по путям, до чтения: отчёт обязан сказать, что именно получится.
fn preserves_workbook(input: &str, output: &str) -> bool {
    let is_xlsx = |path: &str| {
        Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("xlsx"))
    };
    is_xlsx(input) && is_xlsx(output)
}

/// Какие ячейки листа изменит экспорт.
///
/// Колонки выходов пишутся по физическим координатам исходного листа: строка
/// берётся у самой таблицы, колонка — от её начала. Новая колонка получает и
/// заголовок: без него книга откроется с безымянным столбцом чисел.
fn sheet_edits(
    table: &Table,
    headers: &[String],
    output_cols: &[usize],
    predictions: &Predictions,
) -> Result<Vec<CellEdit>, String> {
    let anchor = table
        .anchor()
        .ok_or_else(|| "у источника нет координат листа".to_string())?;
    let header_row = anchor
        .header_row
        .ok_or_else(|| "лист прочитан без строки заголовков".to_string())?;
    let existing = table.n_columns();

    let mut edits = Vec::with_capacity(output_cols.len() * (table.n_rows() + 1));
    for (slot, &col) in output_cols.iter().enumerate() {
        let column = anchor.first_column + col;
        if col >= existing {
            edits.push(CellEdit {
                row: header_row,
                column,
                value: CellValue::Text(headers[col].clone()),
            });
        }
        for r in 0..table.n_rows() {
            edits.push(CellEdit {
                row: table.file_row(r),
                column,
                value: CellValue::Number(predictions.outputs[[r, slot]] as f64),
            });
        }
    }
    Ok(edits)
}

fn header_index(headers: &[String]) -> Result<HashMap<&str, usize>, String> {
    let mut index = HashMap::with_capacity(headers.len());
    for (column, header) in headers.iter().enumerate() {
        let name = header.trim();
        if name.is_empty() {
            continue;
        }
        if index.insert(name, column).is_some() {
            return Err(format!("дублирующийся заголовок '{name}'"));
        }
    }
    Ok(index)
}

/// Записать минимальную книгу: один лист, только значения.
fn write_sheet(path: &str, headers: &[String], rows: &[Vec<SheetCell>]) -> Result<(), String> {
    // Экспорт часто пишут поверх вчерашнего результата: до успешного
    // завершения архива прежний файл трогать нельзя.
    write_atomically(Path::new(path), |file| {
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
        // finish дописывает центральный каталог: без него архив не читается.
        zip.finish().map_err(io::Error::other)?;
        Ok(())
    })
    .map_err(|e| format!("запись {path}: {e}"))
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
) -> io::Result<()> {
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file(name, options).map_err(io::Error::other)?;
    zip.write_all(contents.as_bytes())
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

    /// Прогноз считается по выбранному листу. Без выбора экспорт не начинается:
    /// молча посчитать по титульному листу хуже, чем не посчитать вовсе.
    #[test]
    fn export_reads_the_chosen_sheet() {
        let input = tmp_path("export_book.xlsx");
        let output = tmp_path("export_book_out.xlsx");
        crate::table::write_test_workbook(
            &input,
            &[
                ("Титульный", &[&["отчёт"]]),
                (
                    "Опыты",
                    &[
                        &["материал", "температура", "влажность"],
                        &["глина", "70", "0"],
                    ],
                ),
            ],
        );

        let err = export_predictions(
            input.to_str().unwrap(),
            None,
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap_err();
        assert!(err.contains("Опыты"), "{err}");

        let summary = export_predictions(
            input.to_str().unwrap(),
            Some("Опыты"),
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap();
        assert_eq!(summary.rows, 1);
        assert!(summary.preserved, "книга .xlsx сохраняется");

        // Прогноз лёг в выбранный лист, а титульный остался на месте.
        let table = Table::read_sheet(&output, Some("Опыты"), Delimiter::Auto, true).unwrap();
        assert!(table.header().unwrap().contains(&"влажность".to_string()));
        assert_eq!(table.rows().len(), 1);
        let title = Table::read_sheet(&output, Some("Титульный"), Delimiter::Auto, false).unwrap();
        assert_eq!(title.rows()[0], ["отчёт"]);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Экспорт в .xlsx сохраняет исходную книгу: меняется выбранный лист, всё
    /// остальное остаётся тем же файлом.
    #[test]
    fn export_into_xlsx_preserves_the_source_workbook() {
        let input = tmp_path("preserve_in.xlsx");
        let output = tmp_path("preserve_out.xlsx");
        crate::table::write_test_workbook(
            &input,
            &[(
                "Опыты",
                &[
                    &["заметка", "материал", "температура", "влажность"],
                    &["хорошо", "глина", "70", "0"],
                    &["плохо", "песок", "60", "0"],
                ],
            )],
        );

        let summary = export_predictions(
            input.to_str().unwrap(),
            None,
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap();
        assert!(summary.preserved);
        assert_eq!(summary.replaced, vec!["влажность".to_string()]);
        assert_eq!(summary.added, vec!["плотность".to_string()]);

        let table = Table::read_path(&output, Delimiter::Auto, true).unwrap();
        // Недостающий выход добавлен справа вместе с заголовком.
        assert_eq!(
            table.header().unwrap(),
            [
                "заметка",
                "материал",
                "температура",
                "влажность",
                "плотность"
            ]
        );
        // Посторонние колонки остались собой, выход перезаписан прогнозом:
        // 70 + 1 (глина) = 71 и 70 - 1 = 69.
        assert_eq!(table.rows()[0], ["хорошо", "глина", "70", "71", "69"]);
        assert_eq!(table.rows()[1], ["плохо", "песок", "60", "60", "60"]);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// У текстовой таблицы сохранять нечего: получается новая книга, и отчёт
    /// говорит об этом прямо.
    #[test]
    fn export_from_a_text_table_builds_a_new_workbook() {
        let input = tmp_path("plain_in.csv");
        let output = tmp_path("plain_out.xlsx");
        std::fs::write(
            &input,
            "материал,температура,влажность
глина,70,0
",
        )
        .unwrap();

        let summary = export_predictions(
            input.to_str().unwrap(),
            None,
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap();
        assert!(!summary.preserved, "исходной книги не было");
        let (header, rows) = read_back(&output);
        assert_eq!(
            header,
            ["материал", "температура", "влажность", "плотность"]
        );
        assert_eq!(rows[0], ["глина", "70", "71", "69"]);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
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
            None,
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
            None,
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
    fn export_rejects_ambiguous_headers_and_bad_prediction_shapes() {
        let duplicate = tmp_path("duplicate.csv");
        std::fs::write(
            &duplicate,
            "температура,материал,температура\n70,глина,80\n",
        )
        .unwrap();
        let err = export_predictions(
            duplicate.to_str().unwrap(),
            None,
            tmp_path("unused_duplicate.xlsx").to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap_err();
        assert!(
            err.contains("дублирующийся заголовок 'температура'"),
            "{err}"
        );

        let valid = tmp_path("bad_shape.csv");
        std::fs::write(&valid, "температура,материал\n70,глина\n").unwrap();
        let err = export_predictions(
            valid.to_str().unwrap(),
            None,
            tmp_path("unused_shape.xlsx").to_str().unwrap(),
            &schema(),
            |inputs| {
                Ok(Predictions {
                    outputs: Array2::zeros((inputs.nrows(), 1)),
                    warnings: Vec::new(),
                })
            },
        )
        .unwrap_err();
        assert!(err.contains("ожидалось 1×2"), "{err}");

        let _ = std::fs::remove_file(duplicate);
        let _ = std::fs::remove_file(valid);
    }

    #[test]
    fn export_refuses_to_overwrite_the_source() {
        let input = tmp_path("same_path.csv");
        let contents = "температура,материал\n70,глина\n";
        std::fs::write(&input, contents).unwrap();
        let err = export_predictions(
            input.to_str().unwrap(),
            None,
            input.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap_err();
        assert!(err.contains("не должен перезаписывать"), "{err}");
        assert_eq!(std::fs::read_to_string(&input).unwrap(), contents);
        let _ = std::fs::remove_file(input);
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
            None,
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
            None,
            tmp_path("unused2.xlsx").to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap_err();
        assert!(err.contains("строка 4"), "{err}");
        assert!(err.contains("мрамор"), "{err}");
        let _ = std::fs::remove_file(input);
    }

    /// Экспорт поверх вчерашнего результата заменяет его целиком и читается
    /// обратно: архив дописывается во временный файл и лишь потом становится
    /// назначением.
    #[test]
    fn export_replaces_an_existing_workbook() {
        let input = tmp_path("replace.csv");
        let output = tmp_path("replace_out.xlsx");
        std::fs::write(&input, "температура,материал\n70,глина\n").unwrap();
        std::fs::write(&output, "не книга, а обрубок прошлого запуска").unwrap();

        export_predictions(
            input.to_str().unwrap(),
            None,
            output.to_str().unwrap(),
            &schema(),
            double,
        )
        .unwrap();

        let (headers, rows) = read_back(&output);
        assert!(headers.contains(&"влажность".to_string()), "{headers:?}");
        assert_eq!(rows.len(), 1);

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
