//! Точечная правка книги `.xlsx`: копия пакета, изменённый лист.
//!
//! Экспорт прогнозов писал НОВУЮ книгу, и от исходного файла не оставалось
//! ничего: ни стилей, ни формул, ни других листов, ни ширины колонок. Здесь
//! результат получается иначе — копией пакета, в которой изменён только
//! выбранный worksheet.
//!
//! Контракт ровно такой: **изменяются выбранный лист и служебная
//! метаинформация расчёта**. Обещать «ровно один worksheet.xml» нельзя —
//! записанное значение может кормить формулы на других листах, поэтому кэш
//! порядка вычислений удаляется, а книге предписывается полный пересчёт при
//! открытии. Всё остальное — другие листы, стили, `sharedStrings`, картинки,
//! настройки, неизвестные нам части — переносится байт в байт.
//!
//! Что НЕ обещано и потому отвергается явно:
//!
//! - подписанные пакеты: любая правка делает подпись недействительной;
//! - `.xlsm` и книги с `vbaProject.bin`: сохранность макросов — отдельная
//!   работа со своей проверкой;
//! - расширение «умных таблиц» (`tableN.xml`) и автофильтров: их диапазоны
//!   живут в своих частях, и молча разойтись с данными они не должны.

use crate::atomic_write::{same_file, write_atomically};
use std::collections::BTreeMap;
use std::io::{self, Read, Seek, Write};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

/// Что кладут в ячейку.
///
/// Текст пишется как `inlineStr`: так `sharedStrings.xml` остаётся
/// неизменным, а значит неизменными остаются и все ссылки на его индексы.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CellValue {
    Number(f64),
    Text(String),
}

/// Правка одной ячейки по физическим координатам листа (1-based).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CellEdit {
    pub row: usize,
    pub column: usize,
    pub value: CellValue,
}

/// Части пакета, которые правка затрагивает помимо самого листа.
const CONTENT_TYPES: &str = "[Content_Types].xml";
const WORKBOOK: &str = "xl/workbook.xml";
const WORKBOOK_RELS: &str = "xl/_rels/workbook.xml.rels";

/// Записать значения в лист `sheet` книги `input`, сохранив остальную книгу.
///
/// `output` пишется атомарно: до успешного завершения архива прежний файл по
/// этому пути не трогается.
pub(crate) fn patch_sheet(
    input: &Path,
    output: &Path,
    sheet: &str,
    edits: &[CellEdit],
) -> Result<(), String> {
    if same_file(input, output) {
        return Err(
            "входной и выходной путь совпадают: правка не должна идти поверх исходной книги"
                .to_string(),
        );
    }
    let file =
        std::fs::File::open(input).map_err(|e| format!("чтение {}: {e}", input.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| format!("{}: не читается как .xlsx: {e}", input.display()))?;
    reject_unsupported(&archive, input)?;

    let content_types = read_part(&mut archive, CONTENT_TYPES)?;
    let workbook = read_part(&mut archive, WORKBOOK)?;
    let rels = read_part(&mut archive, WORKBOOK_RELS)?;
    let sheet_part =
        worksheet_part(&workbook, &rels, sheet).map_err(|e| format!("{}: {e}", input.display()))?;
    let patched_sheet = patch_worksheet(&read_part(&mut archive, &sheet_part)?, edits)
        .map_err(|e| format!("{}, лист '{sheet}': {e}", input.display()))?;

    // Кэш порядка вычислений восстанавливается приложением, а согласовать его
    // с правкой мы не можем: удаляем целиком, когда он есть. Полный пересчёт
    // просим всегда: calcChain необязателен, поэтому его отсутствие ничего не
    // говорит о наличии формул и актуальности их сохранённых значений.
    let calc_chain = calculation_chain_part(&rels)?;
    if let Some(part) = &calc_chain {
        if archive.index_for_name(part).is_none() {
            return Err(format!(
                "{}: связь calcChain указывает на отсутствующую часть {part}",
                input.display()
            ));
        }
    }

    let mut replacements: BTreeMap<&str, String> = BTreeMap::new();
    replacements.insert(sheet_part.as_str(), patched_sheet);
    replacements.insert(WORKBOOK, force_full_calc(&workbook));
    if let Some(part) = &calc_chain {
        replacements.insert(WORKBOOK_RELS, drop_calc_chain_relationship(&rels));
        replacements.insert(
            CONTENT_TYPES,
            drop_calc_chain_override(&content_types, part),
        );
    }

    write_atomically(output, |file| {
        copy_package(&mut archive, file, &replacements, calc_chain.as_deref())
    })
    .map_err(|e| format!("запись {}: {e}", output.display()))
}

/// Пакеты, которые эта правка обслуживать не берётся.
fn reject_unsupported<R: Read + Seek>(archive: &ZipArchive<R>, path: &Path) -> Result<(), String> {
    let has = |name: &str| archive.index_for_name(name).is_some();
    if archive
        .file_names()
        .any(|name| name.starts_with("_xmlsignatures/"))
    {
        return Err(format!(
            "{}: книга подписана, а любая правка делает подпись недействительной",
            path.display()
        ));
    }
    if has("xl/vbaProject.bin") {
        return Err(format!(
            "{}: книга с макросами (.xlsm) пока не правится: сохранность vbaProject не проверена",
            path.display()
        ));
    }
    if archive
        .file_names()
        .any(|name| name.starts_with("xl/tables/"))
    {
        return Err(format!(
            "{}: в книге есть «умная таблица»: её диапазон живёт отдельной частью и разойдётся с данными",
            path.display()
        ));
    }
    Ok(())
}

fn read_part<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Result<String, String> {
    let mut part = archive
        .by_name(name)
        .map_err(|_| format!("в книге нет обязательной части {name}"))?;
    let mut text = String::new();
    part.read_to_string(&mut text)
        .map_err(|e| format!("{name}: не читается как XML: {e}"))?;
    Ok(text)
}

/// Скопировать пакет, заменив перечисленные части.
fn copy_package<R: Read + Seek, W: Write + Seek>(
    archive: &mut ZipArchive<R>,
    out: W,
    replacements: &BTreeMap<&str, String>,
    drop_part: Option<&str>,
) -> io::Result<()> {
    let mut zip = ZipWriter::new(out);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for index in 0..archive.len() {
        let entry = archive.by_index_raw(index).map_err(io::Error::other)?;
        let name = entry.name().to_string();
        if drop_part == Some(name.as_str()) {
            continue;
        }
        match replacements.get(name.as_str()) {
            // Изменённые части пишутся заново...
            Some(text) => {
                let text = text.clone();
                drop(entry);
                zip.start_file(&name, options).map_err(io::Error::other)?;
                zip.write_all(text.as_bytes())?;
            }
            // ...а всё остальное переносится как есть, без перекодирования:
            // так непричастные части остаются теми же байтами.
            None => zip.raw_copy_file(entry).map_err(io::Error::other)?,
        }
    }
    zip.finish().map_err(io::Error::other)?;
    Ok(())
}

// --- workbook.xml: лист -> часть пакета ---

/// Найти часть выбранного листа.
///
/// Имя листа связано с частью через `r:id` и `workbook.xml.rels`;
/// предполагать `worksheets/sheetN.xml` нельзя — нумерация частей и порядок
/// листов совпадают далеко не всегда.
fn worksheet_part(workbook: &str, rels: &str, sheet: &str) -> Result<String, String> {
    let mut rest = workbook;
    let mut rid = None;
    while let Some(tag) = next_tag(&mut rest, "<sheet ") {
        if attribute(tag, "name").as_deref() == Some(sheet) {
            rid = attribute(tag, "r:id").or_else(|| attribute(tag, "relationshipId"));
            break;
        }
    }
    let rid = rid.ok_or_else(|| format!("листа '{sheet}' нет в workbook.xml"))?;

    let mut rest = rels;
    while let Some(tag) = next_tag(&mut rest, "<Relationship ") {
        if attribute(tag, "Id").as_deref() != Some(rid.as_str()) {
            continue;
        }
        let target = attribute(tag, "Target").ok_or_else(|| format!("связь {rid} без Target"))?;
        return Ok(resolve_target(&target));
    }
    Err(format!("в workbook.xml.rels нет связи {rid}"))
}

/// Путь части относительно `xl/`; абсолютный Target начинается с `/`.
fn resolve_target(target: &str) -> String {
    match target.strip_prefix('/') {
        Some(absolute) => absolute.to_string(),
        None => format!("xl/{}", target.trim_start_matches("./")),
    }
}

/// Следующий тег, начинающийся с `prefix`; курсор сдвигается за него.
fn next_tag<'a>(rest: &mut &'a str, prefix: &str) -> Option<&'a str> {
    let start = rest.find(prefix)?;
    let end = rest[start..].find('>')? + start;
    let tag = &rest[start..=end];
    *rest = &rest[end + 1..];
    Some(tag)
}

/// Значение атрибута тега. Кавычки бывают и одинарные.
fn attribute(tag: &str, name: &str) -> Option<String> {
    let mut rest = tag;
    while let Some(at) = rest.find(name) {
        let after = &rest[at + name.len()..];
        let before_ok = at == 0
            || rest[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_whitespace() || c == '<');
        let value = after.trim_start().strip_prefix('=').map(str::trim_start);
        match (before_ok, value) {
            (true, Some(value)) => {
                let quote = value.chars().next()?;
                if quote != '"' && quote != '\'' {
                    return None;
                }
                let end = value[1..].find(quote)? + 1;
                return Some(unescape(&value[1..end]));
            }
            _ => rest = &rest[at + name.len()..],
        }
    }
    None
}

// --- calcChain: удаление и принудительный пересчёт ---

/// Убрать связь на `calcChain.xml` из `workbook.xml.rels`.
fn drop_calc_chain_relationship(rels: &str) -> String {
    remove_elements(rels, "<Relationship ", |tag| {
        attribute(tag, "Type").is_some_and(|kind| relationship_kind(&kind) == "calcChain")
    })
}

/// Убрать объявление типа `calcChain.xml` из `[Content_Types].xml`.
fn drop_calc_chain_override(types: &str, part: &str) -> String {
    remove_elements(types, "<Override ", |tag| {
        attribute(tag, "PartName").is_some_and(|name| name.trim_start_matches('/') == part)
    })
}

/// Часть цепочки определяется связью workbook, а не условным именем файла.
fn calculation_chain_part(rels: &str) -> Result<Option<String>, String> {
    let mut rest = rels;
    let mut found = None;
    while let Some(tag) = next_tag(&mut rest, "<Relationship ") {
        let Some(kind) = attribute(tag, "Type") else {
            continue;
        };
        if relationship_kind(&kind) != "calcChain" {
            continue;
        }
        if found.is_some() {
            return Err("в workbook.xml.rels несколько связей calcChain".to_string());
        }
        let target =
            attribute(tag, "Target").ok_or_else(|| "связь calcChain без Target".to_string())?;
        found = Some(resolve_target(&target));
    }
    Ok(found)
}

fn relationship_kind(kind: &str) -> &str {
    kind.rsplit('/').next().unwrap_or(kind)
}

fn remove_elements(xml: &str, prefix: &str, drop: impl Fn(&str) -> bool) -> String {
    let mut out = String::with_capacity(xml.len());
    let mut rest = xml;
    while let Some(start) = rest.find(prefix) {
        let Some(end) = rest[start..].find('>').map(|e| e + start) else {
            break;
        };
        let tag = &rest[start..=end];
        out.push_str(&rest[..start]);
        if !drop(tag) {
            out.push_str(tag);
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Предписать книге полный пересчёт при открытии.
///
/// Режим пересчёта (`calcMode`) — настройка пользователя и сохраняется как
/// есть: мы говорим «пересчитай всё сейчас», а не «считай иначе».
fn force_full_calc(workbook: &str) -> String {
    let full = |tag: &str| {
        let tag = set_attribute(tag, "fullCalcOnLoad", "1");
        set_attribute(&tag, "forceFullCalc", "1")
    };
    let mut rest = workbook;
    let mut out = String::with_capacity(workbook.len() + 64);
    if let Some(start) = rest.find("<calcPr") {
        let Some(end) = rest[start..].find('>').map(|e| e + start) else {
            return workbook.to_string();
        };
        out.push_str(&rest[..start]);
        out.push_str(&full(&rest[start..=end]));
        rest = &rest[end + 1..];
        out.push_str(rest);
        return out;
    }
    // Книги без <calcPr> существуют: тогда элемент добавляется перед первым
    // элементом, который по схеме CT_Workbook идёт после calcPr. Простое
    // добавление перед </workbook> поставило бы calcPr после extLst и сделало
    // бы формально валидный XML невалидным SpreadsheetML.
    match calc_pr_insertion_point(workbook) {
        Some(at) => {
            let mut out = String::with_capacity(workbook.len() + 64);
            out.push_str(&workbook[..at]);
            out.push_str(r#"<calcPr fullCalcOnLoad="1" forceFullCalc="1"/>"#);
            out.push_str(&workbook[at..]);
            out
        }
        None => workbook.to_string(),
    }
}

fn calc_pr_insertion_point(workbook: &str) -> Option<usize> {
    const AFTER_CALC_PR: [&str; 9] = [
        "<oleSize",
        "<customWorkbookViews",
        "<pivotCaches",
        "<smartTagPr",
        "<smartTagTypes",
        "<webPublishing",
        "<fileRecoveryPr",
        "<webPublishObjects",
        "<extLst",
    ];
    AFTER_CALC_PR
        .iter()
        .filter_map(|tag| workbook.find(tag))
        .min()
        .or_else(|| workbook.rfind("</workbook>"))
}

/// Заменить значение атрибута или дописать его в конец тега.
fn set_attribute(tag: &str, name: &str, value: &str) -> String {
    let self_closing = tag.trim_end().ends_with("/>");
    let body_end = tag.len() - if self_closing { 2 } else { 1 };
    let body = &tag[..body_end];
    let tail = if self_closing { "/>" } else { ">" };

    if let Some(at) = find_attribute(body, name) {
        let after = &body[at + name.len()..];
        let Some(value_start) = after.find(['"', '\'']) else {
            return tag.to_string();
        };
        let quote = after[value_start..].chars().next().unwrap_or('"');
        let Some(value_end) = after[value_start + 1..].find(quote) else {
            return tag.to_string();
        };
        let head = &body[..at];
        let rest = &after[value_start + 1 + value_end + 1..];
        return format!(r#"{head}{name}="{value}"{rest}{tail}"#);
    }
    format!(r#"{} {name}="{value}"{tail}"#, body.trim_end())
}

/// Позиция атрибута `name` в теле тега — именно атрибута, а не подстроки
/// внутри имени соседнего.
fn find_attribute(body: &str, name: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(at) = body[from..].find(name).map(|a| a + from) {
        let before_ok = body[..at]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace);
        let after_ok = body[at + name.len()..].trim_start().starts_with('=');
        if before_ok && after_ok {
            return Some(at);
        }
        from = at + name.len();
    }
    None
}

// --- worksheet: ячейки и dimension ---

/// Записать значения в лист, не пересобирая остальной XML.
fn patch_worksheet(xml: &str, edits: &[CellEdit]) -> Result<String, String> {
    if edits.is_empty() {
        return Ok(xml.to_string());
    }
    let mut by_row: BTreeMap<usize, BTreeMap<usize, &CellValue>> = BTreeMap::new();
    for edit in edits {
        if edit.row == 0 || edit.column == 0 {
            return Err("координаты ячейки считаются с единицы".to_string());
        }
        by_row
            .entry(edit.row)
            .or_default()
            .insert(edit.column, &edit.value);
    }

    let (data_start, data_end, open_tag) = sheet_data_bounds(xml)?;
    let patched = patch_sheet_data(&xml[data_start..data_end], &by_row)?;
    let mut out = String::with_capacity(xml.len() + patched.len());
    out.push_str(&xml[..data_start]);
    out.push_str(&patched);
    out.push_str(&xml[data_end..]);
    if open_tag {
        // `<sheetData/>` раскрывается в пару тегов только если в лист что-то
        // добавили.
        out = out.replacen("<sheetData/>", "<sheetData></sheetData>", 1);
    }

    let max_row = by_row.keys().copied().max().unwrap_or(1);
    let max_col = by_row
        .values()
        .filter_map(|row| row.keys().copied().max())
        .max()
        .unwrap_or(1);
    Ok(expand_dimension(&out, max_row, max_col))
}

/// Границы содержимого `<sheetData>`; третий элемент — был ли тег пустым.
fn sheet_data_bounds(xml: &str) -> Result<(usize, usize, bool), String> {
    if let Some(at) = xml.find("<sheetData/>") {
        let inner = at + "<sheetData>".len();
        return Ok((inner, inner, true));
    }
    let open = xml
        .find("<sheetData")
        .ok_or_else(|| "в листе нет <sheetData>".to_string())?;
    let open_end = xml[open..]
        .find('>')
        .map(|e| e + open + 1)
        .ok_or_else(|| "незакрытый тег <sheetData>".to_string())?;
    let close = xml
        .find("</sheetData>")
        .ok_or_else(|| "в листе нет </sheetData>".to_string())?;
    if close < open_end {
        return Err("повреждённый <sheetData>".to_string());
    }
    Ok((open_end, close, false))
}

/// Пройти строки листа, подставляя значения и добавляя недостающие строки.
fn patch_sheet_data(
    data: &str,
    by_row: &BTreeMap<usize, BTreeMap<usize, &CellValue>>,
) -> Result<String, String> {
    let mut out = String::with_capacity(data.len());
    let mut rest = data;
    let mut pending = by_row.iter().peekable();

    while let Some(start) = rest.find("<row") {
        let (element, after) = element_at(rest, start, "row")?;
        let number: usize = attribute(element, "r")
            .ok_or_else(|| "строка листа без номера r".to_string())?
            .parse()
            .map_err(|_| "номер строки листа не число".to_string())?;
        out.push_str(&rest[..start]);
        // Все правки строк, которых в листе нет, дописываются перед текущей.
        while pending.peek().is_some_and(|(&row, _)| row < number) {
            let (&row, cells) = pending.next().expect("peek подтвердил элемент");
            out.push_str(&new_row(row, cells));
        }
        match pending.peek() {
            Some((&row, _)) if row == number => {
                let (_, cells) = pending.next().expect("peek подтвердил элемент");
                out.push_str(&patch_row(element, cells)?);
            }
            _ => out.push_str(element),
        }
        rest = after;
    }
    // Хвост строк, которые идут после последней существующей.
    let mut tail = String::new();
    for (&row, cells) in pending {
        tail.push_str(&new_row(row, cells));
    }
    match rest.rfind("</row>") {
        // Новые строки идут после последней существующей, а не после её
        // хвостовых пробелов.
        Some(at) if !tail.is_empty() => {
            let split = at + "</row>".len();
            out.push_str(&rest[..split]);
            out.push_str(&tail);
            out.push_str(&rest[split..]);
        }
        _ => {
            out.push_str(&tail);
            out.push_str(rest);
        }
    }
    Ok(out)
}

/// Элемент целиком: `<tag .../>` или `<tag ...>…</tag>`.
fn element_at<'a>(xml: &'a str, start: usize, tag: &str) -> Result<(&'a str, &'a str), String> {
    let head_end = xml[start..]
        .find('>')
        .map(|e| e + start)
        .ok_or_else(|| format!("незакрытый тег <{tag}>"))?;
    if xml[start..head_end].ends_with('/') {
        return Ok((&xml[start..=head_end], &xml[head_end + 1..]));
    }
    let close = format!("</{tag}>");
    let close_at = xml[head_end..]
        .find(&close)
        .map(|e| e + head_end)
        .ok_or_else(|| format!("нет закрывающего </{tag}>"))?;
    let end = close_at + close.len();
    Ok((&xml[start..end], &xml[end..]))
}

/// Заменить ячейки внутри строки, сохранив всё остальное.
fn patch_row(element: &str, cells: &BTreeMap<usize, &CellValue>) -> Result<String, String> {
    let row: usize = attribute(element, "r")
        .ok_or_else(|| "строка листа без номера r".to_string())?
        .parse()
        .map_err(|_| "номер строки листа не число".to_string())?;

    let (head, inner, tail) = if element.trim_end().ends_with("/>") {
        // Пустая строка: раскрываем в пару тегов, чтобы было куда писать.
        let head_end = element.len() - 2;
        (
            format!("{}>", element[..head_end].trim_end()),
            "",
            "</row>".to_string(),
        )
    } else {
        let head_end = element
            .find('>')
            .ok_or_else(|| "незакрытый тег <row>".to_string())?;
        let close = element
            .rfind("</row>")
            .ok_or_else(|| "нет закрывающего </row>".to_string())?;
        (
            element[..=head_end].to_string(),
            &element[head_end + 1..close],
            element[close..].to_string(),
        )
    };

    let mut out = String::with_capacity(element.len() + 64 * cells.len());
    out.push_str(&head);
    let mut rest = inner;
    let mut pending = cells.iter().peekable();
    while let Some(start) = rest.find("<c") {
        let (cell, after) = element_at(rest, start, "c")?;
        let reference = attribute(cell, "r").ok_or_else(|| "ячейка без адреса r".to_string())?;
        let column = column_of(&reference)?;
        out.push_str(&rest[..start]);
        while pending.peek().is_some_and(|(&c, _)| c < column) {
            let (&c, value) = pending.next().expect("peek подтвердил элемент");
            out.push_str(&cell_xml(row, c, None, value));
        }
        match pending.peek() {
            Some((&c, _)) if c == column => {
                let (_, value) = pending.next().expect("peek подтвердил элемент");
                // Стиль ячейки — оформление, заданное человеком: оно
                // переживает запись значения. Формула не переживает: значение
                // пишут вместо неё.
                out.push_str(&cell_xml(
                    row,
                    column,
                    attribute(cell, "s").as_deref(),
                    value,
                ));
            }
            _ => out.push_str(cell),
        }
        rest = after;
    }
    for (&c, value) in pending {
        out.push_str(&cell_xml(row, c, None, value));
    }
    out.push_str(rest);
    out.push_str(&tail);
    Ok(out)
}

fn new_row(row: usize, cells: &BTreeMap<usize, &CellValue>) -> String {
    let mut out = format!(r#"<row r="{row}">"#);
    for (&column, value) in cells {
        out.push_str(&cell_xml(row, column, None, value));
    }
    out.push_str("</row>");
    out
}

fn cell_xml(row: usize, column: usize, style: Option<&str>, value: &CellValue) -> String {
    let reference = format!("{}{row}", column_letters(column));
    let style = style.map(|s| format!(r#" s="{s}""#)).unwrap_or_default();
    match value {
        CellValue::Number(v) => format!(r#"<c r="{reference}"{style}><v>{v}</v></c>"#),
        CellValue::Text(text) => format!(
            r#"<c r="{reference}"{style} t="inlineStr"><is><t>{}</t></is></c>"#,
            escape(text)
        ),
    }
}

/// Расширить объявленный диапазон листа. Только расширить: сузить его — значит
/// соврать про данные, которых мы не читали.
fn expand_dimension(xml: &str, max_row: usize, max_col: usize) -> String {
    let Some(start) = xml.find("<dimension") else {
        return xml.to_string();
    };
    let Some(end) = xml[start..].find('>').map(|e| e + start) else {
        return xml.to_string();
    };
    let tag = &xml[start..=end];
    let Some(reference) = attribute(tag, "ref") else {
        return xml.to_string();
    };
    let (first, last) = match reference.split_once(':') {
        Some((first, last)) => (first.to_string(), last.to_string()),
        None => (reference.clone(), reference.clone()),
    };
    let (Ok(last_col), Ok(last_row)) = (column_of(&last), row_of(&last)) else {
        return xml.to_string();
    };
    let widened = format!(
        "{first}:{}{}",
        column_letters(last_col.max(max_col)),
        last_row.max(max_row)
    );
    if widened == reference {
        return xml.to_string();
    }
    let mut out = String::with_capacity(xml.len() + 8);
    out.push_str(&xml[..start]);
    out.push_str(&set_attribute(tag, "ref", &widened));
    out.push_str(&xml[end + 1..]);
    out
}

// --- адреса ячеек ---

/// Номер колонки (1-based) из адреса вида `BC12`.
fn column_of(reference: &str) -> Result<usize, String> {
    let letters: String = reference
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if letters.is_empty() {
        return Err(format!("адрес ячейки '{reference}' без колонки"));
    }
    Ok(letters.chars().fold(0usize, |acc, c| {
        acc * 26 + (c.to_ascii_uppercase() as usize - 'A' as usize + 1)
    }))
}

fn row_of(reference: &str) -> Result<usize, String> {
    reference
        .chars()
        .skip_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .parse()
        .map_err(|_| format!("адрес ячейки '{reference}' без строки"))
}

/// Буквенное имя колонки (1 -> A, 27 -> AA).
pub(crate) fn column_letters(mut column: usize) -> String {
    let mut letters = Vec::new();
    while column > 0 {
        let rem = (column - 1) % 26;
        letters.push((b'A' + rem as u8) as char);
        column = (column - 1) / 26;
    }
    letters.iter().rev().collect()
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
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

fn unescape(text: &str) -> String {
    text.replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const DEFAULT_CALC_CHAIN: &str = "xl/calcChain.xml";

    /// Книга собирается из явных частей: правка обязана не трогать ни одну из
    /// них, кроме листа и метаданных расчёта.
    fn write_book(path: &Path, parts: &[(&str, &str)]) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, content) in parts {
            zip.start_file(*name, options).unwrap();
            zip.write_all(content.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
    }

    fn parts_of(path: &Path) -> BTreeMap<String, Vec<u8>> {
        let file = std::fs::File::open(path).unwrap();
        let mut archive = ZipArchive::new(file).unwrap();
        let mut out = BTreeMap::new();
        for i in 0..archive.len() {
            let mut part = archive.by_index(i).unwrap();
            let mut bytes = Vec::new();
            part.read_to_end(&mut bytes).unwrap();
            out.insert(part.name().to_string(), bytes);
        }
        out
    }

    fn part_names(path: &Path) -> Vec<String> {
        let file = std::fs::File::open(path).unwrap();
        let archive = ZipArchive::new(file).unwrap();
        archive.file_names().map(str::to_string).collect()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("transformer_xlsx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    const TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/data.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/calcChain.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.calcChain+xml"/></Types>"#;

    const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;

    /// Порядок листов и имена частей намеренно не совпадают: путь берётся из
    /// связи, а не из «sheetN.xml».
    const BOOK: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Титульный" sheetId="1" r:id="rId7"/><sheet name="Опыты" sheetId="2" r:id="rId4"/></sheets><calcPr calcId="191029" calcMode="manual"/></workbook>"#;

    const BOOK_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/data.xml"/><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/calcChain" Target="calcChain.xml"/></Relationships>"#;

    const TITLE_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>отчёт</t></is></c></row></sheetData></worksheet>"#;

    /// Лист данных: стиль у B2, формула у C2 — обе детали важны при правке.
    const DATA_SHEET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:C3"/><cols><col min="1" max="3" width="12"/></cols><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>x0</t></is></c><c r="B1" t="inlineStr"><is><t>y0</t></is></c></row><row r="2"><c r="A2"><v>1</v></c><c r="B2" s="4"><v>2</v></c><c r="C2"><f>A2*2</f><v>2</v></c></row><row r="3"><c r="A3"><v>3</v></c><c r="B3"><v>4</v></c></row></sheetData><pageMargins left="0.7"/></worksheet>"#;

    const CALC_CHAIN_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<calcChain xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><c r="C2" i="2"/></calcChain>"#;

    const STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>"#;

    fn sample(path: &Path) {
        write_book(
            path,
            &[
                ("[Content_Types].xml", TYPES),
                ("_rels/.rels", ROOT_RELS),
                ("xl/workbook.xml", BOOK),
                ("xl/_rels/workbook.xml.rels", BOOK_RELS),
                ("xl/worksheets/sheet1.xml", TITLE_SHEET),
                ("xl/worksheets/data.xml", DATA_SHEET),
                ("xl/calcChain.xml", CALC_CHAIN_XML),
                ("xl/styles.xml", STYLES),
            ],
        );
    }

    fn number(row: usize, column: usize, v: f64) -> CellEdit {
        CellEdit {
            row,
            column,
            value: CellValue::Number(v),
        }
    }

    fn text(row: usize, column: usize, t: &str) -> CellEdit {
        CellEdit {
            row,
            column,
            value: CellValue::Text(t.to_string()),
        }
    }

    /// Главное обещание: посторонние части книги остаются теми же байтами.
    #[test]
    fn untouched_parts_survive_byte_for_byte() {
        let input = tmp("keep_in.xlsx");
        let output = tmp("keep_out.xlsx");
        sample(&input);

        patch_sheet(&input, &output, "Опыты", &[number(2, 2, 42.0)]).unwrap();

        let before = parts_of(&input);
        let after = parts_of(&output);
        for name in ["_rels/.rels", "xl/worksheets/sheet1.xml", "xl/styles.xml"] {
            assert_eq!(before[name], after[name], "часть {name} изменилась");
        }
        // Порядок частей в пакете тоже сохраняется, кроме удалённой. BTreeMap
        // выше этого не проверяет: он сортирует имена сам.
        let expected_names: Vec<String> = part_names(&input)
            .into_iter()
            .filter(|name| name != DEFAULT_CALC_CHAIN)
            .collect();
        assert_eq!(part_names(&output), expected_names);
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Значение попадает в нужную ячейку нужного листа, стиль остаётся,
    /// формула уступает место значению.
    #[test]
    fn a_value_replaces_the_cell_keeping_its_style() {
        let input = tmp("cell_in.xlsx");
        let output = tmp("cell_out.xlsx");
        sample(&input);

        patch_sheet(
            &input,
            &output,
            "Опыты",
            &[number(2, 2, 42.5), number(2, 3, 7.0)],
        )
        .unwrap();

        let sheet = String::from_utf8(parts_of(&output)["xl/worksheets/data.xml"].clone()).unwrap();
        assert!(
            sheet.contains(r#"<c r="B2" s="4"><v>42.5</v></c>"#),
            "{sheet}"
        );
        assert!(sheet.contains(r#"<c r="C2"><v>7</v></c>"#), "{sheet}");
        assert!(!sheet.contains("<f>"), "формула уступает место значению");
        // Соседние ячейки не тронуты.
        assert!(sheet.contains(r#"<c r="A2"><v>1</v></c>"#), "{sheet}");
        assert!(sheet.contains("<pageMargins"), "хвост листа на месте");
        assert!(sheet.contains(r#"<cols><col min="1" max="3" width="12"/></cols>"#));
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Новые колонка и строка появляются на своих местах: порядок `<c>` и
    /// `<row>` в SpreadsheetML возрастающий.
    #[test]
    fn new_cells_and_rows_are_inserted_in_order() {
        let input = tmp("insert_in.xlsx");
        let output = tmp("insert_out.xlsx");
        sample(&input);

        patch_sheet(
            &input,
            &output,
            "Опыты",
            &[text(1, 3, "прогноз"), number(3, 3, 9.0), number(5, 1, 5.0)],
        )
        .unwrap();

        let sheet = String::from_utf8(parts_of(&output)["xl/worksheets/data.xml"].clone()).unwrap();
        let at = |needle: &str| {
            sheet
                .find(needle)
                .unwrap_or_else(|| panic!("нет {needle}: {sheet}"))
        };
        assert!(
            at(r#"<c r="B1""#) < at(r#"<c r="C1""#),
            "колонка справа: {sheet}"
        );
        assert!(
            at(r#"<row r="3""#) < at(r#"<row r="5""#),
            "строка ниже: {sheet}"
        );
        assert!(sheet.contains(r#"<c r="C1" t="inlineStr"><is><t>прогноз</t></is></c>"#));
        assert!(
            sheet.contains(r#"<row r="5"><c r="A5"><v>5</v></c></row>"#),
            "{sheet}"
        );
        // Диапазон расширен до правой нижней ячейки.
        assert!(sheet.contains(r#"<dimension ref="A1:C5"/>"#), "{sheet}");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Кэш порядка вычислений удаляется целиком и согласованно, а книга
    /// получает указание пересчитать всё при открытии. Режим пересчёта —
    /// настройка пользователя и сохраняется.
    #[test]
    fn the_calculation_chain_is_dropped_consistently() {
        let input = tmp("calc_in.xlsx");
        let output = tmp("calc_out.xlsx");
        sample(&input);

        patch_sheet(&input, &output, "Опыты", &[number(2, 2, 1.0)]).unwrap();

        let after = parts_of(&output);
        assert!(!after.contains_key("xl/calcChain.xml"), "часть удалена");
        let rels = String::from_utf8(after["xl/_rels/workbook.xml.rels"].clone()).unwrap();
        assert!(!rels.contains("calcChain.xml"), "связь удалена: {rels}");
        assert!(rels.contains("worksheets/data.xml"), "остальные связи целы");
        let types = String::from_utf8(after["[Content_Types].xml"].clone()).unwrap();
        assert!(
            !types.contains("calcChain"),
            "объявление типа удалено: {types}"
        );
        assert!(
            types.contains("/xl/worksheets/data.xml"),
            "остальные типы целы"
        );
        let book = String::from_utf8(after["xl/workbook.xml"].clone()).unwrap();
        assert!(book.contains(r#"fullCalcOnLoad="1""#), "{book}");
        assert!(book.contains(r#"forceFullCalc="1""#), "{book}");
        assert!(
            book.contains(r#"calcMode="manual""#),
            "режим пользователя: {book}"
        );
        assert!(book.contains(r#"calcId="191029""#), "{book}");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// OPC не закрепляет имя части за `xl/calcChain.xml`: как и worksheet,
    /// цепочку нужно находить через relationship target.
    #[test]
    fn a_calculation_chain_with_a_custom_part_name_is_dropped() {
        let input = tmp("custom_calc_in.xlsx");
        let output = tmp("custom_calc_out.xlsx");
        let custom_part = "xl/calculation/cache.xml";
        let types = TYPES.replace("/xl/calcChain.xml", "/xl/calculation/cache.xml");
        let rels = BOOK_RELS.replace("calcChain.xml", "calculation/cache.xml");
        write_book(
            &input,
            &[
                ("[Content_Types].xml", &types),
                ("_rels/.rels", ROOT_RELS),
                ("xl/workbook.xml", BOOK),
                ("xl/_rels/workbook.xml.rels", &rels),
                ("xl/worksheets/sheet1.xml", TITLE_SHEET),
                ("xl/worksheets/data.xml", DATA_SHEET),
                (custom_part, CALC_CHAIN_XML),
                ("xl/styles.xml", STYLES),
            ],
        );

        patch_sheet(&input, &output, "Опыты", &[number(2, 2, 1.0)]).unwrap();

        let after = parts_of(&output);
        assert!(!after.contains_key(custom_part));
        assert!(!String::from_utf8(after[WORKBOOK_RELS].clone())
            .unwrap()
            .contains("calcChain"));
        assert!(!String::from_utf8(after[CONTENT_TYPES].clone())
            .unwrap()
            .contains("calcChain"));
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Формулы допустимы без calcChain: цепочка — необязательный кэш, поэтому
    /// её отсутствие не позволяет оставить сохранённые результаты формул как
    /// будто они актуальны после изменения входных значений.
    #[test]
    fn formulas_without_a_calculation_chain_still_force_recalculation() {
        let input = tmp("formula_without_chain_in.xlsx");
        let output = tmp("formula_without_chain_out.xlsx");
        let types = drop_calc_chain_override(TYPES, DEFAULT_CALC_CHAIN);
        let rels = drop_calc_chain_relationship(BOOK_RELS);
        write_book(
            &input,
            &[
                ("[Content_Types].xml", &types),
                ("_rels/.rels", ROOT_RELS),
                ("xl/workbook.xml", BOOK),
                ("xl/_rels/workbook.xml.rels", &rels),
                ("xl/worksheets/sheet1.xml", TITLE_SHEET),
                ("xl/worksheets/data.xml", DATA_SHEET),
                ("xl/styles.xml", STYLES),
            ],
        );

        patch_sheet(&input, &output, "Опыты", &[number(2, 2, 1.0)]).unwrap();

        let before = parts_of(&input);
        let after = parts_of(&output);
        for name in [
            "[Content_Types].xml",
            "_rels/.rels",
            "xl/_rels/workbook.xml.rels",
            "xl/worksheets/sheet1.xml",
            "xl/styles.xml",
        ] {
            assert_eq!(before[name], after[name], "часть {name} изменилась");
        }
        assert_ne!(
            before["xl/worksheets/data.xml"],
            after["xl/worksheets/data.xml"]
        );
        let book = String::from_utf8(after["xl/workbook.xml"].clone()).unwrap();
        assert!(book.contains(r#"fullCalcOnLoad="1""#), "{book}");
        assert!(book.contains(r#"forceFullCalc="1""#), "{book}");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Книга без `<calcPr>` тоже должна получить указание на пересчёт.
    #[test]
    fn a_workbook_without_calc_settings_gets_them() {
        let book = BOOK.replace(r#"<calcPr calcId="191029" calcMode="manual"/>"#, "");
        let patched = force_full_calc(&book);
        assert!(patched.contains(r#"<calcPr fullCalcOnLoad="1" forceFullCalc="1"/></workbook>"#));

        let with_tail = book.replace(
            "</workbook>",
            "<pivotCaches/><extLst><ext uri=\"keep\"/></extLst></workbook>",
        );
        let patched = force_full_calc(&with_tail);
        let calc = patched.find("<calcPr ").unwrap();
        assert!(calc < patched.find("<pivotCaches").unwrap(), "{patched}");
        assert!(patched.contains("<ext uri=\"keep\"/>"), "{patched}");
    }

    /// Подписанная книга и книга с макросами не правятся: подпись стала бы
    /// недействительной, а сохранность макросов не проверена.
    #[test]
    fn signed_and_macro_workbooks_are_refused() {
        let signed = tmp("signed.xlsx");
        write_book(
            &signed,
            &[
                ("[Content_Types].xml", TYPES),
                ("xl/workbook.xml", BOOK),
                ("_xmlsignatures/sig1.xml", "<Signature/>"),
            ],
        );
        let err = patch_sheet(
            &signed,
            &tmp("signed_out.xlsx"),
            "Опыты",
            &[number(1, 1, 1.0)],
        )
        .unwrap_err();
        assert!(err.contains("подписана"), "{err}");

        let macros = tmp("macros.xlsx");
        write_book(
            &macros,
            &[
                ("[Content_Types].xml", TYPES),
                ("xl/workbook.xml", BOOK),
                ("xl/vbaProject.bin", "не xml"),
            ],
        );
        let err = patch_sheet(
            &macros,
            &tmp("macros_out.xlsx"),
            "Опыты",
            &[number(1, 1, 1.0)],
        )
        .unwrap_err();
        assert!(err.contains("макрос"), "{err}");
        std::fs::remove_file(&signed).ok();
        std::fs::remove_file(&macros).ok();
    }

    /// «Умная таблица» держит свой диапазон отдельной частью: расширять его мы
    /// не обещали, поэтому такую книгу не правим вовсе.
    #[test]
    fn a_structured_table_is_refused() {
        let path = tmp("smart.xlsx");
        write_book(
            &path,
            &[
                ("[Content_Types].xml", TYPES),
                ("xl/workbook.xml", BOOK),
                ("xl/tables/table1.xml", "<table/>"),
            ],
        );
        let err =
            patch_sheet(&path, &tmp("smart_out.xlsx"), "Опыты", &[number(1, 1, 1.0)]).unwrap_err();
        assert!(err.contains("умная таблица"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_unknown_sheet_and_a_shared_path_are_refused() {
        let input = tmp("refuse_in.xlsx");
        sample(&input);
        let err = patch_sheet(
            &input,
            &tmp("refuse_out.xlsx"),
            "Нет такого",
            &[number(1, 1, 1.0)],
        )
        .unwrap_err();
        assert!(err.contains("нет в workbook.xml"), "{err}");

        let err = patch_sheet(&input, &input, "Опыты", &[number(1, 1, 1.0)]).unwrap_err();
        assert!(err.contains("совпадают"), "{err}");
        std::fs::remove_file(&input).ok();
    }

    /// Диапазон только расширяется: сузить его — соврать про данные, которых
    /// мы не читали.
    #[test]
    fn the_dimension_only_grows() {
        let xml = r#"<worksheet><dimension ref="A1:D10"/><sheetData></sheetData></worksheet>"#;
        assert!(expand_dimension(xml, 2, 2).contains(r#"ref="A1:D10""#));
        assert!(expand_dimension(xml, 12, 2).contains(r#"ref="A1:D12""#));
        assert!(expand_dimension(xml, 2, 27).contains(r#"ref="A1:AA10""#));
    }

    #[test]
    fn column_names_follow_excel() {
        assert_eq!(column_letters(1), "A");
        assert_eq!(column_letters(26), "Z");
        assert_eq!(column_letters(27), "AA");
        assert_eq!(column_of("AA12").unwrap(), 27);
        assert_eq!(row_of("AA12").unwrap(), 12);
    }

    /// Значение читается обратно тем же путём, которым его читает приложение.
    #[test]
    fn calamine_reads_the_patched_values() {
        let input = tmp("read_in.xlsx");
        let output = tmp("read_out.xlsx");
        sample(&input);

        patch_sheet(
            &input,
            &output,
            "Опыты",
            &[text(1, 3, "прогноз"), number(2, 3, 4.5), number(3, 3, 8.5)],
        )
        .unwrap();

        let table = crate::table::Table::read_sheet(
            &output,
            Some("Опыты"),
            crate::table::Delimiter::Auto,
            true,
        )
        .unwrap();
        assert_eq!(table.header().unwrap(), ["x0", "y0", "прогноз"]);
        assert_eq!(table.rows()[0], ["1", "2", "4.5"]);
        assert_eq!(table.rows()[1], ["3", "4", "8.5"]);
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
