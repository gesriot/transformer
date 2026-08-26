//! Данные и нормализация.
//!
//! - `Normalizer` — z-score по континуальным колонкам, identity по
//!   категориальным; хранит диапазоны для предупреждения об экстраполяции.
//! - `NumericDataset` — пары вход/выход и выборка строк. Разбиение живёт в
//!   `split.rs`: оно отвечает не за данные, а за протокол оценки.
//! - `TextDataset` + `Vocab` — char-уровень, нарезка контекст→продолжение.

use crate::encoders::FeatureSpec;
use crate::schema::{Column, ColumnRole, ColumnType, ModelSchema};
use ndarray::Array2;
#[cfg(feature = "demo")]
use rand::rngs::StdRng;
#[cfg(feature = "demo")]
use rand::Rng;
#[cfg(feature = "demo")]
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::{self, ErrorKind};

/// Деталь экстраполяции: континуальный признак вне обученного диапазона.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutOfRange {
    pub feature: usize,
    pub value: f32,
    pub min: f32,
    pub max: f32,
}

/// Нормализатор столбцов. Континуальные колонки — z-score `(x-mean)/std`,
/// категориальные — identity (коды не трогаем). Подгоняется ТОЛЬКО на train.
pub struct Normalizer {
    pub(crate) mean: Vec<f32>,
    pub(crate) std: Vec<f32>,
    pub(crate) min: Vec<f32>,
    pub(crate) max: Vec<f32>,
    pub(crate) specs: Vec<FeatureSpec>,
}

impl Normalizer {
    /// Список из `n` континуальных признаков (удобно для выходов регрессии).
    pub fn all_continuous(n: usize) -> Vec<FeatureSpec> {
        vec![FeatureSpec::Continuous; n]
    }

    pub fn fit(data: &Array2<f32>, specs: &[FeatureSpec]) -> Self {
        let (n, f) = data.dim();
        assert!(n > 0, "Normalizer::fit: пустые данные");
        assert_eq!(specs.len(), f, "specs должны покрывать все колонки");

        let mut mean = vec![0.0; f];
        let mut std = vec![1.0; f];
        let mut min = vec![f32::INFINITY; f];
        let mut max = vec![f32::NEG_INFINITY; f];

        for j in 0..f {
            let col = data.column(j);
            for &v in col {
                min[j] = min[j].min(v);
                max[j] = max[j].max(v);
            }
            if let FeatureSpec::Continuous = specs[j] {
                let m = col.sum() / n as f32;
                let var = col.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / n as f32;
                mean[j] = m;
                std[j] = var.sqrt().max(1e-8); // защита от константной колонки
            }
            // Категориальные: mean=0, std=1 (identity) — коды сохраняются целыми.
        }

        Self {
            mean,
            std,
            min,
            max,
            specs: specs.to_vec(),
        }
    }

    /// Число колонок (признаков), на которое подогнан нормализатор.
    pub fn n_features(&self) -> usize {
        self.mean.len()
    }

    pub fn transform(&self, data: &Array2<f32>) -> Array2<f32> {
        self.apply(data, |x, m, s| (x - m) / s)
    }

    pub fn inverse_transform(&self, data: &Array2<f32>) -> Array2<f32> {
        self.apply(data, |x, m, s| x * s + m)
    }

    fn apply<F: Fn(f32, f32, f32) -> f32>(&self, data: &Array2<f32>, f: F) -> Array2<f32> {
        let (n, cols) = data.dim();
        assert_eq!(cols, self.mean.len(), "число колонок != числу при fit");
        let mut out = data.clone();
        for j in 0..cols {
            for i in 0..n {
                out[[i, j]] = f(data[[i, j]], self.mean[j], self.std[j]);
            }
        }
        out
    }

    /// Координаты `(row, col)` континуальных значений вне обученного диапазона
    /// `[min, max]` — сигнал экстраполяции: модель ненадёжна вне
    /// распределения обучения).
    pub fn out_of_range(&self, data: &Array2<f32>) -> Vec<(usize, usize)> {
        let (n, cols) = data.dim();
        let mut flags = Vec::new();
        for i in 0..n {
            for j in 0..cols {
                if matches!(self.specs[j], FeatureSpec::Continuous) {
                    let v = data[[i, j]];
                    if v < self.min[j] || v > self.max[j] {
                        flags.push((i, j));
                    }
                }
            }
        }
        flags
    }

    /// Детали экстраполяции для одной строки входа (континуальные признаки вне
    /// `[min, max]`) — чтобы UI показал «признак 3 = 9.1 вне [0, 5]».
    pub fn out_of_range_details(&self, row: &[f32]) -> Vec<OutOfRange> {
        let mut out = Vec::new();
        for (j, &v) in row.iter().enumerate().take(self.specs.len()) {
            if matches!(self.specs[j], FeatureSpec::Continuous)
                && (v < self.min[j] || v > self.max[j])
            {
                out.push(OutOfRange {
                    feature: j,
                    value: v,
                    min: self.min[j],
                    max: self.max[j],
                });
            }
        }
        out
    }
}

/// Числовой датасет: сырые (ненормализованные) пары вход/выход.
#[derive(Debug)]
pub struct NumericDataset {
    pub inputs: Array2<f32>,
    pub outputs: Array2<f32>,
}

impl NumericDataset {
    pub fn new(inputs: Array2<f32>, outputs: Array2<f32>) -> Self {
        assert_eq!(
            inputs.nrows(),
            outputs.nrows(),
            "число строк входов и выходов должно совпадать"
        );
        Self { inputs, outputs }
    }

    pub fn len(&self) -> usize {
        self.inputs.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Собрать подвыборку по индексам строк.
    pub fn gather(&self, indices: &[usize]) -> NumericDataset {
        let f = self.inputs.ncols();
        let o = self.outputs.ncols();
        let mut inputs = Array2::zeros((indices.len(), f));
        let mut outputs = Array2::zeros((indices.len(), o));
        for (r, &i) in indices.iter().enumerate() {
            inputs.row_mut(r).assign(&self.inputs.row(i));
            outputs.row_mut(r).assign(&self.outputs.row(i));
        }
        NumericDataset { inputs, outputs }
    }
}

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg.into())
}

fn parse_feature_spec(token: &str) -> io::Result<FeatureSpec> {
    let lower = token.to_ascii_lowercase();
    if lower == "c" || lower == "continuous" {
        return Ok(FeatureSpec::Continuous);
    }
    if let Some(rest) = lower
        .strip_prefix("k:")
        .or_else(|| lower.strip_prefix("categorical:"))
    {
        let cardinality = rest
            .parse::<usize>()
            .map_err(|_| invalid_data(format!("неверная cardinality в specs: {token}")))?;
        return Ok(FeatureSpec::Categorical { cardinality });
    }
    Err(invalid_data(format!(
        "неизвестный тип признака в specs: {token}"
    )))
}

// --- формат .tnum ---
//
// TRNUM1 (устаревший, читается) знает только типы признаков:
//
//     TRNUM1 / inputs 2 / outputs 1 / specs C K:3 / rows 2 / data / ...
//
// TRNUM2 добавляет имена, единицы и подписи уровней категорий. Имена могут
// содержать пробелы, кавычки и Unicode, поэтому они пишутся в кавычках, а
// токенизатор понимает кавычки и экранирование:
//
//     TRNUM2
//     inputs 2
//     outputs 1
//     specs C K:3
//     names "temperature, °C" "материал" "влажность"
//     units "°C" - "%"
//     levels 1 "песок" "глина" "торф"
//     rows 2
//     data
//     80 0 12.5
//     60 2 18.1

/// Токен заголовка. Различать кавычки обязательно: `rows` — это директива, а
/// `"rows"` — допустимое имя колонки.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Bare(String),
    Quoted(String),
}

impl Token {
    /// `Some` только для директив и чисел — имя в кавычках директивой не станет.
    fn as_bare(&self) -> Option<&str> {
        match self {
            Token::Bare(s) => Some(s),
            Token::Quoted(_) => None,
        }
    }
}

/// Разбор в токены: `#` вне кавычек начинает комментарий до конца строки,
/// внутри кавычек это обычный символ. Поддерживаются `\"`, `\\`, `\n`, `\r` и `\t`.
fn tokenize(text: &str) -> io::Result<Vec<Token>> {
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line_no = lineno + 1;
        let mut chars = line.chars().peekable();
        loop {
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }
            match chars.peek() {
                None | Some('#') => break,
                Some('"') => {
                    chars.next();
                    let mut value = String::new();
                    loop {
                        match chars.next() {
                            None => {
                                return Err(invalid_data(format!(
                                    "строка {line_no}: незакрытая кавычка"
                                )))
                            }
                            Some('"') => break,
                            Some('\\') => match chars.next() {
                                Some('"') => value.push('"'),
                                Some('\\') => value.push('\\'),
                                Some('n') => value.push('\n'),
                                Some('r') => value.push('\r'),
                                Some('t') => value.push('\t'),
                                Some(other) => {
                                    return Err(invalid_data(format!(
                                        "строка {line_no}: неизвестная escape-последовательность \\{other}"
                                    )))
                                }
                                None => {
                                    return Err(invalid_data(format!(
                                        "строка {line_no}: незакрытая кавычка"
                                    )))
                                }
                            },
                            Some(c) => value.push(c),
                        }
                    }
                    if let Some(&next) = chars.peek() {
                        if !next.is_whitespace() && next != '#' {
                            return Err(invalid_data(format!(
                                "строка {line_no}: после закрывающей кавычки нужен пробел или комментарий"
                            )));
                        }
                    }
                    out.push(Token::Quoted(value));
                }
                Some(_) => {
                    let mut value = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_whitespace() || c == '#' {
                            break;
                        }
                        value.push(c);
                        chars.next();
                    }
                    out.push(Token::Bare(value));
                }
            }
        }
    }
    Ok(out)
}

/// Курсор по токенам заголовка: каждая директива знает свою арность, поэтому
/// переводы строк для разбора не нужны.
struct Cursor<'a> {
    tokens: &'a [Token],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn peek_bare(&self) -> Option<&str> {
        self.tokens.get(self.i).and_then(Token::as_bare)
    }

    fn next_token(&mut self, what: &str) -> io::Result<&'a Token> {
        let t = self
            .tokens
            .get(self.i)
            .ok_or_else(|| invalid_data(format!("ожидали {what}, файл закончился")))?;
        self.i += 1;
        Ok(t)
    }

    fn expect_bare(&mut self, word: &str) -> io::Result<()> {
        match self.next_token(word)? {
            Token::Bare(s) if s == word => Ok(()),
            other => Err(invalid_data(format!(
                "ожидали '{word}', получили {other:?}"
            ))),
        }
    }

    fn next_usize(&mut self, what: &str) -> io::Result<usize> {
        match self.next_token(what)? {
            Token::Bare(s) => s
                .parse()
                .map_err(|_| invalid_data(format!("не удалось прочитать {what}: '{s}'"))),
            Token::Quoted(s) => Err(invalid_data(format!("{what} не может быть строкой: '{s}'"))),
        }
    }

    fn next_f32(&mut self, what: &str) -> io::Result<f32> {
        match self.next_token(what)? {
            Token::Bare(s) => s
                .parse()
                .map_err(|_| invalid_data(format!("не удалось прочитать {what}: '{s}'"))),
            Token::Quoted(s) => Err(invalid_data(format!("{what} не может быть строкой: '{s}'"))),
        }
    }

    /// Строковое поле: обязано быть в кавычках, иначе имя вида `rows` было бы
    /// неотличимо от директивы.
    fn next_quoted(&mut self, what: &str) -> io::Result<String> {
        match self.next_token(what)? {
            Token::Quoted(s) => Ok(s.clone()),
            Token::Bare(s) => Err(invalid_data(format!(
                "{what} должно быть в кавычках, получили '{s}'"
            ))),
        }
    }

    fn finished(&self) -> io::Result<()> {
        match self.tokens.get(self.i) {
            None => Ok(()),
            Some(extra) => Err(invalid_data(format!("лишние данные в конце: {extra:?}"))),
        }
    }

    fn remaining(&self) -> usize {
        self.tokens.len() - self.i
    }
}

/// Прочитать числовой датасет и его схему из `.tnum` (TRNUM1 или TRNUM2).
pub fn read_numeric_tnum(path: &str) -> io::Result<(NumericDataset, ModelSchema)> {
    parse_numeric_tnum(&std::fs::read_to_string(path)?)
}

/// Разбор `.tnum` из строки. У TRNUM1 имён нет, поэтому схема достраивается
/// синтетически — с сохранением категориальных типов.
pub fn parse_numeric_tnum(text: &str) -> io::Result<(NumericDataset, ModelSchema)> {
    let tokens = tokenize(text)?;
    let mut c = Cursor {
        tokens: &tokens,
        i: 0,
    };

    let version = match c.next_token("магию TRNUM1/TRNUM2")? {
        Token::Bare(s) if s == "TRNUM1" => 1,
        Token::Bare(s) if s == "TRNUM2" => 2,
        other => {
            return Err(invalid_data(format!(
                "ожидали магию TRNUM1 или TRNUM2, получили {other:?}"
            )))
        }
    };

    c.expect_bare("inputs")?;
    let n_inputs = c.next_usize("число входов")?;
    c.expect_bare("outputs")?;
    let n_outputs = c.next_usize("число выходов")?;
    let n_columns = n_inputs
        .checked_add(n_outputs)
        .ok_or_else(|| invalid_data("суммарное число колонок не помещается в usize"))?;
    c.expect_bare("specs")?;
    // Не резервируем память по ещё не проверенному числу из внешнего файла.
    let mut specs = Vec::new();
    for _ in 0..n_inputs {
        match c.next_token("тип признака")? {
            Token::Bare(s) => specs.push(parse_feature_spec(s)?),
            Token::Quoted(s) => return Err(invalid_data(format!("тип признака не строка: '{s}'"))),
        }
    }

    let schema = if version == 1 {
        ModelSchema::synthetic_from_specs(&specs, n_outputs).map_err(invalid_data)?
    } else {
        read_v2_schema(&mut c, &specs, n_inputs, n_outputs, n_columns)?
    };

    c.expect_bare("rows")?;
    let rows = c.next_usize("число строк")?;
    c.expect_bare("data")?;
    let expected_values = rows
        .checked_mul(n_columns)
        .ok_or_else(|| invalid_data("размер секции data не помещается в usize"))?;
    if c.remaining() != expected_values {
        return Err(invalid_data(format!(
            "в секции data ожидалось {expected_values} значений ({rows} × {n_columns}), получено {}",
            c.remaining()
        )));
    }

    let mut inputs = Array2::<f32>::zeros((rows, n_inputs));
    let mut outputs = Array2::<f32>::zeros((rows, n_outputs));
    for r in 0..rows {
        for col in 0..n_inputs {
            inputs[[r, col]] = c.next_f32("входное значение")?;
        }
        for col in 0..n_outputs {
            outputs[[r, col]] = c.next_f32("выходное значение")?;
        }
    }
    c.finished()?;

    let data = NumericDataset::new(inputs, outputs);
    validate_numeric_tnum(&schema, &data).map_err(invalid_data)?;
    Ok((data, schema))
}

/// Заголовок TRNUM2: обязательный `names`, необязательные `units` и `levels`.
fn read_v2_schema(
    c: &mut Cursor<'_>,
    specs: &[FeatureSpec],
    n_inputs: usize,
    n_outputs: usize,
    n_columns: usize,
) -> io::Result<ModelSchema> {
    c.expect_bare("names")?;
    // Как и specs, не резервируем память по непроверенному размеру файла.
    let mut names = Vec::new();
    for i in 0..n_columns {
        names.push(c.next_quoted(&format!("имя колонки {i}"))?);
    }

    let mut units: Vec<Option<String>> = vec![None; n_columns];
    if c.peek_bare() == Some("units") {
        c.expect_bare("units")?;
        for unit in units.iter_mut() {
            match c.next_token("единицу измерения")? {
                // Голый дефис — «единицы нет»; иначе имя '-' было бы неотличимо.
                Token::Bare(s) if s == "-" => {}
                Token::Quoted(s) if !s.trim().is_empty() => *unit = Some(s.clone()),
                Token::Quoted(_) => {
                    return Err(invalid_data("пустая единица измерения: используйте '-'"))
                }
                Token::Bare(s) => {
                    return Err(invalid_data(format!(
                        "единица измерения должна быть в кавычках или '-', получили '{s}'"
                    )))
                }
            }
        }
    }

    // levels <индекс входа> "уровень"...; по строке на каждый категориальный вход.
    let mut levels: HashMap<usize, Vec<String>> = HashMap::new();
    while c.peek_bare() == Some("levels") {
        c.expect_bare("levels")?;
        let idx = c.next_usize("индекс категориального входа")?;
        let cardinality = match specs.get(idx) {
            Some(FeatureSpec::Categorical { cardinality }) => *cardinality,
            Some(FeatureSpec::Continuous) => {
                return Err(invalid_data(format!(
                    "levels {idx}: вход не категориальный"
                )))
            }
            None => {
                return Err(invalid_data(format!(
                    "levels {idx}: индекс вне диапазона 0..{n_inputs}"
                )))
            }
        };
        if levels.contains_key(&idx) {
            return Err(invalid_data(format!("levels {idx}: повторная секция")));
        }
        let mut values = Vec::with_capacity(cardinality);
        for l in 0..cardinality {
            values.push(c.next_quoted(&format!("подпись уровня {l} входа {idx}"))?);
        }
        levels.insert(idx, values);
    }

    let mut columns = Vec::with_capacity(n_columns);
    for (i, spec) in specs.iter().enumerate() {
        let column = match *spec {
            FeatureSpec::Continuous => Column::numeric(&names[i], ColumnRole::Input),
            FeatureSpec::Categorical { cardinality } => {
                // Отсутствие подписей у K:n — испорченный файл: writer их всегда
                // пишет, а тихий откат к «0…n-1» подменил бы данные.
                let values = levels.remove(&i).ok_or_else(|| {
                    invalid_data(format!(
                        "вход {i} объявлен как K:{cardinality}, но секции levels {i} нет"
                    ))
                })?;
                Column::categorical(&names[i], ColumnRole::Input, values)
            }
        }
        .map_err(invalid_data)?;
        columns.push(match &units[i] {
            Some(u) => column.with_unit(u),
            None => column,
        });
    }
    let inputs = columns;

    let mut outputs = Vec::with_capacity(n_outputs);
    for j in 0..n_outputs {
        let k = n_inputs + j;
        let column = Column::numeric(&names[k], ColumnRole::Output).map_err(invalid_data)?;
        outputs.push(match &units[k] {
            Some(u) => column.with_unit(u),
            None => column,
        });
    }

    ModelSchema::new(inputs, outputs).map_err(invalid_data)
}

fn quote(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            other => escaped.push(other),
        }
    }
    format!("\"{escaped}\"")
}

/// Общая валидация данных формата. Writer не создаёт битый файл, а reader
/// не доверяет кодам категорий и NaN/Inf из внешнего файла.
fn validate_numeric_tnum(schema: &ModelSchema, data: &NumericDataset) -> Result<(), String> {
    schema.check_dims(data.inputs.ncols(), data.outputs.ncols())?;

    for r in 0..data.len() {
        for (i, column) in schema.inputs().iter().enumerate() {
            let raw = data.inputs[[r, i]];
            if !raw.is_finite() {
                return Err(format!("строка {r}, вход {i}: значение не конечно: {raw}"));
            }
            let Some(cardinality) = column.cardinality() else {
                continue;
            };
            let rounded = raw.round();
            if (raw - rounded).abs() >= 1e-4 {
                return Err(format!(
                    "строка {r}, вход {i}: код категории должен быть целым, получено {raw}"
                ));
            }
            if rounded < 0.0 || (rounded as usize) >= cardinality {
                return Err(format!(
                    "строка {r}, вход {i}: категория {rounded} вне [0, {cardinality})"
                ));
            }
        }
        for j in 0..data.outputs.ncols() {
            let raw = data.outputs[[r, j]];
            if !raw.is_finite() {
                return Err(format!("строка {r}, выход {j}: значение не конечно: {raw}"));
            }
        }
    }
    Ok(())
}

/// Записать датасет в TRNUM2 с явной схемой.
///
/// Схема обязательна: без неё имена пришлось бы выдумывать, а именно этого
/// формат и должен избежать.
pub fn write_numeric_tnum(schema: &ModelSchema, data: &NumericDataset) -> Result<String, String> {
    validate_numeric_tnum(schema, data)?;

    let specs: Vec<String> = schema
        .feature_specs()
        .iter()
        .map(|spec| match spec {
            FeatureSpec::Continuous => "C".to_string(),
            FeatureSpec::Categorical { cardinality } => format!("K:{cardinality}"),
        })
        .collect();
    let columns = || schema.inputs().iter().chain(schema.outputs().iter());

    let mut out = String::new();
    out.push_str("TRNUM2\n");
    out.push_str(&format!("inputs {}\n", schema.n_inputs()));
    out.push_str(&format!("outputs {}\n", schema.n_outputs()));
    out.push_str(&format!("specs {}\n", specs.join(" ")));
    let names: Vec<String> = columns().map(|c| quote(c.name())).collect();
    out.push_str(&format!("names {}\n", names.join(" ")));
    if columns().any(|c| c.unit().is_some()) {
        let units: Vec<String> = columns()
            .map(|c| c.unit().map_or("-".to_string(), quote))
            .collect();
        out.push_str(&format!("units {}\n", units.join(" ")));
    }
    for (i, column) in schema.inputs().iter().enumerate() {
        if let ColumnType::Categorical { levels } = column.ty() {
            let quoted: Vec<String> = levels.iter().map(|l| quote(l)).collect();
            out.push_str(&format!("levels {i} {}\n", quoted.join(" ")));
        }
    }
    out.push_str(&format!("rows {}\n", data.len()));
    out.push_str("data\n");
    for r in 0..data.len() {
        let row: Vec<String> = data
            .inputs
            .row(r)
            .iter()
            .chain(data.outputs.row(r).iter())
            .map(|v| format!("{v}"))
            .collect();
        out.push_str(&row.join(" "));
        out.push('\n');
    }
    Ok(out)
}

/// Словарь char-уровня: отсортированные уникальные символы корпуса.
#[cfg(feature = "demo")]
pub struct Vocab {
    itos: Vec<char>,
    stoi: HashMap<char, usize>,
}

#[cfg(feature = "demo")]
impl Vocab {
    pub fn from_text(text: &str) -> Self {
        let itos: Vec<char> = text.chars().collect::<BTreeSet<_>>().into_iter().collect();
        let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { itos, stoi }
    }

    pub fn len(&self) -> usize {
        self.itos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.itos.is_empty()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.stoi.get(&c).expect("символ отсутствует в словаре"))
            .collect()
    }

    pub fn contains(&self, c: char) -> bool {
        self.stoi.contains_key(&c)
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.itos[i]).collect()
    }

    /// Символы словаря по порядку id (для сериализации).
    pub(crate) fn chars(&self) -> &[char] {
        &self.itos
    }

    /// Восстановить словарь из упорядоченного списка символов.
    pub(crate) fn from_chars(itos: Vec<char>) -> Self {
        let stoi = itos.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self { itos, stoi }
    }
}

/// Текстовый датасет char-уровня: весь корпус как поток id + словарь.
#[cfg(feature = "demo")]
pub struct TextDataset {
    pub vocab: Vocab,
    data: Vec<usize>,
}

#[cfg(feature = "demo")]
impl TextDataset {
    pub fn new(text: &str) -> Self {
        let vocab = Vocab::from_text(text);
        let data = vocab.encode(text);
        Self { vocab, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Случайный батч окон `контекст -> продолжение` для seq2seq.
    /// Возвращает `(src [B, ctx_len], tgt [B, tgt_len])` — оба в id.
    pub fn sample_batch(
        &self,
        batch: usize,
        ctx_len: usize,
        tgt_len: usize,
        rng: &mut StdRng,
    ) -> (Array2<usize>, Array2<usize>) {
        let span = ctx_len + tgt_len;
        assert!(self.data.len() >= span, "корпус короче окна ctx+tgt");

        let mut src = Array2::<usize>::zeros((batch, ctx_len));
        let mut tgt = Array2::<usize>::zeros((batch, tgt_len));
        for b in 0..batch {
            let start = rng.gen_range(0..=self.data.len() - span);
            for t in 0..ctx_len {
                src[[b, t]] = self.data[start + t];
            }
            for t in 0..tgt_len {
                tgt[[b, t]] = self.data[start + ctx_len + t];
            }
        }
        (src, tgt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[cfg(feature = "demo")]
    use rand::SeedableRng;

    #[test]
    fn normalizer_round_trip() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let norm = Normalizer::fit(&data, &Normalizer::all_continuous(2));
        let back = norm.inverse_transform(&norm.transform(&data));
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-4, "round-trip разошёлся: {a} vs {b}");
        }
        // После transform континуальные колонки имеют mean≈0, std≈1.
        let z = norm.transform(&data);
        let m: f32 = z.column(0).sum() / 4.0;
        assert!(m.abs() < 1e-5);
    }

    #[test]
    fn normalizer_skips_categorical() {
        // Колонка 1 категориальная — должна остаться без изменений.
        let data = array![[1.0, 2.0], [3.0, 0.0], [5.0, 1.0]];
        let specs = [
            FeatureSpec::Continuous,
            FeatureSpec::Categorical { cardinality: 3 },
        ];
        let norm = Normalizer::fit(&data, &specs);
        let z = norm.transform(&data);
        for i in 0..3 {
            assert_eq!(z[[i, 1]], data[[i, 1]], "категориальный код изменился");
        }
    }

    #[test]
    fn out_of_range_details_reports_feature_and_bounds() {
        // Признак 0 континуальный с диапазоном [0, 2]; признак 1 константа 5.
        let data = array![[0.0, 5.0], [2.0, 5.0]];
        let norm = Normalizer::fit(&data, &Normalizer::all_continuous(2));
        let d = norm.out_of_range_details(&[3.0, 5.0]); // x0=3 > 2
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].feature, 0);
        assert_eq!(d[0].value, 3.0);
        assert_eq!((d[0].min, d[0].max), (0.0, 2.0));
        assert!(norm.out_of_range_details(&[1.0, 5.0]).is_empty()); // в диапазоне
    }

    #[test]
    fn normalizer_detects_extrapolation() {
        let data = array![[0.0], [1.0], [2.0]];
        let norm = Normalizer::fit(&data, &Normalizer::all_continuous(1));
        let probe = array![[1.0], [5.0], [-3.0]];
        let flags = norm.out_of_range(&probe);
        assert_eq!(flags, vec![(1, 0), (2, 0)]); // 5 и -3 вне [0, 2]
    }

    #[test]
    fn read_numeric_tnum_file() {
        let path = std::env::temp_dir().join("transformer_read_numeric_tnum_test.tnum");
        std::fs::write(
            &path,
            "TRNUM1\ninputs 2\noutputs 1\nspecs C C\nrows 2\ndata\n1 2 3\n4 5 9\n",
        )
        .unwrap();
        let (ds, schema) = read_numeric_tnum(path.to_str().unwrap()).unwrap();
        assert_eq!(
            schema.feature_specs(),
            vec![FeatureSpec::Continuous, FeatureSpec::Continuous]
        );
        assert_eq!(ds.inputs.dim(), (2, 2));
        assert_eq!(ds.outputs.dim(), (2, 1));
        assert_eq!(ds.outputs[[1, 0]], 9.0);
        std::fs::remove_file(path).ok();
    }

    /// Схема с именами, единицами, категорией и «неудобными» подписями.
    fn rich_schema() -> ModelSchema {
        ModelSchema::new(
            vec![
                Column::numeric("температура, °C", ColumnRole::Input)
                    .unwrap()
                    .with_unit("°C"),
                Column::categorical(
                    "материал",
                    ColumnRole::Input,
                    vec![
                        "песок".into(),
                        "глина \"жирная\"".into(),
                        "торф # верховой\nвлажный\\слой".into(),
                    ],
                )
                .unwrap(),
                // Имя, совпадающее с директивой формата: без кавычек разбор бы сломался.
                Column::numeric("rows", ColumnRole::Input).unwrap(),
            ],
            vec![Column::numeric("влажность", ColumnRole::Output)
                .unwrap()
                .with_unit("%\tмас.")],
        )
        .unwrap()
    }

    fn rich_data() -> NumericDataset {
        NumericDataset::new(
            array![[80.0, 0.0, 1.5], [60.0, 2.0, 2.5]],
            array![[12.5], [18.25]],
        )
    }

    #[test]
    fn trnum2_round_trip_keeps_names_units_and_levels() {
        let schema = rich_schema();
        let text = write_numeric_tnum(&schema, &rich_data()).unwrap();
        assert!(text.starts_with("TRNUM2\n"), "{text}");

        let (ds, back) = parse_numeric_tnum(&text).unwrap();
        assert_eq!(back, schema);
        assert_eq!(
            back.input_names(),
            vec!["температура, °C", "материал", "rows"]
        );
        assert_eq!(back.output_names(), vec!["влажность"]);
        assert_eq!(back.inputs()[0].unit(), Some("°C"));
        assert_eq!(back.outputs()[0].unit(), Some("%\tмас."));
        assert_eq!(
            back.inputs()[1].category_level(1).unwrap(),
            "глина \"жирная\""
        );
        assert_eq!(
            back.inputs()[1].category_level(2).unwrap(),
            "торф # верховой\nвлажный\\слой"
        );
        assert!(text.contains("\\n"), "{text}");
        assert!(text.contains("\\t"), "{text}");
        assert_eq!(ds.inputs, rich_data().inputs);
        assert_eq!(ds.outputs, rich_data().outputs);
    }

    #[test]
    fn trnum2_omits_units_line_when_none() {
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let data = NumericDataset::new(array![[1.0, 2.0]], array![[3.0]]);
        let text = write_numeric_tnum(&schema, &data).unwrap();
        assert!(!text.contains("units"), "{text}");
        assert_eq!(parse_numeric_tnum(&text).unwrap().1, schema);
    }

    #[test]
    fn comment_inside_quotes_is_kept_but_outside_is_stripped() {
        let text = concat!(
            "TRNUM2  # заголовок\n",
            "inputs 1\noutputs 1\nspecs C\n",
            "names \"a # b\" \"y\"\n",
            "rows 1\ndata\n1 2\n",
        );
        let (_, schema) = parse_numeric_tnum(text).unwrap();
        assert_eq!(schema.input_names(), vec!["a # b"]);
    }

    #[test]
    fn trnum2_rejects_malformed_headers() {
        let base = "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"x\" \"y\"\nrows 1\ndata\n1 2\n";
        assert!(parse_numeric_tnum(base).is_ok());

        // Имя без кавычек неотличимо от директивы.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames x y\nrows 1\ndata\n1 2\n"
        )
        .is_err());
        // Незакрытая кавычка.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"x \"y\"\nrows 1\ndata\n1 2\n"
        )
        .is_err());
        // Имён меньше, чем колонок.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 2\noutputs 1\nspecs C C\nnames \"x\" \"y\"\nrows 1\ndata\n1 2 3\n"
        )
        .is_err());
        // Лишние данные после объявленных строк.
        assert!(parse_numeric_tnum(&format!("{base}9 9\n")).is_err());
        // Данных меньше объявленного.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"x\" \"y\"\nrows 2\ndata\n1 2\n"
        )
        .is_err());
        // Неизвестная escape-последовательность.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"a\\qb\" \"y\"\nrows 1\ndata\n1 2\n"
        )
        .is_err());
        // Между двумя строковыми полями нужен разделитель.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"x\"\"y\"\nrows 1\ndata\n1 2\n"
        )
        .is_err());
        // Пустая единица не является вторым способом записать отсутствие.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"x\" \"y\"\nunits \"\" -\nrows 1\ndata\n1 2\n"
        )
        .is_err());
        // Объявленный размер проверяется до выделения массивов.
        let oversized = format!(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"x\" \"y\"\nrows {}\ndata\n",
            usize::MAX
        );
        assert!(parse_numeric_tnum(&oversized)
            .err()
            .unwrap()
            .to_string()
            .contains("не помещается"));
    }

    #[test]
    fn categorical_without_levels_is_an_error() {
        // K:2 без секции levels — испорченный файл, а не повод выдумать подписи.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs K:2\nnames \"m\" \"y\"\nrows 1\ndata\n0 2\n"
        )
        .is_err());
        // levels на числовом входе.
        assert!(parse_numeric_tnum(
            "TRNUM2\ninputs 1\noutputs 1\nspecs C\nnames \"m\" \"y\"\nlevels 0 \"a\"\nrows 1\ndata\n0 2\n"
        )
        .is_err());
        // Повторная секция levels.
        assert!(parse_numeric_tnum(concat!(
            "TRNUM2\ninputs 1\noutputs 1\nspecs K:1\nnames \"m\" \"y\"\n",
            "levels 0 \"a\"\nlevels 0 \"b\"\nrows 1\ndata\n0 2\n"
        ))
        .is_err());
    }

    #[test]
    fn writer_checks_dims_and_category_codes() {
        let schema = rich_schema();
        // Схема на 3 входа, данные на 2.
        let wrong = NumericDataset::new(array![[1.0, 0.0]], array![[1.0]]);
        assert!(write_numeric_tnum(&schema, &wrong).is_err());

        let fractional = NumericDataset::new(array![[80.0, 0.5, 1.0]], array![[1.0]]);
        assert!(write_numeric_tnum(&schema, &fractional)
            .unwrap_err()
            .contains("целым"));

        let out_of_range = NumericDataset::new(array![[80.0, 7.0, 1.0]], array![[1.0]]);
        assert!(write_numeric_tnum(&schema, &out_of_range)
            .unwrap_err()
            .contains("вне [0, 3)"));

        let non_finite = NumericDataset::new(array![[f32::NAN, 0.0, 1.0]], array![[1.0]]);
        assert!(write_numeric_tnum(&schema, &non_finite)
            .unwrap_err()
            .contains("не конечно"));
    }

    #[test]
    fn reader_checks_values_from_external_files() {
        // TRNUM1 тоже не должен обходить проверку категориальных кодов.
        assert!(parse_numeric_tnum(
            "TRNUM1\ninputs 1\noutputs 1\nspecs K:2\nrows 1\ndata\n0.5 2\n"
        )
        .err()
        .unwrap()
        .to_string()
        .contains("целым"));

        assert!(parse_numeric_tnum(concat!(
            "TRNUM2\ninputs 1\noutputs 1\nspecs K:2\n",
            "names \"material\" \"y\"\nlevels 0 \"a\" \"b\"\n",
            "rows 1\ndata\n2 3\n"
        ))
        .err()
        .unwrap()
        .to_string()
        .contains("вне [0, 2)"));

        assert!(
            parse_numeric_tnum("TRNUM1\ninputs 1\noutputs 1\nspecs C\nrows 1\ndata\nNaN 2\n")
                .err()
                .unwrap()
                .to_string()
                .contains("не конечно")
        );
    }

    #[test]
    fn legacy_trnum1_still_reads_with_categorical() {
        let (ds, schema) = parse_numeric_tnum(
            "TRNUM1\ninputs 2\noutputs 1\nspecs C K:3\nrows 2\ndata\n1 0 3\n4 2 9\n",
        )
        .unwrap();
        assert_eq!(schema.input_names(), vec!["x0", "x1"]);
        assert_eq!(schema.output_names(), vec!["y0"]);
        assert_eq!(
            schema.feature_specs(),
            vec![
                FeatureSpec::Continuous,
                FeatureSpec::Categorical { cardinality: 3 }
            ]
        );
        // Подписи старого файла — честные коды, а не выдуманные названия.
        assert_eq!(schema.inputs()[1].category_level(2).unwrap(), "2");
        assert_eq!(ds.outputs[[1, 0]], 9.0);
        // Комментарии и произвольные переводы строк по-прежнему допустимы.
        assert!(parse_numeric_tnum(
            "TRNUM1 inputs 1 outputs 1 specs C rows 1 # хвост\ndata\n1 2\n"
        )
        .is_ok());
    }

    #[test]
    #[cfg(feature = "demo")]
    fn vocab_round_trip() {
        let text = "hello world";
        let vocab = Vocab::from_text(text);
        assert_eq!(vocab.decode(&vocab.encode(text)), text);
        // Уникальные символы: ' dehlorw' = 8.
        assert_eq!(vocab.len(), 8);
    }

    #[test]
    #[cfg(feature = "demo")]
    fn text_batch_shapes() {
        let ds = TextDataset::new("abcdefghijklmnop");
        let mut rng = StdRng::seed_from_u64(1);
        let (src, tgt) = ds.sample_batch(4, 3, 2, &mut rng);
        assert_eq!(src.dim(), (4, 3));
        assert_eq!(tgt.dim(), (4, 2));
        // id должны быть валидными индексами словаря.
        assert!(src.iter().all(|&id| id < ds.vocab.len()));
    }
}
