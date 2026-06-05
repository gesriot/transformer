//! Данные и нормализация (Plan.md §4, §6).
//!
//! - `Normalizer` — z-score по континуальным колонкам, identity по
//!   категориальным; хранит диапазоны для предупреждения об экстраполяции.
//! - `NumericDataset` — пары вход/выход, детерминированный split, выборка строк.
//! - `TextDataset` + `Vocab` — char-уровень, нарезка контекст→продолжение.

use crate::encoders::FeatureSpec;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeSet, HashMap};
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
    /// `[min, max]` — сигнал экстраполяции (Plan.md §4: модель ненадёжна вне
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

    /// Детерминированный train/test split: фиксированный seed -> одинаковое
    /// разбиение. Перемешивает индексы и режет по `train_frac`.
    pub fn split(&self, train_frac: f32, seed: u64) -> (NumericDataset, NumericDataset) {
        assert!((0.0..=1.0).contains(&train_frac), "train_frac в [0,1]");
        let n = self.len();
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        idx.shuffle(&mut rng);
        let n_train = (n as f32 * train_frac).round() as usize;
        (self.gather(&idx[..n_train]), self.gather(&idx[n_train..]))
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

fn parse_usize(token: Option<&str>, what: &str) -> io::Result<usize> {
    token
        .ok_or_else(|| invalid_data(format!("ожидали {what}")))?
        .parse::<usize>()
        .map_err(|_| invalid_data(format!("не удалось прочитать {what}")))
}

fn parse_f32(token: Option<&str>, what: &str) -> io::Result<f32> {
    token
        .ok_or_else(|| invalid_data(format!("ожидали {what}")))?
        .parse::<f32>()
        .map_err(|_| invalid_data(format!("не удалось прочитать {what}")))
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

/// Прочитать числовой датасет из простого текстового формата `.tnum`.
///
/// Формат:
/// ```text
/// TRNUM1
/// inputs 2
/// outputs 1
/// specs C C
/// rows 3
/// data
/// 0.1 0.2 0.3
/// 0.4 0.5 0.9
/// 0.7 0.8 1.5
/// ```
pub fn read_numeric_tnum(path: &str) -> io::Result<(NumericDataset, Vec<FeatureSpec>)> {
    let text = std::fs::read_to_string(path)?;
    let mut tokens = Vec::new();
    for line in text.lines() {
        let clean = line.split('#').next().unwrap_or("").trim();
        tokens.extend(clean.split_whitespace().map(str::to_string));
    }

    let mut it = tokens.iter().map(String::as_str);
    if it.next() != Some("TRNUM1") {
        return Err(invalid_data("ожидали магию TRNUM1"));
    }

    if it.next() != Some("inputs") {
        return Err(invalid_data("ожидали строку: inputs <N>"));
    }
    let n_inputs = parse_usize(it.next(), "число входов")?;

    if it.next() != Some("outputs") {
        return Err(invalid_data("ожидали строку: outputs <M>"));
    }
    let n_outputs = parse_usize(it.next(), "число выходов")?;

    if it.next() != Some("specs") {
        return Err(invalid_data("ожидали строку: specs ..."));
    }
    let mut specs = Vec::with_capacity(n_inputs);
    for _ in 0..n_inputs {
        specs.push(parse_feature_spec(
            it.next()
                .ok_or_else(|| invalid_data("specs короче числа входов"))?,
        )?);
    }

    if it.next() != Some("rows") {
        return Err(invalid_data("ожидали строку: rows <N>"));
    }
    let rows = parse_usize(it.next(), "число строк")?;

    if it.next() != Some("data") {
        return Err(invalid_data("ожидали маркер data"));
    }

    let mut inputs = Array2::<f32>::zeros((rows, n_inputs));
    let mut outputs = Array2::<f32>::zeros((rows, n_outputs));
    for r in 0..rows {
        for c in 0..n_inputs {
            inputs[[r, c]] = parse_f32(it.next(), "входное значение")?;
        }
        for c in 0..n_outputs {
            outputs[[r, c]] = parse_f32(it.next(), "выходное значение")?;
        }
    }
    if let Some(extra) = it.next() {
        return Err(invalid_data(format!(
            "лишние данные после {rows} строк: {extra}"
        )));
    }

    Ok((NumericDataset::new(inputs, outputs), specs))
}

/// Словарь char-уровня: отсортированные уникальные символы корпуса.
pub struct Vocab {
    itos: Vec<char>,
    stoi: HashMap<char, usize>,
}

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
pub struct TextDataset {
    pub vocab: Vocab,
    data: Vec<usize>,
}

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
    fn split_is_deterministic_and_partitions() {
        let inputs = Array2::from_shape_fn((10, 2), |(i, j)| (i * 2 + j) as f32);
        let outputs = Array2::from_shape_fn((10, 1), |(i, _)| i as f32);
        let ds = NumericDataset::new(inputs, outputs);

        let (tr1, te1) = ds.split(0.7, 42);
        let (tr2, te2) = ds.split(0.7, 42);
        assert_eq!(tr1.len(), 7);
        assert_eq!(te1.len(), 3);
        assert_eq!(tr1.len() + te1.len(), ds.len());
        // Один seed -> идентичное разбиение.
        assert_eq!(tr1.inputs, tr2.inputs);
        assert_eq!(te1.inputs, te2.inputs);
        // Другой seed -> другое (с большой вероятностью) разбиение.
        let (tr3, _) = ds.split(0.7, 7);
        assert_ne!(tr1.inputs, tr3.inputs);
    }

    #[test]
    fn read_numeric_tnum_file() {
        let path = std::env::temp_dir().join("transformer_read_numeric_tnum_test.tnum");
        std::fs::write(
            &path,
            "TRNUM1\ninputs 2\noutputs 1\nspecs C C\nrows 2\ndata\n1 2 3\n4 5 9\n",
        )
        .unwrap();
        let (ds, specs) = read_numeric_tnum(path.to_str().unwrap()).unwrap();
        assert_eq!(
            specs,
            vec![FeatureSpec::Continuous, FeatureSpec::Continuous]
        );
        assert_eq!(ds.inputs.dim(), (2, 2));
        assert_eq!(ds.outputs.dim(), (2, 1));
        assert_eq!(ds.outputs[[1, 0]], 9.0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn vocab_round_trip() {
        let text = "hello world";
        let vocab = Vocab::from_text(text);
        assert_eq!(vocab.decode(&vocab.encode(text)), text);
        // Уникальные символы: ' dehlorw' = 8.
        assert_eq!(vocab.len(), 8);
    }

    #[test]
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
