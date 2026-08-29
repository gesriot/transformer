//! Бинарная сериализация моделей без внешних зависимостей (Plan.md §6, M9).
//!
//! Формат v2 — секционный, little-endian:
//!   MAGIC, VERSION, KIND, section_count, [section]*
//!   section = name_len, name, payload_len, payload
//! Читатель грузит все секции в map по имени → **неизвестные секции
//! игнорируются** (forward-compat). Конфиг внутри — поля с тегами
//! (tag, len, value): неизвестный тег пропускается, отсутствующее
//! необязательное поле берёт default. Это позволяет расширять `ModelConfig`
//! новыми полями (fourier_bands и т.п.) без поломки старых файлов.

use crate::atomic_write::write_atomically;
use crate::config::ModelConfig;
use crate::data::Normalizer;
#[cfg(feature = "demo")]
use crate::data::Vocab;
use crate::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use crate::fingerprint::DatasetFingerprint;
use crate::interpret::{InterpretProfile, InterpretReport, INTERPRET_PROFILE_VERSION};
use crate::kan::CompactReport;
use crate::lifecycle::{CandidateSpec, RunStamp};
use crate::metrics::{EvalSource, Metrics};
use crate::numeric_model::{KanConfig, ModelKind, NumericConfig, NumericModel};
use crate::report::{CheckRecord, FinalRecord, Selection, TrainingReport, TRAINING_REPORT_VERSION};
use crate::schema::{Column, ColumnRole, ColumnType, ModelSchema};
use crate::split::{FinalEval, FinalOrigin, SplitPlan};
use crate::tensor::Tensor;
#[cfg(feature = "demo")]
use crate::textmodel::TextModel;
use crate::train::{LrSchedule, TrainConfig};
use crate::training::{EpochPoint, SearchObjective, TrainingHistory};
use ndarray::{Array2, ArrayD, Ix2, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: u32 = 0x5452_4653; // "TRFS"
const VERSION: u32 = 2;
const KIND_SURROGATE: u32 = 0;
#[cfg(feature = "demo")]
const KIND_TEXT: u32 = 1;

// Теги полей конфига (стабильные; новые поля получают новые теги).
const TAG_D_MODEL: u16 = 1;
const TAG_N_HEADS: u16 = 2;
const TAG_N_ENC: u16 = 3;
const TAG_N_DEC: u16 = 4;
const TAG_D_FF: u16 = 5;
const TAG_LN_EPS: u16 = 6;
const TAG_MODEL_KIND: u16 = 7; // 0=transformer, 1=mlp, 2=kan; отсутствует -> transformer
const TAG_MLP_WIDTH: u16 = 8;
const TAG_MLP_LAYERS: u16 = 9;
const TAG_VALUE_ENC: u16 = 10; // 0=linear, 1=mlp, 2=fourier; отсутствует -> linear
const TAG_FOURIER_BANDS: u16 = 11;
const TAG_FOURIER_SCALE: u16 = 12;
const TAG_KAN_WIDTH: u16 = 13;
const TAG_KAN_LAYERS: u16 = 14;
const TAG_KAN_GRID: u16 = 15;
// Тег секции meta.
const TAG_NUM_OUTPUTS: u16 = 1;

// Известные секции по типам моделей (всё остальное при чтении пропускается).
const SURROGATE_SECTIONS: &[&str] = &[
    "config",
    "meta",
    "feature_specs",
    "schema",
    "interpret",
    "params",
    "kan_masks",
    "kan_dims",
    "calibration",
    "in_norm",
    "out_norm",
    "training_report",
];
#[cfg(feature = "demo")]
const TEXT_SECTIONS: &[&str] = &["config", "vocab", "params"];

/// Полное содержимое численного checkpoint-а.
#[non_exhaustive]
pub struct NumericCheckpoint {
    pub model: NumericModel,
    pub in_norm: Normalizer,
    pub out_norm: Normalizer,
    pub config: NumericConfig,
    /// Схема данных. У старых checkpoint-ов достраивается синтетически из
    /// сохранённых `feature_specs`, поэтому поле есть всегда.
    pub schema: ModelSchema,
    /// Какой конвейер интерпретации применён. `None` — не применялся или
    /// checkpoint старый.
    pub interpret: Option<InterpretProfile>,
    /// Выборка СЫРЫХ train-входов — калибровка для symbolic extraction
    /// после загрузки. `None` у старых checkpoint-ов.
    pub calibration: Option<Array2<f32>>,
    /// Происхождение модели: данные, выбор конфигурации, проверка и финальный
    /// замер. `None` у старых файлов и у пересохранённых моделей — это
    /// «неизвестно», а не «test не открывался».
    pub report: Option<TrainingReport>,
}

/// Равномерная выборка строк для калибровочной секции checkpoint-а.
pub fn calibration_sample(inputs: &Array2<f32>, max_rows: usize) -> Array2<f32> {
    let n = inputs.nrows();
    if n <= max_rows {
        return inputs.clone();
    }
    let stride = n as f32 / max_rows as f32;
    let rows: Vec<usize> = (0..max_rows)
        .map(|k| (k as f32 * stride) as usize)
        .collect();
    inputs.select(ndarray::Axis(0), &rows)
}

/// Читает размеры KAN после structural compaction и проверяет, что checkpoint
/// сохраняет исходный интерфейс модели. Это внешние данные: не позволяем
/// повреждённой секции дойти до `assert!` в конструкторе сети.
fn read_kan_dims(
    bytes: &[u8],
    expected_inputs: usize,
    expected_outputs: usize,
) -> io::Result<Vec<(usize, usize)>> {
    let mut cur = bytes;
    let count = usize::try_from(r_u64(&mut cur)?)
        .map_err(|_| invalid("kan_dims: число слоёв не помещается в usize"))?;
    if count == 0 || count > cur.len() / 16 {
        return Err(invalid("kan_dims: неверное число слоёв"));
    }
    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        let n_in = usize::try_from(r_u64(&mut cur)?)
            .map_err(|_| invalid("kan_dims: размер входа не помещается в usize"))?;
        let n_out = usize::try_from(r_u64(&mut cur)?)
            .map_err(|_| invalid("kan_dims: размер выхода не помещается в usize"))?;
        if n_in == 0 || n_out == 0 {
            return Err(invalid("kan_dims: размеры слоёв должны быть > 0"));
        }
        dims.push((n_in, n_out));
    }
    if !cur.is_empty() {
        return Err(invalid("kan_dims: лишние байты в секции"));
    }
    if dims[0].0 != expected_inputs || dims.last().unwrap().1 != expected_outputs {
        return Err(invalid(
            "kan_dims: размеры не совпадают с интерфейсом checkpoint-а",
        ));
    }
    if dims.windows(2).any(|w| w[0].1 != w[1].0) {
        return Err(invalid("kan_dims: несогласованные размеры соседних слоёв"));
    }
    Ok(dims)
}

// --- примитивы ---

fn w_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn w_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn w_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn r_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn r_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn r_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// --- TLV-поля (для конфига и meta) ---

fn w_field_u64(buf: &mut Vec<u8>, tag: u16, v: u64) {
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}
fn w_field_f32(buf: &mut Vec<u8>, tag: u16, v: f32) {
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Разобрать поток `(tag, len, value)` в map тег→байты. Неизвестные теги
/// останутся в map и просто не будут прочитаны.
fn parse_tlv(mut bytes: &[u8]) -> io::Result<HashMap<u16, Vec<u8>>> {
    let mut map = HashMap::new();
    while !bytes.is_empty() {
        if bytes.len() < 6 {
            return Err(invalid("обрезанный TLV-заголовок"));
        }
        let tag = u16::from_le_bytes([bytes[0], bytes[1]]);
        let len = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]) as usize;
        bytes = &bytes[6..];
        if bytes.len() < len {
            return Err(invalid("обрезанное TLV-значение"));
        }
        map.insert(tag, bytes[..len].to_vec());
        bytes = &bytes[len..];
    }
    Ok(map)
}
fn field_u64(map: &HashMap<u16, Vec<u8>>, tag: u16) -> Option<u64> {
    let b = map.get(&tag)?;
    (b.len() == 8).then(|| u64::from_le_bytes(b[..8].try_into().unwrap()))
}
fn field_f32(map: &HashMap<u16, Vec<u8>>, tag: u16) -> Option<f32> {
    let b = map.get(&tag)?;
    (b.len() == 4).then(|| f32::from_le_bytes(b[..4].try_into().unwrap()))
}

// --- секции ---

fn w_section<W: Write>(w: &mut W, name: &str, payload: &[u8]) -> io::Result<()> {
    let nb = name.as_bytes();
    w_u64(w, nb.len() as u64)?;
    w.write_all(nb)?;
    w_u64(w, payload.len() as u64)?;
    w.write_all(payload)
}
fn read_sections<R: Read + Seek>(
    r: &mut R,
    known: &[&str],
) -> io::Result<HashMap<String, Vec<u8>>> {
    let count = r_u64(r)?;
    let mut map = HashMap::new();
    for _ in 0..count {
        let nlen = r_u64(r)? as usize;
        let mut nb = vec![0u8; nlen];
        r.read_exact(&mut nb)?;
        let name = String::from_utf8(nb).map_err(|_| invalid("имя секции не UTF-8"))?;
        let plen = r_u64(r)? as usize;
        if known.contains(&name.as_str()) {
            let mut payload = vec![0u8; plen];
            r.read_exact(&mut payload)?;
            map.insert(name, payload);
        } else {
            // Неизвестная секция (из будущей версии) — пропускаем по payload_len,
            // не загружая её в память.
            r.seek(SeekFrom::Current(plen as i64))?;
        }
    }
    Ok(map)
}
fn section<'a>(secs: &'a HashMap<String, Vec<u8>>, name: &str) -> io::Result<&'a [u8]> {
    secs.get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("отсутствует обязательная секция {name}")))
}

// --- тензоры и параметры ---

fn w_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<()> {
    let data = t.data();
    w_u32(w, data.ndim() as u32)?;
    for &d in data.shape() {
        w_u64(w, d as u64)?;
    }
    for &v in data.iter() {
        w_f32(w, v)?;
    }
    Ok(())
}
fn r_tensor<R: Read>(r: &mut R) -> io::Result<ArrayD<f32>> {
    let ndim = r_u32(r)? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(r_u64(r)? as usize);
    }
    let len: usize = shape.iter().product();
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(r_f32(r)?);
    }
    ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|_| invalid("несогласованная форма тензора"))
}

fn build_params(params: &[Tensor]) -> io::Result<Vec<u8>> {
    let mut p = Vec::new();
    w_u64(&mut p, params.len() as u64)?;
    for t in params {
        w_tensor(&mut p, t)?;
    }
    Ok(p)
}
fn load_params(bytes: &[u8], params: &[Tensor]) -> io::Result<()> {
    let mut cur = bytes;
    let n = usize::try_from(r_u64(&mut cur)?)
        .map_err(|_| invalid("число параметров не помещается в usize"))?;
    if n != params.len() {
        return Err(invalid("число параметров не совпадает с архитектурой"));
    }
    for (i, p) in params.iter().enumerate() {
        let loaded = r_tensor(&mut cur)?;
        if p.shape() != loaded.shape() {
            return Err(invalid(format!(
                "параметр {i}: форма {:?} не совпадает с ожидаемой {:?}",
                loaded.shape(),
                p.shape()
            )));
        }
        p.set_data(loaded);
    }
    if !cur.is_empty() {
        return Err(invalid("лишние байты в секции параметров"));
    }
    Ok(())
}

// --- specs / нормализатор / словарь (как полезная нагрузка секций) ---

fn build_specs(specs: &[FeatureSpec]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&(specs.len() as u64).to_le_bytes());
    for s in specs {
        match *s {
            FeatureSpec::Continuous => p.extend_from_slice(&0u32.to_le_bytes()),
            FeatureSpec::Categorical { cardinality } => {
                p.extend_from_slice(&1u32.to_le_bytes());
                p.extend_from_slice(&(cardinality as u64).to_le_bytes());
            }
        }
    }
    p
}
fn read_specs(bytes: &[u8]) -> io::Result<Vec<FeatureSpec>> {
    let mut r = bytes;
    let n = usize::try_from(r_u64(&mut r)?)
        .map_err(|_| invalid("число спецификаций не помещается в usize"))?;
    // Даже continuous занимает четыре байта. Проверяем внешний count до
    // Vec::with_capacity, иначе испорченная дублирующая секция обходила бы
    // защиту новой schema.
    if n > r.len() / 4 {
        return Err(invalid(
            "число спецификаций не согласовано с размером секции",
        ));
    }
    let mut specs = Vec::with_capacity(n);
    for _ in 0..n {
        specs.push(match r_u32(&mut r)? {
            0 => FeatureSpec::Continuous,
            1 => {
                let cardinality = usize::try_from(r_u64(&mut r)?)
                    .map_err(|_| invalid("cardinality не помещается в usize"))?;
                if cardinality == 0 {
                    return Err(invalid("cardinality должна быть > 0"));
                }
                FeatureSpec::Categorical { cardinality }
            }
            _ => return Err(invalid("неизвестный тип признака")),
        });
    }
    if !r.is_empty() {
        return Err(invalid("лишние байты в секции feature_specs"));
    }
    Ok(specs)
}

/// Строка в секции: длина + UTF-8. Длина проверяется по остатку среза ДО
/// выделения памяти, иначе испорченный файл заказал бы гигабайты.
fn w_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn r_string(r: &mut &[u8], what: &str) -> io::Result<String> {
    let len = usize::try_from(r_u64(r)?)
        .map_err(|_| invalid(format!("{what}: длина не помещается в usize")))?;
    if len > r.len() {
        return Err(invalid(format!(
            "{what}: длина {len} больше остатка секции"
        )));
    }
    let (head, tail) = r.split_at(len);
    *r = tail;
    String::from_utf8(head.to_vec()).map_err(|_| invalid(format!("{what}: не UTF-8")))
}

/// Секция `schema`: имена, единицы и подписи уровней. Пишется ДОПОЛНИТЕЛЬНО к
/// `feature_specs`, а не вместо неё, — тогда версия формата не меняется и
/// старый бинарь читает новый checkpoint, просто игнорируя незнакомую секцию.
fn build_schema(schema: &ModelSchema) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&(schema.n_inputs() as u64).to_le_bytes());
    p.extend_from_slice(&(schema.n_outputs() as u64).to_le_bytes());
    for column in schema.inputs().iter().chain(schema.outputs().iter()) {
        w_string(&mut p, column.name());
        w_string(&mut p, column.unit().unwrap_or(""));
        match column.ty() {
            ColumnType::Numeric => p.extend_from_slice(&0u32.to_le_bytes()),
            ColumnType::Categorical { levels } => {
                p.extend_from_slice(&1u32.to_le_bytes());
                p.extend_from_slice(&(levels.len() as u64).to_le_bytes());
                for level in levels {
                    w_string(&mut p, level);
                }
            }
        }
    }
    p
}

fn read_schema(bytes: &[u8]) -> io::Result<ModelSchema> {
    let mut r = bytes;
    let n_inputs = usize::try_from(r_u64(&mut r)?)
        .map_err(|_| invalid("schema: число входов не помещается в usize"))?;
    let n_outputs = usize::try_from(r_u64(&mut r)?)
        .map_err(|_| invalid("schema: число выходов не помещается в usize"))?;
    let n_columns = n_inputs
        .checked_add(n_outputs)
        .ok_or_else(|| invalid("schema: суммарное число колонок не помещается в usize"))?;
    // Каждая колонка занимает минимум 20 байт (две длины строк + тип), поэтому
    // заведомо невозможные размеры отсекаются до аллокаций.
    if n_columns > r.len() / 20 {
        return Err(invalid(
            "schema: число колонок не согласовано с размером секции",
        ));
    }

    let mut columns = Vec::with_capacity(n_columns);
    for i in 0..n_columns {
        let role = if i < n_inputs {
            ColumnRole::Input
        } else {
            ColumnRole::Output
        };
        let name = r_string(&mut r, "schema: имя колонки")?;
        let unit = r_string(&mut r, "schema: единица измерения")?;
        let column = match r_u32(&mut r)? {
            0 => Column::numeric(name, role),
            1 => {
                let count = usize::try_from(r_u64(&mut r)?)
                    .map_err(|_| invalid("schema: число уровней не помещается в usize"))?;
                if count > r.len() / 8 {
                    return Err(invalid("schema: число уровней больше остатка секции"));
                }
                let levels = (0..count)
                    .map(|_| r_string(&mut r, "schema: подпись уровня"))
                    .collect::<io::Result<Vec<_>>>()?;
                Column::categorical(name, role, levels)
            }
            _ => return Err(invalid("schema: неизвестный тип колонки")),
        }
        .map_err(invalid)?;
        columns.push(if unit.is_empty() {
            column
        } else {
            column.with_unit(unit)
        });
    }
    if !r.is_empty() {
        return Err(invalid("schema: лишние байты в секции"));
    }

    let outputs = columns.split_off(n_inputs);
    ModelSchema::new(columns, outputs).map_err(invalid)
}

/// Секция `interpret`: какой конвейер интерпретации получила эта модель.
///
/// Пишутся РАЗРЕШЁННЫЕ значения, а не «был профиль»: иначе через полгода не
/// восстановить, с каким порогом прунинга модель стала такой.
fn build_interpret(profile: &InterpretProfile) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&profile.version.to_le_bytes());
    p.extend_from_slice(&profile.l1.to_le_bytes());
    // Прунинг может отсутствовать, поэтому у него отдельный флаг наличия.
    p.extend_from_slice(&u32::from(profile.prune.is_some()).to_le_bytes());
    p.extend_from_slice(&profile.prune.unwrap_or(0.0).to_le_bytes());
    p.extend_from_slice(&(profile.finetune_epochs as u64).to_le_bytes());
    p.extend_from_slice(&u32::from(profile.compact).to_le_bytes());
    p
}

fn read_interpret(bytes: &[u8]) -> io::Result<InterpretProfile> {
    let mut r = bytes;
    let version = r_u32(&mut r)?;
    if version != INTERPRET_PROFILE_VERSION {
        return Err(invalid(format!(
            "interpret: версия профиля {version} не поддерживается (ожидалась {INTERPRET_PROFILE_VERSION})"
        )));
    }
    let l1 = r_f32(&mut r)?;
    let has_prune = r_u32(&mut r)?;
    if has_prune > 1 {
        return Err(invalid("interpret: has_prune должен быть 0 или 1"));
    }
    let prune_value = r_f32(&mut r)?;
    let finetune_epochs = usize::try_from(r_u64(&mut r)?)
        .map_err(|_| invalid("interpret: число эпох не помещается в usize"))?;
    let compact = r_u32(&mut r)?;
    if compact > 1 {
        return Err(invalid("interpret: compact должен быть 0 или 1"));
    }
    if !r.is_empty() {
        return Err(invalid("interpret: лишние байты в секции"));
    }
    let profile = InterpretProfile {
        version,
        l1,
        prune: (has_prune != 0).then_some(prune_value),
        finetune_epochs,
        compact: compact != 0,
    };
    profile.validate().map_err(invalid)?;
    Ok(profile)
}

fn build_norm(n: &Normalizer) -> Vec<u8> {
    let mut p = Vec::new();
    for vec in [&n.mean, &n.std, &n.min, &n.max] {
        p.extend_from_slice(&(vec.len() as u64).to_le_bytes());
        for &x in vec {
            p.extend_from_slice(&x.to_le_bytes());
        }
    }
    p.extend_from_slice(&build_specs(&n.specs));
    p
}
fn read_norm(bytes: &[u8]) -> io::Result<Normalizer> {
    let mut r = bytes;
    let read_vec = |r: &mut &[u8], what: &str| -> io::Result<Vec<f32>> {
        let len = usize::try_from(r_u64(r)?)
            .map_err(|_| invalid(format!("{what}: длина не помещается в usize")))?;
        if len > r.len() / std::mem::size_of::<f32>() {
            return Err(invalid(format!(
                "{what}: длина не согласована с размером секции"
            )));
        }
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(r_f32(r)?);
        }
        Ok(v)
    };
    let mean = read_vec(&mut r, "normalizer mean")?;
    let std = read_vec(&mut r, "normalizer std")?;
    let min = read_vec(&mut r, "normalizer min")?;
    let max = read_vec(&mut r, "normalizer max")?;
    let specs = read_specs(r)?;
    Ok(Normalizer {
        mean,
        std,
        min,
        max,
        specs,
    })
}

#[cfg(feature = "demo")]
fn build_vocab(v: &Vocab) -> Vec<u8> {
    let chars = v.chars();
    let mut p = Vec::new();
    p.extend_from_slice(&(chars.len() as u64).to_le_bytes());
    for &c in chars {
        p.extend_from_slice(&(c as u32).to_le_bytes());
    }
    p
}
#[cfg(feature = "demo")]
fn read_vocab(bytes: &[u8]) -> io::Result<Vocab> {
    let mut r = bytes;
    let n = r_u64(&mut r)? as usize;
    let mut chars = Vec::with_capacity(n);
    for _ in 0..n {
        let c = char::from_u32(r_u32(&mut r)?).ok_or_else(|| invalid("неверный символ"))?;
        chars.push(c);
    }
    Ok(Vocab::from_chars(chars))
}

// --- config / meta ---

fn build_config(c: &ModelConfig) -> Vec<u8> {
    let mut p = Vec::new();
    w_field_u64(&mut p, TAG_D_MODEL, c.d_model as u64);
    w_field_u64(&mut p, TAG_N_HEADS, c.n_heads as u64);
    w_field_u64(&mut p, TAG_N_ENC, c.n_enc_layers as u64);
    w_field_u64(&mut p, TAG_N_DEC, c.n_dec_layers as u64);
    w_field_u64(&mut p, TAG_D_FF, c.d_ff as u64);
    w_field_f32(&mut p, TAG_LN_EPS, c.ln_eps);
    p
}
fn config_from_map(f: &std::collections::HashMap<u16, Vec<u8>>) -> io::Result<ModelConfig> {
    let req = |tag, name| field_u64(f, tag).ok_or_else(|| invalid(format!("config: нет {name}")));
    Ok(ModelConfig {
        d_model: req(TAG_D_MODEL, "d_model")? as usize,
        n_heads: req(TAG_N_HEADS, "n_heads")? as usize,
        n_enc_layers: req(TAG_N_ENC, "n_enc_layers")? as usize,
        n_dec_layers: req(TAG_N_DEC, "n_dec_layers")? as usize,
        d_ff: req(TAG_D_FF, "d_ff")? as usize,
        // Необязательное поле: при отсутствии берём default.
        ln_eps: field_f32(f, TAG_LN_EPS).unwrap_or(1e-5),
    })
}
#[cfg(any(feature = "demo", test))]
fn read_config(bytes: &[u8]) -> io::Result<ModelConfig> {
    config_from_map(&parse_tlv(bytes)?)
}

fn build_numeric_config(nc: &NumericConfig) -> Vec<u8> {
    let mut p = build_config(&nc.transformer);
    let kind = match nc.kind {
        ModelKind::Transformer => 0,
        ModelKind::Mlp => 1,
        ModelKind::Kan => 2,
    };
    w_field_u64(&mut p, TAG_MODEL_KIND, kind);
    w_field_u64(&mut p, TAG_MLP_WIDTH, nc.mlp_width as u64);
    w_field_u64(&mut p, TAG_MLP_LAYERS, nc.mlp_layers as u64);
    w_field_u64(&mut p, TAG_KAN_WIDTH, nc.kan.width as u64);
    w_field_u64(&mut p, TAG_KAN_LAYERS, nc.kan.layers as u64);
    w_field_u64(&mut p, TAG_KAN_GRID, nc.kan.grid as u64);
    let venc = match nc.value.kind {
        ValueEncoderKind::Linear => 0,
        ValueEncoderKind::Mlp => 1,
        ValueEncoderKind::Fourier => 2,
    };
    w_field_u64(&mut p, TAG_VALUE_ENC, venc);
    w_field_u64(&mut p, TAG_FOURIER_BANDS, nc.value.fourier_bands as u64);
    w_field_f32(&mut p, TAG_FOURIER_SCALE, nc.value.fourier_scale);
    p
}
fn read_numeric_config(bytes: &[u8]) -> io::Result<NumericConfig> {
    let f = parse_tlv(bytes)?;
    let transformer = config_from_map(&f)?;
    // Старые transformer-файлы без тега kind грузятся как transformer.
    let kind = match field_u64(&f, TAG_MODEL_KIND).unwrap_or(0) {
        0 => ModelKind::Transformer,
        1 => ModelKind::Mlp,
        2 => ModelKind::Kan,
        other => return Err(invalid(format!("неизвестный model_kind {other}"))),
    };
    // Старые файлы без тега value_encoder грузятся как linear.
    let value_kind = match field_u64(&f, TAG_VALUE_ENC).unwrap_or(0) {
        0 => ValueEncoderKind::Linear,
        1 => ValueEncoderKind::Mlp,
        2 => ValueEncoderKind::Fourier,
        other => return Err(invalid(format!("неизвестный value_encoder {other}"))),
    };
    Ok(NumericConfig {
        kind,
        transformer,
        value: ValueEncoderConfig {
            kind: value_kind,
            fourier_bands: field_u64(&f, TAG_FOURIER_BANDS).unwrap_or(6) as usize,
            fourier_scale: field_f32(&f, TAG_FOURIER_SCALE).unwrap_or(8.0),
        },
        mlp_width: field_u64(&f, TAG_MLP_WIDTH).unwrap_or(128) as usize,
        mlp_layers: field_u64(&f, TAG_MLP_LAYERS).unwrap_or(3) as usize,
        kan: {
            let d = KanConfig::default();
            KanConfig {
                width: field_u64(&f, TAG_KAN_WIDTH).unwrap_or(d.width as u64) as usize,
                layers: field_u64(&f, TAG_KAN_LAYERS).unwrap_or(d.layers as u64) as usize,
                grid: field_u64(&f, TAG_KAN_GRID).unwrap_or(d.grid as u64) as usize,
            }
        },
    })
}

fn build_meta_surrogate(num_outputs: usize) -> Vec<u8> {
    let mut p = Vec::new();
    w_field_u64(&mut p, TAG_NUM_OUTPUTS, num_outputs as u64);
    p
}

// --- заголовок ---

fn w_header<W: Write>(w: &mut W, kind: u32) -> io::Result<()> {
    w_u32(w, MAGIC)?;
    w_u32(w, VERSION)?;
    w_u32(w, kind)
}
fn r_header<R: Read>(r: &mut R, expected_kind: u32) -> io::Result<()> {
    if r_u32(r)? != MAGIC {
        return Err(invalid("не файл модели (неверная магия)"));
    }
    let v = r_u32(r)?;
    if v != VERSION {
        return Err(invalid(format!(
            "неподдерживаемая версия формата: {v} (ожидали {VERSION})"
        )));
    }
    if r_u32(r)? != expected_kind {
        return Err(invalid("неверный тип модели в файле"));
    }
    Ok(())
}

fn write_file(path: &str, kind: u32, sections: &[(&str, Vec<u8>)]) -> io::Result<()> {
    // Через временный файл: неудачное сохранение не должно оставлять на месте
    // прежнего checkpoint обрубок, который потом «не грузится».
    write_atomically(Path::new(path), |file| {
        let mut w = BufWriter::new(file);
        w_header(&mut w, kind)?;
        w_u64(&mut w, sections.len() as u64)?;
        for (name, payload) in sections {
            w_section(&mut w, name, payload)?;
        }
        // flush до возврата: ошибку из Drop у BufWriter никто не увидит.
        w.flush()
    })
}

fn validate_normalizer(
    normalizer: &Normalizer,
    expected_specs: &[FeatureSpec],
    name: &str,
) -> io::Result<()> {
    let n = expected_specs.len();
    if normalizer.mean.len() != n
        || normalizer.std.len() != n
        || normalizer.min.len() != n
        || normalizer.max.len() != n
    {
        return Err(invalid(format!(
            "{name}: размер статистик не совпадает со схемой ({n} колонок)"
        )));
    }
    if normalizer.specs != expected_specs {
        return Err(invalid(format!(
            "{name}: типы колонок не совпадают со схемой"
        )));
    }
    Ok(())
}

fn validate_output_normalizer(normalizer: &Normalizer, expected_outputs: usize) -> io::Result<()> {
    if normalizer.mean.len() != expected_outputs
        || normalizer.std.len() != expected_outputs
        || normalizer.min.len() != expected_outputs
        || normalizer.max.len() != expected_outputs
        || normalizer.specs.len() != expected_outputs
    {
        return Err(invalid(format!(
            "out_norm: размер статистик не совпадает со схемой ({expected_outputs} колонок)"
        )));
    }
    if normalizer
        .specs
        .iter()
        .any(|spec| !matches!(spec, FeatureSpec::Continuous))
    {
        return Err(invalid("out_norm: выходы регрессии должны быть числовыми"));
    }
    Ok(())
}

fn validate_checkpoint_components(
    nc: &NumericConfig,
    schema: &ModelSchema,
    model: &NumericModel,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
) -> io::Result<Vec<FeatureSpec>> {
    if model.kind() != nc.kind {
        return Err(invalid("тип модели не совпадает с NumericConfig"));
    }
    let expected_dims = (schema.n_inputs(), schema.n_outputs());
    if model.interface_dims() != expected_dims {
        return Err(invalid(format!(
            "интерфейс модели {:?} не совпадает со схемой {:?}",
            model.interface_dims(),
            expected_dims
        )));
    }

    let specs = schema.feature_specs();
    if let NumericModel::Transformer(transformer) = model {
        if transformer.input_specs() != specs {
            return Err(invalid(
                "типы входов transformer-модели не совпадают со схемой",
            ));
        }
    }
    validate_normalizer(in_norm, &specs, "in_norm")?;
    validate_output_normalizer(out_norm, schema.n_outputs())?;
    Ok(specs)
}

// --- публичный API ---

/// Сохраняет численную модель (transformer, MLP или KAN): конфиг с `model_kind`,
/// схема данных, параметры и нормализаторы — каждая часть отдельной секцией.
///
/// Схема заменила пару «спецификации + число выходов» в сигнатуре: и то и
/// другое из неё выводится, а рассогласовать их больше нельзя.
#[allow(clippy::too_many_arguments)]
pub fn save_numeric(
    path: &str,
    nc: &NumericConfig,
    schema: &ModelSchema,
    model: &NumericModel,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    calibration: Option<&Array2<f32>>,
    interpret: Option<&InterpretProfile>,
    // Происхождение модели. `None` — его просто нет (модель загружена и
    // пересохранена), и это не то же самое, что «test не открывался».
    report: Option<&TrainingReport>,
) -> io::Result<()> {
    let specs = validate_checkpoint_components(nc, schema, model, in_norm, out_norm)?;
    let num_outputs = schema.n_outputs();
    let specs = specs.as_slice();
    let mut sections = vec![
        ("config", build_numeric_config(nc)),
        ("meta", build_meta_surrogate(num_outputs)),
        ("feature_specs", build_specs(specs)),
        // Имена/единицы/уровни едут отдельной секцией: старый бинарь её
        // пропустит и прочитает checkpoint как раньше.
        ("schema", build_schema(schema)),
        ("params", build_params(&model.parameters())?),
        ("in_norm", build_norm(in_norm)),
        ("out_norm", build_norm(out_norm)),
    ];
    // Маски — не обучаемые параметры и не должны раздувать parameter_count,
    // но фиксируют hard-prune при последующем fine-tune из checkpoint-а.
    if let Some(profile) = interpret {
        if nc.kind != ModelKind::Kan {
            return Err(invalid("interpret допустим только у KAN checkpoint-а"));
        }
        profile.validate().map_err(invalid)?;
        sections.push(("interpret", build_interpret(profile)));
    }
    if let Some(report) = report {
        sections.push(("training_report", build_report(report)));
    }
    if let Some(masks) = model.kan_masks() {
        sections.push(("kan_masks", build_params(&masks)?));
    }
    // Точные размеры слоёв: после структурного сжатия ширина скрытых слоёв
    // неоднородна и не выводится из конфига.
    if let Some(kan) = model.as_kan() {
        let mut p = Vec::new();
        let dims = kan.layer_dims();
        w_u64(&mut p, dims.len() as u64)?;
        for (i, o) in dims {
            w_u64(&mut p, i as u64)?;
            w_u64(&mut p, o as u64)?;
        }
        sections.push(("kan_dims", p));
    }
    // Калибровка — сырые train-строки, поэтому сохраняем её только у KAN,
    // где она нужна для symbolic extraction после загрузки.
    if nc.kind == ModelKind::Kan {
        if let Some(c) = calibration {
            if c.nrows() == 0 || c.ncols() != in_norm.n_features() {
                return Err(invalid(
                    "calibration должна быть непустой [N, F] с F = числу входов",
                ));
            }
            let mut p = Vec::new();
            w_tensor(&mut p, &Tensor::constant(c.clone().into_dyn()))?;
            sections.push(("calibration", p));
        }
    }
    write_file(path, KIND_SURROGATE, &sections)
}

/// Загружает численную модель: по `model_kind` восстанавливает архитектуру
/// (transformer/MLP/KAN) и заполняет веса. Возвращает модель и нормализаторы.
pub fn load_numeric(path: &str) -> io::Result<(NumericModel, Normalizer, Normalizer)> {
    let checkpoint = load_numeric_full(path)?;
    Ok((checkpoint.model, checkpoint.in_norm, checkpoint.out_norm))
}

/// Загружает численную модель вместе с метаданными, нужными для повторного
/// сохранения из GUI или других оболочек.
pub fn load_numeric_full(path: &str) -> io::Result<NumericCheckpoint> {
    let mut r = BufReader::new(File::open(path)?);
    r_header(&mut r, KIND_SURROGATE)?;
    let secs = read_sections(&mut r, SURROGATE_SECTIONS)?;

    let nc = read_numeric_config(section(&secs, "config")?)?;
    let num_outputs = usize::try_from(
        field_u64(&parse_tlv(section(&secs, "meta")?)?, TAG_NUM_OUTPUTS)
            .ok_or_else(|| invalid("meta: нет num_outputs"))?,
    )
    .map_err(|_| invalid("meta: num_outputs не помещается в usize"))?;
    let specs = read_specs(section(&secs, "feature_specs")?)?;
    let in_norm = read_norm(section(&secs, "in_norm")?)?;
    let out_norm = read_norm(section(&secs, "out_norm")?)?;
    validate_normalizer(&in_norm, &specs, "in_norm")?;
    // Эта проверка идёт до synthetic fallback и тем самым ограничивает
    // num_outputs реальным размером уже прочитанной секции нормализатора.
    validate_output_normalizer(&out_norm, num_outputs)?;

    // Схема: из секции у новых файлов, синтетическая — у старых. Категориальные
    // типы при этом сохраняются, подписями становятся сами коды.
    let schema = match secs.get("schema") {
        Some(bytes) => {
            let schema = read_schema(bytes)?;
            // Две секции описывают одно и то же; расхождение означает битый
            // файл, а не повод выбрать одну из версий.
            if schema.feature_specs() != specs {
                return Err(invalid(
                    "schema и feature_specs описывают разные типы входов",
                ));
            }
            schema
        }
        None => ModelSchema::synthetic_from_specs(&specs, num_outputs).map_err(invalid)?,
    };
    schema
        .check_dims(specs.len(), num_outputs)
        .map_err(invalid)?;

    // Структурно сжатая KAN имеет неоднородные слои: их размеры лежат в
    // секции kan_dims; без неё (legacy) строим по конфигу.
    let model = match (nc.kind, secs.get("kan_dims")) {
        (ModelKind::Kan, Some(bytes)) => {
            let dims = read_kan_dims(bytes, specs.len(), num_outputs)?;
            NumericModel::Kan(crate::kan::KanNet::from_dims(&dims, nc.kan.grid))
        }
        _ => nc.build(&specs, num_outputs),
    };
    load_params(section(&secs, "params")?, &model.parameters())?;
    // Секция появилась после первоначальной поддержки KAN. Её отсутствие
    // означает старый checkpoint без hard-prune — конструктор оставляет маски
    // единичными, поэтому он продолжает читаться без изменения predict.
    if let Some(bytes) = secs.get("kan_masks") {
        let masks = model
            .kan_masks()
            .ok_or_else(|| invalid("kan_masks есть у не-KAN checkpoint-а"))?;
        load_params(bytes, &masks)?;
    }
    let interpret = match secs.get("interpret") {
        Some(bytes) => Some(read_interpret(bytes)?),
        None => None,
    };
    if interpret.is_some() && nc.kind != ModelKind::Kan {
        return Err(invalid("interpret есть у не-KAN checkpoint-а"));
    }
    let calibration = if nc.kind == ModelKind::Kan {
        match secs.get("calibration") {
            Some(bytes) => {
                let calibration = r_tensor(&mut bytes.as_slice())?
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| invalid("calibration должна быть [N, F]"))?;
                if calibration.nrows() == 0 || calibration.ncols() != specs.len() {
                    return Err(invalid(
                        "calibration должна быть непустой [N, F] с F = числу входов",
                    ));
                }
                Some(calibration)
            }
            None => None,
        }
    } else {
        None
    };
    // Отчёт необязателен и версионирован: незнакомая версия читается как его
    // отсутствие, чтобы из-за неё не потерять саму модель.
    let report = match secs.get("training_report") {
        Some(bytes) => read_report(bytes)?,
        None => None,
    };
    Ok(NumericCheckpoint {
        model,
        in_norm,
        out_norm,
        config: nc,
        schema,
        interpret,
        calibration,
        report,
    })
}

/// Секция `training_report`: происхождение модели.
///
/// Необязательная и со своей версией: старый checkpoint читается как раньше и
/// даёт `None` — «неизвестно», а не «test не открывался». Незнакомую версию
/// тоже читаем как отсутствие отчёта: терять из-за неё саму модель нельзя.
fn build_report(report: &TrainingReport) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&TRAINING_REPORT_VERSION.to_le_bytes());
    p.extend_from_slice(report.dataset.as_bytes());
    w_blob(&mut p, &build_schema(&report.schema));
    w_blob(&mut p, &build_stamp(&report.stamp));
    build_selection(&mut p, &report.selection);
    w_opt_blob(&mut p, report.check.as_ref().map(build_check));
    w_opt_blob(&mut p, report.final_run.as_ref().map(build_final));
    p
}

fn read_report(bytes: &[u8]) -> io::Result<Option<TrainingReport>> {
    let mut r = bytes;
    let version = r_u32(&mut r)?;
    if version != TRAINING_REPORT_VERSION {
        return Ok(None);
    }
    let mut fingerprint = [0u8; 32];
    r.read_exact(&mut fingerprint)?;
    let schema = read_schema(&r_blob(&mut r, "training_report: схема")?)?;
    let mut stamp = read_stamp(&r_blob(&mut r, "training_report: отпечаток запуска")?)?;
    // Отпечаток данных хранится один раз — в самом отчёте. Здесь он
    // восстанавливается в stamp, чтобы дальше эти два поля не расходились.
    stamp.dataset = DatasetFingerprint::from_bytes(fingerprint);
    let selection = read_selection(&mut r)?;
    let check = r_opt_blob(&mut r, "training_report: проверка")?
        .map(|bytes| read_check(&bytes))
        .transpose()?;
    let final_run = r_opt_blob(&mut r, "training_report: финал")?
        .map(|bytes| read_final(&bytes))
        .transpose()?;
    if !r.is_empty() {
        return Err(invalid("training_report: лишние байты"));
    }
    Ok(Some(TrainingReport {
        dataset: DatasetFingerprint::from_bytes(fingerprint),
        schema,
        stamp,
        selection,
        check,
        final_run,
    }))
}

/// Длина + содержимое: вложенный блок читается, не зная его устройства.
fn w_blob(buf: &mut Vec<u8>, payload: &[u8]) {
    buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    buf.extend_from_slice(payload);
}

fn r_blob(r: &mut &[u8], what: &str) -> io::Result<Vec<u8>> {
    let len = usize::try_from(r_u64(r)?)
        .map_err(|_| invalid(format!("{what}: длина не помещается в usize")))?;
    // Длина сверяется с остатком ДО выделения памяти: иначе битый файл
    // попросил бы гигабайты.
    if len > r.len() {
        return Err(invalid(format!(
            "{what}: длина {len} больше остатка секции"
        )));
    }
    let (head, tail) = r.split_at(len);
    *r = tail;
    Ok(head.to_vec())
}

fn w_opt_blob(buf: &mut Vec<u8>, payload: Option<Vec<u8>>) {
    match payload {
        Some(bytes) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            w_blob(buf, &bytes);
        }
        None => buf.extend_from_slice(&0u32.to_le_bytes()),
    }
}

fn r_opt_blob(r: &mut &[u8], what: &str) -> io::Result<Option<Vec<u8>>> {
    match r_u32(r)? {
        0 => Ok(None),
        1 => Ok(Some(r_blob(r, what)?)),
        other => Err(invalid(format!("{what}: флаг наличия {other} не 0 и не 1"))),
    }
}

fn w_count(buf: &mut Vec<u8>, n: usize) {
    buf.extend_from_slice(&(n as u64).to_le_bytes());
}

/// Количество элементов с проверкой против остатка секции.
///
/// `min_item` — сколько байт занимает самый короткий возможный элемент. Без
/// этой проверки заявленное «миллиард точек» привело бы к выделению памяти по
/// числу из битого файла.
fn r_count(r: &mut &[u8], min_item: usize, what: &str) -> io::Result<usize> {
    let n = usize::try_from(r_u64(r)?)
        .map_err(|_| invalid(format!("{what}: количество не помещается в usize")))?;
    let needed = n
        .checked_mul(min_item)
        .ok_or_else(|| invalid(format!("{what}: количество {n} переполняет размер")))?;
    if needed > r.len() {
        return Err(invalid(format!(
            "{what}: заявлено {n} элементов, а в секции осталось {} байт",
            r.len()
        )));
    }
    Ok(n)
}

fn build_stamp(stamp: &RunStamp) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&stamp.dataset_revision.to_le_bytes());
    p.extend_from_slice(&stamp.final_init_seed.to_le_bytes());
    build_split(&mut p, stamp.split);
    w_blob(&mut p, &build_numeric_config(&stamp.candidate.config));
    build_train_config(&mut p, &stamp.candidate.train);
    w_opt_blob(
        &mut p,
        stamp.candidate.interpret.as_ref().map(build_interpret),
    );
    p
}

fn read_stamp(bytes: &[u8]) -> io::Result<RunStamp> {
    let mut r = bytes;
    let dataset_revision = r_u64(&mut r)?;
    let final_init_seed = r_u64(&mut r)?;
    let split = read_split(&mut r)?;
    let config = read_numeric_config(&r_blob(&mut r, "stamp: конфигурация")?)?;
    let train = read_train_config(&mut r)?;
    let interpret = r_opt_blob(&mut r, "stamp: профиль интерпретации")?
        .map(|bytes| read_interpret(&bytes))
        .transpose()?;
    if !r.is_empty() {
        return Err(invalid("stamp: лишние байты"));
    }
    Ok(RunStamp {
        // Отпечаток данных лежит в самом отчёте: две копии одного числа рано
        // или поздно разойдутся.
        dataset: DatasetFingerprint::from_bytes([0; 32]),
        dataset_revision,
        split,
        candidate: CandidateSpec {
            config,
            train,
            interpret,
        },
        final_init_seed,
    })
}

fn build_split(buf: &mut Vec<u8>, split: SplitPlan) {
    match split {
        SplitPlan::Holdout {
            train_frac,
            val_frac,
            split_seed,
        } => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&train_frac.to_le_bytes());
            buf.extend_from_slice(&val_frac.to_le_bytes());
            buf.extend_from_slice(&split_seed.to_le_bytes());
        }
        SplitPlan::KFold {
            k,
            folds_seed,
            test_frac,
            test_seed,
        } => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&(k as u64).to_le_bytes());
            buf.extend_from_slice(&folds_seed.to_le_bytes());
            buf.extend_from_slice(&test_frac.to_le_bytes());
            buf.extend_from_slice(&test_seed.to_le_bytes());
        }
    }
}

fn read_split(r: &mut &[u8]) -> io::Result<SplitPlan> {
    match r_u32(r)? {
        0 => Ok(SplitPlan::Holdout {
            train_frac: r_f32(r)?,
            val_frac: r_f32(r)?,
            split_seed: r_u64(r)?,
        }),
        1 => Ok(SplitPlan::KFold {
            k: usize::try_from(r_u64(r)?).map_err(|_| invalid("split: k не помещается в usize"))?,
            folds_seed: r_u64(r)?,
            test_frac: r_f32(r)?,
            test_seed: r_u64(r)?,
        }),
        other => Err(invalid(format!("split: неизвестный вид разбиения {other}"))),
    }
}

fn build_train_config(buf: &mut Vec<u8>, cfg: &TrainConfig) {
    buf.extend_from_slice(&(cfg.epochs as u64).to_le_bytes());
    buf.extend_from_slice(&(cfg.batch_size as u64).to_le_bytes());
    buf.extend_from_slice(&cfg.lr.to_le_bytes());
    buf.extend_from_slice(&cfg.seed.to_le_bytes());
    match cfg.schedule {
        LrSchedule::Constant => buf.extend_from_slice(&0u32.to_le_bytes()),
        LrSchedule::WarmupCosine {
            warmup_frac,
            min_lr_ratio,
        } => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&warmup_frac.to_le_bytes());
            buf.extend_from_slice(&min_lr_ratio.to_le_bytes());
        }
    }
}

fn read_train_config(r: &mut &[u8]) -> io::Result<TrainConfig> {
    let epochs =
        usize::try_from(r_u64(r)?).map_err(|_| invalid("train: epochs не помещается в usize"))?;
    let batch_size = usize::try_from(r_u64(r)?)
        .map_err(|_| invalid("train: batch_size не помещается в usize"))?;
    let lr = r_f32(r)?;
    let seed = r_u64(r)?;
    let schedule = match r_u32(r)? {
        0 => LrSchedule::Constant,
        1 => LrSchedule::WarmupCosine {
            warmup_frac: r_f32(r)?,
            min_lr_ratio: r_f32(r)?,
        },
        other => return Err(invalid(format!("train: неизвестное расписание {other}"))),
    };
    Ok(TrainConfig {
        epochs,
        batch_size,
        lr,
        seed,
        schedule,
    })
}

fn build_selection(buf: &mut Vec<u8>, selection: &Selection) {
    match selection {
        Selection::Manual => buf.extend_from_slice(&0u32.to_le_bytes()),
        Selection::Search {
            objective,
            seeds,
            objective_value,
            label,
        } => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&objective_code(*objective).to_le_bytes());
            w_count(buf, seeds.len());
            for seed in seeds {
                buf.extend_from_slice(&seed.to_le_bytes());
            }
            buf.extend_from_slice(&objective_value.to_le_bytes());
            w_string(buf, label);
        }
    }
}

fn read_selection(r: &mut &[u8]) -> io::Result<Selection> {
    match r_u32(r)? {
        0 => Ok(Selection::Manual),
        1 => {
            let objective = read_objective(r_u32(r)?)?;
            let n = r_count(r, 8, "selection: seeds")?;
            let mut seeds = Vec::with_capacity(n);
            for _ in 0..n {
                seeds.push(r_u64(r)?);
            }
            Ok(Selection::Search {
                objective,
                seeds,
                objective_value: r_f32(r)?,
                label: r_string(r, "selection: подпись строки")?,
            })
        }
        other => Err(invalid(format!(
            "selection: неизвестный способ выбора {other}"
        ))),
    }
}

fn objective_code(objective: SearchObjective) -> u32 {
    match objective {
        SearchObjective::WorstOutputR2 => 0,
        SearchObjective::AggregateR2 => 1,
        SearchObjective::MeanOutputR2 => 2,
        SearchObjective::Nrmse => 3,
    }
}

fn read_objective(code: u32) -> io::Result<SearchObjective> {
    match code {
        0 => Ok(SearchObjective::WorstOutputR2),
        1 => Ok(SearchObjective::AggregateR2),
        2 => Ok(SearchObjective::MeanOutputR2),
        3 => Ok(SearchObjective::Nrmse),
        other => Err(invalid(format!(
            "selection: неизвестная цель поиска {other}"
        ))),
    }
}

fn build_metrics(buf: &mut Vec<u8>, m: &Metrics) {
    buf.extend_from_slice(&m.rmse.to_le_bytes());
    buf.extend_from_slice(&m.mae.to_le_bytes());
    buf.extend_from_slice(&m.rel_error.to_le_bytes());
    buf.extend_from_slice(&m.r2.to_le_bytes());
}

/// Самая короткая запись метрик — четыре `f32`.
const METRICS_BYTES: usize = 16;

fn read_metrics(r: &mut &[u8]) -> io::Result<Metrics> {
    Ok(Metrics {
        rmse: r_f32(r)?,
        mae: r_f32(r)?,
        rel_error: r_f32(r)?,
        r2: r_f32(r)?,
    })
}

fn build_source(buf: &mut Vec<u8>, source: EvalSource) {
    match source {
        EvalSource::Validation => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
        }
        EvalSource::Cv { k } => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&(k as u64).to_le_bytes());
        }
        EvalSource::Test => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
        }
    }
}

fn read_source(r: &mut &[u8]) -> io::Result<EvalSource> {
    let tag = r_u32(r)?;
    let k = usize::try_from(r_u64(r)?).map_err(|_| invalid("source: k не помещается в usize"))?;
    match tag {
        0 => Ok(EvalSource::Validation),
        1 => Ok(EvalSource::Cv { k }),
        2 => Ok(EvalSource::Test),
        other => Err(invalid(format!(
            "source: неизвестное происхождение {other}"
        ))),
    }
}

/// Точка истории: эпоха, train loss и — только там, где был замер — метрики.
const POINT_MIN_BYTES: usize = 8 + 4 + 4;

fn build_history(buf: &mut Vec<u8>, history: &TrainingHistory) {
    build_source(buf, history.source);
    match history.best_epoch {
        Some(epoch) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&(epoch as u64).to_le_bytes());
        }
        None => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes());
        }
    }
    buf.extend_from_slice(&u32::from(history.stopped_early).to_le_bytes());
    w_count(buf, history.points.len());
    for point in &history.points {
        buf.extend_from_slice(&(point.epoch as u64).to_le_bytes());
        buf.extend_from_slice(&point.train_loss.to_le_bytes());
        match &point.val {
            Some(m) => {
                buf.extend_from_slice(&1u32.to_le_bytes());
                build_metrics(buf, m);
            }
            None => buf.extend_from_slice(&0u32.to_le_bytes()),
        }
    }
}

fn read_history(r: &mut &[u8]) -> io::Result<TrainingHistory> {
    let source = read_source(r)?;
    let has_best = r_u32(r)?;
    if has_best > 1 {
        return Err(invalid("history: флаг лучшей эпохи не 0 и не 1"));
    }
    let best = usize::try_from(r_u64(r)?)
        .map_err(|_| invalid("history: номер эпохи не помещается в usize"))?;
    let stopped_early = match r_u32(r)? {
        0 => false,
        1 => true,
        other => {
            return Err(invalid(format!(
                "history: флаг остановки {other} не 0 и не 1"
            )))
        }
    };
    let n = r_count(r, POINT_MIN_BYTES, "history: точки")?;
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        let epoch = usize::try_from(r_u64(r)?)
            .map_err(|_| invalid("history: эпоха не помещается в usize"))?;
        let train_loss = r_f32(r)?;
        let val = match r_u32(r)? {
            0 => None,
            1 => Some(read_metrics(r)?),
            other => {
                return Err(invalid(format!(
                    "history: флаг наличия метрик {other} не 0 и не 1"
                )))
            }
        };
        points.push(EpochPoint {
            epoch,
            train_loss,
            val,
        });
    }
    Ok(TrainingHistory {
        points,
        source,
        best_epoch: (has_best == 1).then_some(best),
        stopped_early,
    })
}

fn build_interpret_report(buf: &mut Vec<u8>, report: &InterpretReport) {
    w_blob(buf, &build_interpret(&report.profile));
    w_count(buf, report.per_layer.len());
    for (active, total) in &report.per_layer {
        buf.extend_from_slice(&(*active as u64).to_le_bytes());
        buf.extend_from_slice(&(*total as u64).to_le_bytes());
    }
    buf.extend_from_slice(&(report.active_edges.0 as u64).to_le_bytes());
    buf.extend_from_slice(&(report.active_edges.1 as u64).to_le_bytes());
    for value in [
        report.r2_before,
        report.r2_after_prune,
        report.r2_after_finetune,
        report.r2_after_compact,
    ] {
        buf.extend_from_slice(&u32::from(value.is_some()).to_le_bytes());
        buf.extend_from_slice(&value.unwrap_or(0.0).to_le_bytes());
    }
    match report.compaction {
        Some(c) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            for value in [
                c.nodes_before,
                c.nodes_after,
                c.params_before,
                c.params_after,
            ] {
                buf.extend_from_slice(&(value as u64).to_le_bytes());
            }
        }
        None => buf.extend_from_slice(&0u32.to_le_bytes()),
    }
    buf.extend_from_slice(&u32::from(report.cancelled).to_le_bytes());
}

fn read_interpret_report(r: &mut &[u8]) -> io::Result<InterpretReport> {
    let profile = read_interpret(&r_blob(r, "interpret-отчёт: профиль")?)?;
    let n = r_count(r, 16, "interpret-отчёт: слои")?;
    let mut per_layer = Vec::with_capacity(n);
    for _ in 0..n {
        let active = r_usize(r, "interpret-отчёт: активные рёбра")?;
        let total = r_usize(r, "interpret-отчёт: всего рёбер")?;
        per_layer.push((active, total));
    }
    let active_edges = (
        r_usize(r, "interpret-отчёт: активные рёбра")?,
        r_usize(r, "interpret-отчёт: всего рёбер")?,
    );
    let mut r2 = [None; 4];
    for slot in &mut r2 {
        let has = r_u32(r)?;
        if has > 1 {
            return Err(invalid("interpret-отчёт: флаг наличия R² не 0 и не 1"));
        }
        let value = r_f32(r)?;
        *slot = (has == 1).then_some(value);
    }
    let compaction = match r_u32(r)? {
        0 => None,
        1 => Some(CompactReport {
            nodes_before: r_usize(r, "interpret-отчёт: узлы до")?,
            nodes_after: r_usize(r, "interpret-отчёт: узлы после")?,
            params_before: r_usize(r, "interpret-отчёт: параметры до")?,
            params_after: r_usize(r, "interpret-отчёт: параметры после")?,
        }),
        other => {
            return Err(invalid(format!(
                "interpret-отчёт: флаг сжатия {other} не 0 и не 1"
            )))
        }
    };
    let cancelled = match r_u32(r)? {
        0 => false,
        1 => true,
        other => {
            return Err(invalid(format!(
                "interpret-отчёт: флаг отмены {other} не 0 и не 1"
            )))
        }
    };
    Ok(InterpretReport {
        profile,
        per_layer,
        active_edges,
        r2_before: r2[0],
        r2_after_prune: r2[1],
        r2_after_finetune: r2[2],
        compaction,
        r2_after_compact: r2[3],
        cancelled,
    })
}

fn r_usize(r: &mut &[u8], what: &str) -> io::Result<usize> {
    usize::try_from(r_u64(r)?).map_err(|_| invalid(format!("{what}: не помещается в usize")))
}

fn build_check(check: &CheckRecord) -> Vec<u8> {
    let mut p = Vec::new();
    build_source(&mut p, check.source);
    build_metrics(&mut p, &check.metrics);
    p.extend_from_slice(&check.r2_std_folds.to_le_bytes());
    w_count(&mut p, check.per_output.len());
    for m in &check.per_output {
        build_metrics(&mut p, m);
    }
    w_count(&mut p, check.histories.len());
    for history in &check.histories {
        let mut block = Vec::new();
        build_history(&mut block, history);
        w_blob(&mut p, &block);
    }
    w_count(&mut p, check.interpret.len());
    for report in &check.interpret {
        let mut block = Vec::new();
        build_interpret_report(&mut block, report);
        w_blob(&mut p, &block);
    }
    p
}

fn read_check(bytes: &[u8]) -> io::Result<CheckRecord> {
    let mut r = bytes;
    let source = read_source(&mut r)?;
    let metrics = read_metrics(&mut r)?;
    let r2_std_folds = r_f32(&mut r)?;
    let n = r_count(&mut r, METRICS_BYTES, "проверка: метрики выходов")?;
    let mut per_output = Vec::with_capacity(n);
    for _ in 0..n {
        per_output.push(read_metrics(&mut r)?);
    }
    let n = r_count(&mut r, 8, "проверка: истории folds")?;
    let mut histories = Vec::with_capacity(n);
    for _ in 0..n {
        let block = r_blob(&mut r, "проверка: история fold")?;
        histories.push(read_history(&mut block.as_slice())?);
    }
    let n = r_count(&mut r, 8, "проверка: отчёты конвейера")?;
    let mut interpret = Vec::with_capacity(n);
    for _ in 0..n {
        let block = r_blob(&mut r, "проверка: отчёт конвейера")?;
        interpret.push(read_interpret_report(&mut block.as_slice())?);
    }
    if !r.is_empty() {
        return Err(invalid("проверка: лишние байты"));
    }
    Ok(CheckRecord {
        source,
        metrics,
        per_output,
        r2_std_folds,
        histories,
        interpret,
    })
}

fn build_final(record: &FinalRecord) -> Vec<u8> {
    let mut p = Vec::new();
    build_history(&mut p, &record.history);
    build_metrics(&mut p, &record.eval.metrics);
    w_count(&mut p, record.eval.per_output.len());
    for m in &record.eval.per_output {
        build_metrics(&mut p, m);
    }
    p.extend_from_slice(&(record.eval.origin.test_rows as u64).to_le_bytes());
    p.extend_from_slice(&record.eval.origin.final_init_seed.to_le_bytes());
    build_split(&mut p, record.eval.origin.plan);
    let mut interpret = None;
    if let Some(report) = &record.interpret {
        let mut block = Vec::new();
        build_interpret_report(&mut block, report);
        interpret = Some(block);
    }
    w_opt_blob(&mut p, interpret);
    p
}

fn read_final(bytes: &[u8]) -> io::Result<FinalRecord> {
    let mut r = bytes;
    let history = read_history(&mut r)?;
    let metrics = read_metrics(&mut r)?;
    let n = r_count(&mut r, METRICS_BYTES, "финал: метрики выходов")?;
    let mut per_output = Vec::with_capacity(n);
    for _ in 0..n {
        per_output.push(read_metrics(&mut r)?);
    }
    let test_rows = r_usize(&mut r, "финал: строк в test")?;
    let final_init_seed = r_u64(&mut r)?;
    let plan = read_split(&mut r)?;
    let interpret = r_opt_blob(&mut r, "финал: отчёт конвейера")?
        .map(|block| read_interpret_report(&mut block.as_slice()))
        .transpose()?;
    if !r.is_empty() {
        return Err(invalid("финал: лишние байты"));
    }
    Ok(FinalRecord {
        history,
        eval: FinalEval {
            metrics,
            per_output,
            origin: FinalOrigin {
                test_rows,
                final_init_seed,
                plan,
            },
        },
        interpret,
    })
}

/// Сохраняет char-LM модель вместе с конфигом и словарём.
#[cfg(feature = "demo")]
pub fn save_text(
    path: &str,
    cfg: &ModelConfig,
    vocab: &Vocab,
    model: &TextModel,
) -> io::Result<()> {
    let sections = [
        ("config", build_config(cfg)),
        ("vocab", build_vocab(vocab)),
        ("params", build_params(&model.parameters())?),
    ];
    write_file(path, KIND_TEXT, &sections)
}

/// Загружает char-LM модель и словарь.
#[cfg(feature = "demo")]
pub fn load_text(path: &str) -> io::Result<(TextModel, Vocab)> {
    let mut r = BufReader::new(File::open(path)?);
    r_header(&mut r, KIND_TEXT)?;
    let secs = read_sections(&mut r, TEXT_SECTIONS)?;

    let cfg = read_config(section(&secs, "config")?)?;
    let vocab = read_vocab(section(&secs, "vocab")?)?;

    let model = TextModel::new(&cfg, vocab.len());
    load_params(section(&secs, "params")?, &model.parameters())?;
    Ok((model, vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::optim::Adam;
    use ndarray::Array2;

    fn tmp_path(name: &str) -> String {
        std::env::temp_dir()
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_enc_layers: 1,
            n_dec_layers: 1,
            d_ff: 32,
            ln_eps: 1e-5,
        }
    }

    fn numeric_cfg(kind: ModelKind) -> NumericConfig {
        NumericConfig {
            kind,
            transformer: tiny_cfg(),
            value: ValueEncoderConfig::default(),
            mlp_width: 16,
            mlp_layers: 2,
            kan: KanConfig {
                width: 8,
                layers: 2,
                grid: 5,
            },
        }
    }

    /// save -> load -> predict даёт тот же результат для каждого типа модели.
    fn round_trip_for(kind: ModelKind, name: &str) {
        let nc = numeric_cfg(kind);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();
        let model = nc.build(&specs, 1);

        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        let x = Tensor::constant(
            Array2::from_shape_vec((2, 2), vec![0.1, -0.2, 0.3, 0.4])
                .unwrap()
                .into_dyn(),
        );
        let before = model.predict(&x).data();

        let path = tmp_path(name);
        save_numeric(
            &path, &nc, &schema, &model, &in_norm, &out_norm, None, None, None,
        )
        .unwrap();
        let (loaded, _in2, _out2) = load_numeric(&path).unwrap();
        let after = loaded.predict(&x).data();
        let full = load_numeric_full(&path).unwrap();

        assert_eq!(
            before, after,
            "{kind:?}: предсказания после загрузки разошлись"
        );
        assert_eq!(full.config.kind, kind);
        assert_eq!(full.schema, schema);
        assert_eq!(full.schema.feature_specs(), specs);
        assert_eq!(full.schema.n_outputs(), 1);
        assert_eq!(full.in_norm.n_features(), 2);
        assert_eq!(full.out_norm.n_features(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// Сохранение поверх существующего checkpoint заменяет его целиком, и
    /// заменённый файл грузится: запись идёт через временный файл.
    #[test]
    fn saving_over_an_existing_checkpoint_replaces_it_and_loads() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        let path = tmp_path("replace_checkpoint.bin");
        // На месте назначения лежит мусор прежнего запуска.
        std::fs::write(&path, [0xff_u8, 0xff, 0x00, 0x01]).unwrap();

        let model = nc.build(&specs, 1);
        save_numeric(
            &path, &nc, &schema, &model, &in_norm, &out_norm, None, None, None,
        )
        .unwrap();

        let x = Tensor::constant(
            Array2::from_shape_vec((1, 2), vec![0.25, -0.5])
                .unwrap()
                .into_dyn(),
        );
        let (loaded, _, _) = load_numeric(&path).unwrap();
        assert_eq!(model.predict(&x).data(), loaded.predict(&x).data());

        // Временных файлов рядом не осталось.
        let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
        let name = std::path::Path::new(&path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.ok()?.file_name().to_string_lossy().into_owned();
                n.starts_with(&format!(".{name}.tmp")).then_some(n)
            })
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        std::fs::remove_file(&path).ok();
    }

    fn sample_history(source: EvalSource) -> TrainingHistory {
        TrainingHistory {
            points: vec![
                EpochPoint {
                    epoch: 1,
                    train_loss: 0.5,
                    val: None,
                },
                EpochPoint {
                    epoch: 2,
                    train_loss: 0.25,
                    val: Some(Metrics {
                        rmse: 1.0,
                        mae: 0.5,
                        rel_error: 0.1,
                        r2: 0.9,
                    }),
                },
            ],
            source,
            best_epoch: Some(2),
            stopped_early: true,
        }
    }

    fn sample_report(schema: &ModelSchema) -> TrainingReport {
        let metrics = Metrics {
            rmse: 1.0,
            mae: 0.5,
            rel_error: 0.1,
            r2: 0.9,
        };
        let split = SplitPlan::KFold {
            k: 3,
            folds_seed: 7,
            test_frac: 0.2,
            test_seed: 9,
        };
        TrainingReport {
            dataset: DatasetFingerprint::from_bytes([7; 32]),
            schema: schema.clone(),
            stamp: RunStamp {
                dataset: DatasetFingerprint::from_bytes([0; 32]),
                dataset_revision: 4,
                split,
                candidate: CandidateSpec {
                    config: numeric_cfg(ModelKind::Kan),
                    train: TrainConfig {
                        epochs: 7,
                        batch_size: 16,
                        lr: 3e-3,
                        seed: 11,
                        schedule: LrSchedule::WarmupCosine {
                            warmup_frac: 0.1,
                            min_lr_ratio: 0.01,
                        },
                    },
                    interpret: Some(InterpretProfile::v1()),
                },
                final_init_seed: 5,
            },
            selection: Selection::Search {
                objective: SearchObjective::Nrmse,
                seeds: vec![0, 1, 2],
                objective_value: -0.25,
                label: "kan width=16 L=2".to_string(),
            },
            check: Some(CheckRecord {
                source: EvalSource::Cv { k: 3 },
                metrics: metrics.clone(),
                per_output: vec![metrics.clone()],
                r2_std_folds: 0.02,
                histories: vec![
                    sample_history(EvalSource::Cv { k: 3 }),
                    sample_history(EvalSource::Cv { k: 3 }),
                ],
                interpret: vec![InterpretReport {
                    profile: InterpretProfile::v1(),
                    per_layer: vec![(3, 8), (2, 4)],
                    active_edges: (5, 12),
                    r2_before: Some(0.8),
                    r2_after_prune: Some(0.7),
                    r2_after_finetune: Some(0.85),
                    compaction: Some(CompactReport {
                        nodes_before: 16,
                        nodes_after: 9,
                        params_before: 400,
                        params_after: 220,
                    }),
                    r2_after_compact: None,
                    cancelled: false,
                }],
            }),
            final_run: Some(FinalRecord {
                history: sample_history(EvalSource::Validation),
                eval: FinalEval {
                    metrics,
                    per_output: vec![Metrics {
                        rmse: 2.0,
                        mae: 1.0,
                        rel_error: 0.2,
                        r2: 0.8,
                    }],
                    origin: FinalOrigin {
                        test_rows: 18,
                        final_init_seed: 5,
                        plan: split,
                    },
                },
                interpret: None,
            }),
        }
    }

    /// Отчёт переживает запись и чтение целиком: истории по folds пишутся
    /// полностью, без прореживания, и `val` остаётся только там, где замер был.
    #[test]
    fn training_report_survives_a_round_trip() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let report = sample_report(&schema);

        let path = tmp_path("training_report.bin");
        save_numeric(
            &path,
            &nc,
            &schema,
            &model,
            &in_norm,
            &out_norm,
            None,
            None,
            Some(&report),
        )
        .unwrap();
        let loaded = load_numeric_full(&path).unwrap().report.expect("отчёт");

        assert_eq!(loaded.dataset, report.dataset);
        assert_eq!(loaded.schema, report.schema);
        assert_eq!(loaded.stamp.split, report.stamp.split);
        assert_eq!(loaded.stamp.candidate, report.stamp.candidate);
        assert_eq!(loaded.stamp.final_init_seed, 5);
        assert_eq!(
            loaded.stamp.candidate.train.seed, 11,
            "seed проверки и финальный seed — разные величины"
        );
        assert_eq!(loaded.selection, report.selection);

        assert!(loaded.test_disclosed());
        let check = loaded.check.expect("запись о проверке");
        assert_eq!(check.source, EvalSource::Cv { k: 3 });
        assert_eq!(check.histories.len(), 2);
        assert_eq!(check.histories[0].points.len(), 2);
        assert!(check.histories[0].points[0].val.is_none());
        assert_eq!(check.histories[0].points[1].val.as_ref().unwrap().r2, 0.9);
        assert!(check.histories[0].stopped_early);
        assert_eq!(check.interpret.len(), 1);
        assert_eq!(check.interpret[0].per_layer, vec![(3, 8), (2, 4)]);
        assert_eq!(check.interpret[0].compaction.unwrap().params_after, 220);
        assert!(check.interpret[0].r2_after_compact.is_none());

        let final_run = loaded.final_run.expect("запись о финале");
        assert_eq!(final_run.eval.origin.test_rows, 18);
        assert_eq!(final_run.eval.origin.plan, report.stamp.split);
        assert!(final_run.interpret.is_none());
        std::fs::remove_file(&path).ok();
    }

    /// Секция необязательна: checkpoint без неё читается как раньше и даёт
    /// «неизвестно», а не «test не открывался».
    #[test]
    fn a_checkpoint_without_a_report_still_loads() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        let path = tmp_path("no_training_report.bin");
        save_numeric(
            &path, &nc, &schema, &model, &in_norm, &out_norm, None, None, None,
        )
        .unwrap();
        assert!(load_numeric_full(&path).unwrap().report.is_none());
        std::fs::remove_file(&path).ok();
    }

    /// Незнакомая версия отчёта не должна стоить модели: секция читается как
    /// отсутствующая.
    #[test]
    fn an_unknown_report_version_is_ignored() {
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let mut bytes = build_report(&sample_report(&schema));
        bytes[..4].copy_from_slice(&(TRAINING_REPORT_VERSION + 1).to_le_bytes());
        assert!(read_report(&bytes).unwrap().is_none());
    }

    /// Обрезанная секция — ошибка, а не попытка выделить память по числу из
    /// битого файла.
    #[test]
    fn a_truncated_report_is_rejected() {
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let bytes = build_report(&sample_report(&schema));
        for cut in [8, bytes.len() / 3, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                read_report(&bytes[..cut]).is_err(),
                "обрезка до {cut} байт принята"
            );
        }
        // Заявленная длина больше остатка тоже отвергается до выделения.
        let mut lying = bytes.clone();
        let count_at = 4 + 32;
        lying[count_at..count_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(read_report(&lying).is_err());
    }

    #[test]
    fn transformer_round_trip_identical() {
        round_trip_for(ModelKind::Transformer, "surr_tf.bin");
    }

    #[test]
    fn mlp_round_trip_identical() {
        round_trip_for(ModelKind::Mlp, "surr_mlp.bin");
    }

    #[test]
    fn kan_round_trip_identical() {
        round_trip_for(ModelKind::Kan, "surr_kan.bin");
    }

    #[test]
    fn kan_dims_rejects_invalid_checkpoint_topology() {
        let mut valid = Vec::new();
        w_u64(&mut valid, 2).unwrap();
        w_u64(&mut valid, 2).unwrap();
        w_u64(&mut valid, 3).unwrap();
        w_u64(&mut valid, 3).unwrap();
        w_u64(&mut valid, 1).unwrap();
        assert_eq!(read_kan_dims(&valid, 2, 1).unwrap(), vec![(2, 3), (3, 1)]);

        let mut wrong_interface = valid.clone();
        // Первый размер входа в первой паре: 2 -> 4.
        wrong_interface[8..16].copy_from_slice(&4_u64.to_le_bytes());
        assert!(read_kan_dims(&wrong_interface, 2, 1).is_err());

        let mut broken_chain = Vec::new();
        w_u64(&mut broken_chain, 2).unwrap();
        w_u64(&mut broken_chain, 2).unwrap();
        w_u64(&mut broken_chain, 3).unwrap();
        w_u64(&mut broken_chain, 4).unwrap();
        w_u64(&mut broken_chain, 1).unwrap();
        assert!(read_kan_dims(&broken_chain, 2, 1).is_err());
    }

    #[test]
    fn calibration_is_stored_only_for_kan() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let path = tmp_path("surr_mlp_without_calibration.bin");
        save_numeric(
            &path,
            &nc,
            &ModelSchema::synthetic_from_specs(&specs, 1).unwrap(),
            &model,
            &in_norm,
            &out_norm,
            Some(&data.inputs),
            None,
            None,
        )
        .unwrap();
        assert!(load_numeric_full(&path).unwrap().calibration.is_none());
        std::fs::remove_file(&path).ok();
    }

    /// Схема с именами, единицами и уровнями переживает round-trip, а секция
    /// `feature_specs` остаётся на месте для старых читателей.
    #[test]
    fn schema_round_trip_keeps_names_units_and_levels() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let schema = ModelSchema::new(
            vec![
                Column::numeric("температура", ColumnRole::Input)
                    .unwrap()
                    .with_unit("°C"),
                Column::categorical(
                    "материал",
                    ColumnRole::Input,
                    vec!["песок".into(), "глина".into(), "торф".into()],
                )
                .unwrap(),
            ],
            vec![Column::numeric("влажность", ColumnRole::Output)
                .unwrap()
                .with_unit("%")],
        )
        .unwrap();
        let specs = schema.feature_specs();
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        let path = tmp_path("surr_schema.bin");
        save_numeric(
            &path, &nc, &schema, &model, &in_norm, &out_norm, None, None, None,
        )
        .unwrap();

        // Симуляция старого reader-а: в списке известных секций schema нет,
        // но все прежние обязательные секции остаются читаемыми.
        let mut legacy_reader = BufReader::new(File::open(&path).unwrap());
        r_header(&mut legacy_reader, KIND_SURROGATE).unwrap();
        let legacy_sections = read_sections(
            &mut legacy_reader,
            &[
                "config",
                "meta",
                "feature_specs",
                "params",
                "in_norm",
                "out_norm",
            ],
        )
        .unwrap();
        assert!(!legacy_sections.contains_key("schema"));
        assert_eq!(
            read_specs(section(&legacy_sections, "feature_specs").unwrap()).unwrap(),
            specs
        );

        let checkpoint = load_numeric_full(&path).unwrap();
        assert_eq!(checkpoint.schema, schema);
        assert_eq!(checkpoint.schema.inputs()[0].unit(), Some("°C"));
        assert_eq!(
            checkpoint.schema.inputs()[1].category_level(2).unwrap(),
            "торф"
        );
        assert_eq!(checkpoint.schema.n_outputs(), 1);
        std::fs::remove_file(&path).ok();
    }

    /// Конвейер интерпретации сохраняется РАЗРЕШЁННЫМИ значениями: через
    /// полгода должно быть видно, с каким порогом модель стала такой.
    #[test]
    fn interpret_profile_round_trip() {
        let nc = numeric_cfg(ModelKind::Kan);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        let profile = InterpretProfile {
            version: INTERPRET_PROFILE_VERSION,
            l1: 2e-3,
            prune: Some(0.1),
            finetune_epochs: 7,
            compact: false,
        };
        let path = tmp_path("surr_interpret.bin");
        save_numeric(
            &path,
            &nc,
            &schema,
            &model,
            &in_norm,
            &out_norm,
            None,
            Some(&profile),
            None,
        )
        .unwrap();
        let checkpoint = load_numeric_full(&path).unwrap();
        assert_eq!(checkpoint.interpret, Some(profile));

        // Reader старой версии не знает секцию interpret, но пропускает её и
        // продолжает читать все прежние обязательные секции.
        let mut legacy_reader = BufReader::new(File::open(&path).unwrap());
        r_header(&mut legacy_reader, KIND_SURROGATE).unwrap();
        let legacy_sections = read_sections(
            &mut legacy_reader,
            &[
                "config",
                "meta",
                "feature_specs",
                "schema",
                "params",
                "kan_masks",
                "kan_dims",
                "calibration",
                "in_norm",
                "out_norm",
            ],
        )
        .unwrap();
        assert!(!legacy_sections.contains_key("interpret"));
        assert!(legacy_sections.contains_key("params"));

        // Без конвейера секции нет, и это не ошибка.
        let plain = tmp_path("surr_interpret_none.bin");
        save_numeric(
            &plain, &nc, &schema, &model, &in_norm, &out_norm, None, None, None,
        )
        .unwrap();
        assert_eq!(load_numeric_full(&plain).unwrap().interpret, None);

        // Профиль без прунинга: флаг наличия переживает round-trip.
        let no_prune = InterpretProfile {
            prune: None,
            ..profile
        };
        let third = tmp_path("surr_interpret_no_prune.bin");
        save_numeric(
            &third,
            &nc,
            &schema,
            &model,
            &in_norm,
            &out_norm,
            None,
            Some(&no_prune),
            None,
        )
        .unwrap();
        assert_eq!(load_numeric_full(&third).unwrap().interpret, Some(no_prune));

        for p in [path, plain, third] {
            std::fs::remove_file(&p).ok();
        }
    }

    /// Чужая версия профиля читаться не должна: «интерпретируемая KAN» обязана
    /// означать одно и то же.
    #[test]
    fn unknown_interpret_version_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&0.001f32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0.05f32.to_le_bytes());
        bytes.extend_from_slice(&20u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        let err = read_interpret(&bytes).unwrap_err();
        assert!(err.to_string().contains("версия профиля 99"), "{err}");

        // Обрезанная секция — тоже ошибка, а не молчаливые нули.
        assert!(read_interpret(&bytes[..6]).is_err());

        // Поля-флаги имеют каноническое представление, другие значения не
        // должны молча трактоваться как true.
        let mut invalid_bool = build_interpret(&InterpretProfile::v1());
        invalid_bool[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(read_interpret(&invalid_bool).is_err());
        invalid_bool = build_interpret(&InterpretProfile::v1());
        invalid_bool[24..28].copy_from_slice(&2u32.to_le_bytes());
        assert!(read_interpret(&invalid_bool).is_err());
    }

    #[test]
    fn interpret_metadata_is_only_saved_for_supported_kan_profile() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let path = tmp_path("surr_interpret_mlp.bin");
        std::fs::remove_file(&path).ok();

        let err = save_numeric(
            &path,
            &nc,
            &schema,
            &model,
            &in_norm,
            &out_norm,
            None,
            Some(&InterpretProfile::v1()),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("только у KAN"), "{err}");

        // Внешний файл может обойти writer: reader обязан независимо
        // отвергнуть ту же невозможную комбинацию.
        write_file(
            &path,
            KIND_SURROGATE,
            &[
                ("config", build_numeric_config(&nc)),
                ("meta", build_meta_surrogate(1)),
                ("feature_specs", build_specs(&specs)),
                ("schema", build_schema(&schema)),
                ("params", build_params(&model.parameters()).unwrap()),
                ("in_norm", build_norm(&in_norm)),
                ("out_norm", build_norm(&out_norm)),
                ("interpret", build_interpret(&InterpretProfile::v1())),
            ],
        )
        .unwrap();
        let err = load_numeric_full(&path)
            .err()
            .expect("файл должен отвергаться");
        assert!(err.to_string().contains("не-KAN"), "{err}");
        std::fs::remove_file(&path).ok();

        let unsupported = InterpretProfile {
            version: 99,
            ..InterpretProfile::v1()
        };
        let kan = numeric_cfg(ModelKind::Kan);
        let kan_model = kan.build(&specs, 1);
        let err = save_numeric(
            &path,
            &kan,
            &schema,
            &kan_model,
            &in_norm,
            &out_norm,
            None,
            Some(&unsupported),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("версия профиля 99"), "{err}");
        assert!(!std::path::Path::new(&path).exists());
    }

    /// Checkpoint без секции `schema` (старый файл) читается и получает
    /// синтетическую схему — с сохранением категориального типа.
    #[test]
    fn legacy_checkpoint_without_schema_gets_synthetic_one() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![
            FeatureSpec::Continuous,
            FeatureSpec::Categorical { cardinality: 3 },
        ];
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        // Пишем те же секции, что и старая версия: без `schema`.
        let path = tmp_path("surr_legacy_schema.bin");
        let sections = vec![
            ("config", build_numeric_config(&nc)),
            ("meta", build_meta_surrogate(1)),
            ("feature_specs", build_specs(&specs)),
            ("params", build_params(&model.parameters()).unwrap()),
            ("in_norm", build_norm(&in_norm)),
            ("out_norm", build_norm(&out_norm)),
        ];
        write_file(&path, KIND_SURROGATE, &sections).unwrap();

        let checkpoint = load_numeric_full(&path).unwrap();
        assert_eq!(checkpoint.schema.input_names(), vec!["x0", "x1"]);
        assert_eq!(checkpoint.schema.output_names(), vec!["y0"]);
        assert_eq!(checkpoint.schema.feature_specs(), specs);
        assert_eq!(
            checkpoint.schema.inputs()[1].category_level(1).unwrap(),
            "1"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Секции `schema` и `feature_specs` описывают одно и то же; расхождение —
    /// битый файл, а не повод выбрать одну из версий.
    #[test]
    fn inconsistent_schema_and_specs_are_rejected() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        // В схеме второй вход категориальный, в feature_specs — числовой.
        let mismatched = ModelSchema::synthetic_from_specs(
            &[
                FeatureSpec::Continuous,
                FeatureSpec::Categorical { cardinality: 4 },
            ],
            1,
        )
        .unwrap();
        let path = tmp_path("surr_schema_mismatch.bin");
        let sections = vec![
            ("config", build_numeric_config(&nc)),
            ("meta", build_meta_surrogate(1)),
            ("feature_specs", build_specs(&specs)),
            ("schema", build_schema(&mismatched)),
            ("params", build_params(&model.parameters()).unwrap()),
            ("in_norm", build_norm(&in_norm)),
            ("out_norm", build_norm(&out_norm)),
        ];
        write_file(&path, KIND_SURROGATE, &sections).unwrap();
        assert!(load_numeric_full(&path).is_err());

        // Схема на другое число выходов, чем meta.
        let wrong_outputs = ModelSchema::synthetic_from_specs(&specs, 3).unwrap();
        let sections = vec![
            ("config", build_numeric_config(&nc)),
            ("meta", build_meta_surrogate(1)),
            ("feature_specs", build_specs(&specs)),
            ("schema", build_schema(&wrong_outputs)),
            ("params", build_params(&model.parameters()).unwrap()),
            ("in_norm", build_norm(&in_norm)),
            ("out_norm", build_norm(&out_norm)),
        ];
        write_file(&path, KIND_SURROGATE, &sections).unwrap();
        assert!(load_numeric_full(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// Испорченная секция не должна приводить к гигантским аллокациям.
    #[test]
    fn corrupt_schema_section_is_rejected() {
        // Заявлено 2 входа и 1 выход, но байтов на них нет.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        assert!(read_schema(&bytes).is_err());

        // Длина имени больше остатка секции.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 64]);
        assert!(read_schema(&bytes).is_err());
    }

    #[test]
    fn corrupt_metadata_counts_are_rejected_before_allocation_or_build() {
        let mut oversized = Vec::new();
        oversized.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(read_specs(&oversized).is_err());

        let mut zero_cardinality = Vec::new();
        zero_cardinality.extend_from_slice(&1u64.to_le_bytes());
        zero_cardinality.extend_from_slice(&1u32.to_le_bytes());
        zero_cardinality.extend_from_slice(&0u64.to_le_bytes());
        assert!(read_specs(&zero_cardinality).is_err());

        let mut trailing = build_specs(&[FeatureSpec::Continuous]);
        trailing.push(0);
        assert!(read_specs(&trailing).is_err());

        let mut oversized_norm = Vec::new();
        oversized_norm.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(read_norm(&oversized_norm).is_err());
    }

    #[test]
    fn save_rejects_components_inconsistent_with_schema() {
        let nc = numeric_cfg(ModelKind::Mlp);
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let specs = schema.feature_specs();
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(8, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let path = tmp_path("surr_inconsistent_save.bin");
        std::fs::remove_file(&path).ok();

        let wrong_dims = ModelSchema::synthetic(3, 1).unwrap();
        assert!(save_numeric(
            &path,
            &nc,
            &wrong_dims,
            &model,
            &in_norm,
            &out_norm,
            None,
            None,
            None
        )
        .unwrap_err()
        .to_string()
        .contains("интерфейс модели"));

        let wrong_kind = numeric_cfg(ModelKind::Kan);
        assert!(save_numeric(
            &path,
            &wrong_kind,
            &schema,
            &model,
            &in_norm,
            &out_norm,
            None,
            None,
            None
        )
        .unwrap_err()
        .to_string()
        .contains("тип модели"));

        let categorical_specs = vec![
            FeatureSpec::Continuous,
            FeatureSpec::Categorical { cardinality: 3 },
        ];
        let wrong_norm = Normalizer::fit(&data.inputs, &categorical_specs);
        assert!(save_numeric(
            &path,
            &nc,
            &schema,
            &model,
            &wrong_norm,
            &out_norm,
            None,
            None,
            None
        )
        .unwrap_err()
        .to_string()
        .contains("in_norm"));
        assert!(!std::path::Path::new(&path).exists());
    }

    /// Структурно сжатая KAN (неоднородные слои) восстанавливается по секции
    /// kan_dims; калибровочная выборка переживает round-trip.
    #[test]
    fn compacted_kan_round_trip_with_calibration() {
        let nc = numeric_cfg(ModelKind::Kan);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let mut model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(16, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));

        // Омертвляем два скрытых узла и физически сжимаем топологию.
        {
            let kan = model.as_kan().unwrap();
            let (_, width) = kan.layer_dims()[0];
            assert!(width >= 3, "тесту нужен скрытый слой шире 2");
            kan.layers[1].prune_edge(0, 0);
            kan.layers[1].prune_edge(1, 0);
        }
        let report = model.as_kan_mut().unwrap().compact();
        assert!(report.nodes_after < report.nodes_before);
        let dims = model.as_kan().unwrap().layer_dims();

        let x = Tensor::constant(
            Array2::from_shape_vec((3, 2), vec![0.1, -0.4, 0.9, 0.2, -1.1, 0.6])
                .unwrap()
                .into_dyn(),
        );
        let before = model.predict(&x).data();

        let calib = calibration_sample(&data.inputs, 8);
        let path = tmp_path("surr_compact_kan.bin");
        save_numeric(
            &path,
            &nc,
            &ModelSchema::synthetic_from_specs(&specs, 1).unwrap(),
            &model,
            &in_norm,
            &out_norm,
            Some(&calib),
            None,
            None,
        )
        .unwrap();
        let checkpoint = load_numeric_full(&path).unwrap();
        assert_eq!(checkpoint.model.as_kan().unwrap().layer_dims(), dims);
        assert_eq!(before, checkpoint.model.predict(&x).data());
        assert_eq!(checkpoint.calibration.as_ref(), Some(&calib));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pruned_kan_round_trip_preserves_hard_masks() {
        let nc = numeric_cfg(ModelKind::Kan);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(32, 0);
        let in_norm = Normalizer::fit(&data.inputs, &specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let calibration = in_norm.transform(&data.inputs);

        let kan = model.as_kan().unwrap();
        let before = kan.active_edges();
        // Оставляем только почти максимальные рёбра; при случайной инициализации
        // это гарантированно меняет маску хотя бы в одном слое.
        let report = kan.prune_edges(0.999, &calibration);
        let pruned = report.totals();
        assert!(pruned.0 < before.0, "тест должен реально что-то отсечь");

        let path = tmp_path("surr_pruned_kan.bin");
        save_numeric(
            &path,
            &nc,
            &ModelSchema::synthetic_from_specs(&specs, 1).unwrap(),
            &model,
            &in_norm,
            &out_norm,
            None,
            None,
            None,
        )
        .unwrap();
        let checkpoint = load_numeric_full(&path).unwrap();
        let loaded_kan = checkpoint.model.as_kan().unwrap();
        assert_eq!(loaded_kan.active_edges(), pruned);

        // Маски после загрузки по-прежнему блокируют градиенты отсечённых рёбер.
        let x = Tensor::constant(calibration.into_dyn());
        let y = Tensor::constant(Array2::<f32>::zeros((data.len(), 1)).into_dyn());
        let mut opt = Adam::new(checkpoint.model.parameters(), 1e-3);
        for _ in 0..3 {
            opt.zero_grad();
            let loss = checkpoint.model.loss(&x, &y);
            loss.backward();
            opt.step();
        }
        assert_eq!(checkpoint.model.as_kan().unwrap().active_edges(), pruned);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(feature = "demo")]
    fn text_round_trip_identical() {
        let cfg = tiny_cfg();
        let vocab = Vocab::from_text("hello world abcdef");
        let model = TextModel::new(&cfg, vocab.len());

        let src = vocab.encode("hello ");
        let memory = model.encode_src(&src);
        let dec = vocab.encode("wor");
        let before = model.next_logits(&dec, &memory);

        let path = tmp_path("text_test.bin");
        save_text(&path, &cfg, &vocab, &model).unwrap();
        let (loaded, vocab2) = load_text(&path).unwrap();

        let memory2 = loaded.encode_src(&vocab2.encode("hello "));
        let after = loaded.next_logits(&vocab2.encode("wor"), &memory2);

        assert_eq!(before, after, "логиты после загрузки разошлись");
        std::fs::remove_file(&path).ok();
    }

    /// Конфиг: неизвестный тег пропускается, отсутствующее необязательное
    /// поле (ln_eps) берёт default. Это и есть forward-compat поля.
    #[test]
    fn config_tlv_skips_unknown_and_defaults() {
        let cfg = tiny_cfg();
        let mut payload = build_config(&cfg);
        w_field_u64(&mut payload, 999, 12345); // «будущее» неизвестное поле
        let back = read_config(&payload).unwrap();
        assert_eq!(back.d_model, cfg.d_model);
        assert_eq!(back.d_ff, cfg.d_ff);

        // Без ln_eps -> default.
        let mut p2 = Vec::new();
        w_field_u64(&mut p2, TAG_D_MODEL, 8);
        w_field_u64(&mut p2, TAG_N_HEADS, 2);
        w_field_u64(&mut p2, TAG_N_ENC, 1);
        w_field_u64(&mut p2, TAG_N_DEC, 1);
        w_field_u64(&mut p2, TAG_D_FF, 16);
        assert_eq!(read_config(&p2).unwrap().ln_eps, 1e-5);
    }

    /// Неизвестная секция пропускается по payload_len, не грузится в память.
    #[test]
    fn sections_skip_unknown() {
        let mut buf = Vec::new();
        w_u64(&mut buf, 2).unwrap();
        w_section(&mut buf, "known", &[1u8, 2, 3]).unwrap();
        w_section(&mut buf, "future_section", &[9u8, 9]).unwrap();

        let mut cur = std::io::Cursor::new(buf);
        let map = read_sections(&mut cur, &["known"]).unwrap();
        assert_eq!(map.get("known").unwrap(), &vec![1u8, 2, 3]);
        assert!(!map.contains_key("future_section")); // пропущена, в map не попала
    }
}
