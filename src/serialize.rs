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

use crate::config::ModelConfig;
use crate::data::{Normalizer, Vocab};
use crate::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use crate::numeric_model::{KanConfig, ModelKind, NumericConfig, NumericModel};
use crate::schema::{Column, ColumnRole, ColumnType, ModelSchema};
use crate::tensor::Tensor;
use crate::textmodel::TextModel;
use ndarray::{Array2, ArrayD, Ix2, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

const MAGIC: u32 = 0x5452_4653; // "TRFS"
const VERSION: u32 = 2;
const KIND_SURROGATE: u32 = 0;
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
    "params",
    "kan_masks",
    "kan_dims",
    "calibration",
    "in_norm",
    "out_norm",
];
const TEXT_SECTIONS: &[&str] = &["config", "vocab", "params"];

/// Полное содержимое численного checkpoint-а.
pub struct NumericCheckpoint {
    pub model: NumericModel,
    pub in_norm: Normalizer,
    pub out_norm: Normalizer,
    pub config: NumericConfig,
    /// Схема данных. У старых checkpoint-ов достраивается синтетически из
    /// сохранённых `feature_specs`, поэтому поле есть всегда.
    pub schema: ModelSchema,
    /// Выборка СЫРЫХ train-входов — калибровка для symbolic extraction
    /// после загрузки. `None` у старых checkpoint-ов.
    pub calibration: Option<Array2<f32>>,
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

fn build_vocab(v: &Vocab) -> Vec<u8> {
    let chars = v.chars();
    let mut p = Vec::new();
    p.extend_from_slice(&(chars.len() as u64).to_le_bytes());
    for &c in chars {
        p.extend_from_slice(&(c as u32).to_le_bytes());
    }
    p
}
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
    let mut w = BufWriter::new(File::create(path)?);
    w_header(&mut w, kind)?;
    w_u64(&mut w, sections.len() as u64)?;
    for (name, payload) in sections {
        w_section(&mut w, name, payload)?;
    }
    w.flush()
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
pub fn save_numeric(
    path: &str,
    nc: &NumericConfig,
    schema: &ModelSchema,
    model: &NumericModel,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    calibration: Option<&Array2<f32>>,
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
    Ok(NumericCheckpoint {
        model,
        in_norm,
        out_norm,
        config: nc,
        schema,
        calibration,
    })
}

/// Сохраняет char-LM модель вместе с конфигом и словарём.
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
        save_numeric(&path, &nc, &schema, &model, &in_norm, &out_norm, None).unwrap();
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
        save_numeric(&path, &nc, &schema, &model, &in_norm, &out_norm, None).unwrap();

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
        assert!(
            save_numeric(&path, &nc, &wrong_dims, &model, &in_norm, &out_norm, None)
                .unwrap_err()
                .to_string()
                .contains("интерфейс модели")
        );

        let wrong_kind = numeric_cfg(ModelKind::Kan);
        assert!(save_numeric(
            &path,
            &wrong_kind,
            &schema,
            &model,
            &in_norm,
            &out_norm,
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
        assert!(
            save_numeric(&path, &nc, &schema, &model, &wrong_norm, &out_norm, None)
                .unwrap_err()
                .to_string()
                .contains("in_norm")
        );
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
