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
use crate::numeric_model::{ModelKind, NumericConfig, NumericModel};
use crate::tensor::Tensor;
use crate::textmodel::TextModel;
use ndarray::{ArrayD, IxDyn};
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
const TAG_MODEL_KIND: u16 = 7; // 0=transformer, 1=mlp; отсутствует -> transformer
const TAG_MLP_WIDTH: u16 = 8;
const TAG_MLP_LAYERS: u16 = 9;
const TAG_VALUE_ENC: u16 = 10; // 0=linear, 1=mlp, 2=fourier; отсутствует -> linear
const TAG_FOURIER_BANDS: u16 = 11;
const TAG_FOURIER_SCALE: u16 = 12;
// Тег секции meta.
const TAG_NUM_OUTPUTS: u16 = 1;

// Известные секции по типам моделей (всё остальное при чтении пропускается).
const SURROGATE_SECTIONS: &[&str] = &[
    "config",
    "meta",
    "feature_specs",
    "params",
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
    pub specs: Vec<FeatureSpec>,
    pub num_outputs: usize,
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
    let n = r_u64(&mut cur)? as usize;
    if n != params.len() {
        return Err(invalid("число параметров не совпадает с архитектурой"));
    }
    for p in params {
        p.set_data(r_tensor(&mut cur)?);
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
    let n = r_u64(&mut r)? as usize;
    let mut specs = Vec::with_capacity(n);
    for _ in 0..n {
        specs.push(match r_u32(&mut r)? {
            0 => FeatureSpec::Continuous,
            1 => FeatureSpec::Categorical {
                cardinality: r_u64(&mut r)? as usize,
            },
            _ => return Err(invalid("неизвестный тип признака")),
        });
    }
    Ok(specs)
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
    let read_vec = |r: &mut &[u8]| -> io::Result<Vec<f32>> {
        let len = r_u64(r)? as usize;
        let mut v = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(r_f32(r)?);
        }
        Ok(v)
    };
    let mean = read_vec(&mut r)?;
    let std = read_vec(&mut r)?;
    let min = read_vec(&mut r)?;
    let max = read_vec(&mut r)?;
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
    };
    w_field_u64(&mut p, TAG_MODEL_KIND, kind);
    w_field_u64(&mut p, TAG_MLP_WIDTH, nc.mlp_width as u64);
    w_field_u64(&mut p, TAG_MLP_LAYERS, nc.mlp_layers as u64);
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

// --- публичный API ---

/// Сохраняет численную модель (трансформер или MLP): конфиг с `model_kind`,
/// спецификация признаков, число выходов, параметры и нормализаторы — каждая
/// часть отдельной секцией.
pub fn save_numeric(
    path: &str,
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    num_outputs: usize,
    model: &NumericModel,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
) -> io::Result<()> {
    let sections = [
        ("config", build_numeric_config(nc)),
        ("meta", build_meta_surrogate(num_outputs)),
        ("feature_specs", build_specs(specs)),
        ("params", build_params(&model.parameters())?),
        ("in_norm", build_norm(in_norm)),
        ("out_norm", build_norm(out_norm)),
    ];
    write_file(path, KIND_SURROGATE, &sections)
}

/// Загружает численную модель: по `model_kind` восстанавливает архитектуру
/// (трансформер/MLP) и заполняет веса. Возвращает модель и нормализаторы.
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
    let num_outputs = field_u64(&parse_tlv(section(&secs, "meta")?)?, TAG_NUM_OUTPUTS)
        .ok_or_else(|| invalid("meta: нет num_outputs"))? as usize;
    let specs = read_specs(section(&secs, "feature_specs")?)?;

    let model = nc.build(&specs, num_outputs);
    load_params(section(&secs, "params")?, &model.parameters())?;
    let in_norm = read_norm(section(&secs, "in_norm")?)?;
    let out_norm = read_norm(section(&secs, "out_norm")?)?;
    Ok(NumericCheckpoint {
        model,
        in_norm,
        out_norm,
        config: nc,
        specs,
        num_outputs,
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
        }
    }

    /// save -> load -> predict даёт тот же результат для каждого типа модели.
    fn round_trip_for(kind: ModelKind, name: &str) {
        let nc = numeric_cfg(kind);
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
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
        save_numeric(&path, &nc, &specs, 1, &model, &in_norm, &out_norm).unwrap();
        let (loaded, _in2, _out2) = load_numeric(&path).unwrap();
        let after = loaded.predict(&x).data();
        let full = load_numeric_full(&path).unwrap();

        assert_eq!(
            before, after,
            "{kind:?}: предсказания после загрузки разошлись"
        );
        assert_eq!(full.config.kind, kind);
        assert_eq!(full.specs, specs);
        assert_eq!(full.num_outputs, 1);
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
