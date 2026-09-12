//! Отпечаток набора данных: что именно считается «теми же данными».
//!
//! Нужен там, где ревизии сессии недостаточно. Ревизия — счётчик внутри одного
//! запуска приложения: повторное открытие того же файла и перезапуск дают новый
//! номер, и потраченный test «возвращается». Отпечаток же берётся из самих
//! чисел, поэтому переживает и то, и другое.
//!
//! В отпечаток входит МОДЕЛЬНОЕ представление данных: размеры, типы входов с
//! кардинальностью категорий и сами значения в порядке строк и колонок. Не
//! входят путь и формат файла, отброшенные при разметке колонки, имена и
//! единицы — переименование `x0` в «температура» не меняет задачу и не должно
//! сбрасывать бюджет test. Полная схема хранится рядом отдельно.
//!
//! Кодирование каноническое и версионированное: длины пишутся явно, `f32` —
//! своими битами, а не текстом. Так одна и та же таблица, пришедшая из XLSX и
//! из TRNUM2, даёт один отпечаток.

use crate::data::{Normalizer, NumericDataset};
use crate::encoders::{FeatureSpec, ValueEncoderKind};
use crate::numeric_model::{ModelKind, NumericConfig, NumericModel};
use crate::schema::ModelSchema;

/// Доменный префикс: отделяет наш хеш от любого другого использования BLAKE3.
const DOMAIN: &[u8] = b"transformer/dataset-fingerprint";

/// Версия кодирования. Меняется вместе с составом или порядком полей: старый и
/// новый отпечаток одних и тех же данных обязаны различаться, иначе несовпадение
/// форматов выглядело бы как «другие данные».
const VERSION: u32 = 1;

/// Отпечаток данных — 32 байта BLAKE3 от канонической кодировки.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DatasetFingerprint([u8; 32]);

impl DatasetFingerprint {
    /// Посчитать отпечаток набора данных вместе с его схемой.
    ///
    /// Схема нужна не именами, а типами: категориальный вход с тремя уровнями
    /// и континуальный — разные задачи, даже если числа совпали.
    pub fn of(data: &NumericDataset, schema: &ModelSchema) -> Result<Self, String> {
        schema.check_dims(data.inputs.ncols(), data.outputs.ncols())?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN);
        hasher.update(&VERSION.to_le_bytes());
        hasher.update(&(data.len() as u64).to_le_bytes());
        hasher.update(&(data.inputs.ncols() as u64).to_le_bytes());
        hasher.update(&(data.outputs.ncols() as u64).to_le_bytes());

        // Типы входов: тег и кардинальность. Тег пишется всегда, поэтому
        // континуальный вход нельзя спутать с категориальным.
        for spec in schema.feature_specs() {
            match spec {
                FeatureSpec::Continuous => {
                    hasher.update(&[0u8]);
                    hasher.update(&0u64.to_le_bytes());
                }
                FeatureSpec::Categorical { cardinality } => {
                    hasher.update(&[1u8]);
                    hasher.update(&(cardinality as u64).to_le_bytes());
                }
            }
        }

        // Значения: сначала все входы, затем все выходы, оба — по строкам.
        // Порядок строк и колонок существенен: перестановка строк даёт другое
        // разбиение, а значит и другую задачу.
        for value in data.inputs.iter().chain(data.outputs.iter()) {
            hasher.update(&canonical_bits(*value).to_le_bytes());
        }

        Ok(Self(*hasher.finalize().as_bytes()))
    }

    /// Короткая подпись для интерфейса и отчётов.
    pub fn short(&self) -> String {
        self.0[..4].iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl std::fmt::Debug for DatasetFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DatasetFingerprint({})", self.short())
    }
}

/// Биты `f32` в каноническом виде.
///
/// У нуля два представления, а у NaN — множество; без нормализации одни и те же
/// данные давали бы разные отпечатки в зависимости от того, как они были
/// прочитаны.
fn canonical_bits(value: f32) -> u32 {
    if value.is_nan() {
        f32::NAN.to_bits()
    } else if value == 0.0 {
        0f32.to_bits()
    } else {
        value.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Зафиксированное значение отпечатка `fixed_model()`.
    const GOLDEN_MODEL_FINGERPRINT: [u8; 32] = [
        184, 44, 36, 129, 40, 247, 160, 167, 199, 24, 8, 179, 0, 37, 145, 218, 13, 109, 80, 52,
        254, 37, 21, 109, 109, 220, 50, 11, 225, 136, 167, 216,
    ];

    fn dataset(inputs: Vec<f32>, outputs: Vec<f32>, rows: usize, cols: usize) -> NumericDataset {
        NumericDataset::new(
            Array2::from_shape_vec((rows, cols), inputs).unwrap(),
            Array2::from_shape_vec((rows, outputs.len() / rows), outputs).unwrap(),
        )
    }

    fn schema(n_inputs: usize, n_outputs: usize) -> ModelSchema {
        ModelSchema::synthetic(n_inputs, n_outputs).unwrap()
    }

    /// Модель с заранее заданными весами: отпечаток обязан быть стабильным.
    fn fixed_model() -> (NumericModel, NumericConfig, Normalizer, Normalizer) {
        let config = NumericConfig {
            kind: crate::numeric_model::ModelKind::Mlp,
            transformer: crate::config::ModelConfig::default(),
            value: crate::encoders::ValueEncoderConfig::default(),
            mlp_width: 3,
            mlp_layers: 1,
            kan: crate::numeric_model::KanConfig::default(),
        };
        let specs = vec![FeatureSpec::Continuous, FeatureSpec::Continuous];
        let model = config.build(&specs, 1);
        // Веса заполняются детерминированно: golden-значение не должно
        // зависеть от инициализации.
        for (i, tensor) in model.parameters().iter().enumerate() {
            tensor.update_data(|data, _| {
                for (j, value) in data.iter_mut().enumerate() {
                    *value = (i * 100 + j) as f32 / 32.0;
                }
            });
        }
        let norm = |offset: f32, n: usize| Normalizer {
            mean: (0..n).map(|i| offset + i as f32).collect(),
            std: (0..n).map(|i| 1.0 + i as f32).collect(),
            min: (0..n).map(|i| -(i as f32)).collect(),
            max: (0..n).map(|i| 10.0 + i as f32).collect(),
            specs: vec![FeatureSpec::Continuous; n],
        };
        (model, config, norm(0.5, 2), norm(1.5, 1))
    }

    /// Golden-значение: отпечаток зависит от порядка обхода `parameters()`,
    /// который является частью формата checkpoint. Если этот тест упал, а
    /// изменение обхода было намеренным, версия отпечатка обязана вырасти —
    /// иначе старые отчёты молча перестанут сходиться со своими моделями.
    #[test]
    fn model_fingerprint_is_stable() {
        let (model, config, in_norm, out_norm) = fixed_model();
        let fp = ModelFingerprint::of(&model, &config, &in_norm, &out_norm);
        assert_eq!(
            fp.as_bytes(),
            &GOLDEN_MODEL_FINGERPRINT,
            "отпечаток изменился: {}",
            fp.short()
        );
    }

    /// Любое изменение весов, конфигурации или нормализаторов меняет отпечаток:
    /// он связывает отчёт с конкретной моделью, а не с похожей.
    #[test]
    fn every_part_of_the_model_changes_its_fingerprint() {
        let (model, config, in_norm, out_norm) = fixed_model();
        let base = ModelFingerprint::of(&model, &config, &in_norm, &out_norm);

        // Один бит веса.
        let (other, _, _, _) = fixed_model();
        other.parameters()[0].update_data(|data, _| {
            if let Some(first) = data.iter_mut().next() {
                *first += 1e-7;
            }
        });
        assert_ne!(
            base,
            ModelFingerprint::of(&other, &config, &in_norm, &out_norm)
        );

        // Конфигурация, не видимая в параметрах: ln_eps.
        let mut other_config = config.clone();
        other_config.transformer.ln_eps = 1e-4;
        assert_ne!(
            base,
            ModelFingerprint::of(&model, &other_config, &in_norm, &out_norm)
        );

        // Нормализатор: он меняет и предсказание, и предупреждения об
        // экстраполяции.
        let mut other_norm = in_norm.clone();
        other_norm.max[0] += 1.0;
        assert_ne!(
            base,
            ModelFingerprint::of(&model, &config, &other_norm, &out_norm)
        );

        // Та же модель, посчитанная заново, даёт тот же отпечаток.
        let (same, same_config, same_in, same_out) = fixed_model();
        assert_eq!(
            base,
            ModelFingerprint::of(&same, &same_config, &same_in, &same_out)
        );
    }

    #[test]
    fn the_same_numbers_give_the_same_fingerprint() {
        let a = dataset(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], 2, 2);
        let b = dataset(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], 2, 2);
        assert_eq!(
            DatasetFingerprint::of(&a, &schema(2, 1)).unwrap(),
            DatasetFingerprint::of(&b, &schema(2, 1)).unwrap()
        );
        // Нули с разным знаком — те же данные.
        let zero = dataset(vec![0.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], 2, 2);
        let neg_zero = dataset(vec![-0.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], 2, 2);
        assert_eq!(
            DatasetFingerprint::of(&zero, &schema(2, 1)).unwrap(),
            DatasetFingerprint::of(&neg_zero, &schema(2, 1)).unwrap()
        );
    }

    #[test]
    fn values_order_and_shape_all_matter() {
        let base = dataset(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], 2, 2);
        let base_fp = DatasetFingerprint::of(&base, &schema(2, 1)).unwrap();

        // Другое значение.
        let changed = dataset(vec![1.0, 2.0, 3.0, 4.5], vec![5.0, 6.0], 2, 2);
        assert_ne!(
            base_fp,
            DatasetFingerprint::of(&changed, &schema(2, 1)).unwrap()
        );

        // Переставленные строки: другое разбиение, другая задача.
        let reordered = dataset(vec![3.0, 4.0, 1.0, 2.0], vec![6.0, 5.0], 2, 2);
        assert_ne!(
            base_fp,
            DatasetFingerprint::of(&reordered, &schema(2, 1)).unwrap()
        );

        // Переставленные колонки.
        let swapped = dataset(vec![2.0, 1.0, 4.0, 3.0], vec![5.0, 6.0], 2, 2);
        assert_ne!(
            base_fp,
            DatasetFingerprint::of(&swapped, &schema(2, 1)).unwrap()
        );

        // Те же числа другой формы: 4 строки по одному входу.
        let reshaped = dataset(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], 4, 1);
        assert_ne!(
            base_fp,
            DatasetFingerprint::of(&reshaped, &schema(1, 1)).unwrap()
        );
    }

    /// Тип входа — часть задачи: те же числа как коды категорий и как
    /// континуальные значения означают разное.
    #[test]
    fn feature_types_are_part_of_the_fingerprint() {
        let data = dataset(vec![0.0, 1.0, 1.0, 2.0], vec![5.0, 6.0], 2, 2);
        let continuous = schema(2, 1);
        let categorical = ModelSchema::synthetic_from_specs(
            &[
                FeatureSpec::Continuous,
                FeatureSpec::Categorical { cardinality: 3 },
            ],
            1,
        )
        .unwrap();
        assert_ne!(
            DatasetFingerprint::of(&data, &continuous).unwrap(),
            DatasetFingerprint::of(&data, &categorical).unwrap()
        );
    }

    /// Схема, не совпадающая с данными по размерам, — ошибка, а не отпечаток
    /// «чего-то»: молчаливый хеш скрыл бы рассогласование.
    #[test]
    fn a_schema_that_does_not_match_the_data_is_an_error() {
        let data = dataset(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0], 2, 2);
        assert!(DatasetFingerprint::of(&data, &schema(3, 1)).is_err());
    }
}

/// Доменный префикс отпечатка модели. Отдельный от данных: это разные вещи, и
/// совпадение их хешей ничего не должно значить.
const MODEL_DOMAIN: &[u8] = b"transformer/model-fingerprint";

/// Версия кодирования отпечатка модели.
///
/// Своя, а не общая с checkpoint: изменение несвязанной секции файла не должно
/// менять отпечаток самой модели. Меняется вместе с составом, порядком полей
/// или порядком обхода параметров — см. golden-тест.
pub const MODEL_FINGERPRINT_VERSION: u32 = 1;

/// Отпечаток модели — 32 байта BLAKE3 от её конфигурации, весов и
/// нормализаторов.
///
/// Нужен, чтобы отчёт нельзя было переставить к другой модели с теми же
/// конфигурацией и схемой: без него происхождение утверждает о весах, которых
/// не видело.
///
/// Здесь, в отличие от [`DatasetFingerprint`], равенство ПОБИТОВОЕ: NaN и −0
/// не нормализуются. Связывается конкретный файл с конкретным утверждением, и
/// любое изменение бита обязано это ломать. Неконечные параметры отсекаются
/// проверкой отдельно — отпечаток их просто фиксирует.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelFingerprint([u8; 32]);

impl ModelFingerprint {
    /// Посчитать отпечаток активной модели.
    ///
    /// Конфигурация входит целиком: `parameters()` не содержит всего, что
    /// влияет на вычисление — `ln_eps`, частоты Fourier и настройки KAN живут
    /// в конфиге.
    pub fn of(
        model: &NumericModel,
        config: &NumericConfig,
        in_norm: &Normalizer,
        out_norm: &Normalizer,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(MODEL_DOMAIN);
        hasher.update(&MODEL_FINGERPRINT_VERSION.to_le_bytes());
        hash_config(&mut hasher, config);

        // Интерфейс модели: сколько входов и выходов она принимает.
        hasher.update(&(in_norm.n_features() as u64).to_le_bytes());
        hasher.update(&(out_norm.n_features() as u64).to_le_bytes());

        // Параметры: порядковый номер, форма, затем точные биты. Порядок
        // обхода — часть формата checkpoint (загрузка позиционная), поэтому и
        // часть отпечатка.
        let params = model.parameters();
        hasher.update(&(params.len() as u64).to_le_bytes());
        for (ordinal, tensor) in params.iter().enumerate() {
            hasher.update(&(ordinal as u64).to_le_bytes());
            hash_tensor(&mut hasher, tensor);
        }

        // Hard-prune маски и фактическая топология KAN: после сжатия ширина
        // слоёв не выводится из конфига.
        match model.kan_masks() {
            Some(masks) => {
                hasher.update(&(masks.len() as u64).to_le_bytes());
                for mask in &masks {
                    hash_tensor(&mut hasher, mask);
                }
            }
            None => {
                hasher.update(&0u64.to_le_bytes());
            }
        }
        match model.as_kan() {
            Some(kan) => {
                let dims = kan.layer_dims();
                hasher.update(&(dims.len() as u64).to_le_bytes());
                for (n_in, n_out) in dims {
                    hasher.update(&(n_in as u64).to_le_bytes());
                    hasher.update(&(n_out as u64).to_le_bytes());
                }
            }
            None => {
                hasher.update(&0u64.to_le_bytes());
            }
        }

        // Нормализаторы: они меняют и предсказание, и предупреждения об
        // экстраполяции, поэтому входят целиком.
        for norm in [in_norm, out_norm] {
            hash_norm(&mut hasher, norm);
        }

        Self(*hasher.finalize().as_bytes())
    }

    pub fn short(&self) -> String {
        self.0[..4].iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl std::fmt::Debug for ModelFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelFingerprint({})", self.short())
    }
}

fn hash_config(hasher: &mut blake3::Hasher, config: &NumericConfig) {
    let kind = match config.kind {
        ModelKind::Transformer => 0u32,
        ModelKind::Mlp => 1,
        ModelKind::Kan => 2,
    };
    hasher.update(&kind.to_le_bytes());
    let t = &config.transformer;
    for value in [
        t.d_model,
        t.n_heads,
        t.n_enc_layers,
        t.n_dec_layers,
        t.d_ff,
        config.mlp_width,
        config.mlp_layers,
        config.kan.width,
        config.kan.layers,
        config.kan.grid,
        config.value.fourier_bands,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    hasher.update(&t.ln_eps.to_bits().to_le_bytes());
    hasher.update(&config.value.fourier_scale.to_bits().to_le_bytes());
    let encoder = match config.value.kind {
        ValueEncoderKind::Linear => 0u32,
        ValueEncoderKind::Mlp => 1,
        ValueEncoderKind::Fourier => 2,
    };
    hasher.update(&encoder.to_le_bytes());
}

fn hash_tensor(hasher: &mut blake3::Hasher, tensor: &crate::tensor::Tensor) {
    let data = tensor.data();
    let shape = data.shape();
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for dim in shape {
        hasher.update(&(*dim as u64).to_le_bytes());
    }
    for value in data.iter() {
        // Точные биты: отпечаток связывает отчёт с конкретными весами, а не с
        // «численно такой же» моделью.
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_norm(hasher: &mut blake3::Hasher, norm: &Normalizer) {
    for stats in [&norm.mean, &norm.std, &norm.min, &norm.max] {
        hasher.update(&(stats.len() as u64).to_le_bytes());
        for value in stats {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }
    hasher.update(&(norm.specs.len() as u64).to_le_bytes());
    for spec in &norm.specs {
        match spec {
            FeatureSpec::Continuous => {
                hasher.update(&[0u8]);
                hasher.update(&0u64.to_le_bytes());
            }
            FeatureSpec::Categorical { cardinality } => {
                hasher.update(&[1u8]);
                hasher.update(&(*cardinality as u64).to_le_bytes());
            }
        }
    }
}
