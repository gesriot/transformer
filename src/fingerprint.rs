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

use crate::data::NumericDataset;
use crate::encoders::FeatureSpec;
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

    fn dataset(inputs: Vec<f32>, outputs: Vec<f32>, rows: usize, cols: usize) -> NumericDataset {
        NumericDataset::new(
            Array2::from_shape_vec((rows, cols), inputs).unwrap(),
            Array2::from_shape_vec((rows, outputs.len() / rows), outputs).unwrap(),
        )
    }

    fn schema(n_inputs: usize, n_outputs: usize) -> ModelSchema {
        ModelSchema::synthetic(n_inputs, n_outputs).unwrap()
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
