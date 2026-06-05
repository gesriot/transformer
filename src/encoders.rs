//! Преобразование входа X в токены `[B, T, d_model]` для ядра (Plan.md §1, §3).
//!
//! - `NumericInputEncoder` — для surrogate-модели: каждый числовой параметр
//!   расчёта становится токеном `feature_emb[id] + value_contrib`, где вклад
//!   значения — `value_proj(value)` для континуальных признаков и
//!   `category_emb[code]` для категориальных (Plan.md §3).
//! - `TokenInputEncoder` — для текста: embedding + sinusoidal positions.

use crate::nn::embedding::{sinusoidal_positions, Embedding};
use crate::nn::linear::Linear;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Тип признака. Категориальный (например `material_id`) нельзя кодировать
/// числовой проекцией — это навязало бы ложную геометрию (код 7 «ближе» к 8,
/// чем к 20). Для него используется отдельный embedding по коду категории.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSpec {
    Continuous,
    Categorical { cardinality: usize },
}

/// Как кодировать скаляр значения в вектор `d_model` (roadmap шаг 4).
/// `Linear` навязывает низкочастотный (спектральный) bias; `Mlp` добавляет
/// нелинейность; `Fourier` инъектирует высокочастотный базис напрямую —
/// помогает осцилляторным/резким зависимостям от признака.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueEncoderKind {
    Linear,
    Mlp,
    Fourier,
}

#[derive(Clone, Copy, Debug)]
pub struct ValueEncoderConfig {
    pub kind: ValueEncoderKind,
    pub fourier_bands: usize,
    pub fourier_scale: f32,
}

impl Default for ValueEncoderConfig {
    fn default() -> Self {
        Self {
            kind: ValueEncoderKind::Linear,
            fourier_bands: 6,
            fourier_scale: 8.0,
        }
    }
}

/// Кодировщик скаляра `[B, F, 1] -> [B, F, d_model]`.
enum ValueEncoder {
    Linear(Linear),
    Mlp {
        up: Linear,
        down: Linear,
    },
    Fourier {
        freqs: Tensor,
        lin_raw: Linear,
        lin_sin: Linear,
        lin_cos: Linear,
    },
}

impl ValueEncoder {
    fn new(d_model: usize, cfg: &ValueEncoderConfig) -> Self {
        match cfg.kind {
            ValueEncoderKind::Linear => ValueEncoder::Linear(Linear::new(1, d_model)),
            ValueEncoderKind::Mlp => ValueEncoder::Mlp {
                up: Linear::new(1, d_model),
                down: Linear::new(d_model, d_model),
            },
            ValueEncoderKind::Fourier => {
                let bands = cfg.fourier_bands;
                // Инварианты (CLI валидирует флаги до этого места).
                assert!(bands >= 1, "fourier_bands должен быть >= 1");
                assert!(cfg.fourier_scale > 0.0, "fourier_scale должна быть > 0");
                // Геометрические частоты от 2π до 2π·scale (на z-score входе).
                let freqs: Vec<f32> = (0..bands)
                    .map(|i| {
                        let t = if bands > 1 {
                            i as f32 / (bands - 1) as f32
                        } else {
                            0.0
                        };
                        2.0 * std::f32::consts::PI * cfg.fourier_scale.powf(t)
                    })
                    .collect();
                ValueEncoder::Fourier {
                    freqs: Tensor::constant(
                        ArrayD::from_shape_vec(IxDyn(&[1, 1, bands]), freqs).unwrap(),
                    ),
                    // Сырой канал: Fourier строго богаче linear (сохраняет тренд).
                    lin_raw: Linear::new(1, d_model),
                    lin_sin: Linear::new(bands, d_model),
                    lin_cos: Linear::new(bands, d_model),
                }
            }
        }
    }

    fn forward(&self, scalar: &Tensor) -> Tensor {
        match self {
            ValueEncoder::Linear(l) => l.forward(scalar),
            ValueEncoder::Mlp { up, down } => down.forward(&up.forward(scalar).gelu()),
            ValueEncoder::Fourier {
                freqs,
                lin_raw,
                lin_sin,
                lin_cos,
            } => {
                let arg = scalar.mul(freqs); // [B, F, bands]
                lin_raw
                    .forward(scalar)
                    .add(&lin_sin.forward(&arg.sin()))
                    .add(&lin_cos.forward(&arg.cos()))
            }
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        match self {
            ValueEncoder::Linear(l) => l.parameters(),
            ValueEncoder::Mlp { up, down } => {
                let mut p = up.parameters();
                p.extend(down.parameters());
                p
            }
            ValueEncoder::Fourier {
                lin_raw,
                lin_sin,
                lin_cos,
                ..
            } => {
                let mut p = lin_raw.parameters();
                p.extend(lin_sin.parameters());
                p.extend(lin_cos.parameters());
                p
            }
        }
    }
}

/// Кодировщик числовых признаков. Вход — `[B, F]` значений, где столбец `j` это
/// один и тот же признак для всех примеров (слот `j`). Континуальные значения
/// должны быть нормализованы; категориальные хранят целочисленный код категории.
pub struct NumericInputEncoder {
    pub feature_emb: Embedding,
    value_enc: ValueEncoder,
    /// Общая таблица для всех категориальных признаков: строка 0 — паддинг
    /// (для континуальных слотов, потом маскируется), далее блоки по признакам.
    pub category_emb: Embedding,
    specs: Vec<FeatureSpec>,
    /// Начальная строка каждого признака в `category_emb` (0 для континуальных).
    cat_offsets: Vec<usize>,
    has_categorical: bool,
    /// Константные маски `[1, F, 1]`: какие слоты континуальные / категориальные.
    cont_mask: Tensor,
    cat_mask: Tensor,
    num_features: usize,
}

impl NumericInputEncoder {
    /// Все признаки континуальные.
    pub fn new(num_features: usize, d_model: usize) -> Self {
        Self::with_specs(
            &vec![FeatureSpec::Continuous; num_features],
            d_model,
            &ValueEncoderConfig::default(),
        )
    }

    pub fn with_specs(
        specs: &[FeatureSpec],
        d_model: usize,
        value_cfg: &ValueEncoderConfig,
    ) -> Self {
        let f = specs.len();
        assert!(f > 0, "нужен хотя бы один признак");

        // Оффсеты в общей category-таблице; строка 0 зарезервирована под паддинг.
        let mut cat_offsets = vec![0usize; f];
        let mut total_categories = 1;
        let mut cont = vec![0.0f32; f];
        let mut cat = vec![0.0f32; f];
        for (j, spec) in specs.iter().enumerate() {
            match *spec {
                FeatureSpec::Continuous => cont[j] = 1.0,
                FeatureSpec::Categorical { cardinality } => {
                    assert!(cardinality > 0, "cardinality должна быть > 0");
                    cat_offsets[j] = total_categories;
                    total_categories += cardinality;
                    cat[j] = 1.0;
                }
            }
        }
        let has_categorical = total_categories > 1;

        let mask =
            |v: Vec<f32>| Tensor::constant(ArrayD::from_shape_vec(IxDyn(&[1, f, 1]), v).unwrap());
        Self {
            feature_emb: Embedding::new(f, d_model),
            value_enc: ValueEncoder::new(d_model, value_cfg),
            category_emb: Embedding::new(total_categories, d_model),
            specs: specs.to_vec(),
            cat_offsets,
            has_categorical,
            cont_mask: mask(cont),
            cat_mask: mask(cat),
            num_features: f,
        }
    }

    /// `values` — `[B, F]`. Возвращает токены `[B, F, d_model]`.
    pub fn forward(&self, values: &Tensor) -> Tensor {
        let shape = values.shape();
        assert_eq!(shape.len(), 2, "NumericInputEncoder ожидает values [B, F]");
        let (batch, f) = (shape[0], shape[1]);
        assert_eq!(f, self.num_features, "число признаков != num_features");

        let slot_ids = self.slot_ids(batch, f);
        let feat = self.feature_emb.forward(&slot_ids); // [B, F, d_model]
        let val = self.value_enc.forward(&values.reshape(&[batch, f, 1])); // [B, F, d_model]

        // Чисто континуальный случай — путь без масок (как раньше).
        if !self.has_categorical {
            return feat.add(&val);
        }

        let cat_ids = self.category_ids(&values.data(), batch, f);
        let cat = self.category_emb.forward(&cat_ids); // [B, F, d_model]
        feat.add(&val.mul(&self.cont_mask))
            .add(&cat.mul(&self.cat_mask))
    }

    /// Идентификаторы слотов: каждая строка = 0..F.
    fn slot_ids(&self, batch: usize, f: usize) -> ArrayD<usize> {
        let mut ids = Vec::with_capacity(batch * f);
        for _ in 0..batch {
            ids.extend(0..f);
        }
        ArrayD::from_shape_vec(IxDyn(&[batch, f]), ids).unwrap()
    }

    /// Индексы в `category_emb`: для категориальных — `offset + код`, для
    /// континуальных — 0 (паддинг, вклад маскируется в `forward`).
    fn category_ids(&self, values: &ArrayD<f32>, batch: usize, f: usize) -> ArrayD<usize> {
        let mut ids = vec![0usize; batch * f];
        for b in 0..batch {
            for j in 0..f {
                if let FeatureSpec::Categorical { cardinality } = self.specs[j] {
                    let raw = values[IxDyn(&[b, j])];
                    let code = raw.round();
                    // Категория должна приходить целым кодом. Дробное значение —
                    // признак ошибки (например случайной нормализации колонки).
                    assert!(
                        (raw - code).abs() < 1e-4,
                        "категориальный признак {j} получил нецелый код {raw} \
                         (возможно колонка ошибочно нормализована)"
                    );
                    assert!(
                        code >= 0.0 && (code as usize) < cardinality,
                        "категория {code} вне диапазона [0, {cardinality}) для признака {j}"
                    );
                    ids[b * f + j] = self.cat_offsets[j] + code as usize;
                }
            }
        }
        ArrayD::from_shape_vec(IxDyn(&[batch, f]), ids).unwrap()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.feature_emb.parameters();
        p.extend(self.value_enc.parameters());
        if self.has_categorical {
            p.extend(self.category_emb.parameters());
        }
        p
    }

    /// Параметры кодировщика значений (для проверок, что путь значения учится).
    pub fn value_parameters(&self) -> Vec<Tensor> {
        self.value_enc.parameters()
    }
}

/// Кодировщик токенов текста: embedding + синусоидальные позиции.
pub struct TokenInputEncoder {
    pub emb: Embedding,
    d_model: usize,
}

impl TokenInputEncoder {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            emb: Embedding::new(vocab_size, d_model),
            d_model,
        }
    }

    /// `ids` — `[B, T]`. Возвращает `[B, T, d_model]` с позиционным кодированием.
    pub fn forward(&self, ids: &ArrayD<usize>) -> Tensor {
        let seq_len = *ids
            .shape()
            .last()
            .expect("ids должен иметь ось последовательности");
        let x = self.emb.forward(ids);
        let pos = sinusoidal_positions(seq_len, self.d_model);
        x.add(&pos)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.emb.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::{grad_check, rand_tensor};

    #[test]
    fn numeric_encoder_shape_and_grad() {
        let enc = NumericInputEncoder::new(3, 8);
        let values = rand_tensor(&[2, 3]);
        let out = enc.forward(&values);
        assert_eq!(out.shape(), vec![2, 3, 8]);

        // Градиент должен течь к value_proj через значения.
        let mut inputs = vec![values.clone()];
        inputs.extend(enc.parameters());
        grad_check(&inputs, |t| enc.forward(&t[0]).mean());
    }

    #[test]
    fn value_encoders_grad() {
        for kind in [ValueEncoderKind::Mlp, ValueEncoderKind::Fourier] {
            let cfg = ValueEncoderConfig {
                kind,
                fourier_bands: 4,
                fourier_scale: 6.0,
            };
            let enc = NumericInputEncoder::with_specs(
                &[FeatureSpec::Continuous, FeatureSpec::Continuous],
                8,
                &cfg,
            );
            let values = rand_tensor(&[2, 2]);
            assert_eq!(enc.forward(&values).shape(), vec![2, 2, 8]);

            let mut inputs = vec![values.clone()];
            inputs.extend(enc.parameters());
            grad_check(&inputs, |t| enc.forward(&t[0]).mean());
        }
    }

    #[test]
    fn token_encoder_adds_positions() {
        let enc = TokenInputEncoder::new(10, 6);
        let ids = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1, 2, 3, 4]).unwrap();
        let out = enc.forward(&ids);
        assert_eq!(out.shape(), vec![1, 4, 6]);
    }

    #[test]
    fn mixed_features_grad() {
        // Признак 0 континуальный, признак 1 категориальный (cardinality=4).
        let enc = NumericInputEncoder::with_specs(
            &[
                FeatureSpec::Continuous,
                FeatureSpec::Categorical { cardinality: 4 },
            ],
            8,
            &ValueEncoderConfig::default(),
        );
        // Значения констант: категориальный столбец хранит коды (2.0, 0.0).
        let values = Tensor::constant(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.3, 2.0, -0.5, 0.0]).unwrap(),
        );
        assert_eq!(enc.forward(&values).shape(), vec![2, 2, 8]);

        // Градиент-чек ТОЛЬКО по параметрам: вход категории недифференцируем
        // (округляется в индекс), поэтому values оставляем константой.
        grad_check(&enc.parameters(), |_| enc.forward(&values).mean());
    }

    #[test]
    #[should_panic(expected = "нецелый код")]
    fn categorical_rejects_fractional_code() {
        let enc = NumericInputEncoder::with_specs(
            &[FeatureSpec::Categorical { cardinality: 3 }],
            4,
            &ValueEncoderConfig::default(),
        );
        // 1.4 — не целый код: должно упасть, а не округлиться до 1.
        enc.forward(&Tensor::constant(
            ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.4]).unwrap(),
        ));
    }

    #[test]
    fn categorical_distinguishes_codes() {
        // Один категориальный признак: разные коды -> разные токены, т.к. слот
        // один и тот же, а различает только category_emb.
        let enc = NumericInputEncoder::with_specs(
            &[FeatureSpec::Categorical { cardinality: 3 }],
            4,
            &ValueEncoderConfig::default(),
        );
        let out0 = enc
            .forward(&Tensor::constant(
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap(),
            ))
            .data();
        let out2 = enc
            .forward(&Tensor::constant(
                ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![2.0]).unwrap(),
            ))
            .data();
        assert_ne!(out0, out2, "разные категории должны давать разные токены");
    }
}
