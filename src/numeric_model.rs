//! Единый интерфейс численных моделей: transformer, MLP или KAN.
//!
//! Один numeric-pipeline (данные/нормализация/обучение/метрики) работает с
//! `NumericModel` через общие `predict`/`loss`/`parameters`; архитектура
//! выбирается флагом `--model-kind`.

use crate::config::ModelConfig;
use crate::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use crate::kan::KanNet;
use crate::mlp::MlpBaseline;
use crate::surrogate::SurrogateModel;
use crate::tensor::Tensor;

/// Проверка осмысленности численного конфига (единый источник правды для CLI
/// и GUI). Падает понятной ошибкой вместо assert/panic в конструкторах.
pub fn validate_numeric(nc: &NumericConfig) -> Result<(), String> {
    let c = &nc.transformer;
    if c.d_model == 0 {
        return Err("d_model должен быть > 0".to_string());
    }
    if c.d_ff == 0 {
        return Err("d_ff должен быть > 0".to_string());
    }
    if c.n_enc_layers < 1 || c.n_dec_layers < 1 {
        return Err("число слоёв должно быть >= 1".to_string());
    }
    if c.n_heads == 0 || !c.d_model.is_multiple_of(c.n_heads) {
        return Err(format!(
            "d_model={} должен делиться на heads={}",
            c.d_model, c.n_heads
        ));
    }
    if matches!(nc.value.kind, ValueEncoderKind::Fourier) {
        if nc.value.fourier_bands < 1 {
            return Err("fourier_bands должен быть >= 1".to_string());
        }
        if !nc.value.fourier_scale.is_finite() || nc.value.fourier_scale <= 0.0 {
            return Err("fourier_scale должен быть конечным и > 0".to_string());
        }
    }
    if matches!(nc.kind, ModelKind::Mlp) {
        if nc.mlp_width == 0 {
            return Err("mlp_width должен быть > 0".to_string());
        }
        if nc.mlp_layers < 1 {
            return Err("mlp_layers должен быть >= 1".to_string());
        }
    }
    if matches!(nc.kind, ModelKind::Kan) {
        if nc.kan.width == 0 {
            return Err("kan_width должен быть > 0".to_string());
        }
        if nc.kan.layers < 1 {
            return Err("kan_layers должен быть >= 1".to_string());
        }
        if nc.kan.grid < 2 {
            return Err("kan_grid должен быть >= 2".to_string());
        }
    }
    if !matches!(nc.kind, ModelKind::Transformer)
        && !matches!(nc.value.kind, ValueEncoderKind::Linear)
    {
        return Err(
            "value-encoder применим только к transformer (MLP/KAN берут значения напрямую)"
                .to_string(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Transformer,
    Mlp,
    Kan,
}

/// Параметры KAN: ширина скрытых слоёв, число слоёв и размер сплайн-сетки
/// (число интервалов на [-3, 3]; базисных функций на ребро — grid + 3).
#[derive(Clone, Copy, Debug)]
pub struct KanConfig {
    pub width: usize,
    pub layers: usize,
    pub grid: usize,
}

impl Default for KanConfig {
    fn default() -> Self {
        Self {
            width: 16,
            layers: 2,
            grid: 8,
        }
    }
}

/// Конфиг численной модели: общий выбор архитектуры + параметры обеих.
#[derive(Clone)]
pub struct NumericConfig {
    pub kind: ModelKind,
    pub transformer: ModelConfig,
    pub value: ValueEncoderConfig,
    pub mlp_width: usize,
    pub mlp_layers: usize,
    pub kan: KanConfig,
}

impl NumericConfig {
    /// Построить свежую модель выбранного типа.
    pub fn build(&self, specs: &[FeatureSpec], n_outputs: usize) -> NumericModel {
        match self.kind {
            ModelKind::Transformer => {
                NumericModel::Transformer(Box::new(SurrogateModel::with_value_encoder(
                    &self.transformer,
                    &self.value,
                    specs,
                    n_outputs,
                )))
            }
            ModelKind::Mlp => NumericModel::Mlp(MlpBaseline::new(
                specs.len(),
                self.mlp_width,
                self.mlp_layers,
                n_outputs,
            )),
            ModelKind::Kan => NumericModel::Kan(KanNet::new(
                specs.len(),
                self.kan.width,
                self.kan.layers,
                self.kan.grid,
                n_outputs,
            )),
        }
    }
}

pub enum NumericModel {
    Transformer(Box<SurrogateModel>),
    Mlp(MlpBaseline),
    Kan(KanNet),
}

impl NumericModel {
    pub fn kind(&self) -> ModelKind {
        match self {
            NumericModel::Transformer(_) => ModelKind::Transformer,
            NumericModel::Mlp(_) => ModelKind::Mlp,
            NumericModel::Kan(_) => ModelKind::Kan,
        }
    }

    /// Размеры внешнего интерфейса модели независимо от её архитектуры.
    pub fn interface_dims(&self) -> (usize, usize) {
        match self {
            NumericModel::Transformer(m) => m.interface_dims(),
            NumericModel::Mlp(m) => m.interface_dims(),
            NumericModel::Kan(m) => {
                let dims = m.layer_dims();
                (dims[0].0, dims.last().unwrap().1)
            }
        }
    }

    pub(crate) fn predict(&self, values: &Tensor) -> Tensor {
        match self {
            NumericModel::Transformer(m) => m.predict(values),
            NumericModel::Mlp(m) => m.predict(values),
            NumericModel::Kan(m) => m.predict(values),
        }
    }

    pub(crate) fn loss(&self, values: &Tensor, targets: &Tensor) -> Tensor {
        match self {
            NumericModel::Transformer(m) => m.loss(values, targets),
            NumericModel::Mlp(m) => m.loss(values, targets),
            NumericModel::Kan(m) => m.loss(values, targets),
        }
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        match self {
            NumericModel::Transformer(m) => m.parameters(),
            NumericModel::Mlp(m) => m.parameters(),
            NumericModel::Kan(m) => m.parameters(),
        }
    }

    /// Число обучаемых скалярных параметров модели.
    pub fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.data().len()).sum()
    }

    /// Доступ к KAN-специфичному API (кривые функций рёбер) — `None` для
    /// остальных архитектур.
    pub fn as_kan(&self) -> Option<&KanNet> {
        match self {
            NumericModel::Kan(m) => Some(m),
            _ => None,
        }
    }

    /// Изменяемый доступ к KAN (структурное сжатие меняет топологию слоёв).
    pub fn as_kan_mut(&mut self) -> Option<&mut KanNet> {
        match self {
            NumericModel::Kan(m) => Some(m),
            _ => None,
        }
    }

    /// Непараметрическое состояние KAN (hard-prune masks) для checkpoint-а.
    pub(crate) fn kan_masks(&self) -> Option<Vec<Tensor>> {
        self.as_kan().map(KanNet::masks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kan_parameter_count_matches_formula() {
        let cfg = NumericConfig {
            kind: ModelKind::Kan,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 128,
            mlp_layers: 3,
            kan: KanConfig {
                width: 16,
                layers: 2,
                grid: 8,
            },
        };
        let model = cfg.build(&[FeatureSpec::Continuous; 3], 3);
        assert_eq!(model.parameter_count(), 1_171);
    }
}
