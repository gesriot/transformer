//! Обучение и оценка surrogate-модели.
//!
//! Модель учится в НОРМАЛИЗОВАННОМ пространстве; метрики считаются в исходных
//! единицах (предсказания денормализуются перед сравнением).

use crate::data::{Normalizer, NumericDataset, TextDataset};
use crate::encoders::FeatureSpec;
use crate::metrics::{evaluate, Metrics};
use crate::numeric_model::NumericModel;
use crate::optim::Adam;
use crate::tensor::Tensor;
use crate::textmodel::TextModel;
use ndarray::{Array2, ArrayD, Ix2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};

/// Проверка осмысленности параметров обучения (единый источник правды для CLI
/// и GUI). lr должен быть конечным и > 0, batch >= 1.
pub fn validate_train(lr: f32, batch_size: usize) -> Result<(), String> {
    if !lr.is_finite() || lr <= 0.0 {
        return Err("lr должен быть конечным и > 0".to_string());
    }
    if batch_size == 0 {
        return Err("batch_size должен быть >= 1".to_string());
    }
    Ok(())
}

/// Политика learning rate по шагам. Живёт здесь, а не в Adam: оптимизатор
/// хранит только моменты, расписание решает, какой lr дать на шаге.
#[derive(Clone, Copy, Debug)]
pub enum LrSchedule {
    Constant,
    /// Линейный warmup до `base_lr`, затем косинусный спад до
    /// `base_lr * min_lr_ratio`. `warmup_frac` — доля всех шагов под warmup.
    WarmupCosine {
        warmup_frac: f32,
        min_lr_ratio: f32,
    },
}

impl LrSchedule {
    pub fn lr_at(&self, base_lr: f32, step: usize, total_steps: usize) -> f32 {
        match *self {
            LrSchedule::Constant => base_lr,
            LrSchedule::WarmupCosine {
                warmup_frac,
                min_lr_ratio,
            } => {
                let warmup = ((warmup_frac * total_steps as f32).round() as usize).min(total_steps);
                if step < warmup {
                    base_lr * (step + 1) as f32 / warmup.max(1) as f32
                } else {
                    let denom = (total_steps - warmup).max(1) as f32;
                    let progress = ((step - warmup) as f32 / denom).clamp(0.0, 1.0);
                    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                    base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub seed: u64,
    pub schedule: LrSchedule,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 30,
            batch_size: 64,
            lr: 1e-3,
            seed: 0,
            schedule: LrSchedule::Constant,
        }
    }
}

fn to_tensor(a: &Array2<f32>) -> Tensor {
    Tensor::constant(a.clone().into_dyn())
}

/// Единая точка создания нормализаторов. В поиске сюда передаётся только
/// train конкретного fold; при финальном refit — весь train+validation pool.
/// Test не участвует в `fit` никогда.
pub fn fit_normalizers(
    train: &NumericDataset,
    in_specs: &[FeatureSpec],
) -> (Normalizer, Normalizer) {
    let in_norm = Normalizer::fit(&train.inputs, in_specs);
    let out_norm = Normalizer::fit(
        &train.outputs,
        &Normalizer::all_continuous(train.outputs.ncols()),
    );
    (in_norm, out_norm)
}

/// Обучает модель, возвращает средний loss по эпохам (для диагностики).
pub fn train_surrogate(
    model: &NumericModel,
    data: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    cfg: &TrainConfig,
) -> Vec<f32> {
    let never = AtomicBool::new(false);
    train_surrogate_cb(
        model,
        data,
        in_norm,
        out_norm,
        cfg,
        &mut |_, _| true,
        &never,
    )
}

/// Как `train_surrogate`, но зовёт `on_epoch(epoch_index_0based, mean_loss)` после
/// каждой эпохи (живая кривая GUI / снапшоты epoch-sweep) и проверяет `cancel`
/// ВНУТРИ батч-цикла — при взводе флага обучение прерывается на ближайшем
/// minibatch (а не ждёт конца эпохи) и возвращает накопленную историю.
///
/// `on_epoch` возвращает `false`, чтобы остановить обучение после этой эпохи:
/// так работает ранняя остановка по validation. Отмена пользователем и ранняя
/// остановка — разные вещи и разными путями и приходят: первая может сработать
/// посреди эпохи, вторая осмысленна только на её границе.
pub fn train_surrogate_cb(
    model: &NumericModel,
    data: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    cfg: &TrainConfig,
    on_epoch: &mut dyn FnMut(usize, f32) -> bool,
    cancel: &AtomicBool,
) -> Vec<f32> {
    let mut opt = Adam::new(model.parameters(), cfg.lr);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let n = data.len();
    let batch = cfg.batch_size.max(1);
    let batches_per_epoch = n.div_ceil(batch);
    let total_steps = (cfg.epochs * batches_per_epoch).max(1);

    let mut history = Vec::with_capacity(cfg.epochs);
    let mut step = 0;
    for epoch in 0..cfg.epochs {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut n_batches = 0;
        for chunk in idx.chunks(batch) {
            if cancel.load(Ordering::Relaxed) {
                return history; // кооперативная отмена между minibatch-ами
            }
            let batch = data.gather(chunk);
            let x = to_tensor(&in_norm.transform(&batch.inputs));
            let y = to_tensor(&out_norm.transform(&batch.outputs));

            opt.zero_grad();
            let loss = model.loss(&x, &y);
            epoch_loss += loss.item();
            n_batches += 1;
            loss.backward();
            opt.step_with_lr(cfg.schedule.lr_at(cfg.lr, step, total_steps));
            step += 1;
        }
        let mean = epoch_loss / n_batches as f32;
        history.push(mean);
        if !on_epoch(epoch, mean) {
            return history;
        }
    }
    history
}

/// Предсказания модели на датасете в ИСХОДНЫХ единицах (денормализованные).
pub fn predict_dataset(
    model: &NumericModel,
    data: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
) -> Array2<f32> {
    let x = to_tensor(&in_norm.transform(&data.inputs));
    let pred_norm = model
        .predict(&x)
        .data()
        .into_dimensionality::<Ix2>()
        .expect("predict возвращает [N, O]");
    out_norm.inverse_transform(&pred_norm)
}

/// Оценивает модель на тесте: предсказывает, денормализует и сравнивает с
/// сырыми таргетами.
pub fn evaluate_surrogate(
    model: &NumericModel,
    data: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
) -> Metrics {
    let pred = predict_dataset(model, data, in_norm, out_norm);
    evaluate(&pred, &data.outputs)
}

#[derive(Clone)]
pub struct TextTrainConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub ctx_len: usize,
    pub tgt_len: usize,
    pub lr: f32,
    pub seed: u64,
}

/// Строит вход декодера (teacher forcing) и метки из окна контекст/продолжение.
/// Вход декодера = [последний символ контекста] + продолжение[..-1];
/// метки = продолжение. Так позиция `t` декодера предсказывает `labels[t]`.
fn build_decoder_io(src: &Array2<usize>, tgt: &Array2<usize>) -> (ArrayD<usize>, ArrayD<usize>) {
    let (b, tgt_len) = tgt.dim();
    let ctx_len = src.ncols();
    let mut dec_in = Array2::<usize>::zeros((b, tgt_len));
    for i in 0..b {
        dec_in[[i, 0]] = src[[i, ctx_len - 1]];
        for t in 1..tgt_len {
            dec_in[[i, t]] = tgt[[i, t - 1]];
        }
    }
    (dec_in.into_dyn(), tgt.clone().into_dyn())
}

/// Обучает char-LM на случайных окнах. Возвращает усреднённый loss по
/// контрольным точкам (для построения кривой / perplexity).
pub fn train_text(model: &TextModel, dataset: &TextDataset, cfg: &TextTrainConfig) -> Vec<f32> {
    let never = AtomicBool::new(false);
    train_text_cb(model, dataset, cfg, &mut |_, _| {}, &never)
}

pub fn train_text_cb(
    model: &TextModel,
    dataset: &TextDataset,
    cfg: &TextTrainConfig,
    on_report: &mut dyn FnMut(usize, f32),
    cancel: &AtomicBool,
) -> Vec<f32> {
    let mut opt = Adam::new(model.parameters(), cfg.lr);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let report_every = (cfg.steps / 20).max(1);

    let mut history = Vec::new();
    let mut running = 0.0;
    let mut count = 0;
    for step in 0..cfg.steps {
        if cancel.load(Ordering::Relaxed) {
            return history;
        }
        let (src, tgt) = dataset.sample_batch(cfg.batch_size, cfg.ctx_len, cfg.tgt_len, &mut rng);
        let (dec_in, labels) = build_decoder_io(&src, &tgt);

        opt.zero_grad();
        let loss = model.loss(&src.into_dyn(), &dec_in, &labels);
        running += loss.item();
        count += 1;
        loss.backward();
        opt.step();

        if (step + 1) % report_every == 0 {
            let mean = running / count as f32;
            history.push(mean);
            on_report(step + 1, mean);
            running = 0.0;
            count = 0;
        }
    }
    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::config::ModelConfig;
    use crate::encoders::FeatureSpec;
    use crate::numeric_model::NumericModel;
    use crate::surrogate::SurrogateModel;

    /// Лёгкий smoke-тест: на простом ящике `sum` средний loss за эпоху должен
    /// заметно упасть. Полное обучение с метриками — через CLI (main.rs).
    #[test]
    fn smoke_train_decreases_loss() {
        let bb = blackbox::sum();
        let data = bb.generate(128, 0);
        let in_specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
        let in_norm = Normalizer::fit(&data.inputs, &in_specs);
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(bb.n_outputs));

        let cfg = ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_enc_layers: 1,
            n_dec_layers: 1,
            d_ff: 32,
            ln_eps: 1e-5,
        };
        let model =
            NumericModel::Transformer(Box::new(SurrogateModel::new(&cfg, &in_specs, bb.n_outputs)));
        let tcfg = TrainConfig {
            epochs: 10,
            batch_size: 32,
            lr: 3e-3,
            seed: 0,
            schedule: LrSchedule::Constant,
        };
        let history = train_surrogate(&model, &data, &in_norm, &out_norm, &tcfg);
        let first = history.first().unwrap();
        let last = history.last().unwrap();
        assert!(
            last < &(first * 0.5),
            "loss не упал достаточно: {first:.4} -> {last:.4}"
        );
    }

    #[test]
    fn warmup_cosine_shape() {
        let s = LrSchedule::WarmupCosine {
            warmup_frac: 0.2,
            min_lr_ratio: 0.1,
        };
        let (base, total) = (1.0, 100);
        assert!(s.lr_at(base, 0, total) < base); // старт ниже base
        assert!((s.lr_at(base, 19, total) - base).abs() < 1e-6); // конец warmup ≈ base
        assert!(s.lr_at(base, 50, total) < base); // косинусный спад
        assert!((s.lr_at(base, total - 1, total) - base * 0.1).abs() < 0.05); // к концу ≈ base*min
        assert_eq!(LrSchedule::Constant.lr_at(base, 7, total), base); // constant игнорит step
    }

    /// Smoke-тест char-LM: loss заметно падает на маленьком тексте.
    #[test]
    fn smoke_text_decreases_loss() {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(8);
        let ds = TextDataset::new(&text);
        let cfg = ModelConfig {
            d_model: 16,
            n_heads: 2,
            n_enc_layers: 1,
            n_dec_layers: 1,
            d_ff: 32,
            ln_eps: 1e-5,
        };
        let model = TextModel::new(&cfg, ds.vocab.len());
        let tcfg = TextTrainConfig {
            steps: 60,
            batch_size: 16,
            ctx_len: 8,
            tgt_len: 8,
            lr: 3e-3,
            seed: 0,
        };
        let history = train_text(&model, &ds, &tcfg);
        assert!(
            history.last().unwrap() < &(history.first().unwrap() * 0.7),
            "char-LM loss не упал: {:.3} -> {:.3}",
            history.first().unwrap(),
            history.last().unwrap()
        );
    }
}
