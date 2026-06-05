//! Авторегрессионная генерация текста (Plan.md §8): greedy / temperature / top-k.

use crate::data::Vocab;
use crate::textmodel::TextModel;
use rand::rngs::StdRng;
use rand::Rng;

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

/// Сэмплирует индекс из логитов. `temperature <= 0` -> greedy (argmax).
/// `top_k > 0` ограничивает выбор k наиболее вероятными символами.
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut StdRng) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    if top_k > 0 && top_k < scaled.len() {
        let mut sorted = scaled.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[top_k - 1];
        for s in scaled.iter_mut() {
            if *s < threshold {
                *s = f32::NEG_INFINITY;
            }
        }
    }

    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut r = rng.gen::<f32>() * sum;
    for (i, &e) in exps.iter().enumerate() {
        r -= e;
        if r <= 0.0 {
            return i;
        }
    }
    exps.len() - 1
}

/// Генерирует `total_new` символов продолжая `seed`. Контекст скользит:
/// каждый rollout кодирует последние `ctx_len` символов и декодирует до
/// `tgt_len` новых, затем окно сдвигается вперёд.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &TextModel,
    vocab: &Vocab,
    seed: &str,
    total_new: usize,
    ctx_len: usize,
    tgt_len: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut StdRng,
) -> String {
    let mut full = vocab.encode(seed);
    assert!(full.len() >= ctx_len, "seed короче ctx_len");

    let mut out = String::new();
    let mut produced = 0;
    while produced < total_new {
        let src = full[full.len() - ctx_len..].to_vec();
        let memory = model.encode_src(&src);
        let mut dec = vec![*src.last().unwrap()];
        for _ in 0..tgt_len {
            let logits = model.next_logits(&dec, &memory);
            let next = sample(&logits, temperature, top_k, rng);
            dec.push(next);
            full.push(next);
            out.push_str(&vocab.decode(&[next]));
            produced += 1;
            if produced >= total_new {
                break;
            }
        }
    }
    out
}
