//! Воспроизводимая инициализация весов (PlanUI обсуждение §2).
//!
//! Случайность при построении модели — только в `Linear`/`Embedding`. Они
//! рисуют веса через `rand_uniform`, который берёт сидируемый thread-local RNG,
//! если задан `set_init_seed`, иначе — энтропию (`thread_rng`). Это делает
//! `search` (per-seed) и `train --seed`/validation-кривые воспроизводимыми, не
//! протягивая RNG через все конструкторы.

use ndarray::{Array, ArrayD, IxDyn};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;

thread_local! {
    static INIT_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Зафиксировать seed инициализации весов для текущего потока: после вызова
/// все `Linear`/`Embedding` инициализируются детерминированно (в порядке их
/// построения).
pub fn set_init_seed(seed: u64) {
    INIT_RNG.with(|c| *c.borrow_mut() = Some(StdRng::seed_from_u64(seed)));
}

/// Вернуться к недетерминированной инициализации (из энтропии).
#[cfg(test)]
pub(crate) fn clear_init_seed() {
    INIT_RNG.with(|c| *c.borrow_mut() = None);
}

/// Равномерная инициализация в `[low, high]`: из сидированного RNG, если задан,
/// иначе из `thread_rng`.
pub(crate) fn rand_uniform(shape: &[usize], low: f32, high: f32) -> ArrayD<f32> {
    let dist = Uniform::new(low, high);
    INIT_RNG.with(|c| match c.borrow_mut().as_mut() {
        Some(rng) => Array::random_using(IxDyn(shape), dist, rng),
        None => Array::random(IxDyn(shape), dist),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::Linear;

    #[test]
    fn seeded_init_is_reproducible() {
        set_init_seed(123);
        let a = Linear::new(4, 8).weight.data();
        set_init_seed(123);
        let b = Linear::new(4, 8).weight.data();
        assert_eq!(a, b, "один seed -> одинаковые веса");

        set_init_seed(999);
        let c = Linear::new(4, 8).weight.data();
        assert_ne!(a, c, "другой seed -> другие веса");

        clear_init_seed();
    }
}
