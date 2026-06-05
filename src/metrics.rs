//! Метрики регрессии (Plan.md §5). Считаются в денормализованных единицах.
//! Относительная ошибка — основная для расчётов; MSE недостаточно.

use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct Metrics {
    pub rmse: f32,
    pub mae: f32,
    /// Средняя относительная ошибка `|pred - target| / (|target| + eps)`.
    pub rel_error: f32,
    /// Коэффициент детерминации R² (доля объяснённой дисперсии).
    pub r2: f32,
}

pub fn evaluate(pred: &Array2<f32>, target: &Array2<f32>) -> Metrics {
    assert_eq!(
        pred.dim(),
        target.dim(),
        "формы pred и target должны совпадать"
    );
    let n = pred.len() as f32;
    assert!(n > 0.0, "пустые данные для метрик");

    let mut se = 0.0;
    let mut ae = 0.0;
    let mut rel = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let d = p - t;
        se += d * d;
        ae += d.abs();
        rel += d.abs() / (t.abs() + 1e-8);
    }

    let mean = target.iter().sum::<f32>() / n;
    let ss_tot: f32 = target.iter().map(|t| (t - mean) * (t - mean)).sum();
    let ss_res = se;
    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Metrics {
        rmse: (se / n).sqrt(),
        mae: ae / n,
        rel_error: rel / n,
        r2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn perfect_prediction() {
        let y = array![[1.0], [2.0], [3.0]];
        let m = evaluate(&y, &y);
        assert!(m.rmse < 1e-6);
        assert!(m.rel_error < 1e-6);
        assert!((m.r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn known_error() {
        let pred = array![[2.0], [2.0]];
        let target = array![[1.0], [3.0]];
        let m = evaluate(&pred, &target);
        assert!((m.rmse - 1.0).abs() < 1e-6); // обе ошибки по 1
        assert!((m.mae - 1.0).abs() < 1e-6);
    }
}
