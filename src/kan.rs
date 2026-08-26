//! KAN (Kolmogorov–Arnold Network) для численной регрессии.
//!
//! В отличие от MLP (фиксированная нелинейность на узлах, обучаемые линейные
//! веса на рёбрах), KAN обучает САМИ функции на рёбрах: каждое ребро — сумма
//! кубических B-сплайнов на равномерной сетке плюс базовая ветка
//! `W_base·gelu(x)` (аналог residual в pykan, несёт градиент вне сетки).
//! Слой: `y = gelu(x)·W_base + b + B(x)·W_spline`, где `B(x)` — матрица
//! значений базисных сплайнов `[B, I*M]`, `M = grid + k` функций на вход.
//!
//! Входы ожидаются z-нормализованными, сетка фиксирована на [-3, 3].

use crate::init::rand_uniform;
use crate::nn::linear::Linear;
use crate::tensor::{BackwardFn, Tensor};
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use std::cell::Cell;

/// Степень B-сплайнов (кубические).
const SPLINE_ORDER: usize = 3;
/// Диапазон сетки: входы z-нормализованы, ±3σ покрывает почти все данные.
/// За границей базис плавно затухает на расширенных узлах и равен нулю лишь
/// за ±(3 + k·h); дальше сигнал несёт только базовая gelu-ветка.
const GRID_MIN: f32 = -3.0;
const GRID_MAX: f32 = 3.0;

/// Значения и производные `M = grid + k` базисных сплайнов степени k в точке x
/// (Кокс–де Бур на равномерных узлах, производная через базис степени k-1).
fn basis_and_deriv(x: f32, grid: usize) -> (Vec<f32>, Vec<f32>) {
    let k = SPLINE_ORDER;
    let h = (GRID_MAX - GRID_MIN) / grid as f32;
    let knot = |j: usize| GRID_MIN + (j as f32 - k as f32) * h;

    // Степень 0: индикаторы интервалов [t_j, t_{j+1}).
    let n0 = grid + 2 * k;
    let mut b: Vec<f32> = (0..n0)
        .map(|j| {
            if knot(j) <= x && x < knot(j + 1) {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // Поднимаем степень до k-1 (in-place: b[j] читает старые b[j], b[j+1]).
    for d in 1..k {
        let dh = d as f32 * h;
        for j in 0..(n0 - d) {
            b[j] = (x - knot(j)) / dh * b[j] + (knot(j + d + 1) - x) / dh * b[j + 1];
        }
    }

    // Финальная степень k и производная: B'_{j,k} = (B_{j,k-1} - B_{j+1,k-1})/h.
    let m = grid + k;
    let kh = k as f32 * h;
    let mut vals = vec![0.0; m];
    let mut derivs = vec![0.0; m];
    for j in 0..m {
        vals[j] = (x - knot(j)) / kh * b[j] + (knot(j + k + 1) - x) / kh * b[j + 1];
        derivs[j] = (b[j] - b[j + 1]) / h;
    }
    (vals, derivs)
}

/// Скалярный gelu — та же tanh-аппроксимация, что в `Tensor::gelu` (ops.rs);
/// нужен для поточечной выборки рёберных функций без построения графа.
fn gelu_scalar(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let k = 0.044_715_f32;
    0.5 * x * (1.0 + (c * (x + k * x.powi(3))).tanh())
}

/// B-сплайновый базис как операция autograd: `[B, I] -> [B, I*M]`.
/// Дифференцируема по входу (нужно для глубоких KAN: вход слоя 2 зависит от
/// параметров слоя 1), производная аналитическая.
fn bspline_basis(x: &Tensor, grid: usize) -> Tensor {
    let data = x.data();
    let shape = data.shape().to_vec();
    assert_eq!(shape.len(), 2, "bspline_basis ожидает вход [B, I]");
    let (n, f) = (shape[0], shape[1]);
    let m = grid + SPLINE_ORDER;

    let mut vals = Array2::<f32>::zeros((n, f * m));
    let mut dvals = Array2::<f32>::zeros((n, f * m));
    for i in 0..n {
        for j in 0..f {
            let (v, d) = basis_and_deriv(data[[i, j]], grid);
            for (t, (&vv, &dd)) in v.iter().zip(d.iter()).enumerate() {
                vals[[i, j * m + t]] = vv;
                dvals[[i, j * m + t]] = dd;
            }
        }
    }

    let lhs = x.clone();
    let dvals = dvals.into_dyn();
    let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
        // dL/dx[b,i] = сумма по базису: g[b, i*M+t] * B'_t(x[b,i]).
        let weighted = g * &dvals;
        let dx = weighted
            .into_shape_with_order(IxDyn(&[n, f, m]))
            .expect("формы согласованы по построению")
            .sum_axis(Axis(2));
        lhs.add_grad(&dx.into_dyn());
    });
    Tensor::from_op(vals.into_dyn(), vec![x.clone()], backward)
}

/// Один слой KAN: обучаемая функция на каждом ребре (вход, выход).
///
/// Маски рёбер (константы, не параметры) — механизм мягкого прунинга: они
/// умножаются на веса ВНУТРИ графа, поэтому у отсечённого ребра зануляется и
/// вклад в forward, и градиент — fine-tune не «отращивает» его обратно.
pub(crate) struct KanLayer {
    base: Linear,
    spline_weight: Tensor, // [I*M, O]
    mask: Tensor,          // [I, O], 1 = ребро активно
    spline_mask: Tensor,   // [I*M, O], маска, размноженная на строки базиса
    grid: usize,
}

impl KanLayer {
    pub(crate) fn new(in_features: usize, out_features: usize, grid: usize) -> Self {
        let m = grid + SPLINE_ORDER;
        // Сплайны стартуют малыми: начальная функция ребра ≈ базовая gelu-ветка.
        let limit = (6.0 / (in_features * m + out_features) as f32).sqrt();
        let spline_weight = Tensor::new(rand_uniform(
            &[in_features * m, out_features],
            -limit,
            limit,
        ));
        Self {
            base: Linear::new(in_features, out_features),
            spline_weight,
            mask: Tensor::constant(ArrayD::ones(IxDyn(&[in_features, out_features]))),
            spline_mask: Tensor::constant(ArrayD::ones(IxDyn(&[in_features * m, out_features]))),
            grid,
        }
    }

    fn dims(&self) -> (usize, usize) {
        let s = self.mask.shape();
        (s[0], s[1])
    }

    /// Веса с применёнными масками рёбер (узлы графа — градиент маскируется).
    fn masked_weights(&self) -> (Tensor, Tensor) {
        (
            self.base.weight.mul(&self.mask),
            self.spline_weight.mul(&self.spline_mask),
        )
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Tensor {
        let (wb, ws) = self.masked_weights();
        let base = x.gelu().matmul(&wb).add(&self.base.bias);
        let spline = bspline_basis(x, self.grid).matmul(&ws);
        base.add(&spline)
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.base.parameters();
        p.push(self.spline_weight.clone());
        p
    }

    /// Непараметрическое состояние hard-prune. Маски не входят в optimizer и
    /// число параметров, но должны жить в checkpoint, чтобы отсечённые рёбра
    /// не оживали после загрузки и fine-tune.
    fn masks(&self) -> Vec<Tensor> {
        vec![self.mask.clone(), self.spline_mask.clone()]
    }

    /// Слой с подмножеством ВЫХОДНЫХ узлов (структурное сжатие).
    fn retain_outputs(&self, kept: &[usize]) -> KanLayer {
        let ax1 = Axis(1);
        KanLayer {
            base: Linear::from_tensors(
                Tensor::new(self.base.weight.data().select(ax1, kept)),
                Tensor::new(self.base.bias.data().select(ax1, kept)),
            ),
            spline_weight: Tensor::new(self.spline_weight.data().select(ax1, kept)),
            mask: Tensor::constant(self.mask.data().select(ax1, kept)),
            spline_mask: Tensor::constant(self.spline_mask.data().select(ax1, kept)),
            grid: self.grid,
        }
    }

    /// Слой с подмножеством ВХОДНЫХ узлов: у base-весов и масок выбираются
    /// строки, у сплайновых — блоки строк по M базисных функций на вход.
    fn retain_inputs(&self, kept: &[usize]) -> KanLayer {
        let m = self.grid + SPLINE_ORDER;
        let spline_rows: Vec<usize> = kept.iter().flat_map(|&i| (i * m)..(i * m + m)).collect();
        let ax0 = Axis(0);
        KanLayer {
            base: Linear::from_tensors(
                Tensor::new(self.base.weight.data().select(ax0, kept)),
                Tensor::new(self.base.bias.data()),
            ),
            spline_weight: Tensor::new(self.spline_weight.data().select(ax0, &spline_rows)),
            mask: Tensor::constant(self.mask.data().select(ax0, kept)),
            spline_mask: Tensor::constant(self.spline_mask.data().select(ax0, &spline_rows)),
            grid: self.grid,
        }
    }

    /// Дифференцируемая L1-важность рёбер слоя: `Σ_ij mean_batch |φ_ij(x)|`,
    /// включая ОБЕ ветки (gelu и сплайн) — L1 только по `w_spline` оставил бы
    /// значимые gelu-рёбра безнаказанными. Ребро i вырезается константными
    /// селекторами, чтобы |φ| считался поэлементно до суммирования по входам.
    fn edge_l1(&self, x: &Tensor) -> Tensor {
        let (n_in, _) = self.dims();
        let m = self.grid + SPLINE_ORDER;
        let (wb, ws) = self.masked_weights();
        let g = x.gelu();
        let basis = bspline_basis(x, self.grid);
        let batch = x.shape()[0] as f32;

        let mut total: Option<Tensor> = None;
        for i in 0..n_in {
            let mut col = Array2::<f32>::zeros((n_in, 1));
            col[[i, 0]] = 1.0;
            let mut row = Array2::<f32>::zeros((1, n_in));
            row[[0, i]] = 1.0;
            let mut sel = Array2::<f32>::zeros((n_in * m, m));
            let mut sel_t = Array2::<f32>::zeros((m, n_in * m));
            for t in 0..m {
                sel[[i * m + t, t]] = 1.0;
                sel_t[[t, i * m + t]] = 1.0;
            }

            // φ_i = gelu(x_i)·w_base[i,:] + B(x_i)·w_spline[i*M.., :]  -> [B, O]
            let g_i = g.matmul(&Tensor::constant(col.into_dyn()));
            let w_row = Tensor::constant(row.into_dyn()).matmul(&wb);
            let basis_i = basis.matmul(&Tensor::constant(sel.into_dyn()));
            let w_blk = Tensor::constant(sel_t.into_dyn()).matmul(&ws);
            let phi_i = g_i.matmul(&w_row).add(&basis_i.matmul(&w_blk));

            let reg_i = phi_i.abs().sum().scale(1.0 / batch);
            total = Some(match total {
                Some(acc) => acc.add(&reg_i),
                None => reg_i,
            });
        }
        total.expect("KAN-слой имеет >= 1 входа")
    }

    /// p95 |φ_ij| по калибровочным строкам `x` `[N, I]` — метрика ДЛЯ УДАЛЕНИЯ
    /// (не для регуляризации): среднее занизило бы узкие, но важные сплайны.
    fn edge_importance(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n_in, n_out) = self.dims();
        let m = self.grid + SPLINE_ORDER;
        let wb = &self.base.weight.data() * &self.mask.data();
        let ws = &self.spline_weight.data() * &self.spline_mask.data();
        let n = x.nrows();

        let mut abs_phi = vec![vec![0.0_f32; n]; n_in * n_out];
        for (s, xrow) in x.rows().into_iter().enumerate() {
            for i in 0..n_in {
                let gx = gelu_scalar(xrow[i]);
                let (basis, _) = basis_and_deriv(xrow[i], self.grid);
                for o in 0..n_out {
                    let spline: f32 = basis
                        .iter()
                        .enumerate()
                        .map(|(t, b)| b * ws[[i * m + t, o]])
                        .sum();
                    abs_phi[i * n_out + o][s] = (gx * wb[[i, o]] + spline).abs();
                }
            }
        }

        let p95 = ((n - 1) as f32 * 0.95).round() as usize;
        let mut imp = Array2::<f32>::zeros((n_in, n_out));
        for i in 0..n_in {
            for o in 0..n_out {
                let vals = &mut abs_phi[i * n_out + o];
                vals.select_nth_unstable_by(p95, |a, b| a.total_cmp(b));
                imp[[i, o]] = vals[p95];
            }
        }
        imp
    }

    /// Зануляет ребро (i, o): маски и сами веса. Маски сохраняются отдельно
    /// от параметров, чтобы checkpoint сохранял и блокировку градиента.
    pub(crate) fn prune_edge(&self, i: usize, o: usize) {
        let m = self.grid + SPLINE_ORDER;
        self.mask.update_data(|d, _| d[[i, o]] = 0.0);
        self.base.weight.update_data(|d, _| d[[i, o]] = 0.0);
        self.spline_mask.update_data(|d, _| {
            for t in 0..m {
                d[[i * m + t, o]] = 0.0;
            }
        });
        self.spline_weight.update_data(|d, _| {
            for t in 0..m {
                d[[i * m + t, o]] = 0.0;
            }
        });
    }

    /// (активных, всего) рёбер слоя по маске.
    fn active_edges(&self) -> (usize, usize) {
        let mask = self.mask.data();
        let active = mask.iter().filter(|&&v| v > 0.5).count();
        (active, mask.len())
    }
}

/// Отчёт прунинга: (активных, всего) рёбер по слоям.
#[derive(Clone, Debug)]
pub struct PruneReport {
    pub per_layer: Vec<(usize, usize)>,
}

/// Отчёт структурного сжатия: скрытые узлы и параметры до/после.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactReport {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub params_before: usize,
    pub params_after: usize,
}

impl PruneReport {
    /// (активных, всего) суммарно по сети.
    pub fn totals(&self) -> (usize, usize) {
        self.per_layer
            .iter()
            .fold((0, 0), |(a, t), &(la, lt)| (a + la, t + lt))
    }
}

/// Сеть из KAN-слоёв. Между слоями нет активаций — нелинейность живёт
/// в самих рёбрах.
pub struct KanNet {
    pub(crate) layers: Vec<KanLayer>,
    /// Коэффициент activation-L1 в loss (0 = выключено; fine-tune после
    /// прунинга идёт с λ=0). `Cell` — чтобы менять фазу без `&mut` через
    /// общий `NumericModel`-интерфейс.
    l1_lambda: Cell<f32>,
}

impl KanNet {
    /// `n_layers` — число KAN-слоёв: 1 -> [in->out], 2 -> [in->width->out],
    /// 3 -> [in->width->width->out] и т.д.
    pub fn new(
        n_inputs: usize,
        width: usize,
        n_layers: usize,
        grid: usize,
        n_outputs: usize,
    ) -> Self {
        assert!(n_layers >= 1, "KAN: нужен хотя бы один слой");
        let mut dims = vec![n_inputs];
        dims.extend(std::iter::repeat_n(width, n_layers - 1));
        dims.push(n_outputs);
        let layers = dims
            .windows(2)
            .map(|w| KanLayer::new(w[0], w[1], grid))
            .collect();
        Self {
            layers,
            l1_lambda: Cell::new(0.0),
        }
    }

    /// Сеть с явными размерами слоёв — для загрузки структурно сжатых
    /// checkpoint-ов, где ширина скрытых слоёв неоднородна.
    pub fn from_dims(dims: &[(usize, usize)], grid: usize) -> Self {
        assert!(!dims.is_empty(), "KAN: нужен хотя бы один слой");
        for w in dims.windows(2) {
            assert_eq!(
                w[0].1, w[1].0,
                "KAN: выход слоя должен совпадать со входом следующего"
            );
        }
        Self {
            layers: dims
                .iter()
                .map(|&(i, o)| KanLayer::new(i, o, grid))
                .collect(),
            l1_lambda: Cell::new(0.0),
        }
    }

    /// Задать коэффициент activation-L1 (обучение с разрежением: λ > 0;
    /// fine-tune после прунинга: λ = 0).
    pub fn set_l1_lambda(&self, lambda: f32) {
        self.l1_lambda.set(lambda);
    }

    /// `values` — `[B, F]` (нормализованные) -> `[B, O]`.
    pub(crate) fn predict(&self, values: &Tensor) -> Tensor {
        let mut x = values.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    pub(crate) fn loss(&self, values: &Tensor, targets: &Tensor) -> Tensor {
        let lambda = self.l1_lambda.get();
        if lambda == 0.0 {
            return self.predict(values).mse_loss(targets);
        }
        self.predict(values)
            .mse_loss(targets)
            .add(&self.l1(values).scale(lambda))
    }

    /// Дифференцируемый L1-терм всей сети: `Σ по слоям Σ_ij mean_batch |φ_ij|`
    /// на активациях данного батча (для слоя k — выход слоя k-1).
    pub(crate) fn l1(&self, values: &Tensor) -> Tensor {
        let mut x = values.clone();
        let mut total: Option<Tensor> = None;
        for layer in &self.layers {
            let reg = layer.edge_l1(&x);
            total = Some(match total {
                Some(acc) => acc.add(&reg),
                None => reg,
            });
            x = layer.forward(&x);
        }
        total.expect("KAN имеет >= 1 слоя")
    }

    /// Важность рёбер (p95 |φ|) по слоям на калибровочных строках `[N, F]`
    /// (нормализованных): активации прогоняются через сеть послойно.
    pub fn edge_importance(&self, calibration: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut x = calibration.clone();
        let mut result = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            result.push(layer.edge_importance(&x));
            x = layer
                .forward(&Tensor::constant(x.into_dyn()))
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("KAN forward возвращает [N, O]");
        }
        result
    }

    /// Hard-prune: зануляет рёбра с важностью ниже `rel_threshold · max`
    /// важности своего слоя (p95 |φ| на `calibration`). Маски замораживают
    /// отсечённые рёбра на время fine-tune. Число параметров не меняется —
    /// структурное сжатие топологии отдельный шаг.
    pub fn prune_edges(&self, rel_threshold: f32, calibration: &Array2<f32>) -> PruneReport {
        let importances = self.edge_importance(calibration);
        for (layer, imp) in self.layers.iter().zip(&importances) {
            let max = imp.iter().fold(0.0_f32, |a, &b| a.max(b));
            let cutoff = rel_threshold * max;
            let (n_in, n_out) = layer.dims();
            for i in 0..n_in {
                for o in 0..n_out {
                    if imp[[i, o]] < cutoff {
                        layer.prune_edge(i, o);
                    }
                }
            }
        }
        PruneReport {
            per_layer: self.layers.iter().map(|l| l.active_edges()).collect(),
        }
    }

    /// Структурное сжатие: физически удаляет мёртвые скрытые узлы, реально
    /// уменьшая число параметров. Узел удаляется, если у него нет активных
    /// исходящих рёбер (выход не используется) ИЛИ нет входящих — тогда его
    /// выход константен (= bias) и вклад каждого исходящего ребра phi(bias)
    /// сворачивается в bias следующего слоя. Функция сети не меняется.
    /// Итерирует до неподвижной точки: удаление узла может омертвить соседей.
    /// Входы сети и выходы последнего слоя не трогаются (интерфейс фиксирован).
    pub fn compact(&mut self) -> CompactReport {
        let hidden_total =
            |layers: &[KanLayer]| -> usize { layers.iter().skip(1).map(|l| l.dims().0).sum() };
        let params_total = |layers: &[KanLayer]| -> usize {
            layers
                .iter()
                .flat_map(|l| l.parameters())
                .map(|p| p.data().len())
                .sum()
        };
        let nodes_before = hidden_total(&self.layers);
        let params_before = params_total(&self.layers);

        loop {
            let mut changed = false;
            for k in 0..self.layers.len() - 1 {
                let (in_k, n_nodes) = self.layers[k].dims();
                let (_, out_next) = self.layers[k + 1].dims();
                let mask_in = self.layers[k].mask.data();
                let mask_out = self.layers[k + 1].mask.data();
                let bias_k = self.layers[k].base.bias.data();

                let mut kept: Vec<usize> = Vec::with_capacity(n_nodes);
                for j in 0..n_nodes {
                    let incoming = (0..in_k).any(|i| mask_in[[i, j]] > 0.5);
                    let outgoing = (0..out_next).any(|o| mask_out[[j, o]] > 0.5);
                    if incoming && outgoing {
                        kept.push(j);
                        continue;
                    }
                    if !incoming && outgoing {
                        // Константный узел: выход = bias, вклад phi(bias)
                        // каждого активного исходящего ребра уходит в bias
                        // следующего слоя — удаление точное.
                        let b = bias_k[[0, j]];
                        for o in 0..out_next {
                            if mask_out[[j, o]] > 0.5 {
                                let phi = self.edge_curve(k + 1, j, o, &[b])[0];
                                self.layers[k + 1]
                                    .base
                                    .bias
                                    .update_data(|d, _| d[[0, o]] += phi);
                            }
                        }
                    }
                }
                if kept.len() == n_nodes {
                    continue;
                }
                if kept.is_empty() {
                    // Все узлы границы мертвы (вырожденная сеть): оставляем
                    // один с нулевыми масками, чтобы формы остались валидными.
                    kept.push(0);
                    let cleared_live_edges = (0..in_k).any(|i| mask_in[[i, 0]] > 0.5)
                        || (0..out_next).any(|o| mask_out[[0, o]] > 0.5);
                    for i in 0..in_k {
                        self.layers[k].prune_edge(i, 0);
                    }
                    for o in 0..out_next {
                        self.layers[k + 1].prune_edge(0, o);
                    }
                    if n_nodes == 1 {
                        // Форма уже минимальная, а обнуление масок идемпотентно:
                        // новая итерация нужна только когда реально очистились
                        // рёбра — они могли омертвить предыдущую границу.
                        changed |= cleared_live_edges;
                        continue;
                    }
                }
                let narrowed = self.layers[k].retain_outputs(&kept);
                self.layers[k] = narrowed;
                let narrowed_next = self.layers[k + 1].retain_inputs(&kept);
                self.layers[k + 1] = narrowed_next;
                changed = true;
            }
            if !changed {
                break;
            }
        }

        CompactReport {
            nodes_before,
            nodes_after: hidden_total(&self.layers),
            params_before,
            params_after: params_total(&self.layers),
        }
    }

    /// (активных, всего) рёбер сети по маскам.
    pub fn active_edges(&self) -> (usize, usize) {
        self.layers.iter().fold((0, 0), |(a, t), l| {
            let (la, lt) = l.active_edges();
            (a + la, t + lt)
        })
    }

    pub(crate) fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// Маски hard-prune для checkpoint-а; это состояние, а не обучаемые
    /// параметры, поэтому оно намеренно вынесено из `parameters()`.
    pub(crate) fn masks(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.masks()).collect()
    }

    /// Размеры слоёв `[(in, out); n_layers]` — для перебора рёбер в UI.
    pub fn layer_dims(&self) -> Vec<(usize, usize)> {
        self.layers
            .iter()
            .map(|l| {
                let s = l.spline_weight.shape();
                let m = l.grid + SPLINE_ORDER;
                (s[0] / m, s[1])
            })
            .collect()
    }

    /// Базовый домен рёберных функций — диапазон основной сплайн-сетки.
    /// На расширенных узлах базис продолжается ещё на `k·h` с каждой стороны;
    /// за их пределами остаётся только gelu-ветка.
    pub fn domain(&self) -> (f32, f32) {
        (GRID_MIN, GRID_MAX)
    }

    /// Узловые bias слоя `[O]` — нужны для послойной сборки формул:
    /// выход узла = bias + Σ по входам phi(x_i).
    pub fn layer_bias(&self, layer: usize) -> Vec<f32> {
        let b = self.layers[layer].base.bias.data();
        b.iter().copied().collect()
    }

    /// Активно ли ребро (не отсечено hard-prune).
    pub fn edge_active(&self, layer: usize, input: usize, output: usize) -> bool {
        self.layers[layer].mask.data()[[input, output]] > 0.5
    }

    /// Входные активации каждого слоя на данных `[N, F]`: `[0]` — сами входы,
    /// `[k]` — выход слоя k-1. Для symbolic-фита рёбра слоя k нужны именно
    /// его реальные входы, а не равномерная сетка.
    pub fn activations(&self, inputs: &Array2<f32>) -> Vec<Array2<f32>> {
        let mut acts = vec![inputs.clone()];
        for layer in &self.layers {
            let last = acts.last().expect("acts непуст");
            let next = layer
                .forward(&Tensor::constant(last.clone().into_dyn()))
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("KAN forward возвращает [N, O]");
            acts.push(next);
        }
        acts
    }

    /// Обученная функция ребра `(layer, input, output)` в точках `xs`:
    /// `phi(x) = w_base·gelu(x) + Σ_t B_t(x)·w_spline[t]`. Узловой bias сюда
    /// не входит — он смещение выхода, а не свойство ребра. Это API
    /// интерпретируемости: выход узла = bias + Σ по входам phi(x_i).
    pub fn edge_curve(&self, layer: usize, input: usize, output: usize, xs: &[f32]) -> Vec<f32> {
        let l = &self.layers[layer];
        let m = l.grid + SPLINE_ORDER;
        let w_base = l.base.weight.data();
        let w_spline = l.spline_weight.data();
        xs.iter()
            .map(|&x| {
                let (basis, _) = basis_and_deriv(x, l.grid);
                let spline: f32 = basis
                    .iter()
                    .enumerate()
                    .map(|(t, b)| b * w_spline[[input * m + t, output]])
                    .sum();
                gelu_scalar(x) * w_base[[input, output]] + spline
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::data::Normalizer;
    use crate::gradcheck::{grad_check, rand_tensor};
    use crate::optim::Adam;

    /// Базис — разбиение единицы внутри сетки: сумма значений = 1.
    #[test]
    fn basis_partition_of_unity() {
        for &x in &[-2.9, -1.0, 0.0, 0.5, 2.99] {
            let (vals, _) = basis_and_deriv(x, 8);
            let s: f32 = vals.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "sum B(x={x}) = {s}, ожидалось 1");
        }
        // За расширяющими узлами (GRID_MAX + k*h) базис нулевой — сигнал
        // несёт базовая ветка.
        let (vals, _) = basis_and_deriv(10.0, 8);
        assert!(vals.iter().all(|&v| v == 0.0));
    }

    /// Аналитическая производная базиса совпадает с конечной разностью
    /// (сквозной градиент по входу — критично для глубоких KAN).
    #[test]
    fn check_bspline_basis_grad() {
        let x = rand_tensor(&[4, 3]);
        grad_check(std::slice::from_ref(&x), |_| bspline_basis(&x, 5).mean());
    }

    /// Градиенты полного слоя (вход + оба вида весов).
    #[test]
    fn check_kan_layer_grad() {
        let layer = KanLayer::new(3, 2, 5);
        let x = rand_tensor(&[4, 3]);
        let mut inputs = vec![x.clone()];
        inputs.extend(layer.parameters());
        grad_check(&inputs, |_| layer.forward(&x).mean());
    }

    /// Интерпретируемость: выход слоя точно раскладывается в
    /// bias + Σ по входам edge_curve(x_i) — кривые рёбер ЯВЛЯЮТСЯ моделью.
    #[test]
    fn edge_curves_decompose_forward() {
        let net = KanNet::new(3, 5, 2, 6, 2);
        assert_eq!(net.layer_dims(), vec![(3, 5), (5, 2)]);

        for (layer, (n_inputs, n_outputs)) in net.layer_dims().into_iter().enumerate() {
            // Ненулевой bias обязателен для проверки, что он не попал в phi
            // каждого ребра и прибавляется к выходу узла ровно один раз.
            net.layers[layer].base.bias.set_data(
                ndarray::Array2::from_shape_fn((1, n_outputs), |(_, output)| {
                    0.2 + output as f32 * 0.1
                })
                .into_dyn(),
            );
            let x: Vec<f32> = (0..n_inputs).map(|i| -0.8 + i as f32 * 0.35).collect();
            let xt = Tensor::constant(
                ndarray::Array2::from_shape_vec((1, n_inputs), x.clone())
                    .unwrap()
                    .into_dyn(),
            );
            let out = net.layers[layer].forward(&xt).data();
            let bias = net.layers[layer].base.bias.data();
            for output in 0..n_outputs {
                let edges: f32 = (0..n_inputs)
                    .map(|input| net.edge_curve(layer, input, output, &[x[input]])[0])
                    .sum();
                let expect = bias[[0, output]] + edges;
                assert!(
                    (out[[0, output]] - expect).abs() < 1e-5,
                    "слой {layer}, выход {output}: forward={} != bias+edges={expect}",
                    out[[0, output]]
                );
            }
        }
    }

    /// L1-терм согласован с edge_curve: Σ_ij mean_b |φ_ij| по обеим веткам.
    #[test]
    fn l1_matches_manual_edge_curves() {
        let net = KanNet::new(3, 4, 2, 5, 2);
        let rows = 8;
        let data: Vec<f32> = (0..rows * 3)
            .map(|k| (k as f32 * 0.37).sin() * 2.0)
            .collect();
        let x0 = Array2::from_shape_vec((rows, 3), data).unwrap();

        let mut manual = 0.0_f32;
        let mut x = x0.clone();
        for (layer, &(n_in, n_out)) in net.layer_dims().iter().enumerate() {
            for i in 0..n_in {
                let xs: Vec<f32> = x.column(i).to_vec();
                for o in 0..n_out {
                    let phis = net.edge_curve(layer, i, o, &xs);
                    manual += phis.iter().map(|p| p.abs()).sum::<f32>() / rows as f32;
                }
            }
            x = net.layers[layer]
                .forward(&Tensor::constant(x.into_dyn()))
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
        }

        let l1 = net.l1(&Tensor::constant(x0.into_dyn())).item();
        assert!(
            (l1 - manual).abs() < 1e-4 * manual.max(1.0),
            "l1={l1}, вручную={manual}"
        );
    }

    /// Прунинг: мёртвое ребро отсекается, живые остаются, предсказания
    /// не меняются; после прунинга градиент ребра заморожен маской.
    #[test]
    fn prune_removes_dead_edge_and_freezes_it() {
        let net = KanNet::new(2, 0, 1, 5, 2); // один слой 2 -> 2
        let layer = &net.layers[0];
        // Убиваем ребро (0, 0) вручную: обе ветки в ноль.
        let m = 5 + SPLINE_ORDER;
        layer.base.weight.update_data(|d, _| d[[0, 0]] = 0.0);
        layer.spline_weight.update_data(|d, _| {
            for t in 0..m {
                d[[t, 0]] = 0.0;
            }
        });

        let calib_data: Vec<f32> = (0..32).map(|k| (k as f32 * 0.19).sin() * 2.0).collect();
        let calib = Array2::from_shape_vec((16, 2), calib_data).unwrap();
        let x = Tensor::constant(calib.clone().into_dyn());
        let before = net.predict(&x).data();

        // Порог ~0: отсекаются только рёбра с нулевой важностью.
        let report = net.prune_edges(1e-6, &calib);
        assert_eq!(report.totals(), (3, 4), "должно уйти ровно мёртвое ребро");
        assert_eq!(net.active_edges(), (3, 4));
        let after = net.predict(&x).data();
        assert_eq!(before, after, "прунинг мёртвого ребра не меняет выход");

        // Fine-tune (λ=0) не отращивает отсечённое ребро обратно.
        let targets = Tensor::constant(ArrayD::ones(IxDyn(&[16, 2])));
        let mut opt = Adam::new(net.parameters(), 1e-2);
        for _ in 0..20 {
            opt.zero_grad();
            let loss = net.loss(&x, &targets);
            loss.backward();
            opt.step();
        }
        let wb = layer.base.weight.data();
        let ws = layer.spline_weight.data();
        assert_eq!(wb[[0, 0]], 0.0, "gelu-ветка ребра осталась нулевой");
        assert!(
            (0..m).all(|t| ws[[t, 0]] == 0.0),
            "сплайн-ветка ребра осталась нулевой"
        );
        // Живые рёбра при этом обучаются.
        assert!(wb[[1, 0]] != 0.0 || wb[[1, 1]] != 0.0);
    }

    /// Структурное сжатие: мёртвый узел (нет исходящих) удаляется, узел без
    /// входящих сворачивается в bias следующего слоя; предсказания сети
    /// не меняются, параметров становится меньше.
    #[test]
    fn compact_removes_dead_nodes_exactly() {
        crate::init::set_init_seed(11);
        let mut net = KanNet::new(2, 4, 2, 5, 1);
        // Узел 0: без исходящих (его единственное ребро 0->0 слоя 1 отсечено).
        net.layers[1].prune_edge(0, 0);
        // Узел 1: без входящих (оба ребра слоя 0 в него отсечены), но с
        // активным исходящим — его вклад должен уйти в bias выхода.
        net.layers[0].prune_edge(0, 1);
        net.layers[0].prune_edge(1, 1);

        let x_data: Vec<f32> = (0..24).map(|k| (k as f32 * 0.29).sin() * 2.0).collect();
        let x = Tensor::constant(Array2::from_shape_vec((12, 2), x_data).unwrap().into_dyn());
        let before = net.predict(&x).data();
        let params_before: usize = net.parameters().iter().map(|p| p.data().len()).sum();

        let report = net.compact();
        assert_eq!(report.nodes_before, 4);
        assert_eq!(report.nodes_after, 2, "узлы 0 и 1 должны уйти");
        assert!(report.params_after < params_before);
        assert_eq!(net.layer_dims(), vec![(2, 2), (2, 1)]);

        let after = net.predict(&x).data();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < 1e-5,
                "сжатие изменило предсказание: {b} vs {a}"
            );
        }
    }

    /// Аналог mlp_overfits_sum: KAN должен выучить простой ящик.
    #[test]
    fn kan_overfits_sum() {
        let data = blackbox::sum().generate(64, 0);
        let in_norm = Normalizer::fit(&data.inputs, &Normalizer::all_continuous(2));
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let x = Tensor::constant(in_norm.transform(&data.inputs).into_dyn());
        let y = Tensor::constant(out_norm.transform(&data.outputs).into_dyn());

        let model = KanNet::new(2, 8, 2, 5, 1);
        let mut opt = Adam::new(model.parameters(), 3e-3);
        let first = model.loss(&x, &y).item();
        for _ in 0..200 {
            opt.zero_grad();
            let loss = model.loss(&x, &y);
            loss.backward();
            opt.step();
        }
        let last = model.loss(&x, &y).item();
        assert!(
            last < first * 0.1,
            "KAN не выучил sum: {first:.4} -> {last:.4}"
        );
    }

    /// Вырожденная сеть: все рёбра слоя отсечены. Раньше `compact` крутился на
    /// ней вечно — обнуление масок он считал изменением, хотя форма уже была
    /// минимальной.
    #[test]
    fn compact_terminates_on_a_fully_pruned_network() {
        crate::init::set_init_seed(0);
        let mut net = KanNet::new(2, 4, 1, 5, 2);
        for layer in net.layers.iter_mut() {
            let (n_in, n_out) = layer.dims();
            for i in 0..n_in {
                for o in 0..n_out {
                    layer.prune_edge(i, o);
                }
            }
        }
        let report = net.compact();
        assert!(report.nodes_after <= report.nodes_before);
        // Сеть осталась работоспособной по форме.
        assert_eq!(net.active_edges().0, 0);
        assert!(!net.layer_dims().is_empty());
    }

    /// Очистка рёбер единственного узла должна запустить ещё одну итерацию:
    /// иначе мёртвые узлы на уже пройденной предыдущей границе останутся.
    #[test]
    fn compact_reaches_fixed_point_through_a_single_node_boundary() {
        crate::init::set_init_seed(0);
        let mut net = KanNet::from_dims(&[(2, 2), (2, 1), (1, 1)], 5);
        net.layers[2].prune_edge(0, 0);

        let report = net.compact();

        assert_eq!(net.layer_dims(), vec![(2, 1), (1, 1), (1, 1)]);
        assert_eq!(report.nodes_before, 3);
        assert_eq!(report.nodes_after, 2);
        assert_eq!(net.active_edges().0, 0);
    }
}
