//! Symbolic extraction из обученной KAN: мост «сплайны -> формулы».
//!
//! Каждой рёберной функции phi(x) подбирается примитив в аффинной форме
//! `a·f(b·x + c) + d`: сетчатый поиск по (b, c) с зумом, a и d — замкнутой
//! формулой МНК. Фит идёт по РЕАЛЬНЫМ активациям слоя (не по равномерной
//! сетке): рёбра глубоких слоёв живут на распределении выходов предыдущего.
//!
//! Примитивы с ограниченным доменом (log, sqrt, 1/x, exp) защищены: кандидат
//! (b, c), выводящий хоть одну точку выборки за домен, пропускается — иначе
//! NaN в МНК ложно отверг бы весь примитив.
//!
//! Формулы собираются ПОСЛОЙНО с явными bias: промежуточные узлы получают
//! имена `h{k}_{j}`, подстановки слоёв друг в друга нет (взрыв выражений).
//! `symbolize` даёт формулы в нормализованном пространстве модели;
//! `SymbolicKan::denormalize` сворачивает z-score в аффинные коэффициенты и
//! выдаёт формулы в исходных единицах данных — точно, без повторного фита.

use crate::data::Normalizer;
use crate::kan::KanNet;
use crate::schema::ModelSchema;
use ndarray::Array2;

/// Примитив f с доменной защитой: `valid(u)` — точка u = b·x + c допустима
/// при подгонке; `f` тотально-безопасен (клампит u) для eval после подгонки.
struct Primitive {
    name: &'static str,
    f: fn(f32) -> f32,
    valid: fn(f32) -> bool,
}

fn any(_: f32) -> bool {
    true
}

/// Порядок — от простого к сложному: при равном R² выигрывает более простой
/// примитив (сравнение строгое `>`).
const PRIMITIVES: &[Primitive] = &[
    Primitive {
        name: "x",
        f: |u| u,
        valid: any,
    },
    Primitive {
        name: "x^2",
        f: |u| u * u,
        valid: any,
    },
    Primitive {
        name: "x^3",
        f: |u| u * u * u,
        valid: any,
    },
    Primitive {
        name: "sqrt",
        f: |u| u.max(0.0).sqrt(),
        valid: |u| u >= 0.0,
    },
    Primitive {
        name: "exp",
        f: |u| u.min(15.0).exp(),
        valid: |u| u < 15.0,
    },
    Primitive {
        name: "log",
        f: |u| u.max(1e-6).ln(),
        valid: |u| u > 1e-4,
    },
    Primitive {
        name: "1/x",
        // `signum(0.0)` is zero, so it cannot serve as a safe clamp here:
        // without this branch an out-of-calibration `u == 0` would make the
        // symbolic model return NaN despite the documented total-safe eval.
        f: |u| {
            let denom = if u.abs() < 1e-3 {
                if u.is_sign_negative() {
                    -1e-3
                } else {
                    1e-3
                }
            } else {
                u
            };
            1.0 / denom
        },
        valid: |u| u.abs() > 1e-3,
    },
    Primitive {
        name: "sin",
        f: f32::sin,
        valid: any,
    },
    Primitive {
        name: "tanh",
        f: f32::tanh,
        valid: any,
    },
];

const CONST_PRIM: Primitive = Primitive {
    name: "const",
    f: |_| 0.0,
    valid: any,
};

/// Подогнанное ребро: `phi(x) ≈ a·f(b·x + c) + d`.
#[derive(Clone, Debug)]
pub struct EdgeFit {
    pub layer: usize,
    pub input: usize,
    pub output: usize,
    pub name: &'static str,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub r2: f32,
    f: fn(f32) -> f32,
}

impl EdgeFit {
    pub fn eval(&self, x: f32) -> f32 {
        self.a * (self.f)(self.b * x + self.c) + self.d
    }
}

/// МНК по a, d при фиксированных z_i = f(b·x_i + c): точное решение.
/// Возвращает (a, d, r2); `None` если z вырожден (константа).
fn solve_affine(zs: &[f32], ys: &[f32], sst: f64, y_mean: f64) -> Option<(f32, f32, f32)> {
    let n = zs.len() as f64;
    let z_mean = zs.iter().map(|&z| z as f64).sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut var = 0.0_f64;
    for (&z, &y) in zs.iter().zip(ys) {
        let dz = z as f64 - z_mean;
        cov += dz * (y as f64 - y_mean);
        var += dz * dz;
    }
    if var < 1e-9 {
        return None;
    }
    let a = cov / var;
    let d = y_mean - a * z_mean;
    let mut sse = 0.0_f64;
    for (&z, &y) in zs.iter().zip(ys) {
        let e = y as f64 - (a * z as f64 + d);
        sse += e * e;
    }
    let r2 = 1.0 - sse / sst.max(1e-12);
    Some((a as f32, d as f32, r2 as f32))
}

/// Подгонка одного примитива: сетка по (b, c) с тремя зумами, a/d — МНК.
/// `None`, если ни один кандидат не прошёл доменную защиту.
fn fit_primitive(xs: &[f32], ys: &[f32], prim: &Primitive) -> Option<(f32, f32, f32, f32, f32)> {
    let n = xs.len() as f64;
    let y_mean = ys.iter().map(|&y| y as f64).sum::<f64>() / n;
    let sst: f64 = ys.iter().map(|&y| (y as f64 - y_mean).powi(2)).sum();

    const STEPS: usize = 13;
    const ZOOMS: usize = 3;
    let (mut b_lo, mut b_hi) = (-3.0_f32, 3.0_f32);
    let (mut c_lo, mut c_hi) = (-4.0_f32, 4.0_f32);
    let mut best: Option<(f32, f32, f32, f32, f32)> = None; // (a,b,c,d,r2)

    let mut zs = vec![0.0_f32; xs.len()];
    for _ in 0..ZOOMS {
        let b_step = (b_hi - b_lo) / (STEPS - 1) as f32;
        let c_step = (c_hi - c_lo) / (STEPS - 1) as f32;
        for bi in 0..STEPS {
            let b = b_lo + b_step * bi as f32;
            'cand: for ci in 0..STEPS {
                let c = c_lo + c_step * ci as f32;
                for (z, &x) in zs.iter_mut().zip(xs) {
                    let u = b * x + c;
                    if !(prim.valid)(u) {
                        continue 'cand; // доменная защита: кандидат вне домена
                    }
                    *z = (prim.f)(u);
                }
                if let Some((a, d, r2)) = solve_affine(&zs, ys, sst, y_mean) {
                    if r2.is_finite() && best.is_none_or(|(.., br2)| r2 > br2) {
                        best = Some((a, b, c, d, r2));
                    }
                }
            }
        }
        let Some((_, bb, bc, _, _)) = best else {
            return None; // домен не достижим в диапазоне поиска
        };
        b_lo = bb - b_step;
        b_hi = bb + b_step;
        c_lo = bc - c_step;
        c_hi = bc + c_step;
    }
    best
}

/// Лучший примитив для ребра. Константное ребро (нулевая дисперсия phi)
/// получает fit "const" с точным R² = 1.
fn fit_edge(layer: usize, input: usize, output: usize, xs: &[f32], ys: &[f32]) -> EdgeFit {
    let n = ys.len() as f64;
    let y_mean = (ys.iter().map(|&y| y as f64).sum::<f64>() / n) as f32;
    let sst: f64 = ys.iter().map(|&y| (y as f64 - y_mean as f64).powi(2)).sum();
    let const_fit = EdgeFit {
        layer,
        input,
        output,
        name: CONST_PRIM.name,
        a: 0.0,
        b: 0.0,
        c: 0.0,
        d: y_mean,
        r2: if sst < 1e-10 { 1.0 } else { 0.0 },
        f: CONST_PRIM.f,
    };
    if sst < 1e-10 {
        return const_fit;
    }

    let mut best = const_fit;
    for prim in PRIMITIVES {
        if let Some((a, b, c, d, r2)) = fit_primitive(xs, ys, prim) {
            if r2 > best.r2 {
                best = EdgeFit {
                    layer,
                    input,
                    output,
                    name: prim.name,
                    a,
                    b,
                    c,
                    d,
                    r2,
                    f: prim.f,
                };
            }
        }
    }
    best
}

/// Символьная копия одного слоя: bias узлов + фиты активных рёбер.
pub struct SymbolicLayer {
    pub bias: Vec<f32>,
    pub fits: Vec<EdgeFit>,
    pub n_inputs: usize,
    pub n_outputs: usize,
}

/// Символьная копия KAN: послойные формулы, вычислимые как модель.
/// Имя переменной в формуле. Имя со спецсимволами берётся в обратные кавычки:
/// иначе `12·температура, °C` читалось бы как два разных терма.
fn var_name(raw: &str) -> String {
    let mut chars = raw.chars();
    let identifier = chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_');
    // h{k}_{j} — зарезервированное пространство имён промежуточных узлов.
    let hidden = raw
        .strip_prefix('h')
        .and_then(|rest| rest.split_once('_'))
        .is_some_and(|(layer, node)| {
            !layer.is_empty()
                && !node.is_empty()
                && layer.chars().all(|c| c.is_ascii_digit())
                && node.chars().all(|c| c.is_ascii_digit())
        });
    if identifier && !hidden {
        raw.to_string()
    } else {
        let escaped = raw.chars().fold(String::new(), |mut out, c| {
            match c {
                '\\' => out.push_str("\\\\"),
                '`' => out.push_str("\\`"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                control if control.is_control() => out.extend(control.escape_default()),
                other => out.push(other),
            }
            out
        });
        format!("`{escaped}`")
    }
}

fn input_var(schema: &ModelSchema, i: usize) -> String {
    var_name(schema.inputs()[i].name())
}

fn output_var(schema: &ModelSchema, o: usize) -> String {
    var_name(schema.outputs()[o].name())
}

pub struct SymbolicKan {
    pub layers: Vec<SymbolicLayer>,
}

/// Равномерное прореживание до `max_samples` точек.
fn subsample(col: &[f32], max_samples: usize) -> Vec<f32> {
    if col.len() <= max_samples {
        return col.to_vec();
    }
    let stride = col.len() as f32 / max_samples as f32;
    (0..max_samples)
        .map(|k| col[(k as f32 * stride) as usize])
        .collect()
}

/// Извлекает символьную копию KAN: фит всех активных рёбер по активациям
/// `calibration` `[N, F]` (нормализованным), не более `max_samples` точек
/// на ребро.
pub fn symbolize(kan: &KanNet, calibration: &Array2<f32>, max_samples: usize) -> SymbolicKan {
    let acts = kan.activations(calibration);
    let mut layers = Vec::new();
    for (layer, &(n_in, n_out)) in kan.layer_dims().iter().enumerate() {
        let mut fits = Vec::new();
        for i in 0..n_in {
            let xs = subsample(&acts[layer].column(i).to_vec(), max_samples);
            for o in 0..n_out {
                if !kan.edge_active(layer, i, o) {
                    continue;
                }
                let ys = kan.edge_curve(layer, i, o, &xs);
                fits.push(fit_edge(layer, i, o, &xs, &ys));
            }
        }
        layers.push(SymbolicLayer {
            bias: kan.layer_bias(layer),
            fits,
            n_inputs: n_in,
            n_outputs: n_out,
        });
    }
    SymbolicKan { layers }
}

fn fmt_num(v: f32) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() >= 1e-5 && v.abs() < 1e6 {
        // Шести знаков достаточно для читаемого отчёта и не превращает
        // небольшие, но реальные коэффициенты в ноль при печати.
        let mut text = format!("{v:.6}");
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    } else {
        format!("{v:.6e}")
    }
}

/// Аргумент примитива: `b·var + c` (без множителя при b=1, без слагаемого
/// при c=0 — чтобы формулы читались).
fn fmt_arg(b: f32, var: &str, c: f32) -> String {
    let scaled = if (b - 1.0).abs() < 1e-6 {
        var.to_string()
    } else {
        format!("{}·{var}", fmt_num(b))
    };
    if c.abs() < 1e-6 {
        scaled
    } else if c > 0.0 {
        format!("{scaled} + {}", fmt_num(c))
    } else {
        format!("{scaled} - {}", fmt_num(-c))
    }
}

impl SymbolicKan {
    fn check_schema(&self, schema: &ModelSchema) -> Result<(), String> {
        let first = self
            .layers
            .first()
            .ok_or_else(|| "символьная KAN не содержит слоёв".to_string())?;
        let last = self.layers.last().unwrap();
        if self
            .layers
            .windows(2)
            .any(|pair| pair[0].n_outputs != pair[1].n_inputs)
        {
            return Err("несогласованные размерности слоёв символьной KAN".to_string());
        }
        schema.check_dims(first.n_inputs, last.n_outputs)
    }

    /// Сворачивает нормализацию в коэффициенты: результат принимает СЫРЫЕ
    /// входы и выдаёт СЫРЫЕ выходы. Преобразование точное (не новый фит):
    /// вход слоя 0 `b·x_norm + c = (b/σ)·x_raw + (c − b·μ/σ)`; последний слой
    /// умножается на σ_out (a, d, bias), bias дополнительно сдвигается на
    /// μ_out. R² рёбер не меняется — аффинная инвариантность.
    /// Промежуточные узлы `h` остаются безразмерными.
    pub fn denormalize(&self, in_norm: &Normalizer, out_norm: &Normalizer) -> SymbolicKan {
        let n_layers = self.layers.len();
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(k, layer)| {
                let first = k == 0;
                let last = k + 1 == n_layers;
                let mut bias = layer.bias.clone();
                if last {
                    for (o, b) in bias.iter_mut().enumerate() {
                        *b = out_norm.std[o] * *b + out_norm.mean[o];
                    }
                }
                let fits = layer
                    .fits
                    .iter()
                    .map(|fit| {
                        let mut f = fit.clone();
                        if first {
                            let (mu, sigma) = (in_norm.mean[f.input], in_norm.std[f.input]);
                            f.c -= f.b * mu / sigma;
                            f.b /= sigma;
                        }
                        if last {
                            let sigma = out_norm.std[f.output];
                            f.a *= sigma;
                            f.d *= sigma;
                        }
                        f
                    })
                    .collect();
                SymbolicLayer {
                    bias,
                    fits,
                    n_inputs: layer.n_inputs,
                    n_outputs: layer.n_outputs,
                }
            })
            .collect();
        SymbolicKan { layers }
    }

    /// Послойные формулы: входы и выходы названы по схеме, промежуточные узлы —
    /// `h{k}_{j}`. Пространство — то, в котором живёт модель: нормализованное
    /// после `symbolize`, исходное после `denormalize`.
    ///
    /// У модели без имён (старый checkpoint, чёрный ящик) схема синтетическая,
    /// поэтому формулы выглядят как раньше: `x0`, `y0`.
    pub fn formulas(&self, schema: &ModelSchema) -> Result<String, String> {
        self.check_schema(schema)?;
        let mut out = String::new();
        for (k, layer) in self.layers.iter().enumerate() {
            let last = k + 1 == self.layers.len();
            for o in 0..layer.n_outputs {
                let lhs = if last {
                    output_var(schema, o)
                } else {
                    format!("h{}_{o}", k + 1)
                };
                // Константы (bias узла + все d рёбер) складываются в один терм.
                let mut constant = layer.bias[o] as f64;
                let mut terms: Vec<String> = Vec::new();
                for fit in layer.fits.iter().filter(|f| f.output == o) {
                    let var = if k == 0 {
                        input_var(schema, fit.input)
                    } else {
                        format!("h{k}_{}", fit.input)
                    };
                    match fit.name {
                        "const" => constant += fit.d as f64,
                        // Линейное ребро сворачивается: a·(b·x+c)+d = ab·x + (ac+d).
                        "x" => {
                            constant += (fit.a * fit.c + fit.d) as f64;
                            let coeff = fit.a * fit.b;
                            if coeff != 0.0 {
                                terms.push(format!("{}·{var}", fmt_num(coeff)));
                            }
                        }
                        name => {
                            constant += fit.d as f64;
                            let arg = fmt_arg(fit.b, &var, fit.c);
                            let call = match name {
                                "x^2" => format!("({arg})^2"),
                                "x^3" => format!("({arg})^3"),
                                "1/x" => format!("1/({arg})"),
                                _ => format!("{name}({arg})"),
                            };
                            terms.push(format!("{}·{call}", fmt_num(fit.a)));
                        }
                    }
                }
                let mut rhs = fmt_num(constant as f32);
                for t in terms {
                    if let Some(stripped) = t.strip_prefix('-') {
                        rhs.push_str(&format!(" - {stripped}"));
                    } else {
                        rhs.push_str(&format!(" + {t}"));
                    }
                }
                out.push_str(&format!("{lhs} = {rhs}\n"));
            }
        }
        Ok(out)
    }

    /// Человекочитаемые концы ребра: у первого/последнего слоя это колонки
    /// схемы, внутри сети — промежуточные узлы `h{k}_{j}`.
    pub fn edge_labels(
        &self,
        edge: &EdgeFit,
        schema: &ModelSchema,
    ) -> Result<(String, String), String> {
        self.check_schema(schema)?;
        let layer = self
            .layers
            .get(edge.layer)
            .ok_or_else(|| format!("ребро ссылается на отсутствующий слой {}", edge.layer))?;
        if edge.input >= layer.n_inputs || edge.output >= layer.n_outputs {
            return Err(format!(
                "ребро слоя {} имеет индексы {}→{} вне размерности {}→{}",
                edge.layer, edge.input, edge.output, layer.n_inputs, layer.n_outputs
            ));
        }
        let input = if edge.layer == 0 {
            schema.inputs()[edge.input].display_name()
        } else {
            format!("h{}_{}", edge.layer, edge.input)
        };
        let output = if edge.layer + 1 == self.layers.len() {
            schema.outputs()[edge.output].display_name()
        } else {
            format!("h{}_{}", edge.layer + 1, edge.output)
        };
        Ok((input, output))
    }

    /// Вычисление формул как модели: `[N, F]` (норм.) -> `[N, O]` (норм.).
    pub fn predict(&self, inputs: &Array2<f32>) -> Array2<f32> {
        let mut x = inputs.clone();
        for layer in &self.layers {
            let n = x.nrows();
            let mut out = Array2::<f32>::zeros((n, layer.n_outputs));
            for o in 0..layer.n_outputs {
                out.column_mut(o).fill(layer.bias[o]);
            }
            for fit in &layer.fits {
                for r in 0..n {
                    out[[r, fit.output]] += fit.eval(x[[r, fit.input]]);
                }
            }
            x = out;
        }
        x
    }

    /// (min, среднее) R² подгонки по всем рёбрам.
    pub fn edge_r2_stats(&self) -> (f32, f32) {
        let all: Vec<f32> = self
            .layers
            .iter()
            .flat_map(|l| l.fits.iter().map(|f| f.r2))
            .collect();
        if all.is_empty() {
            return (1.0, 1.0);
        }
        let min = all.iter().copied().fold(f32::INFINITY, f32::min);
        let mean = all.iter().sum::<f32>() / all.len() as f32;
        (min, mean)
    }

    /// Рёбра с подгонкой хуже порога — кандидаты на ручной разбор.
    pub fn weak_edges(&self, r2_threshold: f32) -> Vec<&EdgeFit> {
        self.layers
            .iter()
            .flat_map(|l| l.fits.iter())
            .filter(|f| f.r2 < r2_threshold)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Normalizer;
    use crate::optim::Adam;
    use crate::tensor::Tensor;
    use crate::{blackbox, kan::KanNet};

    fn linspace(lo: f32, hi: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
            .collect()
    }

    /// Аффинная подгонка восстанавливает известный примитив.
    #[test]
    fn fit_recovers_sin() {
        let xs = linspace(-3.0, 3.0, 120);
        let ys: Vec<f32> = xs
            .iter()
            .map(|&x| 2.0 * (1.5 * x - 0.5).sin() + 0.3)
            .collect();
        let fit = fit_edge(0, 0, 0, &xs, &ys);
        assert_eq!(fit.name, "sin", "ожидался sin, получен {}", fit.name);
        assert!(fit.r2 > 0.999, "R²={} слишком низкий", fit.r2);
    }

    /// Доменная защита: log фитится на данных, где часть аффинных кандидатов
    /// уводит аргумент в минус — они пропускаются, а не дают NaN/отказ.
    #[test]
    fn log_domain_is_protected() {
        let xs = linspace(-2.0, 2.0, 120);
        let ys: Vec<f32> = xs.iter().map(|&x| 1.7 * (x + 3.0).ln() - 0.4).collect();
        let log_prim = PRIMITIVES.iter().find(|p| p.name == "log").unwrap();
        let (a, b, c, _d, r2) = fit_primitive(&xs, &ys, log_prim).expect("log должен фититься");
        assert!(r2 > 0.999, "R²={r2} слишком низкий");
        assert!(r2.is_finite() && a.is_finite());
        // Найденный аффинный сдвиг держит весь домен положительным.
        assert!(xs.iter().all(|&x| b * x + c > 0.0));

        let fit = fit_edge(0, 0, 0, &xs, &ys);
        assert!(fit.r2 > 0.999);
    }

    /// Вне калибровочного домена symbolic-модель всё равно не должна
    /// производить NaN: у обратной функции нулевой аргумент клампится.
    #[test]
    fn reciprocal_eval_is_finite_at_zero() {
        let reciprocal = PRIMITIVES.iter().find(|p| p.name == "1/x").unwrap();
        let fit = EdgeFit {
            layer: 0,
            input: 0,
            output: 0,
            name: reciprocal.name,
            a: 1.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
            r2: 1.0,
            f: reciprocal.f,
        };
        assert!(fit.eval(0.0).is_finite());
    }

    /// Текстовая формула не должна терять малое активное ребро: иначе
    /// напечатанное выражение и `SymbolicKan::predict` были бы разными
    /// моделями.
    #[test]
    fn formulas_keep_small_linear_edges() {
        let sym = SymbolicKan {
            layers: vec![SymbolicLayer {
                bias: vec![0.0],
                fits: vec![EdgeFit {
                    layer: 0,
                    input: 0,
                    output: 0,
                    name: "x",
                    a: 1.0,
                    b: 1e-4,
                    c: 0.0,
                    d: 0.0,
                    r2: 1.0,
                    f: PRIMITIVES[0].f,
                }],
                n_inputs: 1,
                n_outputs: 1,
            }],
        };
        let synthetic = ModelSchema::synthetic(1, 1).unwrap();
        assert!(sym.formulas(&synthetic).unwrap().contains("0.0001·x0"));

        // Схема с именами подставляет их вместо x0/y0; имя со спецсимволами
        // берётся в обратные кавычки.
        let named = ModelSchema::new(
            vec![crate::schema::Column::numeric(
                "температура, °C",
                crate::schema::ColumnRole::Input,
            )
            .unwrap()],
            vec![
                crate::schema::Column::numeric("moisture", crate::schema::ColumnRole::Output)
                    .unwrap(),
            ],
        )
        .unwrap();
        let text = sym.formulas(&named).unwrap();
        assert!(text.starts_with("moisture = "), "формула: {text}");
        assert!(text.contains("`температура, °C`"), "формула: {text}");
        assert!(sym
            .formulas(&ModelSchema::synthetic(2, 1).unwrap())
            .is_err());

        let (input, output) = sym.edge_labels(&sym.layers[0].fits[0], &named).unwrap();
        assert_eq!(input, "температура, °C");
        assert_eq!(output, "moisture");
    }

    #[test]
    fn formula_variable_names_are_unambiguous() {
        assert_eq!(var_name("temperature"), "temperature");
        assert_eq!(var_name("_температура2"), "_температура2");
        assert_eq!(var_name("2temperature"), "`2temperature`");
        assert_eq!(var_name("h1_0"), "`h1_0`");
        assert_eq!(var_name("a`b\n\\c\0"), "`a\\`b\\n\\\\c\\u{0}`");
    }

    /// Символьная копия воспроизводит обученную KAN: формулы — это модель.
    #[test]
    fn symbolic_matches_trained_kan() {
        // Инициализация весов детерминирована: без seed исход зависит от
        // состояния thread-local RNG, оставленного другими тестами.
        crate::init::set_init_seed(0);
        let data = blackbox::sum().generate(64, 0);
        let in_norm = Normalizer::fit(&data.inputs, &Normalizer::all_continuous(2));
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(1));
        let xn = in_norm.transform(&data.inputs);
        let x = Tensor::constant(xn.clone().into_dyn());
        let y = Tensor::constant(out_norm.transform(&data.outputs).into_dyn());

        let net = KanNet::new(2, 0, 1, 5, 1);
        let mut opt = Adam::new(net.parameters(), 3e-3);
        for _ in 0..400 {
            opt.zero_grad();
            let loss = net.loss(&x, &y);
            loss.backward();
            opt.step();
        }

        let sym = symbolize(&net, &xn, 128);
        let text = sym
            .formulas(&ModelSchema::synthetic(2, 1).unwrap())
            .unwrap();
        assert!(text.starts_with("y0 = "), "формула: {text}");

        // Верность формул по отношению к самой KAN.
        let kan_pred = net.predict(&x).data();
        let sym_pred = sym.predict(&xn);
        let mean = kan_pred.iter().sum::<f32>() / kan_pred.len() as f32;
        let sst: f32 = kan_pred.iter().map(|p| (p - mean).powi(2)).sum();
        let sse: f32 = kan_pred
            .iter()
            .zip(sym_pred.iter())
            .map(|(k, s)| (k - s).powi(2))
            .sum();
        let r2 = 1.0 - sse / sst.max(1e-12);
        assert!(r2 > 0.99, "формулы разошлись с KAN: R²={r2}\n{text}");
    }

    /// denormalize сворачивает z-score точно: сырой пайплайн совпадает с
    /// «нормализовать -> формулы -> денормализовать» до машинной точности.
    #[test]
    fn denormalize_folds_zscore_exactly() {
        let data = blackbox::projectile().generate(48, 3);
        let n_in = data.inputs.ncols();
        let n_out = data.outputs.ncols();
        let in_norm = Normalizer::fit(&data.inputs, &Normalizer::all_continuous(n_in));
        let out_norm = Normalizer::fit(&data.outputs, &Normalizer::all_continuous(n_out));
        let xn = in_norm.transform(&data.inputs);

        // Двухслойная сырая (без обучения) KAN: важна точность свёртки,
        // а не качество модели.
        let net = KanNet::new(n_in, 4, 2, 5, n_out);
        let sym = symbolize(&net, &xn, 64);
        let sym_raw = sym.denormalize(&in_norm, &out_norm);

        let via_norm = out_norm.inverse_transform(&sym.predict(&xn));
        let direct = sym_raw.predict(&data.inputs);
        for (a, b) in via_norm.iter().zip(direct.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * a.abs().max(1.0),
                "расхождение денормализации: {a} vs {b}"
            );
        }

        // R² рёбер не тронут (аффинная инвариантность).
        let r2_before: Vec<f32> = sym.layers[0].fits.iter().map(|f| f.r2).collect();
        let r2_after: Vec<f32> = sym_raw.layers[0].fits.iter().map(|f| f.r2).collect();
        assert_eq!(r2_before, r2_after);
    }
}
