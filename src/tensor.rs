//! Минимальный autograd-движок поверх `ndarray`.
//!
//! Дизайн — динамический граф в стиле micrograd/PyTorch:
//! `Tensor` это `Rc<RefCell<TensorData>>`. Каждая операция строит новый узел,
//! запоминая родителей и замыкание `backward`, которое принимает градиент
//! выхода и аккумулирует градиенты в родителей. Полный обратный проход —
//! топологическая сортировка графа и вызов замыканий в обратном порядке.

use ndarray::{Array2, ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Тип замыкания обратного прохода: получает градиент ВЫХОДА этого узла и
/// распределяет его по родителям. Не захватывает сам узел — это разрывает
/// цикл ссылок (узел владеет замыканием, замыкание владело бы узлом).
pub(crate) type BackwardFn = Box<dyn Fn(&ArrayD<f32>)>;

pub struct TensorData {
    pub data: ArrayD<f32>,
    pub grad: ArrayD<f32>,
    pub backward: Option<BackwardFn>,
    pub parents: Vec<Tensor>,
    pub requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    /// Лист графа (параметр или вход). По умолчанию требует градиент.
    pub fn new(data: ArrayD<f32>) -> Tensor {
        let grad = ArrayD::zeros(data.raw_dim());
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad,
            backward: None,
            parents: Vec::new(),
            requires_grad: true,
        })))
    }

    /// Константа: не участвует в накоплении градиента (вход данных, маска).
    pub fn constant(data: ArrayD<f32>) -> Tensor {
        let t = Tensor::new(data);
        t.0.borrow_mut().requires_grad = false;
        t
    }

    /// Внутренний конструктор для результата операции.
    pub(crate) fn from_op(data: ArrayD<f32>, parents: Vec<Tensor>, backward: BackwardFn) -> Tensor {
        let grad = ArrayD::zeros(data.raw_dim());
        let requires_grad = parents.iter().any(|p| p.requires_grad());
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad,
            backward: Some(backward),
            parents,
            requires_grad,
        })))
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> ArrayD<f32> {
        self.0.borrow().grad.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().data.shape().to_vec()
    }

    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    /// Скалярное значение (для loss). Паникует если не один элемент —
    /// это инвариант вызывающего кода, не рантайм-ошибка.
    pub fn item(&self) -> f32 {
        let b = self.0.borrow();
        assert_eq!(b.data.len(), 1, "item() требует тензор из одного элемента");
        b.data.iter().next().copied().unwrap()
    }

    pub(crate) fn add_grad(&self, g: &ArrayD<f32>) {
        let mut b = self.0.borrow_mut();
        if !b.requires_grad {
            return;
        }
        b.grad += g;
    }

    pub fn zero_grad(&self) {
        let mut b = self.0.borrow_mut();
        b.grad.fill(0.0);
    }

    /// In-place обновление данных (используется оптимизатором).
    pub fn update_data<F: FnOnce(&mut ArrayD<f32>, &ArrayD<f32>)>(&self, f: F) {
        let mut b = self.0.borrow_mut();
        let grad = b.grad.clone();
        f(&mut b.data, &grad);
    }

    /// Перезаписать данные (загрузка сохранённых весов). Форма должна совпадать.
    pub fn set_data(&self, new: ArrayD<f32>) {
        let mut b = self.0.borrow_mut();
        assert_eq!(
            b.data.shape(),
            new.shape(),
            "set_data: форма должна совпадать с параметром"
        );
        b.data = new;
    }

    /// Обратный проход от этого (скалярного) узла.
    pub fn backward(&self) {
        // Топологическая сортировка графа.
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: Vec<*const RefCell<TensorData>> = Vec::new();
        build_topo(self, &mut topo, &mut visited);

        // Градиент корня = 1.
        {
            let mut b = self.0.borrow_mut();
            b.grad = ArrayD::ones(b.data.raw_dim());
        }

        for node in topo.iter().rev() {
            let (grad, backward_present) = {
                let b = node.0.borrow();
                (b.grad.clone(), b.backward.is_some())
            };
            if backward_present {
                // Берём замыкание во временное владение, чтобы не держать
                // borrow во время вызова (внутри оно трогает родителей).
                let bw = node.0.borrow_mut().backward.take();
                if let Some(bw) = bw {
                    bw(&grad);
                    node.0.borrow_mut().backward = Some(bw);
                }
            }
        }
    }
}

fn build_topo(
    node: &Tensor,
    topo: &mut Vec<Tensor>,
    visited: &mut Vec<*const RefCell<TensorData>>,
) {
    let ptr = Rc::as_ptr(&node.0);
    if visited.contains(&ptr) {
        return;
    }
    visited.push(ptr);
    let parents = node.0.borrow().parents.clone();
    for p in &parents {
        build_topo(p, topo, visited);
    }
    topo.push(node.clone());
}

/// Сумма градиента по broadcast-осям обратно к форме `target`.
/// Нужно когда forward делал broadcasting (например bias [d] + [n, d]).
fn unbroadcast(mut grad: ArrayD<f32>, target: &[usize]) -> ArrayD<f32> {
    // Сначала сворачиваем лишние ведущие оси.
    while grad.ndim() > target.len() {
        grad = grad.sum_axis(ndarray::Axis(0));
    }
    // Затем оси, которые в target были размера 1.
    for (axis, &t) in target.iter().enumerate() {
        if t == 1 && grad.shape()[axis] != 1 {
            grad = grad
                .sum_axis(ndarray::Axis(axis))
                .insert_axis(ndarray::Axis(axis));
        }
    }
    grad.into_shape_with_order(IxDyn(target))
        .expect("unbroadcast: форма совпадает")
}

// ---------------------------------------------------------------------------
// Операции
// ---------------------------------------------------------------------------

impl Tensor {
    /// Поэлементное сложение с broadcasting.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow().data.clone();
        let b = other.0.borrow().data.clone();
        let out = &a + &b;

        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        let (lhs, rhs) = (self.clone(), other.clone());
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&unbroadcast(g.clone(), &a_shape));
            rhs.add_grad(&unbroadcast(g.clone(), &b_shape));
        });
        Tensor::from_op(out, vec![self.clone(), other.clone()], backward)
    }

    /// Поэлементное умножение с broadcasting.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow().data.clone();
        let b = other.0.borrow().data.clone();
        let out = &a * &b;

        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        let (lhs, rhs) = (self.clone(), other.clone());
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&unbroadcast(g * &b, &a_shape));
            rhs.add_grad(&unbroadcast(g * &a, &b_shape));
        });
        Tensor::from_op(out, vec![self.clone(), other.clone()], backward)
    }

    /// Умножение на скаляр.
    pub fn scale(&self, s: f32) -> Tensor {
        let out = &self.0.borrow().data * s;
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&(g * s));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    /// Матричное умножение 2D: [m, k] · [k, n] = [m, n].
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = to_2d(&self.0.borrow().data);
        let b = to_2d(&other.0.borrow().data);
        let out = a.dot(&b);

        let (lhs, rhs) = (self.clone(), other.clone());
        let a_for_b = a.clone();
        let b_for_a = b.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let g2 = to_2d(g);
            // dA = G · B^T,  dB = A^T · G
            let da = g2.dot(&b_for_a.t());
            let db = a_for_b.t().dot(&g2);
            lhs.add_grad(&da.into_dyn());
            rhs.add_grad(&db.into_dyn());
        });
        Tensor::from_op(out.into_dyn(), vec![self.clone(), other.clone()], backward)
    }

    /// ReLU.
    pub fn relu(&self) -> Tensor {
        let a = self.0.borrow().data.clone();
        let mask = a.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let out = a.mapv(|x| x.max(0.0));
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            lhs.add_grad(&(g * &mask));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    /// Сумма всех элементов в скаляр.
    pub fn sum(&self) -> Tensor {
        let a = self.0.borrow().data.clone();
        let total = a.sum();
        let out = ArrayD::from_elem(IxDyn(&[1]), total);
        let shape = a.shape().to_vec();
        let lhs = self.clone();
        let backward: BackwardFn = Box::new(move |g: &ArrayD<f32>| {
            let scalar = g.iter().next().copied().unwrap();
            lhs.add_grad(&ArrayD::from_elem(IxDyn(&shape), scalar));
        });
        Tensor::from_op(out, vec![self.clone()], backward)
    }

    /// Среднее всех элементов.
    pub fn mean(&self) -> Tensor {
        let n = self.0.borrow().data.len() as f32;
        self.sum().scale(1.0 / n)
    }
}

/// Привести динамический массив к 2D-виду (для matmul).
fn to_2d(a: &ArrayD<f32>) -> Array2<f32> {
    a.clone()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("matmul ожидает 2D тензоры")
}

// ---------------------------------------------------------------------------
// Тесты: проверка градиентов конечными разностями
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn rand_tensor(shape: &[usize]) -> Tensor {
        let a = Array::random(IxDyn(shape), Uniform::new(-1.0, 1.0));
        Tensor::new(a)
    }

    /// Численная проверка: для скалярного выхода f(inputs) сравнивает
    /// аналитический градиент с конечной разностью.
    fn grad_check<F: Fn(&[Tensor]) -> Tensor>(inputs: &[Tensor], f: F) {
        let eps = 1e-3_f32;
        let tol = 1e-2_f32;

        // Аналитический градиент.
        for t in inputs {
            t.zero_grad();
        }
        let out = f(inputs);
        out.backward();
        let analytic: Vec<ArrayD<f32>> = inputs.iter().map(|t| t.grad()).collect();

        // Численный градиент.
        for (i, t) in inputs.iter().enumerate() {
            let n = t.0.borrow().data.len();
            for j in 0..n {
                let orig = t.0.borrow().data.as_slice().unwrap()[j];

                t.0.borrow_mut().data.as_slice_mut().unwrap()[j] = orig + eps;
                let fp = f(inputs).item();
                t.0.borrow_mut().data.as_slice_mut().unwrap()[j] = orig - eps;
                let fm = f(inputs).item();
                t.0.borrow_mut().data.as_slice_mut().unwrap()[j] = orig;

                let numeric = (fp - fm) / (2.0 * eps);
                let a = analytic[i].as_slice().unwrap()[j];
                let diff = (numeric - a).abs();
                let denom = numeric.abs().max(a.abs()).max(1.0);
                assert!(
                    diff / denom < tol,
                    "градиент input {i}[{j}]: analytic={a}, numeric={numeric}"
                );
            }
        }
    }

    #[test]
    fn check_add_mul() {
        let a = rand_tensor(&[3, 4]);
        let b = rand_tensor(&[3, 4]);
        grad_check(&[a, b], |t| t[0].add(&t[1]).mul(&t[1]).sum());
    }

    #[test]
    fn check_broadcast_add() {
        let a = rand_tensor(&[3, 4]);
        let bias = rand_tensor(&[1, 4]);
        grad_check(&[a, bias], |t| t[0].add(&t[1]).sum());
    }

    #[test]
    fn check_matmul() {
        let a = rand_tensor(&[3, 5]);
        let b = rand_tensor(&[5, 2]);
        grad_check(&[a, b], |t| t[0].matmul(&t[1]).sum());
    }

    #[test]
    fn check_relu_chain() {
        let a = rand_tensor(&[4, 6]);
        let w = rand_tensor(&[6, 3]);
        grad_check(&[a, w], |t| t[0].matmul(&t[1]).relu().mean());
    }
}
