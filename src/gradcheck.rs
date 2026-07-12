use crate::tensor::Tensor;
use ndarray::ArrayD;

/// Через `init::rand_uniform`: тест с изломом (relu, abs) может вызвать
/// `set_init_seed` и стать детерминированным — конечная разность у излома
/// иначе флакает при неудачной энтропии.
pub(crate) fn rand_tensor(shape: &[usize]) -> Tensor {
    Tensor::new(crate::init::rand_uniform(shape, -0.7, 0.7))
}

pub(crate) fn grad_check<F: Fn(&[Tensor]) -> Tensor>(inputs: &[Tensor], f: F) {
    grad_check_with_tol(inputs, 1e-3, 1e-2, f);
}

pub(crate) fn grad_check_with_tol<F: Fn(&[Tensor]) -> Tensor>(
    inputs: &[Tensor],
    eps: f32,
    tol: f32,
    f: F,
) {
    for t in inputs {
        t.zero_grad();
    }
    let out = f(inputs);
    out.backward();
    let analytic: Vec<ArrayD<f32>> = inputs.iter().map(|t| t.grad()).collect();

    for (i, t) in inputs.iter().enumerate() {
        let n = t.data().len();
        for j in 0..n {
            let orig = t.data().as_slice().expect("grad_check: contiguous data")[j];

            t.update_data(|data, _| {
                data.as_slice_mut().expect("grad_check: contiguous data")[j] = orig + eps;
            });
            let fp = f(inputs).item();

            t.update_data(|data, _| {
                data.as_slice_mut().expect("grad_check: contiguous data")[j] = orig - eps;
            });
            let fm = f(inputs).item();

            t.update_data(|data, _| {
                data.as_slice_mut().expect("grad_check: contiguous data")[j] = orig;
            });

            let numeric = (fp - fm) / (2.0 * eps);
            let a = analytic[i].as_slice().expect("grad_check: contiguous grad")[j];
            let diff = (numeric - a).abs();
            let denom = numeric.abs().max(a.abs()).max(1.0);
            assert!(
                diff / denom < tol,
                "градиент input {i}[{j}]: analytic={a}, numeric={numeric}, rel_diff={}",
                diff / denom
            );
        }
    }
}
