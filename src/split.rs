//! Протокол оценки: план разбиения и изоляция test.
//!
//! Отбор конфигурации идёт по validation (или CV), test открывается один раз.
//! Это обеспечено не договорённостью, а типами: поисковые функции получают
//! [`SearchPool`] и физически не видят [`HoldoutTest`], а сам `HoldoutTest`
//! наружу отдаёт не данные, а метрики — через [`HoldoutTest::evaluate`].
//!
//! Модуль отдельный именно поэтому: приватные поля доступны всему модулю,
//! поэтому отдельный модуль задаёт границу API.

use crate::data::NumericDataset;
use crate::metrics::{evaluate, evaluate_per_output, EvalSource, Metrics, RunOrigin};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Умолчания в одном месте, чтобы разные вызовы не получали разные «разумные»
/// значения. Seed-ы данных и финальной модели к разбиению не относятся, но
/// фиксируются здесь же по той же причине.
pub const DEFAULT_TRAIN_FRAC: f32 = 0.70;
pub const DEFAULT_VAL_FRAC: f32 = 0.15;
pub const DEFAULT_SPLIT_SEED: u64 = 1;
pub const DEFAULT_DATA_SEED: u64 = 0;
pub const DEFAULT_FINAL_INIT_SEED: u64 = 0;
pub const DEFAULT_K: usize = 5;
pub const DEFAULT_TEST_FRAC: f32 = 0.15;

/// План разбиения. Ровно два варианта: «ручная настройка» в интерфейсе — это
/// другие числа в `Holdout`, а не отдельный доменный случай.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitPlan {
    Holdout {
        train_frac: f32,
        val_frac: f32,
        split_seed: u64,
    },
    /// Test откладывается СРАЗУ и в поиске не участвует; K-fold идёт по
    /// оставшемуся pool.
    KFold {
        k: usize,
        folds_seed: u64,
        test_frac: f32,
        test_seed: u64,
    },
}

impl Default for SplitPlan {
    fn default() -> Self {
        SplitPlan::Holdout {
            train_frac: DEFAULT_TRAIN_FRAC,
            val_frac: DEFAULT_VAL_FRAC,
            split_seed: DEFAULT_SPLIT_SEED,
        }
    }
}

impl SplitPlan {
    /// K-fold с умолчаниями (для «мало данных» в интерфейсе).
    pub fn kfold_default() -> Self {
        SplitPlan::KFold {
            k: DEFAULT_K,
            folds_seed: DEFAULT_SPLIT_SEED,
            test_frac: DEFAULT_TEST_FRAC,
            test_seed: DEFAULT_SPLIT_SEED,
        }
    }

    /// Проверка плана вместе с числом строк: доли могут быть корректны, а
    /// разбиение при этом давать пустую часть после округления.
    pub fn validate(&self, n_rows: usize) -> Result<(), String> {
        match *self {
            SplitPlan::Holdout {
                train_frac,
                val_frac,
                ..
            } => {
                if !train_frac.is_finite() || !val_frac.is_finite() {
                    return Err("доли разбиения должны быть конечными".to_string());
                }
                if train_frac <= 0.0 {
                    return Err("train_frac должен быть > 0".to_string());
                }
                if val_frac <= 0.0 {
                    return Err("val_frac должен быть > 0 (отбор идёт по validation)".to_string());
                }
                if train_frac + val_frac >= 1.0 {
                    return Err(format!(
                        "train_frac + val_frac = {:.3} должно быть < 1: на test ничего не остаётся",
                        train_frac + val_frac
                    ));
                }
                let (n_train, n_val, n_test) = holdout_counts(n_rows, train_frac, val_frac);
                if n_train == 0 || n_val == 0 || n_test == 0 {
                    return Err(format!(
                        "{n_rows} строк при {:.0}/{:.0}/{:.0} дают train {n_train}, validation \
                         {n_val}, test {n_test}: каждая часть должна быть непустой",
                        train_frac * 100.0,
                        val_frac * 100.0,
                        (1.0 - train_frac - val_frac) * 100.0
                    ));
                }
                Ok(())
            }
            SplitPlan::KFold { k, test_frac, .. } => {
                if !test_frac.is_finite() {
                    return Err("test_frac должен быть конечным".to_string());
                }
                if k < 2 {
                    return Err("k должно быть >= 2".to_string());
                }
                if test_frac <= 0.0 || test_frac >= 1.0 {
                    return Err("test_frac должен быть в (0, 1)".to_string());
                }
                let n_test = (n_rows as f32 * test_frac).round() as usize;
                let n_pool = n_rows.saturating_sub(n_test);
                if n_test == 0 {
                    return Err(format!(
                        "{n_rows} строк при test_frac {test_frac} дают пустой test"
                    ));
                }
                if n_pool < k {
                    return Err(format!(
                        "в pool остаётся {n_pool} строк — меньше, чем folds ({k})"
                    ));
                }
                Ok(())
            }
        }
    }

    /// Разбить датасет. Test отделяется здесь и дальше существует только как
    /// [`HoldoutTest`].
    pub fn prepare(&self, data: &NumericDataset) -> Result<PreparedSplit, String> {
        let n = data.len();
        self.validate(n)?;
        let mut idx: Vec<usize> = (0..n).collect();

        match *self {
            SplitPlan::Holdout {
                train_frac,
                val_frac,
                split_seed,
            } => {
                idx.shuffle(&mut StdRng::seed_from_u64(split_seed));
                let (n_train, n_val, _) = holdout_counts(n, train_frac, val_frac);
                let pool_rows = &idx[..n_train + n_val];
                Ok(PreparedSplit {
                    search: SearchPool {
                        pool: data.gather(pool_rows),
                        folds: vec![FoldIndices {
                            train: (0..n_train).collect(),
                            val: (n_train..n_train + n_val).collect(),
                        }],
                        source: EvalSource::Validation,
                    },
                    test: HoldoutTest {
                        data: data.gather(&idx[n_train + n_val..]),
                        plan: *self,
                    },
                })
            }
            SplitPlan::KFold {
                k,
                folds_seed,
                test_frac,
                test_seed,
            } => {
                idx.shuffle(&mut StdRng::seed_from_u64(test_seed));
                let n_test = (n as f32 * test_frac).round() as usize;
                let test = data.gather(&idx[..n_test]);
                let pool = data.gather(&idx[n_test..]);

                // Индексы внутри pool: перемешиваем отдельным seed, режем на k
                // частей, отличающихся не больше чем на строку.
                let m = pool.len();
                let mut pool_idx: Vec<usize> = (0..m).collect();
                pool_idx.shuffle(&mut StdRng::seed_from_u64(folds_seed));
                let base = m / k;
                let rem = m % k;
                let mut folds = Vec::with_capacity(k);
                let mut start = 0;
                for f in 0..k {
                    let size = base + usize::from(f < rem);
                    let val: Vec<usize> = pool_idx[start..start + size].to_vec();
                    let train: Vec<usize> = pool_idx[..start]
                        .iter()
                        .chain(pool_idx[start + size..].iter())
                        .copied()
                        .collect();
                    folds.push(FoldIndices { train, val });
                    start += size;
                }

                Ok(PreparedSplit {
                    search: SearchPool {
                        pool,
                        folds,
                        source: EvalSource::Cv { k },
                    },
                    test: HoldoutTest {
                        data: test,
                        plan: *self,
                    },
                })
            }
        }
    }
}

/// Округление долей holdout в строки. Test получает остаток, поэтому сумма
/// всегда равна `n`.
fn holdout_counts(n: usize, train_frac: f32, val_frac: f32) -> (usize, usize, usize) {
    let n_train = (n as f32 * train_frac).round() as usize;
    let n_val = (n as f32 * val_frac).round() as usize;
    if n_train + n_val >= n {
        // Патология округления на крошечных наборах: validate() это отвергнет.
        return (n_train.min(n), n.saturating_sub(n_train), 0);
    }
    (n_train, n_val, n - n_train - n_val)
}

/// Результат разбиения: то, с чем работает поиск, и отдельно отложенный test.
pub struct PreparedSplit {
    pub search: SearchPool,
    pub test: HoldoutTest,
}

struct FoldIndices {
    train: Vec<usize>,
    val: Vec<usize>,
}

/// Данные, доступные поиску: holdout train/validation либо K-fold pool. Наружу
/// — единый интерфейс folds, чтобы поиск не различал эти случаи.
///
/// Поля закрыты: достать отсюда test невозможно, потому что его здесь нет.
pub struct SearchPool {
    pool: NumericDataset,
    folds: Vec<FoldIndices>,
    source: EvalSource,
}

impl SearchPool {
    pub fn n_folds(&self) -> usize {
        self.folds.len()
    }

    /// Строк в pool (train + validation).
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Чем является метрика, посчитанная на этом pool: validation или CV.
    pub fn source(&self) -> EvalSource {
        self.source
    }

    /// Train и validation части fold. Материализуются по требованию: хранить k
    /// копий данных незачем.
    pub fn fold(&self, i: usize) -> Result<(NumericDataset, NumericDataset), String> {
        let f = self
            .folds
            .get(i)
            .ok_or_else(|| format!("fold {i} вне диапазона 0..{}", self.folds.len()))?;
        Ok((self.pool.gather(&f.train), self.pool.gather(&f.val)))
    }

    /// Весь pool — для переобучения выбранной конфигурации на train+validation
    /// перед единственным замером на test.
    pub fn all(&self) -> NumericDataset {
        self.pool.gather(&(0..self.pool.len()).collect::<Vec<_>>())
    }

    /// Происхождение прогона на fold `i`. Номер fold проставляет pool, а не
    /// потребитель: у holdout он обязан быть `None`, у CV — `Some(i)`, и
    /// `aggregate_runs` это проверяет.
    pub(crate) fn run_origin(&self, fold: usize, init_seed: u64) -> RunOrigin {
        RunOrigin {
            fold: match self.source {
                EvalSource::Validation => None,
                _ => Some(fold),
            },
            init_seed,
        }
    }
}

/// Отложенный test. Не конвертируется в [`SearchPool`] и не отдаёт свои данные:
/// наружу уходят только метрики.
pub struct HoldoutTest {
    data: NumericDataset,
    plan: SplitPlan,
}

impl HoldoutTest {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Единственный способ измерить качество на test: замыкание получает
    /// входы и возвращает предсказания в исходных единицах, наружу уходит
    /// [`FinalEval`]. Сам набор тип не покидает. Метод потребляет test, поэтому
    /// один подготовленный split нельзя оценить повторно с другой моделью.
    pub fn evaluate<F>(self, predict: F, final_init_seed: u64) -> Result<FinalEval, String>
    where
        F: FnOnce(&Array2<f32>) -> Array2<f32>,
    {
        let pred = predict(&self.data.inputs);
        if pred.dim() != self.data.outputs.dim() {
            return Err(format!(
                "final_eval: модель вернула форму {:?}, ожидалась {:?}",
                pred.dim(),
                self.data.outputs.dim()
            ));
        }
        Ok(FinalEval {
            metrics: evaluate(&pred, &self.data.outputs),
            per_output: evaluate_per_output(&pred, &self.data.outputs),
            origin: FinalOrigin {
                final_init_seed,
                plan: self.plan,
                test_rows: self.data.len(),
            },
        })
    }
}

/// Происхождение единственного финального замера. `final_init_seed` фиксируется
/// ДО открытия test: выбор seed по результату — та же форма подбора.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinalOrigin {
    pub final_init_seed: u64,
    pub plan: SplitPlan,
    pub test_rows: usize,
}

/// Результат единственного замера на test.
#[derive(Debug, Clone)]
pub struct FinalEval {
    pub metrics: Metrics,
    pub per_output: Vec<Metrics>,
    pub origin: FinalOrigin,
}

impl FinalEval {
    /// Источник всегда test — метрика из этого типа не может выдать себя за
    /// validation.
    pub fn source(&self) -> EvalSource {
        EvalSource::Test
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::BTreeSet;

    /// Датасет, где значение строки равно её исходному индексу: так любую
    /// строку можно опознать в любой части разбиения.
    fn labeled(n: usize) -> NumericDataset {
        let inputs = Array2::from_shape_fn((n, 2), |(i, j)| (i * 10 + j) as f32);
        let outputs = Array2::from_shape_fn((n, 1), |(i, _)| i as f32);
        NumericDataset::new(inputs, outputs)
    }

    fn labels(d: &NumericDataset) -> BTreeSet<i64> {
        d.outputs.iter().map(|&v| v as i64).collect()
    }

    #[test]
    fn default_plan_is_70_15_15_with_seed_1() {
        assert_eq!(
            SplitPlan::default(),
            SplitPlan::Holdout {
                train_frac: 0.70,
                val_frac: 0.15,
                split_seed: 1,
            }
        );
    }

    #[test]
    fn holdout_parts_are_disjoint_and_cover_all() {
        let data = labeled(100);
        let s = SplitPlan::default().prepare(&data).unwrap();
        let (train, val) = s.search.fold(0).unwrap();

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 15);
        assert_eq!(s.test.len(), 15);

        let (lt, lv) = (labels(&train), labels(&val));
        assert!(lt.is_disjoint(&lv));
        let pool: BTreeSet<i64> = lt.union(&lv).copied().collect();
        assert_eq!(pool, labels(&s.search.all()));
        assert_eq!(pool.len(), 85);
    }

    #[test]
    fn holdout_is_deterministic_and_seed_sensitive() {
        let data = labeled(50);
        let a = SplitPlan::default().prepare(&data).unwrap();
        let b = SplitPlan::default().prepare(&data).unwrap();
        assert_eq!(labels(&a.search.all()), labels(&b.search.all()));

        let other = SplitPlan::Holdout {
            train_frac: 0.70,
            val_frac: 0.15,
            split_seed: 99,
        }
        .prepare(&data)
        .unwrap();
        assert_ne!(labels(&a.search.all()), labels(&other.search.all()));
    }

    #[test]
    fn holdout_rejects_empty_part_after_rounding() {
        // Доли валидны, но на 5 строках validation округляется в ноль.
        let plan = SplitPlan::default();
        assert!(plan.validate(100).is_ok());
        let err = plan.validate(5).unwrap_err();
        assert!(err.contains("validation"), "текст ошибки: {err}");
        assert!(plan.prepare(&labeled(5)).is_err());
    }

    #[test]
    fn holdout_rejects_bad_fractions() {
        let bad = [
            (0.0, 0.15),
            (0.7, 0.0),
            (0.9, 0.1),
            (0.9, 0.2),
            (f32::NAN, 0.15),
        ];
        for (train_frac, val_frac) in bad {
            let plan = SplitPlan::Holdout {
                train_frac,
                val_frac,
                split_seed: 1,
            };
            assert!(
                plan.validate(100).is_err(),
                "{train_frac}/{val_frac} должно отвергаться"
            );
        }
    }

    #[test]
    fn kfold_covers_each_pool_row_exactly_once() {
        let data = labeled(100);
        let s = SplitPlan::kfold_default().prepare(&data).unwrap();
        assert_eq!(s.search.n_folds(), 5);
        assert_eq!(s.test.len(), 15);
        assert_eq!(s.search.len(), 85);

        let mut seen: Vec<i64> = Vec::new();
        for i in 0..s.search.n_folds() {
            let (train, val) = s.search.fold(i).unwrap();
            // Внутри fold train и validation не пересекаются...
            assert!(labels(&train).is_disjoint(&labels(&val)));
            // ...и вместе дают весь pool.
            assert_eq!(train.len() + val.len(), s.search.len());
            seen.extend(labels(&val));
        }
        // Каждая строка pool побывала в validation ровно один раз.
        seen.sort_unstable();
        let unique: BTreeSet<i64> = seen.iter().copied().collect();
        assert_eq!(seen.len(), 85);
        assert_eq!(unique.len(), 85);
        assert_eq!(unique, labels(&s.search.all()));
    }

    #[test]
    fn kfold_balances_fold_sizes() {
        // 85 строк на 5 folds: 17 в каждом; 83 на 5 — 17,17,17,16,16.
        let s = SplitPlan::kfold_default().prepare(&labeled(98)).unwrap();
        let mut sizes: Vec<usize> = (0..s.search.n_folds())
            .map(|i| s.search.fold(i).unwrap().1.len())
            .collect();
        sizes.sort_unstable();
        assert!(sizes[sizes.len() - 1] - sizes[0] <= 1, "{sizes:?}");
    }

    #[test]
    fn kfold_rejects_bad_parameters() {
        let plan = |k, test_frac| SplitPlan::KFold {
            k,
            folds_seed: 1,
            test_frac,
            test_seed: 1,
        };
        assert!(plan(1, 0.15).validate(100).is_err()); // k < 2
        assert!(plan(5, 0.0).validate(100).is_err()); // пустой test
        assert!(plan(5, 1.0).validate(100).is_err()); // пустой pool
        assert!(plan(10, 0.15).validate(11).is_err()); // pool меньше k
        assert!(plan(5, 0.15).validate(100).is_ok());
    }

    /// Ключевой инвариант Э1 на уровне данных: ни одна строка test не попадает
    /// ни в один train или validation fold.
    #[test]
    fn test_rows_never_appear_in_search_pool() {
        for plan in [SplitPlan::default(), SplitPlan::kfold_default()] {
            let data = labeled(100);
            let s = plan.prepare(&data).unwrap();
            let test_len = s.test.len();
            // Test виден только внутри теста — через evaluate, куда приходят
            // именно его входы.
            let mut test_labels = BTreeSet::new();
            s.test
                .evaluate(
                    |inputs| {
                        for row in inputs.rows() {
                            test_labels.insert((row[0] / 10.0).round() as i64);
                        }
                        Array2::zeros((inputs.nrows(), 1))
                    },
                    DEFAULT_FINAL_INIT_SEED,
                )
                .unwrap();

            assert_eq!(test_labels.len(), test_len);
            assert!(test_labels.is_disjoint(&labels(&s.search.all())));
            for i in 0..s.search.n_folds() {
                let (train, val) = s.search.fold(i).unwrap();
                assert!(test_labels.is_disjoint(&labels(&train)));
                assert!(test_labels.is_disjoint(&labels(&val)));
            }
        }
    }

    #[test]
    fn final_eval_records_its_origin() {
        let data = labeled(100);
        let plan = SplitPlan::default();
        let s = plan.prepare(&data).unwrap();
        // Идеальное предсказание: y равен первой колонке входа / 10.
        let f = s
            .test
            .evaluate(
                |inputs| {
                    Array2::from_shape_fn((inputs.nrows(), 1), |(i, _)| {
                        (inputs[[i, 0]] / 10.0).round()
                    })
                },
                7,
            )
            .unwrap();
        assert!((f.metrics.r2 - 1.0).abs() < 1e-6);
        assert_eq!(f.per_output.len(), 1);
        assert_eq!(f.origin.final_init_seed, 7);
        assert_eq!(f.origin.test_rows, 15);
        assert_eq!(f.origin.plan, plan);
        assert_eq!(f.source(), EvalSource::Test);
    }

    #[test]
    fn final_eval_reacts_to_test_targets() {
        let data = labeled(100);
        let plan = SplitPlan::default();
        let prepared = plan.prepare(&data).unwrap();
        let mut test_labels = BTreeSet::new();
        let baseline = prepared
            .test
            .evaluate(
                |inputs| {
                    Array2::from_shape_fn((inputs.nrows(), 1), |(i, _)| {
                        let label = (inputs[[i, 0]] / 10.0).round();
                        test_labels.insert(label as i64);
                        label
                    })
                },
                DEFAULT_FINAL_INIT_SEED,
            )
            .unwrap();
        assert!((baseline.metrics.r2 - 1.0).abs() < 1e-6);

        let mut poisoned = NumericDataset::new(data.inputs.clone(), data.outputs.clone());
        for i in 0..poisoned.len() {
            if test_labels.contains(&((poisoned.inputs[[i, 0]] / 10.0).round() as i64)) {
                poisoned.outputs[[i, 0]] += 1_000.0;
            }
        }
        let changed = plan
            .prepare(&poisoned)
            .unwrap()
            .test
            .evaluate(
                |inputs| {
                    Array2::from_shape_fn((inputs.nrows(), 1), |(i, _)| {
                        (inputs[[i, 0]] / 10.0).round()
                    })
                },
                DEFAULT_FINAL_INIT_SEED,
            )
            .unwrap();

        assert!(
            (changed.metrics.r2 - baseline.metrics.r2).abs() > 1e-3,
            "подмена test-targets должна менять FinalEval"
        );
    }

    #[test]
    fn final_eval_rejects_wrong_prediction_shape() {
        let s = SplitPlan::default().prepare(&labeled(100)).unwrap();
        let err = s
            .test
            .evaluate(|inputs| Array2::zeros((inputs.nrows(), 2)), 0)
            .unwrap_err();
        assert!(err.contains("модель вернула форму"), "текст ошибки: {err}");
    }

    #[test]
    fn pool_source_distinguishes_holdout_from_cv() {
        let data = labeled(100);
        assert_eq!(
            SplitPlan::default().prepare(&data).unwrap().search.source(),
            EvalSource::Validation
        );
        assert_eq!(
            SplitPlan::kfold_default()
                .prepare(&data)
                .unwrap()
                .search
                .source(),
            EvalSource::Cv { k: 5 }
        );
    }

    #[test]
    fn fold_index_out_of_range_is_an_error() {
        let s = SplitPlan::default().prepare(&labeled(100)).unwrap();
        assert!(s.search.fold(1).is_err());
    }
}
