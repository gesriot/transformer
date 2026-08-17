//! Разметка таблицы и диагностика её качества — до обучения и без GUI.
//!
//! Три части, потому что они отвечают на три разных вопроса:
//!
//! - [`TableProfile`] — что вообще лежит в файле: пропуски, нечисловые значения,
//!   константы, дубликаты. От ролей не зависит.
//! - [`SchemaDraft`] — незавершённая разметка. [`TableSchema`] для этого не
//!   годится: она обязана быть валидной и окончательной, а черновик существует
//!   именно для промежуточных состояний.
//! - [`RoleReport`] — то, что можно сказать только после выбора ролей: связи
//!   между числовыми входами.

use crate::schema::{Column, ColumnRole, TableSchema};
use crate::table::Table;
use std::collections::BTreeSet;

/// Сколько различных значений колонки сохранять для показа и для уровней
/// категории. Больше — это уже не категория, а свободный текст.
const MAX_DISTINCT: usize = 256;

// --- профиль таблицы (роли ещё не выбраны) ---

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnProfile {
    pub index: usize,
    pub name: String,
    /// Строк всего (без заголовка).
    pub total: usize,
    /// Пустые ячейки.
    pub missing: usize,
    /// Непустые ячейки, которые не читаются как конечное число.
    pub non_numeric: usize,
    /// Различные непустые значения; `None` — их больше [`MAX_DISTINCT`].
    pub distinct: Option<Vec<String>>,
}

impl ColumnProfile {
    /// Колонку можно считать числовой, только если нечисловых значений нет.
    pub fn is_numeric(&self) -> bool {
        self.non_numeric == 0
    }

    /// Одно значение на всю колонку: для обучения бесполезна, для зависимостей
    /// вырожденна.
    pub fn is_constant(&self) -> bool {
        self.missing == 0 && self.distinct.as_ref().is_some_and(|d| d.len() == 1)
    }

    pub fn n_distinct(&self) -> Option<usize> {
        self.distinct.as_ref().map(Vec::len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableProfile {
    pub rows: usize,
    pub columns: Vec<ColumnProfile>,
    /// Полностью совпадающие строки (второе и последующие вхождения).
    pub duplicate_rows: usize,
    /// Строки, ширина которых отличается от заголовка (номера как в файле).
    pub ragged_rows: Vec<usize>,
}

impl TableProfile {
    pub fn of(table: &Table) -> Self {
        let n_columns = table.n_columns();
        let header = table.header();
        let mut missing = vec![0usize; n_columns];
        let mut non_numeric = vec![0usize; n_columns];
        let mut distinct: Vec<Option<BTreeSet<String>>> = vec![Some(BTreeSet::new()); n_columns];
        let mut ragged_rows = Vec::new();
        let mut seen_rows: BTreeSet<&Vec<String>> = BTreeSet::new();
        let mut duplicate_rows = 0;

        for (r, row) in table.rows().iter().enumerate() {
            if row.len() != n_columns {
                ragged_rows.push(table.file_row(r));
            }
            if !seen_rows.insert(row) {
                duplicate_rows += 1;
            }
            for c in 0..n_columns {
                // У короткой строки отсутствующий хвост — такой же пропуск,
                // как явно пустая ячейка. Иначе профиль занижает missing.
                let text = row.get(c).map_or("", |cell| cell.trim());
                if text.is_empty() {
                    missing[c] += 1;
                    continue;
                }
                if !is_finite_f32(text) {
                    non_numeric[c] += 1;
                }
                if let Some(set) = &mut distinct[c] {
                    set.insert(text.to_string());
                    if set.len() > MAX_DISTINCT {
                        distinct[c] = None;
                    }
                }
            }
        }

        let columns = (0..n_columns)
            .map(|c| ColumnProfile {
                index: c,
                name: header
                    .and_then(|h| h.get(c))
                    .cloned()
                    .unwrap_or_else(|| format!("колонка {}", c + 1)),
                total: table.n_rows(),
                missing: missing[c],
                non_numeric: non_numeric[c],
                distinct: distinct[c].take().map(sorted_values),
            })
            .collect();

        Self {
            rows: table.n_rows(),
            columns,
            duplicate_rows,
            ragged_rows,
        }
    }

    /// Короткий человекочитаемый отчёт: только то, что действительно найдено.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.ragged_rows.is_empty() {
            let shown: Vec<String> = self
                .ragged_rows
                .iter()
                .take(5)
                .map(usize::to_string)
                .collect();
            out.push(format!(
                "строк с другим числом колонок: {} (например {})",
                self.ragged_rows.len(),
                shown.join(", ")
            ));
        }
        if self.duplicate_rows > 0 {
            out.push(format!("повторяющихся строк: {}", self.duplicate_rows));
        }
        for c in &self.columns {
            if c.missing > 0 {
                out.push(format!(
                    "'{}': пропусков {} из {}",
                    c.name, c.missing, c.total
                ));
            }
            if c.non_numeric > 0 {
                out.push(format!(
                    "'{}': нечисловых значений {} из {}",
                    c.name, c.non_numeric, c.total
                ));
            }
            if c.is_constant() {
                out.push(format!("'{}': одно значение на всю колонку", c.name));
            }
        }
        out
    }
}

/// Значения по возрастанию: числа — по величине, остальное — лексикографически.
/// Иначе коды «10» и «9» встали бы не в том порядке.
fn sorted_values(set: BTreeSet<String>) -> Vec<String> {
    let mut values: Vec<String> = set.into_iter().collect();
    if values.iter().all(|v| {
        v.parse::<f64>()
            .map(|number| number.is_finite())
            .unwrap_or(false)
    }) {
        values.sort_by(|a, b| {
            a.parse::<f64>()
                .unwrap_or(f64::NAN)
                .partial_cmp(&b.parse::<f64>().unwrap_or(f64::NAN))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    values
}

fn is_finite_f32(text: &str) -> bool {
    text.parse::<f32>().map(|v| v.is_finite()).unwrap_or(false)
}

// --- черновик разметки ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftType {
    Numeric,
    Categorical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftColumn {
    pub name: String,
    pub role: ColumnRole,
    pub ty: DraftType,
    pub unit: Option<String>,
}

/// Незавершённая разметка: может быть неполной и противоречивой. Проверка —
/// только в [`SchemaDraft::finish`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaDraft {
    columns: Vec<DraftColumn>,
    /// Различные значения каждой колонки — источник уровней категорий.
    distinct: Vec<Option<Vec<String>>>,
    /// Совместимость с тем же `f32`, в который затем конвертируется таблица.
    numeric_compatible: Vec<bool>,
}

impl SchemaDraft {
    /// Стартовое состояние: имена из заголовка, последняя колонка — выход,
    /// остальные — входы. Тип числовой везде, где данные это позволяют.
    ///
    /// Это предположение, а не решение: пользователь подтверждает разметку.
    pub fn from_profile(profile: &TableProfile) -> Self {
        let last = profile.columns.len().saturating_sub(1);
        let columns = profile
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| DraftColumn {
                name: c.name.clone(),
                role: if i == last {
                    ColumnRole::Output
                } else {
                    ColumnRole::Input
                },
                // Текст числовым быть не может — это ограничение данных, а не
                // догадка о смысле колонки.
                ty: if c.is_numeric() {
                    DraftType::Numeric
                } else {
                    DraftType::Categorical
                },
                unit: None,
            })
            .collect();
        Self {
            columns,
            distinct: profile.columns.iter().map(|c| c.distinct.clone()).collect(),
            numeric_compatible: profile
                .columns
                .iter()
                .map(ColumnProfile::is_numeric)
                .collect(),
        }
    }

    pub fn columns(&self) -> &[DraftColumn] {
        &self.columns
    }

    /// Первые `n_inputs` колонок — входы, остальные — выходы.
    ///
    /// Черновик сам ничего не угадывает: разбиение приходит снаружи — от
    /// автоопределения по заголовку либо прямо от пользователя.
    pub fn set_output_split(&mut self, n_inputs: usize) -> Result<(), String> {
        if n_inputs == 0 || n_inputs >= self.columns.len() {
            return Err(format!(
                "граница входов {n_inputs} вне диапазона 1..{}",
                self.columns.len()
            ));
        }
        for (i, column) in self.columns.iter_mut().enumerate() {
            column.role = if i < n_inputs {
                ColumnRole::Input
            } else {
                ColumnRole::Output
            };
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub fn set_role(&mut self, i: usize, role: ColumnRole) -> Result<(), String> {
        self.column_mut(i)?.role = role;
        Ok(())
    }

    pub fn set_name(&mut self, i: usize, name: impl Into<String>) -> Result<(), String> {
        self.column_mut(i)?.name = name.into();
        Ok(())
    }

    pub fn set_unit(&mut self, i: usize, unit: Option<String>) -> Result<(), String> {
        self.column_mut(i)?.unit = unit;
        Ok(())
    }

    /// Сменить тип. Числовым можно объявить только колонку без текста, а
    /// категориальной — только колонку с обозримым числом значений.
    pub fn set_type(&mut self, i: usize, ty: DraftType) -> Result<(), String> {
        let distinct = self
            .distinct
            .get(i)
            .ok_or_else(|| format!("колонка {i} вне диапазона"))?
            .clone();
        let column = self.column_mut(i)?;
        match ty {
            DraftType::Categorical if distinct.is_none() => {
                return Err(format!(
                    "'{}': слишком много различных значений для категории (> {MAX_DISTINCT})",
                    column.name
                ))
            }
            _ => {}
        }
        column.ty = ty;
        Ok(())
    }

    fn column_mut(&mut self, i: usize) -> Result<&mut DraftColumn, String> {
        self.columns
            .get_mut(i)
            .ok_or_else(|| format!("колонка {i} вне диапазона"))
    }

    /// Что мешает завершить разметку. Пустой список — черновик готов.
    pub fn issues(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.columns.iter().any(|c| c.role == ColumnRole::Input) {
            out.push("нужен хотя бы один вход".to_string());
        }
        if !self.columns.iter().any(|c| c.role == ColumnRole::Output) {
            out.push("нужен хотя бы один выход".to_string());
        }
        let mut names = BTreeSet::new();
        for (i, c) in self.columns.iter().enumerate() {
            let clean_name = c.name.trim();
            if clean_name.is_empty() {
                out.push(format!("колонка {}: имя не может быть пустым", i + 1));
            } else if !names.insert(clean_name) {
                out.push(format!("имя колонки '{clean_name}' повторяется"));
            }
            if c.role == ColumnRole::Output && c.ty == DraftType::Categorical {
                out.push(format!(
                    "'{}': выход не может быть категориальным (это регрессия)",
                    c.name
                ));
            }
            if c.role == ColumnRole::Ignore {
                continue;
            }
            match c.ty {
                DraftType::Numeric if !self.numeric_compatible[i] => {
                    let bad = self.distinct[i]
                        .as_ref()
                        .and_then(|values| values.iter().find(|value| !is_finite_f32(value)));
                    out.push(match bad {
                        Some(bad) => {
                            format!("'{}': объявлена числовой, но содержит '{bad}'", c.name)
                        }
                        None => format!(
                            "'{}': объявлена числовой, но содержит нечисловые или неконечные значения",
                            c.name
                        ),
                    });
                }
                DraftType::Categorical => match &self.distinct[i] {
                    None => out.push(format!(
                        "'{}': слишком много различных значений для категории (> {MAX_DISTINCT})",
                        c.name
                    )),
                    Some(levels) if levels.is_empty() => {
                        out.push(format!("'{}': у категории нет непустых уровней", c.name));
                    }
                    Some(_) => {}
                },
                DraftType::Numeric => {}
            }
        }
        out
    }

    fn finish_column(&self, i: usize, draft: &DraftColumn) -> Result<Column, String> {
        // Тип игнорируемой колонки после импорта нигде не используется. Если
        // уровни нельзя сохранить (свободный текст или все значения пусты),
        // не заставляем пользователя маскировать её под Numeric вручную.
        if draft.role == ColumnRole::Ignore {
            if let (DraftType::Categorical, Some(levels)) = (draft.ty, &self.distinct[i]) {
                if !levels.is_empty() {
                    return Column::categorical(&draft.name, draft.role, levels.clone());
                }
            }
            return Column::numeric(&draft.name, draft.role);
        }

        match draft.ty {
            DraftType::Numeric => Column::numeric(&draft.name, draft.role),
            DraftType::Categorical => Column::categorical(
                &draft.name,
                draft.role,
                self.distinct[i]
                    .clone()
                    .ok_or_else(|| format!("'{}': слишком много значений", draft.name))?,
            ),
        }
    }

    /// Завершить разметку. Уровни категорий берутся из данных в порядке
    /// возрастания, поэтому коды воспроизводимы между запусками.
    pub fn finish(&self) -> Result<TableSchema, String> {
        let issues = self.issues();
        if !issues.is_empty() {
            return Err(issues.join("; "));
        }
        let mut columns = Vec::with_capacity(self.columns.len());
        for (i, c) in self.columns.iter().enumerate() {
            let column = self.finish_column(i, c)?;
            columns.push(match &c.unit {
                Some(u) => column.with_unit(u),
                None => column,
            });
        }
        TableSchema::new(columns)
    }
}

// --- отчёт, зависящий от ролей ---

/// Аффинная зависимость между числовыми входами:
/// `target ≈ intercept + Σ coeff·column`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearDependency {
    pub target: usize,
    pub terms: Vec<(usize, f64)>,
    pub intercept: f64,
    pub r2: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoleReport {
    /// Индексы числовых входов, попавших в анализ.
    pub numeric_inputs: Vec<usize>,
    /// Константные входы: они не несут информации и вырождают зависимости.
    pub constant_inputs: Vec<usize>,
    pub dependencies: Vec<LinearDependency>,
}

/// Порог «зависимость точная»: `x0+x1+x2=100` даёт R² = 1 с точностью до f32.
const EXACT_R2: f64 = 0.999_999;
/// Порог «почти зависимость» — стоит показать, но она может быть случайной.
const NEAR_R2: f64 = 0.999;
/// На нескольких строках регрессия легко выглядит точной случайно. Кроме
/// общего минимума оставляем не меньше пяти степеней свободы сверх числа
/// коэффициентов и свободного члена.
const MIN_DEPENDENCY_ROWS: usize = 10;
const MIN_RESIDUAL_DOF: usize = 5;

impl RoleReport {
    pub fn warnings(&self, draft: &SchemaDraft) -> Vec<String> {
        let name = |i: usize| {
            draft
                .columns()
                .get(i)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("колонка {}", i + 1))
        };
        let mut out = Vec::new();
        for &i in &self.constant_inputs {
            out.push(format!("вход '{}' постоянен — модели он не нужен", name(i)));
        }
        for d in &self.dependencies {
            let terms: String = d
                .terms
                .iter()
                // Терм, который на выводимой точности равен нулю, читается как
                // «+ 0·y0» и только мешает: точность связи показывает R².
                .filter(|(_, k)| fmt_coeff(k.abs()) != "0")
                .map(|(i, k)| {
                    format!(
                        " {} {}·{}",
                        if *k < 0.0 { '-' } else { '+' },
                        fmt_coeff(k.abs()),
                        name(*i)
                    )
                })
                .collect();
            out.push(format!(
                "входы связаны{}: '{}' ≈ {}{terms} (R² = {:.6}). Вклад можно \
                 переложить между зависимыми входами, поэтому формулы и \
                 чувствительность по ним неоднозначны.",
                if d.r2 >= EXACT_R2 {
                    " точно"
                } else {
                    " почти"
                },
                name(d.target),
                fmt_coeff(d.intercept),
                d.r2
            ));
        }
        out
    }
}

fn fmt_coeff(v: f64) -> String {
    let rounded = (v * 1e6).round() / 1e6;
    if rounded == 0.0 {
        "0".to_string()
    } else {
        format!("{rounded}")
    }
}

/// Поиск аффинных зависимостей среди числовых входов.
///
/// Ищем именно АФФИННУЮ связь (со свободным членом): `x0 + x1 + x2 = 100` — это
/// зависимость, которую ранг самой матрицы не обнаружит. Поэтому каждая колонка
/// по очереди объясняется остальными методом наименьших квадратов со свободным
/// членом; колонки стандартизуются, иначе разномасштабные входы плохо
/// обусловлены.
///
/// Найдя зависимость, колонку исключаем из числа объясняющих: иначе одна связь
/// трёх колонок отчиталась бы трижды.
pub fn analyze_roles(table: &Table, draft: &SchemaDraft) -> RoleReport {
    let numeric_inputs: Vec<usize> = draft
        .columns()
        .iter()
        .enumerate()
        .filter(|(_, c)| c.role == ColumnRole::Input && c.ty == DraftType::Numeric)
        .map(|(i, _)| i)
        .collect();

    // Строки, где все нужные колонки читаются: пропуск не должен ломать анализ.
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); numeric_inputs.len()];
    for row in table.rows() {
        let parsed: Option<Vec<f64>> = numeric_inputs
            .iter()
            .map(|&c| {
                row.get(c)?
                    .trim()
                    // Конвертация в NumericDataset использует f32. Анализ на
                    // f64 исходного текста мог бы видеть различия, которые
                    // фактически исчезнут на входе модели.
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .map(f64::from)
            })
            .collect();
        if let Some(parsed) = parsed {
            for (slot, v) in parsed.into_iter().enumerate() {
                values[slot].push(v);
            }
        }
    }

    let n = values.first().map_or(0, Vec::len);
    let stats: Vec<(f64, f64)> = values.iter().map(|col| mean_std(col)).collect();
    let constant_inputs: Vec<usize> = numeric_inputs
        .iter()
        .zip(stats.iter())
        .filter(|(_, (_, std))| *std == 0.0)
        .map(|(&i, _)| i)
        .collect();

    let mut dependencies = Vec::new();
    // Кандидаты в объясняющие: непостоянные колонки.
    let mut pool: Vec<usize> = (0..numeric_inputs.len())
        .filter(|&s| stats[s].1 > 0.0)
        .collect();
    if n >= MIN_DEPENDENCY_ROWS && pool.len() >= 2 {
        for slot in pool.clone() {
            let predictors: Vec<usize> = pool.iter().copied().filter(|&s| s != slot).collect();
            if predictors.is_empty() || n < predictors.len() + 1 + MIN_RESIDUAL_DOF {
                continue;
            }
            if let Some((coeffs, intercept, r2)) = fit_affine(&values, &stats, slot, &predictors) {
                if r2 >= NEAR_R2 {
                    dependencies.push(LinearDependency {
                        target: numeric_inputs[slot],
                        terms: predictors
                            .iter()
                            .zip(coeffs.iter())
                            .filter(|(_, k)| k.abs() > 1e-9)
                            .map(|(&s, &k)| (numeric_inputs[s], k))
                            .collect(),
                        intercept,
                        r2,
                    });
                    pool.retain(|&s| s != slot);
                }
            }
        }
    }

    RoleReport {
        numeric_inputs,
        constant_inputs,
        dependencies,
    }
}

fn mean_std(col: &[f64]) -> (f64, f64) {
    if col.is_empty() {
        return (0.0, 0.0);
    }
    let n = col.len() as f64;
    let mean = col.iter().sum::<f64>() / n;
    let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// МНК со свободным членом. Решается в стандартизованном пространстве (там
/// свободный член равен нулю по построению), затем коэффициенты переводятся в
/// исходные единицы.
/// Индексы здесь выражают матричную формулу напрямую: переписывание циклов на
/// итераторы сделало бы нормальные уравнения нечитаемыми.
#[allow(clippy::needless_range_loop)]
fn fit_affine(
    values: &[Vec<f64>],
    stats: &[(f64, f64)],
    target: usize,
    predictors: &[usize],
) -> Option<(Vec<f64>, f64, f64)> {
    let n = values[target].len();
    let p = predictors.len();
    let (y_mean, y_std) = stats[target];
    if y_std == 0.0 {
        return None;
    }

    // Нормальные уравнения в стандартизованных переменных.
    let z = |slot: usize, r: usize| (values[slot][r] - stats[slot].0) / stats[slot].1;
    let mut a = vec![vec![0.0f64; p]; p];
    let mut b = vec![0.0f64; p];
    for r in 0..n {
        let yr = (values[target][r] - y_mean) / y_std;
        for i in 0..p {
            let zi = z(predictors[i], r);
            b[i] += zi * yr;
            for j in i..p {
                a[i][j] += zi * z(predictors[j], r);
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            a[i][j] = a[j][i];
        }
    }

    // Если среди объясняющих колонок есть своя точная зависимость, обычные
    // нормальные уравнения вырождены и скрывают даже очевидную связь target.
    // Малая регуляризация выбирает одно из эквивалентных решений, практически
    // не меняя прогноз и итоговый R².
    let diagonal_scale = (0..p).map(|i| a[i][i].abs()).fold(0.0, f64::max);
    let ridge = diagonal_scale * 1e-12;
    for i in 0..p {
        a[i][i] += ridge;
    }

    let solved = solve(&mut a, &mut b)?;
    // Обратно в исходные единицы: b_raw = b_std · std_y / std_x.
    let coeffs: Vec<f64> = solved
        .iter()
        .zip(predictors.iter())
        .map(|(k, &slot)| k * y_std / stats[slot].1)
        .collect();
    let intercept = y_mean
        - coeffs
            .iter()
            .zip(predictors.iter())
            .map(|(k, &slot)| k * stats[slot].0)
            .sum::<f64>();

    let mut sse = 0.0;
    let mut sst = 0.0;
    for r in 0..n {
        let pred = intercept
            + coeffs
                .iter()
                .zip(predictors.iter())
                .map(|(k, &slot)| k * values[slot][r])
                .sum::<f64>();
        let actual = values[target][r];
        sse += (actual - pred) * (actual - pred);
        sst += (actual - y_mean) * (actual - y_mean);
    }
    if sst == 0.0 {
        return None;
    }
    Some((coeffs, intercept, 1.0 - sse / sst))
}

/// Гаусс с частичным выбором ведущего элемента. `None` — система вырождена
/// (сами объясняющие колонки линейно зависимы).
///
/// Индексы, как и в `fit_affine`, повторяют школьную запись метода.
#[allow(clippy::needless_range_loop)]
fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&x, &y| {
            a[x][col]
                .abs()
                .partial_cmp(&a[y][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Delimiter;

    fn table(text: &str) -> Table {
        Table::parse_text(text, Delimiter::Auto, true).unwrap()
    }

    #[test]
    fn profile_counts_gaps_text_and_duplicates() {
        let t = table("a,b,c\n1,x,5\n,y,5\n1,x,5\n2,,5\n");
        let p = TableProfile::of(&t);
        assert_eq!(p.rows, 4);
        assert_eq!(p.columns[0].missing, 1);
        assert_eq!(p.columns[0].non_numeric, 0);
        assert_eq!(p.columns[1].non_numeric, 3); // x, y, x
        assert_eq!(p.columns[1].missing, 1);
        assert!(p.columns[2].is_constant());
        assert!(!p.columns[0].is_constant());
        assert_eq!(p.duplicate_rows, 1); // строка «1,x,5» встречается дважды
        assert!(p.ragged_rows.is_empty());

        let warnings = p.warnings().join("\n");
        assert!(warnings.contains("'a': пропусков 1 из 4"), "{warnings}");
        assert!(warnings.contains("одно значение"), "{warnings}");
    }

    #[test]
    fn profile_reports_ragged_rows_with_file_numbers() {
        let t = table("a,b\n1,2\n3\n4,5\n");
        let p = TableProfile::of(&t);
        // Заголовок — строка 1, значит короткая строка это третья в файле.
        assert_eq!(p.ragged_rows, vec![3]);
        assert_eq!(p.columns[1].missing, 1, "отсутствующий хвост — пропуск");
    }

    #[test]
    fn draft_defaults_last_column_to_output() {
        let t = table("temp,mat,moisture\n80,песок,12\n60,глина,18\n");
        let d = SchemaDraft::from_profile(&TableProfile::of(&t));
        assert_eq!(d.columns()[0].role, ColumnRole::Input);
        assert_eq!(d.columns()[1].role, ColumnRole::Input);
        assert_eq!(d.columns()[2].role, ColumnRole::Output);
        // Текстовая колонка не может быть числовой.
        assert_eq!(d.columns()[1].ty, DraftType::Categorical);
        assert_eq!(d.columns()[0].ty, DraftType::Numeric);
        assert!(d.issues().is_empty(), "{:?}", d.issues());
    }

    #[test]
    fn draft_finish_builds_schema_with_levels() {
        let t = table("temp,mat,moisture\n80,песок,12\n60,глина,18\n70,песок,15\n");
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));
        d.set_unit(0, Some("°C".to_string())).unwrap();
        let schema = d.finish().unwrap();

        assert_eq!(schema.columns()[1].cardinality(), Some(2));
        // Уровни отсортированы, поэтому коды воспроизводимы.
        assert_eq!(schema.columns()[1].category_level(0).unwrap(), "глина");
        assert_eq!(schema.columns()[1].category_level(1).unwrap(), "песок");
        let model = schema.to_model_schema().unwrap();
        assert_eq!(model.input_names(), vec!["temp", "mat"]);
        assert_eq!(model.inputs()[0].unit(), Some("°C"));
    }

    #[test]
    fn draft_reports_what_blocks_finishing() {
        let t = table("a,b\n1,x\n2,y\n");
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));
        // Текстовая колонка назначена выходом.
        let issues = d.issues();
        assert!(
            issues
                .iter()
                .any(|i| i.contains("не может быть категориальным")),
            "{issues:?}"
        );
        assert!(d.finish().is_err());

        // Числовой её объявить нельзя — в данных текст.
        d.set_type(1, DraftType::Numeric).unwrap();
        assert!(d.issues().iter().any(|i| i.contains("содержит 'x'")));

        // Без выхода тоже нельзя.
        d.set_type(1, DraftType::Categorical).unwrap();
        d.set_role(1, ColumnRole::Ignore).unwrap();
        assert!(d.issues().iter().any(|i| i.contains("хотя бы один выход")));
    }

    #[test]
    fn draft_issues_cover_all_schema_failures() {
        let t = table("a,b,y\n1,2,3\n4,5,6\n");
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));

        d.set_name(0, " ").unwrap();
        assert!(d.issues().iter().any(|issue| issue.contains("имя")));
        assert!(d.finish().is_err());

        d.set_name(0, "b").unwrap();
        assert!(d.issues().iter().any(|issue| issue.contains("повторяется")));
        assert!(d.finish().is_err());
    }

    #[test]
    fn numeric_issue_survives_distinct_value_limit_and_non_finite_values() {
        let mut text = String::from("x,label,y\n");
        for i in 0..=MAX_DISTINCT {
            text.push_str(&format!("{i},value-{i},{i}\n"));
        }
        let t = table(&text);
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));
        assert!(d.set_type(1, DraftType::Numeric).is_ok());
        assert!(d.issues().iter().any(|issue| issue.contains("нечисловые")));

        // Свободный текст можно честно исключить, не подменяя его тип вручную.
        d.set_role(1, ColumnRole::Ignore).unwrap();
        assert!(d.issues().is_empty(), "{:?}", d.issues());
        assert!(d.finish().is_ok());

        let non_finite = table("x,y\nNaN,1\n2,3\n");
        let mut non_finite_draft = SchemaDraft::from_profile(&TableProfile::of(&non_finite));
        non_finite_draft.set_type(0, DraftType::Numeric).unwrap();
        assert!(non_finite_draft
            .issues()
            .iter()
            .any(|issue| issue.contains("NaN")));
    }

    #[test]
    fn output_split_sets_roles() {
        let t = table("a,b,c,d\n1,2,3,4\n5,6,7,8\n");
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));
        d.set_output_split(2).unwrap();
        let roles: Vec<ColumnRole> = d.columns().iter().map(|c| c.role).collect();
        assert_eq!(
            roles,
            vec![
                ColumnRole::Input,
                ColumnRole::Input,
                ColumnRole::Output,
                ColumnRole::Output
            ]
        );
        assert!(d.set_output_split(0).is_err());
        assert!(d.set_output_split(4).is_err());
    }

    #[test]
    fn draft_rejects_out_of_range_column() {
        let t = table("a,b\n1,2\n");
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));
        assert!(d.set_role(9, ColumnRole::Input).is_err());
        assert!(d.set_type(9, DraftType::Numeric).is_err());
    }

    /// Главный случай: доли состава, сумма которых постоянна. Ранг матрицы
    /// такую связь не показывает — нужна именно аффинная зависимость.
    #[test]
    fn finds_affine_dependency_of_a_simplex() {
        let mut text = String::from("x0,x1,x2,y\n");
        for i in 0..40 {
            let x0 = 1.0 + i as f64;
            let x1 = 3.0 + (i % 7) as f64;
            let x2 = 100.0 - x0 - x1;
            text.push_str(&format!("{x0},{x1},{x2},{}\n", i));
        }
        let t = table(&text);
        let d = SchemaDraft::from_profile(&TableProfile::of(&t));
        let report = analyze_roles(&t, &d);

        assert_eq!(report.numeric_inputs, vec![0, 1, 2]);
        assert_eq!(report.dependencies.len(), 1, "одна связь — один отчёт");
        let dep = &report.dependencies[0];
        assert!(dep.r2 >= EXACT_R2, "R² = {}", dep.r2);
        assert!((dep.intercept - 100.0).abs() < 1e-6, "{}", dep.intercept);
        assert_eq!(dep.terms.len(), 2);
        for (_, k) in &dep.terms {
            assert!((k + 1.0).abs() < 1e-6, "коэффициент {k}");
        }

        let warnings = report.warnings(&d).join("\n");
        assert!(warnings.contains("входы связаны точно"), "{warnings}");
        assert!(warnings.contains("неоднозначны"), "{warnings}");
    }

    #[test]
    fn independent_inputs_produce_no_dependency() {
        let mut text = String::from("a,b,y\n");
        for i in 0..40 {
            let a = (i % 5) as f64 * 1.7 + 0.3;
            let b = ((i * 7) % 11) as f64;
            text.push_str(&format!("{a},{b},{}\n", i));
        }
        let t = table(&text);
        let d = SchemaDraft::from_profile(&TableProfile::of(&t));
        let report = analyze_roles(&t, &d);
        assert!(report.dependencies.is_empty(), "{:?}", report.dependencies);
        assert!(report.constant_inputs.is_empty());
    }

    #[test]
    fn constant_input_is_reported_not_treated_as_dependency() {
        let mut text = String::from("a,c,y\n");
        for i in 0..20 {
            text.push_str(&format!("{i},5,{}\n", i * 2));
        }
        let t = table(&text);
        let d = SchemaDraft::from_profile(&TableProfile::of(&t));
        let report = analyze_roles(&t, &d);
        assert_eq!(report.constant_inputs, vec![1]);
        assert!(report.dependencies.is_empty());
        assert!(report.warnings(&d).iter().any(|w| w.contains("постоянен")));
    }

    #[test]
    fn dependencies_survive_redundant_predictors() {
        let mut text = String::from("a,b,c,d,y\n");
        for i in 0..40 {
            let a = i as f64 + 1.0;
            let c = (i % 7) as f64 + 0.25;
            text.push_str(&format!("{a},{},{c},{},{}\n", 2.0 * a, 3.0 * c, i));
        }
        let t = table(&text);
        let d = SchemaDraft::from_profile(&TableProfile::of(&t));
        let report = analyze_roles(&t, &d);

        assert_eq!(
            report.dependencies.len(),
            2,
            "две независимые связи не должны делать МНК неразрешимым: {:?}",
            report.dependencies
        );
        assert!(report
            .dependencies
            .iter()
            .all(|dependency| dependency.r2 >= EXACT_R2));
    }

    #[test]
    fn tiny_but_varying_inputs_are_not_constant() {
        let mut text = String::from("a,b,y\n");
        for i in 1..=30 {
            let a = i as f32 * 1e-20;
            text.push_str(&format!("{a},{},{}\n", 2.0 * a, i));
        }
        let t = table(&text);
        let d = SchemaDraft::from_profile(&TableProfile::of(&t));
        let report = analyze_roles(&t, &d);

        assert!(report.constant_inputs.is_empty());
        assert_eq!(report.dependencies.len(), 1);
        assert!(report.dependencies[0].r2 >= EXACT_R2);
    }

    /// Роли меняют отчёт: колонка, выведенная из входов, в анализ не попадает.
    #[test]
    fn report_follows_roles() {
        let mut text = String::from("x0,x1,x2,y\n");
        for i in 0..30 {
            let x0 = 2.0 + i as f64;
            let x1 = 5.0 + (i % 4) as f64;
            text.push_str(&format!("{x0},{x1},{},{}\n", 100.0 - x0 - x1, i));
        }
        let t = table(&text);
        let mut d = SchemaDraft::from_profile(&TableProfile::of(&t));
        assert_eq!(analyze_roles(&t, &d).dependencies.len(), 1);

        d.set_role(2, ColumnRole::Ignore).unwrap();
        let report = analyze_roles(&t, &d);
        assert_eq!(report.numeric_inputs, vec![0, 1]);
        assert!(report.dependencies.is_empty());
    }
}
