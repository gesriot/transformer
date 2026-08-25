//! Прогноз: числовое ядро и разбор входов по схеме.
//!
//! Ядро принимает готовую матрицу строк и ничего не знает про источник:
//! одна строка из командной строки и тысяча строк из таблицы проходят один и
//! тот же путь. Иначе единичный и пакетный прогноз со временем разъезжаются —
//! ровно так и было раньше.
//!
//! Разбор входов вынесен отдельным слоем: он превращает текст (аргументы CLI,
//! ячейки таблицы) в числа по [`ModelSchema`] — числовые значения как числа,
//! категории по ПОДПИСЯМ уровней — и адресует ошибку колонкой и строкой.

use crate::data::{Normalizer, OutOfRange};
use crate::numeric_model::NumericModel;
use crate::schema::ModelSchema;
use crate::tensor::Tensor;
use ndarray::{Array2, Ix2};

/// Строка, вышедшая за обученный диапазон входов.
#[derive(Clone, Debug, PartialEq)]
pub struct RowWarning {
    /// Номер строки в переданной матрице (0-based).
    pub row: usize,
    pub details: Vec<OutOfRange>,
}

#[derive(Clone, Debug)]
pub struct Predictions {
    pub outputs: Array2<f32>,
    /// Предупреждения об экстраполяции — по строкам, где она есть.
    pub warnings: Vec<RowWarning>,
}

impl Predictions {
    pub fn rows(&self) -> usize {
        self.outputs.nrows()
    }

    /// Сколько строк вышло за обученный диапазон.
    pub fn extrapolated_rows(&self) -> usize {
        self.warnings.len()
    }
}

/// Числовое ядро прогноза: одна или несколько строк.
///
/// Проверяет форму и конечность входа: молча посчитанный прогноз по NaN хуже
/// отказа, потому что выглядит как результат.
pub fn predict_rows(
    model: &NumericModel,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    inputs: &Array2<f32>,
) -> Result<Predictions, String> {
    if inputs.nrows() == 0 {
        return Err("нет строк для прогноза".to_string());
    }
    if inputs.ncols() != in_norm.n_features() {
        return Err(format!(
            "модель ожидает {} входов, получено {}",
            in_norm.n_features(),
            inputs.ncols()
        ));
    }
    if let Some((r, c)) = inputs
        .indexed_iter()
        .find(|(_, v)| !v.is_finite())
        .map(|((r, c), _)| (r, c))
    {
        return Err(format!(
            "строка {}, вход {}: значение не конечно",
            r + 1,
            c + 1
        ));
    }

    let x = Tensor::constant(in_norm.transform(inputs).into_dyn());
    let normalized = model
        .predict(&x)
        .data()
        .into_dimensionality::<Ix2>()
        .map_err(|_| "модель вернула не матрицу [N, O]".to_string())?;
    if normalized.nrows() != inputs.nrows() {
        return Err(format!(
            "модель вернула {} строк вместо {}",
            normalized.nrows(),
            inputs.nrows()
        ));
    }
    if normalized.ncols() != out_norm.n_features() {
        return Err(format!(
            "модель вернула {} выходов вместо {}",
            normalized.ncols(),
            out_norm.n_features()
        ));
    }
    let outputs = out_norm.inverse_transform(&normalized);
    if let Some(((r, c), _)) = outputs.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "строка {}, выход {}: модель вернула не конечное значение",
            r + 1,
            c + 1
        ));
    }

    let warnings = (0..inputs.nrows())
        .filter_map(|row| {
            let details = in_norm.out_of_range_details(&inputs.row(row).to_vec());
            (!details.is_empty()).then_some(RowWarning { row, details })
        })
        .collect();
    Ok(Predictions { outputs, warnings })
}

/// Разобрать одну строку значений по схеме.
///
/// `where_` — адрес для сообщения об ошибке: «строка 5» у таблицы, пусто у
/// одиночного вызова.
pub fn parse_row(schema: &ModelSchema, values: &[&str], where_: &str) -> Result<Vec<f32>, String> {
    if values.len() != schema.n_inputs() {
        return Err(format!(
            "{}модель ожидает {} входов ({}), получено {}",
            prefix(where_),
            schema.n_inputs(),
            schema.input_names().join(", "),
            values.len()
        ));
    }
    let mut row = Vec::with_capacity(values.len());
    for (i, raw) in values.iter().enumerate() {
        let column = &schema.inputs()[i];
        let text = raw.trim();
        if text.is_empty() {
            return Err(format!(
                "{}колонка '{}': пустое значение",
                prefix(where_),
                column.name()
            ));
        }
        // Категория задаётся ПОДПИСЬЮ уровня: код пришлось бы помнить, а
        // ошибиться в нём — молча получить другой материал.
        let value = match column.cardinality() {
            Some(_) => column
                .category_code(text)
                .map_err(|e| format!("{}{e}", prefix(where_)))? as f32,
            None => text.parse::<f32>().map_err(|_| {
                format!(
                    "{}колонка '{}': ожидалось число, получено '{text}'",
                    prefix(where_),
                    column.name()
                )
            })?,
        };
        if !value.is_finite() {
            return Err(format!(
                "{}колонка '{}': значение не конечно",
                prefix(where_),
                column.name()
            ));
        }
        row.push(value);
    }
    Ok(row)
}

/// Разобрать несколько строк: ошибки адресуются номером строки.
pub fn parse_rows(
    schema: &ModelSchema,
    rows: &[Vec<String>],
    row_labels: &[usize],
) -> Result<Array2<f32>, String> {
    if rows.is_empty() {
        return Err("нет строк для прогноза".to_string());
    }
    let mut values = Array2::<f32>::zeros((rows.len(), schema.n_inputs()));
    for (r, row) in rows.iter().enumerate() {
        let label = row_labels.get(r).copied().unwrap_or(r + 1);
        let cells: Vec<&str> = row.iter().map(String::as_str).collect();
        let parsed = parse_row(schema, &cells, &format!("строка {label}"))?;
        for (c, v) in parsed.into_iter().enumerate() {
            values[[r, c]] = v;
        }
    }
    Ok(values)
}

fn prefix(where_: &str) -> String {
    if where_.is_empty() {
        String::new()
    } else {
        format!("{where_}, ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::config::ModelConfig;
    use crate::encoders::{FeatureSpec, ValueEncoderConfig};
    use crate::numeric_model::{KanConfig, ModelKind, NumericConfig};
    use crate::schema::{Column, ColumnRole};
    use crate::train::fit_normalizers;

    fn model_and_norms() -> (NumericModel, Normalizer, Normalizer) {
        let data = blackbox::sum().generate(32, 0);
        let specs = vec![FeatureSpec::Continuous; 2];
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        crate::init::set_init_seed(0);
        let nc = NumericConfig {
            kind: ModelKind::Mlp,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 8,
            mlp_layers: 1,
            kan: KanConfig::default(),
        };
        (nc.build(&specs, 1), in_norm, out_norm)
    }

    fn schema_with_category() -> ModelSchema {
        ModelSchema::new(
            vec![
                Column::numeric("температура", ColumnRole::Input).unwrap(),
                Column::categorical(
                    "материал",
                    ColumnRole::Input,
                    vec!["песок".into(), "глина".into()],
                )
                .unwrap(),
            ],
            vec![Column::numeric("влажность", ColumnRole::Output).unwrap()],
        )
        .unwrap()
    }

    /// Одна строка и много строк идут одним путём: результат для строки внутри
    /// пакета совпадает с результатом её же отдельного прогноза.
    #[test]
    fn single_row_matches_the_same_row_in_a_batch() {
        let (model, in_norm, out_norm) = model_and_norms();
        let batch = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.5, -0.4, 0.9, 0.3]).unwrap();
        let all = predict_rows(&model, &in_norm, &out_norm, &batch).unwrap();

        let single = Array2::from_shape_vec((1, 2), vec![0.5, -0.4]).unwrap();
        let one = predict_rows(&model, &in_norm, &out_norm, &single).unwrap();

        assert_eq!(all.rows(), 3);
        assert_eq!(one.rows(), 1);
        assert_eq!(all.outputs[[1, 0]], one.outputs[[0, 0]]);
    }

    #[test]
    fn core_rejects_bad_shapes_and_values() {
        let (model, in_norm, out_norm) = model_and_norms();
        let empty = Array2::<f32>::zeros((0, 2));
        assert!(predict_rows(&model, &in_norm, &out_norm, &empty).is_err());

        let wide = Array2::<f32>::zeros((1, 5));
        let err = predict_rows(&model, &in_norm, &out_norm, &wide).unwrap_err();
        assert!(err.contains("ожидает 2 входов"), "{err}");

        let nan = Array2::from_shape_vec((1, 2), vec![f32::NAN, 0.0]).unwrap();
        let err = predict_rows(&model, &in_norm, &out_norm, &nan).unwrap_err();
        assert!(err.contains("не конечно"), "{err}");

        let wrong_outputs = Normalizer::fit(
            &Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap(),
            &Normalizer::all_continuous(2),
        );
        let err =
            predict_rows(&model, &in_norm, &wrong_outputs, &Array2::zeros((1, 2))).unwrap_err();
        assert!(err.contains("1 выходов вместо 2"), "{err}");
    }

    #[test]
    fn warnings_point_at_extrapolated_rows() {
        let (model, in_norm, out_norm) = model_and_norms();
        // Вторая строка заведомо вне обученного диапазона.
        let inputs = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 500.0, 0.0]).unwrap();
        let p = predict_rows(&model, &in_norm, &out_norm, &inputs).unwrap();
        assert_eq!(p.extrapolated_rows(), 1);
        assert_eq!(p.warnings[0].row, 1);
        assert_eq!(p.warnings[0].details[0].feature, 0);
    }

    #[test]
    fn schema_layer_reads_numbers_and_category_labels() {
        let schema = schema_with_category();
        let row = parse_row(&schema, &["70", "глина"], "").unwrap();
        assert_eq!(row, vec![70.0, 1.0]);
        // Пробелы вокруг подписи не должны мешать.
        assert_eq!(
            parse_row(&schema, &[" 70 ", " песок "], "").unwrap()[1],
            0.0
        );
    }

    #[test]
    fn schema_layer_addresses_its_errors() {
        let schema = schema_with_category();
        let err = parse_row(&schema, &["70", "гранит"], "строка 5").unwrap_err();
        assert!(err.starts_with("строка 5, "), "{err}");
        assert!(
            err.contains("гранит") && err.contains("песок, глина"),
            "{err}"
        );

        let err = parse_row(&schema, &["ой", "песок"], "строка 2").unwrap_err();
        assert!(err.contains("колонка 'температура'"), "{err}");
        assert!(err.contains("ожидалось число"), "{err}");

        let err = parse_row(&schema, &["70"], "").unwrap_err();
        assert!(err.contains("ожидает 2 входов"), "{err}");
        assert!(err.contains("температура, материал"), "{err}");

        let err = parse_row(&schema, &["70", "  "], "строка 3").unwrap_err();
        assert!(err.contains("пустое значение"), "{err}");
    }

    #[test]
    fn parse_rows_reports_the_file_row_number() {
        let schema = schema_with_category();
        let rows = vec![
            vec!["70".to_string(), "песок".to_string()],
            vec!["60".to_string(), "мрамор".to_string()],
        ];
        // Метки строк приходят из файла: вторая строка данных — четвёртая в нём.
        let err = parse_rows(&schema, &rows, &[2, 4]).unwrap_err();
        assert!(err.starts_with("строка 4, "), "{err}");

        let ok = parse_rows(&schema, &rows[..1], &[2]).unwrap();
        assert_eq!(ok.dim(), (1, 2));
        assert_eq!(ok[[0, 1]], 0.0);
    }
}
