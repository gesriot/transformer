//! Схема данных: имена, роли и типы колонок.
//!
//! Две схемы, потому что они отвечают на разные вопросы.
//!
//! [`TableSchema`] описывает ИСХОДНУЮ таблицу: все её колонки, включая те, что
//! пользователь пометил как игнорируемые. Она живёт в слое импорта.
//!
//! [`ModelSchema`] описывает ВХОДЫ И ВЫХОДЫ модели в порядке тензоров.
//! Игнорируемым колонкам здесь места нет: после преобразования датасет состоит
//! только из inputs и outputs. Именно эта схема едет в `.tnum` и checkpoint,
//! чтобы прогноз и формулы говорили `temperature_C`, а не `x0`.
//!
//! Обе схемы валидируются при построении: имена уникальны и непусты, уровни
//! категорий уникальны, размерности сходятся. Поля закрыты — иначе инвариант
//! можно было бы нарушить после проверки.

use crate::encoders::FeatureSpec;
use std::collections::BTreeSet;

/// Роль колонки исходной таблицы.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRole {
    Input,
    Output,
    /// Колонка есть в файле, но в модель не идёт (комментарий, id, дубль).
    Ignore,
}

impl ColumnRole {
    pub fn label(self) -> &'static str {
        match self {
            ColumnRole::Input => "вход",
            ColumnRole::Output => "выход",
            ColumnRole::Ignore => "игнорировать",
        }
    }
}

/// Тип колонки.
///
/// `Categorical` хранит ПОДПИСИ уровней, а не только их количество: без них
/// прогноз и формулы вынуждены говорить кодами, а пользователь — помнить, что
/// «2» означало `sand`. Порядок подписей задаёт коды `0..levels.len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    Numeric,
    Categorical { levels: Vec<String> },
}

/// Описание одной колонки. Роль осмысленна для [`TableSchema`]; в
/// [`ModelSchema`] она гарантирована конструктором и служит защитой от
/// перепутанных местами списков.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    name: String,
    role: ColumnRole,
    ty: ColumnType,
    unit: Option<String>,
}

impl Column {
    pub fn numeric(name: impl Into<String>, role: ColumnRole) -> Result<Self, String> {
        Ok(Self {
            name: check_name(name.into())?,
            role,
            ty: ColumnType::Numeric,
            unit: None,
        })
    }

    pub fn categorical(
        name: impl Into<String>,
        role: ColumnRole,
        levels: Vec<String>,
    ) -> Result<Self, String> {
        let name = check_name(name.into())?;
        if levels.is_empty() {
            return Err(format!("колонка '{name}': категория без уровней"));
        }
        let mut seen = BTreeSet::new();
        let mut clean = Vec::with_capacity(levels.len());
        for level in levels {
            let level = level.trim().to_string();
            if level.is_empty() {
                return Err(format!("колонка '{name}': пустая подпись уровня"));
            }
            if !seen.insert(level.clone()) {
                return Err(format!("колонка '{name}': уровень '{level}' повторяется"));
            }
            clean.push(level);
        }
        Ok(Self {
            name,
            role,
            ty: ColumnType::Categorical { levels: clean },
            unit: None,
        })
    }

    /// Единица измерения — только подпись для отчётов, в вычислениях не
    /// участвует.
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        let unit = unit.into().trim().to_string();
        self.unit = if unit.is_empty() { None } else { Some(unit) };
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn role(&self) -> ColumnRole {
        self.role
    }
    pub fn ty(&self) -> &ColumnType {
        &self.ty
    }
    pub fn unit(&self) -> Option<&str> {
        self.unit.as_deref()
    }

    /// `Some(n)` для категориальной колонки, `None` для числовой.
    pub fn cardinality(&self) -> Option<usize> {
        match &self.ty {
            ColumnType::Numeric => None,
            ColumnType::Categorical { levels } => Some(levels.len()),
        }
    }

    /// Имя с единицей измерения для заголовков таблиц и отчётов.
    pub fn display_name(&self) -> String {
        match &self.unit {
            Some(u) => format!("{} [{u}]", self.name),
            None => self.name.clone(),
        }
    }

    fn feature_spec(&self) -> FeatureSpec {
        match self.cardinality() {
            None => FeatureSpec::Continuous,
            Some(cardinality) => FeatureSpec::Categorical { cardinality },
        }
    }

    /// Код уровня по подписи. Неизвестное значение — СТРОГАЯ ошибка: молча
    /// подставленный код дал бы правдоподобный, но бессмысленный прогноз.
    pub fn category_code(&self, value: &str) -> Result<usize, String> {
        match &self.ty {
            ColumnType::Numeric => Err(format!("колонка '{}' не категориальная", self.name)),
            ColumnType::Categorical { levels } => levels
                .iter()
                .position(|l| l == value.trim())
                .ok_or_else(|| {
                    format!(
                        "колонка '{}': неизвестный уровень '{value}'; известны: {}",
                        self.name,
                        levels.join(", ")
                    )
                }),
        }
    }

    /// Подпись уровня по коду — обратная операция для отчётов и прогноза.
    pub fn category_level(&self, code: usize) -> Result<&str, String> {
        match &self.ty {
            ColumnType::Numeric => Err(format!("колонка '{}' не категориальная", self.name)),
            ColumnType::Categorical { levels } => {
                levels.get(code).map(String::as_str).ok_or_else(|| {
                    format!(
                        "колонка '{}': код {code} вне диапазона 0..{}",
                        self.name,
                        levels.len()
                    )
                })
            }
        }
    }
}

fn check_name(name: String) -> Result<String, String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("имя колонки не может быть пустым".to_string());
    }
    Ok(name)
}

/// Имена уникальны в пределах всей схемы, включая игнорируемые колонки:
/// одинаковые имена делают диалог разметки и отчёты неоднозначными.
fn check_unique_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(format!("имя колонки '{name}' повторяется"));
        }
    }
    Ok(())
}

/// Выход регрессии обязан быть числом: категориальный таргет не имеет смысла
/// для MSE и R².
fn check_numeric_output(column: &Column) -> Result<(), String> {
    if column.role == ColumnRole::Output && column.cardinality().is_some() {
        return Err(format!(
            "колонка '{}': категориальный выход не поддерживается (регрессия)",
            column.name
        ));
    }
    Ok(())
}

/// Схема исходной таблицы: все колонки в порядке файла.
///
/// Это РЕЗУЛЬТАТ разметки, а не её промежуточное состояние: незавершённый
/// выбор ролей в интерфейсе `TableSchema`-ой не является.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSchema {
    columns: Vec<Column>,
}

impl TableSchema {
    pub fn new(columns: Vec<Column>) -> Result<Self, String> {
        if columns.is_empty() {
            return Err("таблица без колонок".to_string());
        }
        check_unique_names(columns.iter().map(Column::name))?;
        for column in &columns {
            check_numeric_output(column)?;
        }
        if !columns.iter().any(|c| c.role == ColumnRole::Input) {
            return Err("нужен хотя бы один вход".to_string());
        }
        if !columns.iter().any(|c| c.role == ColumnRole::Output) {
            return Err("нужен хотя бы один выход".to_string());
        }
        Ok(Self { columns })
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Индексы колонок таблицы с данной ролью — порядок сохраняется, поэтому
    /// по ним можно резать строки файла.
    pub fn indices(&self, role: ColumnRole) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.role == role)
            .map(|(i, _)| i)
            .collect()
    }

    /// Схема модели: игнорируемые колонки отбрасываются, порядок остальных
    /// сохраняется.
    pub fn to_model_schema(&self) -> Result<ModelSchema, String> {
        let pick = |role: ColumnRole| -> Vec<Column> {
            self.columns
                .iter()
                .filter(|c| c.role == role)
                .cloned()
                .collect()
        };
        ModelSchema::new(pick(ColumnRole::Input), pick(ColumnRole::Output))
    }
}

/// Схема модели: входы и выходы в порядке тензоров.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSchema {
    inputs: Vec<Column>,
    outputs: Vec<Column>,
}

impl ModelSchema {
    pub fn new(inputs: Vec<Column>, outputs: Vec<Column>) -> Result<Self, String> {
        if inputs.is_empty() {
            return Err("нужен хотя бы один вход".to_string());
        }
        if outputs.is_empty() {
            return Err("нужен хотя бы один выход".to_string());
        }
        for c in &inputs {
            if c.role != ColumnRole::Input {
                return Err(format!(
                    "колонка '{}' в списке входов имеет роль {}",
                    c.name,
                    c.role.label()
                ));
            }
        }
        for c in &outputs {
            if c.role != ColumnRole::Output {
                return Err(format!(
                    "колонка '{}' в списке выходов имеет роль {}",
                    c.name,
                    c.role.label()
                ));
            }
            check_numeric_output(c)?;
        }
        check_unique_names(inputs.iter().chain(outputs.iter()).map(Column::name))?;
        Ok(Self { inputs, outputs })
    }

    /// Схема без имён: `x0…xN → y0…yM`, все колонки числовые.
    ///
    /// Нужна для встроенных чёрных ящиков и других заведомо числовых данных.
    /// Для старых TRNUM1/checkpoint-ов нужен [`Self::synthetic_from_specs`]:
    /// они могут хранить категориальные `K:n`, даже если не хранят их подписи.
    pub fn synthetic(n_inputs: usize, n_outputs: usize) -> Result<Self, String> {
        Self::synthetic_from_specs(&vec![FeatureSpec::Continuous; n_inputs], n_outputs)
    }

    /// Fallback-схема для старого формата, где есть типы и cardinality, но нет
    /// имён и подписей уровней. Для категорий подписями становятся известные коды
    /// `"0"..."n-1"`: это единственная информация, которую старый артефакт может восстановить.
    pub fn synthetic_from_specs(specs: &[FeatureSpec], n_outputs: usize) -> Result<Self, String> {
        let inputs = specs
            .iter()
            .enumerate()
            .map(|(i, spec)| match *spec {
                FeatureSpec::Continuous => Column::numeric(format!("x{i}"), ColumnRole::Input),
                FeatureSpec::Categorical { cardinality } => Column::categorical(
                    format!("x{i}"),
                    ColumnRole::Input,
                    (0..cardinality).map(|code| code.to_string()).collect(),
                ),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = (0..n_outputs)
            .map(|j| Column::numeric(format!("y{j}"), ColumnRole::Output))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(inputs, outputs)
    }

    pub fn inputs(&self) -> &[Column] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[Column] {
        &self.outputs
    }
    pub fn n_inputs(&self) -> usize {
        self.inputs.len()
    }
    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }

    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(Column::name).collect()
    }
    pub fn output_names(&self) -> Vec<&str> {
        self.outputs.iter().map(Column::name).collect()
    }

    /// Спецификации признаков для энкодеров — единственный мост от схемы к
    /// существующему коду модели.
    pub fn feature_specs(&self) -> Vec<FeatureSpec> {
        self.inputs.iter().map(Column::feature_spec).collect()
    }

    /// Совместима ли модель с этим набором данных.
    ///
    /// Сравниваются имена, порядок, типы и подписи уровней — всё, что влияет на
    /// смысл числа в тензоре. Единицы измерения НЕ сравниваются: это подпись
    /// для отчётов, и модель от неё не зависит.
    ///
    /// Причина несовместимости возвращается текстом: «не подходит» без
    /// объяснения заставляет угадывать, что именно разошлось.
    pub fn compatibility_with(&self, data: &ModelSchema) -> Result<(), String> {
        fn compare(
            singular: &str,
            plural: &str,
            model: &[Column],
            data: &[Column],
        ) -> Result<(), String> {
            if model.len() != data.len() {
                return Err(format!(
                    "{plural}: у модели {}, у данных {}",
                    model.len(),
                    data.len()
                ));
            }
            for (i, (m, d)) in model.iter().zip(data.iter()).enumerate() {
                if m.name() != d.name() {
                    return Err(format!(
                        "{singular} {}: модель ждёт '{}', в данных '{}'",
                        i + 1,
                        m.name(),
                        d.name()
                    ));
                }
                match (m.ty(), d.ty()) {
                    (ColumnType::Numeric, ColumnType::Numeric) => {}
                    (
                        ColumnType::Categorical { levels: ml },
                        ColumnType::Categorical { levels: dl },
                    ) => {
                        if ml != dl {
                            return Err(format!(
                                "{singular} '{}': уровни категории различаются ({} против {})",
                                m.name(),
                                ml.join(", "),
                                dl.join(", ")
                            ));
                        }
                    }
                    (ColumnType::Numeric, ColumnType::Categorical { .. }) => {
                        return Err(format!(
                            "{singular} '{}': у модели число, в данных категория",
                            m.name()
                        ))
                    }
                    (ColumnType::Categorical { .. }, ColumnType::Numeric) => {
                        return Err(format!(
                            "{singular} '{}': у модели категория, в данных число",
                            m.name()
                        ))
                    }
                }
            }
            Ok(())
        }
        compare("вход", "входов", &self.inputs, &data.inputs)?;
        compare("выход", "выходов", &self.outputs, &data.outputs)
    }

    /// Схема обязана совпадать с формой данных: несовпадение означает, что
    /// разметка и файл разъехались.
    pub fn check_dims(&self, n_inputs: usize, n_outputs: usize) -> Result<(), String> {
        if self.n_inputs() != n_inputs || self.n_outputs() != n_outputs {
            return Err(format!(
                "схема описывает {} входов и {} выходов, а данные — {n_inputs} и {n_outputs}",
                self.n_inputs(),
                self.n_outputs()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str) -> Column {
        Column::numeric(name, ColumnRole::Input).unwrap()
    }
    fn output(name: &str) -> Column {
        Column::numeric(name, ColumnRole::Output).unwrap()
    }
    fn ignored(name: &str) -> Column {
        Column::numeric(name, ColumnRole::Ignore).unwrap()
    }
    fn material() -> Column {
        Column::categorical(
            "material",
            ColumnRole::Input,
            vec!["sand".into(), "clay".into(), "peat".into()],
        )
        .unwrap()
    }

    #[test]
    fn synthetic_names_are_x_and_y() {
        let s = ModelSchema::synthetic(3, 2).unwrap();
        assert_eq!(s.input_names(), vec!["x0", "x1", "x2"]);
        assert_eq!(s.output_names(), vec!["y0", "y1"]);
        assert_eq!(s.feature_specs(), vec![FeatureSpec::Continuous; 3]);
        assert!(ModelSchema::synthetic(0, 1).is_err());
        assert!(ModelSchema::synthetic(1, 0).is_err());
    }

    #[test]
    fn legacy_specs_keep_categorical_inputs() {
        let specs = vec![
            FeatureSpec::Continuous,
            FeatureSpec::Categorical { cardinality: 3 },
        ];
        let schema = ModelSchema::synthetic_from_specs(&specs, 1).unwrap();

        assert_eq!(schema.input_names(), vec!["x0", "x1"]);
        assert_eq!(schema.output_names(), vec!["y0"]);
        assert_eq!(schema.feature_specs(), specs);
        assert_eq!(schema.inputs()[1].category_level(2).unwrap(), "2");
        assert!(ModelSchema::synthetic_from_specs(
            &[FeatureSpec::Categorical { cardinality: 0 }],
            1
        )
        .is_err());
    }

    #[test]
    fn feature_specs_carry_cardinality() {
        let s = ModelSchema::new(
            vec![input("temperature_C"), material()],
            vec![output("moisture")],
        )
        .unwrap();
        assert_eq!(
            s.feature_specs(),
            vec![
                FeatureSpec::Continuous,
                FeatureSpec::Categorical { cardinality: 3 }
            ]
        );
        assert_eq!(s.inputs()[1].cardinality(), Some(3));
        assert_eq!(s.inputs()[0].cardinality(), None);
    }

    #[test]
    fn names_must_be_unique_and_non_empty() {
        assert!(Column::numeric("   ", ColumnRole::Input).is_err());
        // Имя тримится, поэтому «x » и «x» — одно и то же имя.
        let err = TableSchema::new(vec![input("x "), input("x"), output("y")]).unwrap_err();
        assert!(err.contains("повторяется"), "{err}");
        // Дубль с игнорируемой колонкой тоже отвергается.
        assert!(TableSchema::new(vec![input("a"), ignored("a"), output("y")]).is_err());
        // Вход и выход не могут называться одинаково.
        assert!(ModelSchema::new(vec![input("a")], vec![output("a")]).is_err());
    }

    #[test]
    fn levels_must_be_unique_and_non_empty() {
        let dup = Column::categorical("m", ColumnRole::Input, vec!["sand".into(), "sand".into()]);
        assert!(dup.unwrap_err().contains("повторяется"));
        let dup_after_trim =
            Column::categorical("m", ColumnRole::Input, vec!["sand".into(), " sand ".into()]);
        assert!(dup_after_trim.unwrap_err().contains("повторяется"));
        assert!(Column::categorical("m", ColumnRole::Input, vec![" ".into()]).is_err());
        assert!(Column::categorical("m", ColumnRole::Input, vec![]).is_err());
    }

    #[test]
    fn names_and_levels_accept_spaces_quotes_and_unicode() {
        let column = Column::categorical(
            "  тип материала ",
            ColumnRole::Input,
            vec!["мелкий песок".into(), "глина \"А\"".into()],
        )
        .unwrap()
        .with_unit("марка №");

        assert_eq!(column.name(), "тип материала");
        assert_eq!(column.category_code("глина \"А\"").unwrap(), 1);
        assert_eq!(column.category_level(0).unwrap(), "мелкий песок");
        assert_eq!(column.display_name(), "тип материала [марка №]");
    }

    #[test]
    fn categorical_output_is_rejected() {
        let target =
            Column::categorical("grade", ColumnRole::Output, vec!["a".into(), "b".into()]).unwrap();
        let err = ModelSchema::new(vec![input("x")], vec![target.clone()]).unwrap_err();
        assert!(err.contains("категориальный выход"), "{err}");
        assert!(TableSchema::new(vec![input("x"), target]).is_err());
    }

    #[test]
    fn table_schema_needs_input_and_output() {
        assert!(TableSchema::new(vec![]).is_err());
        assert!(TableSchema::new(vec![input("x"), ignored("z")]).is_err());
        assert!(TableSchema::new(vec![output("y"), ignored("z")]).is_err());
    }

    #[test]
    fn model_schema_drops_ignored_and_keeps_order() {
        let table = TableSchema::new(vec![
            input("temperature_C"),
            ignored("operator_note"),
            material(),
            output("moisture"),
            ignored("row_id"),
        ])
        .unwrap();
        assert_eq!(table.columns().len(), 5);
        assert_eq!(table.indices(ColumnRole::Input), vec![0, 2]);
        assert_eq!(table.indices(ColumnRole::Ignore), vec![1, 4]);

        let model = table.to_model_schema().unwrap();
        assert_eq!(model.input_names(), vec!["temperature_C", "material"]);
        assert_eq!(model.output_names(), vec!["moisture"]);
        assert_eq!(model.n_inputs(), 2);
        assert_eq!(model.n_outputs(), 1);
    }

    #[test]
    fn model_schema_rejects_wrong_roles() {
        // Списки, перепутанные местами, не должны проходить молча.
        assert!(ModelSchema::new(vec![output("y")], vec![output("z")]).is_err());
        assert!(ModelSchema::new(vec![input("x")], vec![input("w")]).is_err());
        assert!(ModelSchema::new(vec![ignored("x")], vec![output("y")]).is_err());
    }

    #[test]
    fn compatibility_ignores_units_but_not_meaning() {
        let model = ModelSchema::new(
            vec![input("temperature").with_unit("°C"), material()],
            vec![output("moisture")],
        )
        .unwrap();

        // Единицы — подпись для отчётов: модель от них не зависит.
        let other_units = ModelSchema::new(
            vec![input("temperature").with_unit("K"), material()],
            vec![output("moisture").with_unit("%")],
        )
        .unwrap();
        assert!(model.compatibility_with(&other_units).is_ok());

        // Имя входа другое — данные описывают другую величину.
        let renamed =
            ModelSchema::new(vec![input("temp"), material()], vec![output("moisture")]).unwrap();
        let err = model.compatibility_with(&renamed).unwrap_err();
        assert!(
            err.contains("'temperature'") && err.contains("'temp'"),
            "{err}"
        );

        // Порядок входов важен: это порядок колонок тензора.
        let swapped = ModelSchema::new(
            vec![material(), input("temperature")],
            vec![output("moisture")],
        )
        .unwrap();
        assert!(model.compatibility_with(&swapped).is_err());

        // Число входов.
        let narrow =
            ModelSchema::new(vec![input("temperature")], vec![output("moisture")]).unwrap();
        let err = model.compatibility_with(&narrow).unwrap_err();
        assert!(err.contains("входов"), "{err}");

        // Тип колонки.
        let numeric_material = ModelSchema::new(
            vec![input("temperature"), input("material")],
            vec![output("moisture")],
        )
        .unwrap();
        let err = model.compatibility_with(&numeric_material).unwrap_err();
        assert!(err.contains("категория"), "{err}");

        // Выходы имеют ту же смысловую проверку, что и входы.
        let renamed_output = ModelSchema::new(
            vec![input("temperature"), material()],
            vec![output("humidity")],
        )
        .unwrap();
        let err = model.compatibility_with(&renamed_output).unwrap_err();
        assert!(err.contains("выход 1") && err.contains("humidity"), "{err}");
    }

    #[test]
    fn compatibility_compares_category_levels() {
        let model = ModelSchema::new(vec![material()], vec![output("y")]).unwrap();
        let other_levels = ModelSchema::new(
            vec![Column::categorical(
                "material",
                ColumnRole::Input,
                vec!["sand".into(), "clay".into()],
            )
            .unwrap()],
            vec![output("y")],
        )
        .unwrap();
        let err = model.compatibility_with(&other_levels).unwrap_err();
        assert!(err.contains("уровни категории"), "{err}");
        assert!(model.compatibility_with(&model.clone()).is_ok());
    }

    #[test]
    fn dims_must_match_data() {
        let s = ModelSchema::synthetic(3, 2).unwrap();
        assert!(s.check_dims(3, 2).is_ok());
        let err = s.check_dims(4, 2).unwrap_err();
        assert!(err.contains("3") && err.contains("4"), "{err}");
        assert!(s.check_dims(3, 1).is_err());
    }

    #[test]
    fn category_codes_round_trip_and_reject_unknown() {
        let m = material();
        assert_eq!(m.category_code("sand").unwrap(), 0);
        assert_eq!(m.category_code(" peat ").unwrap(), 2);
        assert_eq!(m.category_level(1).unwrap(), "clay");

        let err = m.category_code("granite").unwrap_err();
        assert!(err.contains("granite"), "{err}");
        assert!(err.contains("sand, clay, peat"), "известные уровни: {err}");
        assert!(m.category_level(9).is_err());

        // Числовая колонка кодов не имеет.
        assert!(input("temperature_C").category_code("sand").is_err());
    }

    #[test]
    fn unit_is_a_label_only() {
        let c = input("temperature").with_unit("°C");
        assert_eq!(c.name(), "temperature");
        assert_eq!(c.unit(), Some("°C"));
        assert_eq!(c.display_name(), "temperature [°C]");
        // Пустая единица — это отсутствие единицы, а не пустая подпись.
        assert_eq!(input("x").with_unit("  ").unit(), None);
        assert_eq!(input("x").display_name(), "x");
    }
}
