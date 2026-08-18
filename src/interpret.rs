//! Профиль интерпретируемой KAN: регуляризация → прунинг → fine-tune →
//! структурное сжатие.
//!
//! Конвейер почти всегда запускают целиком, поэтому у него есть готовый профиль
//! — и одновременно каждый его параметр можно переопределить. Профиль задаёт
//! значения по умолчанию, явные флаги их перекрывают.
//!
//! Профиль версионируется: «интерпретируемая KAN» не должна означать разное в
//! разных сборках. Разрешённые значения сохраняются в checkpoint, чтобы через
//! полгода было видно, какой именно конвейер получил эту модель.
//!
//! Символьные формулы в профиль НЕ входят: они считаются лениво, по запросу, и
//! обучение не меняют.

use crate::data::{Normalizer, NumericDataset};
use crate::kan::CompactReport;
use crate::numeric_model::NumericModel;
use crate::train::{evaluate_surrogate, train_surrogate, TrainConfig};

/// Версия профиля. Меняется, если меняется смысл или состав параметров.
pub const INTERPRET_PROFILE_VERSION: u32 = 1;

/// Разрешённые параметры конвейера — то, что реально применяется к модели.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpretProfile {
    pub version: u32,
    /// Коэффициент activation-L1 при обучении; 0 — регуляризации нет.
    pub l1: f32,
    /// Относительный порог важности ребра; `None` — прунинг не выполняется.
    pub prune: Option<f32>,
    /// Эпохи дообучения после прунинга (с λ=0).
    pub finetune_epochs: usize,
    /// Физически удалить мёртвые скрытые узлы после прунинга.
    pub compact: bool,
}

impl InterpretProfile {
    /// Готовый профиль v1 — тот самый обучающий конвейер, который в
    /// документации запускают четырьмя флагами подряд.
    pub fn v1() -> Self {
        Self {
            version: INTERPRET_PROFILE_VERSION,
            l1: 1e-3,
            prune: Some(0.05),
            finetune_epochs: 20,
            compact: true,
        }
    }

    /// Пустая основа: конвейер собирается только из явных флагов.
    fn bare() -> Self {
        Self {
            version: INTERPRET_PROFILE_VERSION,
            l1: 0.0,
            prune: None,
            finetune_epochs: 10,
            compact: false,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != INTERPRET_PROFILE_VERSION {
            return Err(format!(
                "версия профиля {} не поддерживается (ожидалась {INTERPRET_PROFILE_VERSION})",
                self.version
            ));
        }
        if !self.l1.is_finite() || self.l1 < 0.0 {
            return Err("l1 должен быть конечным и >= 0".to_string());
        }
        if let Some(p) = self.prune {
            if !p.is_finite() || !(0.0..1.0).contains(&p) {
                return Err("prune (отн. порог важности) должен быть в [0, 1)".to_string());
            }
        }
        if self.finetune_epochs == 0 {
            return Err("finetune-epochs должен быть >= 1".to_string());
        }
        Ok(())
    }

    /// Строка для отчёта и лога: профиль не должен быть непрозрачной галочкой.
    pub fn describe(&self) -> String {
        let prune = match self.prune {
            Some(p) => format!("prune {p}, fine-tune {} эпох", self.finetune_epochs),
            None => "без прунинга".to_string(),
        };
        let compact = if self.compact {
            ", структурное сжатие"
        } else {
            ""
        };
        format!("v{}: L1 {}, {prune}{compact}", self.version, self.l1)
    }
}

/// Явные переопределения параметров профиля.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InterpretOverrides {
    pub l1: Option<f32>,
    pub prune: Option<f32>,
    pub finetune_epochs: Option<usize>,
    pub compact: Option<bool>,
}

impl InterpretOverrides {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Собрать конвейер из профиля и переопределений.
///
/// `Ok(None)` — конвейер не запрашивали вовсе. Иначе профиль задаёт значения по
/// умолчанию, а явные переопределения их перекрывают: `--interpret --kan-prune
/// 0.1` означает «профиль, но с другим порогом».
pub fn resolve(
    use_profile: bool,
    overrides: &InterpretOverrides,
) -> Result<Option<InterpretProfile>, String> {
    if !use_profile && overrides.is_empty() {
        return Ok(None);
    }
    let mut profile = if use_profile {
        InterpretProfile::v1()
    } else {
        InterpretProfile::bare()
    };
    if let Some(l1) = overrides.l1 {
        profile.l1 = l1;
    }
    if let Some(prune) = overrides.prune {
        profile.prune = Some(prune);
    }
    if let Some(epochs) = overrides.finetune_epochs {
        profile.finetune_epochs = epochs;
    }
    if let Some(compact) = overrides.compact {
        profile.compact = compact;
    }
    // Дообучение без прунинга нечего дообучать: это ошибка в команде, а не
    // безобидная опечатка.
    if overrides.finetune_epochs.is_some() && profile.prune.is_none() {
        return Err("finetune-epochs имеет смысл только вместе с prune".to_string());
    }
    profile.validate()?;
    Ok(Some(profile))
}

// --- применение конвейера ---

/// Отчёт о применении конвейера: числа, а не текст.
///
/// Печатать их — дело вызывающего: CLI выводит строки, GUI рисует. Иначе один
/// и тот же конвейер расходился бы между поверхностями в мелочах.
#[derive(Clone, Debug, PartialEq)]
pub struct InterpretReport {
    pub profile: InterpretProfile,
    /// Активные рёбра по слоям после прунинга: (активных, всего).
    pub per_layer: Vec<(usize, usize)>,
    /// Активные рёбра всего после прунинга, до возможного сжатия. Когда
    /// прунинг выполнялся, это точная сумма `per_layer`; изменение физической
    /// топологии отдельно описывает `compaction`.
    pub active_edges: (usize, usize),
    /// R² на контрольном наборе по фазам. `None` — контрольного набора не было
    /// (у финальной модели его и не может быть: test тратить нельзя).
    pub r2_before: Option<f32>,
    pub r2_after_prune: Option<f32>,
    pub r2_after_finetune: Option<f32>,
    /// Структурное сжатие: узлы и параметры до/после.
    pub compaction: Option<CompactReport>,
    pub r2_after_compact: Option<f32>,
}

impl InterpretReport {
    /// Параметров стало меньше — сжатие реально что-то удалило.
    pub fn params_removed(&self) -> usize {
        self.compaction
            .map_or(0, |c| c.params_before.saturating_sub(c.params_after))
    }
}

/// Включить activation-L1 ДО обучения: иначе регуляризация не участвует в нём.
pub fn apply_l1(model: &NumericModel, profile: &InterpretProfile) -> Result<(), String> {
    profile.validate()?;
    let kan = model
        .as_kan()
        .ok_or_else(|| "activation-L1 применим только к KAN".to_string())?;
    // Устанавливаем и ноль: повторное применение другого профиля обязано
    // сбросить ранее включённую регуляризацию, а не зависеть от истории модели.
    kan.set_l1_lambda(profile.l1);
    Ok(())
}

/// Конвейер после обучения: прунинг → fine-tune → структурное сжатие.
///
/// `eval` — набор для отчёта о влиянии (validation в фазе разработки). Его
/// отсутствие меняет только отчёт, но не сами операции: модель обязана
/// получиться той же.
pub fn run_pipeline(
    model: &mut NumericModel,
    train: &NumericDataset,
    eval: Option<&NumericDataset>,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    tcfg: &TrainConfig,
    profile: &InterpretProfile,
) -> Result<InterpretReport, String> {
    profile.validate()?;
    if model.as_kan().is_none() {
        return Err("конвейер интерпретации применим только к KAN".to_string());
    }
    let measure =
        |model: &NumericModel| eval.map(|d| evaluate_surrogate(model, d, in_norm, out_norm).r2);

    let r2_before = measure(model);
    let mut per_layer = Vec::new();
    let mut r2_after_prune = None;
    let mut r2_after_finetune = None;

    if let Some(threshold) = profile.prune {
        let kan = model.as_kan().expect("проверено выше");
        let calibration = in_norm.transform(&train.inputs);
        per_layer = kan.prune_edges(threshold, &calibration).per_layer;
        r2_after_prune = measure(model);

        // Дообучение идёт с λ=0: регуляризация своё дело уже сделала.
        kan.set_l1_lambda(0.0);
        let ft_cfg = TrainConfig {
            epochs: profile.finetune_epochs,
            ..tcfg.clone()
        };
        train_surrogate(model, train, in_norm, out_norm, &ft_cfg);
        r2_after_finetune = measure(model);
    }

    let active_edges = model.as_kan().expect("проверено выше").active_edges();
    let mut compaction = None;
    let mut r2_after_compact = None;
    if profile.compact {
        compaction = Some(model.as_kan_mut().expect("проверено выше").compact());
        r2_after_compact = measure(model);
    }

    Ok(InterpretReport {
        profile: *profile,
        per_layer,
        active_edges,
        r2_before,
        r2_after_prune,
        r2_after_finetune,
        compaction,
        r2_after_compact,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::blackbox;
    use crate::config::ModelConfig;
    use crate::encoders::{FeatureSpec, ValueEncoderConfig};
    use crate::numeric_model::{KanConfig, ModelKind, NumericConfig};
    use crate::split::SplitPlan;
    use crate::train::{fit_normalizers, LrSchedule};

    fn kan_config() -> NumericConfig {
        NumericConfig {
            kind: ModelKind::Kan,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 16,
            mlp_layers: 1,
            kan: KanConfig {
                width: 6,
                layers: 2,
                grid: 5,
            },
        }
    }

    fn train_cfg() -> TrainConfig {
        TrainConfig {
            epochs: 3,
            batch_size: 32,
            lr: 3e-3,
            seed: 0,
            schedule: LrSchedule::Constant,
        }
    }

    /// Прогон конвейера «как поверхность»: собрать модель, обучить, применить.
    fn run_surface(profile: &InterpretProfile, with_eval: bool) -> (NumericModel, InterpretReport) {
        let data = blackbox::sum().generate(96, 0);
        let prepared = SplitPlan::default().prepare(&data).unwrap();
        let (train, val) = prepared.search.fold(0).unwrap();
        let specs = vec![FeatureSpec::Continuous; 2];
        let (in_norm, out_norm) = fit_normalizers(&train, &specs);

        crate::init::set_init_seed(0);
        let nc = kan_config();
        let mut model = nc.build(&specs, 1);
        apply_l1(&model, profile).unwrap();
        crate::train::train_surrogate(&model, &train, &in_norm, &out_norm, &train_cfg());
        let report = run_pipeline(
            &mut model,
            &train,
            with_eval.then_some(&val),
            &in_norm,
            &out_norm,
            &train_cfg(),
            profile,
        )
        .unwrap();
        (model, report)
    }

    fn model_state(model: &NumericModel) -> (Vec<ndarray::ArrayD<f32>>, Vec<ndarray::ArrayD<f32>>) {
        let parameters = model.parameters().into_iter().map(|p| p.data()).collect();
        let masks = model
            .kan_masks()
            .expect("тестовая модель KAN")
            .into_iter()
            .map(|m| m.data())
            .collect();
        (parameters, masks)
    }

    /// Главный критерий общего конвейера: одна и та же модель с одним профилем
    /// даёт одинаковые маски и одинаковое число активных рёбер независимо от
    /// того, кто его запустил — CLI или GUI.
    #[test]
    fn same_model_and_profile_give_the_same_masks() {
        let profile = InterpretProfile::v1();
        let (cli_model, cli) = run_surface(&profile, true);
        let (gui_model, gui) = run_surface(&profile, true);
        assert_eq!(cli.active_edges, gui.active_edges);
        assert_eq!(cli.per_layer, gui.per_layer);
        assert_eq!(cli.compaction, gui.compaction);
        assert_eq!(model_state(&cli_model), model_state(&gui_model));

        // Наличие контрольного набора меняет ТОЛЬКО отчёт, но не модель.
        let (without_eval_model, without_eval) = run_surface(&profile, false);
        assert_eq!(cli.active_edges, without_eval.active_edges);
        assert_eq!(cli.per_layer, without_eval.per_layer);
        assert_eq!(cli.compaction, without_eval.compaction);
        assert_eq!(model_state(&cli_model), model_state(&without_eval_model));
        assert!(without_eval.r2_before.is_none());
        assert!(cli.r2_before.is_some());
    }

    #[test]
    fn report_carries_numbers_for_every_phase() {
        let profile = InterpretProfile {
            prune: Some(0.9),
            ..InterpretProfile::v1()
        };
        let (_, report) = run_surface(&profile, true);
        assert_eq!(report.profile, profile);
        assert!(!report.per_layer.is_empty());
        assert_eq!(
            report.active_edges,
            report
                .per_layer
                .iter()
                .fold((0, 0), |(a, t), (la, lt)| (a + la, t + lt))
        );
        // Жёсткий порог обязан что-то отсечь.
        assert!(report.active_edges.0 < report.active_edges.1);
        assert!(report.r2_before.is_some());
        assert!(report.r2_after_prune.is_some());
        assert!(report.r2_after_finetune.is_some());
        // Сжатие включено профилем v1.
        assert!(report.compaction.is_some());
        assert!(report.r2_after_compact.is_some());
    }

    #[test]
    fn profile_without_prune_only_reports_edges() {
        let profile = InterpretProfile {
            prune: None,
            compact: false,
            ..InterpretProfile::v1()
        };
        let (_, report) = run_surface(&profile, true);
        assert!(report.per_layer.is_empty());
        assert!(report.r2_after_prune.is_none());
        assert!(report.compaction.is_none());
        // Ни одно ребро не отсечено.
        assert_eq!(report.active_edges.0, report.active_edges.1);
    }

    #[test]
    fn pipeline_rejects_non_kan_models() {
        let specs = vec![FeatureSpec::Continuous; 2];
        let nc = NumericConfig {
            kind: ModelKind::Mlp,
            ..kan_config()
        };
        let mut model = nc.build(&specs, 1);
        let data = blackbox::sum().generate(32, 0);
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let err = run_pipeline(
            &mut model,
            &data,
            None,
            &in_norm,
            &out_norm,
            &train_cfg(),
            &InterpretProfile::v1(),
        )
        .unwrap_err();
        assert!(err.contains("только к KAN"), "{err}");
        assert!(apply_l1(&model, &InterpretProfile::v1()).is_err());
        assert!(apply_l1(
            &model,
            &InterpretProfile {
                l1: 0.0,
                ..InterpretProfile::v1()
            }
        )
        .is_err());
        assert!(apply_l1(
            &kan_config().build(&specs, 1),
            &InterpretProfile {
                version: 99,
                ..InterpretProfile::v1()
            }
        )
        .is_err());
    }

    #[test]
    fn nothing_requested_means_no_pipeline() {
        assert_eq!(
            resolve(false, &InterpretOverrides::default()).unwrap(),
            None
        );
    }

    #[test]
    fn profile_alone_gives_the_documented_pipeline() {
        let p = resolve(true, &InterpretOverrides::default())
            .unwrap()
            .expect("профиль запрошен");
        assert_eq!(p, InterpretProfile::v1());
        assert_eq!(p.version, INTERPRET_PROFILE_VERSION);
        assert!(p.prune.is_some() && p.compact);
        // Описание показывает реальные значения, а не название профиля.
        let text = p.describe();
        assert!(text.contains("v1"), "{text}");
        assert!(text.contains("0.001"), "{text}");
        assert!(text.contains("20 эпох"), "{text}");
    }

    #[test]
    fn explicit_flags_override_the_profile() {
        let p = resolve(
            true,
            &InterpretOverrides {
                prune: Some(0.2),
                compact: Some(false),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(p.prune, Some(0.2));
        assert!(!p.compact);
        // Не переопределённое остаётся профильным.
        assert_eq!(p.l1, InterpretProfile::v1().l1);
        assert_eq!(p.finetune_epochs, InterpretProfile::v1().finetune_epochs);
    }

    #[test]
    fn flags_without_profile_build_only_what_was_asked() {
        let only_l1 = resolve(
            false,
            &InterpretOverrides {
                l1: Some(1e-4),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(only_l1.l1, 1e-4);
        // Без явного prune конвейер не прунит и не сжимает.
        assert_eq!(only_l1.prune, None);
        assert!(!only_l1.compact);
    }

    #[test]
    fn finetune_without_prune_is_an_error() {
        let err = resolve(
            false,
            &InterpretOverrides {
                finetune_epochs: Some(5),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.contains("prune"), "{err}");
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(InterpretProfile {
            version: 99,
            ..InterpretProfile::v1()
        }
        .validate()
        .is_err());

        let bad_l1 = resolve(
            false,
            &InterpretOverrides {
                l1: Some(-1.0),
                ..Default::default()
            },
        );
        assert!(bad_l1.is_err());

        for prune in [-0.1, 1.0, f32::NAN] {
            assert!(
                resolve(
                    false,
                    &InterpretOverrides {
                        prune: Some(prune),
                        ..Default::default()
                    }
                )
                .is_err(),
                "порог {prune} должен отвергаться"
            );
        }

        assert!(resolve(
            true,
            &InterpretOverrides {
                finetune_epochs: Some(0),
                ..Default::default()
            }
        )
        .is_err());
    }
}
