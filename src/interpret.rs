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

#[cfg(test)]
mod tests {
    use super::*;

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
