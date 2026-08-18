#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

//! CLI: обучение surrogate-модели на чёрном ящике / числовой таблице, либо char-LM.
//!
//! Использование:
//!   transformer numeric <blackbox> [--epochs N] [--model out.bin] [конфиг-флаги]
//!   transformer numeric-file <data> [--epochs N] [--model out.bin] [флаги]
//!   transformer text <file.txt> [steps]
//!   transformer predict <model.bin> <v1> <v2> ...
//!
//! Конфиг-флаги: --d-model --heads --layers --enc-layers --dec-layers --d-ff
//!               --lr --batch-size --seed
//! Старая позиционная форма (`numeric-file f.tnum 40 out.bin`) тоже работает.

use ndarray::{Array2, Ix2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use transformer::blackbox;
use transformer::config::ModelConfig;
use transformer::data::{Normalizer, NumericDataset, TextDataset};
use transformer::diagnostics;
use transformer::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use transformer::epoch_sweep;
use transformer::generate::generate;
use transformer::init::set_init_seed;
use transformer::interpret::{self, InterpretOverrides, InterpretProfile};
use transformer::metrics::{evaluate, Metrics};
use transformer::numeric_model::{
    validate_numeric, KanConfig, ModelKind, NumericConfig, NumericModel,
};
use transformer::schema::ModelSchema;
use transformer::serialize::{calibration_sample, load_numeric_full, save_numeric};
use transformer::split::{
    SplitPlan, DEFAULT_DATA_SEED, DEFAULT_FINAL_INIT_SEED, DEFAULT_SPLIT_SEED,
};
use transformer::sweep as sweep_core;
use transformer::symbolic;
use transformer::tensor::Tensor;
use transformer::textmodel::TextModel;
use transformer::tnum::{
    infer_prepare_spec_from_path, parse_categorical, read_numeric_source, table_path_to_tnum,
    Delimiter, PrepareSpec,
};
use transformer::train::{
    evaluate_surrogate, predict_dataset, train_surrogate, train_text, validate_train, LrSchedule,
    TextTrainConfig, TrainConfig,
};
use transformer::training::{
    evaluate_on, run_training, Dataset, Phase, TrainedModel, TrainingSetup,
};

/// Разобранные аргументы: `--key value` во flags, остальное — позиционные.
struct Flags {
    flags: HashMap<String, String>,
    positionals: Vec<String>,
}

/// Допустимые флаги для numeric / numeric-file. Неизвестный флаг отвергается,
/// чтобы опечатка не привела к молчаливому обучению дефолтной конфигурации.
const NUMERIC_FLAGS: &[&str] = &[
    "epochs",
    "model",
    "model-kind",
    "d-model",
    "heads",
    "layers",
    "enc-layers",
    "dec-layers",
    "d-ff",
    "value-encoder",
    "fourier-bands",
    "fourier-scale",
    "mlp-width",
    "mlp-layers",
    "kan-width",
    "kan-layers",
    "kan-grid",
    "kan-l1",
    "kan-prune",
    "kan-finetune-epochs",
    "lr",
    "batch-size",
    "seed",
    "scheduler",
    "warmup",
    "min-lr-ratio",
];

/// Булевы флаги (без значения).
const NUMERIC_BOOL_FLAGS: &[&str] = &["diagnose", "kan-symbolic", "kan-compact", "interpret"];

/// Флаги подкоманды prepare (таблица -> .tnum).
const PREPARE_FLAGS: &[&str] = &["inputs", "outputs", "delimiter", "categorical"];
const PREPARE_BOOL_FLAGS: &[&str] = &["has-header"];

/// Флаги подкоманды sweep (оси — CSV-списки).
const SWEEP_FLAGS: &[&str] = &[
    "model-kinds",
    "seeds",
    "d-models",
    "layers-list",
    "d-ffs",
    "lrs",
    "value-encoders",
    "fourier-scales",
    "fourier-bands",
    "schedulers",
    "mlp-widths",
    "mlp-layers-list",
    "kan-widths",
    "kan-layers-list",
    "kan-grids",
    "epochs",
    "batch-size",
];

fn fail(msg: &str) -> ! {
    eprintln!("Ошибка: {msg}");
    std::process::exit(1);
}

impl Flags {
    /// Разбирает аргументы строго: неизвестный флаг, флаг без значения или
    /// «--flag --other» (значение похоже на флаг) — ошибка, а не тихий default.
    fn parse(rest: &[String], value_flags: &[&str], bool_flags: &[&str]) -> Result<Self, String> {
        let mut flags = HashMap::new();
        let mut positionals = Vec::new();
        let mut i = 0;
        while i < rest.len() {
            if let Some(key) = rest[i].strip_prefix("--") {
                if bool_flags.contains(&key) {
                    flags.insert(key.to_string(), "true".to_string());
                    i += 1;
                    continue;
                }
                if !value_flags.contains(&key) {
                    return Err(format!("неизвестный флаг --{key}"));
                }
                let val = rest
                    .get(i + 1)
                    .filter(|v| !v.starts_with("--"))
                    .ok_or_else(|| format!("у флага --{key} нет значения"))?;
                flags.insert(key.to_string(), val.clone());
                i += 2;
            } else {
                positionals.push(rest[i].clone());
                i += 1;
            }
        }
        Ok(Self { flags, positionals })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.flags.get(key).map(String::as_str)
    }
    fn has(&self, key: &str) -> bool {
        self.flags.contains_key(key)
    }
    /// `Ok(None)` если флаг отсутствует, `Err` если присутствует, но не парсится.
    fn usize(&self, key: &str) -> Result<Option<usize>, String> {
        match self.flags.get(key) {
            None => Ok(None),
            Some(s) => s
                .parse()
                .map(Some)
                .map_err(|_| format!("--{key}: ожидалось целое, получено '{s}'")),
        }
    }
    fn f32(&self, key: &str) -> Result<Option<f32>, String> {
        match self.flags.get(key) {
            None => Ok(None),
            Some(s) => s
                .parse()
                .map(Some)
                .map_err(|_| format!("--{key}: ожидалось число, получено '{s}'")),
        }
    }
    fn pos(&self, i: usize) -> Option<&str> {
        self.positionals.get(i).map(String::as_str)
    }
}

fn model_config_from(f: &Flags) -> Result<ModelConfig, String> {
    let layers = f.usize("layers")?;
    Ok(ModelConfig {
        d_model: f.usize("d-model")?.unwrap_or(32),
        n_heads: f.usize("heads")?.unwrap_or(4),
        n_enc_layers: f.usize("enc-layers")?.or(layers).unwrap_or(2),
        n_dec_layers: f.usize("dec-layers")?.or(layers).unwrap_or(2),
        d_ff: f.usize("d-ff")?.unwrap_or(64),
        ln_eps: 1e-5,
    })
}

fn train_config_from(f: &Flags, epochs: usize) -> Result<TrainConfig, String> {
    let schedule = match f.get("scheduler").unwrap_or("constant") {
        "constant" => LrSchedule::Constant,
        "warmup-cosine" => {
            let warmup_frac = f.f32("warmup")?.unwrap_or(0.1);
            let min_lr_ratio = f.f32("min-lr-ratio")?.unwrap_or(0.1);
            if !(0.0..1.0).contains(&warmup_frac) {
                return Err("--warmup должен быть в [0, 1)".to_string());
            }
            if !(0.0..=1.0).contains(&min_lr_ratio) {
                return Err("--min-lr-ratio должен быть в [0, 1]".to_string());
            }
            LrSchedule::WarmupCosine {
                warmup_frac,
                min_lr_ratio,
            }
        }
        other => {
            return Err(format!(
                "--scheduler: ожидалось constant|warmup-cosine, получено '{other}'"
            ))
        }
    };
    let tc = TrainConfig {
        epochs,
        batch_size: f.usize("batch-size")?.unwrap_or(64),
        lr: f.f32("lr")?.unwrap_or(1e-3),
        seed: f.usize("seed")?.map(|s| s as u64).unwrap_or(0),
        schedule,
    };
    validate_train(tc.lr, tc.batch_size)?;
    Ok(tc)
}

fn numeric_config_from(f: &Flags) -> Result<NumericConfig, String> {
    let kind = match f.get("model-kind").unwrap_or("transformer") {
        "transformer" => ModelKind::Transformer,
        "mlp" => ModelKind::Mlp,
        "kan" => ModelKind::Kan,
        other => {
            return Err(format!(
                "--model-kind: ожидалось transformer|mlp|kan, получено '{other}'"
            ))
        }
    };
    let value_kind = match f.get("value-encoder").unwrap_or("linear") {
        "linear" => ValueEncoderKind::Linear,
        "mlp" => ValueEncoderKind::Mlp,
        "fourier" => ValueEncoderKind::Fourier,
        other => {
            return Err(format!(
                "--value-encoder: ожидалось linear|mlp|fourier, получено '{other}'"
            ))
        }
    };
    let nc = NumericConfig {
        kind,
        transformer: model_config_from(f)?,
        value: ValueEncoderConfig {
            kind: value_kind,
            fourier_bands: f.usize("fourier-bands")?.unwrap_or(6),
            fourier_scale: f.f32("fourier-scale")?.unwrap_or(8.0),
        },
        mlp_width: f.usize("mlp-width")?.unwrap_or(128),
        mlp_layers: f.usize("mlp-layers")?.unwrap_or(3),
        kan: {
            let d = KanConfig::default();
            KanConfig {
                width: f.usize("kan-width")?.unwrap_or(d.width),
                layers: f.usize("kan-layers")?.unwrap_or(d.layers),
                grid: f.usize("kan-grid")?.unwrap_or(d.grid),
            }
        },
    };
    validate_numeric(&nc)?;
    Ok(nc)
}

/// Разбор конвейера интерпретации: `--interpret` задаёт профиль, явные
/// `--kan-*` его переопределяют. Флаги при не-KAN модели — ошибка, а не
/// молчаливое игнорирование.
fn interpret_from(f: &Flags, nc: &NumericConfig) -> Result<Option<InterpretProfile>, String> {
    let overrides = InterpretOverrides {
        l1: f.f32("kan-l1")?,
        prune: f.f32("kan-prune")?,
        finetune_epochs: f.usize("kan-finetune-epochs")?,
        compact: f.has("kan-compact").then_some(true),
    };
    let use_profile = f.has("interpret");
    if !use_profile && overrides.is_empty() {
        return Ok(None);
    }
    if nc.kind != ModelKind::Kan {
        return Err(
            "--interpret и --kan-l1/--kan-prune/--kan-finetune-epochs/--kan-compact \
             применимы только к --model-kind kan"
                .to_string(),
        );
    }
    interpret::resolve(use_profile, &overrides).map_err(|e| format!("конвейер интерпретации: {e}"))
}

/// Включает activation-L1 на построенной модели (до обучения).
fn apply_kan_l1(model: &NumericModel, profile: &Option<InterpretProfile>) {
    if let Some(s) = profile {
        if s.l1 > 0.0 {
            model
                .as_kan()
                .expect("interpret_from гарантирует kind=Kan")
                .set_l1_lambda(s.l1);
            println!("KAN activation-L1: λ={}", s.l1);
        }
    }
}

/// Прунинг + fine-tune обученной KAN: важность p95 |φ| на train, hard-prune
/// ниже относительного порога, дообучение с λ=0.
///
/// `eval` — набор для отчёта о влиянии прунинга (validation в фазе разработки).
/// У финальной модели его нет: test потратить нельзя, а train+validation уже
/// внутри обучения, поэтому там печатаются только активные рёбра.
#[allow(clippy::too_many_arguments)]
fn run_kan_prune(
    model: &NumericModel,
    train: &NumericDataset,
    eval: Option<&NumericDataset>,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    tcfg: &TrainConfig,
    threshold: f32,
    finetune_epochs: usize,
) {
    let kan = model.as_kan().expect("прунинг вызывается только для KAN");
    let before = eval.map(|d| evaluate_surrogate(model, d, in_norm, out_norm));

    let calibration = in_norm.transform(&train.inputs);
    let report = kan.prune_edges(threshold, &calibration);
    println!("\nKAN prune (важность = p95 |φ| на train, порог {threshold} от максимума слоя):");
    for (l, (a, t)) in report.per_layer.iter().enumerate() {
        println!("  слой {l}: {a}/{t} активных рёбер");
    }
    let after_prune = eval.map(|d| evaluate_surrogate(model, d, in_norm, out_norm));

    kan.set_l1_lambda(0.0);
    let ft_cfg = TrainConfig {
        epochs: finetune_epochs,
        ..tcfg.clone()
    };
    train_surrogate(model, train, in_norm, out_norm, &ft_cfg);

    let (active, total) = report.totals();
    if let (Some(before), Some(after_prune), Some(eval)) = (before, after_prune, eval) {
        let after_ft = evaluate_surrogate(model, eval, in_norm, out_norm);
        println!(
            "R² на validation: до прунинга {:.5} -> после {:.5} -> после fine-tune ({finetune_epochs} эпох, λ=0) {:.5}",
            before.r2, after_prune.r2, after_ft.r2
        );
    }
    println!("Активных рёбер: {active}/{total} (параметры не сжимаются — это следующий шаг)");
}

/// `--kan-symbolic` при не-KAN модели — ошибка, не молчаливое игнорирование.
fn validate_kan_symbolic(f: &Flags, nc: &NumericConfig) -> Result<(), String> {
    for flag in ["kan-symbolic", "kan-compact"] {
        if f.has(flag) && nc.kind != ModelKind::Kan {
            return Err(format!("--{flag} применим только к --model-kind kan"));
        }
    }
    Ok(())
}

/// Структурное сжатие KAN: физически удаляет мёртвые скрытые узлы (после
/// hard-prune) с реальным уменьшением числа параметров и проверяет, что
/// функция сети не изменилась (R² на переданной eval-части до/после).
fn run_kan_compact(
    model: &mut NumericModel,
    eval: Option<&NumericDataset>,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
) {
    let before = eval.map(|d| evaluate_surrogate(model, d, in_norm, out_norm));
    let report = model
        .as_kan_mut()
        .expect("структурное сжатие вызывается только для KAN")
        .compact();
    println!(
        "\nСтруктурное сжатие: скрытых узлов {} -> {}, параметров {} -> {}",
        report.nodes_before, report.nodes_after, report.params_before, report.params_after
    );
    if let (Some(before), Some(eval)) = (before, eval) {
        let after = evaluate_surrogate(model, eval, in_norm, out_norm);
        println!(
            "R² на validation: {:.5} -> {:.5} (удаление точное — совпадение ожидаемо)",
            before.r2, after.r2
        );
    }
}

/// Symbolic extraction обученной KAN: фит рёбер примитивами по активациям,
/// послойные формулы и верность формул.
///
/// Запускается на ФИНАЛЬНОЙ модели — той, что сохраняется, — поэтому формулы
/// соответствуют сохранённому checkpoint-у. Верность считается на train+
/// validation: test потратить нельзя, и число рядом с R² самой KAN на тех же
/// данных показывает именно расхождение «формулы против модели».
fn run_kan_symbolic(
    model: &NumericModel,
    train: &NumericDataset,
    eval: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    schema: &ModelSchema,
) {
    let kan = model
        .as_kan()
        .expect("symbolic extraction вызывается только для KAN");
    let calibration = in_norm.transform(&train.inputs);
    // Свёртка z-score в коэффициенты: формулы и предсказания — в исходных
    // единицах данных, промежуточные узлы h остаются безразмерными.
    let sym = symbolic::symbolize(kan, &calibration, 256).denormalize(in_norm, out_norm);

    println!("\n=== SYMBOLIC EXTRACTION (входы и выходы в исходных единицах данных) ===");
    print!(
        "{}",
        sym.formulas(schema)
            .unwrap_or_else(|e| fail(&format!("формулы KAN: {e}")))
    );

    let (min_r2, mean_r2) = sym.edge_r2_stats();
    println!("Подгонка рёбер примитивами: min R² = {min_r2:.4}, среднее R² = {mean_r2:.4}");
    let weak = sym.weak_edges(0.99);
    if !weak.is_empty() {
        println!("Слабо подогнанные рёбра (R² < 0.99) – формула там приближённая:");
        for w in weak {
            let (input, output) = sym
                .edge_labels(w, schema)
                .unwrap_or_else(|e| fail(&format!("слабое ребро KAN: {e}")));
            println!(
                "  слой {}, {input} -> {output}: {} (R²={:.4})",
                w.layer, w.name, w.r2
            );
        }
    }

    let pred = sym.predict(&eval.inputs);
    let m = evaluate(&pred, &eval.outputs);
    let kan_m = evaluate_surrogate(model, eval, in_norm, out_norm);
    println!(
        "Формулы как модель на train+validation: R² = {:.5} (KAN там же: {:.5}), rel = {:.2}%",
        m.r2,
        kan_m.r2,
        m.rel_error * 100.0
    );
    println!("  (обучающие данные — это верность формул модели, а не обобщение)");
}

/// Число эпох: флаг `--epochs`, иначе позиционный аргумент (legacy), иначе 40.
/// Если позиционный аргумент задан, он ОБЯЗАН парситься — иначе ошибка, а не
/// тихий откат к дефолту.
fn resolve_epochs(f: &Flags) -> usize {
    if let Some(e) = f.usize("epochs").unwrap_or_else(|e| fail(&e)) {
        return e;
    }
    match f.pos(1) {
        Some(s) => s.parse().unwrap_or_else(|_| {
            fail(&format!(
                "позиционные эпохи: ожидалось целое, получено '{s}'"
            ))
        }),
        None => 40,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        // Без подкоманды — GUI (основной режим). CLI-подкоманды — по имени.
        None | Some("gui") => run_gui_cmd(),
        Some("numeric") => run_numeric(&args),
        Some("numeric-file") => run_numeric_file(&args),
        Some("sweep") => run_sweep(&args),
        Some("epoch-sweep") => run_epoch_sweep_cmd(&args),
        Some("prepare") => run_prepare(&args),
        Some("text") => run_text(&args),
        Some("predict") => run_predict(&args),
        Some(other) => {
            eprintln!("Неизвестная команда: {other}\n");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Использование:");
    eprintln!("  transformer                 без аргументов — GUI (основной режим)");
    eprintln!("  transformer gui             явный запуск GUI");
    eprintln!("  transformer numeric <blackbox> [--epochs N] [--model out.bin] [флаги]");
    eprintln!("  transformer numeric-file <data> [--epochs N] [--model out.bin] [флаги]");
    eprintln!("  transformer sweep <blackbox> [--model-kinds transformer,mlp,kan]");
    eprintln!("                    [--d-models 32,64 --layers-list 2,3 --mlp-widths 128]");
    eprintln!("                    [--kan-widths 16,32 --kan-layers-list 2 --kan-grids 8,16]");
    eprintln!("  transformer epoch-sweep <data> [--epochs 1,2,5,10,20,40] [конфиг-флаги]");
    eprintln!(
        "  transformer prepare <input> <out.tnum> --inputs N --outputs M [--has-header] [--categorical 9:4]"
    );
    eprintln!("  transformer text <file.txt> [steps]");
    eprintln!("  transformer predict <model.bin> <v1> <v2> ...");
    eprintln!("  флаги: --d-model --heads --layers --d-ff --lr --batch-size --seed");
    eprintln!("         --model-kind transformer|mlp|kan --mlp-width --mlp-layers");
    eprintln!("         --kan-width --kan-layers --kan-grid");
    eprintln!("         --interpret (профиль KAN: L1 → prune → fine-tune → compact)");
    eprintln!("         --kan-l1 <λ> --kan-prune <отн. порог> --kan-finetune-epochs <N>");
    eprintln!("         --kan-symbolic (извлечь формулы из обученной KAN)");
    eprintln!("         --kan-compact (физически удалить мёртвые узлы после прунинга)");
}

fn print_config(nc: &NumericConfig, tcfg: &TrainConfig) {
    match nc.kind {
        ModelKind::Transformer => {
            let c = &nc.transformer;
            let venc = match nc.value.kind {
                ValueEncoderKind::Linear => "linear".to_string(),
                ValueEncoderKind::Mlp => "mlp".to_string(),
                ValueEncoderKind::Fourier => {
                    format!(
                        "fourier(bands={} scale={})",
                        nc.value.fourier_bands, nc.value.fourier_scale
                    )
                }
            };
            println!(
                "Модель: transformer (d_model={} heads={} enc={} dec={} d_ff={}, value={venc})",
                c.d_model, c.n_heads, c.n_enc_layers, c.n_dec_layers, c.d_ff
            );
        }
        ModelKind::Mlp => println!(
            "Модель: mlp (width={} layers={})",
            nc.mlp_width, nc.mlp_layers
        ),
        ModelKind::Kan => println!(
            "Модель: kan (width={} layers={} grid={})",
            nc.kan.width, nc.kan.layers, nc.kan.grid
        ),
    }
    let sched = match tcfg.schedule {
        LrSchedule::Constant => "constant".to_string(),
        LrSchedule::WarmupCosine {
            warmup_frac,
            min_lr_ratio,
        } => format!("warmup-cosine(warmup={warmup_frac} min={min_lr_ratio})"),
    };
    println!(
        "Обучение: lr={} batch={} seed={} sched={sched}",
        tcfg.lr, tcfg.batch_size, tcfg.seed
    );
}

/// MLP и KAN получают категориальный код как обычное число и неявно считают,
/// что код 3 «между» 2 и 4. Embedding есть только у transformer, поэтому здесь
/// предупреждение, а не тихое обучение на ложной геометрии.
fn warn_categorical_without_embedding(nc: &NumericConfig, schema: &ModelSchema) {
    if nc.kind == ModelKind::Transformer {
        return;
    }
    let categorical: Vec<&str> = schema
        .inputs()
        .iter()
        .filter(|c| c.cardinality().is_some())
        .map(|c| c.name())
        .collect();
    if categorical.is_empty() {
        return;
    }
    println!(
        "ВНИМАНИЕ: категориальные входы ({}) в модели {} кодируются числами —\n\
         \x20 порядок кодов будет воспринят как расстояние. Embedding категорий\n\
         \x20 есть только у transformer.",
        categorical.join(", "),
        match nc.kind {
            ModelKind::Mlp => "mlp",
            ModelKind::Kan => "kan",
            ModelKind::Transformer => unreachable!("проверено выше"),
        }
    );
}

/// Печать метрик с явным указанием, откуда они взяты. Выходы называются по
/// схеме: у модели без имён это по-прежнему `y0`, `y1`, …
fn print_metrics(title: &str, m: &Metrics, per: &[Metrics], schema: &ModelSchema) {
    println!("\n{title} (в исходных единицах):");
    println!("  RMSE        = {:.5}", m.rmse);
    println!("  MAE         = {:.5}", m.mae);
    println!("  rel. error  = {:.2}%", m.rel_error * 100.0);
    println!("  R²          = {:.5}", m.r2);

    if per.len() > 1 {
        let names: Vec<String> = schema.outputs().iter().map(|c| c.display_name()).collect();
        let width = names
            .iter()
            .map(|n| n.chars().count())
            .max()
            .unwrap_or(4)
            .max(4);
        println!("\nПо выходам:");
        println!(
            "  {:<width$}  RMSE        MAE       rel.err     R²",
            "выход"
        );
        for (j, pm) in per.iter().enumerate() {
            let name = names.get(j).cloned().unwrap_or_else(|| format!("y{j}"));
            // Ширина считается в символах: у кириллицы и °C байт больше.
            let pad = width.saturating_sub(name.chars().count());
            println!(
                "  {name}{:pad$}  {:>9.5}  {:>9.5}  {:>7.2}%  {:>8.5}",
                "",
                pm.rmse,
                pm.mae,
                pm.rel_error * 100.0,
                pm.r2
            );
        }
    }
}

/// Общий поток обучения для `numeric` и `numeric-file`.
///
/// Сам сценарий живёт в [`transformer::training`]: здесь остаются только
/// печать и KAN-конвейер, который подключается хуком и потому применяется
/// одинаково к модели разработки и к финальной.
fn run_numeric_flow(
    f: &Flags,
    data: NumericDataset,
    schema: ModelSchema,
    bb: Option<&blackbox::BlackBox>,
) {
    let epochs = resolve_epochs(f);
    let save_path = f.get("model").or_else(|| f.pos(2));
    let nc = numeric_config_from(f).unwrap_or_else(|e| fail(&e));
    let tcfg = train_config_from(f, epochs).unwrap_or_else(|e| fail(&e));
    let dataset = Dataset::new(data, schema).unwrap_or_else(|e| fail(&e));

    let plan = SplitPlan::default();
    // Разбиение печатается до обучения: пользователь должен видеть, на чём
    // модель училась и чем её мерили.
    let preview = plan.prepare(dataset.data()).unwrap_or_else(|e| fail(&e));
    let (train_rows, val_rows) = {
        let (t, v) = preview.search.fold(0).unwrap_or_else(|e| fail(&e));
        (t.len(), v.len())
    };
    println!(
        "Разбиение: {train_rows} train / {val_rows} validation / {} test (holdout, seed {})",
        preview.test.len(),
        DEFAULT_SPLIT_SEED
    );
    drop(preview);

    print_config(&nc, &tcfg);
    warn_categorical_without_embedding(&nc, dataset.schema());
    let interpret = interpret_from(f, &nc).unwrap_or_else(|e| fail(&e));
    if let Some(profile) = &interpret {
        println!("Конвейер интерпретации {}", profile.describe());
    }
    validate_kan_symbolic(f, &nc).unwrap_or_else(|e| fail(&e));

    let setup = TrainingSetup::new(nc.clone(), tcfg.clone());
    let never = std::sync::atomic::AtomicBool::new(false);
    let final_tcfg = TrainConfig {
        seed: DEFAULT_FINAL_INIT_SEED,
        ..tcfg.clone()
    };

    let outcome = run_training(
        &dataset,
        plan,
        &setup,
        true,
        DEFAULT_FINAL_INIT_SEED,
        &never,
        &mut |phase, point| {
            if phase == Phase::Development && (point.epoch % 5 == 1 || point.epoch == epochs) {
                println!(
                    "  эпоха {:>3}: train loss (норм.) = {:.5}",
                    point.epoch, point.train_loss
                );
            }
        },
        &mut |phase, model| {
            // И заголовок, и регуляризатор должны появиться до первой эпохи.
            match phase {
                Phase::Development => {
                    println!("\n=== ФАЗА РАЗРАБОТКИ (метрики на validation) ===")
                }
                Phase::Final => println!(
                    "\n=== ФИНАЛЬНАЯ МОДЕЛЬ (train + validation, seed {DEFAULT_FINAL_INIT_SEED}) ==="
                ),
            }
            println!("Параметров: {}", model.parameter_count());
            apply_kan_l1(model, &interpret);
        },
        &mut |phase, trained, train_data, eval| {
            // Один и тот же конвейер в обеих фазах: иначе сохранённая модель
            // отличалась бы от той, по которой принимали решения.
            let phase_tcfg = match phase {
                Phase::Development => &tcfg,
                Phase::Final => &final_tcfg,
            };
            apply_kan_pipeline(trained, train_data, eval, &interpret, phase_tcfg);
        },
    )
    .unwrap_or_else(|e| fail(&e));

    // Метрики фазы разработки: validation той же модели, что прошла конвейер.
    let dev_split = plan.prepare(dataset.data()).unwrap_or_else(|e| fail(&e));
    let (train_data, val) = dev_split.search.fold(0).unwrap_or_else(|e| fail(&e));
    let (metrics, per_output) = evaluate_on(&outcome.development, &val);
    print_metrics(
        "Метрики на validation",
        &metrics,
        &per_output,
        dataset.schema(),
    );
    if f.has("diagnose") {
        run_diagnostics(
            &nc,
            &dataset.schema().feature_specs(),
            dataset.schema().n_outputs(),
            &train_data,
            &val,
            &outcome.development.in_norm,
            &outcome.development.out_norm,
            &outcome.development.model,
            bb,
        );
    }

    let final_model = outcome.final_model.expect("финальная фаза запрошена");
    let final_eval = outcome.final_eval.expect("финальная фаза запрошена");
    let pool = dev_split.search.all();
    if f.has("kan-symbolic") {
        run_kan_symbolic(
            &final_model.model,
            &pool,
            &pool,
            &final_model.in_norm,
            &final_model.out_norm,
            dataset.schema(),
        );
    }
    print_metrics(
        &format!(
            "Метрики на test ({} строк, единственный замер)",
            final_eval.origin.test_rows
        ),
        &final_eval.metrics,
        &final_eval.per_output,
        dataset.schema(),
    );

    if let Some(path) = save_path {
        save_and_verify(
            path,
            &nc,
            dataset.schema(),
            &final_model,
            &pool,
            interpret.as_ref(),
        );
    }
}

/// KAN-конвейер одной фазы: activation-L1 уже применён при построении, здесь —
/// прунинг с fine-tune и структурное сжатие.
fn apply_kan_pipeline(
    trained: &mut TrainedModel,
    train_data: &NumericDataset,
    eval: Option<&NumericDataset>,
    profile: &Option<InterpretProfile>,
    tcfg: &TrainConfig,
) {
    if let Some(threshold) = profile.as_ref().and_then(|p| p.prune) {
        let ft = profile.as_ref().map_or(10, |p| p.finetune_epochs);
        run_kan_prune(
            &trained.model,
            train_data,
            eval,
            &trained.in_norm,
            &trained.out_norm,
            tcfg,
            threshold,
            ft,
        );
    }
    if profile.is_some_and(|p| p.compact) {
        run_kan_compact(
            &mut trained.model,
            eval,
            &trained.in_norm,
            &trained.out_norm,
        );
    }
}

fn run_numeric(args: &[String]) {
    let f =
        Flags::parse(&args[2..], NUMERIC_FLAGS, NUMERIC_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    let name = f.pos(0).unwrap_or("sum");

    let bb = match blackbox::by_name(name) {
        Some(bb) => bb,
        None => {
            eprintln!("Неизвестный чёрный ящик: {name}");
            eprintln!("Доступны: sum, product, sine, polynomial, projectile");
            std::process::exit(1);
        }
    };
    println!(
        "Чёрный ящик: {} ({} вход -> {} выход)",
        bb.name,
        bb.n_inputs(),
        bb.n_outputs
    );

    // Данные генерируются фиксированным data_seed: --seed меняет только
    // инициализацию модели и порядок батчей.
    let data = bb.generate(2000, DEFAULT_DATA_SEED);
    // У встроенного ящика имён нет — схема синтетическая и это осознанно.
    let schema = ModelSchema::synthetic(bb.n_inputs(), bb.n_outputs).unwrap_or_else(|e| fail(&e));
    run_numeric_flow(&f, data, schema, Some(&bb));
}

fn run_numeric_file(args: &[String]) {
    let f =
        Flags::parse(&args[2..], NUMERIC_FLAGS, NUMERIC_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    let path = match f.pos(0) {
        Some(p) => p,
        None => {
            eprintln!("Укажите путь к .tnum, XLSX, CSV или TSV файлу");
            std::process::exit(1);
        }
    };

    let (data, schema) = match read_numeric_source(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Не удалось прочитать {path}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Датасет: {path} ({} строк, {} вход -> {} выход)",
        data.len(),
        data.inputs.ncols(),
        data.outputs.ncols()
    );
    run_numeric_flow(&f, data, schema, None);
}

fn save_and_verify(
    path: &str,
    nc: &NumericConfig,
    schema: &ModelSchema,
    trained: &TrainedModel,
    pool: &NumericDataset,
    interpret: Option<&InterpretProfile>,
) {
    // Модель и её нормализаторы врозь бессмысленны, поэтому едут вместе.
    let (model, in_norm, out_norm) = (&trained.model, &trained.in_norm, &trained.out_norm);
    // Калибровка (выборка сырых обучающих строк) едет в checkpoint: symbolic
    // extraction остаётся доступной после загрузки .bin.
    let calibration = calibration_sample(&pool.inputs, 256);
    save_numeric(
        path,
        nc,
        schema,
        model,
        in_norm,
        out_norm,
        Some(&calibration),
        interpret,
    )
    .expect("сохранение модели");
    // Проверка целостности файла, а не качества: test уже потрачен, поэтому
    // сравниваем предсказания загруженной модели с исходной на тех же данных.
    let checkpoint = load_numeric_full(path).expect("загрузка модели");
    assert_eq!(
        checkpoint.interpret,
        interpret.copied(),
        "checkpoint изменил профиль интерпретации при сохранении"
    );
    let before = predict_dataset(model, pool, in_norm, out_norm);
    let after = predict_dataset(
        &checkpoint.model,
        pool,
        &checkpoint.in_norm,
        &checkpoint.out_norm,
    );
    let max_diff = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\nМодель сохранена в {path}; макс. расхождение после загрузки = {max_diff:.3e}");
}

#[allow(clippy::too_many_arguments)]
fn run_diagnostics(
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    n_outputs: usize,
    train: &NumericDataset,
    val: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    model: &NumericModel,
    bb: Option<&blackbox::BlackBox>,
) {
    // Диагностика — часть принятия решений, поэтому смотрит на validation:
    // счёт экстраполяции и формы остатка по test означали бы подглядывание.
    println!("\n=== ДИАГНОСТИКА (на validation) ===");

    // 1. Ёмкость: overfit маленького подмножества train.
    let subset = train.gather(&(0..train.len().min(48)).collect::<Vec<_>>());
    let probe = diagnostics::overfit_probe(nc, specs, n_outputs, &subset, 80);
    println!(
        "Overfit-проба ({} строк): норм. train MSE = {probe:.5}",
        subset.len()
    );
    println!(
        "  -> {}",
        if probe < 0.02 {
            "ёмкости хватает (ищи проблему в покрытии/обобщении)"
        } else {
            "underfit -> ёмкость или кодирование значений (value encoder / Fourier)"
        }
    );

    // 2. Покрытие: экстраполяция на validation.
    let rr = diagnostics::range_report(in_norm, &val.inputs);
    println!(
        "Экстраполяция: {} из {} validation-строк вне обученного диапазона",
        rr.rows_out, rr.total
    );

    // 3. Форма остатка по входным признакам.
    let pred = predict_dataset(model, val, in_norm, out_norm);
    let res = diagnostics::residual_diagnostics(&val.inputs, &pred, &val.outputs);
    println!("Остаток по входным признакам:");
    for (j, d) in res.iter().enumerate() {
        println!(
            "  признак {j}: смена знака {:>4.0}% | tail/inner {:.2}",
            d.sign_change_rate * 100.0,
            d.tail_ratio
        );
    }
    println!("  (высокая смена знака -> частота/Fourier; tail/inner>1.5 -> масштаб/хвосты)");

    // 4. Чувствительность карты — только при вызываемом чёрном ящике.
    match bb {
        Some(bb) => {
            let (mean, max) = diagnostics::sensitivity_probe(bb, in_norm, out_norm, 300, 0.01, 0);
            println!("Чувствительность ||Δy||/||Δx|| (норм.): среднее {mean:.2}, макс {max:.2}");
            println!(
                "  -> {}",
                if max < 10.0 {
                    "карта гладкая, surrogate надёжен"
                } else {
                    "высокая: чувствительность/возможен хаос -> потолок точности"
                }
            );
        }
        None => println!("Чувствительность: пропущена (нет вызываемого чёрного ящика)"),
    }
}

fn run_predict(args: &[String]) {
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("Укажите путь к сохранённой модели (.bin)");
            std::process::exit(1);
        }
    };
    let raw_args = &args[3..];
    if raw_args.is_empty() {
        fail("укажите входные значения: transformer predict model.bin v1 v2 ...");
    }

    let checkpoint = match load_numeric_full(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Не удалось загрузить {path}: {e}");
            std::process::exit(1);
        }
    };
    let schema = &checkpoint.schema;
    warn_categorical_without_embedding(&checkpoint.config, schema);

    if raw_args.len() != schema.n_inputs() {
        fail(&format!(
            "модель ожидает {} входов ({}), получено {}",
            schema.n_inputs(),
            schema
                .input_names()
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            raw_args.len()
        ));
    }

    // Категориальный вход задаётся ПОДПИСЬЮ уровня, а не кодом: у старых
    // моделей подписи и есть коды («0», «1»), поэтому прежние команды работают.
    let mut values = Vec::with_capacity(raw_args.len());
    for (i, arg) in raw_args.iter().enumerate() {
        let column = &schema.inputs()[i];
        let value = match column.cardinality() {
            Some(_) => column.category_code(arg).unwrap_or_else(|e| fail(&e)) as f32,
            None => {
                let value = arg.parse::<f32>().unwrap_or_else(|_| {
                    fail(&format!(
                        "вход '{}' = '{arg}': ожидалось число",
                        column.name()
                    ))
                });
                if !value.is_finite() {
                    fail(&format!(
                        "вход '{}' = '{arg}': число должно быть конечным",
                        column.name()
                    ));
                }
                value
            }
        };
        values.push(value);
    }

    let f = values.len();
    let raw = Array2::from_shape_vec((1, f), values.clone()).unwrap();
    let x = Tensor::constant(checkpoint.in_norm.transform(&raw).into_dyn());
    let pred_norm = checkpoint
        .model
        .predict(&x)
        .data()
        .into_dimensionality::<Ix2>()
        .expect("predict возвращает [1, O]");
    let pred = checkpoint.out_norm.inverse_transform(&pred_norm);

    println!("Вход:");
    for (i, column) in schema.inputs().iter().enumerate() {
        match column.cardinality() {
            Some(_) => println!(
                "  {} = {} (код {})",
                column.display_name(),
                raw_args[i],
                values[i] as usize
            ),
            None => println!("  {} = {}", column.display_name(), values[i]),
        }
    }
    println!("Выход:");
    for (j, column) in schema.outputs().iter().enumerate() {
        println!("  {} = {}", column.display_name(), pred[[0, j]]);
    }
}

// --- epoch-sweep (свип по эпохам), CLI-only ---

fn sparkline(xs: &[f32]) -> String {
    if xs.is_empty() {
        return String::new();
    }
    let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let (mn, mx) = xs
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let range = (mx - mn).max(1e-12);
    xs.iter()
        .map(|&x| {
            let t = ((x - mn) / range * (bars.len() - 1) as f32).round() as usize;
            bars[t.min(bars.len() - 1)]
        })
        .collect()
}

fn run_epoch_sweep_cmd(args: &[String]) {
    let mut allowed = NUMERIC_FLAGS.to_vec();
    allowed.extend_from_slice(&["out-dir", "target-r2", "min-r2-gain", "plateau-min-r2"]);
    let f = Flags::parse(&args[2..], &allowed, NUMERIC_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    for k in [
        "kan-l1",
        "kan-prune",
        "kan-finetune-epochs",
        "kan-symbolic",
        "kan-compact",
        "interpret",
    ] {
        if f.has(k) {
            fail(&format!("--{k} не поддерживается в epoch-sweep"));
        }
    }

    let path = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите данные: epoch-sweep <data>"));
    let (data, schema) = read_numeric_source(path).unwrap_or_else(|e| fail(&e));
    let dataset = Dataset::new(data, schema).unwrap_or_else(|e| fail(&e));

    let milestones = csv_usize(&f, "epochs", "1,2,5,10,20,40");
    if milestones.is_empty() {
        fail("--epochs: пустой список");
    }
    if milestones.contains(&0) {
        fail("--epochs: все значения должны быть > 0");
    }
    let target_r2 = f
        .f32("target-r2")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or(0.95);
    let min_gain = f
        .f32("min-r2-gain")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or(0.02);
    let plateau = f
        .f32("plateau-min-r2")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or(0.80);
    if !target_r2.is_finite() {
        fail("--target-r2 должен быть конечным");
    }
    if !min_gain.is_finite() || min_gain < 0.0 {
        fail("--min-r2-gain должен быть конечным и >= 0");
    }
    if !plateau.is_finite() {
        fail("--plateau-min-r2 должен быть конечным");
    }
    let out_dir = f.get("out-dir").unwrap_or("runs").to_string();

    let prepared = SplitPlan::default()
        .prepare(dataset.data())
        .unwrap_or_else(|e| fail(&e));
    let nc = numeric_config_from(&f).unwrap_or_else(|e| fail(&e));
    let max_e = milestones.iter().copied().max().unwrap_or(1);
    let base_tcfg = train_config_from(&f, max_e).unwrap_or_else(|e| fail(&e));

    println!(
        "Epoch-sweep {path}: эпохи {milestones:?}, {} строк в поиске, test ({} строк) не трогаем",
        prepared.search.len(),
        prepared.test.len()
    );
    let rows =
        epoch_sweep::run_epoch_sweep(&dataset, &prepared.search, &nc, &base_tcfg, &milestones)
            .unwrap_or_else(|e| fail(&e));

    println!("\nepochs  train_loss     RMSE       MAE      rel.err     R² (validation)");
    for r in &rows {
        println!(
            "{:>6}  {:>10.5}  {:>9.5}  {:>9.5}  {:>7.2}%  {:>8.5}",
            r.epochs,
            r.train_loss,
            r.rmse,
            r.mae,
            r.rel_error * 100.0,
            r.r2
        );
    }
    let r2s: Vec<f32> = rows.iter().map(|r| r.r2).collect();
    let losses: Vec<f32> = rows.iter().map(|r| r.train_loss).collect();
    println!("\nR²    {}", sparkline(&r2s));
    println!("loss  {}", sparkline(&losses));

    if let Some((e, why)) = epoch_sweep::recommended_stop(&rows, target_r2, min_gain, plateau) {
        println!("\nРекомендованная остановка: {e} эпох ({why})");
    }

    std::fs::create_dir_all(&out_dir).ok();
    let csv_path = format!("{out_dir}/epoch_sweep_results.csv");
    std::fs::write(&csv_path, epoch_sweep::rows_to_csv(&rows))
        .unwrap_or_else(|e| fail(&format!("запись {csv_path}: {e}")));
    println!("CSV: {csv_path}");
}

// --- gui (egui), за фичей `gui` ---

fn run_gui_cmd() {
    #[cfg(feature = "gui")]
    {
        if let Err(e) = transformer::gui::run_gui() {
            fail(&format!("GUI: {e}"));
        }
    }
    #[cfg(not(feature = "gui"))]
    {
        eprintln!("GUI недоступен: собрано без фичи gui (cargo build --features gui).\n");
        print_usage();
        std::process::exit(1);
    }
}

// --- prepare (таблица -> .tnum), CLI-only ---

fn run_prepare(args: &[String]) {
    let f =
        Flags::parse(&args[2..], PREPARE_FLAGS, PREPARE_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    let input = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите входную таблицу: prepare <input> <out.tnum>"));
    let output = f
        .pos(1)
        .unwrap_or_else(|| fail("укажите выходной .tnum: prepare <input> <out.tnum>"));
    let delimiter = match f.get("delimiter").unwrap_or("auto") {
        "auto" => Delimiter::Auto,
        "comma" => Delimiter::Comma,
        "tab" => Delimiter::Tab,
        "space" => Delimiter::Space,
        o => fail(&format!(
            "--delimiter: ожидалось auto|comma|tab|space, получено '{o}'"
        )),
    };
    let inferred = match (
        f.usize("inputs").unwrap_or_else(|e| fail(&e)),
        f.usize("outputs").unwrap_or_else(|e| fail(&e)),
        f.get("categorical"),
    ) {
        (Some(_), Some(_), Some(_)) => None,
        _ => infer_prepare_spec_from_path(input, delimiter).ok(),
    };
    let n_inputs = f
        .usize("inputs")
        .unwrap_or_else(|e| fail(&e))
        .or_else(|| inferred.as_ref().map(|i| i.n_inputs))
        .unwrap_or_else(|| fail("--inputs обязателен (или нужен заголовок x.../y... для auto)"));
    let n_outputs = f
        .usize("outputs")
        .unwrap_or_else(|e| fail(&e))
        .or_else(|| inferred.as_ref().map(|i| i.n_outputs))
        .unwrap_or_else(|| fail("--outputs обязателен (или нужен заголовок x.../y... для auto)"));
    let categorical = if let Some(raw) = f.get("categorical") {
        parse_categorical(raw, n_inputs).unwrap_or_else(|e| fail(&e))
    } else {
        inferred
            .as_ref()
            .map(|i| i.categorical.clone())
            .unwrap_or_default()
    };
    let has_header = f.has("has-header") || inferred.as_ref().is_some_and(|i| i.has_header);

    let spec = PrepareSpec {
        n_inputs,
        n_outputs,
        delimiter,
        has_header,
        categorical,
    };

    let tnum = table_path_to_tnum(input, &spec).unwrap_or_else(|e| fail(&e));
    // Число строк берём из самого заголовка: их количество в TRNUM2 зависит от
    // наличия units и levels, поэтому вычитать фиксированную константу нельзя.
    let rows = tnum
        .lines()
        .find_map(|l| l.strip_prefix("rows "))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or_else(|| fail("в записанном .tnum нет строки rows"));
    std::fs::write(output, &tnum).unwrap_or_else(|e| fail(&format!("запись {output}: {e}")));
    println!("Записано {output}: {rows} строк, {n_inputs} вход -> {n_outputs} выход");
}

// --- sweep ---

fn csv_usize(f: &Flags, key: &str, default: &str) -> Vec<usize> {
    f.get(key)
        .unwrap_or(default)
        .split(',')
        .map(|p| {
            p.trim()
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key}: '{}' не целое", p.trim())))
        })
        .collect()
}

fn csv_f32(f: &Flags, key: &str, default: &str) -> Vec<f32> {
    f.get(key)
        .unwrap_or(default)
        .split(',')
        .map(|p| {
            p.trim()
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key}: '{}' не число", p.trim())))
        })
        .collect()
}

fn csv_u64(f: &Flags, key: &str, default: &str) -> Vec<u64> {
    f.get(key)
        .unwrap_or(default)
        .split(',')
        .map(|p| {
            p.trim()
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key}: '{}' не целое", p.trim())))
        })
        .collect()
}

fn parse_venc(s: &str) -> ValueEncoderKind {
    match s {
        "linear" => ValueEncoderKind::Linear,
        "mlp" => ValueEncoderKind::Mlp,
        "fourier" => ValueEncoderKind::Fourier,
        o => fail(&format!("--value-encoders: '{o}' не linear|mlp|fourier")),
    }
}

fn parse_sched(s: &str) -> LrSchedule {
    match s {
        "constant" => LrSchedule::Constant,
        "warmup-cosine" => LrSchedule::WarmupCosine {
            warmup_frac: 0.1,
            min_lr_ratio: 0.1,
        },
        o => fail(&format!("--schedulers: '{o}' не constant|warmup-cosine")),
    }
}

fn run_sweep(args: &[String]) {
    let f = Flags::parse(&args[2..], SWEEP_FLAGS, &[]).unwrap_or_else(|e| fail(&e));
    let name = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите чёрный ящик: sweep <blackbox>"));
    blackbox::by_name(name).unwrap_or_else(|| fail(&format!("неизвестный чёрный ящик: {name}")));

    let seeds = csv_u64(&f, "seeds", "0");
    let d_models = csv_usize(&f, "d-models", "32");
    let layers = csv_usize(&f, "layers-list", "2");
    let d_ffs = csv_usize(&f, "d-ffs", "64");
    let lrs = csv_f32(&f, "lrs", "0.001");
    let vencs: Vec<&str> = f
        .get("value-encoders")
        .unwrap_or("linear")
        .split(',')
        .map(str::trim)
        .collect();
    let fscales = csv_f32(&f, "fourier-scales", "2");
    let scheds: Vec<&str> = f
        .get("schedulers")
        .unwrap_or("constant")
        .split(',')
        .map(str::trim)
        .collect();
    let epochs = f.usize("epochs").unwrap_or_else(|e| fail(&e)).unwrap_or(30);
    let batch = f
        .usize("batch-size")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or(64);
    let bands = f
        .usize("fourier-bands")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or(6);

    let model_kinds: Vec<ModelKind> = f
        .get("model-kinds")
        .unwrap_or("transformer")
        .split(',')
        .map(|s| match s.trim() {
            "transformer" => ModelKind::Transformer,
            "mlp" => ModelKind::Mlp,
            "kan" => ModelKind::Kan,
            other => fail(&format!(
                "--model-kinds: ожидалось transformer|mlp|kan, получено '{other}'"
            )),
        })
        .collect();

    let axes = sweep_core::SweepAxes {
        model_kinds,
        seeds,
        d_models,
        layers,
        d_ffs,
        lrs,
        value_encoders: vencs.iter().map(|v| parse_venc(v)).collect(),
        fourier_scales: fscales,
        fourier_bands: bands,
        mlp_widths: csv_usize(&f, "mlp-widths", "128"),
        mlp_layers: csv_usize(&f, "mlp-layers-list", "3"),
        kan_widths: csv_usize(&f, "kan-widths", "16"),
        kan_layers: csv_usize(&f, "kan-layers-list", "2"),
        kan_grids: csv_usize(&f, "kan-grids", "8"),
        schedules: scheds.iter().map(|s| parse_sched(s)).collect(),
        epochs,
        final_epochs: epochs,
        batch_size: batch,
    };
    // Цена операции — до запуска: она понятнее, чем название набора осей.
    let cost = sweep_core::sweep_cost(&axes, 1).unwrap_or_else(|e| fail(&e));
    println!("Sweep {name}: {}\n", cost.describe());

    let never = std::sync::atomic::AtomicBool::new(false);
    let result = sweep_core::run_blackbox_sweep(name, &axes, &never, |row| {
        println!(
            "  done [{}]: {} -> worst R²={:.5}, aggregate R²={:.5}±{:.5}",
            epoch_sweep::source_label(row.source),
            row.label,
            row.worst_output_r2_mean,
            row.r2_mean,
            row.r2_std
        );
    })
    .unwrap_or_else(|e| fail(&e));

    // Рекомендация — первая строка, по worst-output R² по умолчанию.
    let source = result
        .rows
        .first()
        .map(|row| epoch_sweep::source_label(row.source))
        .unwrap_or_else(|| "validation".to_string());
    println!("\n=== РАНЖИРОВАНИЕ ({source}; по worst-output R²; rel — справочно) ===");
    for (i, r) in result.rows.iter().enumerate() {
        let mark = if i == 0 { "*" } else { " " };
        println!(
            "{mark} worst R²={:.5}  aggregate R²={:.5}±{:.5}  nRMSE={:.5}  rel={:.1}%  | {}",
            r.worst_output_r2_mean,
            r.r2_mean,
            r.r2_std,
            r.nrmse_mean,
            r.rel_mean * 100.0,
            r.label
        );
    }
}

fn run_text(args: &[String]) {
    let path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("Укажите путь к .txt файлу");
            std::process::exit(1);
        }
    };
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Не удалось прочитать {path}: {e}");
            std::process::exit(1);
        }
    };

    let ds = TextDataset::new(&text);
    println!(
        "Корпус: {} символов, словарь {} символов",
        text.chars().count(),
        ds.vocab.len()
    );

    let ctx_len = 32;
    let tgt_len = 32;
    let cfg = ModelConfig {
        d_model: 64,
        n_heads: 4,
        n_enc_layers: 2,
        n_dec_layers: 2,
        d_ff: 128,
        ln_eps: 1e-5,
    };
    set_init_seed(0); // воспроизводимая инициализация char-LM
    let model = TextModel::new(&cfg, ds.vocab.len());
    let tcfg = TextTrainConfig {
        steps,
        batch_size: 32,
        ctx_len,
        tgt_len,
        lr: 1e-3,
        seed: 0,
    };

    println!("Обучение char-LM: {steps} шагов...");
    let history = train_text(&model, &ds, &tcfg);
    for (i, loss) in history.iter().enumerate() {
        let frac = (i + 1) as f32 / history.len() as f32;
        println!(
            "  {:>3}% : loss = {loss:.4}  (perplexity = {:.2})",
            (frac * 100.0) as usize,
            loss.exp()
        );
    }

    let seed: String = text.chars().take(ctx_len).collect();
    let mut rng = StdRng::seed_from_u64(42);
    let sample = generate(
        &model, &ds.vocab, &seed, 400, ctx_len, tgt_len, 0.8, 10, &mut rng,
    );

    println!("\n--- Затравка ---\n{seed}");
    println!("\n--- Генерация ---\n{seed}{sample}");
}
