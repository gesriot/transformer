#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

//! CLI: обучение surrogate-модели на числовых данных и прогноз по ней.
//!
//! Использование:
//!   transformer gui
//!   transformer train <данные> [--epochs N] [--eval-every N] [--model out.bin]
//!   transformer search <данные> [оси сетки]
//!   transformer prepare <таблица> <out.tnum> --inputs N --outputs M
//!   transformer predict <model.bin> <v1> <v2> ...
//!   transformer predict <model.bin> --table <вход> --out <выход>
//!   transformer demo train|search <чёрный ящик> | demo text <file.txt> [steps]
//!
//! Конфиг-флаги: --d-model --heads --layers --enc-layers --dec-layers --d-ff
//!               --lr --batch-size --seed

use ndarray::Array2;
#[cfg(feature = "demo")]
use rand::rngs::StdRng;
#[cfg(feature = "demo")]
use rand::SeedableRng;
use std::collections::HashMap;
use transformer::batch_predict;
#[cfg(feature = "demo")]
use transformer::blackbox;
use transformer::config::ModelConfig;
#[cfg(feature = "demo")]
use transformer::data::TextDataset;
use transformer::data::{Normalizer, NumericDataset};
use transformer::diagnostics;
use transformer::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
#[cfg(feature = "demo")]
use transformer::generate::generate;
#[cfg(feature = "demo")]
use transformer::init::set_init_seed;
use transformer::interpret::{self, InterpretOverrides, InterpretProfile, InterpretReport};
use transformer::metrics::{evaluate, Metrics};
use transformer::numeric_model::{
    validate_numeric, KanConfig, ModelKind, NumericConfig, NumericModel,
};
use transformer::predict;
use transformer::schema::ModelSchema;
use transformer::serialize::{calibration_sample, load_numeric_full, save_numeric};
#[cfg(feature = "demo")]
use transformer::split::DEFAULT_DATA_SEED;
use transformer::split::{SplitPlan, DEFAULT_FINAL_INIT_SEED, DEFAULT_SPLIT_SEED};
use transformer::sweep as sweep_core;
use transformer::symbolic;
#[cfg(feature = "demo")]
use transformer::textmodel::TextModel;
use transformer::tnum::{
    infer_prepare_spec_from_path, parse_categorical, read_numeric_source, table_path_to_tnum,
    Delimiter, PrepareSpec,
};
use transformer::train::{
    evaluate_surrogate, predict_dataset, validate_train, LrSchedule, TrainConfig,
};
#[cfg(feature = "demo")]
use transformer::train::{train_text, TextTrainConfig};
use transformer::training::{
    evaluate_on, recommended_epoch, run_training, Dataset, EvalSchedule, Phase, TrainedModel,
    TrainingHistory, TrainingSetup,
};

/// Разобранные аргументы: `--key value` во flags, остальное — позиционные.
#[derive(Debug)]
struct Flags {
    flags: HashMap<String, String>,
    positionals: Vec<String>,
}

/// Допустимые флаги для train. Неизвестный флаг отвергается,
/// чтобы опечатка не привела к молчаливому обучению дефолтной конфигурации.
const TRAIN_FLAGS: &[&str] = &[
    "epochs",
    "eval-every",
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
const TRAIN_BOOL_FLAGS: &[&str] = &["diagnose", "kan-symbolic", "kan-compact", "interpret"];

/// Флаги подкоманды predict: табличная форма.
const PREDICT_FLAGS: &[&str] = &["table", "out"];

/// Флаги подкоманды prepare (таблица -> .tnum).
const PREPARE_FLAGS: &[&str] = &["inputs", "outputs", "delimiter", "categorical"];
const PREPARE_BOOL_FLAGS: &[&str] = &["has-header"];

/// Флаги подкоманды search (оси — CSV-списки).
const SEARCH_FLAGS: &[&str] = &[
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
                if flags.contains_key(key) {
                    return Err(format!("флаг --{key} указан повторно"));
                }
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

    fn require_positionals(&self, min: usize, max: usize, usage: &str) -> Result<(), String> {
        if (min..=max).contains(&self.positionals.len()) {
            return Ok(());
        }
        Err(format!(
            "неверное число позиционных аргументов (получено {}): ожидается {usage}",
            self.positionals.len(),
        ))
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
        // Явное выключение прунинга из CLI пока не требуется: флага для него
        // нет, поэтому переопределение только задаёт порог.
        prune: f.f32("kan-prune")?.map(Some),
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

/// Печать отчёта конвейера: сам конвейер живёт в [`transformer::interpret`],
/// здесь только вывод.
fn print_interpret_report(report: &InterpretReport) {
    if let Some(threshold) = report.profile.prune {
        println!("\nKAN prune (важность = p95 |φ| на train, порог {threshold} от максимума слоя):");
        for (l, (a, t)) in report.per_layer.iter().enumerate() {
            println!("  слой {l}: {a}/{t} активных рёбер");
        }
        if let (Some(before), Some(after), Some(ft)) = (
            report.r2_before,
            report.r2_after_prune,
            report.r2_after_finetune,
        ) {
            println!(
                "R² на validation: до прунинга {before:.5} -> после {after:.5} -> \
                 после fine-tune ({} эпох, λ=0) {ft:.5}",
                report.profile.finetune_epochs
            );
        }
    }
    let (active, total) = report.active_edges;
    println!("Активных рёбер: {active}/{total}");
    if let Some(c) = report.compaction {
        println!(
            "Структурное сжатие: скрытых узлов {} -> {}, параметров {} -> {}",
            c.nodes_before, c.nodes_after, c.params_before, c.params_after
        );
        if let (Some(before), Some(after)) = (
            report.r2_after_finetune.or(report.r2_before),
            report.r2_after_compact,
        ) {
            println!(
                "R² на validation: {before:.5} -> {after:.5} (удаление точное — совпадение ожидаемо)"
            );
        }
    }
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

/// Замена для команды, удалённой при переходе на train/search/predict/demo.
/// Держим один релиз: молчаливое «неизвестная команда» не подсказывает, куда
/// переехал сценарий.
fn renamed_command(name: &str) -> Option<&'static str> {
    match name {
        "numeric-file" => Some("используйте transformer train <данные>"),
        "numeric" => Some("используйте transformer demo train <чёрный ящик>"),
        "sweep" => Some("используйте transformer search <данные> или demo search <чёрный ящик>"),
        "epoch-sweep" => Some("используйте transformer train <данные> --eval-every N"),
        "text" => Some("используйте transformer demo text <файл.txt>"),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        // Без подкоманды — GUI (основной режим). CLI-подкоманды — по имени.
        None | Some("gui") => run_gui_cmd(),
        Some("train") => run_train(&args[2..]),
        Some("search") => run_search(&args[2..]),
        Some("prepare") => run_prepare(&args[2..]),
        Some("predict") => run_predict(&args[2..]),
        #[cfg(feature = "demo")]
        Some("demo") => run_demo(&args[2..]),
        // Демонстрации — отдельная фича: в сборке без них команда обязана
        // сказать это прямо, а не притвориться опечаткой.
        #[cfg(not(feature = "demo"))]
        Some("demo") => fail("сборка без демонстраций: пересоберите с --features demo"),
        Some(other) => {
            match renamed_command(other) {
                // Переименованную команду не выполняем догадкой: печатаем
                // замену и выходим, иначе старый скрипт молча сделает не то.
                Some(hint) => eprintln!("Команда «{other}» больше не существует: {hint}\n"),
                None => eprintln!("Неизвестная команда: {other}\n"),
            }
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Использование:");
    eprintln!("  transformer                 без аргументов — GUI (основной режим)");
    eprintln!("  transformer gui             явный запуск GUI");
    eprintln!("  transformer train <данные> [--epochs N] [--eval-every N] [--model out.bin]");
    eprintln!("                    данные: .tnum, XLSX, CSV или TSV");
    eprintln!("  transformer search <данные> [--model-kinds transformer,mlp,kan]");
    eprintln!("                    [--d-models 32,64 --layers-list 2,3 --mlp-widths 128]");
    eprintln!("                    [--kan-widths 16,32 --kan-layers-list 2 --kan-grids 8,16]");
    eprintln!(
        "  transformer prepare <input> <out.tnum> --inputs N --outputs M [--has-header] [--categorical 9:4]"
    );
    eprintln!("  transformer predict <model.bin> <v1> <v2> ...");
    eprintln!("  transformer predict <model.bin> --table <вход> --out <выход.xlsx>");
    #[cfg(feature = "demo")]
    {
        eprintln!("  transformer demo train <чёрный ящик> [флаги обучения]");
        eprintln!("  transformer demo search <чёрный ящик> [оси сетки]");
        eprintln!("  transformer demo text <file.txt> [steps]");
    }
    eprintln!("  флаги: --d-model --heads --layers --d-ff --lr --batch-size --seed");
    eprintln!("         --model-kind transformer|mlp|kan --mlp-width --mlp-layers");
    eprintln!("         --kan-width --kan-layers --kan-grid");
    eprintln!("         --eval-every N (кривая validation по эпохам и рекомендованная остановка)");
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
fn categorical_embedding_warning(kinds: &[ModelKind], schema: &ModelSchema) -> Option<String> {
    let categorical: Vec<&str> = schema
        .inputs()
        .iter()
        .filter(|c| c.cardinality().is_some())
        .map(|c| c.name())
        .collect();
    if categorical.is_empty() {
        return None;
    }
    let mut risky = Vec::new();
    for kind in kinds {
        let name = match kind {
            ModelKind::Transformer => continue,
            ModelKind::Mlp => "mlp",
            ModelKind::Kan => "kan",
        };
        if !risky.contains(&name) {
            risky.push(name);
        }
    }
    if risky.is_empty() {
        return None;
    }
    let models = if risky.len() == 1 {
        format!("модели {}", risky[0])
    } else {
        format!("моделях {}", risky.join(", "))
    };
    Some(format!(
        "ВНИМАНИЕ: категориальные входы ({}) в {models} кодируются числами —\n\
         \x20 порядок кодов будет воспринят как расстояние. Embedding категорий\n\
         \x20 есть только у transformer.",
        categorical.join(", ")
    ))
}

fn warn_categorical_without_embedding(kinds: &[ModelKind], schema: &ModelSchema) {
    if let Some(warning) = categorical_embedding_warning(kinds, schema) {
        println!("{warning}");
    }
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

/// Общий поток обучения для файла и встроенной демонстрационной задачи.
///
/// Сам сценарий живёт в [`transformer::training`]: здесь остаются только
/// печать и KAN-конвейер, который подключается хуком и потому применяется
/// одинаково к модели разработки и к финальной.
fn run_train_flow(
    f: &Flags,
    data: NumericDataset,
    schema: ModelSchema,
    reference: Option<&diagnostics::Reference>,
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
    warn_categorical_without_embedding(&[nc.kind], dataset.schema());
    let interpret = interpret_from(f, &nc).unwrap_or_else(|e| fail(&e));
    if let Some(profile) = &interpret {
        println!("Конвейер интерпретации {}", profile.describe());
    }
    validate_kan_symbolic(f, &nc).unwrap_or_else(|e| fail(&e));

    let mut setup = TrainingSetup::new(nc.clone(), tcfg.clone());
    // Кривая validation по эпохам — то, ради чего раньше был отдельный режим
    // epoch-sweep. Расписание проверяет само ядро: «замер реже, чем длится
    // обучение» станет ошибкой до первой эпохи, а не пустой кривой в конце.
    if let Some(n) = f.usize("eval-every").unwrap_or_else(|e| fail(&e)) {
        setup.eval = EvalSchedule::Every(n);
    }
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
            if let Some(profile) = &interpret {
                interpret::apply_l1(model, profile).unwrap_or_else(|e| fail(&e));
            }
        },
        &mut |phase, trained, train_data, eval| {
            // Один и тот же конвейер в обеих фазах: иначе сохранённая модель
            // отличалась бы от той, по которой принимали решения.
            let phase_tcfg = match phase {
                Phase::Development => &tcfg,
                Phase::Final => &final_tcfg,
            };
            apply_kan_pipeline(trained, train_data, eval, &interpret, phase_tcfg);
            // Рекомендация должна быть видна между development и refit, а не
            // после того, как финальная модель уже обучена и test открыт.
            if phase == Phase::Development {
                print_val_curve(&trained.history);
            }
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
            reference,
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

/// KAN-конвейер одной фазы: прунинг с fine-tune и структурное сжатие.
///
/// Сам конвейер общий для CLI и GUI; здесь остаётся только печать отчёта.
fn apply_kan_pipeline(
    trained: &mut TrainedModel,
    train_data: &NumericDataset,
    eval: Option<&NumericDataset>,
    profile: &Option<InterpretProfile>,
    tcfg: &TrainConfig,
) {
    let Some(profile) = profile else {
        return;
    };
    let report = interpret::run_pipeline(
        &mut trained.model,
        train_data,
        eval,
        &trained.in_norm,
        &trained.out_norm,
        tcfg,
        profile,
        // В CLI отменять некому: отмена приходит только из интерфейса.
        &std::sync::atomic::AtomicBool::new(false),
    )
    .unwrap_or_else(|e| fail(&e));
    print_interpret_report(&report);
}

/// `train` и `search` работают только с файлом. Существование проверяем до
/// чтения: иначе `train sum` жаловался бы на формат вместо того, чтобы указать
/// на `demo train sum`. Сам формат определяет читатель.
fn require_data_file(path: &str) {
    if std::path::Path::new(path).is_file() {
        return;
    }
    #[cfg(feature = "demo")]
    if blackbox::by_name(path).is_some() {
        fail(&format!(
            "«{path}» — встроенная задача, а не файл: используйте transformer demo train {path} \
             (или demo search {path})"
        ));
    }
    fail(&format!(
        "файл не найден: {path} (ожидается .tnum, XLSX, CSV или TSV)"
    ));
}

fn validate_train_positionals(f: &Flags) -> Result<(), String> {
    f.require_positionals(1, 3, "<источник> [эпохи] [модель.bin]")?;
    if f.has("epochs") && f.pos(1).is_some() {
        return Err("эпохи заданы и позиционно, и через --epochs".to_string());
    }
    if f.has("model") && f.pos(2).is_some() {
        return Err("путь модели задан и позиционно, и через --model".to_string());
    }
    Ok(())
}

#[cfg(feature = "demo")]
fn run_demo_train(rest: &[String]) {
    let f = Flags::parse(rest, TRAIN_FLAGS, TRAIN_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    // Имя ящика можно опустить: `demo train` остаётся коротким запуском sum.
    if !f.positionals.is_empty() {
        validate_train_positionals(&f).unwrap_or_else(|e| fail(&e));
    }
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
    // Встроенный ящик умеет считать сам себя, поэтому у демо есть эталон, с
    // которым можно сравнить чувствительность обученной модели.
    let eval = |x: &[f32]| bb.eval(x);
    let reference = diagnostics::Reference {
        n_inputs: bb.n_inputs(),
        n_outputs: bb.n_outputs,
        eval: &eval,
    };
    run_train_flow(&f, data, schema, Some(&reference));
}

fn run_train(rest: &[String]) {
    let f = Flags::parse(rest, TRAIN_FLAGS, TRAIN_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    validate_train_positionals(&f).unwrap_or_else(|e| fail(&e));
    let path = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите данные: transformer train <файл>"));
    require_data_file(path);

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
    run_train_flow(&f, data, schema, None);
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
    reference: Option<&diagnostics::Reference>,
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

    // 4. Чувствительность: модель — всегда, исходный процесс — только у демо.
    let report = diagnostics::sensitivity(
        val,
        specs,
        in_norm,
        out_norm,
        |inputs| {
            let ds =
                NumericDataset::new(inputs.clone(), Array2::zeros((inputs.nrows(), n_outputs)));
            predict_dataset(model, &ds, in_norm, out_norm)
        },
        reference,
        1.0,
        300,
    );
    match report {
        Ok(r) => {
            println!(
                "Чувствительность ||Δy||/||Δx|| (норм., {} пар соседних строк):",
                r.pairs
            );
            println!(
                "  модель:  среднее {:.2}, макс {:.2}",
                r.model.mean, r.model.max
            );
            match (r.reference, r.divergence) {
                (Some(reference), Some(divergence)) => {
                    println!(
                        "  процесс: среднее {:.2}, макс {:.2}",
                        reference.mean, reference.max
                    );
                    // Расхождение — это и есть диагностика: сама по себе
                    // чувствительность модели точности не доказывает.
                    println!(
                        "  расхождение средних: {divergence:.2} (надёжность видна по нему \
                         вместе с метриками на validation)"
                    );
                }
                _ => println!(
                    "  процесс: недоступен (чувствительность исходной функции \
                     известна только у встроенной задачи)"
                ),
            }
            if r.categorical_inputs > 0 {
                println!(
                    "  категориальные входы ({}) не возмущались: дробный шаг по коду \
                     не имеет смысла",
                    r.categorical_inputs
                );
            }
        }
        Err(e) => println!("Чувствительность: не посчитана — {e}"),
    }
}

/// Форма подкоманды predict. Их две, и они не смешиваются: «--table вместе со
/// значениями» — две разные просьбы в одной команде, и угадывать настоящую
/// нельзя.
#[derive(Debug)]
enum PredictForm {
    Table { input: String, output: String },
    Row(Vec<String>),
}

fn predict_form(
    table: Option<&str>,
    out: Option<&str>,
    positionals: &[String],
) -> Result<PredictForm, String> {
    match (table, out) {
        (Some(input), Some(output)) if positionals.is_empty() => Ok(PredictForm::Table {
            input: input.to_string(),
            output: output.to_string(),
        }),
        (Some(_), Some(_)) => Err("нельзя смешивать формы: либо --table и --out для \
             таблицы, либо значения одной строки"
            .to_string()),
        (Some(_), None) | (None, Some(_)) => {
            Err("для таблицы нужны оба флага: --table вход и --out выход".to_string())
        }
        (None, None) if positionals.is_empty() => Err(
            "укажите значения одной строки (predict model.bin 70 глина) или таблицу \
                 (predict model.bin --table вход.xlsx --out прогноз.xlsx)"
                .to_string(),
        ),
        (None, None) => Ok(PredictForm::Row(positionals.to_vec())),
    }
}

/// Отчёт об экспорте: сколько строк, что заменено и что добавлено.
fn print_export_summary(output: &str, summary: &batch_predict::ExportSummary) {
    println!("Таблица с прогнозами записана в {output}");
    println!("  строк: {}", summary.rows);
    if summary.extrapolated_rows > 0 {
        println!(
            "  вне обученного диапазона: {} строк",
            summary.extrapolated_rows
        );
    }
    if !summary.replaced.is_empty() {
        println!("  колонки заменены: {}", summary.replaced.join(", "));
    }
    if !summary.added.is_empty() {
        println!("  колонки добавлены: {}", summary.added.join(", "));
    }
    println!(
        "  результат — новая книга только со значениями: стили, формулы, другие \
         листы и структура исходного файла не сохраняются"
    );
}

fn run_predict(rest: &[String]) {
    let path = match rest.first() {
        Some(p) => p,
        None => {
            eprintln!("Укажите путь к сохранённой модели (.bin)");
            std::process::exit(1);
        }
    };
    let f = Flags::parse(&rest[1..], PREDICT_FLAGS, &[]).unwrap_or_else(|e| fail(&e));
    let form =
        predict_form(f.get("table"), f.get("out"), &f.positionals).unwrap_or_else(|e| fail(&e));

    let checkpoint = match load_numeric_full(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Не удалось загрузить {path}: {e}");
            std::process::exit(1);
        }
    };
    let schema = &checkpoint.schema;
    warn_categorical_without_embedding(&[checkpoint.config.kind], schema);

    // Таблица и одна строка идут через один слой схемы и одно ядро прогноза.
    let row = match &form {
        PredictForm::Table { input, output } => {
            let summary = batch_predict::export_predictions(input, output, schema, |inputs| {
                predict::predict_rows(
                    &checkpoint.model,
                    &checkpoint.in_norm,
                    &checkpoint.out_norm,
                    inputs,
                )
            })
            .unwrap_or_else(|e| fail(&e));
            print_export_summary(output, &summary);
            return;
        }
        PredictForm::Row(values) => values,
    };

    let cells: Vec<&str> = row.iter().map(String::as_str).collect();
    let values = predict::parse_row(schema, &cells, "").unwrap_or_else(|e| fail(&e));
    let inputs = Array2::from_shape_vec((1, values.len()), values.clone())
        .expect("одна строка нужной ширины");
    let result = predict::predict_rows(
        &checkpoint.model,
        &checkpoint.in_norm,
        &checkpoint.out_norm,
        &inputs,
    )
    .unwrap_or_else(|e| fail(&e));

    println!("Вход:");
    for (i, column) in schema.inputs().iter().enumerate() {
        match column.cardinality() {
            Some(_) => println!(
                "  {} = {} (код {})",
                column.display_name(),
                row[i],
                values[i] as usize
            ),
            None => println!("  {} = {}", column.display_name(), values[i]),
        }
    }
    println!("Выход:");
    for (j, column) in schema.outputs().iter().enumerate() {
        println!("  {} = {}", column.display_name(), result.outputs[[0, j]]);
    }
    for warning in &result.warnings {
        for d in &warning.details {
            println!(
                "ВНИМАНИЕ: {} = {} вне обученного диапазона [{}, {}]",
                schema.inputs()[d.feature].display_name(),
                d.value,
                d.min,
                d.max
            );
        }
    }
}

// --- validation-кривая по эпохам ---

/// Кривая validation по эпохам и рекомендованная остановка. Печатается только
/// при `--eval-every`: без замеров точек нет.
fn print_val_curve(history: &TrainingHistory) {
    let measured: Vec<(usize, f32, &Metrics)> = history
        .points
        .iter()
        .filter_map(|p| p.val.as_ref().map(|m| (p.epoch, p.train_loss, m)))
        .collect();
    if measured.is_empty() {
        return;
    }
    println!(
        "\nКривая development по эпохам ({}; до post-train конвейера):",
        history.source.label()
    );
    println!("epochs  train_loss     RMSE       MAE      rel.err        R²");
    for (epoch, loss, m) in &measured {
        println!(
            "{epoch:>6}  {loss:>10.5}  {:>9.5}  {:>9.5}  {:>7.2}%  {:>8.5}",
            m.rmse,
            m.mae,
            m.rel_error * 100.0,
            m.r2
        );
    }
    let r2s: Vec<f32> = measured.iter().map(|(_, _, m)| m.r2).collect();
    let losses: Vec<f32> = measured.iter().map(|(_, loss, _)| *loss).collect();
    println!("R²    {}", sparkline(&r2s));
    println!("loss  {}", sparkline(&losses));
    if let Some((epoch, why)) = recommended_epoch(
        measured.iter().map(|(epoch, _, m)| (*epoch, m.r2)),
        0.95,
        0.02,
        0.80,
    ) {
        println!("Рекомендованная остановка: {epoch} эпох ({why})");
    }
    if history.stopped_early {
        println!("Обучение остановлено ранней остановкой до последней эпохи.");
    }
}

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

fn run_prepare(rest: &[String]) {
    let f = Flags::parse(rest, PREPARE_FLAGS, PREPARE_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    f.require_positionals(2, 2, "prepare <input> <out.tnum>")
        .unwrap_or_else(|e| fail(&e));
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

// --- search ---

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

/// Поиск на своих данных: та же сетка, что и в demo, но датасет читается из
/// файла и разбивается общим протоколом.
fn run_search(rest: &[String]) {
    let f = Flags::parse(rest, SEARCH_FLAGS, &[]).unwrap_or_else(|e| fail(&e));
    f.require_positionals(1, 1, "search <файл>")
        .unwrap_or_else(|e| fail(&e));
    let path = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите данные: transformer search <файл>"));
    require_data_file(path);
    let (data, schema) = read_numeric_source(path).unwrap_or_else(|e| fail(&e));
    let dataset = Dataset::new(data, schema).unwrap_or_else(|e| fail(&e));
    let prepared = SplitPlan::default()
        .prepare(dataset.data())
        .unwrap_or_else(|e| fail(&e));

    let axes = axes_from(&f);
    warn_categorical_without_embedding(&axes.model_kinds, dataset.schema());
    announce_search(path, &axes);
    let never = std::sync::atomic::AtomicBool::new(false);
    let result = sweep_core::run_sweep(
        &dataset,
        &prepared.search,
        &axes,
        sweep_core::SweepObjective::default(),
        &never,
        print_search_row,
    )
    .unwrap_or_else(|e| fail(&e));
    print_search_ranking(&result);
}

#[cfg(feature = "demo")]
fn run_demo_search(rest: &[String]) {
    let f = Flags::parse(rest, SEARCH_FLAGS, &[]).unwrap_or_else(|e| fail(&e));
    f.require_positionals(1, 1, "demo search <чёрный ящик>")
        .unwrap_or_else(|e| fail(&e));
    let name = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите чёрный ящик: demo search <blackbox>"));
    blackbox::by_name(name).unwrap_or_else(|| fail(&format!("неизвестный чёрный ящик: {name}")));

    let axes = axes_from(&f);
    announce_search(name, &axes);
    let never = std::sync::atomic::AtomicBool::new(false);
    let result = sweep_core::run_blackbox_sweep(name, &axes, &never, print_search_row)
        .unwrap_or_else(|e| fail(&e));
    print_search_ranking(&result);
}

/// Цена операции — до запуска: она понятнее, чем название набора осей.
fn announce_search(source: &str, axes: &sweep_core::SweepAxes) {
    let cost = sweep_core::sweep_cost(axes, 1).unwrap_or_else(|e| fail(&e));
    println!("Поиск {source}: {}\n", cost.describe());
}

fn print_search_row(row: &sweep_core::SweepRow) {
    println!(
        "  done [{}]: {} -> worst R²={:.5}, aggregate R²={:.5}±{:.5}",
        row.source.label(),
        row.label,
        row.worst_output_r2_mean,
        row.r2_mean,
        row.r2_std
    );
}

/// Ранжирование: первая строка — рекомендация, по worst-output R² по умолчанию.
fn print_search_ranking(result: &sweep_core::SweepResult) {
    let source = result
        .rows
        .first()
        .map(|row| row.source.label())
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

/// Оси сетки из флагов: общие для поиска на своих данных и на чёрном ящике.
fn axes_from(f: &Flags) -> sweep_core::SweepAxes {
    let seeds = csv_u64(f, "seeds", "0");
    let d_models = csv_usize(f, "d-models", "32");
    let layers = csv_usize(f, "layers-list", "2");
    let d_ffs = csv_usize(f, "d-ffs", "64");
    let lrs = csv_f32(f, "lrs", "0.001");
    let vencs: Vec<&str> = f
        .get("value-encoders")
        .unwrap_or("linear")
        .split(',')
        .map(str::trim)
        .collect();
    let fscales = csv_f32(f, "fourier-scales", "2");
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

    sweep_core::SweepAxes {
        model_kinds,
        seeds,
        d_models,
        layers,
        d_ffs,
        lrs,
        value_encoders: vencs.iter().map(|v| parse_venc(v)).collect(),
        fourier_scales: fscales,
        fourier_bands: bands,
        mlp_widths: csv_usize(f, "mlp-widths", "128"),
        mlp_layers: csv_usize(f, "mlp-layers-list", "3"),
        kan_widths: csv_usize(f, "kan-widths", "16"),
        kan_layers: csv_usize(f, "kan-layers-list", "2"),
        kan_grids: csv_usize(f, "kan-grids", "8"),
        schedules: scheds.iter().map(|s| parse_sched(s)).collect(),
        epochs,
        final_epochs: epochs,
        batch_size: batch,
    }
}

/// demo: встроенные задачи и char-LM. К рабочему сценарию не относятся и
/// держатся отдельной командой, чтобы не мешаться со своими данными.
#[cfg(feature = "demo")]
fn run_demo(rest: &[String]) {
    match rest.first().map(String::as_str) {
        Some("train") => run_demo_train(&rest[1..]),
        Some("search") => run_demo_search(&rest[1..]),
        Some("text") => run_demo_text(&rest[1..]),
        Some(other) => fail(&format!(
            "неизвестная демонстрация: {other} (доступны train, search, text)"
        )),
        None => fail("укажите демонстрацию: demo train|search <чёрный ящик> | demo text <файл>"),
    }
}

#[cfg(feature = "demo")]
fn run_demo_text(rest: &[String]) {
    if !(1..=2).contains(&rest.len()) {
        fail("ожидалось: demo text <файл.txt> [steps]");
    }
    let path = match rest.first() {
        Some(p) => p,
        None => {
            eprintln!("Укажите путь к .txt файлу");
            std::process::exit(1);
        }
    };
    let steps: usize = rest
        .get(1)
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| fail(&format!("steps: ожидалось целое, получено '{s}'")))
        })
        .unwrap_or(2000);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn removed_commands_point_at_their_replacement() {
        for (old, hint) in [
            ("numeric-file", "train"),
            ("numeric", "demo train"),
            ("sweep", "search"),
            ("epoch-sweep", "--eval-every"),
            ("text", "demo text"),
        ] {
            let msg = renamed_command(old).unwrap_or_else(|| panic!("нет замены для {old}"));
            assert!(msg.contains(hint), "{old}: {msg}");
        }
        // Действующие команды не должны попадать в список замен.
        for live in ["train", "search", "predict", "prepare", "demo", "gui"] {
            assert!(renamed_command(live).is_none(), "{live}");
        }
    }

    #[test]
    fn cli_rejects_duplicate_flags_and_extra_positionals() {
        let duplicate = row(&["data.tnum", "--epochs", "2", "--epochs", "3"]);
        let err = Flags::parse(&duplicate, TRAIN_FLAGS, TRAIN_BOOL_FLAGS).unwrap_err();
        assert!(err.contains("указан повторно"), "{err}");

        let search = Flags::parse(&row(&["data.tnum", "лишнее"]), SEARCH_FLAGS, &[]).unwrap();
        let err = search
            .require_positionals(1, 1, "search <файл>")
            .unwrap_err();
        assert!(err.contains("получено 2"), "{err}");

        let mixed = Flags::parse(
            &row(&["data.tnum", "20", "--epochs", "40"]),
            TRAIN_FLAGS,
            TRAIN_BOOL_FLAGS,
        )
        .unwrap();
        let err = validate_train_positionals(&mixed).unwrap_err();
        assert!(err.contains("и позиционно, и через --epochs"), "{err}");
    }

    #[test]
    fn file_search_warns_about_categorical_codes_in_mlp_and_kan() {
        let schema =
            ModelSchema::synthetic_from_specs(&[FeatureSpec::Categorical { cardinality: 3 }], 1)
                .unwrap();
        assert!(categorical_embedding_warning(&[ModelKind::Transformer], &schema).is_none());
        let warning =
            categorical_embedding_warning(&[ModelKind::Mlp, ModelKind::Kan], &schema).unwrap();
        assert!(warning.contains("mlp, kan"), "{warning}");
        assert!(warning.contains("x0"), "{warning}");
    }

    #[test]
    fn table_form_needs_both_flags_and_no_values() {
        let both = predict_form(Some("in.xlsx"), Some("out.xlsx"), &[]).unwrap();
        match both {
            PredictForm::Table { input, output } => {
                assert_eq!((input.as_str(), output.as_str()), ("in.xlsx", "out.xlsx"));
            }
            PredictForm::Row(_) => panic!("ожидалась табличная форма"),
        }
        for (t, o) in [(Some("in.xlsx"), None), (None, Some("out.xlsx"))] {
            let e = predict_form(t, o, &[]).unwrap_err();
            assert!(e.contains("оба флага"), "{e}");
        }
    }

    #[test]
    fn forms_do_not_mix() {
        let e = predict_form(Some("in.xlsx"), Some("out.xlsx"), &row(&["70"])).unwrap_err();
        assert!(e.contains("смешивать"), "{e}");
        let e = predict_form(Some("in.xlsx"), None, &row(&["70"])).unwrap_err();
        assert!(e.contains("оба флага"), "{e}");
    }

    #[test]
    fn short_form_keeps_its_values_and_rejects_an_empty_call() {
        match predict_form(None, None, &row(&["70", "глина"])).unwrap() {
            PredictForm::Row(values) => assert_eq!(values, row(&["70", "глина"])),
            PredictForm::Table { .. } => panic!("ожидалась короткая форма"),
        }
        assert!(predict_form(None, None, &[]).is_err());
    }
}
