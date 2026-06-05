//! CLI: обучение surrogate-модели на чёрном ящике / .tnum файле, либо char-LM.
//!
//! Использование:
//!   transformer numeric <blackbox> [--epochs N] [--model out.bin] [конфиг-флаги]
//!   transformer numeric-file <file.tnum> [--epochs N] [--model out.bin] [флаги]
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
use transformer::data::{read_numeric_tnum, Normalizer, NumericDataset, TextDataset};
use transformer::diagnostics;
use transformer::encoders::{FeatureSpec, ValueEncoderConfig, ValueEncoderKind};
use transformer::epoch_sweep;
use transformer::generate::generate;
use transformer::init::set_init_seed;
use transformer::numeric_model::{validate_numeric, ModelKind, NumericConfig, NumericModel};
use transformer::serialize::{load_numeric, save_numeric};
use transformer::sweep as sweep_core;
use transformer::tensor::Tensor;
use transformer::textmodel::TextModel;
use transformer::tnum::{parse_categorical, table_to_tnum, Delimiter, PrepareSpec};
use transformer::train::{
    evaluate_surrogate, predict_dataset, train_surrogate, train_text, validate_train, LrSchedule,
    TextTrainConfig, TrainConfig,
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
    "lr",
    "batch-size",
    "seed",
    "scheduler",
    "warmup",
    "min-lr-ratio",
];

/// Булевы флаги (без значения).
const NUMERIC_BOOL_FLAGS: &[&str] = &["diagnose"];

/// Флаги подкоманды prepare (таблица -> .tnum).
const PREPARE_FLAGS: &[&str] = &["inputs", "outputs", "delimiter", "categorical"];
const PREPARE_BOOL_FLAGS: &[&str] = &["has-header"];

/// Флаги подкоманды sweep (оси — CSV-списки).
const SWEEP_FLAGS: &[&str] = &[
    "seeds",
    "d-models",
    "layers-list",
    "d-ffs",
    "lrs",
    "value-encoders",
    "fourier-scales",
    "fourier-bands",
    "schedulers",
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
        other => {
            return Err(format!(
                "--model-kind: ожидалось transformer|mlp, получено '{other}'"
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
    };
    validate_numeric(&nc)?;
    Ok(nc)
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
    eprintln!("  transformer numeric-file <file.tnum> [--epochs N] [--model out.bin] [флаги]");
    eprintln!("  transformer sweep <blackbox> [--d-models 32,64 --layers-list 2,3 ...]");
    eprintln!("  transformer epoch-sweep <data.tnum> [--epochs 1,2,5,10,20,40] [конфиг-флаги]");
    eprintln!(
        "  transformer prepare <input> <out.tnum> --inputs N --outputs M [--has-header] [--categorical 9:4]"
    );
    eprintln!("  transformer text <file.txt> [steps]");
    eprintln!("  transformer predict <model.bin> <v1> <v2> ...");
    eprintln!("  флаги: --d-model --heads --layers --d-ff --lr --batch-size --seed");
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

fn report_and_eval(
    model: &NumericModel,
    train: &NumericDataset,
    test: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    tcfg: &TrainConfig,
) {
    println!(
        "Обучение: {} эпох, {} train / {} test...",
        tcfg.epochs,
        train.len(),
        test.len()
    );
    let history = train_surrogate(model, train, in_norm, out_norm, tcfg);
    for (e, loss) in history.iter().enumerate() {
        if e % 5 == 0 || e + 1 == history.len() {
            println!("  эпоха {e:>3}: train loss (норм.) = {loss:.5}");
        }
    }
    let m = evaluate_surrogate(model, test, in_norm, out_norm);
    println!("\nМетрики на test (в исходных единицах):");
    println!("  RMSE        = {:.5}", m.rmse);
    println!("  MAE         = {:.5}", m.mae);
    println!("  rel. error  = {:.2}%", m.rel_error * 100.0);
    println!("  R²          = {:.5}", m.r2);
}

fn run_numeric(args: &[String]) {
    let f =
        Flags::parse(&args[2..], NUMERIC_FLAGS, NUMERIC_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    let name = f.pos(0).unwrap_or("sum");
    let epochs = resolve_epochs(&f);
    let save_path = f.get("model").or_else(|| f.pos(2));
    // Конфиг строим сразу — битый флаг падает до любой работы.
    let nc = numeric_config_from(&f).unwrap_or_else(|e| fail(&e));
    let tcfg = train_config_from(&f, epochs).unwrap_or_else(|e| fail(&e));

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

    let data = bb.generate(2000, 0);
    let (train, test) = data.split(0.8, 1);
    let in_specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
    let in_norm = Normalizer::fit(&train.inputs, &in_specs);
    let out_norm = Normalizer::fit(&train.outputs, &Normalizer::all_continuous(bb.n_outputs));

    print_config(&nc, &tcfg);

    set_init_seed(tcfg.seed); // воспроизводимая инициализация
    let model = nc.build(&in_specs, bb.n_outputs);
    report_and_eval(&model, &train, &test, &in_norm, &out_norm, &tcfg);

    if let Some(path) = save_path {
        save_and_verify(
            path,
            &nc,
            &in_specs,
            bb.n_outputs,
            &model,
            &in_norm,
            &out_norm,
            &test,
        );
    }

    if f.has("diagnose") {
        run_diagnostics(
            &nc,
            &in_specs,
            bb.n_outputs,
            &train,
            &test,
            &in_norm,
            &out_norm,
            &model,
            Some(&bb),
        );
    }
}

fn run_numeric_file(args: &[String]) {
    let f =
        Flags::parse(&args[2..], NUMERIC_FLAGS, NUMERIC_BOOL_FLAGS).unwrap_or_else(|e| fail(&e));
    let path = match f.pos(0) {
        Some(p) => p,
        None => {
            eprintln!("Укажите путь к .tnum файлу");
            std::process::exit(1);
        }
    };
    let epochs = resolve_epochs(&f);
    let save_path = f.get("model").or_else(|| f.pos(2));
    // Конфиг строим сразу — битый флаг падает до любой работы.
    let nc = numeric_config_from(&f).unwrap_or_else(|e| fail(&e));
    let tcfg = train_config_from(&f, epochs).unwrap_or_else(|e| fail(&e));

    let (data, in_specs) = match read_numeric_tnum(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Не удалось прочитать {path}: {e}");
            std::process::exit(1);
        }
    };
    let n_outputs = data.outputs.ncols();
    println!(
        "Датасет: {path} ({} строк, {} вход -> {} выход)",
        data.len(),
        data.inputs.ncols(),
        n_outputs
    );

    let (train, test) = data.split(0.8, 1);
    let in_norm = Normalizer::fit(&train.inputs, &in_specs);
    let out_norm = Normalizer::fit(&train.outputs, &Normalizer::all_continuous(n_outputs));

    print_config(&nc, &tcfg);

    set_init_seed(tcfg.seed); // воспроизводимая инициализация
    let model = nc.build(&in_specs, n_outputs);
    report_and_eval(&model, &train, &test, &in_norm, &out_norm, &tcfg);

    if let Some(path) = save_path {
        save_and_verify(
            path, &nc, &in_specs, n_outputs, &model, &in_norm, &out_norm, &test,
        );
    }

    if f.has("diagnose") {
        run_diagnostics(
            &nc, &in_specs, n_outputs, &train, &test, &in_norm, &out_norm, &model, None,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn save_and_verify(
    path: &str,
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    num_outputs: usize,
    model: &NumericModel,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    test: &NumericDataset,
) {
    save_numeric(path, nc, specs, num_outputs, model, in_norm, out_norm)
        .expect("сохранение модели");
    let (loaded, in2, out2) = load_numeric(path).expect("загрузка модели");
    let m2 = evaluate_surrogate(&loaded, test, &in2, &out2);
    println!(
        "\nМодель сохранена в {path}; R² после загрузки = {:.5}",
        m2.r2
    );
}

#[allow(clippy::too_many_arguments)]
fn run_diagnostics(
    nc: &NumericConfig,
    specs: &[FeatureSpec],
    n_outputs: usize,
    train: &NumericDataset,
    test: &NumericDataset,
    in_norm: &Normalizer,
    out_norm: &Normalizer,
    model: &NumericModel,
    bb: Option<&blackbox::BlackBox>,
) {
    println!("\n=== ДИАГНОСТИКА ===");

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

    // 2. Покрытие: экстраполяция на test.
    let rr = diagnostics::range_report(in_norm, &test.inputs);
    println!(
        "Экстраполяция: {} из {} test-строк вне обученного диапазона",
        rr.rows_out, rr.total
    );

    // 3. Форма остатка по входным признакам.
    let pred = predict_dataset(model, test, in_norm, out_norm);
    let res = diagnostics::residual_diagnostics(&test.inputs, &pred, &test.outputs);
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
        None => println!("Чувствительность: пропущена (нет вызываемого ящика для .tnum)"),
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
    let mut values = Vec::new();
    for s in &args[3..] {
        match s.parse::<f32>() {
            Ok(v) => values.push(v),
            Err(_) => fail(&format!("вход '{s}' не является числом")),
        }
    }
    if values.is_empty() {
        fail("укажите входные значения: transformer predict model.bin v1 v2 ...");
    }

    let (model, in_norm, out_norm) = match load_numeric(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Не удалось загрузить {path}: {e}");
            std::process::exit(1);
        }
    };

    if values.len() != in_norm.n_features() {
        fail(&format!(
            "модель ожидает {} входов, получено {}",
            in_norm.n_features(),
            values.len()
        ));
    }

    let f = values.len();
    let raw = Array2::from_shape_vec((1, f), values.clone()).unwrap();
    let x = Tensor::constant(in_norm.transform(&raw).into_dyn());
    let pred_norm = model
        .predict(&x)
        .data()
        .into_dimensionality::<Ix2>()
        .expect("predict возвращает [1, O]");
    let pred = out_norm.inverse_transform(&pred_norm);

    println!("Вход:  {values:?}");
    println!("Выход: {:?}", pred.row(0).to_vec());
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

    let path = f
        .pos(0)
        .unwrap_or_else(|| fail("укажите .tnum: epoch-sweep <data.tnum>"));
    let (data, specs) =
        read_numeric_tnum(path).unwrap_or_else(|e| fail(&format!("чтение {path}: {e}")));
    let n_out = data.outputs.ncols();

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

    let (train, test) = data.split(0.8, 1);
    let in_norm = Normalizer::fit(&train.inputs, &specs);
    let out_norm = Normalizer::fit(&train.outputs, &Normalizer::all_continuous(n_out));
    let nc = numeric_config_from(&f).unwrap_or_else(|e| fail(&e));
    let max_e = milestones.iter().copied().max().unwrap_or(1);
    let base_tcfg = train_config_from(&f, max_e).unwrap_or_else(|e| fail(&e));

    println!(
        "Epoch-sweep {path}: эпохи {milestones:?}, {} train / {} test",
        train.len(),
        test.len()
    );
    let rows = epoch_sweep::run_epoch_sweep(
        &train,
        &test,
        &in_norm,
        &out_norm,
        &nc,
        &specs,
        n_out,
        &base_tcfg,
        &milestones,
    );

    println!("\nepochs  train_loss     RMSE       MAE      rel.err     R²");
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
    let n_inputs = f
        .usize("inputs")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or_else(|| fail("--inputs обязателен"));
    let n_outputs = f
        .usize("outputs")
        .unwrap_or_else(|e| fail(&e))
        .unwrap_or_else(|| fail("--outputs обязателен"));
    let delimiter = match f.get("delimiter").unwrap_or("auto") {
        "auto" => Delimiter::Auto,
        "comma" => Delimiter::Comma,
        "tab" => Delimiter::Tab,
        "space" => Delimiter::Space,
        o => fail(&format!(
            "--delimiter: ожидалось auto|comma|tab|space, получено '{o}'"
        )),
    };
    let categorical = parse_categorical(f.get("categorical").unwrap_or(""), n_inputs)
        .unwrap_or_else(|e| fail(&e));

    let spec = PrepareSpec {
        n_inputs,
        n_outputs,
        delimiter,
        has_header: f.has("has-header"),
        categorical,
    };

    let text =
        std::fs::read_to_string(input).unwrap_or_else(|e| fail(&format!("чтение {input}: {e}")));
    let tnum = table_to_tnum(&text, &spec).unwrap_or_else(|e| fail(&e));
    let rows = tnum.lines().count().saturating_sub(6); // 6 строк заголовка
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

    let axes = sweep_core::SweepAxes {
        seeds,
        d_models,
        layers,
        d_ffs,
        lrs,
        value_encoders: vencs.iter().map(|v| parse_venc(v)).collect(),
        fourier_scales: fscales,
        fourier_bands: bands,
        schedules: scheds.iter().map(|s| parse_sched(s)).collect(),
        epochs,
        batch_size: batch,
    };
    let (total_configs, total_runs) = sweep_core::sweep_size(&axes).unwrap_or_else(|e| fail(&e));

    println!(
        "Sweep {name}: {} конфигов × {} seed = {} прогонов\n",
        total_configs,
        axes.seeds.len(),
        total_runs
    );

    let never = std::sync::atomic::AtomicBool::new(false);
    let result = sweep_core::run_blackbox_sweep(name, &axes, &never, |row| {
        println!(
            "  done: {} -> R²={:.5}±{:.5}",
            row.label, row.r2_mean, row.r2_std
        );
    })
    .unwrap_or_else(|e| fail(&e));

    // Ранжируем по среднему R² (рекомендация — первая строка).
    println!("\n=== РАНЖИРОВАНИЕ (по среднему R²; rel — справочно) ===");
    for (i, r) in result.rows.iter().enumerate() {
        let mark = if i == 0 { "*" } else { " " };
        println!(
            "{mark} R²={:.5}±{:.5}  nRMSE={:.5}  rel={:.1}%  | {}",
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
