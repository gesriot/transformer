//! Worker-поток: владеет Rc-состоянием моделей, выполняет долгие
//! задачи, общается с UI каналами. Отмена — кооперативно через `Arc<AtomicBool>`,
//! проверяемый внутри батч-цикла `train_surrogate_cb` (не ждёт конца эпохи).
//!
//! Обученная/загруженная модель (Rc !Send) живёт ЗДЕСЬ (`current`) и используется
//! для Predict; UI получает только числа/статусы.

use super::messages::{
    Command, DataSource, DiagnosticsResult, Event, KanModelInfo, KanSymbolicInfo, KanWeakEdge,
};
use crate::batch_predict::{read_prediction_xlsx, write_prediction_xlsx};
use crate::blackbox;
use crate::config::ModelConfig;
use crate::data::{read_numeric_tnum, Normalizer, NumericDataset, OutOfRange, TextDataset};
use crate::encoders::FeatureSpec;
use crate::epoch_sweep::{self, EpochRow};
use crate::generate::generate;
use crate::init::set_init_seed;
use crate::metrics::evaluate;
use crate::numeric_model::{validate_numeric, NumericConfig, NumericModel};
use crate::serialize::{calibration_sample, load_numeric_full, save_numeric};
use crate::split::{SplitPlan, DEFAULT_DATA_SEED};
use crate::sweep::{self, SweepAxes, SweepObjective};
use crate::symbolic;
use crate::tensor::Tensor;
use crate::textmodel::TextModel;
use crate::tnum::{table_path_to_tnum, PrepareSpec};
use crate::train::{
    evaluate_surrogate, fit_normalizers, predict_dataset, train_surrogate_cb, train_text_cb,
    validate_train, TextTrainConfig, TrainConfig,
};
use eframe::egui;
use ndarray::{Array2, Ix2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub struct Worker {
    cmd_tx: Sender<Command>,
    evt_rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(ctx: egui::Context) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (evt_tx, evt_rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_w = cancel.clone();
        let handle = thread::spawn(move || worker_loop(cmd_rx, evt_tx, ctx, cancel_w));
        Self {
            cmd_tx,
            evt_rx,
            cancel,
            handle: Some(handle),
        }
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.cmd_tx.send(cmd);
    }
    pub fn try_recv(&self) -> Option<Event> {
        self.evt_rx.try_recv().ok()
    }
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn reset_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed); // прервать обучение, если идёт
        let _ = self.cmd_tx.send(Command::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Текущая модель + нормализаторы (Rc !Send) — живёт в worker-потоке.
struct Loaded {
    model: NumericModel,
    nc: NumericConfig,
    in_specs: Vec<FeatureSpec>,
    in_norm: Normalizer,
    out_norm: Normalizer,
    n_inputs: usize,
    n_outputs: usize,
    /// Данные обучения для диагностики (`None` для загруженной `.bin`).
    diag: Option<DiagData>,
    /// Калибровочная выборка сырых train-входов: у загруженного checkpoint-а
    /// берётся из секции `calibration`, у обученной модели — из train.
    calibration: Option<Array2<f32>>,
}

/// Данные сессии обучения, нужные для диагностики.
struct DiagData {
    nc: NumericConfig,
    source: DataSource,
    in_specs: Vec<FeatureSpec>,
    train: NumericDataset,
    /// Validation текущей сессии. Test сюда не попадает: он отложен в
    /// `HoldoutTest` и в GUI пока не открывается вовсе.
    val: NumericDataset,
}

struct LoadedText {
    model: TextModel,
    dataset: TextDataset,
    ctx_len: usize,
    tgt_len: usize,
}

fn worker_loop(
    cmd_rx: Receiver<Command>,
    evt_tx: Sender<Event>,
    ctx: egui::Context,
    cancel: Arc<AtomicBool>,
) {
    let _ = evt_tx.send(Event::Status("worker запущен".to_string()));
    ctx.request_repaint();
    let mut current: Option<Loaded> = None;
    let mut current_text: Option<LoadedText> = None;

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            Command::TrainNumeric { source, nc, tcfg } => {
                match train_numeric(&source, &nc, &tcfg, &evt_tx, &ctx, &cancel) {
                    Ok(Some(loaded)) => {
                        let _ = evt_tx.send(Event::ModelReady {
                            n_inputs: loaded.n_inputs,
                            n_outputs: loaded.n_outputs,
                            source: source_desc(&source),
                            parameter_count: loaded.model.parameter_count(),
                            kan: kan_model_info(
                                &loaded.model,
                                loaded.diag.is_some() || loaded.calibration.is_some(),
                            ),
                        });
                        current = Some(loaded);
                        ctx.request_repaint();
                    }
                    Ok(None) => {} // отменено — текущую модель не трогаем
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                        ctx.request_repaint();
                    }
                }
            }
            Command::LoadModel(path) => {
                match load_model(&path) {
                    Ok(loaded) => {
                        let _ = evt_tx.send(Event::ModelReady {
                            n_inputs: loaded.n_inputs,
                            n_outputs: loaded.n_outputs,
                            source: format!("файл: {path}"),
                            parameter_count: loaded.model.parameter_count(),
                            kan: kan_model_info(
                                &loaded.model,
                                loaded.diag.is_some() || loaded.calibration.is_some(),
                            ),
                        });
                        current = Some(loaded);
                        let _ = evt_tx.send(Event::Status("модель загружена".to_string()));
                    }
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                    }
                }
                ctx.request_repaint();
            }
            Command::SaveModel(path) => {
                match &current {
                    Some(loaded) => match save_model(loaded, &path) {
                        Ok(()) => {
                            let _ = evt_tx.send(Event::Status(format!("модель сохранена: {path}")));
                        }
                        Err(e) => {
                            let _ = evt_tx.send(Event::Error(e));
                        }
                    },
                    None => {
                        let _ = evt_tx.send(Event::Error(
                            "нет модели: сначала обучите или загрузите .bin".to_string(),
                        ));
                    }
                }
                ctx.request_repaint();
            }
            Command::Diagnose => {
                match &current {
                    Some(l) => match diagnose(l) {
                        Ok(result) => {
                            let _ = evt_tx.send(Event::Diagnostics { result });
                        }
                        Err(e) => {
                            let _ = evt_tx.send(Event::Error(e));
                        }
                    },
                    None => {
                        let _ = evt_tx.send(Event::Error("нет модели для диагностики".to_string()));
                    }
                }
                ctx.request_repaint();
            }
            Command::Predict(values) => {
                match &current {
                    Some(l) => match do_predict(l, &values) {
                        Ok((outputs, extrapolation)) => {
                            let _ = evt_tx.send(Event::PredictResult {
                                outputs,
                                extrapolation,
                            });
                        }
                        Err(e) => {
                            let _ = evt_tx.send(Event::Error(e));
                        }
                    },
                    None => {
                        let _ = evt_tx.send(Event::Error(
                            "нет модели: обучите или загрузите .bin".to_string(),
                        ));
                    }
                }
                ctx.request_repaint();
            }
            Command::PredictFile { input, output } => {
                match &current {
                    Some(l) => match do_predict_file(l, &input, &output) {
                        Ok((rows, extrapolation_rows)) => {
                            let _ = evt_tx.send(Event::PredictFileDone {
                                output,
                                rows,
                                extrapolation_rows,
                            });
                        }
                        Err(e) => {
                            let _ = evt_tx.send(Event::Error(e));
                        }
                    },
                    None => {
                        let _ = evt_tx.send(Event::Error(
                            "нет модели: обучите или загрузите .bin".to_string(),
                        ));
                    }
                }
                ctx.request_repaint();
            }
            Command::SampleKanEdge {
                layer,
                input,
                output,
                samples,
            } => {
                match &current {
                    Some(loaded) => {
                        match sample_kan_edge(&loaded.model, layer, input, output, samples) {
                            Ok(points) => {
                                let _ = evt_tx.send(Event::KanEdgeCurve {
                                    layer,
                                    input,
                                    output,
                                    points,
                                });
                            }
                            Err(e) => {
                                let _ = evt_tx.send(Event::Error(e));
                            }
                        }
                    }
                    None => {
                        let _ = evt_tx.send(Event::Error(
                            "нет модели: обучите или загрузите KAN".to_string(),
                        ));
                    }
                }
                ctx.request_repaint();
            }
            Command::ExtractKanSymbolic => {
                match &current {
                    Some(loaded) => match extract_kan_symbolic(loaded) {
                        Ok(result) => {
                            let _ = evt_tx.send(Event::KanSymbolic { result });
                        }
                        Err(e) => {
                            let _ = evt_tx.send(Event::Error(e));
                        }
                    },
                    None => {
                        let _ = evt_tx
                            .send(Event::Error("нет модели: сначала обучите KAN".to_string()));
                    }
                }
                ctx.request_repaint();
            }
            Command::Sweep { blackbox, axes } => {
                match run_sweep(&blackbox, &axes, &evt_tx, &ctx, &cancel) {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                        ctx.request_repaint();
                    }
                }
            }
            Command::OptimizeFile {
                path,
                axes,
                objective,
            } => match run_optimize_file(&path, &axes, objective, &evt_tx, &ctx, &cancel) {
                Ok(()) => {}
                Err(e) => {
                    let _ = evt_tx.send(Event::Error(e));
                    ctx.request_repaint();
                }
            },
            Command::TrainText {
                path,
                model_cfg,
                train_cfg,
            } => match train_text(&path, &model_cfg, &train_cfg, &evt_tx, &ctx, &cancel) {
                Ok(Some(loaded)) => {
                    current_text = Some(loaded);
                }
                Ok(None) => {}
                Err(e) => {
                    let _ = evt_tx.send(Event::Error(e));
                    ctx.request_repaint();
                }
            },
            Command::GenerateText {
                seed,
                total_new,
                temperature,
                top_k,
                rng_seed,
            } => {
                match &current_text {
                    Some(t) => {
                        match generate_text(t, &seed, total_new, temperature, top_k, rng_seed) {
                            Ok(text) => {
                                let _ = evt_tx.send(Event::GeneratedText { text });
                            }
                            Err(e) => {
                                let _ = evt_tx.send(Event::Error(e));
                            }
                        }
                    }
                    None => {
                        let _ = evt_tx
                            .send(Event::Error("нет text-модели: сначала обучите".to_string()));
                    }
                }
                ctx.request_repaint();
            }
            Command::Prepare {
                input,
                output,
                spec,
            } => {
                match prepare_tnum(&input, &output, &spec) {
                    Ok((rows, n_inputs, n_outputs)) => {
                        let _ = evt_tx.send(Event::PrepareDone {
                            output,
                            rows,
                            n_inputs,
                            n_outputs,
                        });
                    }
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                    }
                }
                ctx.request_repaint();
            }
            Command::EpochSweep {
                path,
                nc,
                base_tcfg,
                milestones,
                target_r2,
                min_gain,
                plateau_min,
            } => {
                match run_epoch_sweep(
                    &path,
                    &nc,
                    &base_tcfg,
                    &milestones,
                    target_r2,
                    min_gain,
                    plateau_min,
                    &evt_tx,
                    &ctx,
                    &cancel,
                ) {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                        ctx.request_repaint();
                    }
                }
            }
            Command::Shutdown => break,
        }
    }
}

fn source_desc(s: &DataSource) -> String {
    match s {
        DataSource::Blackbox(name) => format!("blackbox: {name}"),
        DataSource::File(path) => format!("файл: {path}"),
    }
}

fn kan_model_info(model: &NumericModel, symbolic_available: bool) -> Option<KanModelInfo> {
    model.as_kan().map(|kan| KanModelInfo {
        layer_dims: kan.layer_dims(),
        domain: kan.domain(),
        symbolic_available,
    })
}

/// Вычисляет выборку одной функции ребра в worker-потоке. Это не даёт
/// `Rc`-тензорам пересечь границу потока и защищает GUI от неверных индексов.
fn sample_kan_edge(
    model: &NumericModel,
    layer: usize,
    input: usize,
    output: usize,
    samples: usize,
) -> Result<Vec<(f32, f32)>, String> {
    let kan = model
        .as_kan()
        .ok_or_else(|| "графики функций доступны только для KAN".to_string())?;
    if samples < 2 {
        return Err("для графика нужно минимум 2 точки".to_string());
    }
    let (n_inputs, n_outputs) = *kan
        .layer_dims()
        .get(layer)
        .ok_or_else(|| format!("KAN: нет слоя {layer}"))?;
    if input >= n_inputs {
        return Err(format!("KAN: в слое {layer} нет входа {input}"));
    }
    if output >= n_outputs {
        return Err(format!("KAN: в слое {layer} нет выхода {output}"));
    }

    let (min, max) = kan.domain();
    let step = (max - min) / (samples - 1) as f32;
    let xs: Vec<f32> = (0..samples).map(|i| min + i as f32 * step).collect();
    let ys = kan.edge_curve(layer, input, output, &xs);
    Ok(xs.into_iter().zip(ys).collect())
}

/// Извлекает формулы по реальным train-активациям и возвращает только
/// serializable данные для UI. Checkpoint без train-набора не подходит:
/// равномерная подмена калибровки исказила бы глубокие рёбра.
fn extract_kan_symbolic(loaded: &Loaded) -> Result<KanSymbolicInfo, String> {
    let kan = loaded
        .model
        .as_kan()
        .ok_or_else(|| "символьные формулы доступны только для KAN".to_string())?;
    // Реальные активации: train текущей сессии либо калибровка из checkpoint-а.
    let raw_inputs: &Array2<f32> = match (&loaded.diag, &loaded.calibration) {
        (Some(diag), _) => &diag.train.inputs,
        (None, Some(calib)) => calib,
        (None, None) => {
            return Err(
                "формулы недоступны: checkpoint без секции calibration и без train-данных"
                    .to_string(),
            )
        }
    };
    let calibration = loaded.in_norm.transform(raw_inputs);
    let symbolic =
        symbolic::symbolize(kan, &calibration, 256).denormalize(&loaded.in_norm, &loaded.out_norm);
    let (min_edge_r2, mean_edge_r2) = symbolic.edge_r2_stats();
    let weak_edges = symbolic
        .weak_edges(0.99)
        .into_iter()
        .map(|edge| KanWeakEdge {
            layer: edge.layer,
            input: edge.input,
            output: edge.output,
            primitive: edge.name.to_string(),
            r2: edge.r2,
        })
        .collect();
    // Метрики формул есть только у модели, обученной в этой сессии, и считаются
    // на validation: test не открывается.
    let (formula_metrics, kan_r2) = match &loaded.diag {
        Some(diag) => (
            Some(evaluate(
                &symbolic.predict(&diag.val.inputs),
                &diag.val.outputs,
            )),
            Some(
                evaluate_surrogate(&loaded.model, &diag.val, &loaded.in_norm, &loaded.out_norm).r2,
            ),
        ),
        None => (None, None),
    };
    Ok(KanSymbolicInfo {
        formulas: symbolic.formulas(),
        min_edge_r2,
        mean_edge_r2,
        formula_metrics,
        kan_r2,
        weak_edges,
    })
}

fn load_model(path: &str) -> Result<Loaded, String> {
    let checkpoint = load_numeric_full(path).map_err(|e| format!("загрузка {path}: {e}"))?;
    let n_inputs = checkpoint.in_norm.n_features();
    Ok(Loaded {
        model: checkpoint.model,
        nc: checkpoint.config,
        in_specs: checkpoint.specs,
        in_norm: checkpoint.in_norm,
        out_norm: checkpoint.out_norm,
        n_inputs,
        n_outputs: checkpoint.num_outputs,
        diag: None, // у загруженной .bin нет данных обучения
        calibration: checkpoint.calibration,
    })
}

fn save_model(loaded: &Loaded, path: &str) -> Result<(), String> {
    save_numeric(
        path,
        &loaded.nc,
        &loaded.in_specs,
        loaded.n_outputs,
        &loaded.model,
        &loaded.in_norm,
        &loaded.out_norm,
        loaded.calibration.as_ref(),
    )
    .map_err(|e| format!("сохранение {path}: {e}"))
}

/// Диагностика обученной модели. Доступна только если есть
/// данные обучения (`diag`).
fn diagnose(l: &Loaded) -> Result<DiagnosticsResult, String> {
    let d = l.diag.as_ref().ok_or_else(|| {
        "диагностика доступна после обучения (не для загруженной .bin)".to_string()
    })?;

    let m = d.train.len().min(48);
    let subset = d.train.gather(&(0..m).collect::<Vec<_>>());
    let overfit_loss =
        crate::diagnostics::overfit_probe(&d.nc, &d.in_specs, l.n_outputs, &subset, 80);

    let rr = crate::diagnostics::range_report(&l.in_norm, &d.val.inputs);
    let pred = predict_dataset(&l.model, &d.val, &l.in_norm, &l.out_norm);
    let res = crate::diagnostics::residual_diagnostics(&d.val.inputs, &pred, &d.val.outputs);
    let residuals = res
        .iter()
        .map(|r| (r.sign_change_rate, r.tail_ratio))
        .collect();

    let sensitivity = match &d.source {
        DataSource::Blackbox(name) => blackbox::by_name(name).map(|bb| {
            crate::diagnostics::sensitivity_probe(&bb, &l.in_norm, &l.out_norm, 300, 0.01, 0)
        }),
        DataSource::File(_) => None,
    };

    Ok(DiagnosticsResult {
        overfit_loss,
        extrapolation_rows: rr.rows_out,
        extrapolation_total: rr.total,
        residuals,
        sensitivity,
    })
}

fn do_predict(l: &Loaded, values: &[f32]) -> Result<(Vec<f32>, Vec<OutOfRange>), String> {
    if values.len() != l.n_inputs {
        return Err(format!(
            "ожидалось {} входов, получено {}",
            l.n_inputs,
            values.len()
        ));
    }
    let raw = Array2::from_shape_vec((1, l.n_inputs), values.to_vec()).unwrap();
    let extrapolation = l.in_norm.out_of_range_details(values);
    let x = Tensor::constant(l.in_norm.transform(&raw).into_dyn());
    let pred_norm = l
        .model
        .predict(&x)
        .data()
        .into_dimensionality::<Ix2>()
        .map_err(|_| "predict вернул неверную форму".to_string())?;
    let pred = l.out_norm.inverse_transform(&pred_norm);
    Ok((pred.row(0).to_vec(), extrapolation))
}

fn do_predict_file(l: &Loaded, input: &str, output: &str) -> Result<(usize, usize), String> {
    let sheet = read_prediction_xlsx(input, l.n_inputs, l.n_outputs)?;
    let input_rows = sheet.input_rows()?;
    if input_rows.is_empty() {
        return Err("Excel-файл не содержит строк с входами".to_string());
    }

    let mut predictions = Vec::with_capacity(input_rows.len());
    let mut extrapolation_rows = 0;
    for values in &input_rows {
        let (pred, extrapolation) = do_predict(l, values)?;
        if !extrapolation.is_empty() {
            extrapolation_rows += 1;
        }
        predictions.push(pred);
    }

    let filled = sheet.fill_outputs(&predictions)?;
    write_prediction_xlsx(output, &filled)?;
    Ok((predictions.len(), extrapolation_rows))
}

fn run_sweep(
    blackbox: &str,
    axes: &SweepAxes,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let (total_configs, total_runs) = sweep::sweep_size(axes)?;
    let _ = evt_tx.send(Event::SweepStarted {
        total_configs,
        total_runs,
    });
    ctx.request_repaint();

    let result = sweep::run_blackbox_sweep(blackbox, axes, cancel, |row| {
        let _ = evt_tx.send(Event::SweepRow { row: row.clone() });
        ctx.request_repaint();
    })?;
    let _ = evt_tx.send(Event::SweepDone {
        rows: result.rows,
        cancelled: result.cancelled,
    });
    ctx.request_repaint();
    Ok(())
}

fn run_optimize_file(
    path: &str,
    axes: &SweepAxes,
    objective: SweepObjective,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let (data, schema) = read_numeric_tnum(path).map_err(|e| format!("чтение {path}: {e}"))?;
    let specs = schema.feature_specs();
    let prepared = SplitPlan::default().prepare(&data)?;
    let (total_configs, total_runs) = sweep::sweep_size(axes)?;
    let _ = evt_tx.send(Event::OptimizeStarted {
        total_configs,
        total_runs,
    });
    ctx.request_repaint();

    let result = sweep::run_sweep(&prepared.search, &specs, axes, objective, cancel, |row| {
        let _ = evt_tx.send(Event::OptimizeRow { row: row.clone() });
        ctx.request_repaint();
    })?;
    let _ = evt_tx.send(Event::OptimizeDone {
        rows: result.rows,
        cancelled: result.cancelled,
    });
    ctx.request_repaint();
    Ok(())
}

fn validate_text_config(
    model_cfg: &ModelConfig,
    train_cfg: &TextTrainConfig,
) -> Result<(), String> {
    if model_cfg.d_model == 0 {
        return Err("d_model должен быть > 0".to_string());
    }
    if model_cfg.d_ff == 0 {
        return Err("d_ff должен быть > 0".to_string());
    }
    if model_cfg.n_heads == 0 || !model_cfg.d_model.is_multiple_of(model_cfg.n_heads) {
        return Err(format!(
            "d_model={} должен делиться на heads={}",
            model_cfg.d_model, model_cfg.n_heads
        ));
    }
    if model_cfg.n_enc_layers == 0 || model_cfg.n_dec_layers == 0 {
        return Err("число слоёв должно быть >= 1".to_string());
    }
    if train_cfg.steps == 0 {
        return Err("steps должен быть >= 1".to_string());
    }
    if train_cfg.batch_size == 0 {
        return Err("batch должен быть >= 1".to_string());
    }
    if train_cfg.ctx_len == 0 || train_cfg.tgt_len == 0 {
        return Err("ctx/tgt должны быть >= 1".to_string());
    }
    if !train_cfg.lr.is_finite() || train_cfg.lr <= 0.0 {
        return Err("lr должен быть конечным и > 0".to_string());
    }
    Ok(())
}

fn train_text(
    path: &str,
    model_cfg: &ModelConfig,
    train_cfg: &TextTrainConfig,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<Option<LoadedText>, String> {
    validate_text_config(model_cfg, train_cfg)?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("чтение {path}: {e}"))?;
    let dataset = TextDataset::new(&text);
    if dataset.len() < train_cfg.ctx_len + train_cfg.tgt_len {
        return Err("корпус короче окна ctx+tgt".to_string());
    }
    let seed_hint: String = text.chars().take(train_cfg.ctx_len).collect();

    set_init_seed(train_cfg.seed);
    let model = TextModel::new(model_cfg, dataset.vocab.len());
    let _ = evt_tx.send(Event::TextStarted {
        total_steps: train_cfg.steps,
    });
    ctx.request_repaint();

    let mut last_loss = None;
    train_text_cb(
        &model,
        &dataset,
        train_cfg,
        &mut |step, loss| {
            last_loss = Some(loss);
            let _ = evt_tx.send(Event::TextProgress { step, loss });
            ctx.request_repaint();
        },
        cancel,
    );

    if cancel.load(Ordering::Relaxed) {
        let _ = evt_tx.send(Event::TextDone {
            final_loss: last_loss,
            cancelled: true,
            vocab_size: dataset.vocab.len(),
            seed_hint,
        });
        ctx.request_repaint();
        return Ok(None);
    }

    let _ = evt_tx.send(Event::TextDone {
        final_loss: last_loss,
        cancelled: false,
        vocab_size: dataset.vocab.len(),
        seed_hint,
    });
    ctx.request_repaint();
    Ok(Some(LoadedText {
        model,
        dataset,
        ctx_len: train_cfg.ctx_len,
        tgt_len: train_cfg.tgt_len,
    }))
}

fn generate_text(
    loaded: &LoadedText,
    seed: &str,
    total_new: usize,
    temperature: f32,
    top_k: usize,
    rng_seed: u64,
) -> Result<String, String> {
    if seed.chars().count() < loaded.ctx_len {
        return Err(format!("затравка короче ctx_len={}", loaded.ctx_len));
    }
    if total_new == 0 {
        return Err("число новых символов должно быть >= 1".to_string());
    }
    if !temperature.is_finite() {
        return Err("temperature должен быть конечным".to_string());
    }
    for c in seed.chars() {
        if !loaded.dataset.vocab.contains(c) {
            return Err(format!("символ '{c}' отсутствует в словаре корпуса"));
        }
    }
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let sample = generate(
        &loaded.model,
        &loaded.dataset.vocab,
        seed,
        total_new,
        loaded.ctx_len,
        loaded.tgt_len,
        temperature,
        top_k,
        &mut rng,
    );
    Ok(format!("{seed}{sample}"))
}

fn prepare_tnum(
    input: &str,
    output: &str,
    spec: &PrepareSpec,
) -> Result<(usize, usize, usize), String> {
    let tnum = table_path_to_tnum(input, spec)?;
    let rows = tnum.lines().count().saturating_sub(6);
    std::fs::write(output, &tnum).map_err(|e| format!("запись {output}: {e}"))?;
    Ok((rows, spec.n_inputs, spec.n_outputs))
}

#[allow(clippy::too_many_arguments)]
fn run_epoch_sweep(
    path: &str,
    nc: &NumericConfig,
    base_tcfg: &TrainConfig,
    milestones: &[usize],
    target_r2: f32,
    min_gain: f32,
    plateau_min: f32,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<(), String> {
    validate_numeric(nc)?;
    validate_train(base_tcfg.lr, base_tcfg.batch_size)?;
    if milestones.is_empty() || milestones.contains(&0) {
        return Err("epochs: список должен быть непустым и > 0".to_string());
    }
    if !target_r2.is_finite() {
        return Err("target-r2 должен быть конечным".to_string());
    }
    if !min_gain.is_finite() || min_gain < 0.0 {
        return Err("min-r2-gain должен быть конечным и >= 0".to_string());
    }
    if !plateau_min.is_finite() {
        return Err("plateau-min-r2 должен быть конечным".to_string());
    }

    let (data, schema) = read_numeric_tnum(path).map_err(|e| format!("чтение {path}: {e}"))?;
    let specs = schema.feature_specs();
    let n_out = data.outputs.ncols();
    let prepared = SplitPlan::default().prepare(&data)?;
    let mut points: Vec<usize> = milestones.iter().copied().filter(|&e| e > 0).collect();
    points.sort_unstable();
    points.dedup();

    let _ = evt_tx.send(Event::EpochSweepStarted {
        total_points: points.len(),
    });
    ctx.request_repaint();

    let mut on_row = |row: EpochRow| {
        let _ = evt_tx.send(Event::EpochSweepRow { row });
        ctx.request_repaint();
    };
    let rows = epoch_sweep::run_epoch_sweep_cb(
        &prepared.search,
        nc,
        &specs,
        n_out,
        base_tcfg,
        &points,
        cancel,
        &mut on_row,
    );
    let recommendation = epoch_sweep::recommended_stop(&rows, target_r2, min_gain, plateau_min);
    let _ = evt_tx.send(Event::EpochSweepDone {
        rows,
        recommendation,
        cancelled: cancel.load(Ordering::Relaxed),
    });
    ctx.request_repaint();
    Ok(())
}

/// Обучение в worker-потоке. Возвращает обученную модель (Rc живёт здесь) или
/// `None` при отмене.
fn train_numeric(
    source: &DataSource,
    nc: &NumericConfig,
    tcfg: &TrainConfig,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<Option<Loaded>, String> {
    validate_numeric(nc)?;
    validate_train(tcfg.lr, tcfg.batch_size)?;

    let (data, in_specs) = match source {
        DataSource::Blackbox(name) => {
            let bb = blackbox::by_name(name)
                .ok_or_else(|| format!("неизвестный чёрный ящик: {name}"))?;
            let specs = vec![FeatureSpec::Continuous; bb.n_inputs()];
            // Seed обучения не должен менять саму выборку.
            (bb.generate(2000, DEFAULT_DATA_SEED), specs)
        }
        DataSource::File(path) => {
            let (data, schema) =
                read_numeric_tnum(path).map_err(|e| format!("чтение {path}: {e}"))?;
            let specs = schema.feature_specs();
            (data, specs)
        }
    };
    let n_inputs = data.inputs.ncols();
    let n_out = data.outputs.ncols();
    // Test откладывается и в GUI не открывается: единственный финальный замер
    // появится вместе с явной финальной оценкой (Э4), а до тех пор все числа
    // здесь — validation.
    let prepared = SplitPlan::default().prepare(&data)?;
    let (train, val) = prepared.search.fold(0)?;
    let (in_norm, out_norm) = fit_normalizers(&train, &in_specs);

    set_init_seed(tcfg.seed); // воспроизводимая инициализация
    let model = nc.build(&in_specs, n_out);

    let _ = evt_tx.send(Event::TrainStarted {
        total_epochs: tcfg.epochs,
        parameter_count: model.parameter_count(),
    });
    ctx.request_repaint();

    train_surrogate_cb(
        &model,
        &train,
        &in_norm,
        &out_norm,
        tcfg,
        &mut |epoch, loss| {
            let _ = evt_tx.send(Event::Epoch {
                epoch: epoch + 1,
                loss,
            });
            ctx.request_repaint();
        },
        cancel,
    );

    if cancel.load(Ordering::Relaxed) {
        let _ = evt_tx.send(Event::TrainDone { metrics: None });
        ctx.request_repaint();
        return Ok(None);
    }

    let metrics = evaluate_surrogate(&model, &val, &in_norm, &out_norm);
    let _ = evt_tx.send(Event::TrainDone {
        metrics: Some(metrics),
    });
    ctx.request_repaint();
    let calibration = Some(calibration_sample(&train.inputs, 256));
    Ok(Some(Loaded {
        model,
        nc: nc.clone(),
        in_specs: in_specs.clone(),
        in_norm,
        out_norm,
        n_inputs,
        n_outputs: n_out,
        diag: Some(DiagData {
            nc: nc.clone(),
            source: source.clone(),
            in_specs,
            train,
            val,
        }),
        calibration,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::config::ModelConfig;
    use crate::encoders::ValueEncoderConfig;
    use crate::numeric_model::{KanConfig, ModelKind};

    #[test]
    fn samples_kan_edge_for_gui() {
        let config = NumericConfig {
            kind: ModelKind::Kan,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 16,
            mlp_layers: 1,
            kan: KanConfig {
                width: 4,
                layers: 2,
                grid: 5,
            },
        };
        let model = config.build(&[FeatureSpec::Continuous; 3], 2);
        let points = sample_kan_edge(&model, 0, 1, 3, 7).unwrap();

        assert_eq!(points.len(), 7);
        assert_eq!(points[0].0, -3.0);
        assert_eq!(points[6].0, 3.0);
        assert!(points.iter().all(|(x, y)| x.is_finite() && y.is_finite()));
        assert!(sample_kan_edge(&model, 0, 3, 0, 7).is_err());
    }

    #[test]
    fn extracts_symbolic_formulas_in_raw_units_for_gui() {
        let config = NumericConfig {
            kind: ModelKind::Kan,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 16,
            mlp_layers: 1,
            kan: KanConfig {
                width: 4,
                layers: 1,
                grid: 5,
            },
        };
        let data = blackbox::sum().generate(64, 0);
        let prepared = SplitPlan::default().prepare(&data).unwrap();
        let (train, val) = prepared.search.fold(0).unwrap();
        let in_specs = vec![FeatureSpec::Continuous; train.inputs.ncols()];
        let (in_norm, out_norm) = fit_normalizers(&train, &in_specs);
        let loaded = Loaded {
            model: config.build(&in_specs, train.outputs.ncols()),
            nc: config.clone(),
            in_specs: in_specs.clone(),
            in_norm,
            out_norm,
            n_inputs: train.inputs.ncols(),
            n_outputs: train.outputs.ncols(),
            diag: Some(DiagData {
                nc: config,
                source: DataSource::Blackbox("sum".to_string()),
                in_specs,
                train,
                val,
            }),
            calibration: None,
        };

        let result = extract_kan_symbolic(&loaded).expect("должны извлечься формулы");
        assert!(result.formulas.starts_with("y0 = "));
        let metrics = result
            .formula_metrics
            .expect("после обучения validation-метрики есть");
        assert!(metrics.r2.is_finite());
        assert!(result.kan_r2.expect("R² KAN есть").is_finite());
        assert!(result.weak_edges.iter().all(|edge| edge.r2 < 0.99));
    }
}
