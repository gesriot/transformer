//! Worker-поток: владеет Rc-состоянием моделей, выполняет долгие
//! задачи, общается с UI каналами. Отмена — кооперативно через `Arc<AtomicBool>`,
//! проверяемый внутри батч-цикла `train_surrogate_cb` (не ждёт конца эпохи).
//!
//! Обученная/загруженная модель (Rc !Send) живёт ЗДЕСЬ (`current`) и используется
//! для Predict; UI получает только числа/статусы.

use super::messages::{
    Command, CurvePoint, DatasetOrigin, DiagnosticsResult, Event, KanModelInfo, KanSymbolicInfo,
    KanWeakEdge, ModelOrigin, PreparedData,
};
use crate::batch_predict::{export_predictions, ExportSummary};
#[cfg(any(feature = "demo", test))]
use crate::blackbox;
#[cfg(feature = "demo")]
use crate::config::ModelConfig;
#[cfg(feature = "demo")]
use crate::data::TextDataset;
use crate::data::{Normalizer, NumericDataset, OutOfRange};
use crate::encoders::FeatureSpec;
#[cfg(feature = "demo")]
use crate::generate::generate;
use crate::gui::messages::InterpretReports;
use crate::init::set_init_seed;
use crate::interpret::{self, InterpretProfile, InterpretReport};
use crate::lifecycle::{
    CheckEval, CheckedRun, FinalizeRefusal, Lifecycle, RunStamp, TestDisclosure,
};
use crate::markup::TableProfile;
use crate::metrics::evaluate;
use crate::numeric_model::{validate_numeric, NumericConfig, NumericModel};
use crate::predict::predict_rows;
use crate::schema::ModelSchema;
use crate::serialize::{calibration_sample, load_numeric_full, save_numeric};
#[cfg(any(feature = "demo", test))]
use crate::split::DEFAULT_DATA_SEED;
use crate::split::{FinalEval, SplitPlan};
use crate::sweep::{self, SweepAxes, SweepObjective};
use crate::symbolic;
use crate::table::{Delimiter, Table};
#[cfg(feature = "demo")]
use crate::textmodel::TextModel;
use crate::tnum::{
    infer_prepare_spec_from_table, prepare_tnum_file, read_numeric_source, PrepareSpec,
};
use crate::train::{evaluate_surrogate, predict_dataset, validate_train};
#[cfg(feature = "demo")]
use crate::train::{train_text_cb, TextTrainConfig};
use crate::training::{
    check_candidate, refit, Dataset, EpochPoint, EvalSchedule, Phase, TrainedModel,
    TrainingHistory, TrainingSetup,
};
use eframe::egui;
use ndarray::Array2;
#[cfg(feature = "demo")]
use rand::rngs::StdRng;
#[cfg(feature = "demo")]
use rand::SeedableRng;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

pub(crate) struct Worker {
    cmd_tx: Sender<Command>,
    evt_rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    pub(crate) fn spawn(ctx: egui::Context) -> Self {
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

    pub(crate) fn send(&self, cmd: Command) {
        let _ = self.cmd_tx.send(cmd);
    }
    pub(crate) fn try_recv(&self) -> Option<Event> {
        self.evt_rx.try_recv().ok()
    }
    pub(crate) fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub(crate) fn reset_cancel(&self) {
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
    /// Схема данных модели: имена, единицы, уровни категорий. `feature_specs`
    /// выводятся из неё, поэтому отдельного списка спецификаций здесь нет.
    schema: ModelSchema,
    in_norm: Normalizer,
    out_norm: Normalizer,
    n_inputs: usize,
    n_outputs: usize,
    /// Разрешённый профиль уже применённого конвейера. У обученной пока в GUI
    /// модели его нет; у загруженной модели переносится при повторном
    /// сохранении, чтобы GUI не стирал происхождение checkpoint-а.
    interpret: Option<InterpretProfile>,
    /// Данные обучения для диагностики (`None` для загруженной `.bin`).
    diag: Option<DiagData>,
    /// Калибровочная выборка сырых train-входов: у загруженного checkpoint-а
    /// берётся из секции `calibration`, у обученной модели — из train.
    calibration: Option<Array2<f32>>,
}

/// Данные сессии обучения, нужные для диагностики.
struct DiagData {
    nc: NumericConfig,
    /// Источник нужен только демонстрациям: у встроенного ящика есть эталон,
    /// с которым сравнивают чувствительность модели.
    #[cfg(feature = "demo")]
    origin: DatasetOrigin,
    in_specs: Vec<FeatureSpec>,
    train: Arc<NumericDataset>,
    /// Набор для остаточной диагностики и проверки формул. У development это
    /// validation, у финальной модели — train+validation: test сюда не попадает.
    eval: Arc<NumericDataset>,
    eval_label: &'static str,
}

#[cfg(feature = "demo")]
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
    // Собственный учёт worker-а: безопасность не должна зависеть только от
    // того, включена ли кнопка в интерфейсе.
    let mut lifecycle = Lifecycle::default();
    // Чем является активная модель: по этому решается, можно ли заменить её
    // отладочной моделью очередной проверки.
    let mut current_origin: Option<ModelOrigin> = None;
    #[cfg(feature = "demo")]
    let mut current_text: Option<LoadedText> = None;

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            Command::CheckCandidate { data, stamp, eval } => {
                let origin = data.origin.clone();
                match train_numeric(&data, &stamp, RunKind::Check, eval, &evt_tx, &ctx, &cancel) {
                    Ok(outcome) => {
                        if let Some(eval) = outcome.check {
                            lifecycle.record_check(CheckedRun {
                                stamp: (*stamp).clone(),
                                eval,
                            });
                        }
                        // Проверка не вытесняет готовую модель: финальная и
                        // загруженная — результат работы, а отладочная годится
                        // только пока результата нет.
                        let keep_current = matches!(
                            current_origin,
                            Some(ModelOrigin::Final(_)) | Some(ModelOrigin::Checkpoint)
                        );
                        match (outcome.loaded, keep_current) {
                            (Some(loaded), false) => {
                                let model_origin = ModelOrigin::Development(stamp.clone());
                                let _ = evt_tx.send(model_ready(
                                    &loaded,
                                    &origin,
                                    model_origin.clone(),
                                ));
                                current = Some(loaded);
                                current_origin = Some(model_origin);
                            }
                            (Some(_), true) => {
                                let _ = evt_tx.send(Event::Status(
                                    "проверка завершена; активной осталась прежняя модель"
                                        .to_string(),
                                ));
                            }
                            (None, _) => {}
                        }
                        ctx.request_repaint();
                    }
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                        ctx.request_repaint();
                    }
                }
            }
            Command::FinalizeCandidate { data, stamp } => {
                // Вторая проверка того же правила: кнопка может быть включена
                // по устаревшему состоянию, а test тратится здесь.
                match lifecycle.can_finalize(&stamp) {
                    Ok(()) => {
                        let origin = data.origin.clone();
                        match train_numeric(
                            &data,
                            &stamp,
                            RunKind::Finalize,
                            EvalSchedule::Never,
                            &evt_tx,
                            &ctx,
                            &cancel,
                        ) {
                            Ok(outcome) => {
                                if let Some(eval) = outcome.final_eval {
                                    lifecycle.record_disclosure(TestDisclosure {
                                        stamp: (*stamp).clone(),
                                        eval,
                                    });
                                }
                                if let Some(loaded) = outcome.loaded {
                                    let model_origin = ModelOrigin::Final(stamp.clone());
                                    let _ = evt_tx.send(model_ready(
                                        &loaded,
                                        &origin,
                                        model_origin.clone(),
                                    ));
                                    current = Some(loaded);
                                    current_origin = Some(model_origin);
                                }
                                ctx.request_repaint();
                            }
                            Err(e) => {
                                let _ = evt_tx.send(Event::Error(e));
                                ctx.request_repaint();
                            }
                        }
                    }
                    Err(FinalizeRefusal::AlreadyFinalized) => {
                        // Повторный запрос того же кандидата не тратит test
                        // второй раз: отдаём уже полученный замер.
                        if let Some(disclosed) = lifecycle.disclosure_for(&stamp) {
                            let _ = evt_tx.send(Event::TrainDone {
                                stamp: stamp.clone(),
                                metrics: None,
                                per_output: None,
                                check_source: None,
                                r2_std_folds: None,
                                curves: Vec::new(),
                                final_eval: Some(disclosed.eval.clone()),
                                interpret: None,
                                cancelled: false,
                            });
                            ctx.request_repaint();
                        }
                    }
                    Err(refusal) => {
                        let _ = evt_tx.send(Event::Error(refusal.message().to_string()));
                        ctx.request_repaint();
                    }
                }
            }
            Command::LoadModel(path) => {
                match load_model(&path) {
                    Ok(loaded) => {
                        let _ = evt_tx.send(Event::ModelReady {
                            schema: loaded.schema.clone(),
                            kind: loaded.nc.kind,
                            source: format!("файл: {path}"),
                            model_origin: ModelOrigin::Checkpoint,
                            parameter_count: loaded.model.parameter_count(),
                            kan: kan_model_info(
                                &loaded.model,
                                loaded.diag.is_some() || loaded.calibration.is_some(),
                            ),
                        });
                        current = Some(loaded);
                        current_origin = Some(ModelOrigin::Checkpoint);
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
            Command::ExportPredictions { input, output } => {
                match &current {
                    Some(l) => match do_export(l, &input, &output) {
                        Ok(summary) => {
                            let _ = evt_tx.send(Event::ExportDone { output, summary });
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
            Command::Search {
                data,
                split,
                axes,
                objective,
            } => match run_search(&data, split, &axes, objective, &evt_tx, &ctx, &cancel) {
                Ok(()) => {}
                Err(e) => {
                    let _ = evt_tx.send(Event::Error(e));
                    ctx.request_repaint();
                }
            },
            #[cfg(feature = "demo")]
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
            #[cfg(feature = "demo")]
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
            Command::OpenDataset { origin } => {
                match open_dataset(&origin) {
                    Ok(data) => {
                        let _ = evt_tx.send(Event::DatasetOpened { data });
                    }
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
                    }
                }
                ctx.request_repaint();
            }
            Command::OpenTable { path, has_header } => {
                match open_table(&path, has_header) {
                    Ok((table, profile, suggested_inputs, suggested_categories)) => {
                        let _ = evt_tx.send(Event::TableOpened {
                            path,
                            has_header,
                            table: Box::new(table),
                            profile: Box::new(profile),
                            suggested_inputs,
                            suggested_categories,
                        });
                    }
                    Err(e) => {
                        let _ = evt_tx.send(Event::Error(e));
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
            Command::Shutdown => break,
        }
    }
}

/// Событие о новой активной модели. Происхождение идёт рядом со схемой:
/// отладочную и финальную модель нельзя показывать одинаково.
fn model_ready(loaded: &Loaded, origin: &DatasetOrigin, model_origin: ModelOrigin) -> Event {
    Event::ModelReady {
        schema: loaded.schema.clone(),
        kind: loaded.nc.kind,
        source: source_desc(origin),
        model_origin,
        parameter_count: loaded.model.parameter_count(),
        kan: kan_model_info(
            &loaded.model,
            loaded.diag.is_some() || loaded.calibration.is_some(),
        ),
    }
}

fn source_desc(origin: &DatasetOrigin) -> String {
    match origin {
        #[cfg(any(feature = "demo", test))]
        DatasetOrigin::Blackbox(name) => format!("blackbox: {name}"),
        DatasetOrigin::File(path) => format!("файл: {path}"),
        DatasetOrigin::Table(path) => format!("таблица: {path}"),
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
        .map(|edge| {
            let (input, output) = symbolic.edge_labels(edge, &loaded.schema)?;
            Ok(KanWeakEdge {
                layer: edge.layer,
                input,
                output,
                primitive: edge.name.to_string(),
                r2: edge.r2,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    // Метрики формул есть только у модели, обученной в этой сессии. Test для
    // этой вспомогательной проверки не открывается.
    let (formula_metrics, kan_r2, evaluation_label) = match &loaded.diag {
        Some(diag) => (
            Some(evaluate(
                &symbolic.predict(&diag.eval.inputs),
                &diag.eval.outputs,
            )),
            Some(
                evaluate_surrogate(&loaded.model, &diag.eval, &loaded.in_norm, &loaded.out_norm).r2,
            ),
            Some(diag.eval_label.to_string()),
        ),
        None => (None, None, None),
    };
    Ok(KanSymbolicInfo {
        formulas: symbolic.formulas(&loaded.schema)?,
        min_edge_r2,
        mean_edge_r2,
        formula_metrics,
        kan_r2,
        evaluation_label,
        weak_edges,
    })
}

fn load_model(path: &str) -> Result<Loaded, String> {
    let checkpoint = load_numeric_full(path).map_err(|e| format!("загрузка {path}: {e}"))?;
    let n_inputs = checkpoint.schema.n_inputs();
    let n_outputs = checkpoint.schema.n_outputs();
    Ok(Loaded {
        model: checkpoint.model,
        nc: checkpoint.config,
        schema: checkpoint.schema,
        in_norm: checkpoint.in_norm,
        out_norm: checkpoint.out_norm,
        n_inputs,
        n_outputs,
        interpret: checkpoint.interpret,
        diag: None, // у загруженной .bin нет данных обучения
        calibration: checkpoint.calibration,
    })
}

fn save_model(loaded: &Loaded, path: &str) -> Result<(), String> {
    save_numeric(
        path,
        &loaded.nc,
        &loaded.schema,
        &loaded.model,
        &loaded.in_norm,
        &loaded.out_norm,
        loaded.calibration.as_ref(),
        loaded.interpret.as_ref(),
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

    let rr = crate::diagnostics::range_report(&l.in_norm, &d.eval.inputs);
    let pred = predict_dataset(&l.model, &d.eval, &l.in_norm, &l.out_norm);
    let res = crate::diagnostics::residual_diagnostics(&d.eval.inputs, &pred, &d.eval.outputs);
    let residuals = res
        .iter()
        .map(|r| (r.sign_change_rate, r.tail_ratio))
        .collect();

    // Чувствительность модели считается всегда; исходного процесса — только у
    // демо-ящика, и на тех же самых парах точек.
    #[cfg(feature = "demo")]
    let bb = d.origin.blackbox().and_then(blackbox::by_name);
    #[cfg(feature = "demo")]
    let bb_eval = bb.as_ref().map(|bb| move |x: &[f32]| bb.eval(x));
    #[cfg(feature = "demo")]
    let reference =
        bb.as_ref()
            .zip(bb_eval.as_ref())
            .map(|(bb, eval)| crate::diagnostics::Reference {
                n_inputs: bb.n_inputs(),
                n_outputs: bb.n_outputs,
                eval,
            });
    // Без демонстраций исходного процесса взять неоткуда: только модель.
    #[cfg(not(feature = "demo"))]
    let reference: Option<crate::diagnostics::Reference> = None;
    let sensitivity = crate::diagnostics::sensitivity(
        &d.eval,
        &d.in_specs,
        &l.in_norm,
        &l.out_norm,
        |inputs| {
            let ds =
                NumericDataset::new(inputs.clone(), Array2::zeros((inputs.nrows(), l.n_outputs)));
            predict_dataset(&l.model, &ds, &l.in_norm, &l.out_norm)
        },
        reference.as_ref(),
        1.0,
        300,
    );

    Ok(DiagnosticsResult {
        overfit_loss,
        extrapolation_rows: rr.rows_out,
        extrapolation_total: rr.total,
        evaluation_label: d.eval_label.to_string(),
        residuals,
        sensitivity,
    })
}

/// Единичный прогноз — тот же пакет из одной строки.
fn do_predict(l: &Loaded, values: &[f32]) -> Result<(Vec<f32>, Vec<OutOfRange>), String> {
    if values.len() != l.n_inputs {
        return Err(format!(
            "ожидалось {} входов, получено {}",
            l.n_inputs,
            values.len()
        ));
    }
    let inputs = Array2::from_shape_vec((1, l.n_inputs), values.to_vec())
        .expect("одна строка нужной ширины");
    let result = predict_rows(&l.model, &l.in_norm, &l.out_norm, &inputs)?;
    let extrapolation = result
        .warnings
        .first()
        .map(|w| w.details.clone())
        .unwrap_or_default();
    Ok((result.outputs.row(0).to_vec(), extrapolation))
}

fn do_export(l: &Loaded, input: &str, output: &str) -> Result<ExportSummary, String> {
    export_predictions(input, output, &l.schema, |inputs| {
        predict_rows(&l.model, &l.in_norm, &l.out_norm, inputs)
    })
}

/// Поиск конфигурации на активном наборе данных.
fn run_search(
    prepared: &PreparedData,
    split: SplitPlan,
    axes: &SweepAxes,
    objective: SweepObjective,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let dataset = Dataset::new(clone_data(prepared), prepared.schema.clone())?;
    let splits = split.prepare(dataset.data())?;
    let (total_configs, total_runs) = sweep::sweep_size(axes)?;
    let _ = evt_tx.send(Event::SearchStarted {
        total_configs,
        total_runs,
    });
    ctx.request_repaint();

    let result = sweep::run_sweep(&dataset, &splits.search, axes, objective, cancel, |row| {
        let _ = evt_tx.send(Event::SearchRow { row: row.clone() });
        ctx.request_repaint();
    })?;
    let _ = evt_tx.send(Event::SearchDone {
        rows: result.rows,
        cancelled: result.cancelled,
    });
    ctx.request_repaint();
    Ok(())
}

#[cfg(feature = "demo")]
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

#[cfg(feature = "demo")]
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

#[cfg(feature = "demo")]
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
    let stats = prepare_tnum_file(input, output, spec)?;
    Ok((stats.rows, stats.n_inputs, stats.n_outputs))
}

#[allow(clippy::too_many_arguments)]
/// Копия данных для `Dataset`: активный набор живёт в сессии под `Arc`, а ядру
/// нужен владеющий экземпляр.
fn clone_data(prepared: &PreparedData) -> NumericDataset {
    prepared
        .data
        .gather(&(0..prepared.data.len()).collect::<Vec<_>>())
}

/// Прочитать источник данных: сгенерировать чёрный ящик либо прочитать файл.
///
/// Дальше сессия работает с готовыми данными, поэтому файл открывается ровно
/// один раз — на этом и держится инвариант «один активный датасет».
fn open_dataset(origin: &DatasetOrigin) -> Result<PreparedData, String> {
    let (data, schema) = match origin {
        #[cfg(any(feature = "demo", test))]
        DatasetOrigin::Blackbox(name) => {
            let bb = blackbox::by_name(name)
                .ok_or_else(|| format!("неизвестный чёрный ящик: {name}"))?;
            // Seed обучения не должен менять саму выборку.
            (
                bb.generate(2000, DEFAULT_DATA_SEED),
                ModelSchema::synthetic(bb.n_inputs(), bb.n_outputs)?,
            )
        }
        DatasetOrigin::File(path) => read_numeric_source(path)?,
        DatasetOrigin::Table(path) => {
            return Err(format!(
                "{path}: размеченная таблица приходит из диалога разметки, а не из чтения"
            ))
        }
    };
    Ok(PreparedData {
        origin: origin.clone(),
        data: Arc::new(data),
        schema,
    })
}

/// Прочитать таблицу и посчитать профиль. Автоопределение подсказывает границу
/// вход/выход, но роли всё равно подтверждает пользователь.
fn open_table(
    path: &str,
    has_header: bool,
) -> Result<(Table, TableProfile, Option<usize>, Vec<usize>), String> {
    // Сначала читаем без выделенного заголовка: старая эвристика должна увидеть
    // первую строку, а файл при этом остаётся прочитан ровно один раз.
    let raw = Table::read_path(path, Delimiter::Auto, false)?;
    let inferred = has_header
        .then(|| infer_prepare_spec_from_table(&raw, Delimiter::Auto).ok())
        .flatten();
    let suggested_inputs = inferred.as_ref().map(|spec| spec.n_inputs);
    let suggested_categories = inferred
        .map(|spec| {
            spec.categorical
                .into_iter()
                .map(|(index, _)| index)
                .collect()
        })
        .unwrap_or_default();
    let table = if has_header {
        raw.promote_first_row_to_header()?
    } else {
        raw
    };
    let profile = TableProfile::of(&table);
    Ok((table, profile, suggested_inputs, suggested_categories))
}

/// Что делаем с кандидатом: проверяем на validation/CV или фиксируем и
/// открываем test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RunKind {
    Check,
    Finalize,
}

/// Итог запуска для worker-цикла: что стало активной моделью и что записать в
/// учёт жизненного цикла.
#[derive(Default)]
struct RunOutcome {
    /// Новая активная модель. `None` — отмена или CV-проверка, где моделей
    /// столько же, сколько folds.
    loaded: Option<Loaded>,
    /// Оценка проверенного кандидата.
    check: Option<CheckEval>,
    /// Состоявшийся замер на test.
    final_eval: Option<FinalEval>,
}

/// Обучение в worker-потоке. Модель остаётся здесь: Rc через границу потока не
/// ходит.
fn train_numeric(
    prepared: &PreparedData,
    stamp: &RunStamp,
    kind: RunKind,
    eval: EvalSchedule,
    evt_tx: &Sender<Event>,
    ctx: &egui::Context,
    cancel: &AtomicBool,
) -> Result<RunOutcome, String> {
    // Всё, чем запуск будет подписан, берётся из самого отпечатка: разойтись
    // им негде.
    let split = stamp.split;
    let nc = &stamp.candidate.config;
    let tcfg = &stamp.candidate.train;
    let interpret = stamp.candidate.interpret;
    validate_numeric(nc)?;
    validate_train(tcfg.lr, tcfg.batch_size)?;

    let (data, schema) = (Arc::clone(&prepared.data), prepared.schema.clone());
    let in_specs = schema.feature_specs();
    let n_inputs = data.inputs.ncols();
    let n_out = data.outputs.ncols();
    // Данные уже могут быть общими (размеченная таблица), поэтому Dataset
    // строится из копии: владение остаётся у вызывающего.
    let dataset = Dataset::new(clone_data(prepared), schema.clone())?;

    // Счётчик параметров нужен до обучения; RNG это не сбивает — ядро само
    // выставляет seed перед построением модели.
    set_init_seed(tcfg.seed);
    let _ = evt_tx.send(Event::TrainStarted {
        total_epochs: tcfg.epochs,
        parameter_count: nc.build(&in_specs, n_out).parameter_count(),
    });
    ctx.request_repaint();

    let mut setup = TrainingSetup::new(nc.clone(), tcfg.clone());
    setup.eval = eval;

    // Конвейер интерпретации применяется в обеих фазах или нигде: иначе
    // сохранённая модель отличалась бы от той, по которой принимали решения.
    // Оба хука пишут в общее состояние конвейера, поэтому оно за RefCell:
    // одновременно взять два `&mut` на одну переменную нельзя.
    let pipeline_error: RefCell<Option<String>> = RefCell::new(None);
    let reports: RefCell<Vec<(Phase, InterpretReport)>> = RefCell::new(Vec::new());
    let profile = interpret;
    let configure = |_phase: Phase, model: &NumericModel| {
        if let Some(p) = &profile {
            if pipeline_error.borrow().is_some() {
                return;
            }
            if let Err(e) = interpret::apply_l1(model, p) {
                *pipeline_error.borrow_mut() = Some(e);
                // Ядро проверяет cancel до post-train и до открытия test.
                // Ошибка настройки модели должна остановить тот же путь.
                cancel.store(true, Ordering::Relaxed);
            }
        }
    };
    let post_train = |phase: Phase,
                      trained: &mut TrainedModel,
                      train_data: &NumericDataset,
                      eval_data: Option<&NumericDataset>| {
        let Some(p) = &profile else {
            return;
        };
        if pipeline_error.borrow().is_some() {
            return;
        }
        match interpret::run_pipeline(
            &mut trained.model,
            train_data,
            eval_data,
            &trained.in_norm,
            &trained.out_norm,
            tcfg,
            p,
            cancel,
        ) {
            Ok(report) => reports.borrow_mut().push((phase, report)),
            Err(e) => {
                *pipeline_error.borrow_mut() = Some(e);
                // `refit` проверит флаг сразу после хука и не откроет test по
                // модели, для которой запрошенный конвейер не завершился.
                cancel.store(true, Ordering::Relaxed);
            }
        }
    };
    // Номер fold доходит до интерфейса: при K-fold точки разных folds повторяют
    // номера эпох, и живая кривая по ним была бы ломаной ни о чём.
    let epoch_event = |phase: Phase, fold: usize, point: &EpochPoint| {
        let _ = evt_tx.send(Event::Epoch {
            phase,
            fold,
            epoch: point.epoch,
            loss: point.train_loss,
            val_r2: point.val.as_ref().map(|m| m.r2),
        });
        ctx.request_repaint();
    };

    // Кривые holdout-проверки: у финализации своей кривой нет — она приходит
    // событиями Epoch и уже нарисована.
    let mut holdout_curves: Vec<Vec<CurvePoint>> = Vec::new();
    let mut check_source = None;
    // Проверка идёт по всем folds и не трогает test. Фиксация уже выбранного
    // кандидата сразу делает refit на всём pool — в том числе при K-fold — и
    // только затем один раз открывает test.
    let (trained, check, final_eval, diag_train, diag_eval, eval_label) = match kind {
        RunKind::Finalize => {
            let pool = Arc::new(split.prepare(dataset.data())?.search.all());
            let outcome = refit(
                &dataset,
                split,
                &setup,
                stamp.final_init_seed,
                cancel,
                &mut |phase, point| epoch_event(phase, 0, point),
                &mut |phase, model| configure(phase, model),
                &mut |phase, trained, train_data, eval_data| {
                    post_train(phase, trained, train_data, eval_data)
                },
            )?;
            if let Some(e) = pipeline_error.borrow_mut().take() {
                return Err(e);
            }
            let Some(trained) = outcome.model else {
                send_cancelled(evt_tx, ctx, stamp);
                return Ok(RunOutcome::default());
            };
            (
                trained,
                None,
                outcome.eval,
                Arc::clone(&pool),
                pool,
                "train+validation",
            )
        }
        RunKind::Check => {
            let outcome = check_candidate(
                &dataset,
                split,
                &setup,
                cancel,
                &mut |fold, point| epoch_event(Phase::Development, fold, point),
                &mut |model| configure(Phase::Development, model),
                &mut |_fold, trained, train_data, eval_data| {
                    post_train(Phase::Development, trained, train_data, eval_data)
                },
            )?;
            if let Some(e) = pipeline_error.borrow_mut().take() {
                return Err(e);
            }
            let (Some(outcome), false) = (outcome, cancel.load(Ordering::Relaxed)) else {
                send_cancelled(evt_tx, ctx, stamp);
                return Ok(RunOutcome::default());
            };
            let check = CheckEval {
                metrics: outcome.metrics.clone(),
                per_output: outcome.per_output.clone(),
                r2_std_folds: outcome.r2_std_folds,
            };
            let curves: Vec<Vec<CurvePoint>> = outcome.histories.iter().map(curve).collect();
            let folds = curves.len();
            let Some(trained) = outcome.model else {
                // K-fold: моделей столько же, сколько folds, ни одна не
                // представляет оценку. Отдаём только её и кривые по folds.
                let _ = evt_tx.send(Event::TrainDone {
                    stamp: Box::new(stamp.clone()),
                    metrics: Some(outcome.metrics),
                    per_output: Some(outcome.per_output),
                    check_source: Some(outcome.source),
                    r2_std_folds: Some(outcome.r2_std_folds),
                    curves,
                    final_eval: None,
                    // Конвейер отработал на каждом fold, и структура у каждого
                    // своя. Показывать отчёт первого как общий нельзя.
                    interpret: None,
                    cancelled: false,
                });
                ctx.request_repaint();
                return Ok(RunOutcome {
                    loaded: None,
                    check: Some(check),
                    final_eval: None,
                });
            };
            debug_assert_eq!(folds, 1, "модель отдаётся только у holdout");
            holdout_curves = curves;
            check_source = Some(outcome.source);
            let splits = split.prepare(dataset.data())?;
            let (train, val) = splits.search.fold(0)?;
            (
                trained,
                Some(check),
                None,
                Arc::new(train),
                Arc::new(val),
                "validation",
            )
        }
    };
    let TrainedModel {
        model,
        in_norm,
        out_norm,
        ..
    } = trained;
    let reports = reports.into_inner();
    // Прерванный конвейер оставил модель недообученной: сохранять и открывать
    // test по ней нельзя.
    if reports.iter().any(|(_, r)| r.cancelled) {
        send_cancelled(evt_tx, ctx, stamp);
        return Ok(RunOutcome::default());
    }
    let _ = evt_tx.send(Event::TrainDone {
        stamp: Box::new(stamp.clone()),
        metrics: check.as_ref().map(|c| c.metrics.clone()),
        per_output: check.as_ref().map(|c| c.per_output.clone()),
        check_source,
        r2_std_folds: check.as_ref().map(|c| c.r2_std_folds),
        curves: holdout_curves,
        final_eval: final_eval.clone(),
        interpret: interpret_reports(reports),
        cancelled: false,
    });
    ctx.request_repaint();
    let calibration = Some(calibration_sample(&diag_train.inputs, 256));
    let loaded = Loaded {
        model,
        nc: nc.clone(),
        schema,
        in_norm,
        out_norm,
        n_inputs,
        n_outputs: n_out,
        // Профиль запоминается только после успешного конвейера: иначе
        // checkpoint утверждал бы то, чего с моделью не делали.
        interpret: profile,
        diag: Some(DiagData {
            nc: nc.clone(),
            #[cfg(feature = "demo")]
            origin: prepared.origin.clone(),
            in_specs,
            train: diag_train,
            eval: diag_eval,
            eval_label,
        }),
        calibration,
    };
    Ok(RunOutcome {
        loaded: Some(loaded),
        check,
        final_eval,
    })
}

/// Отчёты конвейера по фазам: у development видно влияние прунинга на
/// validation, у финальной — какой стала структура. Публикуем, если есть хотя
/// бы один: после поиска фаза только финальная, и её отчёт терять нельзя.
fn interpret_reports(reports: Vec<(Phase, InterpretReport)>) -> Option<Box<InterpretReports>> {
    let development = reports
        .iter()
        .find(|(phase, _)| *phase == Phase::Development)
        .map(|(_, r)| r.clone());
    let final_model = reports
        .iter()
        .find(|(phase, _)| *phase == Phase::Final)
        .map(|(_, r)| r.clone());
    (development.is_some() || final_model.is_some()).then(|| {
        Box::new(InterpretReports {
            development,
            final_model,
        })
    })
}

/// История обучения в виде, пригодном для UI.
fn curve(history: &TrainingHistory) -> Vec<CurvePoint> {
    history
        .points
        .iter()
        .map(|point| CurvePoint {
            epoch: point.epoch,
            train_loss: point.train_loss,
            val_r2: point.val.as_ref().map(|m| m.r2),
        })
        .collect()
}

/// Отменённый запуск: ни оценки, ни модели, ни замера на test.
fn send_cancelled(evt_tx: &Sender<Event>, ctx: &egui::Context, stamp: &RunStamp) {
    let _ = evt_tx.send(Event::TrainDone {
        stamp: Box::new(stamp.clone()),
        metrics: None,
        per_output: None,
        check_source: None,
        r2_std_folds: None,
        curves: Vec::new(),
        final_eval: None,
        interpret: None,
        cancelled: true,
    });
    ctx.request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackbox;
    use crate::config::ModelConfig;
    use crate::encoders::ValueEncoderConfig;
    use crate::lifecycle::CandidateSpec;
    use crate::numeric_model::{KanConfig, ModelKind};
    use crate::train::{fit_normalizers, TrainConfig};
    use crate::training::EvalSchedule;

    /// Отпечаток для тестов: всё, чем подписан запуск, в одном месте.
    fn stamp(
        split: SplitPlan,
        config: NumericConfig,
        train: TrainConfig,
        interpret: Option<InterpretProfile>,
    ) -> RunStamp {
        RunStamp {
            dataset_revision: 1,
            split,
            candidate: CandidateSpec {
                config,
                train,
                interpret,
            },
            final_init_seed: 0,
        }
    }

    /// Дождаться первого события, которое не является просто статусом.
    fn next_meaningful(rx: &mpsc::Receiver<Event>) -> Event {
        loop {
            match rx.recv_timeout(std::time::Duration::from_secs(60)) {
                Ok(Event::Status(_)) | Ok(Event::TrainStarted { .. }) | Ok(Event::Epoch { .. }) => {
                    continue
                }
                Ok(event) => return event,
                Err(e) => panic!("worker молчит: {e}"),
            }
        }
    }

    /// Правило «test открывают один раз» проверяется и на границе worker-а:
    /// кнопка могла остаться включённой по устаревшему состоянию, а тратится
    /// test именно здесь.
    #[test]
    fn worker_enforces_the_test_budget_itself() {
        let data = blackbox::sum().generate(64, 0);
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let prepared = PreparedData {
            origin: DatasetOrigin::Blackbox("sum".to_string()),
            data: Arc::new(data),
            schema,
        };
        let config = NumericConfig {
            kind: ModelKind::Mlp,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 4,
            mlp_layers: 1,
            kan: KanConfig::default(),
        };
        let train = TrainConfig {
            epochs: 1,
            batch_size: 16,
            ..Default::default()
        };
        let first = stamp(SplitPlan::default(), config.clone(), train.clone(), None);
        let mut other_config = config;
        other_config.mlp_width = 8;
        let second = stamp(SplitPlan::default(), other_config, train, None);

        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (evt_tx, evt_rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let handle = {
            let ctx = egui::Context::default();
            let cancel = Arc::clone(&cancel);
            std::thread::spawn(move || worker_loop(cmd_rx, evt_tx, ctx, cancel))
        };

        // Без проверки финализировать нельзя.
        cmd_tx
            .send(Command::FinalizeCandidate {
                data: prepared.clone(),
                stamp: Box::new(first.clone()),
            })
            .unwrap();
        match next_meaningful(&evt_rx) {
            Event::Error(msg) => assert!(msg.contains("Проверить конфигурацию"), "{msg}"),
            other => panic!(
                "ожидался отказ, получено другое событие: {:?}",
                event_name(&other)
            ),
        }

        // Проверка -> финализация проходит и открывает test.
        cmd_tx
            .send(Command::CheckCandidate {
                eval: EvalSchedule::Never,
                data: prepared.clone(),
                stamp: Box::new(first.clone()),
            })
            .unwrap();
        let event = next_meaningful(&evt_rx);
        assert!(
            matches!(event, Event::TrainDone { .. }),
            "проверка: {}",
            describe(&event)
        );
        let event = next_meaningful(&evt_rx);
        assert!(
            matches!(event, Event::ModelReady { .. }),
            "проверка: {}",
            describe(&event)
        );
        cmd_tx
            .send(Command::FinalizeCandidate {
                data: prepared.clone(),
                stamp: Box::new(first),
            })
            .unwrap();
        let disclosed = match next_meaningful(&evt_rx) {
            Event::TrainDone { final_eval, .. } => final_eval,
            other => panic!("ожидался финальный замер: {}", event_name(&other)),
        };
        assert!(disclosed.is_some(), "test должен быть открыт один раз");
        // Финальная модель тоже становится активной: событие о ней идёт следом.
        let event = next_meaningful(&evt_rx);
        assert!(
            matches!(event, Event::ModelReady { .. }),
            "финализация: {}",
            describe(&event)
        );

        // Другой кандидат на той же ревизии данных: test уже потрачен, и
        // проверка на validation этого не меняет.
        cmd_tx
            .send(Command::CheckCandidate {
                eval: EvalSchedule::Never,
                data: prepared.clone(),
                stamp: Box::new(second.clone()),
            })
            .unwrap();
        let event = next_meaningful(&evt_rx);
        assert!(
            matches!(event, Event::TrainDone { .. }),
            "вторая проверка: {}",
            describe(&event)
        );
        cmd_tx
            .send(Command::FinalizeCandidate {
                data: prepared,
                stamp: Box::new(second),
            })
            .unwrap();
        // Между двумя событиями нет ModelReady: проверка после финализации не
        // вытесняет финальную модель отладочной.
        match next_meaningful(&evt_rx) {
            Event::Error(msg) => assert!(msg.contains("уже открыт"), "{msg}"),
            other => panic!("ожидался отказ: {}", event_name(&other)),
        }

        cmd_tx.send(Command::Shutdown).unwrap();
        handle.join().unwrap();
    }

    fn describe(event: &Event) -> String {
        match event {
            Event::Error(msg) => format!("Error({msg})"),
            other => event_name(other).to_string(),
        }
    }

    fn event_name(event: &Event) -> &'static str {
        match event {
            Event::Error(_) => "Error",
            Event::TrainDone { .. } => "TrainDone",
            Event::ModelReady { .. } => "ModelReady",
            _ => "другое",
        }
    }

    /// Отмена во время конвейера обязана прерывать всё: модель не сохраняется,
    /// test не открывается, а событие честно сообщает об отмене.
    #[test]
    fn cancelled_pipeline_yields_no_model_and_no_test() {
        let data = blackbox::sum().generate(64, 0);
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let prepared = PreparedData {
            origin: DatasetOrigin::Blackbox("sum".to_string()),
            data: Arc::new(data),
            schema,
        };
        let config = NumericConfig {
            kind: ModelKind::Kan,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 4,
            mlp_layers: 1,
            kan: KanConfig {
                width: 4,
                layers: 2,
                grid: 5,
            },
        };
        let train = TrainConfig {
            epochs: 1,
            batch_size: 16,
            ..Default::default()
        };
        let (tx, rx) = mpsc::channel();
        // Отмена взведена заранее: обучение прервётся на первом же батче.
        let cancel = AtomicBool::new(true);

        let loaded = train_numeric(
            &prepared,
            &stamp(
                SplitPlan::default(),
                config,
                train,
                Some(InterpretProfile::v1()),
            ),
            RunKind::Finalize,
            EvalSchedule::Never,
            &tx,
            &egui::Context::default(),
            &cancel,
        )
        .unwrap();
        assert!(loaded.loaded.is_none(), "модель не должна сохраняться");
        assert!(loaded.final_eval.is_none(), "test открывать нечем");

        let done = rx
            .try_iter()
            .find_map(|e| match e {
                Event::TrainDone {
                    final_eval,
                    interpret,
                    cancelled,
                    ..
                } => Some((final_eval.is_some(), interpret.is_some(), cancelled)),
                _ => None,
            })
            .expect("событие о завершении");
        assert!(done.2, "обучение должно быть помечено отменённым");
        assert!(!done.0, "test открывать нельзя");
        assert!(!done.1, "отчёт неполного конвейера не публикуется");
    }

    /// Ошибка запрошенного interpret-конвейера должна остановить refit до
    /// test. Иначе worker вернул бы ошибку без `TrainDone`, хотя test уже был
    /// измерен внутри ядра и lifecycle об этом никогда не узнал бы.
    #[test]
    fn failed_pipeline_stops_before_final_evaluation() {
        let data = blackbox::sum().generate(64, 0);
        let prepared = PreparedData {
            origin: DatasetOrigin::Blackbox("sum".to_string()),
            data: Arc::new(data),
            schema: ModelSchema::synthetic(2, 1).unwrap(),
        };
        let config = NumericConfig {
            kind: ModelKind::Mlp,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 4,
            mlp_layers: 1,
            kan: KanConfig::default(),
        };
        let train = TrainConfig {
            epochs: 1,
            batch_size: 16,
            ..Default::default()
        };
        let (tx, rx) = mpsc::channel();
        let cancel = AtomicBool::new(false);

        let result = train_numeric(
            &prepared,
            &stamp(
                SplitPlan::default(),
                config,
                train,
                Some(InterpretProfile::v1()),
            ),
            RunKind::Finalize,
            EvalSchedule::Never,
            &tx,
            &egui::Context::default(),
            &cancel,
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("interpret-конвейер на MLP должен быть отвергнут"),
        };

        assert!(error.contains("KAN"), "{error}");
        assert!(
            cancel.load(Ordering::Relaxed),
            "ошибка хука обязана остановить ядро до test"
        );
        assert!(rx.try_iter().all(|event| !matches!(
            event,
            Event::TrainDone {
                final_eval: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn final_gui_training_keeps_the_refit_model() {
        let data = blackbox::sum().generate(64, 0);
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let prepared = PreparedData {
            origin: DatasetOrigin::Blackbox("sum".to_string()),
            data: Arc::new(data),
            schema: schema.clone(),
        };
        let split = SplitPlan::default();
        let splits = split.prepare(prepared.data.as_ref()).unwrap();
        let pool = splits.search.all();
        let specs = schema.feature_specs();
        let (expected_in_norm, expected_out_norm) = fit_normalizers(&pool, &specs);
        let config = NumericConfig {
            kind: ModelKind::Mlp,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 4,
            mlp_layers: 1,
            kan: KanConfig::default(),
        };
        let train = TrainConfig {
            epochs: 1,
            batch_size: 16,
            ..Default::default()
        };
        let (tx, rx) = mpsc::channel();
        let cancel = AtomicBool::new(false);

        let loaded = train_numeric(
            &prepared,
            &stamp(split, config, train, None),
            RunKind::Finalize,
            EvalSchedule::Never,
            &tx,
            &egui::Context::default(),
            &cancel,
        )
        .unwrap()
        .loaded
        .expect("финальная модель");

        assert_eq!(loaded.in_norm.mean, expected_in_norm.mean);
        assert_eq!(loaded.out_norm.mean, expected_out_norm.mean);
        let diag = loaded.diag.expect("данные финальной модели");
        assert_eq!(diag.eval_label, "train+validation");
        assert_eq!(diag.train.len(), pool.len());
        assert!(Arc::ptr_eq(&diag.train, &diag.eval));
        let final_eval = rx.try_iter().find_map(|event| match event {
            Event::TrainDone {
                final_eval,
                cancelled,
                ..
            } => {
                assert!(!cancelled);
                final_eval
            }
            _ => None,
        });
        assert!(final_eval.is_some(), "refit обязан один раз измерить test");
    }

    #[test]
    fn final_gui_interpretation_publishes_a_final_only_report() {
        let data = blackbox::sum().generate(64, 0);
        let prepared = PreparedData {
            origin: DatasetOrigin::Blackbox("sum".to_string()),
            data: Arc::new(data),
            schema: ModelSchema::synthetic(2, 1).unwrap(),
        };
        let config = NumericConfig {
            kind: ModelKind::Kan,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 4,
            mlp_layers: 1,
            kan: KanConfig {
                width: 2,
                layers: 2,
                grid: 3,
            },
        };
        let train = TrainConfig {
            epochs: 1,
            batch_size: 16,
            ..Default::default()
        };
        let profile = InterpretProfile {
            prune: Some(0.9),
            finetune_epochs: 1,
            compact: true,
            ..InterpretProfile::v1()
        };
        let (tx, rx) = mpsc::channel();

        let loaded = train_numeric(
            &prepared,
            &stamp(SplitPlan::default(), config, train, Some(profile)),
            RunKind::Finalize,
            EvalSchedule::Never,
            &tx,
            &egui::Context::default(),
            &AtomicBool::new(false),
        )
        .unwrap()
        .loaded
        .expect("финальная KAN");
        assert_eq!(loaded.interpret, Some(profile));

        let reports = rx
            .try_iter()
            .find_map(|event| match event {
                Event::TrainDone {
                    interpret,
                    final_eval,
                    cancelled,
                    ..
                } => {
                    assert!(!cancelled);
                    assert!(final_eval.is_some(), "test измеряется после конвейера");
                    interpret
                }
                _ => None,
            })
            .expect("отчёт конвейера");
        assert!(reports.development.is_none());
        assert!(reports.final_model.is_some());
        assert_eq!(reports.profile(), Some(&profile));
    }

    #[test]
    fn development_gui_training_reports_each_output() {
        let inputs =
            Array2::from_shape_fn((64, 2), |(row, column)| (row as f32 + column as f32) / 64.0);
        let outputs = Array2::from_shape_fn((64, 2), |(row, column)| {
            let x0 = inputs[[row, 0]];
            let x1 = inputs[[row, 1]];
            if column == 0 {
                x0 + x1
            } else {
                x0 - x1
            }
        });
        let prepared = PreparedData {
            origin: DatasetOrigin::Blackbox("two-output-test".to_string()),
            data: Arc::new(NumericDataset::new(inputs, outputs)),
            schema: ModelSchema::synthetic(2, 2).unwrap(),
        };
        let config = NumericConfig {
            kind: ModelKind::Mlp,
            transformer: ModelConfig::default(),
            value: ValueEncoderConfig::default(),
            mlp_width: 4,
            mlp_layers: 1,
            kan: KanConfig::default(),
        };
        let train = TrainConfig {
            epochs: 1,
            batch_size: 16,
            ..Default::default()
        };
        let (tx, rx) = mpsc::channel();

        train_numeric(
            &prepared,
            &stamp(SplitPlan::default(), config, train, None),
            RunKind::Check,
            EvalSchedule::Never,
            &tx,
            &egui::Context::default(),
            &AtomicBool::new(false),
        )
        .unwrap()
        .loaded
        .expect("development-модель");

        let per_output = rx.try_iter().find_map(|event| match event {
            Event::TrainDone {
                per_output,
                cancelled,
                ..
            } => {
                assert!(!cancelled);
                per_output
            }
            _ => None,
        });
        assert_eq!(per_output.expect("поколоночные метрики").len(), 2);
    }

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
            schema: ModelSchema::synthetic(train.inputs.ncols(), train.outputs.ncols()).unwrap(),
            in_norm,
            out_norm,
            n_inputs: train.inputs.ncols(),
            n_outputs: train.outputs.ncols(),
            interpret: None,
            diag: Some(DiagData {
                nc: config,
                #[cfg(feature = "demo")]
                origin: DatasetOrigin::Blackbox("sum".to_string()),
                in_specs,
                train: Arc::new(train),
                eval: Arc::new(val),
                eval_label: "validation",
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
        assert_eq!(result.evaluation_label.as_deref(), Some("validation"));
        assert!(result.weak_edges.iter().all(|edge| edge.r2 < 0.99));
    }

    #[test]
    fn gui_resave_preserves_interpret_profile() {
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
        let data = blackbox::sum().generate(16, 0);
        let schema = ModelSchema::synthetic(2, 1).unwrap();
        let specs = schema.feature_specs();
        let (in_norm, out_norm) = fit_normalizers(&data, &specs);
        let profile = InterpretProfile::v1();
        let loaded = Loaded {
            model: config.build(&specs, 1),
            nc: config,
            schema,
            in_norm,
            out_norm,
            n_inputs: 2,
            n_outputs: 1,
            interpret: Some(profile),
            diag: None,
            calibration: None,
        };
        let path = std::env::temp_dir().join(format!(
            "transformer_gui_interpret_resave_{}.bin",
            std::process::id()
        ));
        let resaved_path = std::env::temp_dir().join(format!(
            "transformer_gui_interpret_resave_copy_{}.bin",
            std::process::id()
        ));

        save_model(&loaded, path.to_str().unwrap()).unwrap();
        let reloaded = load_model(path.to_str().unwrap()).unwrap();
        save_model(&reloaded, resaved_path.to_str().unwrap()).unwrap();
        let resaved = load_model(resaved_path.to_str().unwrap()).unwrap();

        std::fs::remove_file(path).ok();
        std::fs::remove_file(resaved_path).ok();
        assert_eq!(resaved.interpret, Some(profile));
    }

    #[test]
    fn open_table_preserves_inferred_categorical_columns() {
        let path = std::env::temp_dir().join(format!(
            "transformer_gui_open_table_{}.csv",
            std::process::id()
        ));
        std::fs::write(&path, "x0,material_id,y0\n1,0,2\n3,1,4\n").unwrap();

        let (table, _, suggested_inputs, suggested_categories) =
            open_table(path.to_str().unwrap(), true).unwrap();

        std::fs::remove_file(path).ok();
        assert_eq!(table.header().unwrap(), ["x0", "material_id", "y0"]);
        assert_eq!(table.rows().len(), 2);
        assert_eq!(suggested_inputs, Some(2));
        assert_eq!(suggested_categories, vec![1]);
    }
}
