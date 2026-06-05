//! GUI на egui (PlanUI §2-3). Компилируется только с фичей `gui`.
//! Запуск: `transformer gui`. Окно + worker-поток; всё ML-состояние (Rc !Send)
//! живёт в worker-потоке, UI общается с ним каналами.

mod app;
mod messages;
mod worker;

/// Поднимает нативное окно egui (блокирует поток до закрытия).
pub fn run_gui() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "transformer",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
