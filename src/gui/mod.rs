//! GUI на egui (PlanUI §2-3). Компилируется только с фичей `gui`.
//! Запуск: `transformer gui`. Окно + worker-поток; всё ML-состояние (Rc !Send)
//! живёт в worker-потоке, UI общается с ним каналами.

mod data;
#[cfg(feature = "demo")]
mod demo;
mod messages;
mod model;
mod predict;
mod session;
mod train;
mod worker;

/// Поднимает нативное окно egui (блокирует поток до закрытия).
pub fn run_gui() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([980.0, 760.0])
            .with_min_inner_size([760.0, 560.0])
            .with_icon(app_icon()),
        ..Default::default()
    };
    eframe::run_native(
        "Transformer",
        options,
        Box::new(|cc| Ok(Box::new(session::App::new(cc)))),
    )
}

fn app_icon() -> eframe::egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../../macos/icon-runtime.png"))
        .unwrap_or_default()
}
