use eframe::egui;
use image::RgbaImage;
use std::sync::{Arc, Mutex};
use crate::Individual;
use egui_plot::{Plot, Line, PlotPoints};
use std::sync::mpsc::Receiver;

pub struct EvolutionApp {
    target_image: egui::TextureHandle,
    current_image: egui::TextureHandle,
    best_individual: Arc<Mutex<Individual>>,
    current_image_data: Arc<Mutex<RgbaImage>>,
    fitness_history: Arc<Mutex<Vec<f32>>>,
    generation: Arc<Mutex<usize>>,
    running: Arc<Mutex<bool>>,
    update_receiver: Receiver<RgbaImage>,
}

impl EvolutionApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        target_image: RgbaImage,
        best_individual: Arc<Mutex<Individual>>,
        current_image: Arc<Mutex<RgbaImage>>,
        fitness_history: Arc<Mutex<Vec<f32>>>,
        generation: Arc<Mutex<usize>>,
        running: Arc<Mutex<bool>>,
        update_receiver: Receiver<RgbaImage>,
    ) -> Self {
        let ctx = &cc.egui_ctx;
        let target_texture = load_texture(ctx, &target_image, "target_image");
        let current_texture = load_texture(ctx, &current_image.lock().unwrap(), "current_image");

        Self {
            target_image: target_texture,
            current_image: current_texture,
            best_individual,
            current_image_data: current_image,
            fitness_history,
            generation,
            running,
            update_receiver,
        }
    }

    fn update_current_image(&mut self, ctx: &egui::Context, current_image: &RgbaImage) {
        self.current_image = load_texture(ctx, current_image, "current_image");
        *self.current_image_data.lock().unwrap() = current_image.clone();
    }
}


impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for new images
        if let Ok(new_image) = self.update_receiver.try_recv() {
            self.update_current_image(ctx, &new_image);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Start").clicked() {
                    *self.running.lock().unwrap() = true;
                }
                if ui.button("Stop").clicked() {
                    *self.running.lock().unwrap() = false;
                }
            });

            ui.horizontal(|ui| {
                ui.image(&self.target_image);
                ui.image(&self.current_image);
            });

            ui.label(format!("Generation: {}", *self.generation.lock().unwrap()));
            ui.label(format!("Best Fitness: {:.4}", self.best_individual.lock().unwrap().fitness));

            Plot::new("fitness_plot")
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    let fitness_history = self.fitness_history.lock().unwrap();
                    let points: PlotPoints = PlotPoints::from_iter(
                        fitness_history.iter().enumerate()
                            .map(|(i, &y)| [i as f64, y as f64])
                    );
                    plot_ui.line(Line::new(points));
                });
        });
    }
}



fn load_texture(ctx: &egui::Context, image: &RgbaImage, name: &str) -> egui::TextureHandle {
    let size = [image.width() as _, image.height() as _];
    let image_data = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
    ctx.load_texture(name, image_data, egui::TextureOptions::default())
}

pub fn run_ui_main_thread(
    target_image: RgbaImage,
    best_individual: Arc<Mutex<Individual>>,
    current_image: Arc<Mutex<RgbaImage>>,
    fitness_history: Arc<Mutex<Vec<f32>>>,
    generation: Arc<Mutex<usize>>,
    running: Arc<Mutex<bool>>,
    update_receiver: Receiver<RgbaImage>,
) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Image Evolution",
        options,
        Box::new(move |cc| {
            Ok(Box::new(EvolutionApp::new(
                cc,
                target_image,
                best_individual,
                current_image,
                fitness_history,
                generation,
                running,
                update_receiver,
            )))
        }),
    ).unwrap_or_else(|e| eprintln!("Error running eframe: {:?}", e));
}
