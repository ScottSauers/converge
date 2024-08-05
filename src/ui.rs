use eframe::egui;
use image::{RgbaImage};
use std::sync::{Arc, Mutex};
use crate::{Individual, IndividualInfo, render_individual};
use egui_plot::{Plot, Line, PlotPoints};
use std::sync::mpsc::Receiver;

pub struct EvolutionApp {
    target_image: egui::TextureHandle,
    current_best_image: egui::TextureHandle,
    population: Vec<IndividualInfo>,
    best_individual: Arc<Mutex<Individual>>, population_textures: Vec<(usize, egui::TextureHandle)>,
    current_image_data: Arc<Mutex<RgbaImage>>,
    fitness_history: Arc<Mutex<Vec<f32>>>,
    generation: Arc<Mutex<usize>>,
    running: Arc<Mutex<bool>>,
    update_receiver: Receiver<Vec<IndividualInfo>>,
    selected_individual: Arc<Mutex<Option<usize>>>,
    overall_best_individual: Arc<Mutex<Individual>>,
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
        update_receiver: Receiver<Vec<IndividualInfo>>,
    ) -> Self {
        let ctx = &cc.egui_ctx;
        let target_texture = load_texture(ctx, &target_image, "target_image");
        let current_texture = load_texture(ctx, &current_image.lock().unwrap(), "current_best_image");

        Self {
            target_image: target_texture,
            current_best_image: current_texture,
            population: Vec::new(),
            population_textures: Vec::new(),
            best_individual,
            current_image_data: current_image,
            fitness_history,
            generation,
            running,
            update_receiver,
            selected_individual: Arc::new(Mutex::new(None)),
            overall_best_individual: Arc::new(Mutex::new(Individual { id: 0, paths: Vec::new(), fitness: 0.0 })),
        }
    }

    fn update_population(&mut self, ctx: &egui::Context, new_population: Vec<IndividualInfo>) {
        // Sort population by fitness in descending order
        let mut sorted_population = new_population;
        sorted_population.sort_by(|a, b| b.individual.fitness.partial_cmp(&a.individual.fitness).unwrap());

        // Extract elites (top 5)
        let elite = sorted_population.iter().take(5).cloned().collect::<Vec<IndividualInfo>>();

        // Select top 10 individuals AFTER the elite
        let top_10 = sorted_population.iter().skip(5).take(10).cloned().collect::<Vec<IndividualInfo>>();

        // Select bottom 10 individuals
        let bottom_10 = sorted_population.iter().rev().take(10).cloned().collect::<Vec<IndividualInfo>>();

        // Update the population
        self.population = elite.into_iter()
            .chain(top_10.into_iter())
            .chain(bottom_10.into_iter())
            .collect();

        // Clear and update textures
        self.population_textures.clear();
        for individual_info in &self.population {
            let image = render_individual(&individual_info.individual, 256, 256);
            let texture = load_texture(ctx, &image, &format!("individual_{}", individual_info.individual.id));
            self.population_textures.push((individual_info.individual.id, texture));
        }

        // Update current_best_image
        if let Some(best) = sorted_population.first() {
            let best_image = render_individual(&best.individual, 256, 256);
            self.current_best_image = load_texture(ctx, &best_image, "current_best_image");
            *self.current_image_data.lock().unwrap() = best_image;
        }
    }

    fn display_individuals(&self, ui: &mut egui::Ui, individuals: &[IndividualInfo], id_prefix: &str) {
        ui.horizontal_wrapped(|ui| {
            for (local_index, info) in individuals.iter().enumerate() {
                let unique_id = format!("{}_{}_{}", id_prefix, local_index, info.individual.fitness);
                ui.push_id(unique_id, |ui| {
                    ui.vertical(|ui| {
                        // Find the correct texture for this individual
                        if let Some((_, texture)) = self.population_textures.iter()
                            .find(|(id, _)| *id == info.individual.id) {
                                let image = egui::Image::new(egui::load::SizedTexture::new(texture.id(), egui::vec2(64.0, 64.0)));
                                if ui.add(image).clicked() {
                                    *self.selected_individual.lock().unwrap() = Some(info.individual.id);
                                }
                        } else {
                            ui.label("Image not found");
                        }

                        ui.label(format!("Fitness: {:.4}", info.individual.fitness));
                        
                        if info.is_elite {
                            ui.label(egui::RichText::new("Elite").color(egui::Color32::GOLD));
                        }
                        if info.is_new {
                            ui.label(egui::RichText::new("New").color(egui::Color32::GREEN));
                        }
                    });
                });
                ui.add_space(4.0);
            }
        });
    }
}






impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(new_population) = self.update_receiver.try_recv() {
            self.update_population(ctx, new_population);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Target Image");
                    ui.image(&self.target_image);
                });
                ui.vertical(|ui| {
                    ui.heading("Current Best");
                    ui.image(&self.current_best_image);
                });
                ui.vertical(|ui| {
                    ui.heading("Statistics");
                    ui.label(format!("Generation: {}", *self.generation.lock().unwrap()));
                    ui.label(format!("Best Fitness: {:.4}", self.best_individual.lock().unwrap().fitness));
                    
                    Plot::new("fitness_plot")
                        .view_aspect(2.0)
                        .show(ui, |plot_ui| {
                            let fitness_history = self.fitness_history.lock().unwrap();
                            let points: PlotPoints = fitness_history.iter().enumerate()
                                .map(|(i, &y)| [i as f64, y as f64])
                                .collect();
                            plot_ui.line(Line::new(points));
                        });
                });
            });

            ui.add_space(20.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Elite Individuals");
                let elite_count = 5.min(self.population.len());
                self.display_individuals(ui, &self.population[0..elite_count], "elite");

                ui.add_space(10.0);

                ui.heading("Top 10 Individuals");
                let top_10_start = elite_count;
                let top_10_end = (top_10_start + 10).min(self.population.len());
                self.display_individuals(ui, &self.population[top_10_start..top_10_end], "top_10");

                ui.add_space(10.0);

                ui.heading("Bottom 10 Individuals");
                let bottom_10_start = self.population.len().saturating_sub(10);
                self.display_individuals(ui, &self.population[bottom_10_start..], "bottom_10");
            });
        });

        if let Some(selected) = *self.selected_individual.lock().unwrap() {
            egui::Window::new("Individual Details")
                .show(ctx, |ui| {
                    if let Some(info) = self.population.get(selected) {
                        ui.label(format!("Fitness: {:.4}", info.individual.fitness));
                        if info.is_elite {
                            ui.label(egui::RichText::new("Elite Individual").color(egui::Color32::GOLD));
                        }
                        if info.is_new {
                            ui.label(egui::RichText::new("New Individual").color(egui::Color32::GREEN));
                        }
                        if let Some((parent1, parent2)) = info.parent_ids {
                            ui.label(egui::RichText::new(format!("Parents: {} & {}", parent1, parent2)).color(egui::Color32::LIGHT_BLUE));
                        }
                        if info.crossover {
                            ui.label(egui::RichText::new("Result of Crossover").color(egui::Color32::YELLOW));
                        }
                        if info.survived {
                            ui.label(egui::RichText::new("Survived from Previous Generation").color(egui::Color32::KHAKI));
                        }
                        
                        if let Some((_, texture)) = self.population_textures.iter().find(|(idx, _)| *idx == selected) {
                            ui.image(texture);
                        }
                    }
                });
        }

        ctx.request_repaint();
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
    update_receiver: Receiver<Vec<IndividualInfo>>,
) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0]),
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
