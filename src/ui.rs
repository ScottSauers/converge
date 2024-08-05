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
    population_textures: Vec<egui::TextureHandle>,
    best_individual: Arc<Mutex<Individual>>,
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
            overall_best_individual: Arc::new(Mutex::new(Individual { paths: Vec::new(), fitness: 0.0 })),
        }
    }

    fn update_population(&mut self, ctx: &egui::Context, new_population: Vec<IndividualInfo>) {
        // Sort population by fitness in descending order
        let mut sorted_population = new_population.clone();
        sorted_population.sort_by(|a, b| b.individual.fitness.partial_cmp(&a.individual.fitness).unwrap());

        // Extract elites (assuming you want the top 5 as elite)
        let elite = sorted_population.iter().take(5).cloned().collect::<Vec<IndividualInfo>>();

        // Select top 10 individuals AFTER the elite
        let top_10 = sorted_population.iter().skip(5).take(10).cloned().collect::<Vec<IndividualInfo>>();
        let bottom_10 = sorted_population.iter().rev().take(10).cloned().collect::<Vec<IndividualInfo>>();

        // Update the population and textures (correct order: elite, top_10, bottom_10)
        self.population = elite.iter().chain(top_10.iter()).chain(bottom_10.iter()).cloned().collect();
        self.population_textures.clear();

        for individual_info in &self.population {
            let image = render_individual(&individual_info.individual, 256, 256);
            let texture = load_texture(ctx, &image, &format!("individual_{}", individual_info.individual.fitness));
            self.population_textures.push(texture);
        }

        // Update current_best_image based on the actual best individual
        if let Some(best) = sorted_population.first() { // Use sorted_population here
            let best_image = render_individual(&best.individual, 256, 256);
            self.current_best_image = load_texture(ctx, &best_image, "current_best_image");
            *self.current_image_data.lock().unwrap() = best_image;
        }

        // Update overall_best_individual (for the "big" display)
        let mut overall_best = self.overall_best_individual.lock().unwrap();
        if let Some(best) = sorted_population.first() {
            if best.individual.fitness > overall_best.fitness {
                *overall_best = best.individual.clone();
            }
        }
    }

    fn display_individuals(&self, ctx: &egui::Context, ui: &mut egui::Ui, individuals: Vec<&IndividualInfo>) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("population_grid")
                .max_col_width(100.0)
                .min_col_width(80.0)
                .show(ui, |ui| {
                    for (i, info) in individuals.iter().enumerate() {
                        // Use ui.push_id() to create a unique ID scope for each individual
                        ui.push_id(i, |ui| { 
                            if i % 8 == 0 && i != 0 {
                                ui.end_row();
                            }
                            ui.vertical(|ui| {
                                let texture = &self.population_textures[i];
                                let sized_texture = egui::load::SizedTexture::new(texture.id(), egui::vec2(64.0, 64.0));
                                let mut image_button = egui::ImageButton::new(sized_texture);
                                
                                // Apply different styles based on individual's status
                                if info.is_elite {
                                    image_button = image_button.rounding(egui::Rounding::same(10.0))
                                }
                                if info.is_new {
                                    image_button = image_button.frame(true);
                                }
                                if *self.selected_individual.lock().unwrap() == Some(i) {
                                    image_button = image_button.selected(true);
                                }
                                
                                // Add the button with a fixed size
                                if ui.add_sized(egui::vec2(64.0, 64.0), image_button).clicked() {
                                    *self.selected_individual.lock().unwrap() = Some(i);
                                }

                                ui.label(format!("Fitness: {:.4}", info.individual.fitness));
                                
                                // Add labels for different types of individuals
                                if info.is_elite {
                                    ui.label(egui::RichText::new("Elite").color(egui::Color32::DARK_BLUE));
                                }
                                if let Some((parent1, parent2)) = info.parent_ids {
                                    ui.label(egui::RichText::new(format!("Child of {} & {}", parent1, parent2)).color(egui::Color32::LIGHT_BLUE));
                                }
                                if !info.is_new && !info.is_elite {
                                    ui.label(egui::RichText::new("Survivor").color(egui::Color32::GREEN));
                                }
                            });
                        }); // End ui.push_id()
                    }
                });
        });
    }
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(new_population) = self.update_receiver.try_recv() {
            self.update_population(ctx, new_population);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Top section
            ui.horizontal(|ui| {
                ui.image(&self.target_image);
                ui.image(&self.current_best_image);
            });

            ui.add_space(10.0);

            // Middle section
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Controls");
                    if ui.button("Start").clicked() {
                        *self.running.lock().unwrap() = true;
                    }
                    if ui.button("Stop").clicked() {
                        *self.running.lock().unwrap() = false;
                    }
                });

                ui.vertical(|ui| {
                    ui.label(format!("Generation: {}", *self.generation.lock().unwrap()));
                    ui.label(format!("Best Fitness: {:.4}", self.best_individual.lock().unwrap().fitness));
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    Plot::new("fitness_plot")
                        .view_aspect(3.0)
                        .width(200.0)
                        .height(100.0)
                        .show(ui, |plot_ui| {
                            let fitness_history = self.fitness_history.lock().unwrap();
                            let start_gen = self.generation.lock().unwrap().saturating_sub(fitness_history.len());
                            let points: PlotPoints = PlotPoints::from_iter(
                                fitness_history.iter().enumerate()
                                    .map(|(i, &y)| [(start_gen + i) as f64, y as f64])
                            );
                            plot_ui.line(Line::new(points));
                        });
                });
            });

            ui.add_space(10.0);

            // Bottom section
            ui.label("Elite Individuals:");
            ui.push_id("elite_individuals", |ui| {
                self.display_individuals(ctx, ui, self.population.iter().filter(|info| info.is_elite).collect());
            });

            ui.add_space(10.0);
            
            ui.label("Top 10 Individuals:");
            ui.push_id("top_10_individuals", |ui| {
                let top_10_non_elite: Vec<&IndividualInfo> = self.population.iter()
                    .filter(|info| !info.is_elite)
                    .take(10)
                    .collect();
                self.display_individuals(ctx, ui, top_10_non_elite);
            });

            ui.add_space(10.0);
            
            ui.label("Bottom 10 Individuals:");
            ui.push_id("bottom_10_individuals", |ui| {
                self.display_individuals(ctx, ui, self.population.iter().rev().take(10).collect());
            });
        });

        if let Some(selected) = *self.selected_individual.lock().unwrap() {
            egui::Window::new("Individual Details")
                .show(ctx, |ui| {
                    let info = &self.population[selected];
                    ui.label(format!("Fitness: {:.4}", info.individual.fitness));
                    if info.is_elite {
                        ui.label(egui::RichText::new("Elite Individual").color(egui::Color32::DARK_BLUE));
                    }
                    if info.is_new {
                        ui.label(egui::RichText::new("New Individual").color(egui::Color32::LIGHT_BLUE));
                    }
                    if let Some((parent1, parent2)) = info.parent_ids {
                        ui.label(egui::RichText::new(format!("Parents: {} & {}", parent1, parent2)).color(egui::Color32::LIGHT_BLUE));
                    }
                    if !info.is_new && !info.is_elite {
                        ui.label(egui::RichText::new("Survivor from previous generation").color(egui::Color32::GREEN));
                    }
                    // Add more details about the individual here
                });
        }
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
