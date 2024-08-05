use image::{GenericImageView, Rgba, DynamicImage, GrayImage, ImageBuffer, Luma, RgbaImage};
use rand::Rng;
use std::fs;
use std::thread;
use svg::Document;
use resvg::usvg::{Tree, Options};
use resvg::tiny_skia::{Pixmap, Transform};
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use rand::seq::SliceRandom;
use rand::thread_rng;
use svg::node::element::{Definitions, Stop};
use svg::node::element::path::Data;
use svg::node::element::Path;
use rand_distr::{Geometric, Distribution};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::atomic::AtomicU32;
use imageproc::edges::canny;
use imageproc::gradients::sobel_gradients;
use image_compare::{Algorithm, gray_similarity_structure};
use image::imageops::resize;
use lab::Lab;
use rgb::RGB;



mod ui;
use ui::run_ui_main_thread;


const POPULATION_SIZE: usize = 50;
const GENERATIONS: usize = 10000;
const ELITISM_COUNT: usize = 5;
const PLATEAU_THRESHOLD: usize = 2; // Number of generations to check for plateau
const PLATEAU_IMPROVEMENT_THRESHOLD: f32 = 0.001; // Minimum improvement to not be considered a plateau
const MAX_PATHS: usize = 40;
const MAX_COMMANDS_PER_PATH: usize = 50;
const IMAGE_SIZE: u32 = 256;

#[derive(Clone, Debug)]
enum Command {
    M(f32, f32),
    L(f32, f32),
    H(f32),
    V(f32),
    C(f32, f32, f32, f32, f32, f32),
    S(f32, f32, f32, f32),
    Q(f32, f32, f32, f32),
    T(f32, f32),
    A(f32, f32, f32, bool, bool, f32, f32),
    Z,
}


struct AtomicF32 {
    inner: AtomicU32,
}

impl AtomicF32 {
    fn new(val: f32) -> Self {
        AtomicF32 {
            inner: AtomicU32::new(val.to_bits()),
        }
    }

    fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.inner.load(order))
    }

    fn store(&self, val: f32, order: Ordering) {
        self.inner.store(val.to_bits(), order)
    }
}


#[derive(Clone, Debug)]
struct Individual {
    paths: Vec<(Vec<Command>, PathStyle)>,
    fitness: f32,
}

#[derive(Clone)]
struct IndividualInfo {
    individual: Individual,
    parent_ids: Option<(usize, usize)>,
    is_elite: bool,
    is_new: bool,
    survived: bool,
    crossover: bool,
}



#[derive(Clone, Debug)]
struct LinearGradient {
    start: (f32, f32),
    end: (f32, f32),
    colors: Vec<Rgba<u8>>,
}


#[derive(Clone, Debug)]
struct PathStyle {
    fill: Option<Rgba<u8>>,
    gradient: Option<LinearGradient>,
}

fn adaptive_mutation_rate(generation: usize, max_generations: usize) -> f32 {
    let min_rate = 0.01;
    let max_rate = 0.2;
    let progress = generation as f32 / max_generations as f32;
    max_rate - (max_rate - min_rate) * progress
}

fn random_command(rng: &mut impl Rng, width: u32, height: u32) -> Command {
    match rng.gen_range(0..10) {
        0 => Command::M(rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
        1 => Command::L(rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
        2 => Command::H(rng.gen_range(0.0..width as f32)),
        3 => Command::V(rng.gen_range(0.0..height as f32)),
        4 => Command::C(
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
        ),
        5 => Command::S(
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
        ),
        6 => Command::Q(
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
        ),
        7 => Command::T(rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
        8 => Command::A(
            rng.gen_range(0.0..50.0), rng.gen_range(0.0..50.0),
            rng.gen_range(0.0..360.0), rng.gen_bool(0.5), rng.gen_bool(0.5),
            rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32),
        ),
        _ => Command::Z,
    }
}

fn random_color(rng: &mut impl Rng) -> Rgba<u8> {
    Rgba([rng.gen(), rng.gen(), rng.gen(), rng.gen_range(50..=200)])
}

fn create_random_individual(rng: &mut impl Rng, width: u32, height: u32) -> Individual {
    let paths: Vec<(Vec<Command>, PathStyle)> = (0..2)  // Always start with 2 paths
        .map(|_| {
            let commands: Vec<Command> = (0..2)  // Always start with 2 commands per path
                .map(|_| random_command(rng, width, height))
                .collect();
            (commands, random_path_style(rng, width, height))
        })
        .collect();

    Individual { paths, fitness: 0.0 }
}

fn random_path_style(rng: &mut impl Rng, width: u32, height: u32) -> PathStyle {
    if rng.gen_bool(0.8) { // Chance of having a gradient
        // Create a geometric distribution with p = 0.5
        // This means each subsequent number is half as likely as the previous one
        let geo = Geometric::new(0.5).unwrap();
        
        // Generate a random number from this distribution and add 2
        // (because geometric distribution starts at 0, but we want to start at 2)
        let num_colors = geo.sample(rng) as usize + 2;

        PathStyle {
            fill: None,
            gradient: Some(LinearGradient {
                start: (rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
                end: (rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
                colors: (0..num_colors).map(|_| random_color(rng)).collect(),
            })
        }
    } else {
        PathStyle {
            fill: Some(random_color(rng)),
            gradient: None,
        }
    }
}

fn crossover(parent1: &Individual, parent2: &Individual, rng: &mut impl Rng) -> Individual {
    let mut child_paths = Vec::new();
    let max_len = parent1.paths.len().max(parent2.paths.len());
    
    for i in 0..max_len {
        if i < parent1.paths.len() && i < parent2.paths.len() {
            // Combine paths from both parents
            let mut combined_commands = Vec::new();
            let crossover_point = rng.gen_range(0..parent1.paths[i].0.len().min(parent2.paths[i].0.len()));
            combined_commands.extend_from_slice(&parent1.paths[i].0[0..crossover_point]);
            combined_commands.extend_from_slice(&parent2.paths[i].0[crossover_point..]);
            
            let style = if rng.gen_bool(0.5) {
                parent1.paths[i].1.clone()
            } else {
                parent2.paths[i].1.clone()
            };
            
            child_paths.push((combined_commands, style));
        } else if i < parent1.paths.len() {
            child_paths.push(parent1.paths[i].clone());
        } else if i < parent2.paths.len() {
            child_paths.push(parent2.paths[i].clone());
        }
    }

    Individual { paths: child_paths, fitness: 0.0 }
}




fn render_individual(individual: &Individual, width: u32, height: u32) -> RgbaImage {
    let mut document = Document::new()
        .set("viewBox", (0, 0, IMAGE_SIZE, IMAGE_SIZE))
        .set("width", IMAGE_SIZE)
        .set("height", IMAGE_SIZE);

    let mut defs = Definitions::new();

    for (i, (commands, style)) in individual.paths.iter().enumerate() {
        let mut data = Data::new();
        for command in commands {
            match command {
                Command::M(x, y) => data = data.move_to((*x, *y)),
                Command::L(x, y) => data = data.line_to((*x, *y)),
                Command::H(x) => data = data.horizontal_line_to(*x),
                Command::V(y) => data = data.vertical_line_to(*y),
                Command::C(x1, y1, x2, y2, x, y) => data = data.cubic_curve_to((*x1, *y1, *x2, *y2, *x, *y)),
                Command::S(x2, y2, x, y) => data = data.smooth_cubic_curve_to((*x2, *y2, *x, *y)),
                Command::Q(x1, y1, x, y) => data = data.quadratic_curve_to((*x1, *y1, *x, *y)),
                Command::T(x, y) => data = data.smooth_quadratic_curve_to((*x, *y)),
                Command::A(rx, ry, x_axis_rotation, large_arc, sweep, x, y) => 
                    data = data.elliptical_arc_to((
                        *rx, *ry, *x_axis_rotation, 
                        if *large_arc { 1 } else { 0 }, 
                        if *sweep { 1 } else { 0 }, 
                        *x, *y
                    )),
                Command::Z => data = data.close(),
            }
        }

        let mut path = Path::new().set("d", data);

        if let Some(ref gradient) = style.gradient {
            let gradient_id = format!("gradient-{}", i);
            let mut linear_gradient = svg::node::element::LinearGradient::new()
                .set("id", gradient_id.clone())
                .set("x1", format!("{}%", gradient.start.0 / width as f32 * 100.0))
                .set("y1", format!("{}%", gradient.start.1 / height as f32 * 100.0))
                .set("x2", format!("{}%", gradient.end.0 / width as f32 * 100.0))
                .set("y2", format!("{}%", gradient.end.1 / height as f32 * 100.0));

            for (j, color) in gradient.colors.iter().enumerate() {
                let stop = Stop::new()
                    .set("offset", format!("{}%", j * 100 / (gradient.colors.len() - 1)))
                    .set("stop-color", format!("rgba({},{},{},{})", color[0], color[1], color[2], color[3]));
                linear_gradient = linear_gradient.add(stop);
            }

            defs = defs.add(linear_gradient);
            path = path.set("fill", format!("url(#{})", gradient_id));
        } else if let Some(fill) = style.fill {
            path = path.set("fill", format!("rgba({},{},{},{})", fill[0], fill[1], fill[2], fill[3]));
        } else {
            path = path.set("fill", "none");
        }

        document = document.add(path);
    }

    document = document.add(defs);

    let opt = Options::default();



    let tree = Tree::from_str(&document.to_string(), &opt).unwrap();

    let mut pixmap = Pixmap::new(width, height).unwrap();
    resvg::render(&tree, Transform::default(), &mut pixmap.as_mut());
    let raw_pixels = pixmap.take();
    ImageBuffer::from_raw(width, height, raw_pixels).unwrap()

}




fn draw_line(image: &mut RgbaImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgba<u8>) {
    // Simple line drawing algorithm (Bresenham's algorithm)
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1.0 } else { -1.0 };
    let sy = if y0 < y1 { 1.0 } else { -1.0 };
    let mut err = dx + dy;

    let mut x = x0;
    let mut y = y0;

    loop {
        if x >= 0.0 && x < image.width() as f32 && y >= 0.0 && y < image.height() as f32 {
            image.put_pixel(x as u32, y as u32, color);
        }

        if (x - x1).abs() < 0.01 && (y - y1).abs() < 0.01 {
            break;
        }

        let e2 = 2.0 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}










fn calculate_fitness(img: &DynamicImage, individual: &Individual) -> f32 {
    let (width, height) = img.dimensions();
    let rendered = render_individual(individual, width, height);
    
    let window_size = 8u32;
    let c1 = (0.01f32 * 255.0).powi(2);
    let c2 = (0.03f32 * 255.0).powi(2);
    
    let mut ssim_sum = 0.0;
    let mut windows = 0;
    
    for y in (0..height).step_by(window_size as usize) {
        for x in (0..width).step_by(window_size as usize) {
            let mut img_mean = [0.0f32; 3];
            let mut rendered_mean = [0.0f32; 3];
            let mut covariance = [0.0f32; 3];
            let mut img_variance = [0.0f32; 3];
            let mut rendered_variance = [0.0f32; 3];
            
            let window_width = window_size.min(width - x);
            let window_height = window_size.min(height - y);
            
            let n = (window_width * window_height) as f32;
            if n == 0.0 {
                continue;
            }
            
            for dy in 0..window_height {
                for dx in 0..window_width {
                    let img_pixel = img.get_pixel(x + dx, y + dy);
                    let rendered_pixel = rendered.get_pixel(x + dx, y + dy);
                    
                    // Convert RGB to Lab
                    let img_lab = Lab::from_rgb(&[img_pixel[0], img_pixel[1], img_pixel[2]]);
                    let rendered_lab = Lab::from_rgb(&[rendered_pixel[0], rendered_pixel[1], rendered_pixel[2]]);
                    
                    let img_vals = [img_lab.l, img_lab.a, img_lab.b];
                    let rendered_vals = [rendered_lab.l, rendered_lab.a, rendered_lab.b];
                    
                    for c in 0..3 {
                        img_mean[c] += img_vals[c];
                        rendered_mean[c] += rendered_vals[c];
                        covariance[c] += img_vals[c] * rendered_vals[c];
                        img_variance[c] += img_vals[c].powi(2);
                        rendered_variance[c] += rendered_vals[c].powi(2);
                    }
                }
            }
            
            let mut ssim = 1.0;
            for c in 0..3 {
                img_mean[c] /= n;
                rendered_mean[c] /= n;
                covariance[c] = (covariance[c] / n) - (img_mean[c] * rendered_mean[c]);
                img_variance[c] = (img_variance[c] / n) - img_mean[c].powi(2);
                rendered_variance[c] = (rendered_variance[c] / n) - rendered_mean[c].powi(2);
                
                let numerator = (2.0 * img_mean[c] * rendered_mean[c] + c1) * (2.0 * covariance[c] + c2);
                let denominator = (img_mean[c].powi(2) + rendered_mean[c].powi(2) + c1) * 
                                  (img_variance[c] + rendered_variance[c] + c2);
                
                let channel_ssim = if denominator != 0.0 { numerator / denominator } else { 0.0 };
                
                // Adjust weights to emphasize color
                let weight = if c == 0 { 0.5 } else { 0.75 }; // 0.5 for luminance, 0.75 for color channels
                ssim *= channel_ssim.powf(weight);
            }
            
            if !ssim.is_nan() && !ssim.is_infinite() {
                ssim_sum += ssim;
                windows += 1;
            }
        }
    }
    
    if windows > 0 {
        ssim_sum / windows as f32
    } else {
        0.0
    }
}








fn calculate_fitness_parallel(img: &image::DynamicImage, population: &mut [Individual]) {
    population.par_iter_mut().for_each(|individual| {
        individual.fitness = calculate_fitness(img, individual);
    });
}














fn optimize_image(
    img: &image::DynamicImage,
    best_individual: &Arc<Mutex<Individual>>,
    current_image: &Arc<Mutex<RgbaImage>>,
    fitness_history: &Arc<Mutex<Vec<f32>>>,
    generation: &Arc<Mutex<usize>>,
    running: &Arc<Mutex<bool>>,
    update_sender: mpsc::Sender<Vec<IndividualInfo>>,
) -> Individual {
    let (width, height) = img.dimensions();

    let best_fitness = Arc::new(AtomicF32::new(0.0));
    let generations_without_improvement = Arc::new(AtomicUsize::new(0));

    let mut population: Vec<_> = (0..POPULATION_SIZE)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            create_random_individual(&mut rng, width, height)
        })
        .collect();

    calculate_fitness_parallel(img, &mut population);

    let population = Arc::new(Mutex::new(population));

    let generation_counter = Arc::new(AtomicUsize::new(0));

    (0..GENERATIONS).for_each(|_| {
        if !*running.lock().unwrap() {
            return;
        }

        let gen = generation_counter.fetch_add(1, Ordering::SeqCst);

        let mut pop = population.lock().unwrap();

        pop.par_sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Create mating pool (excluding elites)
        let mating_pool: Vec<_> = pop[ELITISM_COUNT..].to_vec();

        // Select elites
        let elites: Vec<_> = pop[..ELITISM_COUNT].to_vec();

        let best = pop[0].clone();
        println!("Generation {}: Best fitness = {}", gen, best.fitness);

        {
            let mut best_i = best_individual.lock().unwrap();
            *best_i = best.clone();
        }

        let evolved_image = render_individual(&best, width, height);

        {
            let mut current_img = current_image.lock().unwrap();
            *current_img = evolved_image;
        }

        {
            let mut fitness_hist = fitness_history.lock().unwrap();
            fitness_hist.push(best.fitness);
        }

        {
            let mut gen_count = generation.lock().unwrap();
            *gen_count = gen;
        }

        let population_info: Vec<IndividualInfo> = pop.par_iter().enumerate()
            .map(|(i, ind)| IndividualInfo {
                individual: ind.clone(),
                parent_ids: None,
                is_elite: i < ELITISM_COUNT,
                is_new: false,
                survived: true,
                crossover: false,
            })
            .collect();

        update_sender.send(population_info).unwrap();

        let new_population: Vec<_> = (0..POPULATION_SIZE - ELITISM_COUNT)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                
                // Unwrap parents safely
                let parent1 = tournament_selection(&mating_pool, 5, &mut rng)
                    .expect("Failed to select parent1");
                let parent2 = tournament_selection(&mating_pool, 5, &mut rng)
                    .expect("Failed to select parent2");
                
                // Perform crossover
                let mut child = crossover(parent1, parent2, &mut rng);
                
                // Calculate mutation rate
                let mutation_rate = adaptive_mutation_rate(gen, GENERATIONS);
                
                // Apply mutation
                mutate(&mut child, &mut rng, width, height, mutation_rate);
                
                // Calculate fitness
                child.fitness = calculate_fitness(img, &child);
                
                child // Return the child
            })
            .collect();

        let mut new_pop = new_population;
        new_pop.extend(elites);
        *pop = new_pop;

        let current_best_fitness = pop[0].fitness;
        let best = best_fitness.load(Ordering::Relaxed);
        if current_best_fitness > best + PLATEAU_IMPROVEMENT_THRESHOLD {
            best_fitness.store(current_best_fitness, Ordering::Relaxed);
            generations_without_improvement.store(0, Ordering::Relaxed);
        } else {
            generations_without_improvement.fetch_add(1, Ordering::Relaxed);
        }

        if generations_without_improvement.load(Ordering::Relaxed) >= PLATEAU_THRESHOLD {
            pop.par_iter_mut().for_each(|individual| {
                let mut rng = rand::thread_rng();
                if rng.gen_bool(0.2) {
                    add_complexity(individual, &mut rng, width, height);
                }
            });
            generations_without_improvement.store(0, Ordering::Relaxed);
        }

        if gen % 20 == 0 {  // Every 20 generations
            let num_new = POPULATION_SIZE / 10;  // Replace 10% of population
            let new_individuals: Vec<_> = (0..num_new)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    create_random_individual(&mut rng, width, height)
                })
                .collect();
            
            pop.truncate(POPULATION_SIZE - num_new);
            pop.extend(new_individuals);
        }
    });
    let best_individual = population.lock().unwrap().par_iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap().clone();
    best_individual
}



fn add_complexity(individual: &mut Individual, rng: &mut impl Rng, width: u32, height: u32) {
    if rng.gen_bool(0.5) {
        // Add a new path
        individual.paths.push((
            vec![random_command(rng, width, height)],
            random_path_style(rng, width, height),
        ));
    } else {
        // Add a new command to a random existing path
        if let Some((commands, _)) = individual.paths.choose_mut(rng) {
            commands.push(random_command(rng, width, height));
        }
    }
}

fn tournament_selection<'a>(population: &'a [Individual], tournament_size: usize, rng: &mut impl Rng) -> Option<&'a Individual> {
    population
        .choose_multiple(rng, tournament_size)
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
}



fn mutate(individual: &mut Individual, rng: &mut impl Rng, width: u32, height: u32, mutation_rate: f32) {
    let major_mutation_chance = 0.2;

    // Mutate existing paths
    for (commands, style) in &mut individual.paths {
        // Mutate commands
        for command in commands.iter_mut() {
            if rng.gen::<f32>() < mutation_rate {
                if rng.gen::<f32>() < major_mutation_chance {
                    *command = random_command(rng, width, height);
                } else {
                    mutate_command(command, rng, width, height);
                }
            }
        }

        // Mutate style
        if rng.gen::<f32>() < mutation_rate {
            if rng.gen::<f32>() < major_mutation_chance {
                *style = random_path_style(rng, width, height);
            } else {
                // Perform minor mutations on the existing style
                match style {
                    PathStyle { fill: Some(color), gradient: None } => {
                        // Mutate solid color
                        for channel in color.0.iter_mut() {
                            *channel = (*channel as i16 + rng.gen_range(-10..=10)).max(0).min(255) as u8;
                        }
                    },
                    PathStyle { fill: None, gradient: Some(gradient) } => {
                        // Mutate gradient
                        gradient.start.0 += rng.gen_range(-0.1..0.1) * width as f32;
                        gradient.start.1 += rng.gen_range(-0.1..0.1) * height as f32;
                        gradient.end.0 += rng.gen_range(-0.1..0.1) * width as f32;
                        gradient.end.1 += rng.gen_range(-0.1..0.1) * height as f32;
                        if rng.gen_bool(0.5) {
                            // Mutate a single random color in the gradient
                            if let Some(color) = gradient.colors.choose_mut(rng) {
                                for channel in color.0.iter_mut() {
                                    *channel = (*channel as i16 + rng.gen_range(-10..=10)).max(0).min(255) as u8;
                                }
                            }
                        } else {
                            // Mutate all colors in the gradient
                            for color in &mut gradient.colors {
                                for channel in color.0.iter_mut() {
                                    *channel = (*channel as i16 + rng.gen_range(-10..=10)).max(0).min(255) as u8;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }

        // Mutate number of commands in this path
        if rng.gen::<f32>() < mutation_rate {
            if rng.gen_bool(0.5) && commands.len() > 1 {
                commands.remove(rng.gen_range(0..commands.len()));
            } else if commands.len() < MAX_COMMANDS_PER_PATH {
                commands.push(random_command(rng, width, height));
            }
        }
    }

    // Occasionally add or remove entire paths
    if rng.gen::<f32>() < mutation_rate {
        if rng.gen_bool(0.5) && individual.paths.len() > 1 {
            let index = rng.gen_range(0..individual.paths.len());
            individual.paths.remove(index);
        } else if individual.paths.len() < MAX_PATHS {
            individual.paths.push((
                vec![random_command(rng, width, height)],
                random_path_style(rng, width, height),
            ));
        }
    }

    // Ensure at least one path
    while individual.paths.is_empty() {
        individual.paths.push((
            vec![random_command(rng, width, height)],
            random_path_style(rng, width, height),
        ));
    }

    // Ensure at least one command per path
    for (commands, _) in &mut individual.paths {
        if commands.is_empty() {
            commands.push(random_command(rng, width, height));
        }
    }
}








fn mutate_command(command: &mut Command, rng: &mut impl Rng, width: u32, height: u32) {
    let mutation_factor = 0.2; // mutation
    match command {
        Command::M(x, y) | Command::L(x, y) => {
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::H(x) => {
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
        },
        Command::V(y) => {
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::C(x1, y1, x2, y2, x, y) => {
            *x1 += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y1 += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
            *x2 += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y2 += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::S(x2, y2, x, y) => {
            *x2 += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y2 += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::Q(x1, y1, x, y) => {
            *x1 += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y1 += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::T(x, y) => {
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::A(rx, ry, x_axis_rotation, large_arc, sweep, x, y) => {
            *rx += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *ry += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
            *x_axis_rotation += rng.gen_range(-5.0..5.0);
            *large_arc = rng.gen_bool(0.5);
            *sweep = rng.gen_bool(0.5);
            *x += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
            *y += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
        },
        Command::Z => {},
    }
}




fn individual_to_svg(individual: &Individual, width: u32, height: u32) -> Document {
    let mut document = Document::new()
        .set("width", width)
        .set("height", height);

    let mut defs = Definitions::new();

    for (i, (commands, style)) in individual.paths.iter().enumerate() {
        let mut data = Data::new();
        for command in commands {
            match command {
                Command::M(x, y) => data = data.move_to((*x, *y)),
                Command::L(x, y) => data = data.line_to((*x, *y)),
                Command::H(x) => data = data.horizontal_line_to(*x),
                Command::V(y) => data = data.vertical_line_to(*y),
                Command::C(x1, y1, x2, y2, x, y) => data = data.cubic_curve_to((*x1, *y1, *x2, *y2, *x, *y)),
                Command::S(x2, y2, x, y) => data = data.smooth_cubic_curve_to((*x2, *y2, *x, *y)),
                Command::Q(x1, y1, x, y) => data = data.quadratic_curve_to((*x1, *y1, *x, *y)),
                Command::T(x, y) => data = data.smooth_quadratic_curve_to((*x, *y)),
                Command::A(rx, ry, x_axis_rotation, large_arc, sweep, x, y) => 
                    data = data.elliptical_arc_to((
                        *rx, *ry, *x_axis_rotation, 
                        if *large_arc { 1 } else { 0 }, 
                        if *sweep { 1 } else { 0 }, 
                        *x, *y
                    )),
                Command::Z => data = data.close(),
            }
        }

        let mut path = Path::new().set("d", data);

        if let Some(ref gradient) = style.gradient {
            let gradient_id = format!("gradient-{}", i);
            let mut linear_gradient = svg::node::element::LinearGradient::new()
                .set("id", gradient_id.clone())
                .set("x1", format!("{}%", gradient.start.0 / width as f32 * 100.0))
                .set("y1", format!("{}%", gradient.start.1 / height as f32 * 100.0))
                .set("x2", format!("{}%", gradient.end.0 / width as f32 * 100.0))
                .set("y2", format!("{}%", gradient.end.1 / height as f32 * 100.0));

            for (j, color) in gradient.colors.iter().enumerate() {
                let stop = Stop::new()
                    .set("offset", format!("{}%", j * 100 / (gradient.colors.len() - 1)))
                    .set("stop-color", format!("rgba({},{},{},{})", color[0], color[1], color[2], color[3]));
                linear_gradient = linear_gradient.add(stop);
            }

            defs = defs.add(linear_gradient);
            path = path.set("fill", format!("url(#{})", gradient_id));
        } else if let Some(fill) = style.fill {
            path = path.set("fill", format!("rgba({},{},{},{})", fill[0], fill[1], fill[2], fill[3]));
        } else {
            path = path.set("fill", "none");
        }

        document = document.add(path);
    }

    document.add(defs)
}




fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images_dir = "/Users/scott/Downloads/converge/images";
    let output_dir = "/Users/scott/Downloads/converge/output";
    fs::create_dir_all(output_dir)?;

    println!("Reading directory: {}", images_dir);

    let mut image_paths: Vec<_> = fs::read_dir(images_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let extension = path.extension()?;
            if extension == "png" {
                Some(path)
            } else {
                println!("Skipping non-PNG file: {:?}", path);
                None
            }
        })
        .collect();

    println!("Found {} PNG images", image_paths.len());

    if image_paths.is_empty() {
        return Err("No PNG images found in the specified directory".into());
    }







    let mut rng = thread_rng();
    image_paths.shuffle(&mut rng);

    for path in image_paths {
        println!("Processing image: {:?}", path);
        let img = image::open(&path).map_err(|e| format!("Failed to open image {:?}: {}", path, e))?;
        let img_rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        println!("Image dimensions: {}x{}", width, height);

        let best_individual = Arc::new(Mutex::new(Individual { paths: Vec::new(), fitness: 0.0 }));
        let current_image = Arc::new(Mutex::new(RgbaImage::new(width, height)));
        let fitness_history = Arc::new(Mutex::new(Vec::new()));
        let generation = Arc::new(Mutex::new(0));
        let running = Arc::new(Mutex::new(true));

        let (update_sender, update_receiver) = mpsc::channel();

        // Spawn optimization thread
        let optimization_thread = {
            let img_clone = img.clone();
            let best_individual = Arc::clone(&best_individual);
            let current_image = Arc::clone(&current_image);
            let fitness_history = Arc::clone(&fitness_history);
            let generation = Arc::clone(&generation);
            let running = Arc::clone(&running);
            let update_sender = update_sender.clone();
            thread::spawn(move || {
                optimize_image(&img_clone, &best_individual, &current_image, &fitness_history, &generation, &running, update_sender)
            })
        };

        // Run UI on the main thread
        run_ui_main_thread(
            img_rgba,
            Arc::clone(&best_individual),
            Arc::clone(&current_image),
            Arc::clone(&fitness_history),
            Arc::clone(&generation),
            Arc::clone(&running),
            update_receiver,
        );

        // Stop the optimization thread
        *running.lock().unwrap() = false;

        // Wait for the optimization thread to finish
        let final_best_individual = optimization_thread.join().unwrap();

        // Save the final SVG
        let svg_document = individual_to_svg(&final_best_individual, width, height);
        let svg_filename = path.file_stem().unwrap().to_str().unwrap();
        let svg_path = format!("{}/{}.svg", output_dir, svg_filename);
        svg::save(&svg_path, &svg_document).map_err(|e| format!("Failed to save SVG {:?}: {}", svg_path, e))?;

        println!("Processed: {:?}", path);
    }

    Ok(())
}
