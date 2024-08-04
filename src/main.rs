use image::{GenericImageView, Rgba};
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
use image::RgbaImage;


mod ui;
use ui::run_ui_main_thread;


const POPULATION_SIZE: usize = 50;
const GENERATIONS: usize = 10000;
const ELITISM_COUNT: usize = 5;
const PLATEAU_THRESHOLD: usize = 50; // Number of generations to check for plateau
const PLATEAU_IMPROVEMENT_THRESHOLD: f32 = 0.001; // Minimum improvement to not be considered a plateau
const MAX_PATHS: usize = 40;
const MAX_COMMANDS_PER_PATH: usize = 50;

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
    let paths: Vec<(Vec<Command>, PathStyle)> = (0..2)  // Always create 2 paths
        .map(|_| {
            let commands: Vec<Command> = (0..2)  // Always create 2 commands per path
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
            if rng.gen_bool(0.5) {
                child_paths.push(parent1.paths[i].clone());
            } else {
                child_paths.push(parent2.paths[i].clone());
            }
        } else if i < parent1.paths.len() {
            child_paths.push(parent1.paths[i].clone());
        } else if i < parent2.paths.len() {
            child_paths.push(parent2.paths[i].clone());
        }
    }

    Individual { paths: child_paths, fitness: 0.0 }
}



fn render_individual(individual: &Individual, width: u32, height: u32) -> RgbaImage {
    let svg_document = individual_to_svg(individual, width, height);
    let svg_string = svg_document.to_string();

    let opts = Options::default();
    let rtree = Tree::from_str(&svg_string, &opts).unwrap();

    let mut pixmap = Pixmap::new(width, height).unwrap();
    resvg::render(&rtree, Transform::default(), &mut pixmap.as_mut());

    RgbaImage::from_raw(width, height, pixmap.data().to_vec()).unwrap()
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










fn calculate_fitness(img: &image::DynamicImage, individual: &Individual) -> f32 {
    let (width, height) = img.dimensions();
    let rendered = render_individual(individual, width, height);

    let mut total_diff = 0.0;
    let mut pixel_count = 0;

    for (p1, p2) in img.as_rgba8().unwrap().pixels().zip(rendered.pixels()) {
        let d = p1.0.iter().zip(p2.0.iter())
            .map(|(c1, c2)| (*c1 as f32 - *c2 as f32).powi(2))
            .sum::<f32>();
        total_diff += d.sqrt();
        pixel_count += 1;
    }

    let avg_diff = total_diff / pixel_count as f32;
    let fitness = 1.0 / (1.0 + avg_diff);

    //println!("Total diff: {}, Pixel count: {}, Avg diff: {}, Fitness: {}", total_diff, pixel_count, avg_diff, fitness);

    fitness
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
    let mut rng = rand::thread_rng();
    let (width, height) = img.dimensions();

        let mut best_fitness = 0.0;
    let mut generations_without_improvement = 0;

    let mut population: Vec<Individual> = (0..POPULATION_SIZE)
        .map(|_| {
            let mut individual = create_random_individual(&mut rng, width, height);
            individual.fitness = calculate_fitness(img, &individual);
            individual
        })
        .collect();

    for gen in 0..GENERATIONS {
        if !*running.lock().unwrap() {
            break;
        }

        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best = population[0].clone();
        println!("Generation {}: Best fitness = {}", gen, best.fitness);
        
        *best_individual.lock().unwrap() = best.clone();
        let evolved_image = render_individual(&best, width, height);
        *current_image.lock().unwrap() = evolved_image.clone();
        fitness_history.lock().unwrap().push(best.fitness);
        *generation.lock().unwrap() = gen;

        // Create IndividualInfo for each member of the population
        let population_info: Vec<IndividualInfo> = population.iter().enumerate()
            .map(|(i, ind)| IndividualInfo {
                individual: ind.clone(),
                parent_ids: None, // track this during crossover
                is_elite: i < ELITISM_COUNT,
                is_new: false, // set this to true for new individuals
                survived: true, // All individuals in the current population have survived
                crossover: false, // Set this during the crossover process
            })
            .collect();

        // Send the entire population info to the UI thread
        update_sender.send(population_info).unwrap();

        let elites: Vec<Individual> = population.iter().take(ELITISM_COUNT).cloned().collect();
        let mut new_population = Vec::new();

        while new_population.len() < POPULATION_SIZE - ELITISM_COUNT {
            let parent1 = tournament_selection(&population, 5, &mut rng);
            let parent2 = tournament_selection(&population, 5, &mut rng);
            let mut child = crossover(parent1, parent2, &mut rng);
            let mutation_rate = adaptive_mutation_rate(gen, GENERATIONS);
            mutate(&mut child, &mut rng, width, height, mutation_rate);
            child.fitness = calculate_fitness(img, &child);
            new_population.push(child);
        }

        new_population.extend(elites);
        population = new_population;

        let current_best_fitness = population[0].fitness;
        if current_best_fitness > best_fitness + PLATEAU_IMPROVEMENT_THRESHOLD {
            best_fitness = current_best_fitness;
            generations_without_improvement = 0;
        } else {
            generations_without_improvement += 1;
        }

        if generations_without_improvement >= PLATEAU_THRESHOLD {
            // Allow complexity increase
            for individual in &mut population {
                if rng.gen_bool(0.2) { // 20% chance to add complexity
                    add_complexity(individual, &mut rng, width, height);
                }
            }
            generations_without_improvement = 0;
        }

        // Periodic population refresh
        if gen % 50 == 0 && gen > 0 {
            let num_refresh = POPULATION_SIZE / 10;
            for _ in 0..num_refresh {
                let mut new_individual = create_random_individual(&mut rng, width, height);
                new_individual.fitness = calculate_fitness(img, &new_individual);
                population.push(new_individual);
            }
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            population.truncate(POPULATION_SIZE);
        }
    }

    population.into_iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap()
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

fn tournament_selection<'a>(population: &'a [Individual], tournament_size: usize, rng: &mut impl Rng) -> &'a Individual {
    population
        .choose_multiple(rng, tournament_size)
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .unwrap()
}









fn mutate(individual: &mut Individual, rng: &mut impl Rng, width: u32, height: u32, mutation_rate: f32) {
    let major_mutation_chance = 0.1; // 10% chance for a major mutation

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
                style.fill = Some(random_color(rng));
            } else if let Some(fill) = &mut style.fill {
                mutate_color(fill, rng);
            }
        }

        if rng.gen::<f32>() < mutation_rate {
            if rng.gen::<f32>() < major_mutation_chance {
                style.gradient = if rng.gen_bool(0.5) {
                    Some(random_gradient(rng, width, height))
                } else {
                    None
                };
            } else if let Some(gradient) = &mut style.gradient {
                mutate_gradient(gradient, rng, width, height);
            }
        }

        // Mutate number of commands in this path
        if rng.gen::<f32>() < mutation_rate {
            if rng.gen_bool(0.5) && commands.len() > 1 {
                let index = rng.gen_range(0..commands.len());
                commands.remove(index);
            } else if commands.len() < MAX_COMMANDS_PER_PATH {
                commands.push(random_command(rng, width, height));
            }
        }
    }

    // Mutate number of paths 
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

    // Ensure at least two paths and two commands per path 
    while individual.paths.len() < 2 {
        individual.paths.push((
            vec![random_command(rng, width, height), random_command(rng, width, height)],
            random_path_style(rng, width, height),
        ));
    }

    for (commands, _) in &mut individual.paths {
        while commands.len() < 2 {
            commands.push(random_command(rng, width, height));
        }
    }
}







fn mutate_command(command: &mut Command, rng: &mut impl Rng, width: u32, height: u32) {
    let mutation_factor = 0.05; // 5% mutation
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







fn mutate_color(color: &mut Rgba<u8>, rng: &mut impl Rng) {
    let mutation_amount = 10;
    for channel in color.0.iter_mut() {
        *channel = (*channel as i16 + rng.gen_range(-mutation_amount..=mutation_amount))
            .max(0).min(255) as u8;
    }
}

fn mutate_gradient(gradient: &mut LinearGradient, rng: &mut impl Rng, width: u32, height: u32) {
    let mutation_factor = 0.05; // 5% mutation
    gradient.start.0 += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
    gradient.start.1 += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;
    gradient.end.0 += rng.gen_range(-mutation_factor..mutation_factor) * width as f32;
    gradient.end.1 += rng.gen_range(-mutation_factor..mutation_factor) * height as f32;

    for color in &mut gradient.colors {
        mutate_color(color, rng);
    }
}

fn random_gradient(rng: &mut impl Rng, width: u32, height: u32) -> LinearGradient {
    let geo = Geometric::new(0.5).unwrap();
    let num_colors = geo.sample(rng) as usize + 2;
    LinearGradient {
        start: (rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
        end: (rng.gen_range(0.0..width as f32), rng.gen_range(0.0..height as f32)),
        colors: (0..num_colors).map(|_| random_color(rng)).collect(),
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



    let (update_sender, update_receiver) = mpsc::channel::<Vec<IndividualInfo>>();




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
