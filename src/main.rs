use image::{GenericImageView, Rgba, RgbaImage};
use rand::Rng;
use std::fs;
use std::thread;
use std::time::{Duration, Instant};
use svg::Document;
use svg::node::element::path::Data;
use svg::node::element::Path;
use resvg::usvg::{Tree, Options};
use resvg::tiny_skia::{Pixmap, Transform};
use std::sync::{Arc, Mutex};
use std::sync::mpsc;


mod ui;
use ui::run_ui_main_thread;


const POPULATION_SIZE: usize = 50;
const GENERATIONS: usize = 1000;
const MUTATION_RATE: f32 = 0.1;
const MAX_PATHS: usize = 100;
const MAX_COMMANDS: usize = 20;

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
    paths: Vec<(Vec<Command>, Rgba<u8>)>,
    fitness: f32,
}

fn random_command(rng: &mut rand::rngs::ThreadRng, width: u32, height: u32) -> Command {
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

fn random_color(rng: &mut rand::rngs::ThreadRng) -> Rgba<u8> {
    Rgba([rng.gen(), rng.gen(), rng.gen(), rng.gen_range(50..=200)])
}

fn create_random_individual(rng: &mut rand::rngs::ThreadRng, width: u32, height: u32) -> Individual {
    let num_paths = rng.gen_range(1..=MAX_PATHS);
    let paths: Vec<(Vec<Command>, Rgba<u8>)> = (0..num_paths)
        .map(|_| {
            let num_commands = rng.gen_range(1..=MAX_COMMANDS);
            let commands: Vec<Command> = (0..num_commands)
                .map(|_| random_command(rng, width, height))
                .collect();
            (commands, random_color(rng))
        })
        .collect();

    Individual { paths, fitness: 0.0 }
}

fn mutate(individual: &mut Individual, rng: &mut rand::rngs::ThreadRng, width: u32, height: u32) {
    for (commands, color) in &mut individual.paths {
        if rng.gen::<f32>() < MUTATION_RATE {
            let index = rng.gen_range(0..commands.len());
            commands[index] = random_command(rng, width, height);
        }
        if rng.gen::<f32>() < MUTATION_RATE {
            *color = random_color(rng);
        }
    }
    if rng.gen::<f32>() < MUTATION_RATE {
        if individual.paths.len() < MAX_PATHS {
            individual.paths.push((
                vec![random_command(rng, width, height)],
                random_color(rng),
            ));
        } else if !individual.paths.is_empty() {
            let index = rng.gen_range(0..individual.paths.len());
            individual.paths.remove(index);
        }
    }
}

fn crossover(parent1: &Individual, parent2: &Individual, rng: &mut rand::rngs::ThreadRng) -> Individual {
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
    update_sender: mpsc::Sender<RgbaImage>,
) -> Individual {
    let mut rng = rand::thread_rng();
    let (width, height) = img.dimensions();

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

        // Send the updated image to the UI thread
        update_sender.send(evolved_image).unwrap();

        let mut new_population = Vec::new();

        while new_population.len() < POPULATION_SIZE {
            let parent1 = &population[rng.gen_range(0..POPULATION_SIZE / 2)];
            let parent2 = &population[rng.gen_range(0..POPULATION_SIZE / 2)];
            let mut child = crossover(parent1, parent2, &mut rng);
            mutate(&mut child, &mut rng, width, height);
            child.fitness = calculate_fitness(img, &child);
            new_population.push(child);
        }

        population = new_population;
    }

    population.into_iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap()
}










fn individual_to_svg(individual: &Individual, width: u32, height: u32) -> Document {
    let mut document = Document::new()
        .set("width", width)
        .set("height", height);

    for (commands, color) in &individual.paths {
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
        let path = Path::new()
            .set("fill", format!("rgba({},{},{},{})", color[0], color[1], color[2], color[3]))
            .set("stroke", "none")
            .set("d", data);
        document = document.add(path);
    }

    document
}






fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images_dir = "/Users/scott/Downloads/converge/images";
    let output_dir = "/Users/scott/Downloads/converge/output";
    fs::create_dir_all(output_dir)?;

    let image_paths: Vec<_> = fs::read_dir(images_dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()?.to_str()? == "png" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    for path in image_paths {
        let img = image::open(&path).expect("Failed to open image");
        let img_rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

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
        svg::save(svg_path, &svg_document).expect("Failed to save SVG");

        println!("Processed: {:?}", path);
    }

    Ok(())
}
