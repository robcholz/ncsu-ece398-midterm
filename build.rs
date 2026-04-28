use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const WINDOW_SAMPLES: usize = 200;
const WINDOW_LIMIT: usize = 16;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_arch == "arm" && target_os == "none" {
        println!("cargo:rustc-link-arg-bins=--nmagic");
        println!("cargo:rustc-link-arg-bins=-Tlink.x");
    }

    if let Err(err) = generate_benchmark_data() {
        panic!("failed to generate benchmark data: {err}");
    }
}

fn generate_benchmark_data() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let recordings_dir = manifest_dir.join("recordings");
    println!("cargo:rerun-if-changed={}", recordings_dir.display());

    let csv_files = discover_csv_files(&recordings_dir)?;
    let windows = collect_windows(&csv_files)?;
    if windows.is_empty() {
        return Err("no benchmark windows were generated from recordings".into());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let out_path = out_dir.join("benchmark_data.rs");
    let mut file = fs::File::create(out_path)?;

    writeln!(file, "pub const BENCHMARK_WINDOW_SAMPLES: usize = {WINDOW_SAMPLES};")?;
    writeln!(
        file,
        "pub const BENCHMARK_WINDOW_COUNT: usize = {};",
        windows.len()
    )?;
    writeln!(
        file,
        "pub static BENCHMARK_WINDOWS: [[[f32; 4]; {WINDOW_SAMPLES}]; {}] = [",
        windows.len()
    )?;

    for window in &windows {
        writeln!(file, "    [")?;
        for sample in window {
            writeln!(
                file,
                "        [{:.6}, {:.6}, {:.6}, {:.6}],",
                sample[0], sample[1], sample[2], sample[3]
            )?;
        }
        writeln!(file, "    ],")?;
    }

    writeln!(file, "];")?;
    Ok(())
}

fn discover_csv_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    visit_dir(root, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn visit_dir(dir: &Path, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_dir(&path, paths)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("csv") {
            paths.push(path);
        }
    }
    Ok(())
}

fn collect_windows(csv_files: &[PathBuf]) -> Result<Vec<[[f32; 4]; WINDOW_SAMPLES]>, Box<dyn std::error::Error>> {
    let mut windows = Vec::new();

    for csv_path in csv_files {
        let rows = load_recording(csv_path)?;
        if rows.len() < WINDOW_SAMPLES {
            continue;
        }

        windows.push(extract_window(&rows));
        if windows.len() == WINDOW_LIMIT {
            break;
        }
    }

    Ok(windows)
}

fn load_recording(csv_path: &Path) -> Result<Vec<[f32; 4]>, Box<dyn std::error::Error>> {
    let file = fs::File::open(csv_path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for (line_number, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        if line_number == 0 {
            continue;
        }

        let mut fields = line.split(',');
        let _timestamp = fields.next();
        let ax = parse_field(fields.next(), csv_path, line_number + 1, "acc_x")?;
        let ay = parse_field(fields.next(), csv_path, line_number + 1, "acc_y")?;
        let az = parse_field(fields.next(), csv_path, line_number + 1, "acc_z")?;

        let acc_mag = (ax * ax + ay * ay + az * az).sqrt();
        rows.push([ax, ay, az, acc_mag]);
    }

    Ok(rows)
}

fn parse_field(
    field: Option<&str>,
    csv_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let raw = field.ok_or_else(|| {
        format!(
            "{}:{} missing required field {name}",
            csv_path.display(),
            line_number
        )
    })?;

    raw.parse::<f32>().map_err(|err| {
        format!(
            "{}:{} invalid {name} value {raw:?}: {err}",
            csv_path.display(),
            line_number
        )
        .into()
    })
}

fn extract_window(rows: &[[f32; 4]]) -> [[f32; 4]; WINDOW_SAMPLES] {
    let start = if rows.len() > WINDOW_SAMPLES {
        (rows.len() - WINDOW_SAMPLES) / 2
    } else {
        0
    };

    let mut window = [[0.0; 4]; WINDOW_SAMPLES];
    for (index, slot) in window.iter_mut().enumerate() {
        *slot = rows[start + index];
    }
    window
}
