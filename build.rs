use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const BENCHMARK_WINDOW_LIMIT: usize = 16;
const DATASET_SAMPLE_RATE_HZ: f32 = 100.0;

fn main() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    if target_arch == "arm" && target_os == "none" {
        println!("cargo:rustc-link-arg-bins=--nmagic");
        println!("cargo:rustc-link-arg-bins=-Tlink.x");
        compile_cmsis_model(&manifest_dir, &out_dir);
    }

    generate_rust_model_constants(&manifest_dir, &out_dir);
    generate_benchmark_data(&manifest_dir, &out_dir);
}

fn compile_cmsis_model(manifest_dir: &Path, out_dir: &Path) {
    println!("cargo:rerun-if-changed=model/cmsis/imu_model.c");
    println!("cargo:rerun-if-changed=model/cmsis/imu_model.h");
    println!("cargo:rerun-if-changed=model/cmsis/imu_model_weights.h");

    let cmsis_nn = manifest_dir.join("third_party/CMSIS-NN");
    let cmsis_core = manifest_dir.join("third_party/CMSIS_6/CMSIS/Core/Include");
    if !cmsis_nn.exists() || !cmsis_core.exists() {
        panic!(
            "CMSIS sources are missing. Expected third_party/CMSIS-NN and third_party/CMSIS_6. \
             Run: git clone --depth 1 https://github.com/ARM-software/CMSIS-NN.git third_party/CMSIS-NN && \
             git clone --depth 1 https://github.com/ARM-software/CMSIS_6.git third_party/CMSIS_6"
        );
    }

    let mut sources = vec![
        manifest_dir.join("model/cmsis/imu_model.c"),
        manifest_dir.join("model/cmsis/cmsis_compat.c"),
    ];

    for dir in [
        "third_party/CMSIS-NN/Source/ActivationFunctions",
        "third_party/CMSIS-NN/Source/BasicMathFunctions",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions",
        "third_party/CMSIS-NN/Source/FullyConnectedFunctions",
        "third_party/CMSIS-NN/Source/NNSupportFunctions",
        "third_party/CMSIS-NN/Source/PoolingFunctions",
    ] {
        let dir_path = manifest_dir.join(dir);
        for entry in fs::read_dir(&dir_path).expect("read CMSIS-NN source directory") {
            let path = entry.expect("read CMSIS-NN source entry").path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("c") {
                println!("cargo:rerun-if-changed={}", path.display());
                sources.push(path);
            }
        }
    }

    let c_compiler = cmsis_c_compiler();

    for (idx, source) in sources.iter().enumerate() {
        let stem = source
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("cmsis");
        let object = out_dir.join(format!("{idx:03}_{stem}.o"));
        let mut command = std::process::Command::new(&c_compiler.program);
        command.args(&c_compiler.flags);
        let status = command
            .arg("-mthumb")
            .arg("-mcpu=cortex-m33")
            .arg("-ffreestanding")
            .arg("-fno-builtin")
            .arg("-O3")
            .arg("-Wno-unknown-attributes")
            .arg("-Wno-unknown-pragmas")
            .arg("-DARM_MATH_DSP")
            .arg("-DCMSIS_NN_USE_SINGLE_ROUNDING")
            .arg(format!("-I{}", manifest_dir.join("model/cmsis").display()))
            .arg(format!("-I{}", cmsis_nn.join("Include").display()))
            .arg(format!("-I{}", cmsis_core.display()))
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object)
            .status()
            .unwrap_or_else(|err| panic!("run {} for CMSIS-NN source: {err}", c_compiler.program));
        if !status.success() {
            panic!(
                "{} failed while compiling {}",
                c_compiler.program,
                source.display()
            );
        }
        println!("cargo:rustc-link-arg-bins={}", object.display());
    }
}

struct CCompiler {
    program: String,
    flags: Vec<&'static str>,
}

fn cmsis_c_compiler() -> CCompiler {
    let program = env::var("CMSIS_CC").unwrap_or_else(|_| "clang".to_string());
    println!("cargo:rerun-if-env-changed=CMSIS_CC");
    if program.contains("clang") {
        CCompiler {
            program,
            flags: vec!["--target=thumbv8m.main-none-eabihf"],
        }
    } else {
        CCompiler {
            program,
            flags: Vec::new(),
        }
    }
}

fn generate_rust_model_constants(manifest_dir: &Path, out_dir: &Path) {
    let header_path = manifest_dir.join("model/cmsis/imu_model.h");
    println!("cargo:rerun-if-changed={}", header_path.display());
    let header = fs::read_to_string(header_path).expect("read imu_model.h");
    let input_len = read_define_usize(&header, "IMU_MODEL_INPUT_LEN");
    let channels = read_define_usize(&header, "IMU_MODEL_CHANNELS");
    let classes = read_define_usize(&header, "IMU_MODEL_CLASSES");
    let input_size = input_len * channels;
    let generated = format!(
        "\
// Generated by build.rs from model/cmsis/imu_model.h.
pub const INPUT_LEN: usize = {input_len};
pub const CHANNELS: usize = {channels};
pub const INPUT_SIZE: usize = {input_size};
pub const CLASSES: usize = {classes};
"
    );
    fs::write(out_dir.join("model_constants.rs"), generated)
        .expect("write generated Rust model constants");
}

fn read_define_usize(header: &str, name: &str) -> usize {
    for line in header.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("#define") && parts.next() == Some(name) {
            let value = parts
                .next()
                .unwrap_or_else(|| panic!("missing value for {name}"));
            return value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("expected integer value for {name}, got {value}"));
        }
    }
    panic!("missing #define {name} in model/cmsis/imu_model.h");
}

fn generate_benchmark_data(manifest_dir: &Path, out_dir: &Path) {
    let header_path = manifest_dir.join("model/cmsis/imu_model.h");
    let header = fs::read_to_string(&header_path).expect("read imu_model.h");
    let input_len = read_define_usize(&header, "IMU_MODEL_INPUT_LEN");
    let channels = read_define_usize(&header, "IMU_MODEL_CHANNELS");
    if channels != 4 {
        panic!("benchmark data generation expects 4 channels, got {channels}");
    }

    let data_root = manifest_dir.join("dataset/Multimodal Cough Dataset");
    println!("cargo:rerun-if-changed={}", data_root.display());
    if !data_root.exists() {
        panic!(
            "missing dataset for firmware benchmark: {}",
            data_root.display()
        );
    }

    let csv_files = discover_dataset_accelerometer_files(&data_root)
        .expect("discover dataset Accelerometer.csv files");
    let windows = collect_dataset_windows(&csv_files, input_len);
    if windows.is_empty() {
        panic!("no firmware benchmark windows were generated from dataset");
    }

    let out_path = out_dir.join("benchmark_data.rs");
    write_benchmark_data(&out_path, input_len, &windows).expect("write benchmark data");
}

fn discover_dataset_accelerometer_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    visit_dataset_dir(root, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn visit_dataset_dir(dir: &Path, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_dataset_dir(&path, paths)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("Accelerometer.csv") {
            paths.push(path);
        }
    }
    Ok(())
}

fn collect_dataset_windows(csv_files: &[PathBuf], input_len: usize) -> Vec<BenchmarkWindow> {
    let mut windows = Vec::new();
    for csv_path in csv_files {
        let rows = load_dataset_accelerometer(csv_path)
            .unwrap_or_else(|err| panic!("load {}: {err}", csv_path.display()));
        if rows.len() < input_len {
            continue;
        }
        windows.push(BenchmarkWindow {
            samples: extract_resampled_dataset_window(&rows, input_len),
        });
        if windows.len() == BENCHMARK_WINDOW_LIMIT {
            break;
        }
    }
    windows
}

struct BenchmarkWindow {
    samples: Vec<[f32; 4]>,
}

fn load_dataset_accelerometer(
    csv_path: &Path,
) -> Result<Vec<(f32, [f32; 3])>, Box<dyn std::error::Error>> {
    let file = fs::File::open(csv_path)?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let headers: Vec<&str> = header.trim_end().split(',').collect();
    let elapsed_idx = find_header(&headers, "elapsed (s)", csv_path)?;
    let x_idx = find_header(&headers, "x-axis (g)", csv_path)?;
    let y_idx = find_header(&headers, "y-axis (g)", csv_path)?;
    let z_idx = find_header(&headers, "z-axis (g)", csv_path)?;

    let mut rows = Vec::new();
    for (line_number, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() <= z_idx {
            continue;
        }
        let line_number = line_number + 2;
        let elapsed = parse_indexed_field(&fields, elapsed_idx, csv_path, line_number, "elapsed (s)")?;
        let ax = parse_indexed_field(&fields, x_idx, csv_path, line_number, "x-axis (g)")?;
        let ay = parse_indexed_field(&fields, y_idx, csv_path, line_number, "y-axis (g)")?;
        let az = parse_indexed_field(&fields, z_idx, csv_path, line_number, "z-axis (g)")?;
        rows.push((elapsed, [ax, ay, az]));
    }
    Ok(rows)
}

fn find_header(
    headers: &[&str],
    name: &str,
    csv_path: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    headers
        .iter()
        .position(|header| *header == name)
        .ok_or_else(|| format!("{} missing header {name}", csv_path.display()).into())
}

fn parse_indexed_field(
    fields: &[&str],
    index: usize,
    csv_path: &Path,
    line_number: usize,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let raw = fields.get(index).ok_or_else(|| {
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

fn extract_resampled_dataset_window(rows: &[(f32, [f32; 3])], input_len: usize) -> Vec<[f32; 4]> {
    let duration = input_len as f32 / DATASET_SAMPLE_RATE_HZ;
    let first_time = rows.first().map(|row| row.0).unwrap_or(0.0);
    let last_time = rows.last().map(|row| row.0).unwrap_or(first_time);
    let start = if last_time - first_time > duration {
        first_time + (last_time - first_time - duration) / 2.0
    } else {
        first_time
    };

    let mut window = Vec::with_capacity(input_len);
    let mut cursor = 0usize;
    for sample in 0..input_len {
        let target = start + sample as f32 / DATASET_SAMPLE_RATE_HZ;
        while cursor + 1 < rows.len() && rows[cursor + 1].0 < target {
            cursor += 1;
        }
        let [ax, ay, az] = interpolate_dataset_sample(rows, cursor, target);
        let magnitude = (ax * ax + ay * ay + az * az).sqrt();
        window.push([ax, ay, az, magnitude]);
    }
    window
}

fn interpolate_dataset_sample(rows: &[(f32, [f32; 3])], cursor: usize, target: f32) -> [f32; 3] {
    if cursor + 1 >= rows.len() {
        return rows[cursor].1;
    }
    let (t0, v0) = rows[cursor];
    let (t1, v1) = rows[cursor + 1];
    if t1 <= t0 {
        return v0;
    }
    let alpha = ((target - t0) / (t1 - t0)).clamp(0.0, 1.0);
    [
        v0[0] + alpha * (v1[0] - v0[0]),
        v0[1] + alpha * (v1[1] - v0[1]),
        v0[2] + alpha * (v1[2] - v0[2]),
    ]
}

fn write_benchmark_data(
    out_path: &Path,
    input_len: usize,
    windows: &[BenchmarkWindow],
) -> io::Result<()> {
    let mut file = fs::File::create(out_path)?;
    writeln!(file, "// Generated by build.rs from dataset/Multimodal Cough Dataset.")?;
    writeln!(file, "// Values are unnormalized g-units in [ax, ay, az, magnitude] order.")?;
    writeln!(file, "pub const BENCHMARK_WINDOW_SAMPLES: usize = {input_len};")?;
    writeln!(
        file,
        "pub const BENCHMARK_WINDOW_COUNT: usize = {};",
        windows.len()
    )?;
    writeln!(file, "#[allow(clippy::approx_constant, clippy::excessive_precision)]")?;
    writeln!(
        file,
        "pub static BENCHMARK_WINDOWS: [[[f32; 4]; {input_len}]; {}] = [",
        windows.len()
    )?;
    for window in windows {
        writeln!(file, "    [")?;
        for sample in &window.samples {
            writeln!(
                file,
                "        [{:.6}f32, {:.6}f32, {:.6}f32, {:.6}f32],",
                sample[0], sample[1], sample[2], sample[3]
            )?;
        }
        writeln!(file, "    ],")?;
    }
    writeln!(file, "];")?;
    Ok(())
}
