use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const WINDOW_SAMPLES: usize = 200;
const WINDOW_LIMIT: usize = 16;

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

    if let Err(err) = generate_benchmark_data(&manifest_dir, &out_dir) {
        panic!("failed to generate benchmark data: {err}");
    }
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

fn generate_benchmark_data(
    manifest_dir: &Path,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let recordings_dir = manifest_dir.join("recordings");
    println!("cargo:rerun-if-changed={}", recordings_dir.display());

    let csv_files = discover_csv_files(&recordings_dir)?;
    let windows = collect_windows(&csv_files)?;
    if windows.is_empty() {
        return Err("no benchmark windows were generated from recordings".into());
    }

    let out_path = out_dir.join("benchmark_data.rs");
    let mut file = fs::File::create(out_path)?;

    writeln!(
        file,
        "pub const BENCHMARK_WINDOW_SAMPLES: usize = {WINDOW_SAMPLES};"
    )?;
    writeln!(
        file,
        "pub const BENCHMARK_WINDOW_COUNT: usize = {};",
        windows.len()
    )?;
    writeln!(
        file,
        "#[allow(clippy::approx_constant, clippy::excessive_precision)]"
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

fn collect_windows(
    csv_files: &[PathBuf],
) -> Result<Vec<[[f32; 4]; WINDOW_SAMPLES]>, Box<dyn std::error::Error>> {
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
