fn main() {
    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rerun-if-changed=model/cmsis/imu_model.c");
    println!("cargo:rerun-if-changed=model/cmsis/imu_model.h");
    println!("cargo:rerun-if-changed=model/cmsis/imu_model_weights.h");

    let cmsis_nn = std::path::Path::new("third_party/CMSIS-NN");
    let cmsis_core = std::path::Path::new("third_party/CMSIS_6/CMSIS/Core/Include");
    if !cmsis_nn.exists() || !cmsis_core.exists() {
        panic!(
            "CMSIS sources are missing. Expected third_party/CMSIS-NN and third_party/CMSIS_6. \
             Run: git clone --depth 1 https://github.com/ARM-software/CMSIS-NN.git third_party/CMSIS-NN && \
             git clone --depth 1 https://github.com/ARM-software/CMSIS_6.git third_party/CMSIS_6"
        );
    }

    let mut sources = vec![
        std::path::PathBuf::from("model/cmsis/imu_model.c"),
        std::path::PathBuf::from("model/cmsis/cmsis_compat.c"),
    ];

    for dir in [
        "third_party/CMSIS-NN/Source/ActivationFunctions",
        "third_party/CMSIS-NN/Source/BasicMathFunctions",
        "third_party/CMSIS-NN/Source/ConvolutionFunctions",
        "third_party/CMSIS-NN/Source/FullyConnectedFunctions",
        "third_party/CMSIS-NN/Source/NNSupportFunctions",
        "third_party/CMSIS-NN/Source/PoolingFunctions",
    ] {
        for entry in std::fs::read_dir(dir).expect("read CMSIS-NN source directory") {
            let path = entry.expect("read CMSIS-NN source entry").path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("c") {
                println!("cargo:rerun-if-changed={}", path.display());
                sources.push(path);
            }
        }
    }

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    for (idx, source) in sources.iter().enumerate() {
        let stem = source
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("cmsis");
        let object = out_dir.join(format!("{idx:03}_{stem}.o"));
        let status = std::process::Command::new("clang")
            .arg("--target=thumbv8m.main-none-eabihf")
            .arg("-mthumb")
            .arg("-mcpu=cortex-m33")
            .arg("-ffreestanding")
            .arg("-fno-builtin")
            .arg("-O3")
            .arg("-Wno-unknown-attributes")
            .arg("-Wno-unknown-pragmas")
            .arg("-DARM_MATH_DSP")
            .arg("-DCMSIS_NN_USE_SINGLE_ROUNDING")
            .arg("-Imodel/cmsis")
            .arg("-Ithird_party/CMSIS-NN/Include")
            .arg("-Ithird_party/CMSIS_6/CMSIS/Core/Include")
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object)
            .status()
            .expect("run clang for CMSIS-NN source");
        if !status.success() {
            panic!("clang failed while compiling {}", source.display());
        }
        println!("cargo:rustc-link-arg-bins={}", object.display());
    }
}
