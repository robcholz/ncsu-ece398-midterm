# ECE398 Midterm - IMU Dataset Record

## Requirements

- Python toolchain for `uv`
- Rust toolchain with `rustup`
- Rust target: `thumbv8m.main-none-eabihf`
- Rust component: `llvm-tools-preview`
- `dfu-util` for flashing over STM32 USB DFU
- macOS host tools: `stty` for `cargo monitor`
- `clang` for compiling CMSIS-NN C sources

```shell
rustup target add thumbv8m.main-none-eabihf
rustup component add llvm-tools-preview
```

## Model Pipeline

The model pipeline is:

1. Build labeled accelerometer windows from `dataset/Multimodal Cough Dataset`.
2. Train a multiclass CNN and save a PyTorch checkpoint.
3. Export that checkpoint to CMSIS-NN int8 C weights.
4. Evaluate the generated C model on the host.
5. Build the Rust firmware binary that links the CMSIS-NN C model through FFI.

### 1. Train and Save the Checkpoint

Small deployment model:

```shell
uv run python -m benchmark.host \
  --task multiclass \
  --model small \
  --sampling-strategy event-centered \
  --event-windows-per-event 3 \
  --background-windows-per-event 1.0 \
  --background-exclusion-seconds 0.25 \
  --normalization train \
  --balanced-sampler \
  --include-magnitude \
  --window-seconds 1.0 \
  --label-overlap-threshold 0.15 \
  --epochs 30 \
  --max-windows 20000 \
  --max-background-ratio 3.0 \
  --latency-runs 500 \
  --batch-size 64 \
  --output benchmark/results/final_multiclass_small_event_centered.json \
  --save-model model/artifacts/final_multiclass_small_event_centered.pt
```

### 2. Export CMSIS-NN C Weights

This is the CLI that generates the model C header used by the firmware:

```shell
uv run python -m model.export_cmsis \
  --checkpoint model/artifacts/final_multiclass_small_event_centered.pt \
  --output model/cmsis/imu_model_weights.h \
  --max-calibration-windows 2048 \
  --activation-percentile 100
```

The firmware build compiles the hand-written wrapper plus the generated weights header through `build.rs`.

### 3. Evaluate the Quantized C Model

```shell
uv run python -m benchmark.quantized_c \
  --checkpoint model/artifacts/final_multiclass_small_event_centered.pt \
  --output benchmark/results/quantized_c_eval.json
```

This compares the PyTorch checkpoint against the generated C/CMSIS-NN model on the same validation split.

## Flash

```shell
cargo build --release --bin data-collection
cargo flash --bin data-collection
```

For the model benchmark firmware:

```shell
cargo build --release --bin benchmark
cargo flash --bin benchmark
```

## Data Collection

```shell
uv run scripts/monitor_record.py <event_name> --output-dir <output_dir>
uv run scripts/monitor_plot.py --csv <path_to_data>
uv run scripts/interactive_plot.py --csv <path_to_data>
```

## Quality

```shell
cargo fmt --all
cargo clippy --all-targets --all-features --workspace -- -D warnings
```
