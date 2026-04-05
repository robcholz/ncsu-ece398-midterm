# ECE398 Midterm - IMU Dataset Record

## Requirements

- Rust toolchain with `rustup`
- Rust target: `thumbv8m.main-none-eabihf`
- Rust component: `llvm-tools-preview`
- `dfu-util` for flashing over STM32 USB DFU
- macOS host tools: `stty` for `cargo monitor`

```bash
rustup target add thumbv8m.main-none-eabihf
rustup component add llvm-tools-preview
```

## Flash

```markdown
cargo build --release
cargo flash
```

## Record

```python
python scripts/monitor_record.py <event_name> --output-dir <output_dir>
python scripts/monitor_plot.py --csv <path_to_data>
```
