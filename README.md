# ECE398 Midterm - IMU Dataset Record

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
