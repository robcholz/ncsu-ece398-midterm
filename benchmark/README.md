# Benchmark

First run

```shell
uv run python -m benchmark.host \
    --epochs 10 \
    --max-windows 6000 \
    --latency-runs 200 \
    --batch-size 32 \
    --stride 0.5 \
    --output benchmark/results/host_baseline_e0.json
```

Second run

```shell
uv run python -m benchmark.host \
    --epochs 10 \
    --max-windows 6000 \
    --latency-runs 200 \
    --batch-size 32 \
    --stride 0.5 \
    --include-magnitude \
    --output benchmark/results/host_norm_mag_e0.json
```

Third run

```shell
uv run python -m benchmark.host \
    --epochs 10 \
    --max-windows 6000 \
    --latency-runs 200 \
    --batch-size 32 \
    --stride 0.6 \
    --output benchmark/results/host_baseline_e1.json
```

Fourth run (best)

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
    --output benchmark/results/multiclass_small_event_centered_1s_20k_30e.json
```
