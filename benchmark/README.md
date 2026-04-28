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
