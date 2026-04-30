"""Host-side benchmark for weighted checkpoint ensembles."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np

from benchmark.host import (
    apply_task,
    apply_train_normalization_with_stats,
    measure_latency,
    primary_logits,
)
from model.dataset import WindowConfig, build_windows, subject_holdout_split
from model.metrics import classification_report

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from model.cnn import build_model, count_parameters, model_size_bytes
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = (
        "The ensemble benchmark requires PyTorch. Install project dependencies with "
        "`uv sync` or install `torch` in the active environment."
    )


class WeightedEnsemble(nn.Module if nn is not None else object):
    def __init__(self, models: list[nn.Module], weights: list[float]) -> None:
        super().__init__()
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length")
        total = sum(weights)
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.models = nn.ModuleList(models)
        self.register_buffer(
            "weights",
            torch.as_tensor(
                [weight / total for weight in weights], dtype=torch.float32
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = None
        for idx, model in enumerate(self.models):
            weighted = primary_logits(model(x)) * self.weights[idx]
            logits = weighted if logits is None else logits + weighted
        if logits is None:
            raise ValueError("ensemble must contain at least one model")
        return logits


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit(_TORCH_IMPORT_ERROR)

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoint
    ]
    weights = parse_weights(args.weights, len(checkpoints))
    config = checkpoints[0]["config"]
    dataset_config = comparable_dataset_config(config)
    class_names = tuple(checkpoints[0]["class_names"])

    for checkpoint in checkpoints[1:]:
        if comparable_dataset_config(checkpoint["config"]) != dataset_config:
            raise SystemExit(
                "All ensemble checkpoints must use the same dataset config."
            )
        if tuple(checkpoint["class_names"]) != class_names:
            raise SystemExit("All ensemble checkpoints must use the same class order.")

    window_config = WindowConfig(
        window_seconds=config["window_seconds"],
        stride_seconds=config.get("stride", 0.5),
        label_overlap_threshold=config["label_overlap_threshold"],
        include_magnitude=config["include_magnitude"],
        normalize=config["normalization"] == "window",
        max_background_ratio=config["max_background_ratio"],
        sampling_strategy=config.get("sampling_strategy", "sliding"),
        event_windows_per_event=config.get("event_windows_per_event", 3),
        event_jitter_seconds=config.get("event_jitter_seconds", 0.25),
        background_windows_per_event=config.get("background_windows_per_event", 1.0),
        background_exclusion_seconds=config.get("background_exclusion_seconds", 0.25),
        seed=args.seed,
    )
    x, y, metadata = build_windows(args.data_root, window_config, config["max_windows"])
    split = subject_holdout_split(x, y, metadata)
    split = apply_task(split, config["task"])
    if config["normalization"] == "train":
        split, _stats = apply_train_normalization_with_stats(split)

    models = []
    for checkpoint in checkpoints:
        model = build_model(
            checkpoint["model"],
            in_channels=checkpoint["input_shape"][0],
            num_classes=len(class_names),
            dropout=checkpoint["config"].get("dropout", 0.2),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)

    device = torch.device(args.device)
    ensemble = WeightedEnsemble(models, weights).to(device)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x_val), torch.from_numpy(split.y_val)),
        batch_size=args.batch_size,
    )
    y_true, y_pred = predict(ensemble, val_loader, device)
    report = classification_report(y_true, y_pred, class_names)
    latency = measure_latency(ensemble, split.x_val, device, args.latency_runs)

    result = {
        "dataset": {
            "data_root": str(args.data_root),
            "num_windows": int(len(y)),
            "train_windows": int(len(split.y_train)),
            "val_windows": int(len(split.y_val)),
            "train_subjects": split.train_subjects,
            "val_subjects": split.val_subjects,
        },
        "model": {
            "name": "WeightedEnsemble",
            "members": [checkpoint["model"] for checkpoint in checkpoints],
            "weights": weights,
            "input_shape": list(checkpoints[0]["input_shape"]),
            "params": int(sum(count_parameters(model) for model in models)),
            "size_mb": sum(model_size_bytes(model) for model in models) / (1024 * 1024),
        },
        "config": {
            "source_configs": [checkpoint["config"] for checkpoint in checkpoints],
            "batch_size": args.batch_size,
            "latency_runs": args.latency_runs,
        },
        "metrics": serialize_report(report),
        "host": {
            "device": str(device),
            "latency_ms_p50": latency["p50_ms"],
            "latency_ms_p95": latency["p95_ms"],
            "peak_ram_mb": peak_rss_mb(),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated checkpoint weights. Defaults to equal weights.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/Multimodal Cough Dataset"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-runs", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=398)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_weights(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0 / count] * count
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}")
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [weight / total for weight in weights]


def comparable_dataset_config(config: dict) -> dict:
    keys = (
        "task",
        "window_seconds",
        "stride",
        "label_overlap_threshold",
        "include_magnitude",
        "normalization",
        "max_windows",
        "max_background_ratio",
        "sampling_strategy",
        "event_windows_per_event",
        "event_jitter_seconds",
        "background_windows_per_event",
        "background_exclusion_seconds",
    )
    return {key: config.get(key) for key in keys}


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_batches = []
    pred_batches = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            pred_batches.append(logits.argmax(dim=1).cpu().numpy())
            true_batches.append(y_batch.numpy())
    return np.concatenate(true_batches), np.concatenate(pred_batches)


def serialize_report(report: dict) -> dict:
    serialized = dict(report)
    serialized["confusion_matrix"] = report["confusion_matrix"].tolist()
    return serialized


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


if __name__ == "__main__":
    raise SystemExit(main())
