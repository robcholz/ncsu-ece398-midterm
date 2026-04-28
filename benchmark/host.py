"""Host-side benchmark for the baseline accelerometer CNN."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from model.dataset import LABELS, WindowConfig, build_windows, subject_holdout_split
from model.metrics import classification_report

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from model.cnn import SmallAccelCNN, count_parameters, model_size_bytes
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    torch = None
    _TORCH_IMPORT_ERROR = (
        "The host benchmark requires PyTorch. Install project dependencies with "
        "`uv sync` or install `torch` in the active environment."
    )


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit(f"{_TORCH_IMPORT_ERROR}")

    seed_everything(args.seed)
    config = WindowConfig(
        stride_seconds=args.stride,
        label_overlap_threshold=args.label_overlap_threshold,
        include_magnitude=args.include_magnitude,
        normalize=args.normalize,
        max_background_ratio=args.max_background_ratio,
        seed=args.seed,
    )
    x, y, metadata = build_windows(args.data_root, config, max_windows=args.max_windows)
    if len(y) == 0:
        raise SystemExit("No windows were built from the dataset.")
    split = subject_holdout_split(x, y, metadata, args.val_subject)

    model = SmallAccelCNN(in_channels=split.x_train.shape[1], num_classes=len(LABELS))
    device = torch.device(args.device)
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x_train), torch.from_numpy(split.y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(split.x_val), torch.from_numpy(split.y_val)),
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights(split.y_train, len(LABELS)).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = None
    best_macro_f1 = -1.0
    best_report = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = predict(model, val_loader, device)
        report = classification_report(y_true, y_pred, split.class_names)
        if report["macro_f1"] > best_macro_f1:
            best_macro_f1 = report["macro_f1"]
            best_report = report
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"val_accuracy={report['accuracy']:.4f} val_macro_f1={report['macro_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    latency = measure_latency(model, split.x_val, device, args.latency_runs)
    peak_ram_mb = peak_rss_mb()

    result = {
        "dataset": {
            "data_root": str(args.data_root),
            "num_windows": int(len(y)),
            "train_windows": int(len(split.y_train)),
            "val_windows": int(len(split.y_val)),
            "train_subjects": split.train_subjects,
            "val_subjects": split.val_subjects,
            "class_counts": {
                name: int((y == idx).sum()) for idx, name in enumerate(split.class_names)
            },
        },
        "model": {
            "name": "SmallAccelCNN",
            "input_shape": [int(split.x_train.shape[1]), int(split.x_train.shape[2])],
            "params": int(count_parameters(model)),
            "size_mb": model_size_bytes(model) / (1024 * 1024),
        },
        "metrics": serialize_report(best_report),
        "host": {
            "device": str(device),
            "latency_ms_p50": latency["p50_ms"],
            "latency_ms_p95": latency["p95_ms"],
            "peak_ram_mb": peak_ram_mb,
        },
    }
    print(json.dumps(result, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/Multimodal Cough Dataset"),
        help="Path containing DataAnnotation.json and subject folders.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--label-overlap-threshold", type=float, default=0.3)
    parser.add_argument("--max-windows", type=int, default=6000)
    parser.add_argument("--max-background-ratio", type=float, default=3.0)
    parser.add_argument("--val-subject", action="append")
    parser.add_argument("--include-magnitude", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--latency-runs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=398)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_items += len(y_batch)
    return total_loss / max(total_items, 1)


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
            pred = logits.argmax(dim=1).cpu().numpy()
            true_batches.append(y_batch.numpy())
            pred_batches.append(pred)
    return np.concatenate(true_batches), np.concatenate(pred_batches)


def measure_latency(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    runs: int,
) -> dict[str, float]:
    model.eval()
    sample = torch.from_numpy(x[:1]).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(sample)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def serialize_report(report: dict | None) -> dict:
    if report is None:
        return {}
    serialized = dict(report)
    serialized["confusion_matrix"] = report["confusion_matrix"].tolist()
    return serialized


if __name__ == "__main__":
    raise SystemExit(main())
